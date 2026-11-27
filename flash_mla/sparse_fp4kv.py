# Copyright 2026 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for FlashMLA in 2026. Derived from SGLang's
# python/sglang/kernels/ops/attention/sparse_mla_q8kv8_sm100.py
# Changes: calls the flash_mla op; pt_eL and kv4_sf required; TRT-LLM wrapper and packers removed.
# The full license text is in LICENSE-APACHE.

"""Sparse MLA over an fp4 KV pool with per-token e4m3 scales, SM100.

Q is e4m3 on all 576 dims (512 NoPE + 64 RoPE). The KV pool is read directly as the MMA operand, with no
dequantization: each row is 321 B instead of the 576 B of an e4m3 row,

  kv4       [s_kv, 256] uint8  the 512 NoPE dims as packed e2m1, two per byte, low nibble first,
                               stored as nope / (r_t * 2^pt_eL)
  kv4_rope  [s_kv, 64]  uint8  the 64 RoPE dims as e4m3, no scale
  kv4_sf    [s_kv]      uint8  r_t, one e4m3 per row, in (0, 1.75]

`pt_eL` is a per-layer exponent chosen by whoever writes the pool. It only has to keep r_t inside e4m3's
range (r_t <= 1.75 keeps the stored e4m3 P below 448) and cancels out of the result, but the kernel must
be told the value the pool was written with.

This op is ours, not upstream's: it lives beside `get_mla_decoding_metadata` in csrc_ops/.
"""

from typing import Optional, Tuple

import torch

from flash_mla._loader import get_ops

D_QK = 576
D_V = 512
_B_H = 64
# Split only long top-k lists, into chunks of at least 256 entries. Measured on B200: a split never
# paid off at topk=512, and at topk=2048 the best was the largest of 8 / 4 / 2 that keeps every work
# item on its own SM (decode batches of 1-64 query tokens).
_MIN_TOPK_TO_SPLIT = 1024
_MIN_SPLIT_LEN = 256

_NUM_SMS = {}


def _num_sms(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _NUM_SMS:
        _NUM_SMS[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _NUM_SMS[index]


def sparse_fp4kv_num_splits(s_q: int, h_q: int, topk: int, num_sms: int) -> int:
    """The top-k split `sparse_fp4kv_fwd` picks when `num_splits` is None: the largest of 8 / 4 / 2 that
    keeps every work item on its own SM, else 1. `h_q` is the padded head count (a multiple of 64)."""
    if topk < _MIN_TOPK_TO_SPLIT:
        return 1
    items = s_q * (h_q // _B_H)
    for splits in (8, 4, 2):
        if (
            topk % (64 * splits) == 0
            and topk // splits >= _MIN_SPLIT_LEN
            and items * splits <= num_sms
        ):
            return splits
    return 1


def _check_out(t: torch.Tensor, name: str, shape, dtype: torch.dtype, device: torch.device) -> None:
    if tuple(t.shape) != tuple(shape) or t.dtype != dtype or t.device != device or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous {tuple(shape)} {dtype} tensor on {device}, "
            f"got {tuple(t.shape)} {t.dtype} on {t.device}"
        )


def sparse_fp4kv_fwd(
    q: torch.Tensor,
    kv4: torch.Tensor,
    kv4_rope: torch.Tensor,
    kv4_sf: torch.Tensor,
    indices: torch.Tensor,
    pt_eL: float,
    bmm1_scale: float,
    bmm2_scale: float = 1.0,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    num_splits: Optional[int] = None,
    *,
    out: Optional[torch.Tensor] = None,
    max_logits: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse MLA on SM100 over the fp4 KV pool described in this module's docstring.

    V is the first 512 dims of each KV row. Indices < 0 or >= s_kv and positions >= topk_length are
    masked. Masked positions with an in-range index still load their row, which must not hold NaN.
    r_t and `pt_eL` are applied per token inside the kernel, so `bmm1_scale` / `bmm2_scale` are the
    scales an e4m3 KV row would use.

    Args:
        q:           [s_q, h_q, 576], float8_e4m3fn, NoPE then RoPE dims. h_q a multiple of 64, or a
                     divisor of 64 (padded to 64 here)
        kv4:         [s_kv, 256], uint8, contiguous
        kv4_rope:    [s_kv, 64], uint8, contiguous
        kv4_sf:      [s_kv], uint8, contiguous
        indices:     [s_q, topk] or [s_q, 1, topk], int32 rows of the pool; topk a multiple of 64
        pt_eL:       the per-layer exponent the pool was written with
        bmm1_scale:  q_scale * kv_scale * sm_scale
        bmm2_scale:  kv_scale of V
        attn_sink:   optional [h_q], float32. Multiplies the output by exp(lse) / (exp(lse) + exp(sink));
                     no effect on lse and max_logits
        topk_length: optional [s_q], int32 number of leading valid positions per row
        num_splits:  1, 2, 4 or 8 top-k splits; None picks `sparse_fp4kv_num_splits`
        out:         optional [s_q, h_q, 512] bfloat16 buffer for the output
        max_logits:  optional [s_q, h_q] float32 buffer
        lse:         optional [s_q, h_q] float32 buffer

    Returns:
        out:        [s_q, h_q, 512], bfloat16
        max_logits: [s_q, h_q], float32, -inf for rows without valid indices
        lse:        [s_q, h_q], float32, natural log without the sink, +inf for rows without valid indices
    """
    if q.dim() != 3 or q.shape[2] != D_QK:
        raise ValueError(f"q must be [s_q, h_q, {D_QK}], got {tuple(q.shape)}")
    if indices.dim() == 3:
        if indices.shape[1] != 1:
            raise ValueError("only h_kv == 1 is supported")
        indices = indices.squeeze(1)
    s_q, h_q_real, _ = q.shape
    topk = indices.shape[-1]
    device = q.device

    h_q = h_q_real
    out_user = max_logits_user = lse_user = None
    if h_q % _B_H != 0:
        if _B_H % h_q != 0:
            raise ValueError(f"h_q must be a multiple of 64 or divide 64, got {h_q}")
        q_pad = q.new_zeros(s_q, _B_H, D_QK)
        q_pad[:, :h_q] = q
        q, h_q = q_pad, _B_H
        if attn_sink is not None:
            attn_sink = torch.cat([attn_sink, attn_sink.new_full((_B_H - h_q_real,), float("-inf"))])
        out_user, max_logits_user, lse_user = out, max_logits, lse
        out = max_logits = lse = None

    if num_splits is None:
        num_splits = sparse_fp4kv_num_splits(s_q, h_q, topk, _num_sms(device))

    if out is None:
        out = torch.empty(s_q, h_q, D_V, dtype=torch.bfloat16, device=device)
    else:
        _check_out(out, "out", (s_q, h_q, D_V), torch.bfloat16, device)
    if max_logits is None:
        max_logits = torch.empty(s_q, h_q, dtype=torch.float32, device=device)
    else:
        _check_out(max_logits, "max_logits", (s_q, h_q), torch.float32, device)
    if lse is None:
        lse = torch.empty(s_q, h_q, dtype=torch.float32, device=device)
    else:
        _check_out(lse, "lse", (s_q, h_q), torch.float32, device)

    get_ops(use_legacy=False).sparse_fp4kv_fwd(
        q, kv4, kv4_rope, kv4_sf, indices, attn_sink, topk_length,
        out, max_logits, lse,
        float(bmm1_scale), float(bmm2_scale), float(pt_eL), int(num_splits),
    )

    if h_q != h_q_real:
        out, max_logits, lse = out[:, :h_q_real], max_logits[:, :h_q_real], lse[:, :h_q_real]
        if out_user is not None:
            out = out_user.copy_(out)
        if max_logits_user is not None:
            max_logits = max_logits_user.copy_(max_logits)
        if lse_user is not None:
            lse = lse_user.copy_(lse)
    return out, max_logits, lse
