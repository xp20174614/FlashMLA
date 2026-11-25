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
# test/registered/kernels/ops/attention/test_sparse_mla_q8kv8_sm100.py
# Changes: calls flash_mla.sparse_fp4kv_fwd; SGLang-layout cases removed; new cases added.
# The full license text is in LICENSE-APACHE.

"""Correctness of flash_mla.sparse_fp4kv_fwd, the SM100 sparse MLA over the fp4 KV pool.

The reference is exact fp32 attention over the same pool: the e4m3 rows are packed with `pack_pool` and
expanded back with `unpack_pool`, so the quantization of the cache cancels out and what is left is the
kernel's own error (the e4m3 rounding of P and the bf16 output).

`pack_pool` is a reference packer only. A serving stack writes the pool in its own KV-store kernel and
passes the per-layer exponent it used as `pt_eL`.

Needs an SM100 GPU and the compiled flash_mla.  python -m pytest tests/test_sparse_fp4kv.py
"""
import math

import pytest
import torch

import flash_mla

D_QK, D_V = flash_mla.sparse_fp4kv.D_QK, flash_mla.sparse_fp4kv.D_V

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="sparse_fp4kv_fwd needs an SM100 GPU",
)

# The e4m3 rounding of P dominates the output error; max_logits and lse stay in fp32.
OUT_TOL = 3e-2
STAT_TOL = 1e-3

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@torch.no_grad()
def pack_pool(kv: torch.Tensor, rt_max: float = 1.75):
    """kv [s_kv, 576] (natural scale) -> (kv4 [s_kv, 256], kv4_rope [s_kv, 64], kv4_sf [s_kv], eL).

    r_t = smallest e4m3 >= amax_t / 6 / 2^eL, with eL per pool so that max r_t <= rt_max (P r_t <= 448).
    """
    nope = kv[:, :D_V].float()
    rope = kv[:, D_V:].float()
    amax = nope.abs().amax(dim=1)
    want = amax / 6.0
    wmax = float(want.max())
    eL = math.ceil(math.log2(wmax / rt_max)) if wmax > 0 else 0
    while True:
        w = want / 2.0 ** eL
        rt = w.to(torch.float8_e4m3fn).float()
        nxt = (rt.to(torch.float8_e4m3fn).view(torch.uint8) + 1).view(torch.float8_e4m3fn).float()
        rt = torch.where(rt < w, nxt, rt)
        if float(rt.max()) <= rt_max:
            break
        eL += 1
    rt = torch.where(amax > 0, rt, torch.full_like(rt, 2.0 ** -9))
    table = torch.tensor(_E2M1, device=kv.device)
    scaled = nope / (rt * 2.0 ** eL).unsqueeze(1)
    mids = (table[1:] + table[:-1]) / 2
    mag = torch.bucketize(scaled.abs().clamp(max=6.0), mids).to(torch.uint8)
    code = mag | ((scaled < 0).to(torch.uint8) << 3)
    kv4 = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    kv4_rope = rope.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    kv4_sf = rt.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    return kv4, kv4_rope, kv4_sf, eL


@torch.no_grad()
def unpack_pool(kv4, kv4_rope, kv4_sf, eL) -> torch.Tensor:
    """The exact values the kernel multiplies with, float32 [s_kv, 576]."""
    table = torch.tensor(_E2M1, device=kv4.device)

    def dec(c):
        m = table[(c & 7).long()]
        return torch.where((c & 8) != 0, -m, m)

    rt = kv4_sf.view(torch.float8_e4m3fn).float() * 2.0 ** eL
    nope = torch.empty(kv4.shape[0], D_V, device=kv4.device)
    nope[:, 0::2] = dec(kv4 & 0xF)
    nope[:, 1::2] = dec(kv4 >> 4)
    rope = kv4_rope.view(torch.float8_e4m3fn).float()
    return torch.cat([nope * rt.unsqueeze(1), rope], dim=1)


def _fwd(q8, pool, indices, bmm1, bmm2=1.0, attn_sink=None, topk_length=None, num_splits=None, **kw):
    kv4, kv4_rope, kv4_sf, eL = pool
    return flash_mla.sparse_fp4kv_fwd(
        q8, kv4, kv4_rope, kv4_sf, indices, float(eL), bmm1, bmm2,
        attn_sink, topk_length, num_splits, **kw,
    )


def _assert_close(name, actual, expected, tol):
    actual, expected = actual.float(), expected.float()
    finite = torch.isfinite(expected)
    assert torch.equal(torch.isfinite(actual), finite), f"{name}: non-finite entries differ"
    assert torch.equal(actual[~finite], expected[~finite]), f"{name}: inf values differ"
    if finite.any():
        err = ((actual[finite] - expected[finite]).norm() / expected[finite].norm().clamp(min=1e-12)).item()
        assert err < tol, f"{name}: rel_l2 {err:.3e} >= {tol}"


def _quant(x):
    scale = (x.float().abs().amax() / 448.0).clamp(min=1e-8)
    return (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale.item()


def _make_case(s_q, s_kv, h_q, topk, invalid_frac, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(s_q, h_q, D_QK, generator=g, device="cuda") * 0.3).bfloat16()
    kv = (torch.randn(s_kv, D_QK, generator=g, device="cuda") * 0.3).bfloat16()
    kv[:, D_V:] *= 2.0  # RoPE dims with a different range than NoPE, as in real activations
    q8, q_scale = _quant(q)
    # The KV rows are packed at their natural scale, as a serving pool sees them; r_t absorbs the amplitude.
    k8 = kv.float().to(torch.float8_e4m3fn)
    keys = torch.rand(s_q, s_kv, generator=g, device="cuda")
    indices = keys.topk(topk, dim=-1).indices.to(torch.int32)
    if invalid_frac > 0:
        bad = torch.rand(s_q, topk, generator=g, device="cuda") < invalid_frac
        coin = torch.rand(s_q, topk, generator=g, device="cuda") < 0.5
        junk = torch.where(coin, -1, s_kv + 7).to(torch.int32)
        indices = torch.where(bad, junk, indices)
    bmm1 = q_scale / math.sqrt(192 + 64)
    pool = pack_pool(k8)
    return q8, indices, bmm1, pool, unpack_pool(*pool)


def _reference(q8, kf, indices, bmm1, bmm2, attn_sink=None, topk_length=None):
    s_q, s_kv = q8.shape[0], kf.shape[0]
    qf = q8.float()
    outs, max_logits, lses = [], [], []
    for i in range(0, s_q, 16):
        rows = indices[i : i + 16]
        valid = (rows >= 0) & (rows < s_kv)
        if topk_length is not None:
            pos = torch.arange(rows.shape[1], device=rows.device)
            valid &= pos[None, :] < topk_length[i : i + 16, None]
        k_rows = kf[rows.clamp(0, s_kv - 1).long()]
        logits = torch.einsum("qhd,qkd->qhk", qf[i : i + 16], k_rows) * bmm1
        logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
        lse = torch.logsumexp(logits, -1)
        denom = lse if attn_sink is None else torch.logaddexp(lse, attn_sink[None, :])
        p = torch.exp(logits - denom[..., None]).nan_to_num(0.0)
        outs.append(torch.einsum("qhk,qkd->qhd", p, k_rows[..., :D_V]) * bmm2)
        max_logits.append(logits.amax(-1))
        lses.append(lse.masked_fill(lse == float("-inf"), float("inf")))
    return torch.cat(outs), torch.cat(max_logits), torch.cat(lses)


# name -> shape and features; `splits` lists the num_splits values to run (None: chosen by the op).
CASES = {
    "topk2048": dict(s_q=64, s_kv=4096, h_q=64, topk=2048, splits=(None, 1, 2, 4)),
    "decode_split8": dict(s_q=4, s_kv=16384, h_q=64, topk=2048, splits=(None, 8)),
    "decode_split_sink_topk_length": dict(
        s_q=9, s_kv=8192, h_q=64, topk=2048, sink=True, topk_len=True, invalid_frac=0.2, splits=(8, 4, 2),
    ),
    "topk512": dict(s_q=64, s_kv=20480, h_q=64, topk=512, splits=(None, 1, 2)),
    "topk64": dict(s_q=64, s_kv=4096, h_q=64, topk=64),
    "topk2176": dict(s_q=48, s_kv=8192, h_q=64, topk=2176, splits=(None, 2)),
    "invalid_indices": dict(s_q=64, s_kv=4096, h_q=64, topk=1024, invalid_frac=0.3, splits=(1, 2)),
    "mostly_invalid": dict(s_q=32, s_kv=4096, h_q=64, topk=1024, invalid_frac=0.97, splits=(1, 2, 4)),
    "multi_wave_split": dict(s_q=200, s_kv=4096, h_q=64, topk=1024, splits=(2,)),
    "sink_topk_length": dict(
        s_q=64, s_kv=4096, h_q=64, topk=1024, invalid_frac=0.1, sink=True, topk_len=True, splits=(1, 2, 4),
    ),
    "h128": dict(s_q=40, s_kv=4096, h_q=128, topk=512, splits=(1, 2)),
    "h32_padded": dict(s_q=40, s_kv=4096, h_q=32, topk=512, sink=True),
    "s_kv_equals_topk": dict(s_q=64, s_kv=2048, h_q=64, topk=2048),
    "persistent_items": dict(s_q=600, s_kv=4096, h_q=64, topk=512, topk_len=True, sink=True),
    "persistent_topk2048": dict(s_q=400, s_kv=8192, h_q=64, topk=2048, invalid_frac=0.05),
}


@pytest.mark.parametrize("name", list(CASES))
def test_matches_reference(name):
    case = dict(CASES[name])
    splits = case.pop("splits", (None,))
    sink, topk_len = case.pop("sink", False), case.pop("topk_len", False)
    invalid_frac = case.pop("invalid_frac", 0.0)
    s_q, h_q, topk = case["s_q"], case["h_q"], case["topk"]
    q8, indices, bmm1, pool, pool_ref = _make_case(**case, invalid_frac=invalid_frac, seed=0)
    attn_sink = topk_length = None
    if sink:
        g = torch.Generator(device="cuda").manual_seed(1)
        attn_sink = torch.randn(h_q, generator=g, device="cuda")
        attn_sink[:3] = float("inf")  # zeroes the row
        attn_sink[3:6] = float("-inf")  # no-op
    if topk_len:
        g = torch.Generator(device="cuda").manual_seed(2)
        topk_length = torch.randint(0, topk + 1, (s_q,), generator=g, device="cuda", dtype=torch.int32)
        topk_length[:2] = 0
        topk_length[2:4] = topk
    expected = _reference(q8, pool_ref, indices, bmm1, 1.0, attn_sink, topk_length)
    for num_splits in splits:
        actual = _fwd(q8, pool, indices, bmm1, 1.0, attn_sink, topk_length, num_splits)
        for label, a, e, tol in zip(("out", "max_logits", "lse"), actual, expected, (OUT_TOL, STAT_TOL, STAT_TOL)):
            _assert_close(f"{name} splits={num_splits} {label}", a, e, tol)


def test_3d_indices():
    q8, indices, bmm1, pool, _ = _make_case(16, 4096, 64, 512, 0.0, seed=5)
    flat = _fwd(q8, pool, indices, bmm1)
    for a, b in zip(flat, _fwd(q8, pool, indices.unsqueeze(1), bmm1)):
        assert torch.equal(a, b)


@pytest.mark.parametrize("h_q", [32, 64])
def test_caller_owned_outputs(h_q):
    q8, indices, bmm1, pool, _ = _make_case(40, 4096, h_q, 512, 0.0, seed=3)
    expected = _fwd(q8, pool, indices, bmm1)
    buffers = (
        torch.empty(40, h_q, D_V, dtype=torch.bfloat16, device="cuda"),
        torch.empty(40, h_q, dtype=torch.float32, device="cuda"),
        torch.empty(40, h_q, dtype=torch.float32, device="cuda"),
    )
    results = _fwd(q8, pool, indices, bmm1, out=buffers[0], max_logits=buffers[1], lse=buffers[2])
    for result, buffer, ref in zip(results, buffers, expected):
        assert result.data_ptr() == buffer.data_ptr()
        assert torch.equal(buffer, ref)


def test_pack_pool_accepts_large_amax():
    """A pool whose NoPE amax far exceeds 6 must pack and round-trip, not clip: r_t and eL absorb the
    amplitude, and the fp4 round-trip error stays flat across a 300x amplitude change."""
    g = torch.Generator(device="cuda").manual_seed(11)
    base = None
    for sigma in (0.3, 100.0):
        kv = (torch.randn(64, D_QK, generator=g, device="cuda") * sigma).to(torch.float8_e4m3fn)
        pool = pack_pool(kv)
        rt = pool[2].view(torch.float8_e4m3fn).float()
        assert torch.isfinite(rt).all() and rt.max() <= 1.75 + 1e-6
        ref = unpack_pool(*pool)
        err = ((ref - kv.float()).norm() / kv.float().norm()).item()
        assert err < 1.5e-1, f"sigma {sigma}: fp4 round trip rel_l2 {err:.3e}"
        base = err if base is None else base
        assert abs(err - base) < 3e-2, f"round-trip error moved with amplitude: {base:.3f} -> {err:.3f}"


def test_rejects_misaligned_indices():
    q8, indices, bmm1, pool, _ = _make_case(8, 4096, 64, 512, 0.0, seed=4)
    storage = torch.empty(8 * 512 + 1, dtype=torch.int32, device="cuda")
    shifted = storage[1:].view(8, 512)
    shifted.copy_(indices)
    with pytest.raises(Exception, match="must be aligned"):
        _fwd(q8, pool, shifted, bmm1)


def test_rejects_bad_inputs():
    q8, indices, bmm1, pool, _ = _make_case(8, 4096, 64, 512, 0.0, seed=6)
    kv4, kv4_rope, kv4_sf, eL = pool
    with pytest.raises(Exception, match="kv4_sf"):
        flash_mla.sparse_fp4kv_fwd(q8, kv4, kv4_rope, kv4_sf[:-1], indices, float(eL), bmm1)
    with pytest.raises(Exception, match="num_splits"):
        _fwd(q8, pool, indices, bmm1, num_splits=3)
    with pytest.raises(Exception, match="topk"):
        _fwd(q8, pool, indices[:, :100].contiguous(), bmm1)


def test_cuda_graph_replay_matches_eager():
    q8, indices, bmm1, pool, _ = _make_case(16, 8192, 64, 2048, 0.1, seed=7)
    topk_length = torch.full((16,), 1500, dtype=torch.int32, device="cuda")
    eager = _fwd(q8, pool, indices, bmm1, topk_length=topk_length)[0]
    replayed = torch.empty_like(eager)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed.copy_(_fwd(q8, pool, indices, bmm1, topk_length=topk_length)[0])
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replayed, eager)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
