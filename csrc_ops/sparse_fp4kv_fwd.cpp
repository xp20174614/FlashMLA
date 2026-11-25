/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modified for FlashMLA in 2026. Derived from SGLang's
python/sglang/kernels/jit/csrc/sparse_mla_q8kv8_sm100/entry.cuh
Changes: rewritten as a PyTorch stable-ABI op with argument checks.
The full license text is in LICENSE-APACHE.
==============================================================================*/

// Sparse MLA forward over the fp4 KV pool (per-token e4m3 scales), SM100.
//
// Q is e4m3 on all 576 dims. Each KV row is 321 B: the 512 NoPE dims as packed e2m1 (`kv4`, 256 B),
// the 64 RoPE dims as unscaled e4m3 (`kv4_rope`, 64 B) and one e4m3 per-token scale r_t (`kv4_sf`,
// 1 B). NoPE is stored as nope / (r_t * 2^pt_eL), with pt_eL a per-layer constant that cancels out of
// the result. The kernel reads the pool directly as its MMA operand; see
// sparse_fp4kv_sm100/kernel.cuh.
//
// The outputs are caller-owned (the Python wrapper allocates them, or takes the caller's buffers);
// the split partials are allocated here.
//
// This op is ours, not upstream's, which is why it lives outside csrc/: that directory is replaced
// wholesale on a sync.

#include <optional>

#include "api/common.h"
#include "sparse_fp4kv_sm100/run.h"

namespace {

constexpr int D_QK = 576;
constexpr int D_V = 512;
constexpr int B_H = 64;

}  // namespace

static void sparse_fp4kv_fwd(
    const Tensor &q,                          // [s_q, h_q, 576] e4m3, h_q a multiple of 64
    const Tensor &kv4,                        // [s_kv, 256] uint8, packed e2m1 NoPE
    const Tensor &kv4_rope,                   // [s_kv, 64] uint8, e4m3 RoPE, no scale
    const Tensor &kv4_sf,                     // [s_kv] uint8, e4m3 r_t
    const Tensor &indices,                    // [s_q, topk] int32
    const std::optional<Tensor> &attn_sink,   // [h_q] float32
    const std::optional<Tensor> &topk_length, // [s_q] int32
    Tensor out,                               // [s_q, h_q, 512] bf16
    Tensor max_logits,                        // [s_q, h_q] float32
    Tensor lse,                               // [s_q, h_q] float32
    double bmm1_scale,
    double bmm2_scale,
    double pt_eL,
    int64_t num_splits
) {
    Arch arch = Arch();
    STD_TORCH_CHECK(arch.major == 10 && arch.minor == 0,
                    "sparse_fp4kv_fwd is built on tcgen05 and TMA gather4 and validated on SM100 only");

    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(kv4, 2);
    KU_CHECK_NDIM(kv4_rope, 2);
    KU_CHECK_NDIM(kv4_sf, 1);
    KU_CHECK_NDIM(indices, 2);
    KU_CHECK_NDIM(attn_sink, 1);
    KU_CHECK_NDIM(topk_length, 1);

    const int s_q = static_cast<int>(q.size(0));
    const int h_q = static_cast<int>(q.size(1));
    const int s_kv = static_cast<int>(kv4.size(0));
    const int topk = static_cast<int>(indices.size(1));

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv4);
    KU_CHECK_DEVICE(kv4_rope);
    KU_CHECK_DEVICE(kv4_sf);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DEVICE(out);
    KU_CHECK_DEVICE(max_logits);
    KU_CHECK_DEVICE(lse);

    KU_CHECK_DTYPE(q, ScalarType::Float8_e4m3fn);
    KU_CHECK_DTYPE(kv4, ScalarType::Byte);
    KU_CHECK_DTYPE(kv4_rope, ScalarType::Byte);
    KU_CHECK_DTYPE(kv4_sf, ScalarType::Byte);
    KU_CHECK_DTYPE(indices, ScalarType::Int);
    KU_CHECK_DTYPE(attn_sink, ScalarType::Float);
    KU_CHECK_DTYPE(topk_length, ScalarType::Int);
    KU_CHECK_DTYPE(out, ScalarType::BFloat16);
    KU_CHECK_DTYPE(max_logits, ScalarType::Float);
    KU_CHECK_DTYPE(lse, ScalarType::Float);

    KU_CHECK_SHAPE(q, s_q, h_q, D_QK);
    KU_CHECK_SHAPE(kv4, s_kv, D_V / 2);
    KU_CHECK_SHAPE(kv4_rope, s_kv, D_QK - D_V);
    KU_CHECK_SHAPE(kv4_sf, s_kv);
    KU_CHECK_SHAPE(indices, s_q, topk);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(topk_length, s_q);
    KU_CHECK_SHAPE(out, s_q, h_q, D_V);
    KU_CHECK_SHAPE(max_logits, s_q, h_q);
    KU_CHECK_SHAPE(lse, s_q, h_q);

    // The pool rows are fixed-stride TMA rows; the outputs and the partials are written as dense arrays.
    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_CONTIGUOUS(kv4);
    KU_CHECK_CONTIGUOUS(kv4_rope);
    KU_CHECK_CONTIGUOUS(kv4_sf);
    KU_CHECK_CONTIGUOUS(attn_sink);
    KU_CHECK_CONTIGUOUS(topk_length);
    KU_CHECK_CONTIGUOUS(out);
    KU_CHECK_CONTIGUOUS(max_logits);
    KU_CHECK_CONTIGUOUS(lse);

    STD_TORCH_CHECK(h_q > 0 && h_q % B_H == 0, "h_q must be a multiple of 64 (pad Q), got ", h_q);
    STD_TORCH_CHECK(topk > 0 && topk % 64 == 0, "topk must be a multiple of 64, got ", topk);
    STD_TORCH_CHECK(num_splits == 1 || num_splits == 2 || num_splits == 4 || num_splits == 8,
                    "num_splits must be 1, 2, 4 or 8, got ", num_splits);
    STD_TORCH_CHECK(topk % (64 * num_splits) == 0,
                    "topk ", topk, " does not split into ", num_splits, " chunks of a multiple of 64");
    const void *out_ptrs[3] = {out.data_ptr(), max_logits.data_ptr(), lse.data_ptr()};
    STD_TORCH_CHECK(out_ptrs[0] != out_ptrs[1] && out_ptrs[0] != out_ptrs[2] && out_ptrs[1] != out_ptrs[2],
                    "out, max_logits and lse must not alias");

    torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());

    using namespace sm100::sparse_fp4kv;
    SparseFp4KvFwdParams p{};
    p.s_q = s_q;
    p.s_kv = s_kv;
    p.h_q = h_q;
    p.topk = topk;
    p.num_splits = static_cast<int32_t>(num_splits);
    p.split_len = static_cast<int32_t>(topk / num_splits);
    p.scale_log2 = static_cast<float>(bmm1_scale * LOG_2_E);
    p.bmm2_scale = static_cast<float>(bmm2_scale);
    p.pt_eL = static_cast<float>(pt_eL);

    p.q = static_cast<const uint8_t *>(q.data_ptr());
    p.kv4 = static_cast<const uint8_t *>(kv4.data_ptr());
    p.kv4_sf = static_cast<const uint8_t *>(kv4_sf.data_ptr());
    p.kv4_rope = static_cast<const uint8_t *>(kv4_rope.data_ptr());
    p.indices = static_cast<const int32_t *>(indices.data_ptr());
    p.attn_sink = ku::get_optional_tensor_ptr<const float>(attn_sink);
    p.topk_length = ku::get_optional_tensor_ptr<const int32_t>(topk_length);

    p.stride_q_s_q = q.stride(0);
    p.stride_q_h_q = q.stride(1);
    p.stride_indices_s_q = indices.stride(0);

    // Kept alive until the launch has been enqueued; the caching allocator orders their reuse on the stream.
    std::optional<Tensor> partial_out, partial_max_logits, partial_lse;
    if (num_splits > 1) {
        // The main kernel writes the partials, the merge kernel the final outputs.
        partial_out = torch::stable::new_empty(q, {num_splits, s_q, h_q, D_V}, ScalarType::BFloat16);
        partial_max_logits = torch::stable::new_empty(q, {num_splits, s_q, h_q}, ScalarType::Float);
        partial_lse = torch::stable::new_empty(q, {num_splits, s_q, h_q}, ScalarType::Float);
        p.out = static_cast<cutlass::bfloat16_t *>(partial_out->data_ptr());
        p.max_logits = static_cast<float *>(partial_max_logits->data_ptr());
        p.lse = static_cast<float *>(partial_lse->data_ptr());
        p.merged_out = static_cast<cutlass::bfloat16_t *>(out.data_ptr());
        p.merged_max_logits = static_cast<float *>(max_logits.data_ptr());
        p.merged_lse = static_cast<float *>(lse.data_ptr());
    } else {
        p.out = static_cast<cutlass::bfloat16_t *>(out.data_ptr());
        p.max_logits = static_cast<float *>(max_logits.data_ptr());
        p.lse = static_cast<float *>(lse.data_ptr());
        p.merged_out = nullptr;
        p.merged_max_logits = nullptr;
        p.merged_lse = nullptr;
    }
    p.stream = get_current_stream();

    run_sparse_fp4kv_fwd_kernel(p, arch.num_sms);
}

STABLE_TORCH_LIBRARY_FRAGMENT(flash_mla, m) {
    m.def(
        "sparse_fp4kv_fwd(Tensor q, Tensor kv4, Tensor kv4_rope, Tensor kv4_sf, Tensor indices, "
        "Tensor? attn_sink, Tensor? topk_length, Tensor(a!) out, Tensor(b!) max_logits, Tensor(c!) lse, "
        "float bmm1_scale, float bmm2_scale, float pt_eL, int num_splits) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(flash_mla, CUDA, m) {
    m.impl("sparse_fp4kv_fwd", TORCH_BOX(&sparse_fp4kv_fwd));
}
