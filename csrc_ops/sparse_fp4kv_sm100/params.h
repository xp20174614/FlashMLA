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
python/sglang/kernels/jit/csrc/sparse_mla_q8kv8_sm100/params.h
Changes: namespace and struct renamed, doc comment updated.
The full license text is in LICENSE-APACHE.
==============================================================================*/

#pragma once

#include <cutlass/bfloat16.h>

#include <cstdint>
#include <cuda_runtime.h>

namespace sm100::sparse_fp4kv {

/**
 * \brief Launch parameters of the SM100 sparse MLA kernel over the fp4 KV pool.
 *
 * Q is e4m3 on all 576 dims (512 NoPE + 64 RoPE). The KV row is the fp4 pool below; its per-token scale
 * and the per-layer exponent are applied in the kernel, so the scales are those of the e4m3 kernel:
 * logits = bmm1_scale * (q8 . k), out = bmm2_scale * softmax(logits) . v.
 */
struct SparseFp4KvFwdParams {
  int32_t s_q, s_kv, h_q, topk;
  // The top-k of a query is cut into num_splits chunks of split_len (a multiple of 64) that run as
  // separate work items; with num_splits > 1, out / max_logits / lse are [num_splits, s_q, h_q, ...]
  // partials without the sink, merged by a second kernel.
  int32_t num_splits, split_len;
  float scale_log2;  // bmm1_scale * log2(e)
  float bmm2_scale;
  float pt_eL;  // per-layer exponent of the pool: NoPE was stored divided by r_t * 2^pt_eL

  const uint8_t* __restrict__ q;            // [s_q, h_q, 576] e4m3, strides in bytes
  // fp4 KV pool, the MMA operand itself (no dequantization anywhere), 321 B/row instead of 576:
  //   kv4       [s_kv, 256] packed e2m1 of NoPE / (r_t * 2^pt_eL), read by TMA 16U4_ALIGN16B
  //                         (1 B per element nominal in smem)
  //   kv4_rope  [s_kv,  64] e4m3 RoPE dims, no scale (an ALIGN16B map cannot be 64 elements wide)
  //   kv4_sf    [s_kv]      e4m3 r_t in (0, 1.75], loaded by the mask warp; required
  const uint8_t* __restrict__ kv4;
  const uint8_t* __restrict__ kv4_sf;
  const uint8_t* __restrict__ kv4_rope;
  const int32_t* __restrict__ indices;      // [s_q, topk]
  const float* __restrict__ attn_sink;      // [h_q] or nullptr
  const int32_t* __restrict__ topk_length;  // [s_q] or nullptr

  int64_t stride_q_s_q, stride_q_h_q;
  int64_t stride_indices_s_q;

  cutlass::bfloat16_t* __restrict__ out;  // [(num_splits,) s_q, h_q, 512], contiguous
  float* __restrict__ max_logits;         // [(num_splits,) s_q, h_q]
  float* __restrict__ lse;                // [(num_splits,) s_q, h_q]

  // Final outputs of the split merge (num_splits > 1 only).
  cutlass::bfloat16_t* __restrict__ merged_out;
  float* __restrict__ merged_max_logits;
  float* __restrict__ merged_lse;

  cudaStream_t stream;
};

}  // namespace sm100::sparse_fp4kv
