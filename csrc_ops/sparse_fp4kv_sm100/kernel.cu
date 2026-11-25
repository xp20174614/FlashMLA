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
Changes: the template dispatch, split out of the TVM-FFI entry point.
The full license text is in LICENSE-APACHE.
==============================================================================*/

#include "kernel.cuh"
#include "run.h"

namespace sm100::sparse_fp4kv {

void run_sparse_fp4kv_fwd_kernel(const SparseFp4KvFwdParams& params, int num_sms) {
  if (params.topk_length != nullptr) {
    if (params.attn_sink != nullptr) {
      run_sparse_fp4kv_fwd<true, true>(params, num_sms);
    } else {
      run_sparse_fp4kv_fwd<true, false>(params, num_sms);
    }
  } else {
    if (params.attn_sink != nullptr) {
      run_sparse_fp4kv_fwd<false, true>(params, num_sms);
    } else {
      run_sparse_fp4kv_fwd<false, false>(params, num_sms);
    }
  }
}

}  // namespace sm100::sparse_fp4kv
