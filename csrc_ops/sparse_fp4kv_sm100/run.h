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
Changes: declaration of the launcher defined in kernel.cu.
The full license text is in LICENSE-APACHE.
==============================================================================*/

#pragma once

#include "params.h"

namespace sm100::sparse_fp4kv {

// Launch the kernel (and the split merge when params.num_splits > 1) on params.stream.
// Defined in kernel.cu, the only translation unit nvcc compiles for this op.
void run_sparse_fp4kv_fwd_kernel(const SparseFp4KvFwdParams& params, int num_sms);

}  // namespace sm100::sparse_fp4kv
