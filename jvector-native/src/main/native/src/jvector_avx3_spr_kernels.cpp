/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// AVX3_SPR (Sapphire Rapids) tier: ONLY kernels that require SPR-specific
// instructions unavailable in AVX3_DL (-march=icelake-server) belong here
// (e.g. native FP16 arithmetic via avx512fp16).  Generic kernels go in
// jvector_simd_kernels.cpp; ICX-only kernels go in jvector_avx3_dl_kernels.cpp.
// Both are inherited via the vtable chain AVX3 → AVX3_DL → AVX3_SPR, so their
// function pointers are reused here at zero additional .text cost.
//
// Compiled with -march=sapphirerapids.
// Highway will select HWY_AVX3_SPR as the static target.
#include "jvector_simd.h"
#include "hwy/highway.h"
#include "assert_hwy_targets.h"

namespace hn = hwy::HWY_NAMESPACE;

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace AVX3_SPR {
// INT8 query projection on 512-bit registers (see ash_projection_i8-inl.h).
#define JV_ASH_I8_WIDE 1
#include "ash_projection_i8-inl.h"
#undef JV_ASH_I8_WIDE
} // namespace AVX3_SPR
