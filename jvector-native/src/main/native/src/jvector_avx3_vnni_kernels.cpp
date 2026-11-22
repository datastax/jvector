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

// AVX3 CPUs with AVX-512 VNNI but not the rest of the Ice Lake feature set (Cascade Lake,
// Cooper Lake): INT8 query projection only. Every other slot keeps the AVX3 kernels.
// Compiled with -march=cascadelake. Highway selects HWY_AVX3, whose SumOfMulQuadAccumulate
// emulates the dot product, so the kernel's dot step uses vpdpbusd directly; the rest is
// the same Highway code as the AVX3_DL tier, with 256-bit tiles.
#include "jvector_simd.h"
#include "hwy/highway.h"
#include "assert_hwy_targets.h"

namespace hn = hwy::HWY_NAMESPACE;

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <limits>

namespace AVX3_VNNI {
#define JV_ASH_I8_WIDE 0
#define JV_ASH_I8_DOT(d32, a, b, sum) hn::Vec<I8Tag>{_mm256_dpbusd_epi32((sum).raw, (a).raw, (b).raw)}
#include "ash_projection_i8-inl.h"
#undef JV_ASH_I8_DOT
#undef JV_ASH_I8_WIDE
} // namespace AVX3_VNNI
