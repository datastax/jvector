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

// AVX3_DL (Ice Lake / Icelake-SP) tier: ONLY kernels that require ICX-specific
// instructions unavailable in the AVX3 (-march=skylake-avx512) compilation
// belong here.  Generic kernels that Highway auto-vectorises identically under
// both marches must go in jvector_simd_kernels.cpp instead — that file is
// compiled once for AVX3 and its function pointers are reused by this tier and
// AVX3_SPR via vtable inheritance, avoiding any duplication in .text.
//
// ICX adds over AVX3 (HWY_TARGET_STR_AVX3_DL):
//   VNNI, VBMI, VBMI2, IFMA, BITALG, VPOPCNTDQ, GFNI, VAES, VPCLMULQDQ
//
// Compiled with -march=icelake-server.
// Highway will select HWY_AVX3_DL as the static target.
#include "jvector_simd.h"
#include "hwy/highway.h"
#include "assert_hwy_targets.h"

namespace hn = hwy::HWY_NAMESPACE;

namespace AVX3_DL {

} // namespace AVX3_DL
