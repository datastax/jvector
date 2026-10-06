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

#include "jvector_arch.h"

#if JV_ARCH_X86_64

#if defined(JV_REQUIRE_HWY_AVX3_SPR)
#  if HWY_STATIC_TARGET != HWY_AVX3_SPR
#    error "Highway did not select HWY_AVX3_SPR for the Sapphire Rapids build. Check compiler flags, compiler support, and Highway blocklists."
#  endif
#elif defined(JV_REQUIRE_HWY_AVX3_DL)
#  if HWY_STATIC_TARGET != HWY_AVX3_DL
#    error "Highway did not select HWY_AVX3_DL for the Ice Lake build. Check compiler flags, compiler support, and Highway blocklists."
#  endif
#elif defined(JV_REQUIRE_HWY_AVX3)
#  if HWY_STATIC_TARGET != HWY_AVX3
#    error "Highway did not select HWY_AVX3 for the AVX-512 build. Check compiler flags, compiler support, and Highway blocklists."
#  endif
#elif defined(JV_REQUIRE_HWY_AVX2)
#  if HWY_STATIC_TARGET != HWY_AVX2
#    error "Highway did not select HWY_AVX2 for the AVX2 build. Check compiler flags, compiler support, and Highway blocklists."
#  endif
#endif // JV_REQUIRE_HWY_*

#elif JV_ARCH_AARCH64

// Compiler flags per tier and the Highway target each must produce:
//   neon:  -march=armv8-a+crypto   → HWY_NEON
//          (no BF16/dotprod/I8MM features, so never HWY_NEON_BF16)
//   sve:   -march=armv8.4-a+sve    → HWY_SVE  (scalable, VL-agnostic)
//          (Graviton 3 / Neoverse V1 and later)
//   sve2:  -march=armv9-a+sve2     → HWY_SVE2 (scalable, VL-agnostic)
//          (Graviton 4 / Neoverse V2/N2 and later)
#if defined(JV_REQUIRE_HWY_SVE2)
#  if HWY_STATIC_TARGET != HWY_SVE2
#    error "Highway did not select HWY_SVE2 for the SVE2 build. Check compiler flags (-march=armv9-a+sve2) and Highway blocklists."
#  endif
#elif defined(JV_REQUIRE_HWY_SVE)
#  if HWY_STATIC_TARGET != HWY_SVE
#    error "Highway did not select HWY_SVE for the SVE build. Check compiler flags (-march=armv8.4-a+sve) and Highway blocklists."
#  endif
#elif defined(JV_REQUIRE_HWY_NEON)
#  if HWY_STATIC_TARGET != HWY_NEON
#    error "Highway did not select HWY_NEON for the NEON build. Check compiler flags (-march=armv8-a+crypto) and Highway blocklists."
#  endif
#endif // JV_REQUIRE_HWY_*

#endif // JV_ARCH_X86_64 / JV_ARCH_AARCH64
