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

// Each tier is compiled with a fixed -march= flag that pins HWY_STATIC_TARGET
// to exactly one Highway constant — assert that precisely, matching the x86
// assertions above.
//   neon:  -march=armv8-a+crypto  → HWY_NEON  (no BF16/dotprod/I8MM, so never HWY_NEON_BF16)
//   sve:   -march=armv8.4-a+sve   → HWY_SVE   (no fixed-width hint, so never HWY_SVE_256)
//   sve2:  -march=armv9-a+sve2    → HWY_SVE2  (no fixed-width hint, so never HWY_SVE2_128)
#if defined(JV_REQUIRE_HWY_SVE2)
#  if HWY_STATIC_TARGET != HWY_SVE2
#    error "Highway did not select HWY_SVE2 for the SVE2 build. Check compiler flags (-march=armv9-a+sve2), compiler support (GCC >= 10 or Clang >= 22), and Highway blocklists."
#  endif
#elif defined(JV_REQUIRE_HWY_SVE)
#  if HWY_STATIC_TARGET != HWY_SVE
#    error "Highway did not select HWY_SVE for the SVE build. Check compiler flags (-march=armv8.4-a+sve), compiler support (GCC >= 10 or Clang >= 9), and Highway blocklists."
#  endif
#elif defined(JV_REQUIRE_HWY_NEON)
#  if HWY_STATIC_TARGET != HWY_NEON
#    error "Highway did not select HWY_NEON for the NEON build. Check compiler flags (-march=armv8-a+crypto) and Highway blocklists."
#  endif
#endif // JV_REQUIRE_HWY_*

#endif // JV_ARCH_X86_64 / JV_ARCH_AARCH64
