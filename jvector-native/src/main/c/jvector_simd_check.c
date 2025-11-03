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

#include <cpuid.h>
#include "jvector_simd.h"

bool check_compatibility(void) {
    unsigned int eax, ebx, ecx, edx;
    bool avx512f_supported = false, avx512cd_supported = false,
         avx512bw_supported = false, avx512dq_supported = false,
         avx512vl_supported = false;

    // Check for AVX-512 Foundation (AVX-512F) and other AVX-512 features:
    // These are indicated by various bits of EBX from leaf 7, sub-leaf 0.
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        avx512f_supported = ebx & (1 << 16);     // AVX-512F
        avx512cd_supported = ebx & (1 << 28);    // AVX-512CD
        avx512bw_supported = ebx & (1 << 30);    // AVX-512BW
        avx512dq_supported = ebx & (1 << 17);    // AVX-512DQ
        avx512vl_supported = ebx & (1 << 31);    // AVX-512VL
    }

    return avx512f_supported && avx512cd_supported && avx512bw_supported && avx512dq_supported && avx512vl_supported;
}