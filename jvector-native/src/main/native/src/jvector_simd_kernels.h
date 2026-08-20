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

// Header file for the SIMD kernels.
// Kernel declarations are auto-generated from jvector_simd_kernel_list.h.
// Architecture guards use the canonical macros from jvector_arch.h so that
// only the namespaces that exist for the current build target are declared.
#ifndef SIMD_KERNELS_H
#define SIMD_KERNELS_H

#include <cstddef>
#include <cstdint>
#include "jvector_arch.h"

// Macro to declare a kernel function signature from the kernel list.
#define KERNEL_ENTRY(ret_type, name, params, names) \
    ret_type name params;

// Declares all kernel signatures inside a namespace named ISA.
#define DECLARE_SIMD_KERNELS(ISA) \
    namespace ISA { \
    JVECTOR_SIMD_KERNEL_LIST \
    }

#include "jvector_simd_kernel_list.h"

#if JV_ARCH_X86_64
// x86-64 ISA namespaces (SSE4.2 baseline → AVX2 → AVX3 → Ice Lake → Sapphire Rapids)
DECLARE_SIMD_KERNELS(AVX3_SPR)
DECLARE_SIMD_KERNELS(AVX3_DL)
DECLARE_SIMD_KERNELS(AVX3)
DECLARE_SIMD_KERNELS(AVX2)
DECLARE_SIMD_KERNELS(SSE42)
#else // JV_ARCH_AARCH64
// AArch64 ISA namespaces (NEON baseline → SVE → SVE2)
DECLARE_SIMD_KERNELS(NEON)
DECLARE_SIMD_KERNELS(SVE)
DECLARE_SIMD_KERNELS(SVE2)
#endif // JV_ARCH_X86_64 / JV_ARCH_AARCH64

#undef KERNEL_ENTRY

#endif // SIMD_KERNELS_H

