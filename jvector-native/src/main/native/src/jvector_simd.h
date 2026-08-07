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

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef VECTOR_SIMD_DOT_H
#define VECTOR_SIMD_DOT_H

// Mark a symbol as part of the public ABI even when the library is built
// with -fvisibility=hidden.
#define JVECTOR_SIMD_API __attribute__((visibility("default")))

// APIs exposed to Java via FFI
// Auto-generated from jvector_simd_kernel_list.h
#ifdef __cplusplus
extern "C" {
#endif

#define KERNEL_ENTRY(ret_type, name, params, names) \
    JVECTOR_SIMD_API ret_type name params;

#include "jvector_simd_kernel_list.h"

JVECTOR_SIMD_KERNEL_LIST

#undef KERNEL_ENTRY

// Returns the name of the ISA tier that was selected at library init time by
// dispatch_kernels().  Useful for diagnostics, tests, and logging.
// Possible return values: "avx3_spr", "avx3_dl", "avx3", "avx2", "sse42".
// The returned pointer is a string literal; do not free it.
JVECTOR_SIMD_API const char *jvector_simd_get_active_isa(void);

// Returns the value of JVECTOR_MAX_ISA that was read at library init time, or
// NULL if the variable was absent or contained an unrecognised value.
// Possible non-null return values: "avx3_spr", "avx3_dl", "avx3", "avx2", "sse42".
// The returned pointer (when non-null) is a string literal; do not free it.
JVECTOR_SIMD_API const char *jvector_simd_get_max_isa_env(void);

#ifdef __cplusplus
}
#endif // extern "C"
#endif // VECTOR_SIMD_DOT_H

