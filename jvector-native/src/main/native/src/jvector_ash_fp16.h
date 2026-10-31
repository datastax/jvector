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
#pragma once
#include <stdint.h>

#if defined(_WIN32)
#define JV_ASH_FP16_EXPORT __declspec(dllexport)
#else
#define JV_ASH_FP16_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum jvector_ash_fp16_status {
    JV_ASH_FP16_OK = 0,
    JV_ASH_FP16_UNSUPPORTED = 1,
    JV_ASH_FP16_INVALID_ARGUMENT = 2,
    JV_ASH_FP16_RANGE = 3,
    JV_ASH_FP16_NO_MEMORY = 4,
    JV_ASH_FP16_ERROR = 5
};

/* Caller-owned status storage avoids a second, thread-affine last-error call.
 * In particular, diagnostics remain correct if a Java virtual thread migrates. */
typedef struct jvector_ash_fp16_result {
    int status;
    char message[256];
} jvector_ash_fp16_result;

JV_ASH_FP16_EXPORT int jvector_ash_fp16_abi(void);
/* Copies a row-major [rows][columns] FP32 matrix and prepares immutable FP16
 * weights. The caller retains its input; result is required and always written.
 * Successful contexts may be used concurrently, but must not be destroyed while
 * a call is in flight. No C++ exception crosses this ABI. */
JV_ASH_FP16_EXPORT void* jvector_ash_fp16_create(const float* matrix, int rows,
                                               int columns, jvector_ash_fp16_result* result);
JV_ASH_FP16_EXPORT void jvector_ash_fp16_destroy(void* context);
/* Retained packed weight bytes, excluding per-calling-thread workspace. */
JV_ASH_FP16_EXPORT uint64_t jvector_ash_fp16_bytes(const void* context);
/* input has columns floats, output has rows floats; neither may alias the other.
 * Finite operands must be within the FP16 finite range. Output is FP32. */
JV_ASH_FP16_EXPORT int jvector_ash_fp16_project(void* context, const float* input,
                                              float* output, jvector_ash_fp16_result* result);

#ifdef __cplusplus
}
#endif
