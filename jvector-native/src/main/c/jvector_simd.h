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

#ifndef VECTOR_SIMD_DOT_H
#define VECTOR_SIMD_DOT_H

#define JV_INLINE static inline
#define JV_FINLINE static inline __attribute__((always_inline))
// check CPU support
bool check_compatibility(void);

// APIs exposed to Java via FFI
float assemble_and_sum_f32_512(const float* data, int dataBase, const unsigned char* baseOffsets, int baseOffsetsOffset, int baseOffsetsLength);
float pq_decoded_cosine_similarity_f32_512(const unsigned char* baseOffsets, int baseOffsetsOffset, int baseOffsetsLength, int clusterCount, const float* partialSums, const float* aMagnitude, float bMagnitude);
void calculate_partial_sums_f32_512(const float* codebook, int codebookBase, int size, int clusterCount, const float* query, int queryOffset, int similarityFunction, float* partialSums);
#endif
