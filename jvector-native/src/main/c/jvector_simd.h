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
#ifdef __cplusplus
extern "C" {
#endif
/* PQ kernels */
JVECTOR_SIMD_API float assemble_and_sum_f32(const float* data, int dataBase, const unsigned char* baseOffsets, int baseOffsetsOffset, size_t baseOffsetsLength);
JVECTOR_SIMD_API float assemble_and_sum_pq_f32(const float* data, size_t subspaceCount, const unsigned char* baseOffsets1, int baseOffsetsOffset1, const unsigned char* baseOffsets2, int baseOffsetsOffset2, int clusterCount);
JVECTOR_SIMD_API float pq_decoded_cosine_similarity_f32(const unsigned char* baseOffsets, int baseOffsetsOffset, size_t baseOffsetsLength, int clusterCount, const float* partialSums, const float* aMagnitude, float bMagnitude);
JVECTOR_SIMD_API void calculate_partial_sums_euclidean_f32(const float* codebook, int codebookBase, size_t size, int clusterCount, const float* query, int queryOffset, float* partialSums);
JVECTOR_SIMD_API void calculate_partial_sums_dot_f32(const float* codebook, int codebookBase, size_t size, int clusterCount, const float* query, int queryOffset, float* partialSums);
JVECTOR_SIMD_API void calculate_partial_sums_self_magnitude_f32(const float* codebook, int codebookBase, size_t size, int clusterCount, float* partialSums);

/* Vector similarity kernels */
JVECTOR_SIMD_API float dot_product_f32(const float* a, size_t aoffset, const float* b, size_t boffset, size_t length);
JVECTOR_SIMD_API float cosine_f32(const float* a, size_t aoffset, const float* b, size_t boffset, size_t length);
JVECTOR_SIMD_API float euclidean_f32(const float* a, size_t aoffset, const float* b, size_t boffset, size_t length);

/* NVQ kernels */
JVECTOR_SIMD_API void    nvq_quantize_8bit(const float* vector, size_t length, float alpha, float x0, float minValue, float maxValue, unsigned char* destination);
JVECTOR_SIMD_API float   nvq_loss(const float* vector, size_t length, float alpha, float x0, float minValue, float maxValue, int nBits);
JVECTOR_SIMD_API float   nvq_uniform_loss(const float* vector, size_t length, float minValue, float maxValue, int nBits);
JVECTOR_SIMD_API float   nvq_square_l2_distance_8bit(const float* vector, const unsigned char* quantized, size_t length, float alpha, float x0, float minValue, float maxValue);
JVECTOR_SIMD_API float   nvq_dot_product_8bit(const float* vector, const unsigned char* quantized, size_t length, float alpha, float x0, float minValue, float maxValue);
JVECTOR_SIMD_API int64_t nvq_cosine_8bit_packed(const float* vector, const unsigned char* quantized, size_t length, float alpha, float x0, float minValue, float maxValue, const float* centroid);
JVECTOR_SIMD_API void    nvq_shuffle_query_in_place_8bit(float* vector, size_t length);

/* Element-wise in-place vector arithmetic */
JVECTOR_SIMD_API void    add_in_place_f32(float* v1, const float* v2, size_t length);
JVECTOR_SIMD_API void    add_scalar_in_place_f32(float* v1, float value, size_t length);
JVECTOR_SIMD_API void    sub_in_place_f32(float* v1, const float* v2, size_t length);
JVECTOR_SIMD_API void    sub_scalar_in_place_f32(float* v1, float value, size_t length);
JVECTOR_SIMD_API float   max_f32(const float* v, size_t length);
JVECTOR_SIMD_API void    min_in_place_f32(float* v1, const float* v2, size_t length);
#ifdef __cplusplus
}
#endif // extern "C"
#endif // VECTOR_SIMD_DOT_H
