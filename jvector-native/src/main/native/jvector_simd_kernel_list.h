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

// X-Macro list of all SIMD kernels.
// To add a new kernel, simply add a new KERNEL_ENTRY line here.
// The macro will auto-generate:
//   - Namespace declarations in jvector_simd_kernels.h
//   - KernelVTable struct members
//   - Vtable initializers for AVX3, AVX2, SSE42
//   - Public API wrapper functions
//   - Public C API declarations in jvector_simd.h
//
// KERNEL_ENTRY format:
//   KERNEL_ENTRY(return_type, function_name, (param_declarations), (param_names))
//   - param_declarations: full parameter list with types and names
//   - param_names: just the parameter names for forwarding calls

#ifndef JVECTOR_SIMD_KERNEL_LIST_H
#define JVECTOR_SIMD_KERNEL_LIST_H

// clang-format off
#define JVECTOR_SIMD_KERNEL_LIST \
    /* Vector similarity */ \
    KERNEL_ENTRY(float, cosine_f32, (const float *a, size_t aoffset, const float *b, size_t boffset, size_t length), (a, aoffset, b, boffset, length)) \
    KERNEL_ENTRY(float, dot_product_f32, (const float *a, size_t aoffset, const float *b, size_t boffset, size_t length), (a, aoffset, b, boffset, length)) \
    KERNEL_ENTRY(float, euclidean_f32, (const float *a, size_t aoffset, const float *b, size_t boffset, size_t length), (a, aoffset, b, boffset, length)) \
    /* Element-wise in-place arithmetic */ \
    KERNEL_ENTRY(void, add_in_place_f32, (float *v1, const float *v2, size_t length), (v1, v2, length)) \
    KERNEL_ENTRY(void, add_scalar_in_place_f32, (float *v1, float value, size_t length), (v1, value, length)) \
    KERNEL_ENTRY(void, sub_in_place_f32, (float *v1, const float *v2, size_t length), (v1, v2, length)) \
    KERNEL_ENTRY(void, sub_scalar_in_place_f32, (float *v1, float value, size_t length), (v1, value, length)) \
    KERNEL_ENTRY(float, max_f32, (const float *v, size_t length), (v, length)) \
    KERNEL_ENTRY(void, min_in_place_f32, (float *v1, const float *v2, size_t length), (v1, v2, length)) \
    /* PQ kernels */ \
    KERNEL_ENTRY(float, assemble_and_sum_f32, (const float *data, int dataBase, const unsigned char *baseOffsets, int baseOffsetsOffset, size_t baseOffsetsLength), (data, dataBase, baseOffsets, baseOffsetsOffset, baseOffsetsLength)) \
    KERNEL_ENTRY(float, assemble_and_sum_pq_f32, (const float *data, size_t subspaceCount, const unsigned char *baseOffsets1, int baseOffsetsOffset1, const unsigned char *baseOffsets2, int baseOffsetsOffset2, int clusterCount), (data, subspaceCount, baseOffsets1, baseOffsetsOffset1, baseOffsets2, baseOffsetsOffset2, clusterCount)) \
    KERNEL_ENTRY(float, pq_decoded_cosine_similarity_f32, (const unsigned char *baseOffsets, int baseOffsetsOffset, size_t baseOffsetsLength, int clusterCount, const float *partialSums, const float *aMagnitude, float bMagnitude), (baseOffsets, baseOffsetsOffset, baseOffsetsLength, clusterCount, partialSums, aMagnitude, bMagnitude)) \
    KERNEL_ENTRY(void, calculate_partial_sums_dot_f32, (const float *codebook, int codebookIndex, size_t size, int clusterCount, const float *query, int queryOffset, float *partialSums), (codebook, codebookIndex, size, clusterCount, query, queryOffset, partialSums)) \
    KERNEL_ENTRY(void, calculate_partial_sums_euclidean_f32, (const float *codebook, int codebookIndex, size_t size, int clusterCount, const float *query, int queryOffset, float *partialSums), (codebook, codebookIndex, size, clusterCount, query, queryOffset, partialSums)) \
    KERNEL_ENTRY(void, calculate_partial_sums_self_magnitude_f32, (const float *codebook, int codebookIndex, size_t size, int clusterCount, float *partialSums), (codebook, codebookIndex, size, clusterCount, partialSums)) \
    /* NVQ kernels */ \
    KERNEL_ENTRY(void, nvq_quantize_8bit, (const float *vector, size_t length, float alpha, float x0, float minValue, float maxValue, unsigned char *destination), (vector, length, alpha, x0, minValue, maxValue, destination)) \
    KERNEL_ENTRY(float, nvq_loss, (const float *vector, size_t length, float alpha, float x0, float minValue, float maxValue, int nBits), (vector, length, alpha, x0, minValue, maxValue, nBits)) \
    KERNEL_ENTRY(float, nvq_uniform_loss, (const float *vector, size_t length, float minValue, float maxValue, int nBits), (vector, length, minValue, maxValue, nBits)) \
    KERNEL_ENTRY(float, nvq_square_l2_distance_8bit, (const float *vector, const unsigned char *quantized, size_t length, float alpha, float x0, float minValue, float maxValue), (vector, quantized, length, alpha, x0, minValue, maxValue)) \
    KERNEL_ENTRY(float, nvq_dot_product_8bit, (const float *vector, const unsigned char *quantized, size_t length, float alpha, float x0, float minValue, float maxValue), (vector, quantized, length, alpha, x0, minValue, maxValue)) \
    KERNEL_ENTRY(int64_t, nvq_cosine_8bit_packed, (const float *vector, const unsigned char *quantized, size_t length, float alpha, float x0, float minValue, float maxValue, const float *centroid), (vector, quantized, length, alpha, x0, minValue, maxValue, centroid)) \
    KERNEL_ENTRY(void, nvq_shuffle_query_in_place_8bit, (float *vector, size_t length), (vector, length))
// clang-format on

#endif // JVECTOR_SIMD_KERNEL_LIST_H

