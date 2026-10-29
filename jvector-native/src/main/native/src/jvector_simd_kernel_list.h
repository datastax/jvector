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
//   - Vtable initializers for AVX3_SPR, AVX3_DL, AVX3, AVX2, SSE42
//   - Public API wrapper functions
//   - Public C API declarations in jvector_simd.h
//
// KERNEL_ENTRY format:
//   KERNEL_ENTRY(return_type, function_name, (param_declarations), (param_names))
//   - param_declarations: full parameter list with types and names
//   - param_names: just the parameter names for forwarding calls

#ifndef JVECTOR_SIMD_KERNEL_LIST_H
#define JVECTOR_SIMD_KERNEL_LIST_H

// ASH C=1 ABI contract:
//   codes: ceil(capacity/B) complete blocks, each ceil(groups/2)*B + 5*B bytes.
//   lut:   groups*16 read-only floats, out: capacity writable floats, no aliasing.
//   groups/capacity are positive, B is 8/16/32. Inputs use FusedASHLayout bytes.
//   Return 0 on success, 1 for a nonzero landmark, 2 for an invalid shape/pointer.
//   Output may be partially written on error. No allocation, pointer retention,
//   callbacks, or blocking are permitted: ordinary graph calls use critical FFM.

// ASH projection ABI: positive rows/cols, borrowed contiguous row-major matrix,
// query[cols], out[rows]; output must not alias either read-only input.
// Callers validate buffer sizes; the kernel is a synchronous allocation-free leaf.

// clang-format off
#define JVECTOR_SIMD_KERNEL_LIST \
    /* Vector similarity */ \
    KERNEL_ENTRY(float, cosine_f32, (const float *a, size_t aoffset, const float *b, size_t boffset, size_t length), (a, aoffset, b, boffset, length)) \
    KERNEL_ENTRY(float, dot_product_f32, (const float *a, size_t aoffset, const float *b, size_t boffset, size_t length), (a, aoffset, b, boffset, length)) \
    KERNEL_ENTRY(float, euclidean_f32, (const float *a, size_t aoffset, const float *b, size_t boffset, size_t length), (a, aoffset, b, boffset, length)) \
    /* Element-wise arithmetic and reductions */ \
    KERNEL_ENTRY(void, add_in_place_f32, (float *v1, const float *v2, size_t length), (v1, v2, length)) \
    KERNEL_ENTRY(void, add_scalar_in_place_f32, (float *v1, float value, size_t length), (v1, value, length)) \
    KERNEL_ENTRY(void, sub_in_place_f32, (float *v1, const float *v2, size_t length), (v1, v2, length)) \
    KERNEL_ENTRY(void, sub_scalar_in_place_f32, (float *v1, float value, size_t length), (v1, value, length)) \
    KERNEL_ENTRY(void, sub_f32, (const float *a, size_t aoffset, const float *b, size_t boffset, float *res, size_t length), (a, aoffset, b, boffset, res, length)) \
    KERNEL_ENTRY(void, sub_scalar_f32, (const float *a, size_t aoffset, float value, float *res, size_t length), (a, aoffset, value, res, length)) \
    KERNEL_ENTRY(void, scale_f32, (float *v, float multiplier, size_t length), (v, multiplier, length)) \
    KERNEL_ENTRY(float, max_f32, (const float *v, size_t length), (v, length)) \
    KERNEL_ENTRY(float, min_f32, (const float *v, size_t length), (v, length)) \
    KERNEL_ENTRY(void, min_in_place_f32, (float *v1, const float *v2, size_t length), (v1, v2, length)) \
    KERNEL_ENTRY(float, sum_f32, (const float *v, size_t length), (v, length)) \
    /* PQ training */ \
    KERNEL_ENTRY(void, scale_to_f32, (const float *source, float *destination, size_t destinationOffset, float multiplier, size_t length), (source, destination, destinationOffset, multiplier, length)) \
    KERNEL_ENTRY(void, subtract_add_in_place_f32, (float *oldSum, float *newSum, const float *point, size_t length), (oldSum, newSum, point, length)) \
    /* PQ kernels */ \
    KERNEL_ENTRY(int, closest_centroid_f32, (const float *point, size_t pointOffset, const float *centroids, int centroidCount, size_t dimension), (point, pointOffset, centroids, centroidCount, dimension)) \
    KERNEL_ENTRY(float, assemble_and_sum_f32, (const float *data, int dataBase, const unsigned char *baseOffsets, int baseOffsetsOffset, size_t baseOffsetsLength), (data, dataBase, baseOffsets, baseOffsetsOffset, baseOffsetsLength)) \
    KERNEL_ENTRY(float, assemble_and_sum_pq_f32, (const float *data, size_t subspaceCount, const unsigned char *baseOffsets1, int baseOffsetsOffset1, const unsigned char *baseOffsets2, int baseOffsetsOffset2, int clusterCount), (data, subspaceCount, baseOffsets1, baseOffsetsOffset1, baseOffsets2, baseOffsetsOffset2, clusterCount)) \
    KERNEL_ENTRY(float, pq_decoded_cosine_similarity_f32, (const unsigned char *baseOffsets, int baseOffsetsOffset, size_t baseOffsetsLength, int clusterCount, const float *partialSums, const float *aMagnitude, float bMagnitude), (baseOffsets, baseOffsetsOffset, baseOffsetsLength, clusterCount, partialSums, aMagnitude, bMagnitude)) \
    KERNEL_ENTRY(void, calculate_partial_sums_dot_f32, (const float *codebook, int codebookIndex, size_t size, int clusterCount, const float *query, int queryOffset, float *partialSums), (codebook, codebookIndex, size, clusterCount, query, queryOffset, partialSums)) \
    KERNEL_ENTRY(void, calculate_partial_sums_euclidean_f32, (const float *codebook, int codebookIndex, size_t size, int clusterCount, const float *query, int queryOffset, float *partialSums), (codebook, codebookIndex, size, clusterCount, query, queryOffset, partialSums)) \
    KERNEL_ENTRY(void, calculate_partial_sums_self_magnitude_f32, (const float *codebook, int codebookIndex, size_t size, int clusterCount, float *partialSums), (codebook, codebookIndex, size, clusterCount, partialSums)) \
    /* ASH preparation and exact packed construction scoring; synchronous leaf calls. */ \
    KERNEL_ENTRY(void, ash_project_f32, (const float *matrix, const float *query, int rows, int cols, float *out), (matrix, query, rows, cols, out)) \
    /* Whole-neighborhood C=1 ASH, one downcall including headers and similarity. */ \
    KERNEL_ENTRY(int, ash_score_neighborhood_c1_f32, (const unsigned char *codes, const float *lut, float landmarkDot, int groups, int blockSize, int capacity, float *out), (codes, lut, landmarkDot, groups, blockSize, capacity, out)) \
    /* NVQ kernels */ \
    KERNEL_ENTRY(void, nvq_quantize_8bit, (const float *vector, size_t length, float alpha, float x0, float minValue, float maxValue, unsigned char *destination), (vector, length, alpha, x0, minValue, maxValue, destination)) \
    KERNEL_ENTRY(float, nvq_loss, (const float *vector, size_t length, float alpha, float x0, float minValue, float maxValue, int nBits), (vector, length, alpha, x0, minValue, maxValue, nBits)) \
    KERNEL_ENTRY(float, nvq_uniform_loss, (const float *vector, size_t length, float minValue, float maxValue, int nBits), (vector, length, minValue, maxValue, nBits)) \
    KERNEL_ENTRY(float, nvq_growth_rate, (const float *vector, size_t length, float minValue, float maxValue, int nBits), (vector, length, minValue, maxValue, nBits)) \
    KERNEL_ENTRY(float, nvq_square_l2_distance_8bit, (const float *vector, const unsigned char *quantized, size_t length, float alpha, float x0, float minValue, float maxValue), (vector, quantized, length, alpha, x0, minValue, maxValue)) \
    KERNEL_ENTRY(float, nvq_dot_product_8bit, (const float *vector, const unsigned char *quantized, size_t length, float alpha, float x0, float minValue, float maxValue), (vector, quantized, length, alpha, x0, minValue, maxValue)) \
    KERNEL_ENTRY(int64_t, nvq_cosine_8bit_packed, (const float *vector, const unsigned char *quantized, size_t length, float alpha, float x0, float minValue, float maxValue, const float *centroid), (vector, quantized, length, alpha, x0, minValue, maxValue, centroid)) \
    KERNEL_ENTRY(void, nvq_shuffle_query_in_place_8bit, (float *vector, size_t length), (vector, length))
    /* ── ADD NEW KERNEL_ENTRY LINES ABOVE THIS LINE ── */
// clang-format on

#endif // JVECTOR_SIMD_KERNEL_LIST_H

