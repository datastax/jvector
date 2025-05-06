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

// check CPU support
bool check_compatibility(void);

//F32
float dot_product_f32(int preferred_size, const float* a, int aoffset, const float* b, int boffset, int length);
float euclidean_f32(int preferred_size, const float* a, int aoffset, const float* b, int boffset, int length);
void bulk_quantized_shuffle_dot_f32_512(const unsigned char* shuffles, int codebookCount, const char* quantizedPartials, float delta, float minDistance, float* results);
void bulk_quantized_shuffle_euclidean_f32_512(const unsigned char* shuffles, int codebookCount, const char* quantizedPartials, float delta, float minDistance, float* results);
void bulk_quantized_shuffle_cosine_f32_512(const unsigned char* shuffles, int codebookCount, const char* quantizedPartialSums, float sumDelta, float minDistance, const char* quantizedPartialMagnitudes, float magnitudeDelta, float minMagnitude, float queryMagnitudeSquared, float* results);
float assemble_and_sum_f32_512(const float* data, int dataBase, const unsigned char* baseOffsets, int baseOffsetsOffset, int baseOffsetsLength);
float pq_decoded_cosine_similarity_f32_512(const unsigned char* baseOffsets, int baseOffsetsOffset, int baseOffsetsLength, int clusterCount, const float* partialSums, const float* aMagnitude, float bMagnitude);
void calculate_partial_sums_dot_f32_512(const float* codebook, int codebookBase, int size, int clusterCount, const float* query, int queryOffset, float* partialSums);
void calculate_partial_sums_euclidean_f32_512(const float* codebook, int codebookBase, int size, int clusterCount, const float* query, int queryOffset, float* partialSums);
void calculate_partial_sums_best_dot_f32_512(const float* codebook, int codebookBase, int size, int clusterCount, const float* query, int queryOffset, float* partialSums, float* partialBestDistances);
void calculate_partial_sums_best_euclidean_f32_512(const float* codebook, int codebookBase, int size, int clusterCount, const float* query, int queryOffset, float* partialSums, float* partialBestDistances);
#endif
