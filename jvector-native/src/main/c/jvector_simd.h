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

void bulk_quantized_shuffle_dot(const unsigned char* shuffles, int codebookCount, const char* quantizedPartials, float delta, float minDistance, float* results);
void bulk_quantized_shuffle_euclidean(const unsigned char* shuffles, int codebookCount, const char* quantizedPartials, float delta, float minDistance, float* results);
void bulk_quantized_shuffle_cosine(const unsigned char* shuffles, int codebookCount, const char* quantizedPartialSums, float sumDelta, float minDistance, const char* quantizedPartialMagnitudes, float magnitudeDelta, float minMagnitude, float queryMagnitudeSquared, float* results);
#endif
