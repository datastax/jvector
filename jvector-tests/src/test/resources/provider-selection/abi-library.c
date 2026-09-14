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

// Complete symbol table for selection probes; dot product and sum are exercised.
// Compile with ABI_FIRST/ABI_SECOND to simulate a supported set, or OMIT_ABI_QUERY.
#include <stddef.h>
#include <stdint.h>
#ifndef OMIT_ABI_QUERY
#ifndef ABI_FIRST
#define ABI_FIRST 1
#endif
#ifndef ABI_SECOND
#define ABI_SECOND 3
#endif
int32_t jvector_supports_abi(int32_t abi) {
    return abi > 0 && (abi == ABI_FIRST || abi == ABI_SECOND);
}
#endif
const char *jvector_simd_get_active_isa(void) { return "selection-test"; }
const char *jvector_simd_get_max_isa_env(void) { return NULL; }
float dot_product_f32(const float *a, size_t ao, const float *b, size_t bo, size_t n) {
    float sum = 0;
    for (size_t i = 0; i < n; i++) sum += a[ao + i] * b[bo + i];
    return sum;
}
void cosine_f32(void) {}
void euclidean_f32(void) {}
void add_in_place_f32(void) {}
void add_scalar_in_place_f32(void) {}
void sub_in_place_f32(void) {}
void sub_scalar_in_place_f32(void) {}
void sub_f32(void) {}
void sub_scalar_f32(void) {}
void scale_f32(void) {}
void max_f32(void) {}
void min_f32(void) {}
void min_in_place_f32(void) {}
float sum_f32(const float *v, size_t n) {
    float sum = 0;
    for (size_t i = 0; i < n; i++) sum += v[i];
    return sum;
}
void assemble_and_sum_f32(void) {}
void assemble_and_sum_pq_f32(void) {}
void pq_decoded_cosine_similarity_f32(void) {}
void calculate_partial_sums_dot_f32(void) {}
void calculate_partial_sums_euclidean_f32(void) {}
void calculate_partial_sums_self_magnitude_f32(void) {}
void nvq_quantize_8bit(void) {}
void nvq_uniform_loss(void) {}
void nvq_square_l2_distance_8bit(void) {}
void nvq_dot_product_8bit(void) {}
void nvq_cosine_8bit_packed(void) {}
void nvq_shuffle_query_in_place_8bit(void) {}
void nvq_loss(void) {}

// Additional kernels required by the Highway batching bindings.
void closest_centroid_f32(void) {}
void nvq_growth_rate(void) {}
void scale_to_f32(void) {}
void subtract_add_in_place_f32(void) {}
