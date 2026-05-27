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

// Runtime SIMD dispatch: selects AVX3 (AVX-512), AVX2, or SSE42 at startup.
// SSE42 is the baseline and is assumed always available without a CPUID check.
// AVX2 and AVX3 are probed via CPUID.  Function pointers are resolved once at
// static-init time; each public call is a single indirect branch.
#include "jvector_simd.h"
#include "jvector_simd_kernels.h" // AVX3::, AVX2::, SSE42:: kernel declarations
#include "jvector_cpuFeatures.h"  // populate_cpu_features(), CpuFeature enum

#include <array>
#include <cstdlib> // std::getenv
#include <cstring> // std::strcmp

// Everything in this anonymous namespace is translation-unit private; none of
// these symbols are exported from the shared library.
namespace {

// Enumerates the ISA tiers in ascending capability order so that numeric
// comparisons (e.g. max_isa != MaxIsa::AVX2) correctly gate higher tiers.
// Unset (-1) means "no override; use best available CPU capability".
enum class MaxIsa { Unset = -1, SSE42 = 0, AVX2 = 1, AVX3 = 2 };

// Reads the JVECTOR_MAX_ISA environment variable and maps it to a MaxIsa
// value.  This lets callers cap the ISA at runtime without recompiling —
// useful for benchmarking or working around CPU errata.
// Accepted values (case-sensitive): "avx3", "avx2", "sse42".
static MaxIsa read_max_isa() noexcept
{
    const char *val = std::getenv("JVECTOR_MAX_ISA");
    if (!val) return MaxIsa::Unset;
    if (std::strcmp(val, "avx3")  == 0) return MaxIsa::AVX3;
    if (std::strcmp(val, "avx2")  == 0) return MaxIsa::AVX2;
    if (std::strcmp(val, "sse42") == 0) return MaxIsa::SSE42;
    return MaxIsa::Unset; // unrecognised value: ignore and use CPU detection
}

// KernelVTable holds one function pointer per public kernel.  Storing them
// together in a struct means a single pointer comparison during dispatch_kernels()
// selects all kernels for the chosen ISA in one shot.
struct KernelVTable {
    /* Vector similarity */
    float (*cosine_f32)(const float *, size_t, const float *, size_t, size_t);
    float (*dot_product_f32)(const float *, size_t, const float *, size_t, size_t);
    float (*euclidean_f32)(const float *, size_t, const float *, size_t, size_t);
    /* Element-wise in-place arithmetic */
    void  (*add_in_place_f32)(float *, const float *, size_t);
    void  (*add_scalar_in_place_f32)(float *, float, size_t);
    void  (*sub_in_place_f32)(float *, const float *, size_t);
    void  (*sub_scalar_in_place_f32)(float *, float, size_t);
    float (*max_f32)(const float *, size_t);
    void  (*min_in_place_f32)(float *, const float *, size_t);
    /* PQ kernels */
    float (*assemble_and_sum_f32)(const float *, int,
                                  const unsigned char *, int, size_t);
    float (*assemble_and_sum_pq_f32)(const float *, size_t,
                                     const unsigned char *, int,
                                     const unsigned char *, int, int);
    float (*pq_decoded_cosine_similarity_f32)(const unsigned char *, int,
                                              size_t, int,
                                              const float *, const float *,
                                              float);
    void  (*calculate_partial_sums_dot_f32)(const float *, int,
                                            size_t, int,
                                            const float *, int, float *);
    void  (*calculate_partial_sums_euclidean_f32)(const float *, int,
                                                  size_t, int,
                                                  const float *, int, float *);
    void  (*calculate_partial_sums_self_magnitude_f32)(const float *, int,
                                                       size_t, int, float *);
    /* NVQ kernels */
    void    (*nvq_quantize_8bit)(const float *, size_t,
                                 float, float, float, float,
                                 unsigned char *);
    float   (*nvq_loss)(const float *, size_t,
                        float, float, float, float, int);
    float   (*nvq_uniform_loss)(const float *, size_t, float, float, int);
    float   (*nvq_square_l2_distance_8bit)(const float *,
                                           const unsigned char *, size_t,
                                           float, float, float, float);
    float   (*nvq_dot_product_8bit)(const float *,
                                    const unsigned char *, size_t,
                                    float, float, float, float);
    int64_t (*nvq_cosine_8bit_packed)(const float *,
                                      const unsigned char *, size_t,
                                      float, float, float, float,
                                      const float *);
    void    (*nvq_shuffle_query_in_place_8bit)(float *, size_t);
};

// One pre-filled vtable per ISA.  These are constant data; no heap allocation.
#define DEFINE_ISA_VTABLE(ISA)                                          \
    static const KernelVTable ISA##_vtable = {                          \
        ISA::cosine_f32,                                                 \
        ISA::dot_product_f32,                                            \
        ISA::euclidean_f32,                                              \
        ISA::add_in_place_f32,                                           \
        ISA::add_scalar_in_place_f32,                                    \
        ISA::sub_in_place_f32,                                           \
        ISA::sub_scalar_in_place_f32,                                    \
        ISA::max_f32,                                                    \
        ISA::min_in_place_f32,                                           \
        ISA::assemble_and_sum_f32,                                       \
        ISA::assemble_and_sum_pq_f32,                                    \
        ISA::pq_decoded_cosine_similarity_f32,                           \
        ISA::calculate_partial_sums_dot_f32,                             \
        ISA::calculate_partial_sums_euclidean_f32,                       \
        ISA::calculate_partial_sums_self_magnitude_f32,                  \
        ISA::nvq_quantize_8bit,                                          \
        ISA::nvq_loss,                                                   \
        ISA::nvq_uniform_loss,                                           \
        ISA::nvq_square_l2_distance_8bit,                                \
        ISA::nvq_dot_product_8bit,                                       \
        ISA::nvq_cosine_8bit_packed,                                     \
        ISA::nvq_shuffle_query_in_place_8bit,                            \
    }

DEFINE_ISA_VTABLE(AVX3);
DEFINE_ISA_VTABLE(AVX2);
DEFINE_ISA_VTABLE(SSE42);

// Selects and returns the best vtable for the current CPU and environment.
// Called exactly once during static initialisation (before main()).
static KernelVTable dispatch_kernels() noexcept
{
    // Check whether the caller has capped the ISA via the environment variable.
    const MaxIsa max_isa = read_max_isa();

    // Populate a boolean feature array by issuing CPUID and reading XCR0.
    std::array<bool, static_cast<uint32_t>(CpuFeature::COUNT)> features;
    populate_cpu_features(features);

    auto has = [&](CpuFeature f) noexcept {
        return features[static_cast<uint32_t>(f)];
    };

    // AVX3 tier requires the full Skylake-AVX512 (SKX) baseline:
    // AVX512F (foundation) + BW (byte/word) + CD (conflict detect)
    // + DQ (dword/qword) + VL (vector length extensions).
    if (max_isa != MaxIsa::SSE42 && max_isa != MaxIsa::AVX2
        && has(CpuFeature::AVX512F) && has(CpuFeature::AVX512BW)
        && has(CpuFeature::AVX512CD) && has(CpuFeature::AVX512DQ)
        && has(CpuFeature::AVX512VL)) {
        return AVX3_vtable;
    }
    // AVX2 tier: 256-bit integer/FP SIMD, available on Haswell and later.
    if (max_isa != MaxIsa::SSE42 && has(CpuFeature::AVX2)) {
        return AVX2_vtable;
    }
    // SSE42 is the baseline — assumed always present, no CPUID check needed.
    return SSE42_vtable;
}

// 'kernels' is initialised once at static-init time to the vtable chosen by
// dispatch_kernels().  After that every public API call goes through one
// indirect branch to the right ISA implementation — no runtime comparisons.
static const KernelVTable kernels = dispatch_kernels();

} // namespace

// ---- Public API ------------------------------------------------------------
// Each function is a thin wrapper that forwards to the pre-resolved function
// pointer in `kernels`.

/* Vector similarity */

float cosine_f32(const float *a, size_t aoffset,
                 const float *b, size_t boffset, size_t length)
{
    return kernels.cosine_f32(a, aoffset, b, boffset, length);
}

float dot_product_f32(const float *a, size_t aoffset,
                      const float *b, size_t boffset, size_t length)
{
    return kernels.dot_product_f32(a, aoffset, b, boffset, length);
}

float euclidean_f32(const float *a, size_t aoffset,
                    const float *b, size_t boffset, size_t length)
{
    return kernels.euclidean_f32(a, aoffset, b, boffset, length);
}

/* Element-wise in-place arithmetic */

void add_in_place_f32(float *v1, const float *v2, size_t length)
{
    kernels.add_in_place_f32(v1, v2, length);
}

void add_scalar_in_place_f32(float *v1, float value, size_t length)
{
    kernels.add_scalar_in_place_f32(v1, value, length);
}

void sub_in_place_f32(float *v1, const float *v2, size_t length)
{
    kernels.sub_in_place_f32(v1, v2, length);
}

void sub_scalar_in_place_f32(float *v1, float value, size_t length)
{
    kernels.sub_scalar_in_place_f32(v1, value, length);
}

float max_f32(const float *v, size_t length)
{
    return kernels.max_f32(v, length);
}

void min_in_place_f32(float *v1, const float *v2, size_t length)
{
    kernels.min_in_place_f32(v1, v2, length);
}

/* PQ kernels */

float assemble_and_sum_f32(const float *data, int dataBase,
                            const unsigned char *baseOffsets,
                            int baseOffsetsOffset, size_t baseOffsetsLength)
{
    return kernels.assemble_and_sum_f32(
            data, dataBase, baseOffsets, baseOffsetsOffset, baseOffsetsLength);
}

float assemble_and_sum_pq_f32(const float *data, size_t subspaceCount,
                               const unsigned char *baseOffsets1,
                               int baseOffsetsOffset1,
                               const unsigned char *baseOffsets2,
                               int baseOffsetsOffset2, int clusterCount)
{
    return kernels.assemble_and_sum_pq_f32(data, subspaceCount,
                                           baseOffsets1, baseOffsetsOffset1,
                                           baseOffsets2, baseOffsetsOffset2,
                                           clusterCount);
}

float pq_decoded_cosine_similarity_f32(const unsigned char *baseOffsets,
                                        int baseOffsetsOffset,
                                        size_t baseOffsetsLength,
                                        int clusterCount,
                                        const float *partialSums,
                                        const float *aMagnitude,
                                        float bMagnitude)
{
    return kernels.pq_decoded_cosine_similarity_f32(baseOffsets,
                                                    baseOffsetsOffset,
                                                    baseOffsetsLength,
                                                    clusterCount,
                                                    partialSums,
                                                    aMagnitude,
                                                    bMagnitude);
}

void calculate_partial_sums_dot_f32(const float *codebook, int codebookIndex,
                                     size_t size, int clusterCount,
                                     const float *query, int queryOffset,
                                     float *partialSums)
{
    kernels.calculate_partial_sums_dot_f32(codebook, codebookIndex,
                                           size, clusterCount,
                                           query, queryOffset, partialSums);
}

void calculate_partial_sums_euclidean_f32(const float *codebook,
                                           int codebookIndex,
                                           size_t size, int clusterCount,
                                           const float *query, int queryOffset,
                                           float *partialSums)
{
    kernels.calculate_partial_sums_euclidean_f32(codebook, codebookIndex,
                                                 size, clusterCount,
                                                 query, queryOffset,
                                                 partialSums);
}

void calculate_partial_sums_self_magnitude_f32(const float *codebook,
                                               int codebookIndex,
                                               size_t size, int clusterCount,
                                               float *partialSums)
{
    kernels.calculate_partial_sums_self_magnitude_f32(codebook, codebookIndex,
                                                      size, clusterCount,
                                                      partialSums);
}

/* NVQ kernels */

void nvq_quantize_8bit(const float *vector, size_t length,
                        float alpha, float x0,
                        float minValue, float maxValue,
                        unsigned char *destination)
{
    kernels.nvq_quantize_8bit(vector, length, alpha, x0,
                              minValue, maxValue, destination);
}

float nvq_loss(const float *vector, size_t length,
               float alpha, float x0,
               float minValue, float maxValue, int nBits)
{
    return kernels.nvq_loss(vector, length, alpha, x0, minValue, maxValue, nBits);
}

float nvq_uniform_loss(const float *vector, size_t length,
                        float minValue, float maxValue, int nBits)
{
    return kernels.nvq_uniform_loss(vector, length, minValue, maxValue, nBits);
}

float nvq_square_l2_distance_8bit(const float *vector,
                                   const unsigned char *quantized,
                                   size_t length,
                                   float alpha, float x0,
                                   float minValue, float maxValue)
{
    return kernels.nvq_square_l2_distance_8bit(vector, quantized, length,
                                               alpha, x0, minValue, maxValue);
}

float nvq_dot_product_8bit(const float *vector,
                            const unsigned char *quantized,
                            size_t length,
                            float alpha, float x0,
                            float minValue, float maxValue)
{
    return kernels.nvq_dot_product_8bit(vector, quantized, length,
                                        alpha, x0, minValue, maxValue);
}

int64_t nvq_cosine_8bit_packed(const float *vector,
                                const unsigned char *quantized,
                                size_t length,
                                float alpha, float x0,
                                float minValue, float maxValue,
                                const float *centroid)
{
    return kernels.nvq_cosine_8bit_packed(vector, quantized, length,
                                          alpha, x0, minValue, maxValue,
                                          centroid);
}

void nvq_shuffle_query_in_place_8bit(float *vector, size_t length)
{
    kernels.nvq_shuffle_query_in_place_8bit(vector, length);
}
