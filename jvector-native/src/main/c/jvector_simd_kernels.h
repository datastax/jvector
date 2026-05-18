// headerfile for the simd kernels
#ifndef SIMD_KERNELS_H
#define SIMD_KERNELS_H

#include <cstddef>
#include <cstdint>

#define DECLARE_SIMD_KERNELS(ISA) \
    namespace ISA { \
    /* Base Fp32 distance kernels */ \
    float cosine_f32(const float *a, \
                     size_t aoffset, \
                     const float *b, \
                     size_t boffset, \
                     size_t length); \
    float dot_product_f32(const float *a, \
                          size_t aoffset, \
                          const float *b, \
                          size_t boffset, \
                          size_t length); \
    float euclidean_f32(const float *a, \
                        size_t aoffset, \
                        const float *b, \
                        size_t boffset, \
                        size_t length); \
    /* Element-wise in-place arithmetic */ \
    void add_in_place_f32(float *v1, \
                          const float *v2, \
                          size_t length); \
    void add_scalar_in_place_f32(float *v1, \
                                 float value, \
                                 size_t length); \
    void sub_in_place_f32(float *v1, \
                          const float *v2, \
                          size_t length); \
    void sub_scalar_in_place_f32(float *v1, \
                                 float value, \
                                 size_t length); \
    float max_f32(const float *v, \
                  size_t length); \
    void min_in_place_f32(float *v1, \
                          const float *v2, \
                          size_t length); \
    /* PQ kernels */ \
    float assemble_and_sum_f32(const float *data, \
                               int dataBase, \
                               const unsigned char *baseOffsets, \
                               int baseOffsetsOffset, \
                               size_t baseOffsetsLength); \
    float assemble_and_sum_pq_f32(const float *data, \
                                  size_t subspaceCount, \
                                  const unsigned char *baseOffsets1, \
                                  int baseOffsetsOffset1, \
                                  const unsigned char *baseOffsets2, \
                                  int baseOffsetsOffset2, \
                                  int clusterCount); \
    float pq_decoded_cosine_similarity_f32(const unsigned char *baseOffsets, \
                                           int baseOffsetsOffset, \
                                           size_t baseOffsetsLength, \
                                           int clusterCount, \
                                           const float *partialSums, \
                                           const float *aMagnitude, \
                                           float bMagnitude); \
    void calculate_partial_sums_dot_f32(const float *codebook, \
                                        int codebookIndex, \
                                        size_t size, \
                                        int clusterCount, \
                                        const float *query, \
                                        int queryOffset, \
                                        float *partialSums); \
    void calculate_partial_sums_euclidean_f32(const float *codebook, \
                                              int codebookIndex, \
                                              size_t size, \
                                              int clusterCount, \
                                              const float *query, \
                                              int queryOffset, \
                                              float *partialSums); \
    void calculate_partial_sums_self_magnitude_f32(const float *codebook, \
                                                   int codebookIndex, \
                                                   size_t size, \
                                                   int clusterCount, \
                                                   float *partialSums); \
    /* NVQ kernels */ \
    void nvq_quantize_8bit(const float *vector, \
                           size_t length, \
                           float alpha, float x0, \
                           float minValue, float maxValue, \
                           unsigned char *destination); \
    float nvq_loss(const float *vector, \
                   size_t length, \
                   float alpha, float x0, \
                   float minValue, float maxValue, \
                   int nBits); \
    float nvq_uniform_loss(const float *vector, \
                           size_t length, \
                           float minValue, float maxValue, \
                           int nBits); \
    float nvq_square_l2_distance_8bit(const float *vector, \
                                      const unsigned char *quantized, \
                                      size_t length, \
                                      float alpha, float x0, \
                                      float minValue, float maxValue); \
    float nvq_dot_product_8bit(const float *vector, \
                               const unsigned char *quantized, \
                               size_t length, \
                               float alpha, float x0, \
                               float minValue, float maxValue); \
    int64_t nvq_cosine_8bit_packed(const float *vector, \
                                   const unsigned char *quantized, \
                                   size_t length, \
                                   float alpha, float x0, \
                                   float minValue, float maxValue, \
                                   const float *centroid); \
    void nvq_shuffle_query_in_place_8bit(float *vector, \
                                         size_t length); \
    }

DECLARE_SIMD_KERNELS(AVX3)
DECLARE_SIMD_KERNELS(AVX2)
DECLARE_SIMD_KERNELS(SSE42)

#endif // SIMD_KERNELS_H
