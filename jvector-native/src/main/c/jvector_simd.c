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

#include <immintrin.h>
#include <inttypes.h>
#include <math.h>
#include "jvector_simd.h"

__m512i initialIndexRegister;
__m512i indexIncrement;
__m512i maskSeventhBit;
__m512i maskEighthBit;

__attribute__((constructor))
void initialize_constants() {
    if (check_compatibility()) {
        initialIndexRegister = _mm512_setr_epi32(-16, -15, -14, -13, -12, -11, -10, -9,
                                             -8, -7, -6, -5, -4, -3, -2, -1);
        indexIncrement = _mm512_set1_epi32(16);
        maskSeventhBit = _mm512_set1_epi16(0x0040);
        maskEighthBit = _mm512_set1_epi16(0x0080);
    }
}

float dot_product_f32_64(const float* a, int aoffset, const float* b, int boffset) {

     __m128 va = _mm_castsi128_ps(_mm_loadl_epi64((__m128i *)(a + aoffset)));
     __m128 vb = _mm_castsi128_ps(_mm_loadl_epi64((__m128i *)(b + boffset)));
     __m128 r  = _mm_mul_ps(va, vb); // Perform element-wise multiplication

    // Horizontal sum of the vector to get dot product
    __attribute__((aligned(16))) float result[4];
    _mm_store_ps(result, r);
    return result[0] + result[1];
}

float dot_product_f32_128(const float* a, int aoffset, const float* b, int boffset, int length) {
    float dot = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    int simd_length = length - (length % 4);

    if (length >= 4) {
        __m128 sum = _mm_setzero_ps();

        for(; ao < aoffset + simd_length; ao += 4, bo += 4) {
            // Load float32
            __m128 va = _mm_loadu_ps(a + ao);
            __m128 vb = _mm_loadu_ps(b + bo);

            // Multiply and accumulate
            sum = _mm_fmadd_ps(va, vb, sum);
        }

        // Horizontal sum of the vector to get dot product
        __attribute__((aligned(16))) float result[4];
        _mm_store_ps(result, sum);

        for(int i = 0; i < 4; ++i) {
            dot += result[i];
        }
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        dot += a[ao] * b[bo];
    }

    return dot;
}

float dot_product_f32_256(const float* a, int aoffset, const float* b, int boffset, int length) {
    float dot = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    int simd_length = length - (length % 8);

    if (length >= 8) {
        __m256 sum = _mm256_setzero_ps();

        for(; ao < aoffset + simd_length; ao += 8, bo += 8) {
            // Load float32
            __m256 va = _mm256_loadu_ps(a + ao);
            __m256 vb = _mm256_loadu_ps(b + bo);

            // Multiply and accumulate
            sum = _mm256_fmadd_ps(va, vb, sum);
        }

        // Horizontal sum of the vector to get dot product
        __attribute__((aligned(32))) float result[8];
        _mm256_store_ps(result, sum);

        for(int i = 0; i < 8; ++i) {
            dot += result[i];
        }
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        dot += a[ao] * b[bo];
    }

    return dot;
}

float dot_product_f32_512(const float* a, int aoffset, const float* b, int boffset, int length) {
    float dot = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    int simd_length = length - (length % 16);

    if (length >= 16) {
        __m512 sum = _mm512_setzero_ps();
        for(; ao < aoffset + simd_length; ao += 16, bo += 16) {
            // Load float32
            __m512 va = _mm512_loadu_ps(a + ao);
            __m512 vb = _mm512_loadu_ps(b + bo);

            // Multiply and accumulate
            sum = _mm512_fmadd_ps(va, vb, sum);
        }

        // Horizontal sum of the vector to get dot product
        dot = _mm512_reduce_add_ps(sum);
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        dot += a[ao] * b[bo];
    }

    return dot;
}

float dot_product_f32(int preferred_size, const float* a, int aoffset, const float* b, int boffset, int length) {
    if (length == 2)
        return dot_product_f32_64(a, aoffset, b, boffset);
    if (length <= 7)
        return dot_product_f32_128(a, aoffset, b, boffset, length);

    return (preferred_size == 512 && length >= 16)
           ? dot_product_f32_512(a, aoffset, b, boffset, length)
           : dot_product_f32_256(a, aoffset, b, boffset, length);
}

float euclidean_f32_64(const float* a, int aoffset, const float* b, int boffset) {
     __m128 va = _mm_castsi128_ps(_mm_loadl_epi64((__m128i *)(a + aoffset)));
     __m128 vb = _mm_castsi128_ps(_mm_loadl_epi64((__m128i *)(b + boffset)));
     __m128 r  = _mm_sub_ps(va, vb);
     r = _mm_mul_ps(r, r);

    // Horizontal sum of the vector to get square distance
    __attribute__((aligned(8))) float result[2];
    _mm_store_ps(result, r);
    return result[0] + result[1];
}

float euclidean_f32_128(const float* a, int aoffset, const float* b, int boffset, int length) {
    float squareDistance = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    int simd_length = length - (length % 4);

    if (length >= 4) {
        __m128 sum = _mm_setzero_ps();

        for(; ao < aoffset + simd_length; ao += 4, bo += 4) {
            // Load float32
            __m128 va = _mm_loadu_ps(a + ao);
            __m128 vb = _mm_loadu_ps(b + bo);
            __m128 diff = _mm_sub_ps(va, vb);
            // Multiply and accumulate
            sum = _mm_fmadd_ps(diff, diff, sum);
        }

        // Horizontal sum of the vector to get dot product
        __attribute__((aligned(16))) float result[4];
        _mm_store_ps(result, sum);

        for(int i = 0; i < 4; ++i) {
            squareDistance += result[i];
        }
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        float diff = a[ao] - b[bo];
        squareDistance += diff * diff;
    }

    return squareDistance;
}

float euclidean_f32_256(const float* a, int aoffset, const float* b, int boffset, int length) {
    float squareDistance = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    int simd_length = length - (length % 8);

    if (length >= 8) {
        __m256 sum = _mm256_setzero_ps();

        for(; ao < aoffset + simd_length; ao += 8, bo += 8) {
            // Load float32
            __m256 va = _mm256_loadu_ps(a + ao);
            __m256 vb = _mm256_loadu_ps(b + bo);
            __m256 diff = _mm256_sub_ps(va, vb);

            // Multiply and accumulate
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        __attribute__((aligned(32))) float result[8];
        _mm256_store_ps(result, sum);

        for(int i = 0; i < 8; ++i) {
            squareDistance += result[i];
        }
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        float diff = a[ao] - b[bo];
        squareDistance += diff * diff;
    }

    return squareDistance;
}

float euclidean_f32_512(const float* a, int aoffset, const float* b, int boffset, int length) {
    float squareDistance = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    int simd_length = length - (length % 16);

    if (length >= 16) {
        __m512 sum = _mm512_setzero_ps();
        for(; ao < aoffset + simd_length; ao += 16, bo += 16) {
            // Load float32
            __m512 va = _mm512_loadu_ps(a + ao);
            __m512 vb = _mm512_loadu_ps(b + bo);
            __m512 diff = _mm512_sub_ps(va, vb);

            // Multiply and accumulate
            sum = _mm512_fmadd_ps(diff, diff, sum);
        }

        // Horizontal sum of the vector to get dot product
        squareDistance = _mm512_reduce_add_ps(sum);
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        float diff = a[ao] - b[bo];
        squareDistance += diff * diff;
    }

    return squareDistance;
}

float euclidean_f32(int preferred_size, const float* a, int aoffset, const float* b, int boffset, int length) {
    if (length == 2)
        return euclidean_f32_64(a, aoffset, b, boffset);
    if (length <= 7)
        return euclidean_f32_128(a, aoffset, b, boffset, length);

    return (preferred_size == 512 && length >= 16)
           ? euclidean_f32_512(a, aoffset, b, boffset, length)
           : euclidean_f32_256(a, aoffset, b, boffset, length);
}

float assemble_and_sum_f32_512(const float* data, int dataBase, const unsigned char* baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
    __m512 sum = _mm512_setzero_ps();
    int i = 0;
    int limit = baseOffsetsLength - (baseOffsetsLength % 16);
    __m512i indexRegister = initialIndexRegister;
    __m512i dataBaseVec = _mm512_set1_epi32(dataBase);
    baseOffsets = baseOffsets + baseOffsetsOffset;

    for (; i < limit; i += 16) {
        __m128i baseOffsetsRaw = _mm_loadu_si128((__m128i *)(baseOffsets + i));
        __m512i baseOffsetsInt = _mm512_cvtepu8_epi32(baseOffsetsRaw);
        // we have base offsets int, which we need to scale to index into data.
        // first, we want to initialize a vector with the lane number added as an index
        indexRegister = _mm512_add_epi32(indexRegister, indexIncrement);
        // then we want to multiply by dataBase
        __m512i scale = _mm512_mullo_epi32(indexRegister, dataBaseVec);
        // then we want to add the base offsets
        __m512i convOffsets = _mm512_add_epi32(scale, baseOffsetsInt);

        __m512 partials = _mm512_i32gather_ps(convOffsets, data, 4);
        sum = _mm512_add_ps(sum, partials);
    }

    float res = _mm512_reduce_add_ps(sum);
    for (; i < baseOffsetsLength; i++) {
        res += data[dataBase * i + baseOffsets[i]];
    }

    return res;
}

float pq_decoded_cosine_similarity_f32_512(const unsigned char* baseOffsets, int baseOffsetsOffset, int baseOffsetsLength, int clusterCount, const float* partialSums, const float* aMagnitude, float bMagnitude) {
    __m512 sum = _mm512_setzero_ps();
    __m512 vaMagnitude = _mm512_setzero_ps();
    int i = 0;
    int limit = baseOffsetsLength - (baseOffsetsLength % 16);
    __m512i indexRegister = initialIndexRegister;
    __m512i scale = _mm512_set1_epi32(clusterCount);
    baseOffsets = baseOffsets + baseOffsetsOffset;


    for (; i < limit; i += 16) {
        // Load and convert baseOffsets to integers
        __m128i baseOffsetsRaw = _mm_loadu_si128((__m128i *)(baseOffsets + i));
        __m512i baseOffsetsInt = _mm512_cvtepu8_epi32(baseOffsetsRaw);

        indexRegister = _mm512_add_epi32(indexRegister, indexIncrement);
        // Scale the baseOffsets by the cluster count
        __m512i scaledOffsets = _mm512_mullo_epi32(indexRegister, scale);

        // Calculate the final convOffsets by adding the scaled indexes and the base offsets
        __m512i convOffsets = _mm512_add_epi32(scaledOffsets, baseOffsetsInt);

        // Gather and sum values for partial sums and a magnitude
        __m512 partialSumVals = _mm512_i32gather_ps(convOffsets, partialSums, 4);
        sum = _mm512_add_ps(sum, partialSumVals);

        __m512 aMagnitudeVals = _mm512_i32gather_ps(convOffsets, aMagnitude, 4);
        vaMagnitude = _mm512_add_ps(vaMagnitude, aMagnitudeVals);
    }

    // Reduce sums
    float sumResult = _mm512_reduce_add_ps(sum);
    float aMagnitudeResult = _mm512_reduce_add_ps(vaMagnitude);

    // Handle the remaining elements
    for (; i < baseOffsetsLength; i++) {
        int offset = clusterCount * i + baseOffsets[i];
        sumResult += partialSums[offset];
        aMagnitudeResult += aMagnitude[offset];
    }

    return sumResult / sqrtf(aMagnitudeResult * bMagnitude);
}

void calculate_partial_sums_dot_f32_512(const float* codebook, int codebookIndex, int size, int clusterCount, const float* query, int queryOffset, float* partialSums) {
    int codebookBase = codebookIndex * clusterCount;
    for (int i = 0; i < clusterCount; i++) {
      partialSums[codebookBase + i] = dot_product_f32(512, codebook, i * size, query, queryOffset, size);
    }
}

void calculate_partial_sums_euclidean_f32_512(const float* codebook, int codebookIndex, int size, int clusterCount, const float* query, int queryOffset, float* partialSums) {
    int codebookBase = codebookIndex * clusterCount;
    for (int i = 0; i < clusterCount; i++) {
      partialSums[codebookBase + i] = euclidean_f32(512, codebook, i * size, query, queryOffset, size);
    }
}

/* Bulk shuffles for Fused ADC
 * These shuffles take an array of transposed PQ neighbors (in shuffles) and an of quantized partial distances to shuffle.
 * Partial distance quantization depends on the best distance and delta used to quantize.
 * The shuffles for each codebook will be loaded as bytes (supporting up to 256 cluster PQ) and zero-padded to align
 * with 16-bit quantized partial distances. These partial distances will be loaded into SIMD registers, supporting 32 partials
 * per register. Each permutation will take 2 registers, so we need four total permutations to look up against all
 * 256 partial distances. These four permutations will be blended based on the top two bits of each shuffle, allowing 256
 * entry codebook lookup. Quantized partials are quantized based on bounds provided during the search that suggest total
 * distances above the maximum value of an unsigned 16-bit integer will be irrelevant. This allows us to use saturating
 * arithmetic, eliminating the need to widen lanes during accumulation. The total quantized distance is then de-quantized
 * and transformed into the appropriate similarity score.
 *
 * In the case of cosine, we have an additional set of partials used for partial squared magnitudes. These are quantized \
 * with a different pair of delta/base, so they will be aggregated and dequantized separately.
 */


__attribute__((always_inline)) inline __m512i lookup_partial_sums(__m512i shuffle, const char* quantizedPartials, int i) {
    __m512i partialsVecA = _mm512_loadu_epi16(quantizedPartials + i * 512);
    __m512i partialsVecB = _mm512_loadu_epi16(quantizedPartials + i * 512 + 64);
    __m512i partialsVecC = _mm512_loadu_epi16(quantizedPartials + i * 512 + 128);
    __m512i partialsVecD = _mm512_loadu_epi16(quantizedPartials + i * 512 + 192);
    __m512i partialsVecE = _mm512_loadu_epi16(quantizedPartials + i * 512 + 256);
    __m512i partialsVecF = _mm512_loadu_epi16(quantizedPartials + i * 512 + 320);
    __m512i partialsVecG = _mm512_loadu_epi16(quantizedPartials + i * 512 + 384);
    __m512i partialsVecH = _mm512_loadu_epi16(quantizedPartials + i * 512 + 448);

    __m512i partialsVecAB = _mm512_permutex2var_epi16(partialsVecA, shuffle, partialsVecB);
    __m512i partialsVecCD = _mm512_permutex2var_epi16(partialsVecC, shuffle, partialsVecD);
    __m512i partialsVecEF = _mm512_permutex2var_epi16(partialsVecE, shuffle, partialsVecF);
    __m512i partialsVecGH = _mm512_permutex2var_epi16(partialsVecG, shuffle, partialsVecH);

    __mmask32 maskSeven = _mm512_test_epi16_mask(shuffle, maskSeventhBit);
    __mmask32 maskEight = _mm512_test_epi16_mask(shuffle, maskEighthBit);
    __m512i partialsVecABCD = _mm512_mask_blend_epi16(maskSeven, partialsVecAB, partialsVecCD);
    __m512i partialsVecEFGH = _mm512_mask_blend_epi16(maskSeven, partialsVecEF, partialsVecGH);
    __m512i partialSumsVec = _mm512_mask_blend_epi16(maskEight, partialsVecABCD, partialsVecEFGH);

    return partialSumsVec;
}

// dequantize a 256-bit vector containing 16 unsigned 16-bit integers into a 512-bit vector containing 16 32-bit floats
__attribute__((always_inline)) inline __m512 dequantize(__m256i quantizedVec, float delta, float base) {
    __m512i quantizedVecWidened = _mm512_cvtepu16_epi32(quantizedVec);
    __m512 floatVec = _mm512_cvtepi32_ps(quantizedVecWidened);
    __m512 deltaVec = _mm512_set1_ps(delta);
    __m512 baseVec = _mm512_set1_ps(base);
    __m512 dequantizedVec = _mm512_fmadd_ps(floatVec, deltaVec, baseVec);
    return dequantizedVec;
}

void bulk_quantized_shuffle_euclidean_f32_512(const unsigned char* shuffles, int codebookCount, const char* quantizedPartials, float delta, float minDistance, float* results) {
    __m512i sum = _mm512_setzero_epi32();

    for (int i = 0; i < codebookCount; i++) {
         __m256i smallShuffle = _mm256_loadu_epi8(shuffles + i * 32);
         __m512i shuffle = _mm512_cvtepu8_epi16(smallShuffle);
        __m512i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);

        sum = _mm512_adds_epu16(sum, partialsVec);
    }

    __m256i quantizedResultsLeftRaw = _mm512_extracti32x8_epi32(sum, 0);
    __m256i quantizedResultsRightRaw = _mm512_extracti32x8_epi32(sum, 1);
    __m512 resultsLeft = dequantize(quantizedResultsLeftRaw, delta, minDistance);
    __m512 resultsRight = dequantize(quantizedResultsRightRaw, delta, minDistance);

    __m512 ones = _mm512_set1_ps(1.0);
    resultsLeft = _mm512_add_ps(resultsLeft, ones);
    resultsRight = _mm512_add_ps(resultsRight, ones);
    resultsLeft = _mm512_rcp14_ps(resultsLeft);
    resultsRight = _mm512_rcp14_ps(resultsRight);
    _mm512_storeu_ps(results, resultsLeft);
    _mm512_storeu_ps(results + 16, resultsRight);
}

void bulk_quantized_shuffle_dot_f32_512(const unsigned char* shuffles, int codebookCount, const char* quantizedPartials, float delta, float best, float* results) {
    __m512i sum = _mm512_setzero_epi32();

    for (int i = 0; i < codebookCount; i++) {
         __m256i smallShuffle = _mm256_loadu_epi8(shuffles + i * 32);
         __m512i shuffle = _mm512_cvtepu8_epi16(smallShuffle);
        __m512i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
        sum = _mm512_adds_epu16(sum, partialsVec);
    }

    __m256i quantizedResultsLeftRaw = _mm512_extracti32x8_epi32(sum, 0);
    __m256i quantizedResultsRightRaw = _mm512_extracti32x8_epi32(sum, 1);
    __m512 resultsLeft = dequantize(quantizedResultsLeftRaw, delta, best);
    __m512 resultsRight = dequantize(quantizedResultsRightRaw, delta, best);

    __m512 ones = _mm512_set1_ps(1.0);
    resultsLeft = _mm512_add_ps(resultsLeft, ones);
    resultsRight = _mm512_add_ps(resultsRight, ones);
    resultsLeft = _mm512_div_ps(resultsLeft, _mm512_set1_ps(2.0));
    resultsRight = _mm512_div_ps(resultsRight, _mm512_set1_ps(2.0));
    _mm512_storeu_ps(results, resultsLeft);
    _mm512_storeu_ps(results + 16, resultsRight);
}

void bulk_quantized_shuffle_cosine_f32_512(const unsigned char* shuffles, int codebookCount, const char* quantizedPartialSums, float sumDelta, float minDistance, const char* quantizedPartialMagnitudes, float magnitudeDelta, float minMagnitude, float queryMagnitudeSquared, float* results) {
    __m512i sum = _mm512_setzero_epi32();
    __m512i magnitude = _mm512_setzero_epi32();

    for (int i = 0; i < codebookCount; i++) {
        __m256i smallShuffle = _mm256_loadu_epi8((shuffles + i * 32));
        __m512i shuffle = _mm512_cvtepu8_epi16(smallShuffle);
        __m512i partialSumsVec = lookup_partial_sums(shuffle, quantizedPartialSums, i);
        sum = _mm512_adds_epu16(sum, partialSumsVec);

        __m512i partialMagnitudesVec = lookup_partial_sums(shuffle, quantizedPartialMagnitudes, i);
        magnitude = _mm512_adds_epu16(magnitude, partialMagnitudesVec);
    }

    __m256i quantizedSumsLeftRaw = _mm512_extracti32x8_epi32(sum, 0);
    __m256i quantizedSumsRightRaw = _mm512_extracti32x8_epi32(sum, 1);
    __m512 sumsLeft = dequantize(quantizedSumsLeftRaw, sumDelta, minDistance);
    __m512 sumsRight = dequantize(quantizedSumsRightRaw, sumDelta, minDistance);

    __m256i quantizedMagnitudesLeftRaw = _mm512_extracti32x8_epi32(magnitude, 0);
    __m256i quantizedMagnitudesRightRaw = _mm512_extracti32x8_epi32(magnitude, 1);
    __m512 magnitudesLeft = dequantize(quantizedMagnitudesLeftRaw, magnitudeDelta, minMagnitude);
    __m512 magnitudesRight = dequantize(quantizedMagnitudesRightRaw, magnitudeDelta, minMagnitude);

    __m512 queryMagnitudeSquaredVec = _mm512_set1_ps(queryMagnitudeSquared);
    magnitudesLeft = _mm512_mul_ps(magnitudesLeft, queryMagnitudeSquaredVec);
    magnitudesRight = _mm512_mul_ps(magnitudesRight, queryMagnitudeSquaredVec);
    magnitudesLeft = _mm512_sqrt_ps(magnitudesLeft);
    magnitudesRight = _mm512_sqrt_ps(magnitudesRight);
    __m512 resultsLeft = _mm512_div_ps(sumsLeft, magnitudesLeft);
    __m512 resultsRight = _mm512_div_ps(sumsRight, magnitudesRight);

    __m512 ones = _mm512_set1_ps(1.0);
    resultsLeft = _mm512_add_ps(resultsLeft, ones);
    resultsRight = _mm512_add_ps(resultsRight, ones);
    resultsLeft = _mm512_div_ps(resultsLeft, _mm512_set1_ps(2.0));
    resultsRight = _mm512_div_ps(resultsRight, _mm512_set1_ps(2.0));
    _mm512_storeu_ps(results, resultsLeft);
    _mm512_storeu_ps(results + 16, resultsRight);
}

// Partial sum calculations that also record best distances, as this is necessary for Fused ADC quantization
void calculate_partial_sums_best_dot_f32_512(const float* codebook, int codebookIndex, int size, int clusterCount, const float* query, int queryOffset, float* partialSums, float* partialBestDistances) {
    float best = -INFINITY;
    int codebookBase = codebookIndex * clusterCount;
    for (int i = 0; i < clusterCount; i++) {
      float val = dot_product_f32(512, codebook, i * size, query, queryOffset, size);
      partialSums[codebookBase + i] = val;
      if (val > best) {
        best = val;
      }
    }
    partialBestDistances[codebookIndex] = best;
}

void calculate_partial_sums_best_euclidean_f32_512(const float* codebook, int codebookIndex, int size, int clusterCount, const float* query, int queryOffset, float* partialSums, float* partialBestDistances) {
    float best = INFINITY;
    int codebookBase = codebookIndex * clusterCount;
    for (int i = 0; i < clusterCount; i++) {
      float val = euclidean_f32(512, codebook, i * size, query, queryOffset, size);
      partialSums[codebookBase + i] = val;
      if (val < best) {
        best = val;
      }
    }
    partialBestDistances[codebookIndex] = best;
}
