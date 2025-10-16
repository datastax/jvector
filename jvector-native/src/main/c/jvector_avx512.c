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
#include "jvector_common.h"

#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VL__)

__m512i maskSeventhBit;
__m512i maskEighthBit;

__m512i initialIndexRegister;
__m512i indexIncrement;

__attribute__((constructor))
void initialize_constants() {
    if (check_compatibility()) {
        maskSeventhBit = _mm512_set1_epi16(0x0040);
        maskEighthBit = _mm512_set1_epi16(0x0080);

        initialIndexRegister = _mm512_setr_epi32(-16, -15, -14, -13, -12, -11, -10, -9,
                                                     -8, -7, -6, -5, -4, -3, -2, -1);
        indexIncrement = _mm512_set1_epi32(16);
    }
}

int simd_version(void) {
    return 2;
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

/* Layout for 512-wide SIMD:
 * The i-th position of quantizedPartials stores 256 quantized partial distances as 16-bit integers, totalling 512 bytes.
 * The 512 bytes of quantized partials are organized as 8 "segments", each one containing 32 16-bit unsigned integers.
 * The function apply_pairwise_shuffle512 permutes and blends consecutive segment pairs. These resulting blends get merged hierarchically.
 */

__attribute__((always_inline)) inline __m512i apply_pairwise_shuffle(__m512i shuffle, const char* quantizedPartials, int offset) {
    __m512i partialsVecA = _mm512_loadu_epi16(quantizedPartials + offset);
    __m512i partialsVecB = _mm512_loadu_epi16(quantizedPartials + offset + 64);
    __m512i partialsVecAB = _mm512_permutex2var_epi16(partialsVecA, shuffle, partialsVecB);
    return partialsVecAB;
}

__attribute__((always_inline)) inline __m512i lookup_partial_sums(__m512i shuffle, const char* quantizedPartials, int i) {
    __m512i partialsVecAB = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512);
    __m512i partialsVecCD = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 128);
    __m512i partialsVecEF = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 256);
    __m512i partialsVecGH = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 384);

    __mmask32 maskSeven = _mm512_test_epi16_mask(shuffle, maskSeventhBit);
    __m512i partialsVecABCD = _mm512_mask_blend_epi16(maskSeven, partialsVecAB, partialsVecCD);
    __m512i partialsVecEFGH = _mm512_mask_blend_epi16(maskSeven, partialsVecEF, partialsVecGH);

    __mmask32 maskEight = _mm512_test_epi16_mask(shuffle, maskEighthBit);
    __m512i partialSumsVec = _mm512_mask_blend_epi16(maskEight, partialsVecABCD, partialsVecEFGH);

    return partialSumsVec;
}

// Dequantize a 256-bit vector containing 16 unsigned 16-bit integers into a 512-bit vector containing 16 32-bit floats
__attribute__((always_inline)) inline __m512 dequantize(__m256i quantizedVec, float delta, float base) {
    __m512i quantizedVecWidened = _mm512_cvtepu16_epi32(quantizedVec);
    __m512 floatVec = _mm512_cvtepi32_ps(quantizedVecWidened);
    __m512 deltaVec = _mm512_set1_ps(delta);
    __m512 baseVec = _mm512_set1_ps(base);
    __m512 dequantizedVec = _mm512_fmadd_ps(floatVec, deltaVec, baseVec);
    return dequantizedVec;
}

void bulk_quantized_shuffle_euclidean(const unsigned char* shuffles, int codebookCount, int codesCount, const char* quantizedPartials, float delta, float minDistance, float* results) {
    int length = codebookCount * codesCount;
    for (int j = 0; j + 32 <= codesCount; j += 32) {
        __m512i sum = _mm512_setzero_epi32();

        for (int i = 0; i < codebookCount; i++) {
            int byte = j * codebookCount + i * 32;
            __m256i smallShuffle;
            if (j + 32 <= codesCount) {
                smallShuffle = _mm256_loadu_epi8(shuffles + byte);
            } else {
                smallShuffle = _mm256_maskz_loadu_epi8((1 << (codesCount - j)) - 1, shuffles + byte);
            }
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

        if (j + 32 <= codesCount) {
            _mm512_storeu_ps(results + j, resultsLeft);
            _mm512_storeu_ps(results + j + 16, resultsRight);
        } else {
            // The mask saves the first (codesCount - j) * 16 floats, masked if appropriate
            _mm512_mask_store_ps(results + j, ((1 << (codesCount - j)) - 1) && ((1 << 16) - 1), resultsLeft);
            // The mask saves the last (codesCount - j) * 16 floats, masked appropriately because we should never have 16 floats exactly in the else branch
            _mm512_mask_store_ps(results + j + 16, ((1 << (codesCount - j  - 16)) - 1) && (((1 << 16) - 1) << 8), resultsRight);
        }
    }
}

void bulk_quantized_shuffle_dot(const unsigned char* shuffles, int codebookCount, int codesCount, const char* quantizedPartials, float delta, float best, float* results) {
    int length = codebookCount * codesCount;
    for (int j = 0; j + 32 <= codesCount; j += 32) {
        __m512i sum = _mm512_setzero_epi32();

        for (int i = 0; i < codebookCount; i++) {
            int byte = j * codebookCount + i * 32;
            __m256i smallShuffle;
            if (j + 32 <= codesCount) {
                smallShuffle = _mm256_loadu_epi8(shuffles + byte);
            } else {
                smallShuffle = _mm256_maskz_loadu_epi8((1 << (codesCount - j)) - 1, shuffles + byte);
            }
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

        if (j + 32 <= codesCount) {
            _mm512_storeu_ps(results + j, resultsLeft);
            _mm512_storeu_ps(results + j + 16, resultsRight);
        } else {
            // The mask saves the first (codesCount - j) * 16 floats, masked if appropriate
            _mm512_mask_store_ps(results + j, ((1 << (codesCount - j)) - 1) && ((1 << 16) - 1), resultsLeft);
            // The mask saves the last (codesCount - j) * 16 floats, masked appropriately because we should never have 16 floats exactly in the else branch
            _mm512_mask_store_ps(results + j + 16, ((1 << (codesCount - j  - 16)) - 1) && (((1 << 16) - 1) << 8), resultsRight);
        }
    }
}

void bulk_quantized_shuffle_cosine(const unsigned char* shuffles, int codebookCount, int codesCount, const char* quantizedPartialSums, float sumDelta, float minDistance, const char* quantizedPartialMagnitudes, float magnitudeDelta, float minMagnitude, float queryMagnitudeSquared, float* results) {
    int length = codebookCount * codesCount;
    for (int j = 0; j + 32 <= codesCount; j += 32) {
        __m512i sum = _mm512_setzero_epi32();
        __m512i magnitude = _mm512_setzero_epi32();

        for (int i = 0; i < codebookCount; i++) {
            int byte = j * codebookCount + i * 32;
            __m256i smallShuffle;
            if (j + 32 <= codesCount) {
                smallShuffle = _mm256_loadu_epi8(shuffles + byte);
            } else {
                smallShuffle = _mm256_maskz_loadu_epi8((1 << (codesCount - j)) - 1, shuffles + byte);
            }
            __m512i shuffle = _mm512_cvtepu8_epi16(smallShuffle);
            __m512i partialsVec = lookup_partial_sums(shuffle, quantizedPartialSums, i);
            sum = _mm512_adds_epu16(sum, partialsVec);

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

        if (j + 32 <= codesCount) {
            _mm512_storeu_ps(results + j, resultsLeft);
            _mm512_storeu_ps(results + j + 16, resultsRight);
        } else {
            // The mask saves the first (codesCount - j) * 16 floats, masked if appropriate
            _mm512_mask_store_ps(results + j, ((1 << (codesCount - j)) - 1) && ((1 << 16) - 1), resultsLeft);
            // The mask saves the last (codesCount - j) * 16 floats, masked appropriately because we should never have 16 floats exactly in the else branch
            _mm512_mask_store_ps(results + j + 16, ((1 << (codesCount - j  - 16)) - 1) && (((1 << 16) - 1) << 8), resultsRight);
        }
    }
}

float assemble_and_sum(const float* data, int dataBase, const unsigned char* baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
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

#endif
