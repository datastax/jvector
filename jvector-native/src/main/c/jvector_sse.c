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

#elif defined(__SSE4_1__)

__m128i maskFifthBit;
__m128i maskSixthBit;
__m128i maskSeventhBit;
__m128i maskEighthBit;


__attribute__((constructor))
void initialize_constants() {
    if (check_compatibility()) {
        maskFifthBit = _mm_set1_epi16(0x0010);
        maskSixthBit = _mm_set1_epi16(0x0020);
        maskSeventhBit = _mm_set1_epi16(0x0040);
        maskEighthBit = _mm_set1_epi16(0x0080);
    }
}

int simd_version(void) {
    return 0;
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

/* Layout for 128-wide SIMD:
 * The i-th position of quantizedPartials stores 256 quantized partial distances as 16-bit integers, totalling 512 bytes.
 * The 512 bytes of quantized partials are organized as 32 "segments", each one containing 8 16-bit unsigned integers.
 * The function apply_pairwise_shuffle permutes and blends consecutive segment pairs. These resulting blends get merged hierarchically.
 */

__attribute__((always_inline)) inline __m128i apply_pairwise_shuffle(__m128i shuffle, const char* quantizedPartials, int offset) {
    __m128i partialsVecA = _mm_loadu_epi16(quantizedPartials + offset);
    __m128i partialsVecB = _mm_loadu_epi16(quantizedPartials + offset + 16);
    __m128i partialsVecAB = _mm_permutex2var_epi16(partialsVecA, shuffle, partialsVecB);
    return partialsVecAB;
}

__attribute__((always_inline)) inline __m128i lookup_partial_sums(__m128i shuffle, const char* quantizedPartials, int i) {
    __m128i partialsVecA = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512);
    __m128i partialsVecB = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 32);
    __m128i partialsVecC = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 64);
    __m128i partialsVecD = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 96);
    __m128i partialsVecE = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 128);
    __m128i partialsVecF = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 160);
    __m128i partialsVecG = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 192);
    __m128i partialsVecH = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 224);
    __m128i partialsVecI = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 256);
    __m128i partialsVecJ = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 288);
    __m128i partialsVecK = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 320);
    __m128i partialsVecL = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 352);
    __m128i partialsVecM = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 384);
    __m128i partialsVecN = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 416);
    __m128i partialsVecO = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 448);
    __m128i partialsVecP = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 480);

    __mmask8 maskFifth = _mm_test_epi16_mask(shuffle, maskFifthBit);
    __m128i partialsVecAB = _mm_mask_blend_epi16(maskFifth, partialsVecA, partialsVecB);
    __m128i partialsVecCD = _mm_mask_blend_epi16(maskFifth, partialsVecC, partialsVecD);
    __m128i partialsVecEF = _mm_mask_blend_epi16(maskFifth, partialsVecE, partialsVecF);
    __m128i partialsVecGH = _mm_mask_blend_epi16(maskFifth, partialsVecG, partialsVecH);
    __m128i partialsVecIJ = _mm_mask_blend_epi16(maskFifth, partialsVecI, partialsVecJ);
    __m128i partialsVecKL = _mm_mask_blend_epi16(maskFifth, partialsVecK, partialsVecL);
    __m128i partialsVecMN = _mm_mask_blend_epi16(maskFifth, partialsVecM, partialsVecN);
    __m128i partialsVecOP = _mm_mask_blend_epi16(maskFifth, partialsVecO, partialsVecP);

    __mmask8 maskSixth = _mm_test_epi16_mask(shuffle, maskSixthBit);
    __m128i partialsVecABCD = _mm_mask_blend_epi16(maskSixth, partialsVecAB, partialsVecCD);
    __m128i partialsVecEFGH = _mm_mask_blend_epi16(maskSixth, partialsVecEF, partialsVecGH);
    __m128i partialsVecIJKL = _mm_mask_blend_epi16(maskSixth, partialsVecIJ, partialsVecKL);
    __m128i partialsVecMNOP = _mm_mask_blend_epi16(maskSixth, partialsVecMN, partialsVecOP);

    __mmask16 maskSeven = _mm_test_epi16_mask(shuffle, maskSeventhBit);
    __m128i partialsVecABCDEFGH = _mm_mask_blend_epi16(maskSeven, partialsVecABCD, partialsVecEFGH);
    __m128i partialsVecIJKLMNOP = _mm_mask_blend_epi16(maskSeven, partialsVecIJKL, partialsVecMNOP);

    __mmask16 maskEight = _mm_test_epi16_mask(shuffle, maskEighthBit);
    __m128i partialSumsVec = _mm_mask_blend_epi16(maskEight, partialsVecABCDEFGH, partialsVecIJKLMNOP);

    return partialSumsVec;
}

// Dequantize a 128-bit vector containing 4 unsigned 16-bit integers (only the first 4 integers are considered)
// into a 128-bit vector containing 8 32-bit floats
__attribute__((always_inline)) inline __m128 dequantize(__m128i quantizedVec, float delta, float base) {
    __m128i quantizedVecWidened = _mm_cvtepu16_epi32(quantizedVec);
    __m128 floatVec = _mm_cvtepi32_ps(quantizedVecWidened);
    __m128 deltaVec = _mm_set1_ps(delta);
    __m128 baseVec = _mm_set1_ps(base);
    __m128 dequantizedVec = _mm_fmadd_ps(floatVec, deltaVec, baseVec);
    return dequantizedVec;
}

void bulk_quantized_shuffle_euclidean(const unsigned char* shuffles, int codebookCount, int codesCount, const char* quantizedPartials, float delta, float minDistance, float* results) {
    int length = codebookCount * codesCount;
    int j = 0;
    for (; j + 8 <= codesCount; j += 8) {
        __m128i sum = _mm_setzero_si128();

        for (int i = 0; i < codebookCount; i++) {
            int byte = j * codebookCount + i * 8;
            __m128i smallShuffle = _mm_loadl_epi64(shuffles + byte);
            __m128i shuffle = _mm_cvtepu8_epi16(smallShuffle);

            __m128i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
            sum = _mm_adds_epu16(sum, partialsVec);
        }

        __m128i quantizedSumsLeftRaw = _mm_cvtepu16_epi32(_mm_cvtsi64x_si128(_mm_cvtsi128_si64(sum)));
        // Shuffle the upper 64-bit value to the lower position.
        // _MM_SHUFFLE(a, b, c, d) selects 32-bit integers from the source vector.
        // We want elements 3 and 2 to move to positions 1 and 0.
        // The other arguments don't matter because we won't use them.
        sum = _mm_shuffle_epi32(sum, _MM_SHUFFLE(3, 2, 3, 2));
        __m128i quantizedSumsRightRaw = _mm_cvtepu16_epi32(_mm_cvtsi64x_si128(_mm_cvtsi128_si64(sum)));
        __m128 sumsLeft = dequantize(quantizedSumsLeftRaw, sumDelta, minDistance);
        __m128 sumsRight = dequantize(quantizedSumsRightRaw, sumDelta, minDistance);

        __m128 ones = _mm_set1_ps(1.0);
        resultsLeft = _mm_add_ps(sumsLeft, ones);
        resultsRight = _mm_add_ps(sumsRight, ones);
        resultsLeft = _mm_rcp_ps(resultsLeft);
        resultsRight = _mm_rcp_ps(resultsRight);

        _mm_storeu_ps(results + j, resultsLeft);
        _mm_storeu_ps(results + j + 4, resultsRight);
    }
    if (j < codesCount) {
        for (; j < codesCount; j++) {
            float sum = 0;

            for (int i = 0; i < codebookCount; i++) {
                var shuffle = computeSingleShuffle(i, j, shuffles, codesCount);
                var val = combineBytes(i, shuffle, quantizedPartialSums);
                sum += val;
            }

            float unquantizedSum = sumDelta * sum + minDistance;
            results[j] = 1 / (1 + (float) (unquantizedSum));
        }
    }
}

void bulk_quantized_shuffle_dot(const unsigned char* shuffles, int codebookCount, int codesCount, const char* quantizedPartials, float delta, float best, float* results) {
    int length = codebookCount * codesCount;
    int j = 0;
    for (; j + 8 <= codesCount; j += 8) {
        __m128i sum = _mm_setzero_si128();

        for (int i = 0; i < codebookCount; i++) {
            int byte = j * codebookCount + i * 8;
            __m128i smallShuffle = _mm_loadl_epi64(shuffles + byte);
            __m128i shuffle = _mm_cvtepu8_epi16(smallShuffle);

            __m128i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
            sum = _mm_adds_epu16(sum, partialsVec);
        }

        __m128i quantizedSumsLeftRaw = _mm_cvtepu16_epi32(_mm_cvtsi64x_si128(_mm_cvtsi128_si64(sum)));
        // Shuffle the upper 64-bit value to the lower position.
        // _MM_SHUFFLE(a, b, c, d) selects 32-bit integers from the source vector.
        // We want elements 3 and 2 to move to positions 1 and 0.
        // The other arguments don't matter because we won't use them.
        sum = _mm_shuffle_epi32(sum, _MM_SHUFFLE(3, 2, 3, 2));
        __m128i quantizedSumsRightRaw = _mm_cvtepu16_epi32(_mm_cvtsi64x_si128(_mm_cvtsi128_si64(sum)));
        __m128 sumsLeft = dequantize(quantizedSumsLeftRaw, sumDelta, minDistance);
        __m128 sumsRight = dequantize(quantizedSumsRightRaw, sumDelta, minDistance);

        __m128 ones = _mm_set1_ps(1.0);
        resultsLeft = _mm_add_ps(sumsLeft, ones);
        resultsRight = _mm_add_ps(sumsRight, ones);
        resultsLeft = _mm_div_ps(resultsLeft, _mm_set1_ps(2.0));
        resultsRight = _mm_div_ps(resultsRight, _mm_set1_ps(2.0));

        _mm_storeu_ps(results + j, resultsLeft);
        _mm_storeu_ps(results + j + 4, resultsRight);
    }
    if (j < codesCount) {
        for (; j < codesCount; j++) {
            float sum = 0;

            for (int i = 0; i < codebookCount; i++) {
                var shuffle = computeSingleShuffle(i, j, shuffles, codesCount);
                var val = combineBytes(i, shuffle, quantizedPartialSums);
                sum += val;
            }

            float unquantizedSum = sumDelta * sum + minDistance;
            results[j] = (1 + (float) (unquantizedSum)) / 2);
        }
    }
}

void bulk_quantized_shuffle_cosine(const unsigned char* shuffles, int codebookCount, int codesCount, const char* quantizedPartialSums, float sumDelta, float minDistance, const char* quantizedPartialMagnitudes, float magnitudeDelta, float minMagnitude, float queryMagnitudeSquared, float* results) {
    int length = codebookCount * codesCount;
    int j = 0;
    for (; j + 8 <= codesCount; j += 8) {
        __m128i sum = _mm_setzero_si128();
        __m128i magnitude = _mm_setzero_si128();

        for (int i = 0; i < codebookCount; i++) {
            int byte = j * codebookCount + i * 8;
            __m128i smallShuffle = _mm_loadl_epi64(shuffles + byte);
            __m128i shuffle = _mm_cvtepu8_epi16(smallShuffle);

            __m128i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
            sum = _mm_adds_epu16(sum, partialsVec);

            __m128i partialMagnitudesVec = lookup_partial_sums(shuffle, quantizedPartialMagnitudes, i);
            magnitude = _mm_adds_epu16(magnitude, partialMagnitudesVec);
        }

        __m128i quantizedSumsLeftRaw = _mm_cvtepu16_epi32(_mm_cvtsi64x_si128(_mm_cvtsi128_si64(sum)));
        // Shuffle the upper 64-bit value to the lower position.
        // _MM_SHUFFLE(a, b, c, d) selects 32-bit integers from the source vector.
        // We want elements 3 and 2 to move to positions 1 and 0.
        // The other arguments don't matter because we won't use them.
        sum = _mm_shuffle_epi32(sum, _MM_SHUFFLE(3, 2, 3, 2));
        __m128i quantizedSumsRightRaw = _mm_cvtepu16_epi32(_mm_cvtsi64x_si128(_mm_cvtsi128_si64(sum)));
        __m128 sumsLeft = dequantize(quantizedSumsLeftRaw, sumDelta, minDistance);
        __m128 sumsRight = dequantize(quantizedSumsRightRaw, sumDelta, minDistance);

        __m128i quantizedMagnitudesLeftRaw = _mm_cvtepu16_epi32(_mm_cvtsi64x_si128(_mm_cvtsi128_si64(magnitude)));
        // Shuffle the upper 64-bit value to the lower position.
        // _MM_SHUFFLE(a, b, c, d) selects 32-bit integers from the source vector.
        // We want elements 3 and 2 to move to positions 1 and 0.
        // The other arguments don't matter because we won't use them.
        magnitude = _mm_shuffle_epi32(magnitude, _MM_SHUFFLE(3, 2, 3, 2));
        __m128i quantizedMagnitudesRightRaw = _mm_cvtepu16_epi32(_mm_cvtsi64x_si128(_mm_cvtsi128_si64(magnitude)));
        __m128 magnitudesLeft = dequantize(quantizedMagnitudesLeftRaw, magnitudeDelta, minMagnitude);
        __m128 magnitudesRight = dequantize(quantizedMagnitudesRightRaw, magnitudeDelta, minMagnitude);

        __m128 queryMagnitudeSquaredVec = _mm_set1_ps(queryMagnitudeSquared);
        magnitudesLeft = _mm_mul_ps(magnitudesLeft, queryMagnitudeSquaredVec);
        magnitudesRight = _mm_mul_ps(magnitudesRight, queryMagnitudeSquaredVec);
        magnitudesLeft = _mm_sqrt_ps(magnitudesLeft);
        magnitudesRight = _mm_sqrt_ps(magnitudesRight);
        __m128 resultsLeft = _mm_div_ps(sumsLeft, magnitudesLeft);
        __m128 resultsRight = _mm_div_ps(sumsRight, magnitudesRight);

        __m128 ones = _mm_set1_ps(1.0);
        resultsLeft = _mm_add_ps(resultsLeft, ones);
        resultsRight = _mm_add_ps(resultsRight, ones);
        resultsLeft = _mm_div_ps(resultsLeft, _mm_set1_ps(2.0));
        resultsRight = _mm_div_ps(resultsRight, _mm_set1_ps(2.0));

        _mm_storeu_ps(results + j, resultsLeft);
        _mm_storeu_ps(results + j + 4, resultsRight);
    }
    if (j < codesCount) {
        for (; j < codesCount; j++) {
            float sum = 0;
            float magnitude = 0;

            for (int i = 0; i < codebookCount; i++) {
                var shuffle = computeSingleShuffle(i, j, shuffles, codesCount);
                var val = combineBytes(i, shuffle, quantizedPartialSums);
                sum += val;
                val = combineBytes(i, shuffle, quantizedPartialSquaredMagnitudes);
                magnitude += val;
            }

            float unquantizedSum = sumDelta * sum + minDistance;
            float unquantizedMagnitude = magnitudeDelta * magnitude + minMagnitude;
            float divisor = sqrt(unquantizedMagnitude * queryMagnitudeSquared);
            results[j] = (1 + unquantizedSum / divisor) / 2);
        }
    }
}

float assemble_and_sum(const float* data, int dataBase, const unsigned char* baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
    // It seems like this function cannot be implemented in SSE4_1 because of the lack of a gather operation.
    // For reference, see the implementations for AVX512 & AVX2.
    float res = 0;
    for (int i = 0; i < baseOffsetsLength; i++) {
        res += data[dataBase * i + baseOffsets[i]];
    }
    return res;
}

#endif
