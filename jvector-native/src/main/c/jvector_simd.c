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

#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VL__)
__m512i maskSeventhBit;
__m512i maskEighthBit;

__m512i initialIndexRegister;
__m512i indexIncrement;

#elif defined(__AVX2__)
__m256i maskSixthBit;
__m256i maskSeventhBit;
__m256i maskEighthBit;

__m256i initialIndexRegister;
__m256i indexIncrement;

#elif defined(__SSE4_1__)
__m128i maskFifthBit;
__m128i maskSixthBit;
__m128i maskSeventhBit;
__m128i maskEighthBit;
#endif

__attribute__((constructor))
void initialize_constants() {
    if (check_compatibility()) {
        #if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VL__)
        maskSeventhBit = _mm512_set1_epi16(0x0040);
        maskEighthBit = _mm512_set1_epi16(0x0080);

        initialIndexRegister = _mm512_setr_epi32(-16, -15, -14, -13, -12, -11, -10, -9,
                                                     -8, -7, -6, -5, -4, -3, -2, -1);
        indexIncrement = _mm512_set1_epi32(16);

        #elif defined(__AVX2__)
        maskSixthBit = _mm256_set1_epi16(0x0020);
        maskSeventhBit = _mm256_set1_epi16(0x0040);
        maskEighthBit = _mm256_set1_epi16(0x0080);

        initialIndexRegister = _mm256_setr_epi32(-8, -7, -6, -5, -4, -3, -2, -1);
        indexIncrement = _mm256_set1_epi32(8);

        #elif defined(__SSE4_1__)
        maskFifthBit = _mm_set1_epi16(0x0010);
        maskSixthBit = _mm_set1_epi16(0x0020);
        maskSeventhBit = _mm_set1_epi16(0x0040);
        maskEighthBit = _mm_set1_epi16(0x0080);

        #endif
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

// Auxiliary scalar functions common to all methods
unsigned int combineBytes(int i, unsigned int shuffle, const char* quantizedPartials) {
    // This is a 16-bit value stored in two bytes, so we need to move in multiples of two and then combine them.
    unsigned int lowByte = quantizedPartials[i * 512 + shuffle];
    unsigned int highByte = quantizedPartials[i * 512 + shuffle + 1];
    return (highByte << 8) | lowByte;
}

unsigned int computeSingleShuffle(int i, int j, const unsigned char* shuffles, int nNeighbors) {
    // This points to a 16-bit value stored in two bytes, so we need to move in multiples of two.
    unsigned int temp = shuffles[i * nNeighbors + j];
    return temp * 2;
}


/************************************************************************/
/*********************** 512-wide SIMD functions ************************/
/************************************************************************/

#if defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VL__)

/* Layout for 512-wide SIMD:
 * The i-th position of quantizedPartials stores 256 quantized partial distances as 16-bit integers, totalling 512 bytes.
 * The 512 bytes of quantized partials are organized as 8 "segments", each one containing 32 16-bit unsigned integers.
 * The function apply_pairwise_shuffle512 permutes and blends consecutive segment pairs. These resulting blends get merged hierarchically.
 */

int simd_version(void) {
    return 2;
}

// AVX512 implementation
__attribute__((always_inline)) inline __m512i apply_pairwise_shuffle(__m512i shuffle, const char* quantizedPartials, int offset) {
    __m512i partialsVecA = _mm512_loadu_epi16(quantizedPartials + offset);
    __m512i partialsVecB = _mm512_loadu_epi16(quantizedPartials + offset + 64);
    __m512i partialsVecAB = _mm512_permutex2var_epi16(partialsVecA, shuffle, partialsVecB);
    return partialsVecAB;
}

// AVX512 implementation
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

// AVX512 implementation
// Dequantize a 256-bit vector containing 16 unsigned 16-bit integers into a 512-bit vector containing 16 32-bit floats
__attribute__((always_inline)) inline __m512 dequantize(__m256i quantizedVec, float delta, float base) {
    __m512i quantizedVecWidened = _mm512_cvtepu16_epi32(quantizedVec);
    __m512 floatVec = _mm512_cvtepi32_ps(quantizedVecWidened);
    __m512 deltaVec = _mm512_set1_ps(delta);
    __m512 baseVec = _mm512_set1_ps(base);
    __m512 dequantizedVec = _mm512_fmadd_ps(floatVec, deltaVec, baseVec);
    return dequantizedVec;
}

// AVX512 implementation
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

// AVX512 implementation
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

// AVX512 implementation
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

// AVX512 implementation
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

// end of AVX512 section

#elif defined(__AVX2__)

/************************************************************************/
/*********************** 256-wide SIMD functions ************************/
/************************************************************************/

int simd_version(void) {
    return 1;
}


/* Layout for 256-wide SIMD:
 * The i-th position of quantizedPartials stores 256 quantized partial distances as 16-bit integers, totalling 512 bytes.
 * The 512 bytes of quantized partials are organized as 16 "segments", each one containing 16 16-bit unsigned integers.
 * The function apply_pairwise_shuffle permutes and blends consecutive segment pairs. These resulting blends get merged hierarchically.
 */

// AVX2 implementation
inline __m256i apply_pairwise_shuffle(__m256i shuffle, const char* quantizedPartials, int offset) {
    __m256i partialsVecA = _mm256_loadu_epi16(quantizedPartials + offset);
    __m256i partialsVecB = _mm256_loadu_epi16(quantizedPartials + offset + 32);
    __m256i partialsVecAB = _mm256_permutex2var_epi16(partialsVecA, shuffle, partialsVecB);
    return partialsVecAB;
}

// AVX2 implementation
inline __m256i lookup_partial_sums(__m256i shuffle, const char* quantizedPartials, int i) {
    __m256i partialsVecA = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512);
    __m256i partialsVecB = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 64);
    __m256i partialsVecC = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 128);
    __m256i partialsVecD = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 192);
    __m256i partialsVecE = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 256);
    __m256i partialsVecF = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 320);
    __m256i partialsVecG = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 384);
    __m256i partialsVecH = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 448);

    __mmask16 maskSixth = _mm256_test_epi16_mask(shuffle, maskSixthBit);
    __m256i partialsVecAB = _mm256_mask_blend_epi16(maskSixth, partialsVecA, partialsVecB);
    __m256i partialsVecCD = _mm256_mask_blend_epi16(maskSixth, partialsVecC, partialsVecD);
    __m256i partialsVecEF = _mm256_mask_blend_epi16(maskSixth, partialsVecE, partialsVecF);
    __m256i partialsVecGH = _mm256_mask_blend_epi16(maskSixth, partialsVecG, partialsVecH);

    __mmask16 maskSeven = _mm256_test_epi16_mask(shuffle, maskSeventhBit);
    __m256i partialsVecABCD = _mm256_mask_blend_epi16(maskSeven, partialsVecAB, partialsVecCD);
    __m256i partialsVecEFGH = _mm256_mask_blend_epi16(maskSeven, partialsVecEF, partialsVecGH);

    __mmask16 maskEight = _mm256_test_epi16_mask(shuffle, maskEighthBit);
    __m256i partialSumsVec = _mm256_mask_blend_epi16(maskEight, partialsVecABCD, partialsVecEFGH);

    return partialSumsVec;
}

// AVX2 implementation
// Dequantize a 128-bit vector containing 8 unsigned 16-bit integers into a 256-bit vector containing 8 32-bit floats
__attribute__((always_inline)) inline __m256 dequantize(__m128i quantizedVec, float delta, float base) {
    __m256i quantizedVecWidened = _mm256_cvtepu16_epi32(quantizedVec);
    __m256 floatVec = _mm256_cvtepi32_ps(quantizedVecWidened);
    __m256 deltaVec = _mm256_set1_ps(delta);
    __m256 baseVec = _mm256_set1_ps(base);
    __m256 dequantizedVec = _mm256_fmadd_ps(floatVec, deltaVec, baseVec);
    return dequantizedVec;
}

// AVX2 implementation
void bulk_quantized_shuffle_euclidean(const unsigned char* shuffles, int codebookCount, int codesCount, const char* quantizedPartials, float delta, float minDistance, float* results) {
    int length = codebookCount * codesCount;
    int j = 0;
    for (; j + 16 <= codesCount; j += 16) {
        __m256i sum = _mm256_setzero_si256();

        for (int i = 0; i < codebookCount; i++) {
            int byte = j * codebookCount + i * 16;
            __m128i smallShuffle = _mm_loadu_epi8(shuffles + byte);
            __m256i shuffle = _mm256_cvtepu8_epi16(smallShuffle);
            __m256i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
            sum = _mm256_adds_epu16(sum, partialsVec);
        }

        __m128i quantizedResultsLeftRaw = _mm256_extracti32x4_epi32(sum, 0);
        __m128i quantizedResultsRightRaw = _mm256_extracti32x4_epi32(sum, 1);
        __m256 resultsLeft = dequantize(quantizedResultsLeftRaw, delta, minDistance);
        __m256 resultsRight = dequantize(quantizedResultsRightRaw, delta, minDistance);

        __m256 ones = _mm256_set1_ps(1.0);
        resultsLeft = _mm256_add_ps(resultsLeft, ones);
        resultsRight = _mm256_add_ps(resultsRight, ones);
        resultsLeft = _mm256_rcp14_ps(resultsLeft);
        resultsRight = _mm256_rcp14_ps(resultsRight);

        _mm256_storeu_ps(results + j, resultsLeft);
        _mm256_storeu_ps(results + j + 8, resultsRight);
    }
    if (j < codesCount) {
        for (; j < codesCount; j++) {
            unsigned int val = 0;
            for (int i = 0; i < codebookCount; i++) {
                unsigned int shuffle = computeSingleShuffle(i, j, shuffles, codesCount);
                val += combineBytes(i, shuffle, quantizedPartials);
            }
            results[j] = 1 / (1 + delta * val + minDistance);
        }
    }
}

// AVX2 implementation
void bulk_quantized_shuffle(const unsigned char* shuffles, int codebookCount, int codesCount, const char* quantizedPartials, float delta, float minDistance, float* results) {
    int length = codebookCount * codesCount;
    int j = 0;
    for (; j + 16 <= codesCount; j += 16) {
        __m256i sum = _mm256_setzero_si256();

        for (int i = 0; i < codebookCount; i++) {
            int byte = j * codebookCount + i * 16;
            __m128i smallShuffle = _mm_loadu_epi8(shuffles + byte);
            __m256i shuffle = _mm256_cvtepu8_epi16(smallShuffle);
            __m256i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
            sum = _mm256_adds_epu16(sum, partialsVec);
        }

        __m128i quantizedResultsLeftRaw = _mm256_extracti32x4_epi32(sum, 0);
        __m128i quantizedResultsRightRaw = _mm256_extracti32x4_epi32(sum, 1);
        __m256 resultsLeft = dequantize(quantizedResultsLeftRaw, delta, minDistance);
        __m256 resultsRight = dequantize(quantizedResultsRightRaw, delta, minDistance);

        __m256 ones = _mm256_set1_ps(1.0);
        resultsLeft = _mm256_add_ps(resultsLeft, ones);
        resultsRight = _mm256_add_ps(resultsRight, ones);
        resultsLeft = _mm256_rcp14_ps(resultsLeft);
        resultsRight = _mm256_rcp14_ps(resultsRight);

        _mm256_storeu_ps(results + j, resultsLeft);
        _mm256_storeu_ps(results + j + 8, resultsRight);
    }
    if (j < codesCount) {
        for (; j < codesCount; j++) {
            unsigned int val = 0;
            for (int i = 0; i < codebookCount; i++) {
                unsigned int shuffle = computeSingleShuffle(i, j, shuffles, codesCount);
                val += combineBytes(i, shuffle, quantizedPartials);
            }
            results[j] = (1 + delta * val + minDistance) / 2;
        }
    }
}

// AVX2 implementation
void bulk_quantized_shuffle_cosine(const unsigned char* shuffles, int codebookCount, int codesCount, const char* quantizedPartialSums, float sumDelta, float minDistance, const char* quantizedPartialMagnitudes, float magnitudeDelta, float minMagnitude, float queryMagnitudeSquared, float* results) {
    int length = codebookCount * codesCount;
    int j = 0;
    for (; j + 16 <= codesCount; j += 16) {
        __m256i sum = _mm256_setzero_si256();
        __m256i magnitude = _mm256_setzero_si256();

        for (int i = 0; i < codebookCount; i++) {
            int byte = j * codebookCount + i * 16;
            __m128i smallShuffle = _mm_loadu_epi8(shuffles + byte);
            __m256i shuffle = _mm256_cvtepu8_epi16(smallShuffle);

            __m256i partialsVec = lookup_partial_sums(shuffle, quantizedPartialSums, i);
            sum = _mm256_adds_epu16(sum, partialsVec);

            __m256i partialMagnitudesVec = lookup_partial_sums(shuffle, quantizedPartialMagnitudes, i);
            magnitude = _mm256_adds_epu16(magnitude, partialMagnitudesVec);
        }

        __m128i quantizedSumsLeftRaw = _mm256_extracti32x4_epi32(sum, 0);
        __m128i quantizedSumsRightRaw = _mm256_extracti32x4_epi32(sum, 1);
        __m256 sumsLeft = dequantize(quantizedSumsLeftRaw, sumDelta, minDistance);
        __m256 sumsRight = dequantize(quantizedSumsRightRaw, sumDelta, minDistance);

        __m128i quantizedMagnitudesLeftRaw = _mm256_extracti32x4_epi32(magnitude, 0);
        __m128i quantizedMagnitudesRightRaw = _mm256_extracti32x4_epi32(magnitude, 1);
        __m256 magnitudesLeft = dequantize(quantizedMagnitudesLeftRaw, magnitudeDelta, minMagnitude);
        __m256 magnitudesRight = dequantize(quantizedMagnitudesRightRaw, magnitudeDelta, minMagnitude);

        __m256 queryMagnitudeSquaredVec = _mm256_set1_ps(queryMagnitudeSquared);
        magnitudesLeft = _mm256_mul_ps(magnitudesLeft, queryMagnitudeSquaredVec);
        magnitudesRight = _mm256_mul_ps(magnitudesRight, queryMagnitudeSquaredVec);
        magnitudesLeft = _mm256_sqrt_ps(magnitudesLeft);
        magnitudesRight = _mm256_sqrt_ps(magnitudesRight);
        __m256 resultsLeft = _mm256_div_ps(sumsLeft, magnitudesLeft);
        __m256 resultsRight = _mm256_div_ps(sumsRight, magnitudesRight);

        __m256 ones = _mm256_set1_ps(1.0);
        resultsLeft = _mm256_add_ps(resultsLeft, ones);
        resultsRight = _mm256_add_ps(resultsRight, ones);
        resultsLeft = _mm256_div_ps(resultsLeft, _mm256_set1_ps(2.0));
        resultsRight = _mm256_div_ps(resultsRight, _mm256_set1_ps(2.0));

        _mm256_storeu_ps(results + j, resultsLeft);
        _mm256_storeu_ps(results + j + 8, resultsRight);
    }
    if (j < codesCount) {
        for (; j < codesCount; j++) {
            float sum = 0;
            float magnitude = 0;

            for (int i = 0; i < codebookCount; i++) {
                int shuffle = computeSingleShuffle(i, j, shuffles, codesCount);
                sum += combineBytes(i, shuffle, quantizedPartialSums);
                magnitude += combineBytes(i, shuffle, quantizedPartialMagnitudes);
            }

            float unquantizedSum = sumDelta * sum + minDistance;
            float unquantizedMagnitude = magnitudeDelta * magnitude + minMagnitude;
            float divisor = sqrt(unquantizedMagnitude * queryMagnitudeSquared);
            results[j] = (1 + (unquantizedSum / divisor)) / 2;
        }
    }
}

// AVX2 implementation
float assemble_and_sum(const float* data, int dataBase, const unsigned char* baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    int limit = baseOffsetsLength - (baseOffsetsLength % 8);
    __m256i indexRegister = initialIndexRegister;
    __m256i dataBaseVec = _mm256_set1_epi32(dataBase);
    baseOffsets = baseOffsets + baseOffsetsOffset;

    for (; i < limit; i += 8) {
        __m128i baseOffsetsRaw = _mm_loadu_si128((__m128i *)(baseOffsets + i));
        __m256i baseOffsetsInt = _mm256_cvtepu8_epi32(baseOffsetsRaw);
        // we have base offsets int, which we need to scale to index into data.
        // first, we want to initialize a vector with the lane number added as an index
        indexRegister = _mm256_add_epi32(indexRegister, indexIncrement);
        // then we want to multiply by dataBase
        __m256i scale = _mm256_mullo_epi32(indexRegister, dataBaseVec);
        // then we want to add the base offsets
        __m256i convOffsets = _mm256_add_epi32(scale, baseOffsetsInt);

        __m256 partials = _mm256_i32gather_ps(data, convOffsets, 4);
        sum = _mm256_add_ps(sum, partials);
    }

    // This code performs a horixontal reduce add across all 8 32-bit lanes
    __m256 temp = _mm256_permute2f128_ps(sum, sum, 1);
    sum = _mm256_add_ps(sum, temp);
    sum = _mm256_hadd_ps(sum, sum);
    sum = _mm256_hadd_ps(sum, sum);
    // This code extract a single element from sum
    __m128 low_part = _mm256_extractf128_ps(sum, 0);
    float res = _mm_extract_ps(low_part, 0);

    for (; i < baseOffsetsLength; i++) {
        res += data[dataBase * i + baseOffsets[i]];
    }

    return res;
}

// end of AVX2 section

/************************************************************************/
/*********************** 128-wide SIMD functions ************************/
/************************************************************************/

#elif defined(__SSE4_1__)

int simd_version(void) {
    return 0;
}

/* Layout for 128-wide SIMD:
 * The i-th position of quantizedPartials stores 256 quantized partial distances as 16-bit integers, totalling 512 bytes.
 * The 512 bytes of quantized partials are organized as 32 "segments", each one containing 8 16-bit unsigned integers.
 * The function apply_pairwise_shuffle permutes and blends consecutive segment pairs. These resulting blends get merged hierarchically.
 */

// SSE implementation
__attribute__((always_inline)) inline __m128i apply_pairwise_shuffle(__m128i shuffle, const char* quantizedPartials, int offset) {
    __m128i partialsVecA = _mm_loadu_epi16(quantizedPartials + offset);
    __m128i partialsVecB = _mm_loadu_epi16(quantizedPartials + offset + 16);
    __m128i partialsVecAB = _mm_permutex2var_epi16(partialsVecA, shuffle, partialsVecB);
    return partialsVecAB;
}

// SSE implementation
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

// SSE implementation
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

// SSE implementation
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

// SSE implementation
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

// SSE implementation
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

// SSE implementation
float assemble_and_sum(const float* data, int dataBase, const unsigned char* baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
    // It seems like this function cannot be implemented in SSE4_1 because of the lack of a gather operation.
    // For reference, see the implementations for AVX512 & AVX2.
    float res = 0;
    for (int i = 0; i < baseOffsetsLength; i++) {
        res += data[dataBase * i + baseOffsets[i]];
    }
    return res;
}

// end of SSE4_1 section

#endif
