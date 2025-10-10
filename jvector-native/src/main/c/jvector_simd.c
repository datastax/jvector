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

#if defined(SSE4_1)
__m128i maskFifthBit;
__m128i maskSixthBit;
__m128i maskSeventhBit;
__m128i maskEighthBit;
#endif

#if defined(AVX2)
__m256i maskSixthBit;
__m256i maskSeventhBit;
__m256i maskEighthBit;
#endif

#if defined(AVX-512F) && defined(AVX-512CD) && defined( AVX-512BW) && defined(AVX-512DQ) && defined(AVX-512VL)
__m512i maskSeventhBit;
__m512i maskEighthBit;
#endif

__attribute__((constructor))
void initialize_constants() {
    if (check_compatibility()) {
        #if defined(SSE4_1)
        maskFifthBit = _mm_set1_epi16(0x0010);
        maskSixthBit = _mm_set1_epi16(0x0020);
        maskSeventhBit = _mm_set1_epi16(0x0040);
        maskEighthBit = _mm_set1_epi16(0x0080);
        #endif

        #if defined(AVX2)
        maskSixthBit = _mm256_set1_epi16(0x0020);
        maskSeventhBit = _mm256_set1_epi16(0x0040);
        maskEighthBit = _mm256_set1_epi16(0x0080);
        #endif

        #if defined(AVX-512F) && defined(AVX-512CD) && defined( AVX-512BW) && defined(AVX-512DQ) && defined(AVX-512VL)
        maskSeventhBit = _mm512_set1_epi16(0x0040);
        maskEighthBit = _mm512_set1_epi16(0x0080);
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

/************************************************************************/
/*********************** 512-wide SIMD functions ************************/
/************************************************************************/

#if defined(AVX-512F) && defined(AVX-512CD) && defined( AVX-512BW) && defined(AVX-512DQ) && defined(AVX-512VL)

/* Layout for 512-wide SIMD:
 * The i-th position of quantizedPartials stores 256 quantized partial distances as 16-bit integers, totalling 512 bytes.
 * The 512 bytes of quantized partials are organized as 8 "segments", each one containing 32 16-bit unsigned integers.
 * The function apply_pairwise_shuffle512 permutes and blends consecutive segment pairs. These resulting blends get merged hierarchically.
 */

__attribute__((always_inline)) inline __m512i apply_pairwise_shuffle512(__m512i shuffle, const char* quantizedPartials, int offset) {
    __m512i partialsVecA = _mm512_loadu_epi16(quantizedPartials + offset);
    __m512i partialsVecB = _mm512_loadu_epi16(quantizedPartials + offset + 64);
    __m512i partialsVecAB = _mm512_permutex2var_epi16(partialsVecA, shuffle, partialsVecB);
    return partialsVecAB;
}

__attribute__((always_inline)) inline __m512i lookup_partial_sums(__m512i shuffle, const char* quantizedPartials, int i) {
    __m512i partialsVecAB = apply_pairwise_shuffle512(shuffle, quantizedPartials, i * 512);
    __m512i partialsVecCD = apply_pairwise_shuffle512(shuffle, quantizedPartials, i * 512 + 128);
    __m512i partialsVecEF = apply_pairwise_shuffle512(shuffle, quantizedPartials, i * 512 + 256);
    __m512i partialsVecGH = apply_pairwise_shuffle512(shuffle, quantizedPartials, i * 512 + 384);

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
    __m512i sum = _mm512_setzero_epi32();

    int byte = 0;
    int length = codebookCount * codesCount;
    for (int i = 0; byte + 32 <= length; byte += 32, i++) {
        __m256i smallShuffle = _mm256_loadu_epi8(shuffles + byte);
        __m512i shuffle = _mm512_cvtepu8_epi16(smallShuffle);
        __m512i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
        sum = _mm512_adds_epu16(sum, partialsVec);
    }
    if (byte < length) {
        // process the tail:
        // - load 16 32-bit floats, load only the first length-byte floats
        // - other floats are automatically set to zero
        __m256i smallShuffle = _mm256_maskz_loadu_epi8((1 << (length - byte)) - 1, shuffles + byte);
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

void bulk_quantized_shuffle_dot(const unsigned char* shuffles, int codebookCount, const char* quantizedPartials, float delta, float best, float* results) {
    __m512i sum = _mm512_setzero_epi32();

    int byte = 0;
    int length = codebookCount * codesCount;
    for (int i = 0; byte + 32 <= length; byte += 32, i++) {
        __m256i smallShuffle = _mm256_loadu_epi8(shuffles + byte);
        __m512i shuffle = _mm512_cvtepu8_epi16(smallShuffle);
        __m512i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
        sum = _mm512_adds_epu16(sum, partialsVec);
    }
    if (byte < length) {
        // process the tail:
        // - load 16 32-bit floats, load only the first length-byte floats
        // - other floats are automatically set to zero
        __m256i smallShuffle = _mm256_maskz_loadu_epi8((1 << (length - byte)) - 1, shuffles + byte);
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

void bulk_quantized_shuffle_cosine(const unsigned char* shuffles, int codebookCount, const char* quantizedPartialSums, float sumDelta, float minDistance, const char* quantizedPartialMagnitudes, float magnitudeDelta, float minMagnitude, float queryMagnitudeSquared, float* results) {
    __m512i sum = _mm512_setzero_epi32();
    __m512i magnitude = _mm512_setzero_epi32();

    int byte = 0;
    int length = codebookCount * codesCount;
    for (int i = 0; byte + 32 <= length; byte += 32, i++) {
        __m256i smallShuffle = _mm256_loadu_epi8(shuffles + byte);
        __m512i shuffle = _mm512_cvtepu8_epi16(smallShuffle);
        __m512i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
        sum = _mm512_adds_epu16(sum, partialsVec);

        __m512i partialMagnitudesVec = lookup_partial_sums(shuffle, quantizedPartialMagnitudes, i);
        magnitude = _mm512_adds_epu16(magnitude, partialMagnitudesVec);
    }
    if (byte < length) {
        // process the tail:
        // - load 16 32-bit floats, load only the first length-byte floats
        // - other floats are automatically set to zero
        __m256i smallShuffle = _mm256_maskz_loadu_epi8((1 << (length - byte)) - 1, shuffles + byte);
        __m512i shuffle = _mm512_cvtepu8_epi16(smallShuffle);
        __m512i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
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
    _mm512_storeu_ps(results, resultsLeft);
    _mm512_storeu_ps(results + 16, resultsRight);
}

#endif // defined(AVX-512F) && defined(AVX-512CD) && defined( AVX-512BW) && defined(AVX-512DQ) && defined(AVX-512VL)

/************************************************************************/
/*********************** 256-wide SIMD functions ************************/
/************************************************************************/

#if defined(AVX2)

/* Layout for 256-wide SIMD:
 * The i-th position of quantizedPartials stores 256 quantized partial distances as 16-bit integers, totalling 512 bytes.
 * The 512 bytes of quantized partials are organized as 16 "segments", each one containing 16 16-bit unsigned integers.
 * The function apply_pairwise_shuffle permutes and blends consecutive segment pairs. These resulting blends get merged hierarchically.
 */

__attribute__((always_inline)) inline __m256i apply_pairwise_shuffle256(__m256i shuffle, const char* quantizedPartials, int offset) {
    __m256i partialsVecA = _mm256_loadu_epi16(quantizedPartials + offset);
    __m256i partialsVecB = _mm256_loadu_epi16(quantizedPartials + offset + 32);
    __m256i partialsVecAB = _mm256_permutex2var_epi16(partialsVecA, shuffle, partialsVecB);
    return partialsVecAB;
}

__attribute__((always_inline)) inline __m256i lookup_partial_sums(__m256i shuffle, const char* quantizedPartials, int i) {
    __m256i partialsVecA = apply_pairwise_shuffle256(shuffle, quantizedPartials, i * 512);
    __m256i partialsVecB = apply_pairwise_shuffle256(shuffle, quantizedPartials, i * 512 + 64);
    __m256i partialsVecC = apply_pairwise_shuffle256(shuffle, quantizedPartials, i * 512 + 128);
    __m256i partialsVecD = apply_pairwise_shuffle256(shuffle, quantizedPartials, i * 512 + 192);
    __m256i partialsVecE = apply_pairwise_shuffle256(shuffle, quantizedPartials, i * 512 + 256);
    __m256i partialsVecF = apply_pairwise_shuffle256(shuffle, quantizedPartials, i * 512 + 320);
    __m256i partialsVecG = apply_pairwise_shuffle256(shuffle, quantizedPartials, i * 512 + 384);
    __m256i partialsVecH = apply_pairwise_shuffle256(shuffle, quantizedPartials, i * 512 + 448);

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

// Dequantize a 128-bit vector containing 8 unsigned 16-bit integers into a 256-bit vector containing 8 32-bit floats
__attribute__((always_inline)) inline __m256 dequantize(__m128i quantizedVec, float delta, float base) {
    __m256i quantizedVecWidened = _mm256_cvtepu16_epi32(quantizedVec);
    __m256 floatVec = _mm256_cvtepi32_ps(quantizedVecWidened);
    __m256 deltaVec = _mm256_set1_ps(delta);
    __m256 baseVec = _mm256_set1_ps(base);
    __m256 dequantizedVec = _mm256_fmadd_ps(floatVec, deltaVec, baseVec);
    return dequantizedVec;
}

void bulk_quantized_shuffle_euclidean(const unsigned char* shuffles, int codebookCount, int codesCount, const char* quantizedPartials, float delta, float minDistance, float* results) {
    __m256i sum = _mm256_setzero_epi32();

    int byte = 0;
    int length = codebookCount * codesCount;
    for (int i = 0; byte + 16 <= length; byte += 16, i++) {
        __m128i smallShuffle = _mm_loadu_epi8(shuffles + byte);
        __m256i shuffle = _mm256_cvtepu8_epi16(smallShuffle);
        __m256i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
        sum = _mm256_adds_epu16(sum, partialsVec);
    }
    if (byte < length) {
        // process the tail:
        // - load 16 32-bit floats, load only the first length-byte floats
        // - other floats are automatically set to zero
        __m128i smallShuffle = _mm_maskz_loadu_epi8((1 << (length - byte)) - 1, shuffles + byte);
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
    _mm256_storeu_ps(results, resultsLeft);
    _mm256_storeu_ps(results + 8, resultsRight);
}

void bulk_quantized_shuffle(const unsigned char* shuffles, int codebookCount, const char* quantizedPartials, float delta, float best, float* results) {
    __m256i sum = _mm256_setzero_epi32();

    int byte = 0;
    int length = codebookCount * codesCount;
    for (int i = 0; byte + 16 <= length; byte += 16, i++) {
        __m128i smallShuffle = _mm_loadu_epi8(shuffles + byte);
        __m256i shuffle = _mm256_cvtepu8_epi16(smallShuffle);
        __m256i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
        sum = _mm256_adds_epu16(sum, partialsVec);
    }
    if (byte < length) {
        // process the tail:
        // - load 16 32-bit floats, load only the first length-byte floats
        // - other floats are automatically set to zero
        __m128i smallShuffle = _mm_maskz_loadu_epi8((1 << (length - byte)) - 1, shuffles + byte);
        __m256i shuffle = _mm256_cvtepu8_epi16(smallShuffle);
        __m256i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
        sum = _mm256_adds_epu16(sum, partialsVec);
    }

    __m128i quantizedResultsLeftRaw = _mm256_extracti32x4_epi32(sum, 0);
    __m128i quantizedResultsRightRaw = _mm256_extracti32x4_epi32(sum, 1);
    __m256 resultsLeft = dequantize(quantizedResultsLeftRaw, delta, best);
    __m256 resultsRight = dequantize(quantizedResultsRightRaw, delta, best);

    __m256 ones = _mm256_set1_ps(1.0);
    resultsLeft = _mm256_add_ps(resultsLeft, ones);
    resultsRight = _mm256_add_ps(resultsRight, ones);
    resultsLeft = _mm256_div_ps(resultsLeft, _mm512_set1_ps(2.0));
    resultsRight = _mm256_div_ps(resultsRight, _mm512_set1_ps(2.0));
    _mm256_storeu_ps(results, resultsLeft);
    _mm256_storeu_ps(results + 8, resultsRight);
}

void bulk_quantized_shuffle_cosine(const unsigned char* shuffles, int codebookCount, const char* quantizedPartialSums, float sumDelta, float minDistance, const char* quantizedPartialMagnitudes, float magnitudeDelta, float minMagnitude, float queryMagnitudeSquared, float* results) {
    __m256i sum = _mm256_setzero_epi32();
    __m256i magnitude = _mm256_setzero_epi32();

    int byte = 0;
    int length = codebookCount * codesCount;
    for (int i = 0; byte + 16 <= length; byte += 16, i++) {
        __m128i smallShuffle = _mm_loadu_epi8(shuffles + byte);
        __m256i shuffle = _mm256_cvtepu8_epi16(smallShuffle);
        __m256i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
        sum = _mm256_adds_epu16(sum, partialsVec);

        __m256i partialMagnitudesVec = lookup_partial_sums(shuffle, quantizedPartialMagnitudes, i);
        magnitude = _mm256_adds_epu16(magnitude, partialMagnitudesVec);
    }
    if (byte < length) {
        // process the tail:
        // - load 16 32-bit floats, load only the first length-byte floats
        // - other floats are automatically set to zero
        __m128i smallShuffle = _mm_maskz_loadu_epi8((1 << (length - byte)) - 1, shuffles + byte);
        __m256i shuffle = _mm256_cvtepu8_epi16(smallShuffle);
        __m256i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
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
    resultsLeft = _mm256_div_ps(resultsLeft, _mm512_set1_ps(2.0));
    resultsRight = _mm256_div_ps(resultsRight, _mm512_set1_ps(2.0));
    _mm256_storeu_ps(results, resultsLeft);
    _mm256_storeu_ps(results + 8, resultsRight);
}

#endif // defined(AVX2)

/************************************************************************/
/*********************** 128-wide SIMD functions ************************/
/************************************************************************/

#if defined(SSE4_1)

/* Layout for 128-wide SIMD:
 * The i-th position of quantizedPartials stores 256 quantized partial distances as 16-bit integers, totalling 512 bytes.
 * The 512 bytes of quantized partials are organized as 32 "segments", each one containing 8 16-bit unsigned integers.
 * The function apply_pairwise_shuffle permutes and blends consecutive segment pairs. These resulting blends get merged hierarchically.
 */

__attribute__((always_inline)) inline __m128i apply_pairwise_shuffle128(__m128i shuffle, const char* quantizedPartials, int offset) {
    __m128i partialsVecA = _mm_loadu_epi16(quantizedPartials + offset);
    __m128i partialsVecB = _mm_loadu_epi16(quantizedPartials + offset + 16);
    __m128i partialsVecAB = _mm_permutex2var_epi16(partialsVecA, shuffle, partialsVecB);
    return partialsVecAB;
}

__attribute__((always_inline)) inline __m128i lookup_partial_sums(__m128i shuffle, const char* quantizedPartials, int i) {
    __m128i partialsVecA = apply_pairwise_shuffle128(shuffle, quantizedPartials, i * 512);
    __m128i partialsVecB = apply_pairwise_shuffle128(shuffle, quantizedPartials, i * 512 + 32);
    __m128i partialsVecC = apply_pairwise_shuffle128(shuffle, quantizedPartials, i * 512 + 64);
    __m128i partialsVecD = apply_pairwise_shuffle128(shuffle, quantizedPartials, i * 512 + 96);
    __m128i partialsVecE = apply_pairwise_shuffle128(shuffle, quantizedPartials, i * 512 + 128);
    __m128i partialsVecF = apply_pairwise_shuffle128(shuffle, quantizedPartials, i * 512 + 160);
    __m128i partialsVecG = apply_pairwise_shuffle128(shuffle, quantizedPartials, i * 512 + 192);
    __m128i partialsVecH = apply_pairwise_shuffle128(shuffle, quantizedPartials, i * 512 + 224);
    __m128i partialsVecI = apply_pairwise_shuffle128(shuffle, quantizedPartials, i * 512 + 256);
    __m128i partialsVecJ = apply_pairwise_shuffle128(shuffle, quantizedPartials, i * 512 + 288);
    __m128i partialsVecK = apply_pairwise_shuffle128(shuffle, quantizedPartials, i * 512 + 320);
    __m128i partialsVecL = apply_pairwise_shuffle128(shuffle, quantizedPartials, i * 512 + 352);
    __m128i partialsVecM = apply_pairwise_shuffle128(shuffle, quantizedPartials, i * 512 + 384);
    __m128i partialsVecN = apply_pairwise_shuffle128(shuffle, quantizedPartials, i * 512 + 416);
    __m128i partialsVecO = apply_pairwise_shuffle128(shuffle, quantizedPartials, i * 512 + 448);
    __m128i partialsVecP = apply_pairwise_shuffle128(shuffle, quantizedPartials, i * 512 + 480);

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
    __m128i sum = _mm128_setzero_epi32();

    int byte = 0;
    int length = codebookCount * codesCount;
    for (int i = 0; byte + 8 <= length; byte += 8, i++) {
        __m128i smallShuffle = _mm_loadu_epi8(shuffles + byte);
        __m128i shuffle = _mm_cvtepu8_epi16(smallShuffle);
        __m128i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
        sum = _mm_adds_epu16(sum, partialsVec);
    }
    if (byte < length) {
        // process the tail:
        // - load 16 32-bit floats, load only the first length-byte floats
        // - other floats are automatically set to zero
        __m128i smallShuffle = _mm_maskz_loadu_epi8((1 << (length - byte)) - 1, shuffles + byte);
        __m128i shuffle = _mm_cvtepu8_epi16(smallShuffle);
        __m128i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
        sum = _mm_adds_epu16(sum, partialsVec);
    }

    __m128i quantizedResultsLeftRaw = _mm_cvtepu16_epi32(_mm_extract_epi64(sum, 0));
    __m128i quantizedResultsRightRaw = _mm_cvtepu16_epi32(_mm_extract_epi64(sum, 1));
    __m128 resultsLeft = dequantize(quantizedResultsLeftRaw, delta, minDistance);
    __m128 resultsRight = dequantize(quantizedResultsRightRaw, delta, minDistance);

    __m128 ones = _mm_set1_ps(1.0);
    resultsLeft = _mm_add_ps(resultsLeft, ones);
    resultsRight = _mm_add_ps(resultsRight, ones);
    resultsLeft = _mm_rcp14_ps(resultsLeft);
    resultsRight = _mm_rcp14_ps(resultsRight);
    _mm_storeu_ps(results, resultsLeft);
    _mm_storeu_ps(results + 4, resultsRight);
}

void bulk_quantized_shuffle_dot(const unsigned char* shuffles, int codebookCount, const char* quantizedPartials, float delta, float best, float* results) {
    __m128i sum = _mm256_setzero_epi32();

    int byte = 0;
    int length = codebookCount * codesCount;
    for (int i = 0; byte + 8 <= length; byte += 8, i++) {
        __m128i smallShuffle = _mm_loadu_epi8(shuffles + byte);
        __m128i shuffle = _mm_cvtepu8_epi16(smallShuffle);
        __m128i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
        sum = _mm_adds_epu16(sum, partialsVec);
    }
    if (byte < length) {
        // process the tail:
        // - load 16 32-bit floats, load only the first length-byte floats
        // - other floats are automatically set to zero
        __m128i smallShuffle = _mm_maskz_loadu_epi8((1 << (length - byte)) - 1, shuffles + byte);
        __m128i shuffle = _mm_cvtepu8_epi16(smallShuffle);
        __m128i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
        sum = _mm_adds_epu16(sum, partialsVec);
    }

    __m128i quantizedResultsLeftRaw = _mm_cvtepu16_epi32(_mm_extract_epi64(sum, 0));
    __m128i quantizedResultsRightRaw = _mm_cvtepu16_epi32(_mm_extract_epi64(sum, 1));
    __m128 resultsLeft = dequantize(quantizedResultsLeftRaw, delta, best);
    __m128 resultsRight = dequantize(quantizedResultsRightRaw, delta, best);

    __m128 ones = _mm_set1_ps(1.0);
    resultsLeft = _mm_add_ps(resultsLeft, ones);
    resultsRight = _mm_add_ps(resultsRight, ones);
    resultsLeft = _mm_div_ps(resultsLeft, _mm_set1_ps(2.0));
    resultsRight = _mm_div_ps(resultsRight, _mm_set1_ps(2.0));
    _mm_storeu_ps(results, resultsLeft);
    _mm_storeu_ps(results + 4, resultsRight);
}

void bulk_quantized_shuffle_cosine(const unsigned char* shuffles, int codebookCount, const char* quantizedPartialSums, float sumDelta, float minDistance, const char* quantizedPartialMagnitudes, float magnitudeDelta, float minMagnitude, float queryMagnitudeSquared, float* results) {
    __m256i sum = _mm256_setzero_epi32();
    __m256i magnitude = _mm256_setzero_epi32();

    int byte = 0;
    int length = codebookCount * codesCount;
    for (int i = 0; byte + 8 <= length; byte += 8, i++) {
        __m128i smallShuffle = _mm_loadu_epi8(shuffles + byte);
        __m256i shuffle = _mm_cvtepu8_epi16(smallShuffle);
        __m256i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
        sum = _mm256_adds_epu16(sum, partialsVec);

        __m256i partialMagnitudesVec = lookup_partial_sums(shuffle, quantizedPartialMagnitudes, i);
        magnitude = _mm256_adds_epu16(magnitude, partialMagnitudesVec);
    }
    if (byte < length) {
        // process the tail:
        // - load 16 32-bit floats, load only the first length-byte floats
        // - other floats are automatically set to zero
        __m128i smallShuffle = _mm_maskz_loadu_epi8((1 << (length - byte)) - 1, shuffles + byte);
        __m256i shuffle = _mm256_cvtepu8_epi16(smallShuffle);
        __m256i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
        sum = _mm256_adds_epu16(sum, partialsVec);

        __m256i partialMagnitudesVec = lookup_partial_sums(shuffle, quantizedPartialMagnitudes, i);
        magnitude = _mm256_adds_epu16(magnitude, partialMagnitudesVec);
    }

    __m128i quantizedResultsLeftRaw = _mm_cvtepu16_epi32(_mm_extract_epi64(sum, 0));
    __m128i quantizedResultsRightRaw = _mm_cvtepu16_epi32(_mm_extract_epi64(sum, 1));
    __m128 sumsLeft = dequantize(quantizedSumsLeftRaw, sumDelta, minDistance);
    __m128 sumsRight = dequantize(quantizedSumsRightRaw, sumDelta, minDistance);

    __m128i quantizedMagnitudesLeftRaw = _mm_cvtepu16_epi32(_mm_extract_epi64(magnitude, 0));
    __m128i quantizedMagnitudesRightRaw = _mm_cvtepu16_epi32(_mm_extract_epi64(magnitude, 1));
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
    _mm_storeu_ps(results, resultsLeft);
    _mm_storeu_ps(results + 4, resultsRight);
}

#endif // defined(SSE4_1)
