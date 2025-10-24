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

#if defined(__AVX2__)

__m256i maskSixthBit;
__m256i maskSeventhBit;
__m256i maskEighthBit;

__m256i initialIndexRegister;
__m256i indexIncrement;

__attribute__((constructor))
void initialize_constants() {
    if (check_compatibility()) {
        maskSixthBit = _mm256_set1_epi16(0x0020);
        maskSeventhBit = _mm256_set1_epi16(0x0040);
        maskEighthBit = _mm256_set1_epi16(0x0080);

        initialIndexRegister = _mm256_setr_epi32(-8, -7, -6, -5, -4, -3, -2, -1);
        indexIncrement = _mm256_set1_epi32(8);
    }
}

int simd_version(void) {
    return 1;
}

void quantized_partials(float delta, const float* partials, int codebookCount, int codebookSize, const float* partialBases, unsigned char* quantizedPartials) {
    __m256 invDeltaVec = _mm256_set1_ps(1 / delta);
    __m256 zeros = _mm256_setzero_ps();
    __m256 max65535 = _mm256_set1_ps(65535);
    __m256i vectorFF = _mm256_set1_epi32(0xFF);
    __m256i shuffle = _mm256_set_epi8(
        1, 3, 5, 7, 9, 11, 13, 15,
        14, 12, 10, 8, 6, 4, 2, 0,
        17, 19, 21, 23, 25, 27, 29, 31,
        30, 28, 26, 24, 22, 20, 18, 16
    );
    __m128i storeMask = _mm_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0,
        0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000
    );

    for (int i = 0; i < codebookCount; i++) {
        const float codebookBase = partialBases[i];
        __m256 codebookBaseVector = _mm256_set1_ps(codebookBase);

        int j = 0;
        for (; j < codebookSize; j += 8) {
            int store_offset = (2 * i) * codebookSize + j;

            __m256 partialVector = _mm256_loadu_ps(partials + i * codebookSize + j);
            partialVector = _mm256_sub_ps(partialVector, codebookBaseVector);
            partialVector = _mm256_mul_ps(partialVector, invDeltaVec);
            partialVector = _mm256_min_ps(_mm256_max_ps(partialVector, zeros), max65535);
            __m256i quantized = _mm256_cvtps_epi32(partialVector);
            __m256i lowBytes = _mm256_and_si256(quantized, vectorFF);
            __m256i highBytes = _mm256_and_si256(_mm256_srli_epi32(quantized, 8), vectorFF);
            __m256i packed = _mm256_packus_epi32(lowBytes, highBytes);
            packed = _mm256_permute4x64_epi64(packed, 0b11011000);
            packed = _mm256_shuffle_epi8(packed, shuffle);

            // Extract the low bytes
            __m128i lowPacked = _mm256_castsi256_si128(packed);
            _mm_maskmoveu_si128(lowPacked, storeMask, quantizedPartials + store_offset);

            // Put the high bytes in the first 128 bits
            packed = _mm256_permute4x64_epi64(packed, 0b01001110);

            // Extract the high bytes
            __m128i highPacked = _mm256_castsi256_si128(packed);
            _mm_maskmoveu_si128(highPacked, storeMask, quantizedPartials + store_offset + codebookSize);
        }
        for (; j < codebookSize; j++) {
            int store_offset = (2 * i) * codebookSize + j;
            float val = partials[i * codebookSize + j];
            int quantized = (int) fminf(fmaxf((val - codebookBase) / delta, 0), 65535);
            quantizedPartials[store_offset] = (char) (quantized & 0xFF);
            quantizedPartials[store_offset + codebookSize] = (char) ((quantized >> 8) & 0xFF);
        }
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

/* Layout for 256-wide SIMD:
 * The i-th position of quantizedPartials stores 256 quantized partial distances as 16-bit integers, totalling 512 bytes.
 * The 512 bytes of quantized partials are organized as 16 "segments", each one containing 16 16-bit unsigned integers.
 * The function apply_pairwise_shuffle permutes and blends consecutive segment pairs. These resulting blends get merged hierarchically.
 */

//__attribute__((always_inline)) inline __m256i apply_pairwise_shuffle(__m256i shuffle, const char* quantizedPartials, int offset) {
//    __m128i partialsVecA = _mm_loadu_epi16(quantizedPartials + offset);
//    __m128i partialsVecB = _mm_loadu_epi16(quantizedPartials + offset + 16);
//    __m128i partialsVecAB = _mm_permutex2var_epi16(partialsVecA, shuffle, partialsVecB);
//
//    __m128i partialsVecC = _mm_loadu_epi16(quantizedPartials + offset + 32);
//    __m128i partialsVecD = _mm_loadu_epi16(quantizedPartials + offset + 48);
//    __m128i partialsVecCD = _mm_permutex2var_epi16(partialsVecC, shuffle, partialsVecD);
//
//    __m256 partialsABCD = _mm256_castps128_ps256(a);
//    c = _mm256_insertf128_ps(c,b,1);
//
//    __m256i partialsVecA = _mm256_loadu_si256((const __m256i*) quantizedPartials + offset);
//    __m256i partialsVecB = _mm256_loadu_si256((const __m256i*) quantizedPartials + offset + 32);
//    __m256i partialsVecAB = _mm256_permutex2var_epi16(partialsVecA, shuffle, partialsVecB);
//    return partialsVecAB;
//}
//
//__attribute__((always_inline)) inline __m256i lookup_partial_sums(__m256i shuffle, const char* quantizedPartials, int i) {
//    __m256i partialsVecA = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512);
//    __m256i partialsVecB = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 64);
//    __m256i partialsVecC = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 128);
//    __m256i partialsVecD = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 192);
//    __m256i partialsVecE = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 256);
//    __m256i partialsVecF = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 320);
//    __m256i partialsVecG = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 384);
//    __m256i partialsVecH = apply_pairwise_shuffle(shuffle, quantizedPartials, i * 512 + 448);
//
//    __mmask16 maskSixth = _mm256_test_epi16_mask(shuffle, maskSixthBit);
//    __m256i partialsVecAB = _mm256_mask_blend_epi16(maskSixth, partialsVecA, partialsVecB);
//    __m256i partialsVecCD = _mm256_mask_blend_epi16(maskSixth, partialsVecC, partialsVecD);
//    __m256i partialsVecEF = _mm256_mask_blend_epi16(maskSixth, partialsVecE, partialsVecF);
//    __m256i partialsVecGH = _mm256_mask_blend_epi16(maskSixth, partialsVecG, partialsVecH);
//
//    __mmask16 maskSeven = _mm256_test_epi16_mask(shuffle, maskSeventhBit);
//    __m256i partialsVecABCD = _mm256_mask_blend_epi16(maskSeven, partialsVecAB, partialsVecCD);
//    __m256i partialsVecEFGH = _mm256_mask_blend_epi16(maskSeven, partialsVecEF, partialsVecGH);
//
//    __mmask16 maskEight = _mm256_test_epi16_mask(shuffle, maskEighthBit);
//    __m256i partialSumsVec = _mm256_mask_blend_epi16(maskEight, partialsVecABCD, partialsVecEFGH);
//
//    return partialSumsVec;
//}
//
//// Dequantize a 128-bit vector containing 8 unsigned 16-bit integers into a 256-bit vector containing 8 32-bit floats
//__attribute__((always_inline)) inline __m256 dequantize(__m128i quantizedVec, float delta, float base) {
//    __m256i quantizedVecWidened = _mm256_cvtepu16_epi32(quantizedVec);
//    __m256 floatVec = _mm256_cvtepi32_ps(quantizedVecWidened);
//    __m256 deltaVec = _mm256_set1_ps(delta);
//    __m256 baseVec = _mm256_set1_ps(base);
//    __m256 dequantizedVec = _mm256_fmadd_ps(floatVec, deltaVec, baseVec);
//    return dequantizedVec;
//}
//
//void bulk_quantized_shuffle_euclidean(const unsigned char* shuffles, int codebookCount, int codesCount, const char* quantizedPartials, float delta, float minDistance, float* results) {
//    int length = codebookCount * codesCount;
//    int j = 0;
//    for (; j + 16 <= codesCount; j += 16) {
//        __m256i sum = _mm256_setzero_si256();
//
//        for (int i = 0; i < codebookCount; i++) {
//            int byte = j * codebookCount + i * 16;
//            __m128i smallShuffle = _mm_loadu_epi8(shuffles + byte);
//            __m256i shuffle = _mm256_cvtepu8_epi16(smallShuffle);
//            __m256i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
//            sum = _mm256_adds_epu16(sum, partialsVec);
//        }
//
//        __m128i quantizedResultsLeftRaw = _mm256_extracti32x4_epi32(sum, 0);
//        __m128i quantizedResultsRightRaw = _mm256_extracti32x4_epi32(sum, 1);
//        __m256 resultsLeft = dequantize(quantizedResultsLeftRaw, delta, minDistance);
//        __m256 resultsRight = dequantize(quantizedResultsRightRaw, delta, minDistance);
//
//        __m256 ones = _mm256_set1_ps(1.0);
//        resultsLeft = _mm256_add_ps(resultsLeft, ones);
//        resultsRight = _mm256_add_ps(resultsRight, ones);
//        resultsLeft = _mm256_rcp14_ps(resultsLeft);
//        resultsRight = _mm256_rcp14_ps(resultsRight);
//
//        _mm256_storeu_ps(results + j, resultsLeft);
//        _mm256_storeu_ps(results + j + 8, resultsRight);
//    }
//    if (j < codesCount) {
//        for (; j < codesCount; j++) {
//            unsigned int val = 0;
//            for (int i = 0; i < codebookCount; i++) {
//                unsigned int shuffle = computeSingleShuffle(i, j, shuffles, codesCount);
//                val += combineBytes(i, shuffle, quantizedPartials);
//            }
//            results[j] = 1 / (1 + delta * val + minDistance);
//        }
//    }
//}
//
//void bulk_quantized_shuffle(const unsigned char* shuffles, int codebookCount, int codesCount, const char* quantizedPartials, float delta, float minDistance, float* results) {
//    int length = codebookCount * codesCount;
//    int j = 0;
//    for (; j + 16 <= codesCount; j += 16) {
//        __m256i sum = _mm256_setzero_si256();
//
//        for (int i = 0; i < codebookCount; i++) {
//            int byte = j * codebookCount + i * 16;
//            __m128i smallShuffle = _mm_loadu_epi8(shuffles + byte);
//            __m256i shuffle = _mm256_cvtepu8_epi16(smallShuffle);
//            __m256i partialsVec = lookup_partial_sums(shuffle, quantizedPartials, i);
//            sum = _mm256_adds_epu16(sum, partialsVec);
//        }
//
//        __m128i quantizedResultsLeftRaw = _mm256_extracti32x4_epi32(sum, 0);
//        __m128i quantizedResultsRightRaw = _mm256_extracti32x4_epi32(sum, 1);
//        __m256 resultsLeft = dequantize(quantizedResultsLeftRaw, delta, minDistance);
//        __m256 resultsRight = dequantize(quantizedResultsRightRaw, delta, minDistance);
//
//        __m256 ones = _mm256_set1_ps(1.0);
//        resultsLeft = _mm256_add_ps(resultsLeft, ones);
//        resultsRight = _mm256_add_ps(resultsRight, ones);
//        resultsLeft = _mm256_rcp14_ps(resultsLeft);
//        resultsRight = _mm256_rcp14_ps(resultsRight);
//
//        _mm256_storeu_ps(results + j, resultsLeft);
//        _mm256_storeu_ps(results + j + 8, resultsRight);
//    }
//    if (j < codesCount) {
//        for (; j < codesCount; j++) {
//            unsigned int val = 0;
//            for (int i = 0; i < codebookCount; i++) {
//                unsigned int shuffle = computeSingleShuffle(i, j, shuffles, codesCount);
//                val += combineBytes(i, shuffle, quantizedPartials);
//            }
//            results[j] = (1 + delta * val + minDistance) / 2;
//        }
//    }
//}
//
//void bulk_quantized_shuffle_cosine(const unsigned char* shuffles, int codebookCount, int codesCount, const char* quantizedPartialSums, float sumDelta, float minDistance, const char* quantizedPartialMagnitudes, float magnitudeDelta, float minMagnitude, float queryMagnitudeSquared, float* results) {
//    int length = codebookCount * codesCount;
//    int j = 0;
//    for (; j + 16 <= codesCount; j += 16) {
//        __m256i sum = _mm256_setzero_si256();
//        __m256i magnitude = _mm256_setzero_si256();
//
//        for (int i = 0; i < codebookCount; i++) {
//            int byte = j * codebookCount + i * 16;
//            __m128i smallShuffle = _mm_loadu_epi8(shuffles + byte);
//            __m256i shuffle = _mm256_cvtepu8_epi16(smallShuffle);
//
//            __m256i partialsVec = lookup_partial_sums(shuffle, quantizedPartialSums, i);
//            sum = _mm256_adds_epu16(sum, partialsVec);
//
//            __m256i partialMagnitudesVec = lookup_partial_sums(shuffle, quantizedPartialMagnitudes, i);
//            magnitude = _mm256_adds_epu16(magnitude, partialMagnitudesVec);
//        }
//
//        __m128i quantizedSumsLeftRaw = _mm256_extracti32x4_epi32(sum, 0);
//        __m128i quantizedSumsRightRaw = _mm256_extracti32x4_epi32(sum, 1);
//        __m256 sumsLeft = dequantize(quantizedSumsLeftRaw, sumDelta, minDistance);
//        __m256 sumsRight = dequantize(quantizedSumsRightRaw, sumDelta, minDistance);
//
//        __m128i quantizedMagnitudesLeftRaw = _mm256_extracti32x4_epi32(magnitude, 0);
//        __m128i quantizedMagnitudesRightRaw = _mm256_extracti32x4_epi32(magnitude, 1);
//        __m256 magnitudesLeft = dequantize(quantizedMagnitudesLeftRaw, magnitudeDelta, minMagnitude);
//        __m256 magnitudesRight = dequantize(quantizedMagnitudesRightRaw, magnitudeDelta, minMagnitude);
//
//        __m256 queryMagnitudeSquaredVec = _mm256_set1_ps(queryMagnitudeSquared);
//        magnitudesLeft = _mm256_mul_ps(magnitudesLeft, queryMagnitudeSquaredVec);
//        magnitudesRight = _mm256_mul_ps(magnitudesRight, queryMagnitudeSquaredVec);
//        magnitudesLeft = _mm256_sqrt_ps(magnitudesLeft);
//        magnitudesRight = _mm256_sqrt_ps(magnitudesRight);
//        __m256 resultsLeft = _mm256_div_ps(sumsLeft, magnitudesLeft);
//        __m256 resultsRight = _mm256_div_ps(sumsRight, magnitudesRight);
//
//        __m256 ones = _mm256_set1_ps(1.0);
//        resultsLeft = _mm256_add_ps(resultsLeft, ones);
//        resultsRight = _mm256_add_ps(resultsRight, ones);
//        resultsLeft = _mm256_div_ps(resultsLeft, _mm256_set1_ps(2.0));
//        resultsRight = _mm256_div_ps(resultsRight, _mm256_set1_ps(2.0));
//
//        _mm256_storeu_ps(results + j, resultsLeft);
//        _mm256_storeu_ps(results + j + 8, resultsRight);
//    }
//    if (j < codesCount) {
//        for (; j < codesCount; j++) {
//            float sum = 0;
//            float magnitude = 0;
//
//            for (int i = 0; i < codebookCount; i++) {
//                int shuffle = computeSingleShuffle(i, j, shuffles, codesCount);
//                sum += combineBytes(i, shuffle, quantizedPartialSums);
//                magnitude += combineBytes(i, shuffle, quantizedPartialMagnitudes);
//            }
//
//            float unquantizedSum = sumDelta * sum + minDistance;
//            float unquantizedMagnitude = magnitudeDelta * magnitude + minMagnitude;
//            float divisor = sqrt(unquantizedMagnitude * queryMagnitudeSquared);
//            results[j] = (1 + (unquantizedSum / divisor)) / 2;
//        }
//    }
//}


__attribute__((always_inline)) inline float reduce_add_avx2_ps(__m256 ymm_vec) {
    // Step 1: Sum the high and low 128-bit lanes.
    // This reduces the 256-bit vector to a 128-bit vector containing the sum.
    __m128 sum128 = _mm256_extractf128_ps(ymm_vec, 1);
    sum128 = _mm_add_ps(sum128, _mm256_castps256_ps128(ymm_vec));

    // Step 2: Sum adjacent elements within the 128-bit vector.
    sum128 = _mm_hadd_ps(sum128, sum128);

    // Step 3: Sum the remaining two elements.
    sum128 = _mm_hadd_ps(sum128, sum128);

    // Step 4: Extract the final scalar result.
    return _mm_cvtss_f32(sum128);
}

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

    float res = reduce_add_avx2_ps(sum);

    for (; i < baseOffsetsLength; i++) {
        res += data[dataBase * i + baseOffsets[i]];
    }

    return res;
}

float pq_decoded_cosine_similarity(const unsigned char* baseOffsets, int baseOffsetsOffset, int baseOffsetsLength, int clusterCount, const float* partialSums, const float* aMagnitude, float bMagnitude) {
    __m256 sum = _mm256_setzero_ps();
    __m256 vaMagnitude = _mm256_setzero_ps();
    int i = 0;
    int limit = baseOffsetsLength - (baseOffsetsLength % 8);
    __m256i indexRegister = initialIndexRegister;
    __m256i scale = _mm256_set1_epi32(clusterCount);
    baseOffsets = baseOffsets + baseOffsetsOffset;

    for (; i < limit; i += 8) {
        // Load and convert baseOffsets to integers
        __m128i baseOffsetsRaw = _mm_loadu_si128((__m128i *)(baseOffsets + i));
        __m256i baseOffsetsInt = _mm256_cvtepu8_epi32(baseOffsetsRaw);

        indexRegister = _mm256_add_epi32(indexRegister, indexIncrement);
        // Scale the baseOffsets by the cluster count
        __m256i scaledOffsets = _mm256_mullo_epi32(indexRegister, scale);

        // Calculate the final convOffsets by adding the scaled indexes and the base offsets
        __m256i convOffsets = _mm256_add_epi32(scaledOffsets, baseOffsetsInt);

        // Gather and sum values for partial sums and a magnitude
        __m256 partialSumVals = _mm256_i32gather_ps(partialSums, convOffsets, 4);
        sum = _mm256_add_ps(sum, partialSumVals);

        __m256 aMagnitudeVals = _mm256_i32gather_ps(aMagnitude, convOffsets, 4);
        vaMagnitude = _mm256_add_ps(vaMagnitude, aMagnitudeVals);
    }

    // Reduce sums
    float sumResult = reduce_add_avx2_ps(sum);
    float aMagnitudeResult = reduce_add_avx2_ps(vaMagnitude);

    // Handle the remaining elements
    for (; i < baseOffsetsLength; i++) {
        int offset = clusterCount * i + baseOffsets[i];
        sumResult += partialSums[offset];
        aMagnitudeResult += aMagnitude[offset];
    }

    return sumResult / sqrtf(aMagnitudeResult * bMagnitude);
}

#endif
