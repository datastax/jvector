/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.vector;

import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.List;

/**
 * Interface for implementations of VectorUtil support.
 */
public interface VectorUtilSupport {

  /** Calculates the dot product of the given float arrays. */
  float dotProduct(VectorFloat<?> a, VectorFloat<?> b);

  /** Calculates the dot product of float arrays of differing sizes, or a subset of the data */
  float dotProduct(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length);

  /** Returns the cosine similarity between the two vectors. */
  float cosine(VectorFloat<?> v1, VectorFloat<?> v2);

  /** Calculates the cosine similarity of VectorFloats of differing sizes, or a subset of the data */
  float cosine(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length);

  /** Returns the sum of squared differences of the two vectors. */
  float squareDistance(VectorFloat<?> a, VectorFloat<?> b);

  /** Calculates the sum of squared differences of float arrays of differing sizes, or a subset of the data */
  float squareDistance(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length);

  /** returns the sum of the given vectors. */
  VectorFloat<?> sum(List<VectorFloat<?>> vectors);

  /** return the sum of the components of the vector */
  float sum(VectorFloat<?> vector);

  /** Multiply vector by multiplier, in place (vector will be modified) */
  void scale(VectorFloat<?> vector, float multiplier);

  /** Adds v2 into v1, in place (v1 will be modified) */
  void addInPlace(VectorFloat<?> v1, VectorFloat<?> v2);

  /** Adds value to each element of v1, in place (v1 will be modified) */
  void addInPlace(VectorFloat<?> v1, float value);

  /** Subtracts v2 from v1, in place (v1 will be modified) */
  void subInPlace(VectorFloat<?> v1, VectorFloat<?> v2);

  /** Subtracts value from each element of v1, in place (v1 will be modified) */
  void subInPlace(VectorFloat<?> vector, float value);

  /** @return a - b, element-wise */
  VectorFloat<?> sub(VectorFloat<?> a, VectorFloat<?> b);

  /** Subtracts value from each element of a */
  VectorFloat<?> sub(VectorFloat<?> a, float value);

  /** @return a - b, element-wise, starting at aOffset and bOffset respectively */
  VectorFloat<?> sub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length);

  /** Calculates the minimum value for every corresponding lane values in v1 and v2, in place (v1 will be modified) */
  void minInPlace(VectorFloat<?> v1, VectorFloat<?> v2);

  /**
   * Calculates the sum of sparse points in a vector.
   * <p>
   * This assumes the data vector is a 2d matrix which has been flattened into 1 dimension
   * so rather than data[n][m] it's data[n * m].  With this layout this method can quickly
   * assemble the data from this heap and sum it.
   *
   * @param data the vector of all datapoints
   * @param baseIndex the start of the data in the offset table
   *                  (scaled by the index of the lookup table)
   * @param baseOffsets bytes that represent offsets from the baseIndex
   * @return the sum of the points
   */
  float assembleAndSum(VectorFloat<?> data, int baseIndex, ByteSequence<?> baseOffsets);

  /**
   * Calculates the sum of sparse points in a vector.
   *
   * @param data the vector of all datapoints
   * @param baseIndex the start of the data in the offset table
   *                  (scaled by the index of the lookup table)
   * @param baseOffsets bytes that represent offsets from the baseIndex
   * @param baseOffsetsOffset the offset into the baseOffsets ByteSequence
   * @param baseOffsetsLength the length of the baseOffsets ByteSequence to use
   * @return the sum of the points
   */
  float assembleAndSum(VectorFloat<?> data, int baseIndex, ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength);

  int hammingDistance(long[] v1, long[] v2);

  // default implementation used here because Panama SIMD can't express necessary SIMD operations and degrades to scalar
  /**
   * Calculates the similarity score of multiple product quantization-encoded vectors against a single query vector,
   * using quantized precomputed similarity score fragments derived from codebook contents and evaluations during a search.
   * @param shuffles a sequence of shuffles to be used against partial pre-computed fragments. These are transposed PQ-encoded
   *                 vectors using the same codebooks as the partials. Due to the transposition, rather than this being
   *                 contiguous encoded vectors, the first component of all vectors is stored contiguously, then the second, and so on.
   * @param codebookCount The number of codebooks used in the PQ encoding.
   * @param quantizedPartials The quantized precomputed score fragments for each codebook entry. These are stored as a contiguous vector of all
   *                          the fragments for one codebook, followed by all the fragments for the next codebook, and so on. These have been
   *                          quantized by quantizePartialSums.
   * @param vsf      The similarity function to use.
   * @param results  The output vector to store the similarity scores. This should be pre-allocated to the same size as the number of shuffles.
   */
  default void bulkShuffleQuantizedSimilarity(ByteSequence<?> shuffles, int codebookCount, ByteSequence<?> quantizedPartials, float delta, float minDistance, VectorSimilarityFunction vsf, VectorFloat<?> results) {
    for (int i = 0; i < codebookCount; i++) {
      for (int j = 0; j < results.length(); j++) {
        var shuffle = Byte.toUnsignedInt(shuffles.get(i * results.length() + j)) * 2;
        var lowByte = quantizedPartials.get(i * 512 + shuffle);
        var highByte = quantizedPartials.get(i * 512 + shuffle + 1);
        var val = ((Byte.toUnsignedInt(highByte) << 8) | Byte.toUnsignedInt(lowByte));
        results.set(j, results.get(j) + val);
      }
    }

    for (int i = 0; i < results.length(); i++) {
      switch (vsf) {
        case EUCLIDEAN:
          results.set(i, 1 / (1 + (delta * results.get(i)) + minDistance));
          break;
        case DOT_PRODUCT:
            results.set(i, (1 + (delta * results.get(i)) + minDistance) / 2);
            break;
        default:
          throw new UnsupportedOperationException("Unsupported similarity function " + vsf);
      }
    }
  }

  // default implementation used here because Panama SIMD can't express necessary SIMD operations and degrades to scalar
  /**
   * Calculates the similarity score of multiple product quantization-encoded vectors against a single query vector,
   * using quantized precomputed similarity score fragments derived from codebook contents and evaluations during a search.
   * @param shuffles a sequence of shuffles to be used against partial pre-computed fragments. These are transposed PQ-encoded
   *                 vectors using the same codebooks as the partials. Due to the transposition, rather than this being
   *                 contiguous encoded vectors, the first component of all vectors is stored contiguously, then the second, and so on.
   * @param codebookCount The number of codebooks used in the PQ encoding.
   * @param quantizedPartialSums The quantized precomputed dot product fragments between query vector and codebook entries.
   *                             These are stored as a contiguous vector of all the fragments for one codebook, followed by
   *                             all the fragments for the next codebook, and so on. These have been quantized by quantizePartials.
   * @param sumDelta The delta used to quantize quantizedPartialSums.
   * @param minDistance The minimum distance used to quantize quantizedPartialSums.
   * @param quantizedPartialSquaredMagnitudes The quantized precomputed squared magnitudes of each codebook entry. Quantized through the
   *                                          same process as quantizedPartialSums.
   * @param magnitudeDelta The delta used to quantize quantizedPartialSquaredMagnitudes.
   * @param minMagnitude The minimum magnitude used to quantize quantizedPartialSquaredMagnitudes.
   * @param queryMagnitudeSquared The squared magnitude of the query vector.
   * @param results  The output vector to store the similarity distances. This should be pre-allocated to the same size as the number of shuffles.
   */
  default void bulkShuffleQuantizedSimilarityCosine(ByteSequence<?> shuffles, int codebookCount,
                                                    ByteSequence<?> quantizedPartialSums, float sumDelta, float minDistance,
                                                    ByteSequence<?> quantizedPartialSquaredMagnitudes, float magnitudeDelta, float minMagnitude,
                                                    float queryMagnitudeSquared, VectorFloat<?> results) {
    float[] sums = new float[results.length()];
    float[] magnitudes = new float[results.length()];
    for (int i = 0; i < codebookCount; i++) {
      for (int j = 0; j < results.length(); j++) {
        var shuffle = Byte.toUnsignedInt(shuffles.get(i * results.length() + j)) * 2;
        var lowByte = quantizedPartialSums.get(i * 512 + shuffle);
        var highByte = quantizedPartialSums.get(i * 512 + shuffle + 1);
        var val = ((Byte.toUnsignedInt(highByte) << 8) | Byte.toUnsignedInt(lowByte));
        sums[j] += val;
        lowByte = quantizedPartialSquaredMagnitudes.get(i * 512 + shuffle);
        highByte = quantizedPartialSquaredMagnitudes.get(i * 512 + shuffle + 1);
        val = ((Byte.toUnsignedInt(highByte) << 8) | Byte.toUnsignedInt(lowByte));
        magnitudes[j] += val;
      }
    }

    for (int i = 0; i < results.length(); i++) {
        float unquantizedSum = sumDelta * sums[i] + minDistance;
        float unquantizedMagnitude = magnitudeDelta * magnitudes[i] + minMagnitude;
        double divisor = Math.sqrt(unquantizedMagnitude * queryMagnitudeSquared);
        results.set(i, (1 + (float) (unquantizedSum / divisor)) / 2);
    }
  }

  void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int offset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums);

  void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int offset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums, VectorFloat<?> partialMins);

  /**
   * Quantizes values in partials (of length N = M * K) into unsigned little-endian 16-bit integers stored in quantizedPartials in the same order.
   * partialBases is of length M. For each indexed chunk of K values in partials, each value in the chunk is quantized by subtracting the value
   * in partialBases as the same index and dividing by delta. If the value is greater than 65535, 65535 will be used.
   *
   * The caller is responsible for ensuring than no value in partialSums is larger than its corresponding partialBase.
   *
   * @param delta the divisor to use for quantization
   * @param partials the values to quantize
   * @param partialBases the base values to subtract from the partials
   * @param quantizedPartials the output sequence to store the quantized values
   */
  void quantizePartials(float delta, VectorFloat<?> partials, VectorFloat<?> partialBases, ByteSequence<?> quantizedPartials);

  float max(VectorFloat<?> v);
  float min(VectorFloat<?> v);

  default float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude)
  {
    return pqDecodedCosineSimilarity(encoded, 0, encoded.length(), clusterCount, partialSums, aMagnitude, bMagnitude);
  }

  default float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int encodedOffset, int encodedLength, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude)
  {
    float sum = 0.0f;
    float aMag = 0.0f;

    for (int m = 0; m < encodedLength; ++m) {
      int centroidIndex = Byte.toUnsignedInt(encoded.get(m + encodedOffset));
      var index = m * clusterCount + centroidIndex;
      sum += partialSums.get(index);
      aMag += aMagnitude.get(index);
    }

    return (float) (sum / Math.sqrt(aMag * bMagnitude));
  }
  /**
   * Computes the dot product between a vector and a 8-bit quantized vector (described by its parameters).
   * We assume that the number of dimensions of the vector and the quantized vector match.
   * @param vector The query vector
   * @param bytes The byte sequence where the quantized vector is stored.
   * @param growthRate The growth rate of the logistic function
   * @param midpoint the midpoint of the logistic function
   * @param minValue The minimum value of the subvector
   * @param maxValue The maximum value of the subvector
   * @return the dot product
   */
  float nvqDotProduct8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate, float midpoint, float minValue, float maxValue);

  /**
   * Computes the squared Euclidean distance between a vector and a 8-bit quantized vector (described by its parameters).
   * We assume that the number of dimensions of the vector and the quantized vector match.
   * @param vector The query vector
   * @param bytes The byte sequence where the quantized vector is stored.
   * @param growthRate The growth rate of the logistic function
   * @param midpoint the midpoint of the logistic function
   * @param minValue The minimum value of the subvector
   * @param maxValue The maximum value of the subvector
   * @return the squared Euclidean distance
   */
  float nvqSquareL2Distance8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate, float midpoint, float minValue, float maxValue);

  /**
   * Computes the cosine similarity between a vector and a 8-bit quantized vector (described by its parameters).
   * We assume that the number of dimensions of the vector and the quantized vector match.
   * @param vector The query vector
   * @param bytes The byte sequence where the quantized vector is stored.
   * @param growthRate The growth rate of the logistic function
   * @param midpoint the midpoint of the logistic function
   * @param minValue The minimum value of the subvector
   * @param maxValue The maximum value of the subvector
   * @param centroid the global mean vector used to re-center the quantized subvectors.
   * @return the cosine similarity
   */
  float[] nvqCosine8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate, float midpoint, float minValue, float maxValue, VectorFloat<?> centroid);

  /**
   * When computing distances, the unpacking of am NVQ quantized vector is faster if we do not do it in sequential order.
   * Hence, we shuffle the query so that it matches this order
   * See <a href="https://www.vldb.org/pvldb/vol16/p2132-afroozeh.pdf">this reference</a>
   * @param vector the vector to be shuffled
   */
  void nvqShuffleQueryInPlace8bit(VectorFloat<?> vector);

  /**
   * Quantize a subvector as an 8-bit quantized subvector.
   * @param vector The vector to quantized
   * @param growthRate The growth rate of the logistic function
   * @param midpoint the midpoint of the logistic function
   * @param minValue The minimum value of the subvector
   * @param maxValue The maximum value of the subvector
   * @param destination The vector where the reconstructed values are stored
   */
  void nvqQuantize8bit(VectorFloat<?> vector, float growthRate, float midpoint, float minValue, float maxValue, ByteSequence<?> destination);

  /**
   * Compute the squared error of quantizing the vector with NVQ.
   * @param vector The vector to quantized
   * @param growthRate The growth rate of the logistic function
   * @param midpoint the midpoint of the logistic function
   * @param minValue The minimum value of the subvector
   * @param maxValue The maximum value of the subvector
   * @param nBits the number of bits per dimension
   */
  float nvqLoss(VectorFloat<?> vector, float growthRate, float midpoint, float minValue, float maxValue, int nBits);

  /**
   * Compute the squared error of quantizing the vector with a uniform quantizer.
   * @param vector The vector to quantized
   * @param minValue The minimum value of the subvector
   * @param maxValue The maximum value of the subvector
   * @param nBits the number of bits per dimension
   */
  float nvqUniformLoss(VectorFloat<?> vector, float minValue, float maxValue, int nBits);

}
