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

import io.github.jbellis.jvector.quantization.ProductQuantization;
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

  /**
   * Optionally finds the closest contiguous centroid in one bulk operation.
   * Returns {@code -1} when unsupported so callers can retain their Java loop
   * in its original compilation context. Supported implementations retain the
   * first centroid on equal distances and must validate the supplied regions.
   */
  default int tryClosestCentroid(VectorFloat<?> point, int pointOffset,
                                VectorFloat<?> centroids, int centroidCount, int dimension) {
    return -1;
  }

  /**
   * Optionally writes {@code source * multiplier} directly into a destination region.
   * Returns false without modifying the destination when unsupported. The source
   * and destination regions must not overlap.
   */
  default boolean tryScaleTo(VectorFloat<?> source, VectorFloat<?> destination,
                             int destinationOffset, float multiplier) {
    return false;
  }

  /**
   * Optionally subtracts a point from its old cluster sum, then adds it to its new
   * cluster sum in one call. All three vectors must have equal lengths and disjoint
   * storage. Returns false without modifying the sums when unsupported.
   */
  default boolean trySubtractAddInPlace(VectorFloat<?> oldSum, VectorFloat<?> newSum,
                                       VectorFloat<?> point) {
    return false;
  }

  /**
   * Optionally computes squared distances to one KMeans++ centroid into existing
   * scratch, then updates the existing componentwise minima. Points must have the
   * centroid's dimension; all arrays must be disjoint. Returns false without
   * modifying either output when unsupported. The caller retains its original
   * distance/minimum loops for other providers and dimensions.
   */
  default boolean tryUpdateKMeansDistances(VectorFloat<?>[] points, VectorFloat<?> centroid,
                                           VectorFloat<?> minima, VectorFloat<?> scratch) {
    return false;
  }

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

  /**
   * Calculates the distance between 2 vectors, which were quantized using Product Quantization, using a precomputed table of partial results
   *
   * See {@link ProductQuantization#createCodebookPartialSums(VectorSimilarityFunction)}
   *
   * @param codebookPartialSums the vector of all PQ
   * @param subspaceCount the number of PQ subspaces
   * @param vector1Ordinals Specifies which centroid vector is used for each of node1's subvectors
   * @param vector1OrdinalOffset the offset into the vector1Ordinals ByteSequence for node1 (in case vector1Ordinals is a chunk of many nodes)
   * @param node2Ordinals Specifies which centroid vector is used for each of node2's subvectors
   * @param node2OrdinalOffset the offset into the vector1Ordinals ByteSequence for node2 (in case vector1Ordinals is a chunk of many nodes)
   * @param clusterCount the number of PQ clusters per subvector in the codebook
   * @return the sum of the vectors
   */
  float assembleAndSumPQ(VectorFloat<?> codebookPartialSums, int subspaceCount, ByteSequence<?> vector1Ordinals, int vector1OrdinalOffset, ByteSequence<?> node2Ordinals, int node2OrdinalOffset, int clusterCount);

  /** Calculates the dot product of two signed int8 byte vectors. */
  float dotProduct(ByteSequence<?> a, ByteSequence<?> b);

  /** Returns the sum of squared differences of two signed int8 byte vectors. */
  float squareDistance(ByteSequence<?> a, ByteSequence<?> b);

  /** Returns the cosine similarity of two signed int8 byte vectors. */
  float cosine(ByteSequence<?> a, ByteSequence<?> b);

  int hammingDistance(long[] v1, long[] v2);

  void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int offset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums);

  default void calculatePartialSelfMagnitudes(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> partialMagnitudes) {
    int codebookBase = codebookIndex * clusterCount;
    for (int i = 0; i < clusterCount; i++) {
      partialMagnitudes.set(codebookBase + i, dotProduct(codebook, i * size, codebook, i * size, size));
    }
  }

  float max(VectorFloat<?> v);
  float min(VectorFloat<?> v);

  /**
   * Dot products of one vector against {@code count} query vectors: {@code out[i] = dot(vector, queries[i])}.
   * Implementations tile several queries per pass so the vector is loaded once per tile.
   */
  default void dotProductMulti(VectorFloat<?> vector, VectorFloat<?>[] queries, int count, float[] out) {
    for (int i = 0; i < count; i++) {
      out[i] = dotProduct(vector, queries[i]);
    }
  }


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
   * Like {@link #min(VectorFloat)}, allowing providers to tune NVQ setup separately.
   * Does not modify the input; returns {@link Float#MAX_VALUE} for an empty vector.
   * The default delegates to the shared minimum implementation.
   */
  default float nvqMin(VectorFloat<?> vector) { return min(vector); }

  /**
   * Multiplies the NVQ setup vector in place. The default delegates to
   * {@link #scale(VectorFloat, float)}.
   */
  default void nvqScale(VectorFloat<?> vector, float multiplier) { scale(vector, multiplier); }

  /**
   * Returns a new NVQ setup vector containing {@code a - b}; neither input is modified.
   * Inputs must have equal dimensions. The default delegates to {@link #sub(VectorFloat, VectorFloat)}.
   */
  default VectorFloat<?> nvqSub(VectorFloat<?> a, VectorFloat<?> b) { return sub(a, b); }

  /**
   * Returns a new vector containing the difference of the specified NVQ setup regions.
   * Offsets and length are in float elements; both regions must be in bounds.
   * Neither input is modified, including when their storage overlaps.
   * The default delegates to the shared subtraction implementation.
   */
  default VectorFloat<?> nvqSub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length) {
    return sub(a, aOffset, b, bOffset, length);
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
   * Whether NVQ scoring should use the packed cosine result instead of the array result.
   * The default is false because its packed implementation wraps the array-returning method.
   */
  default boolean supportsNvqCosine8bitPacked() { return false; }

  /**
   * Like {@link #nvqCosine8bit}, returning two floats as raw bits in a long:
   * low 32 bits contain the dot sum; high 32 bits contain the reconstructed squared magnitude.
   * Input layout is unchanged. The default packs the array-returning method's result.
   */
  default long nvqCosine8bitPacked(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate,
                                  float midpoint, float minValue, float maxValue, VectorFloat<?> centroid) {
    float[] result = nvqCosine8bit(vector, bytes, growthRate, midpoint, minValue, maxValue, centroid);
    return Integer.toUnsignedLong(Float.floatToRawIntBits(result[0]))
           | ((long)Float.floatToRawIntBits(result[1]) << 32);
  }

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

  /**
   * Optionally performs the existing coarse/fine NVQ growth-rate search in one
   * call. Returns NaN when unsupported; the caller then uses its Java search.
   * Implementations must preserve the candidate sequence and strict winner test.
   */
  default float tryNvqGrowthRate(VectorFloat<?> vector, float minValue, float maxValue, int nBits) {
    return Float.NaN;
  }

}
