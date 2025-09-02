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

/** Utilities for computations with numeric arrays */
public final class VectorUtil {

  private static final VectorUtilSupport impl =
      VectorizationProvider.getInstance().getVectorUtilSupport();

  private VectorUtil() {}

  /**
   * Returns the vector dot product of the two vectors.
   *
   * @throws IllegalArgumentException if the vectors' dimensions differ.
   */
  public static float dotProduct(VectorFloat<?> a, VectorFloat<?> b) {
    if (a.length() != b.length()) {
      throw new IllegalArgumentException("vector dimensions differ: " + a.length() + "!=" + b.length());
    }
    float r = impl.dotProduct(a, b);
    assert Float.isFinite(r) : String.format("dotProduct(%s, %s) = %s", a, b, r);
    return r;
  }

  public static float dotProduct(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
    //This check impacts FLOPS
    /*if ( length > Math.min(a.length - aoffset, b.length - boffset) ) {
      throw new IllegalArgumentException("length must be less than the vectors remaining space at the given offsets: a(" +
              (a.length - aoffset) + "), b(" + (b.length - boffset) + "), length(" + length + ")");
    }*/
    float r = impl.dotProduct(a, aoffset, b, boffset, length);
    assert Float.isFinite(r) : String.format("dotProduct(%s, %s) = %s", a, b, r);
    return r;
  }

  /**
   * Returns the cosine similarity between the two vectors.
   *
   * @throws IllegalArgumentException if the vectors' dimensions differ.
   */
  public static float cosine(VectorFloat<?> a, VectorFloat<?> b) {
    if (a.length() != b.length()) {
      throw new IllegalArgumentException("vector dimensions differ: " + a.length() + "!=" + b.length());
    }
    float r = impl.cosine(a, b);
    assert Float.isFinite(r) : String.format("cosine(%s, %s) = %s", a, b, r);
    return r;
  }

  /**
   * Returns the sum of squared differences of the two vectors.
   *
   * @throws IllegalArgumentException if the vectors' dimensions differ.
   */
  public static float squareL2Distance(VectorFloat<?> a, VectorFloat<?> b) {
    if (a.length() != b.length()) {
      throw new IllegalArgumentException("vector dimensions differ: " + a.length() + "!=" + b.length());
    }
    float r = impl.squareDistance(a, b);
    assert Float.isFinite(r) : String.format("squareDistance(%s, %s) = %s", a, b, r);
    return r;
  }

  /**
   * Returns the sum of squared differences of the two vectors, or subvectors, of the given length.
   */
  public static float squareL2Distance(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
    float r = impl.squareDistance(a, aoffset, b, boffset, length);
    assert Float.isFinite(r);
    return r;
  }

  /**
   * Modifies the argument to be unit length, dividing by its l2-norm. IllegalArgumentException is
   * thrown for zero vectors.
   *
   * @param v the vector to normalize
   */
  public static void l2normalize(VectorFloat<?> v) {
    double squareSum = dotProduct(v, v);
    if (squareSum == 0) {
      throw new IllegalArgumentException("Cannot normalize a zero-length vector");
    }
    double length = Math.sqrt(squareSum);
    scale(v, (float) (1.0 / length));
  }

  public static VectorFloat<?> sum(List<VectorFloat<?>> vectors) {
    if (vectors.isEmpty()) {
      throw new IllegalArgumentException("Input list cannot be empty");
    }

    return impl.sum(vectors);
  }

  public static float sum(VectorFloat<?> vector) {
    return impl.sum(vector);
  }

  public static void scale(VectorFloat<?> vector, float multiplier) {
    impl.scale(vector, multiplier);
  }

  public static void addInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
    impl.addInPlace(v1, v2);
  }

  public static void addInPlace(VectorFloat<?> v1, float value) {
    impl.addInPlace(v1, value);
  }

  public static void subInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
    impl.subInPlace(v1, v2);
  }

  public static void subInPlace(VectorFloat<?> vector, float value) {
    impl.subInPlace(vector, value);
  }

  public static VectorFloat<?> sub(VectorFloat<?> lhs, VectorFloat<?> rhs) {
    return impl.sub(lhs, rhs);
  }

  public static VectorFloat<?> sub(VectorFloat<?> lhs, float value) {
    return impl.sub(lhs, value);
  }

  public static VectorFloat<?> sub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length) {
    return impl.sub(a, aOffset, b, bOffset, length);
  }

  public static void minInPlace(VectorFloat<?> distances1, VectorFloat<?> distances2) {
    impl.minInPlace(distances1, distances2);
  }

  public static float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> dataOffsets) {
    return impl.assembleAndSum(data, dataBase, dataOffsets);
  }

  public static float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> dataOffsets, int dataOffsetsOffset, int dataOffsetsLength) {
    return impl.assembleAndSum(data, dataBase, dataOffsets, dataOffsetsOffset, dataOffsetsLength);
  }

  public static float assembleAndSumPQ(VectorFloat<?> data, int subspaceCount, ByteSequence<?> dataOffsets1, int dataOffsetsOffset1, ByteSequence<?> dataOffsets2, int dataOffsetsOffset2, int clusterCount) {
    return impl.assembleAndSumPQ(data, subspaceCount, dataOffsets1, dataOffsetsOffset1, dataOffsets2, dataOffsetsOffset2, clusterCount);
  }

  public static void bulkShuffleQuantizedSimilarity(ByteSequence<?> shuffles, int codebookCount, ByteSequence<?> quantizedPartials, float delta, float minDistance, VectorFloat<?> results, VectorSimilarityFunction vsf) {
    impl.bulkShuffleQuantizedSimilarity(shuffles, codebookCount, quantizedPartials, delta, minDistance, vsf, results);
  }

  public static void bulkShuffleQuantizedSimilarityCosine(ByteSequence<?> shuffles, int codebookCount,
                                                          ByteSequence<?> quantizedPartialSums, float sumDelta, float minDistance,
                                                          ByteSequence<?> quantizedPartialMagnitudes, float magnitudeDelta, float minMagnitude,
                                                          float queryMagnitudeSquared, VectorFloat<?> results) {
    impl.bulkShuffleQuantizedSimilarityCosine(shuffles, codebookCount, quantizedPartialSums, sumDelta, minDistance, quantizedPartialMagnitudes, magnitudeDelta, minMagnitude, queryMagnitudeSquared, results);
  }

  public static int hammingDistance(long[] v1, long[] v2) {
    return impl.hammingDistance(v1, v2);
  }

  public static void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int offset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums, VectorFloat<?> partialBestDistances) {
    impl.calculatePartialSums(codebook, codebookIndex, size, clusterCount, query, offset, vsf, partialSums, partialBestDistances);
  }

  public static void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int offset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums) {
    impl.calculatePartialSums(codebook, codebookIndex, size, clusterCount, query, offset, vsf, partialSums);
  }

  public static void quantizePartials(float delta, VectorFloat<?> partials, VectorFloat<?> partialBase, ByteSequence<?> quantizedPartials) {
    impl.quantizePartials(delta, partials, partialBase, quantizedPartials);
  }

  /**
   * Calculates the maximum value in the vector.
   * @param v vector
   * @return the maximum value, or -Float.MAX_VALUE if the vector is empty
   */
  public static float max(VectorFloat<?> v) {
    return impl.max(v);
  }

  /**
   * Calculates the minimum value in the vector.
   * @param v vector
   * @return the minimum value, or Float.MAX_VALUE if the vector is empty
   */
  public static float min(VectorFloat<?> v) {
    return impl.min(v);
  }

  public static float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
    return impl.pqDecodedCosineSimilarity(encoded, clusterCount, partialSums, aMagnitude, bMagnitude);
  }

  public static float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int encodedOffset, int encodedLength, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
    return impl.pqDecodedCosineSimilarity(encoded, encodedOffset, encodedLength, clusterCount, partialSums, aMagnitude, bMagnitude);
  }

  public static float nvqDotProduct8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate, float midpoint, float minValue, float maxValue) {
    return impl.nvqDotProduct8bit(vector, bytes, growthRate, midpoint, minValue, maxValue);
  }

  public static float nvqSquareL2Distance8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate, float midpoint, float minValue, float maxValue) {
    return impl.nvqSquareL2Distance8bit(vector, bytes, growthRate, midpoint, minValue, maxValue);
  }

  public static float[] nvqCosine8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate, float midpoint, float minValue, float maxValue, VectorFloat<?> centroid) {
    return impl.nvqCosine8bit(vector, bytes, growthRate, midpoint, minValue, maxValue, centroid);
  }

  public static void nvqShuffleQueryInPlace8bit(VectorFloat<?> vector) {
    impl.nvqShuffleQueryInPlace8bit(vector);
  }

  public static void nvqQuantize8bit(VectorFloat<?> vector, float growthRate, float midpoint, float minValue, float maxValue, ByteSequence<?> destination) {
    impl.nvqQuantize8bit(vector, growthRate, midpoint, minValue, maxValue, destination);
  }

  public static float nvqLoss(VectorFloat<?> vector, float growthRate, float midpoint, float minValue, float maxValue, int nBits) {
    return impl.nvqLoss(vector, growthRate, midpoint, minValue, maxValue, nBits);
  }

  public static float nvqUniformLoss(VectorFloat<?> vector, float minValue, float maxValue, int nBits) {
    return impl.nvqUniformLoss(vector, minValue, maxValue, nBits);
  }
}
