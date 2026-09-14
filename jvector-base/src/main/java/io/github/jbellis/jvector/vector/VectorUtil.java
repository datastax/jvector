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
   * Tries a bulk closest-centroid operation by squared L2 distance. Returns -1
   * when unsupported, allowing callers to retain their existing Java scan.
   * Equal distances retain the first centroid. No distance array is allocated.
   * With assertions enabled, a second distance scan checks that every distance is finite.
   */
  public static int tryClosestCentroid(VectorFloat<?> point, int pointOffset,
                                    VectorFloat<?> centroids, int centroidCount, int dimension) {
    int nearest = impl.tryClosestCentroid(point, pointOffset, centroids, centroidCount, dimension);
    // Preserve the former loop's finite-distance checks; this repeats the scan only with -ea.
    assert nearest < 0 || centroidDistancesAreFinite(point, pointOffset, centroids, centroidCount, dimension);
    return nearest;
  }

  /**
   * Optionally writes {@code source * multiplier} into the destination at the given offset.
   * The source and destination regions must not overlap. Returns {@code false}
   * without modifying the destination when unsupported.
   */
  public static boolean tryScaleTo(VectorFloat<?> source, VectorFloat<?> destination,
                                   int destinationOffset, float multiplier) {
    return impl.tryScaleTo(source, destination, destinationOffset, multiplier);
  }

  /**
   * Optionally subtracts {@code point} from {@code oldSum} and adds it to {@code newSum}.
   * All three vectors must have equal lengths and disjoint storage. Returns
   * {@code false} without modifying the sums when unsupported.
   */
  public static boolean trySubtractAddInPlace(VectorFloat<?> oldSum, VectorFloat<?> newSum,
                                             VectorFloat<?> point) {
    return impl.trySubtractAddInPlace(oldSum, newSum, point);
  }

  /**
   * Optionally writes squared distances from the points to the centroid into scratch,
   * then updates the componentwise minima. Points must have the centroid's dimension;
   * input and output storage must be disjoint. Returns {@code false} without modifying
   * either output when unsupported, so callers can use their existing loops.
   */
  public static boolean tryUpdateKMeansDistances(VectorFloat<?>[] points, VectorFloat<?> centroid,
                                                  VectorFloat<?> minima, VectorFloat<?> scratch) {
    return impl.tryUpdateKMeansDistances(points, centroid, minima, scratch);
  }

  private static boolean centroidDistancesAreFinite(VectorFloat<?> point, int pointOffset,
                                                    VectorFloat<?> centroids, int centroidCount, int dimension) {
    for (int i = 0; i < centroidCount; i++) {
      if (!Float.isFinite(impl.squareDistance(point, pointOffset, centroids, i * dimension, dimension))) {
        return false;
      }
    }
    return true;
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

  /** @see VectorUtilSupport#dotProductMulti */
  public static void dotProductMulti(VectorFloat<?> vector, VectorFloat<?>[] queries, int count, float[] out) {
    impl.dotProductMulti(vector, queries, count, out);
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

  public static int hammingDistance(long[] v1, long[] v2) {
    return impl.hammingDistance(v1, v2);
  }

  public static void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int offset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums) {
    impl.calculatePartialSums(codebook, codebookIndex, size, clusterCount, query, offset, vsf, partialSums);
  }

  public static void calculatePartialSelfMagnitudes(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> partialMagnitudes) {
    impl.calculatePartialSelfMagnitudes(codebook, codebookIndex, size, clusterCount, partialMagnitudes);
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

  /**
   * Like {@link #min(VectorFloat)}, with a provider-specific implementation for NVQ setup.
   * Does not modify the input; returns {@link Float#MAX_VALUE} for an empty vector.
   */
  public static float nvqMin(VectorFloat<?> vector) { return impl.nvqMin(vector); }

  /** Like {@link #scale(VectorFloat, float)}, multiplying the NVQ setup vector in place. */
  public static void nvqScale(VectorFloat<?> vector, float multiplier) { impl.nvqScale(vector, multiplier); }

  /**
   * Returns a new vector containing {@code a - b} for NVQ setup. Inputs must have
   * equal dimensions; neither input is modified.
   */
  public static VectorFloat<?> nvqSub(VectorFloat<?> a, VectorFloat<?> b) { return impl.nvqSub(a, b); }

  /**
   * Returns a new vector containing the difference of the specified NVQ setup regions.
   * Offsets and length are in float elements; both regions must be in bounds.
   * Neither input is modified, including when their storage overlaps.
   */
  public static VectorFloat<?> nvqSub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length) {
    return impl.nvqSub(a, aOffset, b, bOffset, length);
  }

  public static float nvqDotProduct8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate, float midpoint, float minValue, float maxValue) {
    return impl.nvqDotProduct8bit(vector, bytes, growthRate, midpoint, minValue, maxValue);
  }

  public static float nvqSquareL2Distance8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate, float midpoint, float minValue, float maxValue) {
    return impl.nvqSquareL2Distance8bit(vector, bytes, growthRate, midpoint, minValue, maxValue);
  }

  /** Whether NVQ scoring should use the packed cosine result instead of the array result. */
  public static boolean supportsNvqCosine8bitPacked() {
    return impl.supportsNvqCosine8bitPacked();
  }

  /**
   * Like {@link #nvqCosine8bit}, returning two floats as raw bits in a long:
   * low 32 bits contain the dot sum; high 32 bits contain the reconstructed squared magnitude.
   * The vector, centroid and quantized bytes follow the same layout as the array-returning method.
   */
  public static long nvqCosine8bitPacked(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate,
                                        float midpoint, float minValue, float maxValue, VectorFloat<?> centroid) {
    return impl.nvqCosine8bitPacked(vector, bytes, growthRate, midpoint, minValue, maxValue, centroid);
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

  /**
   * Optionally performs NVQ's coarse/fine growth-rate search without modifying the vector.
   * Returns {@link Float#NaN} when unsupported, so callers can use their Java search.
   */
  public static float tryNvqGrowthRate(VectorFloat<?> vector, float minValue, float maxValue, int nBits) {
    return impl.tryNvqGrowthRate(vector, minValue, maxValue, nBits);
  }

  public static float nvqUniformLoss(VectorFloat<?> vector, float minValue, float maxValue, int nBits) {
    return impl.nvqUniformLoss(vector, minValue, maxValue, nBits);
  }
}
