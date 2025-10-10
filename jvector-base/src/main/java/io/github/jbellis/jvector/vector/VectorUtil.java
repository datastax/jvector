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
   * @param a the first vector
   * @param b the second vector
   * @return the dot product of the two vectors
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

  /**
   * Returns the vector dot product of two subvectors.
   *
   * @param a the first vector
   * @param aoffset the starting offset in the first vector
   * @param b the second vector
   * @param boffset the starting offset in the second vector
   * @param length the length of the subvectors to compute
   * @return the dot product of the two subvectors
   */
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
   * @param a the first vector
   * @param b the second vector
   * @return the cosine similarity between the two vectors
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
   * @param a the first vector
   * @param b the second vector
   * @return the squared Euclidean distance between the two vectors
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
   *
   * @param a the first vector
   * @param aoffset the starting offset in the first vector
   * @param b the second vector
   * @param boffset the starting offset in the second vector
   * @param length the length of the subvectors to compute
   * @return the squared Euclidean distance between the two subvectors
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

  /**
   * Computes the element-wise sum of multiple vectors.
   *
   * @param vectors the list of vectors to sum
   * @return a new vector containing the element-wise sum of all input vectors
   * @throws IllegalArgumentException if the input list is empty
   */
  public static VectorFloat<?> sum(List<VectorFloat<?>> vectors) {
    if (vectors.isEmpty()) {
      throw new IllegalArgumentException("Input list cannot be empty");
    }

    return impl.sum(vectors);
  }

  /**
   * Computes the sum of all elements in the vector.
   *
   * @param vector the vector to sum
   * @return the sum of all elements in the vector
   */
  public static float sum(VectorFloat<?> vector) {
    return impl.sum(vector);
  }

  /**
   * Multiplies each element of the vector by the given multiplier in place.
   *
   * @param vector the vector to scale
   * @param multiplier the scaling factor to apply to each element
   */
  public static void scale(VectorFloat<?> vector, float multiplier) {
    impl.scale(vector, multiplier);
  }

  /**
   * Adds the elements of v2 to the corresponding elements of v1 in place.
   *
   * @param v1 the vector to modify (will contain the sum)
   * @param v2 the vector to add to v1
   */
  public static void addInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
    impl.addInPlace(v1, v2);
  }

  /**
   * Adds a scalar value to each element of v1 in place.
   *
   * @param v1 the vector to modify
   * @param value the scalar value to add to each element
   */
  public static void addInPlace(VectorFloat<?> v1, float value) {
    impl.addInPlace(v1, value);
  }

  /**
   * Subtracts the elements of v2 from the corresponding elements of v1 in place.
   *
   * @param v1 the vector to modify (will contain the difference)
   * @param v2 the vector to subtract from v1
   */
  public static void subInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
    impl.subInPlace(v1, v2);
  }

  /**
   * Subtracts a scalar value from each element of the vector in place.
   *
   * @param vector the vector to modify
   * @param value the scalar value to subtract from each element
   */
  public static void subInPlace(VectorFloat<?> vector, float value) {
    impl.subInPlace(vector, value);
  }

  /**
   * Subtracts rhs from lhs element-wise and returns a new vector.
   *
   * @param lhs the vector to subtract from
   * @param rhs the vector to subtract
   * @return a new vector containing the element-wise difference
   */
  public static VectorFloat<?> sub(VectorFloat<?> lhs, VectorFloat<?> rhs) {
    return impl.sub(lhs, rhs);
  }

  /**
   * Subtracts a scalar value from each element of lhs and returns a new vector.
   *
   * @param lhs the vector to subtract from
   * @param value the scalar value to subtract from each element
   * @return a new vector containing the result
   */
  public static VectorFloat<?> sub(VectorFloat<?> lhs, float value) {
    return impl.sub(lhs, value);
  }

  /**
   * Subtracts subvectors element-wise and returns a new vector.
   *
   * @param a the first vector
   * @param aOffset the starting offset in the first vector
   * @param b the second vector
   * @param bOffset the starting offset in the second vector
   * @param length the length of the subvectors
   * @return a new vector containing the element-wise difference
   */
  public static VectorFloat<?> sub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length) {
    return impl.sub(a, aOffset, b, bOffset, length);
  }

  /**
   * Computes the element-wise minimum of two vectors and stores the result in distances1.
   *
   * @param distances1 the first vector (will be modified to contain the minimums)
   * @param distances2 the second vector
   */
  public static void minInPlace(VectorFloat<?> distances1, VectorFloat<?> distances2) {
    impl.minInPlace(distances1, distances2);
  }

  /**
   * Assembles sparse vector elements from a data array using byte offsets and computes their sum.
   *
   * @param data the vector containing all data points
   * @param dataBase the base index in the data vector
   * @param dataOffsets byte sequence containing offsets from the base index
   * @return the sum of the assembled elements
   */
  public static float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> dataOffsets) {
    return impl.assembleAndSum(data, dataBase, dataOffsets);
  }

  /**
   * Assembles sparse vector elements from a data array using a subset of byte offsets and computes their sum.
   *
   * @param data the vector containing all data points
   * @param dataBase the base index in the data vector
   * @param dataOffsets byte sequence containing offsets from the base index
   * @param dataOffsetsOffset the starting offset within the dataOffsets sequence
   * @param dataOffsetsLength the number of offsets to use
   * @return the sum of the assembled elements
   */
  public static float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> dataOffsets, int dataOffsetsOffset, int dataOffsetsLength) {
    return impl.assembleAndSum(data, dataBase, dataOffsets, dataOffsetsOffset, dataOffsetsLength);
  }

  /**
   * Computes distance between two product quantization-encoded vectors using precomputed codebook partial sums.
   *
   * @param data the vector of precomputed partial sums from the codebook
   * @param subspaceCount the number of PQ subspaces
   * @param dataOffsets1 byte sequence specifying centroid indices for the first vector
   * @param dataOffsetsOffset1 the starting offset within dataOffsets1
   * @param dataOffsets2 byte sequence specifying centroid indices for the second vector
   * @param dataOffsetsOffset2 the starting offset within dataOffsets2
   * @param clusterCount the number of clusters per subspace
   * @return the distance between the two encoded vectors
   */
  public static float assembleAndSumPQ(VectorFloat<?> data, int subspaceCount, ByteSequence<?> dataOffsets1, int dataOffsetsOffset1, ByteSequence<?> dataOffsets2, int dataOffsetsOffset2, int clusterCount) {
    return impl.assembleAndSumPQ(data, subspaceCount, dataOffsets1, dataOffsetsOffset1, dataOffsets2, dataOffsetsOffset2, clusterCount);
  }

  /**
   * Computes similarity scores for multiple PQ-encoded vectors against a query using quantized partial sums.
   *
   * @param shuffles transposed PQ-encoded vectors where all first components are contiguous, then all second components, etc.
   * @param codebookCount the number of codebooks used in PQ encoding
   * @param quantizedPartials quantized precomputed score fragments for each codebook entry
   * @param delta the quantization delta used for dequantizing scores
   * @param minDistance the minimum distance used during quantization
   * @param results output vector to store the computed similarity scores
   * @param vsf the vector similarity function to apply
   */
  public static void bulkShuffleQuantizedSimilarity(ByteSequence<?> shuffles, int codebookCount, ByteSequence<?> quantizedPartials, float delta, float minDistance, VectorFloat<?> results, VectorSimilarityFunction vsf) {
    impl.bulkShuffleQuantizedSimilarity(shuffles, codebookCount, quantizedPartials, delta, minDistance, vsf, results);
  }

  /**
   * Computes cosine similarity scores for multiple PQ-encoded vectors against a query using quantized partial sums and magnitudes.
   *
   * @param shuffles transposed PQ-encoded vectors where all first components are contiguous, then all second components, etc.
   * @param codebookCount the number of codebooks used in PQ encoding
   * @param quantizedPartialSums quantized dot product fragments between query and codebook entries
   * @param sumDelta the quantization delta used for quantizedPartialSums
   * @param minDistance the minimum distance used for quantizing sums
   * @param quantizedPartialMagnitudes quantized squared magnitudes of codebook entries
   * @param magnitudeDelta the quantization delta used for quantizedPartialMagnitudes
   * @param minMagnitude the minimum magnitude used for quantizing magnitudes
   * @param queryMagnitudeSquared the squared magnitude of the query vector
   * @param results output vector to store the computed cosine similarity scores
   */
  public static void bulkShuffleQuantizedSimilarityCosine(ByteSequence<?> shuffles, int codebookCount,
                                                          ByteSequence<?> quantizedPartialSums, float sumDelta, float minDistance,
                                                          ByteSequence<?> quantizedPartialMagnitudes, float magnitudeDelta, float minMagnitude,
                                                          float queryMagnitudeSquared, VectorFloat<?> results) {
    impl.bulkShuffleQuantizedSimilarityCosine(shuffles, codebookCount, quantizedPartialSums, sumDelta, minDistance, quantizedPartialMagnitudes, magnitudeDelta, minMagnitude, queryMagnitudeSquared, results);
  }

  /**
   * Computes the Hamming distance between two bit vectors represented as long arrays.
   *
   * @param v1 the first bit vector
   * @param v2 the second bit vector
   * @return the number of bit positions where the two vectors differ
   */
  public static int hammingDistance(long[] v1, long[] v2) {
    return impl.hammingDistance(v1, v2);
  }

  /**
   * Calculates partial similarity sums and best distances for PQ codebook clusters against a query subvector.
   *
   * @param codebook the codebook containing all centroid vectors
   * @param codebookIndex the index of the codebook to use
   * @param size the dimensionality of each centroid vector
   * @param clusterCount the number of clusters in the codebook
   * @param query the query vector
   * @param offset the starting offset in the query vector for this subspace
   * @param vsf the vector similarity function to use
   * @param partialSums output vector to store the computed partial sums
   * @param partialBestDistances output vector to store the best distances per cluster
   */
  public static void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int offset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums, VectorFloat<?> partialBestDistances) {
    impl.calculatePartialSums(codebook, codebookIndex, size, clusterCount, query, offset, vsf, partialSums, partialBestDistances);
  }

  /**
   * Calculates partial similarity sums for PQ codebook clusters against a query subvector.
   *
   * @param codebook the codebook containing all centroid vectors
   * @param codebookIndex the index of the codebook to use
   * @param size the dimensionality of each centroid vector
   * @param clusterCount the number of clusters in the codebook
   * @param query the query vector
   * @param offset the starting offset in the query vector for this subspace
   * @param vsf the vector similarity function to use
   * @param partialSums output vector to store the computed partial sums
   */
  public static void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int offset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums) {
    impl.calculatePartialSums(codebook, codebookIndex, size, clusterCount, query, offset, vsf, partialSums);
  }

  /**
   * Quantizes partial sum values into unsigned 16-bit integers for efficient storage.
   *
   * @param delta the quantization step size (divisor)
   * @param partials the partial sum values to quantize
   * @param partialBase the base values to subtract before quantization
   * @param quantizedPartials output byte sequence to store the quantized values
   */
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

  /**
   * Computes cosine similarity between a PQ-encoded vector and a query using precomputed partial sums and magnitudes.
   *
   * @param encoded the PQ-encoded vector (centroid indices)
   * @param clusterCount the number of clusters per subspace
   * @param partialSums precomputed dot products between query and codebook centroids
   * @param aMagnitude precomputed squared magnitudes of codebook centroids
   * @param bMagnitude the magnitude of the second vector
   * @return the cosine similarity score
   */
  public static float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
    return impl.pqDecodedCosineSimilarity(encoded, clusterCount, partialSums, aMagnitude, bMagnitude);
  }

  /**
   * Computes cosine similarity between a PQ-encoded vector subset and a query using precomputed partial sums and magnitudes.
   *
   * @param encoded the PQ-encoded vector (centroid indices)
   * @param encodedOffset the starting offset within the encoded sequence
   * @param encodedLength the length of the encoded sequence to use
   * @param clusterCount the number of clusters per subspace
   * @param partialSums precomputed dot products between query and codebook centroids
   * @param aMagnitude precomputed squared magnitudes of codebook centroids
   * @param bMagnitude the magnitude of the second vector
   * @return the cosine similarity score
   */
  public static float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int encodedOffset, int encodedLength, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
    return impl.pqDecodedCosineSimilarity(encoded, encodedOffset, encodedLength, clusterCount, partialSums, aMagnitude, bMagnitude);
  }

  /**
   * Computes the dot product between a vector and an 8-bit NVQ-quantized vector.
   *
   * @param vector the query vector
   * @param bytes the 8-bit quantized vector
   * @param growthRate the growth rate parameter of the logistic quantization function
   * @param midpoint the midpoint parameter of the logistic quantization function
   * @param minValue the minimum value used during quantization
   * @param maxValue the maximum value used during quantization
   * @return the dot product
   */
  public static float nvqDotProduct8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate, float midpoint, float minValue, float maxValue) {
    return impl.nvqDotProduct8bit(vector, bytes, growthRate, midpoint, minValue, maxValue);
  }

  /**
   * Computes the squared Euclidean distance between a vector and an 8-bit NVQ-quantized vector.
   *
   * @param vector the query vector
   * @param bytes the 8-bit quantized vector
   * @param growthRate the growth rate parameter of the logistic quantization function
   * @param midpoint the midpoint parameter of the logistic quantization function
   * @param minValue the minimum value used during quantization
   * @param maxValue the maximum value used during quantization
   * @return the squared Euclidean distance
   */
  public static float nvqSquareL2Distance8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate, float midpoint, float minValue, float maxValue) {
    return impl.nvqSquareL2Distance8bit(vector, bytes, growthRate, midpoint, minValue, maxValue);
  }

  /**
   * Computes the cosine similarity between a vector and an 8-bit NVQ-quantized vector.
   *
   * @param vector the query vector
   * @param bytes the 8-bit quantized vector
   * @param growthRate the growth rate parameter of the logistic quantization function
   * @param midpoint the midpoint parameter of the logistic quantization function
   * @param minValue the minimum value used during quantization
   * @param maxValue the maximum value used during quantization
   * @param centroid the global mean vector used to re-center the quantized vector
   * @return an array containing cosine similarity components
   */
  public static float[] nvqCosine8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate, float midpoint, float minValue, float maxValue, VectorFloat<?> centroid) {
    return impl.nvqCosine8bit(vector, bytes, growthRate, midpoint, minValue, maxValue, centroid);
  }

  /**
   * Shuffles the query vector in place to optimize NVQ distance computation order.
   *
   * @param vector the vector to shuffle (will be modified)
   */
  public static void nvqShuffleQueryInPlace8bit(VectorFloat<?> vector) {
    impl.nvqShuffleQueryInPlace8bit(vector);
  }

  /**
   * Quantizes a vector to 8-bit NVQ representation using a logistic quantization function.
   *
   * @param vector the vector to quantize
   * @param growthRate the growth rate parameter of the logistic quantization function
   * @param midpoint the midpoint parameter of the logistic quantization function
   * @param minValue the minimum value for quantization
   * @param maxValue the maximum value for quantization
   * @param destination the byte sequence to store the quantized result
   */
  public static void nvqQuantize8bit(VectorFloat<?> vector, float growthRate, float midpoint, float minValue, float maxValue, ByteSequence<?> destination) {
    impl.nvqQuantize8bit(vector, growthRate, midpoint, minValue, maxValue, destination);
  }

  /**
   * Computes the quantization error (squared loss) for NVQ quantization with logistic function.
   *
   * @param vector the vector to quantize
   * @param growthRate the growth rate parameter of the logistic quantization function
   * @param midpoint the midpoint parameter of the logistic quantization function
   * @param minValue the minimum value for quantization
   * @param maxValue the maximum value for quantization
   * @param nBits the number of bits per dimension
   * @return the squared error of quantization
   */
  public static float nvqLoss(VectorFloat<?> vector, float growthRate, float midpoint, float minValue, float maxValue, int nBits) {
    return impl.nvqLoss(vector, growthRate, midpoint, minValue, maxValue, nBits);
  }

  /**
   * Computes the quantization error (squared loss) for uniform quantization.
   *
   * @param vector the vector to quantize
   * @param minValue the minimum value for quantization
   * @param maxValue the maximum value for quantization
   * @param nBits the number of bits per dimension
   * @return the squared error of quantization
   */
  public static float nvqUniformLoss(VectorFloat<?> vector, float minValue, float maxValue, int nBits) {
    return impl.nvqUniformLoss(vector, minValue, maxValue, nBits);
  }
}
