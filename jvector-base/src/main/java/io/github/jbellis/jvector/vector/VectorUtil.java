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
   * @return the dot product
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
   * Returns the vector dot product of the two vectors, or subvectors, of the given length.
   *
   * @param a the first vector
   * @param aoffset the starting offset in the first vector
   * @param b the second vector
   * @param boffset the starting offset in the second vector
   * @param length the number of elements to compute the dot product over
   * @return the dot product
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
   * @return the cosine similarity
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
   * @return the sum of squared differences
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
   * @param length the number of elements to compare
   * @return the sum of squared differences
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
   * Returns the sum of the given vectors.
   *
   * @param vectors the list of vectors to sum
   * @return a new vector containing the sum of all input vectors
   * @throws IllegalArgumentException if the input list is empty
   */
  public static VectorFloat<?> sum(List<VectorFloat<?>> vectors) {
    if (vectors.isEmpty()) {
      throw new IllegalArgumentException("Input list cannot be empty");
    }

    return impl.sum(vectors);
  }

  /**
   * Returns the sum of all components in the vector.
   *
   * @param vector the vector to sum
   * @return the sum of all elements in the vector
   */
  public static float sum(VectorFloat<?> vector) {
    return impl.sum(vector);
  }

  /**
   * Multiplies each element of the vector by the given multiplier, modifying the vector in place.
   *
   * @param vector the vector to scale (modified in place)
   * @param multiplier the scalar value to multiply each element by
   */
  public static void scale(VectorFloat<?> vector, float multiplier) {
    impl.scale(vector, multiplier);
  }

  /**
   * Adds v2 to v1 element-wise, modifying v1 in place.
   *
   * @param v1 the vector to add to (modified in place)
   * @param v2 the vector to add
   */
  public static void addInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
    impl.addInPlace(v1, v2);
  }

  /**
   * Adds a scalar value to each element of v1, modifying v1 in place.
   *
   * @param v1 the vector to add to (modified in place)
   * @param value the scalar value to add to each element
   */
  public static void addInPlace(VectorFloat<?> v1, float value) {
    impl.addInPlace(v1, value);
  }

  /**
   * Subtracts v2 from v1 element-wise, modifying v1 in place.
   *
   * @param v1 the vector to subtract from (modified in place)
   * @param v2 the vector to subtract
   */
  public static void subInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
    impl.subInPlace(v1, v2);
  }

  /**
   * Subtracts a scalar value from each element of the vector, modifying the vector in place.
   *
   * @param vector the vector to subtract from (modified in place)
   * @param value the scalar value to subtract from each element
   */
  public static void subInPlace(VectorFloat<?> vector, float value) {
    impl.subInPlace(vector, value);
  }

  /**
   * Returns a new vector containing the element-wise difference of lhs and rhs.
   *
   * @param lhs the left-hand side vector
   * @param rhs the right-hand side vector
   * @return a new vector containing lhs - rhs
   */
  public static VectorFloat<?> sub(VectorFloat<?> lhs, VectorFloat<?> rhs) {
    return impl.sub(lhs, rhs);
  }

  /**
   * Returns a new vector containing the result of subtracting a scalar value from each element of lhs.
   *
   * @param lhs the left-hand side vector
   * @param value the scalar value to subtract from each element
   * @return a new vector containing lhs - value
   */
  public static VectorFloat<?> sub(VectorFloat<?> lhs, float value) {
    return impl.sub(lhs, value);
  }

  /**
   * Returns a new vector containing the element-wise difference of two subvectors.
   *
   * @param a the first vector
   * @param aOffset the starting offset in the first vector
   * @param b the second vector
   * @param bOffset the starting offset in the second vector
   * @param length the number of elements to subtract
   * @return a new vector containing a[aOffset:aOffset+length] - b[bOffset:bOffset+length]
   */
  public static VectorFloat<?> sub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length) {
    return impl.sub(a, aOffset, b, bOffset, length);
  }

  /**
   * Computes the element-wise minimum of distances1 and distances2, modifying distances1 in place.
   *
   * @param distances1 the first vector (modified in place to contain the minimum values)
   * @param distances2 the second vector
   */
  public static void minInPlace(VectorFloat<?> distances1, VectorFloat<?> distances2) {
    impl.minInPlace(distances1, distances2);
  }

  /**
   * Assembles values from data using indices in dataOffsets and returns their sum.
   *
   * @param data the vector containing all data points
   * @param dataBase the base index in the data vector
   * @param dataOffsets byte sequence containing offsets from the base index
   * @return the sum of the assembled values
   */
  public static float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> dataOffsets) {
    return impl.assembleAndSum(data, dataBase, dataOffsets);
  }

  /**
   * Assembles values from data using a subset of indices in dataOffsets and returns their sum.
   *
   * @param data the vector containing all data points
   * @param dataBase the base index in the data vector
   * @param dataOffsets byte sequence containing offsets from the base index
   * @param dataOffsetsOffset the starting offset in the dataOffsets sequence
   * @param dataOffsetsLength the number of offsets to use
   * @return the sum of the assembled values
   */
  public static float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> dataOffsets, int dataOffsetsOffset, int dataOffsetsLength) {
    return impl.assembleAndSum(data, dataBase, dataOffsets, dataOffsetsOffset, dataOffsetsLength);
  }

  /**
   * Computes the distance between two product-quantized vectors using precomputed partial results.
   *
   * @param data the vector of product quantization partial sums
   * @param subspaceCount the number of PQ subspaces
   * @param dataOffsets1 the ordinals specifying centroids for the first vector
   * @param dataOffsetsOffset1 the starting offset in dataOffsets1
   * @param dataOffsets2 the ordinals specifying centroids for the second vector
   * @param dataOffsetsOffset2 the starting offset in dataOffsets2
   * @param clusterCount the number of clusters per subspace
   * @return the sum of the partial results
   */
  public static float assembleAndSumPQ(VectorFloat<?> data, int subspaceCount, ByteSequence<?> dataOffsets1, int dataOffsetsOffset1, ByteSequence<?> dataOffsets2, int dataOffsetsOffset2, int clusterCount) {
    return impl.assembleAndSumPQ(data, subspaceCount, dataOffsets1, dataOffsetsOffset1, dataOffsets2, dataOffsetsOffset2, clusterCount);
  }

  /**
   * Computes similarity scores for multiple product-quantized vectors using quantized partial results.
   *
   * @param shuffles the transposed PQ-encoded vectors
   * @param codebookCount the number of codebooks used in PQ encoding
   * @param quantizedPartials the quantized precomputed score fragments
   * @param delta the quantization delta value
   * @param minDistance the minimum distance used in quantization
   * @param results the output vector to store similarity scores (modified in place)
   * @param vsf the vector similarity function to use
   */
  public static void bulkShuffleQuantizedSimilarity(ByteSequence<?> shuffles, int codebookCount, ByteSequence<?> quantizedPartials, float delta, float minDistance, VectorFloat<?> results, VectorSimilarityFunction vsf) {
    impl.bulkShuffleQuantizedSimilarity(shuffles, codebookCount, quantizedPartials, delta, minDistance, vsf, results);
  }

  /**
   * Computes cosine similarity scores for multiple product-quantized vectors using quantized partial results.
   *
   * @param shuffles the transposed PQ-encoded vectors
   * @param codebookCount the number of codebooks used in PQ encoding
   * @param quantizedPartialSums the quantized precomputed dot product fragments
   * @param sumDelta the delta used to quantize the partial sums
   * @param minDistance the minimum distance used in quantization
   * @param quantizedPartialMagnitudes the quantized precomputed squared magnitudes
   * @param magnitudeDelta the delta used to quantize the magnitudes
   * @param minMagnitude the minimum magnitude used in quantization
   * @param queryMagnitudeSquared the squared magnitude of the query vector
   * @param results the output vector to store similarity scores (modified in place)
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
   * @return the Hamming distance (number of differing bits)
   */
  public static int hammingDistance(long[] v1, long[] v2) {
    return impl.hammingDistance(v1, v2);
  }

  /**
   * Calculates partial sums for product quantization, storing results in partialSums and partialBestDistances.
   *
   * @param codebook the PQ codebook vectors
   * @param codebookIndex the starting index in the codebook
   * @param size the size of each codebook entry
   * @param clusterCount the number of clusters per subspace
   * @param query the query vector
   * @param offset the offset in the query vector
   * @param vsf the vector similarity function
   * @param partialSums the output vector for partial sums (modified in place)
   * @param partialBestDistances the output vector for best distances (modified in place)
   */
  public static void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int offset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums, VectorFloat<?> partialBestDistances) {
    impl.calculatePartialSums(codebook, codebookIndex, size, clusterCount, query, offset, vsf, partialSums, partialBestDistances);
  }

  /**
   * Calculates partial sums for product quantization, storing results in partialSums.
   *
   * @param codebook the PQ codebook vectors
   * @param codebookIndex the starting index in the codebook
   * @param size the size of each codebook entry
   * @param clusterCount the number of clusters per subspace
   * @param query the query vector
   * @param offset the offset in the query vector
   * @param vsf the vector similarity function
   * @param partialSums the output vector for partial sums (modified in place)
   */
  public static void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int offset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums) {
    impl.calculatePartialSums(codebook, codebookIndex, size, clusterCount, query, offset, vsf, partialSums);
  }

  /**
   * Quantizes partial sum values into unsigned 16-bit integers stored as bytes.
   *
   * @param delta the quantization delta (divisor)
   * @param partials the values to quantize
   * @param partialBase the base values to subtract before quantization
   * @param quantizedPartials the output byte sequence for quantized values (modified in place)
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
   * Computes the cosine similarity between a query and a product-quantized vector.
   *
   * @param encoded the PQ-encoded vector
   * @param clusterCount the number of clusters per subspace
   * @param partialSums the precomputed partial dot products with codebook centroids
   * @param aMagnitude the precomputed partial magnitudes of codebook centroids
   * @param bMagnitude the magnitude of the query vector
   * @return the cosine similarity
   */
  public static float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
    return impl.pqDecodedCosineSimilarity(encoded, clusterCount, partialSums, aMagnitude, bMagnitude);
  }

  /**
   * Computes the cosine similarity between a query and a subset of a product-quantized vector.
   *
   * @param encoded the PQ-encoded vector
   * @param encodedOffset the starting offset in the encoded vector
   * @param encodedLength the number of encoded values to use
   * @param clusterCount the number of clusters per subspace
   * @param partialSums the precomputed partial dot products with codebook centroids
   * @param aMagnitude the precomputed partial magnitudes of codebook centroids
   * @param bMagnitude the magnitude of the query vector
   * @return the cosine similarity
   */
  public static float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int encodedOffset, int encodedLength, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
    return impl.pqDecodedCosineSimilarity(encoded, encodedOffset, encodedLength, clusterCount, partialSums, aMagnitude, bMagnitude);
  }

  /**
   * Computes the dot product between a vector and an 8-bit NVQ quantized vector.
   *
   * @param vector the query vector
   * @param bytes the 8-bit quantized vector
   * @param growthRate the growth rate parameter of the logistic quantization function
   * @param midpoint the midpoint parameter of the logistic quantization function
   * @param minValue the minimum value of the quantized subvector
   * @param maxValue the maximum value of the quantized subvector
   * @return the dot product
   */
  public static float nvqDotProduct8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate, float midpoint, float minValue, float maxValue) {
    return impl.nvqDotProduct8bit(vector, bytes, growthRate, midpoint, minValue, maxValue);
  }

  /**
   * Computes the squared Euclidean distance between a vector and an 8-bit NVQ quantized vector.
   *
   * @param vector the query vector
   * @param bytes the 8-bit quantized vector
   * @param growthRate the growth rate parameter of the logistic quantization function
   * @param midpoint the midpoint parameter of the logistic quantization function
   * @param minValue the minimum value of the quantized subvector
   * @param maxValue the maximum value of the quantized subvector
   * @return the squared Euclidean distance
   */
  public static float nvqSquareL2Distance8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate, float midpoint, float minValue, float maxValue) {
    return impl.nvqSquareL2Distance8bit(vector, bytes, growthRate, midpoint, minValue, maxValue);
  }

  /**
   * Computes the cosine similarity between a vector and an 8-bit NVQ quantized vector.
   *
   * @param vector the query vector
   * @param bytes the 8-bit quantized vector
   * @param growthRate the growth rate parameter of the logistic quantization function
   * @param midpoint the midpoint parameter of the logistic quantization function
   * @param minValue the minimum value of the quantized subvector
   * @param maxValue the maximum value of the quantized subvector
   * @param centroid the global mean vector used to re-center the quantized subvectors
   * @return an array containing the cosine similarity components
   */
  public static float[] nvqCosine8bit(VectorFloat<?> vector, ByteSequence<?> bytes, float growthRate, float midpoint, float minValue, float maxValue, VectorFloat<?> centroid) {
    return impl.nvqCosine8bit(vector, bytes, growthRate, midpoint, minValue, maxValue, centroid);
  }

  /**
   * Shuffles a query vector in place to optimize NVQ quantized vector unpacking performance.
   *
   * @param vector the vector to shuffle (modified in place)
   */
  public static void nvqShuffleQueryInPlace8bit(VectorFloat<?> vector) {
    impl.nvqShuffleQueryInPlace8bit(vector);
  }

  /**
   * Quantizes a vector as an 8-bit NVQ quantized vector.
   *
   * @param vector the vector to quantize
   * @param growthRate the growth rate parameter of the logistic quantization function
   * @param midpoint the midpoint parameter of the logistic quantization function
   * @param minValue the minimum value of the subvector
   * @param maxValue the maximum value of the subvector
   * @param destination the byte sequence to store the quantized values (modified in place)
   */
  public static void nvqQuantize8bit(VectorFloat<?> vector, float growthRate, float midpoint, float minValue, float maxValue, ByteSequence<?> destination) {
    impl.nvqQuantize8bit(vector, growthRate, midpoint, minValue, maxValue, destination);
  }

  /**
   * Computes the squared error (loss) of quantizing a vector with NVQ.
   *
   * @param vector the vector to quantize
   * @param growthRate the growth rate parameter of the logistic quantization function
   * @param midpoint the midpoint parameter of the logistic quantization function
   * @param minValue the minimum value of the subvector
   * @param maxValue the maximum value of the subvector
   * @param nBits the number of bits per dimension
   * @return the squared error
   */
  public static float nvqLoss(VectorFloat<?> vector, float growthRate, float midpoint, float minValue, float maxValue, int nBits) {
    return impl.nvqLoss(vector, growthRate, midpoint, minValue, maxValue, nBits);
  }

  /**
   * Computes the squared error (loss) of quantizing a vector with a uniform quantizer.
   *
   * @param vector the vector to quantize
   * @param minValue the minimum value of the subvector
   * @param maxValue the maximum value of the subvector
   * @param nBits the number of bits per dimension
   * @return the squared error
   */
  public static float nvqUniformLoss(VectorFloat<?> vector, float minValue, float maxValue, int nBits) {
    return impl.nvqUniformLoss(vector, minValue, maxValue, nBits);
  }
}
