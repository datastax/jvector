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

  /** Effective ASH kernel tuning for benchmark diagnostics. */
  default String ashKernelDescription() { return "portable scalar ASH kernels"; }

  /** Query factories select the tuned entry point once, outside the scoring hot loop. */
  default boolean usesAshProjectionTuning() { return false; }

  /** Explicitly tuned projection kernel; the ordinary entry point retains its compact default loop. */
  default float ashProjectionDotTuned(float[] query, byte[] code, int dimensions, int bitsPerDimension) {
    return ashProjectionDotTuned(query, code, 0, dimensions, bitsPerDimension);
  }

  /** As {@link #ashProjectionDotTuned(float[], byte[], int, int)} for the code starting at {@code codeOffset}. */
  default float ashProjectionDotTuned(float[] query, byte[] code, int codeOffset, int dimensions, int bitsPerDimension) {
    return ashProjectionDot(query, code, codeOffset, dimensions, bitsPerDimension);
  }

  /** Whether ASH nibble lookup and accumulation are implemented with SIMD. */
  default boolean supportsAshLutScoring() { return false; }

  /** Exact integer lookup for encoded C=1 query pairs. LUT stride is 32 (two repeated 16-entry tables). */
    default void ashSymmetricLutScore(byte[] codes, int groups, int stride, int lane, int count,
                                      short[] lut, float[] out, int outOffset) {
        for (int i = 0; i < count; i++) {
            long sum = 0;
            for (int g = 0; g < groups; g++) {
                int nibble = (codes[(g / 2) * stride + lane + i] >>> (4 * (g % 2))) & 15;
                sum += lut[g * 32 + nibble];
            }
            out[outOffset + i] = sum * .25f;
        }
    }

  /** Packed-to-packed integer dot products for symmetric ASH. */
  default boolean supportsAshSymmetricScoring() { return false; }

  default float ashSymmetricDot(long[] aBits, byte[] aCodes, long[] bBits, byte[] bCodes,
                                int dimensions, int bits) {
    throw new UnsupportedOperationException("Symmetric ASH SIMD is unavailable on this backend");
  }

  /** Whether {@link #ashSymmetricDotAt} is implemented (2- and 4-bit codes at array offsets). */
  default boolean supportsAshSymmetricDotAt() { return false; }

  /**
   * Symmetric 2/4-bit packed dot product of codes stored at offsets inside larger arrays.
   * Same result as {@link #ashSymmetricDot} on the corresponding standalone code arrays.
   */
  default float ashSymmetricDotAt(byte[] aCodes, int aOffset, byte[] bCodes, int bOffset,
                                  int dimensions, int bits) {
    throw new UnsupportedOperationException("Offset symmetric ASH SIMD is unavailable on this backend");
  }

  /** Whether {@link #ashSymmetricDotRows} scores all rows in one call rather than one call per row. */
  default boolean supportsAshSymmetricDotRows() { return false; }

  /**
   * Symmetric 2/4-bit dot products of one code against {@code count} codes stored {@code stride} bytes apart from
   * the start of {@code rows}: {@code out[i]} equals {@link #ashSymmetricDotAt} of the query and row {@code i}.
   * Requires {@link #supportsAshSymmetricDotAt()}.
   */
  default void ashSymmetricDotRows(byte[] query, int queryOffset, byte[] rows, int stride, int count,
                                   int dimensions, int bits, float[] out) {
    for (int i = 0; i < count; i++) out[i] = ashSymmetricDotAt(query, queryOffset, rows, i * stride, dimensions, bits);
  }

  /**
   * Symmetric 1-bit dot product, {@code dimensions - 2 * mismatches}, of sign words stored at offsets
   * inside larger arrays. Same result as {@link #ashSymmetricDot} with {@code bits == 1}.
   * Available whenever {@link #supportsAshSymmetricScoring()} is.
   */
  default float ashSymmetricBinaryDotAt(long[] aBits, int aOffset, long[] bBits, int bOffset, int dimensions) {
    throw new UnsupportedOperationException("Symmetric ASH SIMD is unavailable on this backend");
  }

  /** Whether canonical 2/4-bit projection codes have a single-vector SIMD kernel. */
  default boolean supportsAshProjectionScoring() { return false; }

  /** Whether canonical ASH codes can use an already prepared nibble LUT in SIMD. */
  default boolean supportsAshCanonicalLutScoring() { return false; }

  /** Whether canonical LUT scoring is implemented by the native backend. */
  default boolean supportsNativeAshCanonicalLutScoring() { return false; }

  /** Fail before scoring when the caller explicitly requires a native ASH single scorer. */
  default void requireNativeAshSingleScoring(int bitsPerDimension) {
    throw new IllegalStateException("Native ASH single scoring is unavailable for b=" + bitsPerDimension);
  }

  /** Projection dot only; callers apply the vector's scale, offset and landmark term. */
  default float ashCanonicalLutDot(byte[] code, int groups, float[] lut) {
    return ashCanonicalLutDot(code, 0, groups, lut);
  }

  /** As {@link #ashCanonicalLutDot(byte[], int, float[])} for the code starting at {@code codeOffset}. */
  default float ashCanonicalLutDot(byte[] code, int codeOffset, int groups, float[] lut) {
    return ASHLutScoring.dotCanonical(code, codeOffset, groups, lut);
  }

  /** Dot product of a projected query with a canonical signed-magnitude 2/4-bit code. */
  default float ashProjectionDot(float[] query, byte[] code, int dimensions, int bitsPerDimension) {
    return ashProjectionDot(query, code, 0, dimensions, bitsPerDimension);
  }

  /** As {@link #ashProjectionDot(float[], byte[], int, int)} for the code starting at {@code codeOffset}. */
  default float ashProjectionDot(float[] query, byte[] code, int codeOffset, int dimensions, int bitsPerDimension) {
    if (bitsPerDimension != 2 && bitsPerDimension != 4) {
      throw new IllegalArgumentException("Projection scoring requires 2 or 4 bits per dimension");
    }
    java.util.Objects.checkFromIndexSize(0, dimensions, query.length);
    int perByte = 8 / bitsPerDimension;
    java.util.Objects.checkFromIndexSize(codeOffset, dimensions / perByte + (dimensions % perByte == 0 ? 0 : 1), code.length);
    int sign = 1 << (bitsPerDimension - 1);
    float sum = 0;
    for (int i = 0; i < dimensions; i++) {
      int field = (code[codeOffset + i / perByte] & 255) >>> ((i % perByte) * bitsPerDimension);
      float magnitude = (field & (sign - 1)) + 0.5f;
      sum += query[i] * ((field & sign) == 0 ? -magnitude : magnitude);
    }
    return sum;
  }

  /**
   * Scores contiguous lanes in a byte-interleaved ASH block using a 16-entry LUT per nibble.
   * Even groups occupy low nibbles; odd groups occupy high nibbles. Each byte row has
   * {@code stride} lanes. Writes projection dot products, without per-vector headers.
   */
  default void ashLutScore(byte[] codes, int offset, int groups, int stride,
                           int lane, int count, float[] lut, float[] out, int outOffset) {
    ASHLutScoring.score(codes, offset, groups, stride, lane, count, lut, out, outOffset);
  }

  /**
   * Whether this provider implements complete FusedASH neighborhood scoring for one landmark.
   * Query factories use this capability independently of the standalone LUT primitive.
   *
   * @return {@code true} when {@link #ashScoreNeighborhood} is implemented
   */
  default boolean supportsAshNeighborhoodScoring() { return false; }

  /**
   * Scores one complete C=1 FusedASH neighborhood, including all serialized blocks and headers.
   * The result is {@code max(0, (1 + rawDotProduct) / 2)}, matching graph DOT_PRODUCT similarity.
   *
   * <p>Inputs are read-only for the call; output storage must be distinct from {@code lut}.
   * Calls may share immutable inputs across threads when their output arrays are separate.
   * On failure, output contents are unspecified. Implementations validate sizes before native access.</p>
   *
   * @param codes complete packed neighborhood, with full serialized storage for its final block
   * @param groups positive nibble group count, including a possible partial final group
   * @param blockSize serialized block width: 8, 16, or 32
   * @param capacity positive number of lanes to score, including graph padding
   * @param lut query LUT with at least {@code 16 * groups} floats
   * @param landmarkDot query dot product with the single landmark
   * @param out receives {@code capacity} similarities; later elements are left unchanged
   * @throws UnsupportedOperationException if this provider has no implementation
   * @throws IllegalArgumentException for an invalid shape or aliased LUT/output
   * @throws IndexOutOfBoundsException if an array is too small
   * @throws IllegalStateException if a scored lane contains a nonzero landmark ID
   */
  default void ashScoreNeighborhood(byte[] codes, int groups, int blockSize, int capacity,
                                    float[] lut, float landmarkDot, float[] out) {
    throw new UnsupportedOperationException("Native ASH neighborhood scoring is unavailable");
  }

  /** Whether {@link #ashScoreNeighborhoodInt8} is implemented. */
  default boolean supportsAshInt8NeighborhoodScoring() { return false; }

  /**
   * Scores one complete C=1 FusedASH neighborhood of offset-binary codes (3 and 5 to 9 bits) with an
   * INT8 query. Each block's body is {@code [unit][lane][bits bytes]} for {@code units} units of
   * eight dimensions, followed by the usual headers. Per lane,
   * {@code dot = (<q, field> - center) * invScale}, and the result is
   * {@code max(0, (1 + scale * dot + landmarkDot + offset) / 2)}.
   *
   * @param codes complete packed neighborhood, with full serialized storage for its final block
   * @param bits bits per dimension: 3, 5, 6, 7, 8 or 9
   * @param units positive number of eight-dimension units
   * @param blockSize serialized block width: 8, 16, or 32
   * @param capacity positive number of lanes to score, including graph padding
   * @param q INT8 query, {@code units * 8} entries
   * @param invScale reciprocal of the query quantization scale
   * @param center {@code (2^bits - 1)/2 * sum(q)}
   * @param landmarkDot query dot product with the single landmark
   * @param out receives {@code capacity} similarities; later elements are left unchanged
   * @throws UnsupportedOperationException if this provider has no implementation
   * @throws IllegalArgumentException for an invalid shape
   * @throws IndexOutOfBoundsException if an array is too small
   * @throws IllegalStateException if a scored lane contains a nonzero landmark ID
   */
  default void ashScoreNeighborhoodInt8(byte[] codes, int bits, int units, int blockSize, int capacity,
                                        byte[] q, float invScale, float center, float landmarkDot, float[] out) {
    throw new UnsupportedOperationException("Native INT8 ASH neighborhood scoring is unavailable");
  }

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

  // ------------------------------------------------------------------
  // ASH kernels (default scalar implementations)
  // ------------------------------------------------------------------

  /**
   * @return true if this backend provides a SIMD implementation of the masked-load kernel as described in the ASH paper.
   */
  default boolean supportsAshMaskedLoad() {
    return false;
  }

  /** Immutable provider-specific projection, safe to share between query threads. */
  interface AshProjection {
    /** Writes the full matrix-vector product; input and output must not alias. */
    void project(float[] query, float[] out);

    /**
     * Projects the first {@code count} vectors, each stored in its own row.
     * All input rows must have the model's original dimension and output rows
     * its projected dimension. Output rows must be distinct and must not alias
     * any input row. Rows beyond count are ignored; count zero is a no-op.
     * Providers may batch the multiply; the default retains their single-vector
     * arithmetic without allocating temporary arrays.
     */
    default void projectBatch(float[][] inputs, float[][] outputs, int count) {
      validateBatch(inputs, outputs, count);
      for (int i = 0; i < count; i++) project(inputs[i], outputs[i]);
    }

    /**
     * Whether this view has a genuine multi-vector projection implementation.
     * Query dispatch uses this capability to avoid coordinating providers whose
     * batch method simply repeats singleton calls. This is a capability flag,
     * not a performance promise, and may become false after a native fallback.
     */
    default boolean supportsBatchProjection() { return false; }

    /** Validates a batch before any output is changed. */
    static void validateBatch(float[][] inputs, float[][] outputs, int count) {
      java.util.Objects.requireNonNull(inputs, "inputs");
      java.util.Objects.requireNonNull(outputs, "outputs");
      if (count < 0 || count > inputs.length || count > outputs.length)
        throw new IllegalArgumentException("Invalid ASH projection batch count: " + count);
      for (int i = 0; i < count; i++) {
        java.util.Objects.requireNonNull(inputs[i], "input row");
        java.util.Objects.requireNonNull(outputs[i], "output row");
        if (inputs[i].length != inputs[0].length || outputs[i].length != outputs[0].length)
          throw new IllegalArgumentException("Ragged ASH projection batch");
      }
      if (count <= 32) {
        // The normal encoding batch needs no temporary collection. For larger
        // API requests use linear validation rather than an unbounded scan.
        for (int i = 0; i < count; i++)
          for (int j = 0; j < count; j++)
            if (outputs[i] == inputs[j] || (j < i && outputs[i] == outputs[j]))
              throw new IllegalArgumentException("Aliased ASH projection batch");
      } else {
        var used = new java.util.IdentityHashMap<float[], Boolean>(count);
        for (int i = 0; i < count; i++) used.put(inputs[i], Boolean.TRUE);
        for (int i = 0; i < count; i++)
          if (used.put(outputs[i], Boolean.TRUE) != null)
            throw new IllegalArgumentException("Aliased ASH projection batch");
      }
    }

    /**
     * Returns an immutable view for query scoring, which may use lower operand
     * precision. Encoding/training must continue to use this original object.
     * The caller caches the returned view for the model's lifetime. Providers
     * without a query-specific representation return this same object.
     */
    default AshProjection forQueryScoring() {
      String mode = System.getProperty("jvector.ash.queryProjection", "auto");
      switch (mode) {
        case "auto":
        case "fp32": return this;
        case "fp16": throw new UnsupportedOperationException(
                "This provider does not support FP16 ASH query projection");
        case "int8": throw new UnsupportedOperationException(
                "This provider does not support INT8 ASH query projection");
        default: throw new IllegalArgumentException(
                "jvector.ash.queryProjection must be auto, fp32, fp16 or int8: " + mode);
      }
    }

    /** Operand precision of this prepared view; reporting must not initialize another view. */
    default String operandPrecision() { return "fp32"; }

    /** Additional backing storage, excluding the caller-owned matrix. */
    default long ramBytesUsed() { return 0L; }
  }

  /**
   * Prepares an immutable [projected dimension][original dimension] matrix once.
   * The caller must not mutate its rows after preparation. Providers may cache a
   * packed representation; each project call still prepares exactly one query.
   */
  default AshProjection prepareAshProjection(float[][] matrix) {
    java.util.Objects.requireNonNull(matrix, "matrix");
    if (matrix.length == 0 || matrix[0].length == 0)
      throw new IllegalArgumentException("ASH projection dimensions must be positive");
    int columns = matrix[0].length;
    for (float[] row : matrix) {
      if (row.length != columns) throw new IllegalArgumentException("Ragged ASH projection");
    }
    return (query, out) -> {
      if (query.length != columns || out.length != matrix.length || query == out)
        throw new IllegalArgumentException("ASH projection shape or aliasing mismatch");
      for (int row = 0; row < matrix.length; row++) out[row] = ashDotRow(matrix[row], query);
    };
  }

  /**
   * Dot product between a matrix row and a dense vector.
   * Used by ASH projection: {@code y_i = <A_i, x>}.
   *
   * Contract: Arow.length == x.length.
   * Default is scalar; SIMD backends may override.
   */
  default float ashDotRow(float[] Arow, float[] x) {
    assert Arow.length == x.length : "Arow.length != x.length";
    float acc = 0.0f;
    for (int i = 0; i < x.length; i++) {
      acc += Arow[i] * x[i];
    }
    return acc;
  }

  /**
   * Sums {@code tildeQ[qOffset + i]} over the set bits {@code i < d} of one vector's packed sign words.
   * The default is portable scalar code; SIMD backends override it.
   */
  default float ashMaskedAdd_512(float[] tildeQ,
                                         int qOffset,
                                         long[] allPackedVectors,
                                         int packedBase,
                                         int d,
                                         int words) {
    float sum = 0.0f;
    int baseDim = 0;

    for (int w = 0; w < words; w++) {
      long word = allPackedVectors[packedBase + w];

      // Standard bit-twiddling iteration (iterate only set bits)
      while (word != 0L) {
        int bit = Long.numberOfTrailingZeros(word);
        int idx = baseDim + bit;

        // Bounds check required for the last word
        if (idx < d) {
          sum += tildeQ[qOffset + idx];
        }

        // Clear the lowest set bit
        word &= (word - 1);
      }

      baseDim += 64;
    }
    return sum;
  }

  /**
   * Computes maskedAdd for a block slice of vectors stored in packed block-column-major form.
   * For each lane in [0, blockLen), writes:
   *   {@code outMaskedAdd[lane] = <tildeQ, b_lane>}
   * packedBits layout:
   *   packedBits[blockWordBase + w*blockSize + laneIndex] is 64-bit word w for that lane.
   * Default implementation is scalar bit-walk with a register accumulator per lane.
   * SIMD backends may override.
   */
  default void ashMaskedAddBlockAllWords(
          float[] tildeQ,
          int d,
          long[] packedBits,
          int packedBase,  // blockId * nWords * blockSize
          int nWords,
          int blockSize,
          int laneStart,
          int blockLen,
          float[] outMaskedAdd
  ) {
    assert laneStart >= 0;
    assert blockLen >= 0;
    assert laneStart + blockLen <= blockSize;
    assert outMaskedAdd.length >= blockLen : "outMaskedAdd too small";

    for (int lane = 0; lane < blockLen; lane++) {
      float maskedAdd = 0f; // register accumulator (critical!)
      int laneIndex = laneStart + lane;

      for (int w = 0; w < nWords; w++) {
        long word = packedBits[packedBase + (w * blockSize) + laneIndex];
        int baseDim = w * 64;

        while (word != 0L) {
          int bit = Long.numberOfTrailingZeros(word);
          int idx = baseDim + bit;
          if (idx < d) maskedAdd += tildeQ[idx];
          word &= (word - 1);
        }
      }

      outMaskedAdd[lane] = maskedAdd;
    }
  }

  /**
   * Optionally performs the existing coarse/fine NVQ growth-rate search in one
   * call. Returns NaN when unsupported; the caller then uses its Java search.
   * Implementations must preserve the candidate sequence and strict winner test.
   */
  default float tryNvqGrowthRate(VectorFloat<?> vector, float minValue, float maxValue, int nBits) {
    return Float.NaN;
  }

  /**
   * Sets the native BLAS thread count for calls made on the current thread, as
   * {@link io.github.jbellis.jvector.util.BlasThreads} requests. Returns the setting to restore
   * afterward, or -1 when this provider cannot control BLAS threads.
   */
  default int setBlasThreadsLocal(int threads) {
    return -1;
  }

}
