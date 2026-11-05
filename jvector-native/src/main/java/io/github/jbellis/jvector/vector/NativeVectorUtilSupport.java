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

package io.github.jbellis.jvector.vector;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.Objects;

import io.github.jbellis.jvector.vector.cnative.NativeSimdOps;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

/**
 * VectorUtilSupport using Highway kernels and inherited Panama operations.
 */
class NativeVectorUtilSupport extends PanamaVectorUtilSupport
{
    public NativeVectorUtilSupport() {}

    /** Select once at provider creation; leave the wide scan and its call site unchanged. */
    static NativeVectorUtilSupport create() {
        return FloatVector.SPECIES_PREFERRED.vectorBitSize() < 256
                ? new NativeVectorUtilSupport128() : new NativeVectorUtilSupport();
    }

    /**
     * Returns the ISA tier that was selected by the native dispatcher at library
     * initialisation time.  Possible values: {@code "avx3_spr"}, {@code "avx3_dl"},
     * {@code "avx3"}, {@code "avx2"}, {@code "sse42"}.
     */
    public String getActiveIsa() {
        return NativeSimdOps.jvector_simd_get_active_isa()
                .reinterpret(Long.MAX_VALUE)
                .getString(0);
    }

    /**
     * Returns the value of {@code JVECTOR_MAX_ISA} that was in effect when the
     * native library was loaded, or {@code null} if the variable was absent or
     * contained an unrecognised value.
     * Possible non-null return values: {@code "avx3_spr"}, {@code "avx3_dl"},
     * {@code "avx3"}, {@code "avx2"}, {@code "sse42"}.
     */
    public String getMaxIsaEnv() {
        MemorySegment ptr = NativeSimdOps.jvector_simd_get_max_isa_env();
        if (ptr.address() == 0L) return null;
        return ptr.reinterpret(Long.MAX_VALUE).getString(0);
    }

    @Override
    public boolean tryUpdateKMeansDistances(VectorFloat<?>[] points, VectorFloat<?> centroid,
                                           VectorFloat<?> minima, VectorFloat<?> scratch) {
        return NativeKMeansInitialization.update(points, centroid, minima, scratch);
    }

    @Override
    public float nvqMin(VectorFloat<?> vector) {
        return NativeNVQArithmetic.min(vector);
    }

    @Override
    public void nvqScale(VectorFloat<?> vector, float multiplier) {
        NativeNVQArithmetic.scale(vector, multiplier);
    }

    @Override
    public VectorFloat<?> nvqSub(VectorFloat<?> a, VectorFloat<?> b) {
        return NativeNVQArithmetic.sub(a, 0, b, 0, a.length());
    }

    @Override
    public VectorFloat<?> nvqSub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length) {
        return NativeNVQArithmetic.sub(a, aOffset, b, bOffset, length);
    }

    @Override
    protected FloatVector fromVectorFloat(VectorSpecies<Float> SPEC, VectorFloat<?> vector, int offset) {
        return FloatVector.fromMemorySegment(SPEC, ((MemorySegmentVectorFloat) vector).get(), vector.offset(offset), ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    protected FloatVector fromVectorFloat(VectorSpecies<Float> SPEC, VectorFloat<?> vector, int offset, int[] indices, int indicesOffset) {
        // Native vectors are heap-backed; inherited Java gather kernels can read that array directly.
        return FloatVector.fromArray(SPEC, ((MemorySegmentVectorFloat) vector).array(),
                offset, indices, indicesOffset);
    }

    @Override
    protected void intoVectorFloat(FloatVector vector, VectorFloat<?> v, int offset) {
        vector.intoMemorySegment(((MemorySegmentVectorFloat) v).get(), v.offset(offset), ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    protected ByteVector fromByteSequence(VectorSpecies<Byte> SPEC, ByteSequence<?> vector, int offset) {
        return ByteVector.fromMemorySegment(SPEC, ((MemorySegmentByteSequence) vector).get(), offset, ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    protected void intoByteSequence(ByteVector vector, ByteSequence<?> v, int offset) {
        vector.intoMemorySegment(((MemorySegmentByteSequence) v).get(), offset, ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    protected void intoByteSequence(ByteVector vector, ByteSequence<?> v, int offset, VectorMask<Byte> mask) {
        vector.intoMemorySegment(((MemorySegmentByteSequence) v).get(), offset, ByteOrder.LITTLE_ENDIAN, mask);
    }

    @Override
    public float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets) {
        return assembleAndSum(data, dataBase, baseOffsets, 0, baseOffsets.length());
    }

    @Override
    public float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength)
    {
        assert baseOffsets.offset() == 0 : "Base offsets are expected to have an offset of 0. Found: " + baseOffsets.offset();
        // baseOffsets is a pointer into a PQ chunk - we need to index into it by baseOffsetsOffset and provide baseOffsetsLength to the native code
        return NativeSimdOps.assemble_and_sum_f32(((MemorySegmentVectorFloat) data).get(), dataBase, ((MemorySegmentByteSequence) baseOffsets).get(), baseOffsetsOffset, (long) baseOffsetsLength);
    }

    @Override
    public void addInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        NativeSimdOps.add_in_place_f32(((MemorySegmentVectorFloat) v1).get(), ((MemorySegmentVectorFloat) v2).get(), v1.length());
    }

    @Override
    public void addInPlace(VectorFloat<?> v1, float value) {
        NativeSimdOps.add_scalar_in_place_f32(((MemorySegmentVectorFloat) v1).get(), value, v1.length());
    }

    @Override
    public void scale(VectorFloat<?> vector, float multiplier) {
        NativeSimdOps.scale_f32(((MemorySegmentVectorFloat) vector).get(), multiplier, vector.length());
    }

    @Override
    public void subInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        NativeSimdOps.sub_in_place_f32(((MemorySegmentVectorFloat) v1).get(), ((MemorySegmentVectorFloat) v2).get(), v1.length());
    }

    @Override
    public void subInPlace(VectorFloat<?> vector, float value) {
        NativeSimdOps.sub_scalar_in_place_f32(((MemorySegmentVectorFloat) vector).get(), value, vector.length());
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, VectorFloat<?> b) {
        return sub(a, 0, b, 0, a.length());
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, float value) {
        return sub(a, 0, value, a.length());
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length) {
        Objects.checkFromIndexSize(aOffset, length, a.length());
        Objects.checkFromIndexSize(bOffset, length, b.length());
        VectorFloat<?> res = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(length);
        NativeSimdOps.sub_f32(
                ((MemorySegmentVectorFloat) a).get(), aOffset,
                ((MemorySegmentVectorFloat) b).get(), bOffset,
                ((MemorySegmentVectorFloat) res).get(), length);
        return res;
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, int aOffset, float value, int length) {
        Objects.checkFromIndexSize(aOffset, length, a.length());
        VectorFloat<?> res = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(length);
        NativeSimdOps.sub_scalar_f32(
                ((MemorySegmentVectorFloat) a).get(), aOffset,
                value,
                ((MemorySegmentVectorFloat) res).get(), length);
        return res;
    }

    @Override
    public float max(VectorFloat<?> v) {
        return NativeSimdOps.max_f32(((MemorySegmentVectorFloat) v).get(), v.length());
    }

    @Override
    public float min(VectorFloat<?> v) {
        return NativeSimdOps.min_f32(((MemorySegmentVectorFloat) v).get(), v.length());
    }

    @Override
    public void minInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        NativeSimdOps.min_in_place_f32(((MemorySegmentVectorFloat) v1).get(), ((MemorySegmentVectorFloat) v2).get(), v1.length());
    }

    @Override
    public float sum(VectorFloat<?> vector) {
        return NativeSimdOps.sum_f32(((MemorySegmentVectorFloat) vector).get(), vector.length());
    }

    @Override
    public int hammingDistance(long[] a, long[] b) {
        Objects.checkFromIndexSize(0, a.length, b.length);
        return super.hammingDistance(a, b);
    }

    @Override
    public float assembleAndSumPQ(
            VectorFloat<?> codebookPartialSums,
            int subspaceCount,                  // = M
            ByteSequence<?> vector1Ordinals,
            int vector1OrdinalOffset,
            ByteSequence<?> vector2Ordinals,
            int vector2OrdinalOffset,
            int clusterCount                    // = k
    ) {
        assert vector1Ordinals.offset() == 0 : "vector1Ordinals offset must be 0. Found: " + vector1Ordinals.offset();
        assert vector2Ordinals.offset() == 0 : "vector2Ordinals offset must be 0. Found: " + vector2Ordinals.offset();
        return NativeSimdOps.assemble_and_sum_pq_f32(
                ((MemorySegmentVectorFloat) codebookPartialSums).get(),
                (long) subspaceCount,
                ((MemorySegmentByteSequence) vector1Ordinals).get(), vector1OrdinalOffset,
                ((MemorySegmentByteSequence) vector2Ordinals).get(), vector2OrdinalOffset,
                clusterCount);
    }

    @Override
    public float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
        return pqDecodedCosineSimilarity(encoded, 0, encoded.length(), clusterCount, partialSums, aMagnitude, bMagnitude);
    }

    @Override
    public float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int encodedOffset, int encodedLength, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
        assert encoded.offset() == 0 : "Bulk shuffle shuffles are expected to have an offset of 0. Found: " + encoded.offset();
        // encoded is a pointer into a PQ chunk - we need to index into it by encodedOffset and provide encodedLength to the native code
        return NativeSimdOps.pq_decoded_cosine_similarity_f32(((MemorySegmentByteSequence) encoded).get(), encodedOffset, (long) encodedLength, clusterCount, ((MemorySegmentVectorFloat) partialSums).get(), ((MemorySegmentVectorFloat) aMagnitude).get(), bMagnitude);
    }

    @Override
    public float squareDistance(VectorFloat<?> v1, VectorFloat<?> v2) {
        return NativeSimdOps.euclidean_f32(((MemorySegmentVectorFloat) v1).get(), 0,
                                          ((MemorySegmentVectorFloat) v2).get(), 0,
                                          v1.length());
    }

    @Override
    public float squareDistance(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {
        return NativeSimdOps.euclidean_f32(((MemorySegmentVectorFloat) v1).get(), v1offset,
                                          ((MemorySegmentVectorFloat) v2).get(), v2offset,
                                          length);
    }

    @Override
    public float cosine(VectorFloat<?> v1, VectorFloat<?> v2) {
        return NativeSimdOps.cosine_f32(((MemorySegmentVectorFloat) v1).get(), 0,
                                       ((MemorySegmentVectorFloat) v2).get(), 0,
                                       v1.length());
    }

    @Override
    public float cosine(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {
        return NativeSimdOps.cosine_f32(((MemorySegmentVectorFloat) v1).get(), v1offset,
                                       ((MemorySegmentVectorFloat) v2).get(), v2offset,
                                       length);
    }

    @Override
    public float dotProduct(VectorFloat<?> v1, VectorFloat<?> v2) {
        return NativeSimdOps.dot_product_f32(((MemorySegmentVectorFloat) v1).get(), 0,
                                             ((MemorySegmentVectorFloat) v2).get(), 0,
                                             v1.length());
    }

    @Override
    public float dotProduct(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {
        return NativeSimdOps.dot_product_f32(((MemorySegmentVectorFloat) v1).get(), v1offset,
                                             ((MemorySegmentVectorFloat) v2).get(), v2offset,
                                             length);
    }

    @Override
    public int tryClosestCentroid(VectorFloat<?> point, int pointOffset,
                               VectorFloat<?> centroids, int centroidCount, int dimension) {
        if (centroidCount <= 0) {
            throw new IllegalArgumentException("centroidCount must be positive");
        }
        java.util.Objects.checkFromIndexSize(pointOffset, dimension, point.length());
        if ((long) centroidCount * dimension > centroids.length()) {
            throw new IndexOutOfBoundsException("centroids do not contain centroidCount vectors");
        }
        return NativeSimdOps.closest_centroid_f32(
                ((MemorySegmentVectorFloat) point).get(), pointOffset,
                ((MemorySegmentVectorFloat) centroids).get(), centroidCount, dimension);
    }

    @Override
    public boolean tryScaleTo(VectorFloat<?> source, VectorFloat<?> destination,
                              int destinationOffset, float multiplier) {
        java.util.Objects.checkFromIndexSize(destinationOffset, source.length(), destination.length());
        var src = ((MemorySegmentVectorFloat) source).get();
        var dst = ((MemorySegmentVectorFloat) destination).get();
        if (src.asOverlappingSlice(dst).isPresent()) {
            throw new IllegalArgumentException("Source and destination storage must not overlap");
        }
        NativeSimdOps.scale_to_f32(src, dst, destinationOffset, multiplier, source.length());
        return true;
    }

    @Override
    public boolean trySubtractAddInPlace(VectorFloat<?> oldSum, VectorFloat<?> newSum,
                                        VectorFloat<?> point) {
        if (oldSum.length() != point.length() || newSum.length() != point.length()) {
            throw new IllegalArgumentException("Cluster sums and point must have equal lengths");
        }
        var oldSegment = ((MemorySegmentVectorFloat) oldSum).get();
        var newSegment = ((MemorySegmentVectorFloat) newSum).get();
        var pointSegment = ((MemorySegmentVectorFloat) point).get();
        if (oldSegment.asOverlappingSlice(newSegment).isPresent()
                || oldSegment.asOverlappingSlice(pointSegment).isPresent()
                || newSegment.asOverlappingSlice(pointSegment).isPresent()) {
            throw new IllegalArgumentException("Cluster sums and point storage must not overlap");
        }
        NativeSimdOps.subtract_add_in_place_f32(oldSegment, newSegment, pointSegment, point.length());
        return true;
    }

    @Override
    public void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int queryOffset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums) {
        var nativeCodebook = ((MemorySegmentVectorFloat) codebook).get();
        var nativeQuery = ((MemorySegmentVectorFloat) query).get();
        var nativePartialSums = ((MemorySegmentVectorFloat) partialSums).get();
        switch (vsf) {
            case EUCLIDEAN -> NativeSimdOps.calculate_partial_sums_euclidean_f32(nativeCodebook, codebookIndex, (long) size, clusterCount, nativeQuery, queryOffset, nativePartialSums);
            case DOT_PRODUCT -> NativeSimdOps.calculate_partial_sums_dot_f32(nativeCodebook, codebookIndex, (long) size, clusterCount, nativeQuery, queryOffset, nativePartialSums);
            default -> throw new UnsupportedOperationException("Unsupported similarity function " + vsf);
        }
    }

    @Override
    public void calculatePartialSelfMagnitudes(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> partialMagnitudes) {
        NativeSimdOps.calculate_partial_sums_self_magnitude_f32(
                ((MemorySegmentVectorFloat) codebook).get(),
                codebookIndex,
                (long) size,
                clusterCount,
                ((MemorySegmentVectorFloat) partialMagnitudes).get());
    }

    @Override
    public void nvqShuffleQueryInPlace8bit(VectorFloat<?> vector) {
        NativeSimdOps.nvq_shuffle_query_in_place_8bit(
                ((MemorySegmentVectorFloat) vector).get(),
                (long) vector.length());
    }

    @Override
    public void nvqQuantize8bit(VectorFloat<?> vector, float alpha, float x0, float minValue, float maxValue, ByteSequence<?> destination) {
        NativeSimdOps.nvq_quantize_8bit(
                ((MemorySegmentVectorFloat) vector).get(),
                (long) vector.length(),
                alpha, x0, minValue, maxValue,
                ((MemorySegmentByteSequence) destination).get());
    }

    @Override
    public float tryNvqGrowthRate(VectorFloat<?> vector, float minValue, float maxValue, int nBits) {
        if (nBits != 8) return Float.NaN;
        return NativeSimdOps.nvq_growth_rate(((MemorySegmentVectorFloat) vector).get(),
                                            vector.length(), minValue, maxValue, nBits);
    }

    @Override
    public float nvqLoss(VectorFloat<?> vector, float alpha, float x0, float minValue, float maxValue, int nBits) {
        return NativeSimdOps.nvq_loss(
                ((MemorySegmentVectorFloat) vector).get(),
                (long) vector.length(),
                alpha, x0, minValue, maxValue, nBits);
    }

    @Override
    public float nvqUniformLoss(VectorFloat<?> vector, float minValue, float maxValue, int nBits) {
        return NativeSimdOps.nvq_uniform_loss(
                ((MemorySegmentVectorFloat) vector).get(),
                (long) vector.length(),
                minValue, maxValue, nBits);
    }

    @Override
    public float nvqSquareL2Distance8bit(VectorFloat<?> vector, ByteSequence<?> quantizedVector,
                                         float alpha, float x0, float minValue, float maxValue) {
        return NativeSimdOps.nvq_square_l2_distance_8bit(
                ((MemorySegmentVectorFloat) vector).get(),
                ((MemorySegmentByteSequence) quantizedVector).get(),
                (long) vector.length(),
                alpha, x0, minValue, maxValue);
    }

    @Override
    public float nvqDotProduct8bit(VectorFloat<?> vector, ByteSequence<?> quantizedVector,
                                   float alpha, float x0, float minValue, float maxValue) {
        return NativeSimdOps.nvq_dot_product_8bit(
                ((MemorySegmentVectorFloat) vector).get(),
                ((MemorySegmentByteSequence) quantizedVector).get(),
                (long) vector.length(),
                alpha, x0, minValue, maxValue);
    }

    @Override
    public boolean supportsNvqCosine8bitPacked() { return true; }

    @Override
    public long nvqCosine8bitPacked(VectorFloat<?> vector, ByteSequence<?> quantizedVector,
                                    float alpha, float x0, float minValue, float maxValue,
                                    VectorFloat<?> centroid) {
        return NativeSimdOps.nvq_cosine_8bit_packed(
                ((MemorySegmentVectorFloat) vector).get(),
                ((MemorySegmentByteSequence) quantizedVector).get(),
                (long) vector.length(), alpha, x0, minValue, maxValue,
                ((MemorySegmentVectorFloat) centroid).get());
    }

    @Override
    public float[] nvqCosine8bit(VectorFloat<?> vector, ByteSequence<?> quantizedVector,
                                 float alpha, float x0, float minValue, float maxValue,
                                 VectorFloat<?> centroid) {
        long packed = NativeSimdOps.nvq_cosine_8bit_packed(
                ((MemorySegmentVectorFloat) vector).get(),
                ((MemorySegmentByteSequence) quantizedVector).get(),
                (long) vector.length(),
                alpha, x0, minValue, maxValue,
                ((MemorySegmentVectorFloat) centroid).get());
        float sum  = Float.intBitsToFloat((int)(packed & 0xFFFFFFFFL));
        float bMag = Float.intBitsToFloat((int)(packed >>> 32));
        return new float[]{sum, bMag};
    }
}
