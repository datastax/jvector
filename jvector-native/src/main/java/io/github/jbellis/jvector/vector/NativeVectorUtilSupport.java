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

import io.github.jbellis.jvector.pq.LocallyAdaptiveVectorQuantization;
import io.github.jbellis.jvector.vector.cnative.NativeSimdOps;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.List;

/**
 * VectorUtilSupport implementation that prefers native/Panama SIMD.
 */
final class NativeVectorUtilSupport implements VectorUtilSupport
{
    @Override
    public float dotProduct(VectorFloat<?> a, VectorFloat<?> b) {
        return this.dotProduct(a, 0, b, 0, a.length());
    }

    @Override
    public float cosine(VectorFloat<?> v1, VectorFloat<?> v2) {
        return VectorSimdOps.cosineSimilarity((MemorySegmentVectorFloat)v1, (MemorySegmentVectorFloat)v2);
    }

    @Override
    public float cosine(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
        return VectorSimdOps.cosineSimilarity((MemorySegmentVectorFloat)a, aoffset, (MemorySegmentVectorFloat)b, boffset, length);
    }

    @Override
    public float squareDistance(VectorFloat<?> a, VectorFloat<?> b) {
         return this.squareDistance(a, 0, b, 0, a.length());
    }

    @Override
    public float squareDistance(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
        return VectorSimdOps.squareDistance((MemorySegmentVectorFloat) a, aoffset, (MemorySegmentVectorFloat) b, boffset, length);
    }

    @Override
    public float dotProduct(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
        return VectorSimdOps.dotProduct((MemorySegmentVectorFloat) a, aoffset, (MemorySegmentVectorFloat) b, boffset, length);
    }

    @Override
    public VectorFloat<?> sum(List<VectorFloat<?>> vectors) {
        return VectorSimdOps.sum(vectors);
    }

    @Override
    public float sum(VectorFloat<?> vector) {
        return VectorSimdOps.sum((MemorySegmentVectorFloat) vector);
    }

    @Override
    public void scale(VectorFloat<?> vector, float multiplier) {
        VectorSimdOps.scale((MemorySegmentVectorFloat) vector, multiplier);
    }

    @Override
    public void addInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        VectorSimdOps.addInPlace((MemorySegmentVectorFloat)v1, (MemorySegmentVectorFloat)v2);
    }

    @Override
    public void subInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        VectorSimdOps.subInPlace((MemorySegmentVectorFloat)v1, (MemorySegmentVectorFloat)v2);
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, VectorFloat<?> b) {
        if (a.length() != b.length()) {
            throw new IllegalArgumentException("Vectors must be the same length");
        }
        return sub(a, 0, b, 0, a.length());
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length) {
        return VectorSimdOps.sub((MemorySegmentVectorFloat) a, aOffset, (MemorySegmentVectorFloat) b, bOffset, length);
    }

    @Override
    public float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets) {
        return NativeSimdOps.assemble_and_sum_f32_512(((MemorySegmentVectorFloat)data).get(), dataBase, ((MemorySegmentByteSequence)baseOffsets).get(), baseOffsets.length());
    }

    @Override
    public int hammingDistance(long[] v1, long[] v2) {
        return VectorSimdOps.hammingDistance(v1, v2);
    }

    @Override
    public float max(VectorFloat<?> vector) {
        return VectorSimdOps.max((MemorySegmentVectorFloat) vector);
    }

    @Override
    public float min(VectorFloat<?> vector) {
        return VectorSimdOps.min((MemorySegmentVectorFloat) vector);
    }

    @Override
    public void bulkShuffleSimilarity(ByteSequence<?> shuffles, int codebookCount, VectorFloat<?> partials, VectorSimilarityFunction vsf, VectorFloat<?> results) {
        switch (vsf) {
            case DOT_PRODUCT -> NativeSimdOps.bulk_shuffle_dot_f32_512(((MemorySegmentByteSequence) shuffles).get(), codebookCount, ((MemorySegmentVectorFloat) partials).get(), ((MemorySegmentVectorFloat) results).get());
            case EUCLIDEAN -> NativeSimdOps.bulk_shuffle_euclidean_f32_512(((MemorySegmentByteSequence) shuffles).get(), codebookCount, ((MemorySegmentVectorFloat) partials).get(), ((MemorySegmentVectorFloat) results).get());
            case COSINE -> throw new UnsupportedOperationException("Cosine similarity not supported for bulkShuffleSimilarity");
        }
    }

    @Override
    public void calculatePartialSums(VectorFloat<?> codebook, int codebookBase, int size, int clusterCount, VectorFloat<?> query, int queryOffset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums) {
        switch (vsf) {
            case DOT_PRODUCT -> NativeSimdOps.calculate_partial_sums_dot_f32_512(((MemorySegmentVectorFloat)codebook).get(), codebookBase, size, clusterCount, ((MemorySegmentVectorFloat)query).get(), queryOffset, ((MemorySegmentVectorFloat)partialSums).get());
            case EUCLIDEAN -> NativeSimdOps.calculate_partial_sums_euclidean_f32_512(((MemorySegmentVectorFloat)codebook).get(), codebookBase, size, clusterCount, ((MemorySegmentVectorFloat)query).get(), queryOffset, ((MemorySegmentVectorFloat)partialSums).get());
            case COSINE -> throw new UnsupportedOperationException("Cosine similarity not supported for calculatePartialSums");
        }
    }

    @Override
    public void dotProductMultiScore(VectorFloat<?> v1, VectorFloat<?> v2, VectorFloat<?> results) {
        NativeSimdOps.dot_product_multi_f32_512(((MemorySegmentVectorFloat)v1).get(), ((MemorySegmentVectorFloat)v2).get(), v1.length(), results.length(), ((MemorySegmentVectorFloat)results).get());
    }

    @Override
    public void squareL2DistanceMultiScore(VectorFloat<?> v1, VectorFloat<?> v2, VectorFloat<?> results) {
        NativeSimdOps.square_distance_multi_f32_512(((MemorySegmentVectorFloat)v1).get(), ((MemorySegmentVectorFloat)v2).get(), v1.length(), results.length(), ((MemorySegmentVectorFloat)results).get());
    }

    @Override
    public float lvqDotProduct(VectorFloat<?> query, LocallyAdaptiveVectorQuantization.PackedVector vector, float querySum) {
        return VectorSimdOps.lvqDotProduct((MemorySegmentVectorFloat) query, vector, querySum);
    }

    @Override
    public float lvqSquareL2Distance(VectorFloat<?> query, LocallyAdaptiveVectorQuantization.PackedVector vector) {
        return VectorSimdOps.lvqSquareL2Distance((MemorySegmentVectorFloat) query, vector);
    }

    @Override
    public float lvqCosine(VectorFloat<?> query, LocallyAdaptiveVectorQuantization.PackedVector vector, VectorFloat<?> centroid) {
        return VectorSimdOps.lvqCosine((MemorySegmentVectorFloat) query, vector, (MemorySegmentVectorFloat) centroid);
    }


}
