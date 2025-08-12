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

import java.nio.ByteOrder;

import io.github.jbellis.jvector.vector.cnative.NativeSimdOps;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

/**
 * VectorUtilSupport implementation that prefers native/Panama SIMD.
 */
final class NativeVectorUtilSupport extends PanamaVectorUtilSupport
{
    @Override
    protected FloatVector fromVectorFloat(VectorSpecies<Float> SPEC, VectorFloat<?> vector, int offset)
    {
        return FloatVector.fromMemorySegment(SPEC, ((MemorySegmentVectorFloat) vector).get(), vector.offset(offset), ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    protected FloatVector fromVectorFloat(VectorSpecies<Float> SPEC, VectorFloat<?> vector, int offset, int[] indices, int indicesOffset)
    {
        throw new UnsupportedOperationException("Assembly not supported with memory segments.");
    }

    @Override
    protected void intoVectorFloat(FloatVector vector, VectorFloat<?> v, int offset)
    {
        vector.intoMemorySegment(((MemorySegmentVectorFloat) v).get(), v.offset(offset), ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    protected ByteVector fromByteSequence(VectorSpecies<Byte> SPEC, ByteSequence<?> vector, int offset)
    {
        return ByteVector.fromMemorySegment(SPEC, ((MemorySegmentByteSequence) vector).get(), offset, ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    protected void intoByteSequence(ByteVector vector, ByteSequence<?> v, int offset)
    {
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
        return NativeSimdOps.assemble_and_sum_f32_512(((MemorySegmentVectorFloat) data).get(), dataBase, ((MemorySegmentByteSequence) baseOffsets).get(), baseOffsetsOffset, baseOffsetsLength);
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
    public void calculatePartialSums(VectorFloat<?> codebook, int codebookBase, int size, int clusterCount, VectorFloat<?> query, int queryOffset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums, VectorFloat<?> partialBestDistances) {
        switch (vsf) {
            case DOT_PRODUCT -> NativeSimdOps.calculate_partial_sums_best_dot_f32_512(((MemorySegmentVectorFloat)codebook).get(), codebookBase, size, clusterCount, ((MemorySegmentVectorFloat)query).get(), queryOffset, ((MemorySegmentVectorFloat)partialSums).get(), ((MemorySegmentVectorFloat)partialBestDistances).get());
            case EUCLIDEAN -> NativeSimdOps.calculate_partial_sums_best_euclidean_f32_512(((MemorySegmentVectorFloat)codebook).get(), codebookBase, size, clusterCount, ((MemorySegmentVectorFloat)query).get(), queryOffset, ((MemorySegmentVectorFloat)partialSums).get(), ((MemorySegmentVectorFloat)partialBestDistances).get());
            case COSINE -> throw new UnsupportedOperationException("Cosine similarity not supported for calculatePartialSums");
        }
    }

    @Override
    public void bulkShuffleQuantizedSimilarity(ByteSequence<?> shuffles, int codebookCount, ByteSequence<?> quantizedPartials, float delta, float bestDistance, VectorSimilarityFunction vsf, VectorFloat<?> results) {
        assert shuffles.offset() == 0 : "Bulk shuffle shuffles are expected to have an offset of 0. Found: " + shuffles.offset();
        switch (vsf) {
            case DOT_PRODUCT -> NativeSimdOps.bulk_quantized_shuffle_dot_f32_512(((MemorySegmentByteSequence) shuffles).get(), codebookCount, ((MemorySegmentByteSequence) quantizedPartials).get(), delta, bestDistance, ((MemorySegmentVectorFloat) results).get());
            case EUCLIDEAN -> NativeSimdOps.bulk_quantized_shuffle_euclidean_f32_512(((MemorySegmentByteSequence) shuffles).get(), codebookCount, ((MemorySegmentByteSequence) quantizedPartials).get(), delta, bestDistance, ((MemorySegmentVectorFloat) results).get());
            case COSINE -> throw new UnsupportedOperationException("Cosine similarity not supported for bulkShuffleQuantizedSimilarity");
        }
    }

    @Override
    public void bulkShuffleQuantizedSimilarityCosine(ByteSequence<?> shuffles, int codebookCount,
                                                     ByteSequence<?> quantizedPartialSums, float sumDelta, float minDistance,
                                                     ByteSequence<?> quantizedPartialSquaredMagnitudes, float magnitudeDelta, float minMagnitude,
                                                     float queryMagnitudeSquared, VectorFloat<?> results) {
        assert shuffles.offset() == 0 : "Bulk shuffle shuffles are expected to have an offset of 0. Found: " + shuffles.offset();
        NativeSimdOps.bulk_quantized_shuffle_cosine_f32_512(((MemorySegmentByteSequence) shuffles).get(), codebookCount, ((MemorySegmentByteSequence) quantizedPartialSums).get(), sumDelta, minDistance,
                ((MemorySegmentByteSequence) quantizedPartialSquaredMagnitudes).get(), magnitudeDelta, minMagnitude, queryMagnitudeSquared, ((MemorySegmentVectorFloat) results).get());
    }

    @Override
    public float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude)
    {
        return pqDecodedCosineSimilarity(encoded, 0, encoded.length(), clusterCount, partialSums, aMagnitude, bMagnitude);
    }

    @Override
    public float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int encodedOffset, int encodedLength, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude)
    {
        assert encoded.offset() == 0 : "Bulk shuffle shuffles are expected to have an offset of 0. Found: " + encoded.offset();
        // encoded is a pointer into a PQ chunk - we need to index into it by encodedOffset and provide encodedLength to the native code
        return NativeSimdOps.pq_decoded_cosine_similarity_f32_512(((MemorySegmentByteSequence) encoded).get(), encodedOffset, encodedLength, clusterCount, ((MemorySegmentVectorFloat) partialSums).get(), ((MemorySegmentVectorFloat) aMagnitude).get(), bMagnitude);
    }
}
