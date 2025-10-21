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
    private enum SIMDVersion {
        SIMD512,
        SIMD256,
        SIMD128
    }

    private final SIMDVersion SIMD_VERSION;

    public NativeVectorUtilSupport() {
        int version = 0; // NativeSimdOps.simd_version();

        // This mapping is defined in vector_simd.h
        switch (version) {
            case 0 -> // SSE
                    SIMD_VERSION = SIMDVersion.SIMD128;
            case 1 -> // AVX2
                    SIMD_VERSION = SIMDVersion.SIMD256;
            case 2 -> // AVX512
                    SIMD_VERSION = SIMDVersion.SIMD512;
            default -> throw new UnsupportedOperationException("Unsupported SIMD version: " + version);
        }
    }

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
    public float assembleAndSumPQ(
            VectorFloat<?> codebookPartialSums,
            int subspaceCount,                  // = M
            ByteSequence<?> vector1Ordinals,
            int vector1OrdinalOffset,
            ByteSequence<?> vector2Ordinals,
            int vector2OrdinalOffset,
            int clusterCount                    // = k
    ) {
        //Use the non-panama solution for now
        return assembleAndSumPQ_128(codebookPartialSums, subspaceCount, vector1Ordinals, vector1OrdinalOffset, vector2Ordinals, vector2OrdinalOffset, clusterCount);
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

    // These block sizes are tied to the SIMD implementations of the methods in jvector_simd.h,
    // which are found in jvector_sse.c, jvector_avx2.c, and jvector_avx512.c.
    // Since the methods using this from Java are not in the most performance critical path,
    // it is just easier not going through the JNI barrier.
    private int getBlockSize() {
        int blockSize;
        switch (SIMD_VERSION) {
            case SIMDVersion.SIMD128 ->
                    blockSize = 8;
            case SIMDVersion.SIMD256 ->
                    blockSize = 16;
            case SIMDVersion.SIMD512 ->
                    blockSize = 32;
            default -> throw new UnsupportedOperationException("Unsupported SIMD version: " + SIMD_VERSION);
        }
        return blockSize;
    }

    // This implementation stores pqCode in compressedNeighbors using a block-transposed layout.
    // The block size is determined by the SIMD width.
    @Override
    public void storePQCodeInNeighbors(ByteSequence<?> pqCode, int position, ByteSequence<?> compressedNeighbors) {
        int blockSize = getBlockSize();

        int blockIndex = position / blockSize;
        int offset = blockIndex * blockSize * pqCode.length();
        int positionWithinBlock = position % blockSize;
        for (int j = 0; j < pqCode.length(); j++) {
            compressedNeighbors.set(offset + blockSize * j + positionWithinBlock, pqCode.get(j));
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
    public void bulkShuffleRawSimilarity(ByteSequence<?> shuffles, int codebookCount, VectorFloat<?> partials, VectorFloat<?> results) {
        int blockSize = getBlockSize();

        for (int j = 0; j < results.length(); j++) {
            int blockIndex = j / blockSize;
            int offset = blockIndex * blockSize * codebookCount;
            int positionWithinBlock = j % blockSize;
            for (int i = 0; i < codebookCount; i++) {
                int singleShuffle = Byte.toUnsignedInt(shuffles.get(offset + blockSize * i + positionWithinBlock));
                results.set(j, results.get(j) + partials.get(i * codebookCount + singleShuffle));
            }
        }
    }

    @Override
    public void bulkShuffleRawSimilarityCosine(ByteSequence<?> shuffles, int codebookCount,
                                               VectorFloat<?> partialSums,
                                               VectorFloat<?> partialSquaredMagnitudes,
                                               float[] resultSumAggregates, float[] resultMagnitudeAggregates) {
        int blockSize = getBlockSize();

        for (int j = 0; j < partialSums.length(); j++) {
            int blockIndex = j / blockSize;
            int offset = blockIndex * blockSize * codebookCount;
            int positionWithinBlock = j % blockSize;
            for (int i = 0; i < codebookCount; i++) {
                int singleShuffle = Byte.toUnsignedInt(shuffles.get(offset + blockSize * i + positionWithinBlock));
                resultSumAggregates[j] += partialSums.get(i * codebookCount + singleShuffle);
                resultMagnitudeAggregates[j] += partialSquaredMagnitudes.get(i * codebookCount + singleShuffle);
            }
        }
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
