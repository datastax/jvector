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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

public class QuickerADCVectors {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    final ProductQuantization pq;
    private ByteSequence<?>[] compressedDataChunks;
    private final PQLayout layout;

    private QuickerADCVectors(ProductQuantization pq, ByteSequence<?>[] chunks, PQLayout layout) {
        this.pq = pq;
        this.compressedDataChunks = chunks;
        this.layout = layout;
    }

    /**
     * Build a PQVectors instance from the given RandomAccessVectorValues. The vectors are encoded in parallel
     * and split into chunks to avoid exceeding the maximum array size.
     * <p>
     * This is a helper method for the special case where the ordinals mapping in the graph and the RAVV/PQVectors are the same.
     *
     * @param pq           the ProductQuantization to use
     * @param vectorCount  the number of vectors to encode
     * @param ravv         the RandomAccessVectorValues to encode
     * @param simdExecutor the ForkJoinPool to use for SIMD operations
     * @return the PQVectors instance
     */
    public static QuickerADCVectors encodeAndBuild(ProductQuantization pq, int vectorCount, RandomAccessVectorValues ravv, ForkJoinPool simdExecutor) {
        return encodeAndBuild(pq, vectorCount, IntUnaryOperator.identity(), ravv, simdExecutor);
    }

    /**
     * Build a PQVectors instance from the given RandomAccessVectorValues. The vectors are encoded in parallel
     * and split into chunks to avoid exceeding the maximum array size.
     *
     * @param pq           the ProductQuantization to use
     * @param vectorCount  the number of vectors to encode
     * @param ordinalsMapping the graph ordinals to RAVV mapping, the function should be defined in [0, vectorCount)
     * @param ravv         the RandomAccessVectorValues to encode
     * @param simdExecutor the ForkJoinPool to use for SIMD operations
     * @return the PQVectors instance
     */
    public static QuickerADCVectors encodeAndBuild(ProductQuantization pq, int vectorCount, IntUnaryOperator ordinalsMapping, RandomAccessVectorValues ravv, ForkJoinPool simdExecutor) {
        int compressedDimension = pq.compressedVectorSize();
        PQLayout layout = new PQLayout(vectorCount, compressedDimension);
        final ByteSequence<?>[] chunks = new ByteSequence<?>[layout.totalChunks];
        for (int i = 0; i < layout.fullSizeChunks; i++) {
            chunks[i] = vectorTypeSupport.createByteSequence(layout.fullChunkBytes);
        }
        if (layout.lastChunkVectors > 0) {
            chunks[layout.fullSizeChunks] = vectorTypeSupport.createByteSequence(layout.lastChunkBytes);
        }

        // Encode the vectors in parallel into the compressed data chunks
        // The changes are concurrent, but because they are coordinated and do not overlap, we can use parallel streams
        // and then we are guaranteed safe publication because we join the thread after completion.
        ThreadLocal<ByteSequence<?>> codes = ThreadLocal.withInitial(() -> vectorTypeSupport.createByteSequence(pq.getSubspaceCount()));
        var ravvCopy = ravv.threadLocalSupplier();
        simdExecutor.submit(() -> IntStream.range(0, vectorCount)
                        .parallel()
                        .forEach(ordinal -> {
                            // Retrieve the slice and mutate it.
                            var localRavv = ravvCopy.get();
                            var compressedVector = codes.get();
                            var vector = localRavv.getVector(ordinalsMapping.applyAsInt(ordinal));
                            pq.encodeTo(vector, compressedVector);

                            int chunkIndex = ordinal / layout.fullChunkVectors;
                            int vectorIndexInChunk = ordinal % layout.fullChunkVectors;

                            for (int j = 0; j < compressedVector.length(); j++) {
                                chunks[chunkIndex].set(j * layout.fullChunkVectors + vectorIndexInChunk, compressedVector.get(j));
                            }
                        }))
                .join();

        return new QuickerADCVectors(pq, chunks, layout);
    }

    public int count() {
        return layout.vectorCount;
    }

    public int countPerChunk() {
        return layout.fullChunkVectors;
    }

    public ByteSequence<?>[] getChunks() {
        return compressedDataChunks;
    }

    /**
     * We consider two QuickerADCVectors equal if the underlying chunks are equal.
     * @param o the object to check for equality
     * @return true if the objects are equal, false otherwise
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        QuickerADCVectors that = (QuickerADCVectors) o;
        if (!Objects.equals(pq, that.pq)) return false;
        if (this.count() != that.count()) return false;
        for (int i = 0; i < this.compressedDataChunks.length; i++) {
            var thisNode = this.compressedDataChunks[i];
            var thatNode = that.compressedDataChunks[i];
            if (!thisNode.equals(thatNode)) return false;
        }
        return true;
    }

    @Override
    public int hashCode() {
        int result = 1;
        result = 31 * result + pq.hashCode();
        result = 31 * result + count();

        // We don't use the array structure in the hash code calculation because we allow for different chunking
        // strategies. Instead, we use the first entry in the first chunk to provide a stable hash code.
        for (int i = 0; i < count(); i++)
            result = 31 * result + compressedDataChunks[i].hashCode();

        return result;
    }

    public QuickerADCDecoder precomputedScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
            return QuickerADCDecoder.newDecoder(this, q, similarityFunction);
    }

    public int getOriginalSize() {
        return pq.originalDimension * Float.BYTES;
    }

    public int getCompressedSize() {
        return pq.compressedVectorSize();
    }

    public ProductQuantization getCompressor() {
        return pq;
    }

    public long ramBytesUsed() {
        int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        int OH_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        int AH_BYTES = RamUsageEstimator.NUM_BYTES_ARRAY_HEADER;

        long codebooksSize = pq.ramBytesUsed();
        long chunksArraySize = OH_BYTES + AH_BYTES + (long) compressedDataChunks.length * REF_BYTES;
        long dataSize = 0;
        for (int i = 0; i < compressedDataChunks.length; i++) {
            dataSize += compressedDataChunks[i].ramBytesUsed();
        }
        return codebooksSize + chunksArraySize + dataSize;
    }

    @Override
    public String toString() {
        return "QuickerADCVectors{" +
                "pq=" + pq +
                ", count=" + count() +
                '}';
    }

    /**
     * Chunk Dimensions and Layout
     * This is emulative of modern Java records, but keeps to J11 standards.
     * This class consolidates the layout calculations for PQ data into one place
     */
    static class PQLayout {

        /**
         * total number of vectors
         **/
        public final int vectorCount;
        /**
         * total number of chunks, including any partial
         **/
        public final int totalChunks;
        /**
         * total number of fully-filled chunks
         **/
        public final int fullSizeChunks;
        /**
         * number of vectors per fullSize chunk
         **/
        public final int fullChunkVectors;
        /**
         * number of vectors in last partially filled chunk, if any
         **/
        public final int lastChunkVectors;
        /**
         * compressed dimension of vectors
         **/
        public final int compressedDimension;
        /**
         * number of bytes in each fully-filled chunk
         **/
        public final int fullChunkBytes;
        /**
         * number of bytes in the last partially-filled chunk, if any
         **/
        public final int lastChunkBytes;

        public PQLayout(int vectorCount, int compressedDimension) {
            if (vectorCount <= 0) {
                throw new IllegalArgumentException("Invalid vector count " + vectorCount);
            }
            this.vectorCount = vectorCount;

            if (compressedDimension <= 0) {
                throw new IllegalArgumentException("Invalid compressed dimension " + compressedDimension);
            }
            this.compressedDimension = compressedDimension;

            // Get the aligned number of bytes needed to hold a given dimension
            // purely for overflow prevention
            int layoutBytesPerVector = compressedDimension == 1 ? 1 : Integer.highestOneBit(compressedDimension - 1) << 1;
            // truncation welcome here, biasing for smaller chunks
            int addressableVectorsPerChunk = Integer.MAX_VALUE / layoutBytesPerVector;
//            int addressableVectorsPerChunk = 500;

            fullChunkVectors = Math.min(vectorCount, addressableVectorsPerChunk);
            lastChunkVectors = vectorCount % fullChunkVectors;

            fullChunkBytes = fullChunkVectors * compressedDimension;
            lastChunkBytes = lastChunkVectors * compressedDimension;

            fullSizeChunks = vectorCount / fullChunkVectors;
            totalChunks = fullSizeChunks + (lastChunkVectors == 0 ? 0 : 1);
        }

    }
}
