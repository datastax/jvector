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

import io.github.jbellis.jvector.annotations.VisibleForTesting;
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
import java.util.stream.IntStream;

public abstract class PQVectors implements CompressedVectors {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    static final int MAX_CHUNK_SIZE = Integer.MAX_VALUE - 16; // standard Java array size limit with some headroom

    final ProductQuantization pq;
    protected ByteSequence<?>[] compressedDataChunks;
    protected int vectorsPerChunk;

    protected PQVectors(ProductQuantization pq) {
        this.pq = pq;
    }

    public static ImmutablePQVectors load(RandomAccessReader in) throws IOException {
        // pq codebooks
        var pq = ProductQuantization.load(in);

        // read the vectors
        int vectorCount = in.readInt();
        int compressedDimension = in.readInt();

        int[] params = calculateChunkParameters(vectorCount, compressedDimension);
        int vectorsPerChunk = params[0];
        int totalChunks = params[1];
        int fullSizeChunks = params[2];
        int remainingVectors = params[3];

        ByteSequence<?>[] chunks = new ByteSequence<?>[totalChunks];
        int chunkBytes = vectorsPerChunk * compressedDimension;

        for (int i = 0; i < fullSizeChunks; i++) {
            chunks[i] = vectorTypeSupport.readByteSequence(in, chunkBytes);
        }

        // Last chunk might be smaller
        if (totalChunks > fullSizeChunks) {
            chunks[fullSizeChunks] = vectorTypeSupport.readByteSequence(in, remainingVectors * compressedDimension);
        }

        return new ImmutablePQVectors(pq, chunks, vectorCount, vectorsPerChunk);
    }

    /**
     * Calculate chunking parameters for the given vector count and compressed dimension
     * @return array of [vectorsPerChunk, totalChunks, fullSizeChunks, remainingVectors]
     */
    @VisibleForTesting
    static int[] calculateChunkParameters(int vectorCount, int compressedDimension) {
        if (vectorCount < 0) {
            throw new IllegalArgumentException("Invalid vector count " + vectorCount);
        }
        if (compressedDimension < 0) {
            throw new IllegalArgumentException("Invalid compressed dimension " + compressedDimension);
        }

        long totalSize = (long) vectorCount * compressedDimension;
        int vectorsPerChunk = totalSize <= MAX_CHUNK_SIZE ? vectorCount : MAX_CHUNK_SIZE / compressedDimension;
        if (vectorsPerChunk == 0) {
            throw new IllegalArgumentException("Compressed dimension " + compressedDimension + " too large for chunking");
        }

        int fullSizeChunks = vectorCount / vectorsPerChunk;
        int totalChunks = vectorCount % vectorsPerChunk == 0 ? fullSizeChunks : fullSizeChunks + 1;

        int remainingVectors = vectorCount % vectorsPerChunk;
        return new int[] {vectorsPerChunk, totalChunks, fullSizeChunks, remainingVectors};
    }

    public static PQVectors load(RandomAccessReader in, long offset) throws IOException {
        in.seek(offset);
        return load(in);
    }

    /**
     * Build a PQVectors instance from the given RandomAccessVectorValues. The vectors are encoded in parallel
     * and split into chunks to avoid exceeding the maximum array size.
     *
     * @param pq           the ProductQuantization to use
     * @param vectorCount  the number of vectors to encode
     * @param ravv         the RandomAccessVectorValues to encode
     * @param simdExecutor the ForkJoinPool to use for SIMD operations
     * @return the PQVectors instance
     */
    public static ImmutablePQVectors encodeAndBuild(ProductQuantization pq, int vectorCount, RandomAccessVectorValues ravv, ForkJoinPool simdExecutor) {
        // Calculate if we need to split into multiple chunks
        int compressedDimension = pq.compressedVectorSize();
        long totalSize = (long) vectorCount * compressedDimension;
        int vectorsPerChunk = totalSize <= PQVectors.MAX_CHUNK_SIZE ? vectorCount : PQVectors.MAX_CHUNK_SIZE / compressedDimension;

        int numChunks = vectorCount / vectorsPerChunk;
        final ByteSequence<?>[] chunks = new ByteSequence<?>[numChunks];
        int chunkSize = vectorsPerChunk * compressedDimension;
        for (int i = 0; i < numChunks - 1; i++)
            chunks[i] = vectorTypeSupport.createByteSequence(chunkSize);

        // Last chunk might be smaller
        int remainingVectors = vectorCount - (vectorsPerChunk * (numChunks - 1));
        chunks[numChunks - 1] = vectorTypeSupport.createByteSequence(remainingVectors * compressedDimension);

        // Encode the vectors in parallel into the compressed data chunks
        // The changes are concurrent, but because they are coordinated and do not overlap, we can use parallel streams
        // and then we are guaranteed safe publication because we join the thread after completion.
        var ravvCopy = ravv.threadLocalSupplier();
        simdExecutor.submit(() -> IntStream.range(0, ravv.size())
                        .parallel()
                        .forEach(ordinal -> {
                            // Retrieve the slice and mutate it.
                            var localRavv = ravvCopy.get();
                            var slice = PQVectors.get(chunks, ordinal, vectorsPerChunk, pq.getSubspaceCount());
                            var vector = localRavv.getVector(ordinal);
                            if (vector != null)
                                pq.encodeTo(vector, slice);
                            else
                                slice.zero();
                        }))
                .join();

        return new ImmutablePQVectors(pq, chunks, vectorCount, vectorsPerChunk);
    }

    @Override
    public void write(DataOutput out, int version) throws IOException
    {
        // pq codebooks
        pq.write(out, version);

        // compressed vectors
        out.writeInt(count());
        out.writeInt(pq.getSubspaceCount());
        for (int i = 0; i < validChunkCount(); i++) {
            vectorTypeSupport.writeByteSequence(out, compressedDataChunks[i]);
        }
    }

    /**
     * @return the number of chunks that have actually been allocated ({@code <= compressedDataChunks.length})
     */
    protected abstract int validChunkCount();

    /**
     * We consider two PQVectors equal when their PQs are equal and their compressed data is equal. We ignore the
     * chunking strategy in the comparison since this is an implementation detail.
     * @param o the object to check for equality
     * @return true if the objects are equal, false otherwise
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        PQVectors that = (PQVectors) o;
        if (!Objects.equals(pq, that.pq)) return false;
        if (this.count() != that.count()) return false;
        for (int i = 0; i < this.count(); i++) {
            var thisNode = this.get(i);
            var thatNode = that.get(i);
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
            result = 31 * result + get(i).hashCode();

        return result;
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction precomputedScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        switch (similarityFunction) {
            case DOT_PRODUCT:
                return new PQDecoder.DotProductDecoder(this, q);
            case EUCLIDEAN:
                return new PQDecoder.EuclideanDecoder(this, q);
            case COSINE:
                return new PQDecoder.CosineDecoder(this, q);
            default:
                throw new IllegalArgumentException("Unsupported similarity function " + similarityFunction);
        }
    }

    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        VectorFloat<?> centeredQuery = pq.globalCentroid == null ? q : VectorUtil.sub(q, pq.globalCentroid);
        switch (similarityFunction) {
            case DOT_PRODUCT:
                return (node2) -> {
                    var encodedChunk = getChunk(node2);
                    var encodedOffset = getOffsetInChunk(node2);
                    // compute the dot product of the query and the codebook centroids corresponding to the encoded points
                    float dp = 0;
                    for (int m = 0; m < pq.getSubspaceCount(); m++) {
                        int centroidIndex = Byte.toUnsignedInt(encodedChunk.get(m + encodedOffset));
                        int centroidLength = pq.subvectorSizesAndOffsets[m][0];
                        int centroidOffset = pq.subvectorSizesAndOffsets[m][1];
                        dp += VectorUtil.dotProduct(pq.codebooks[m], centroidIndex * centroidLength, centeredQuery, centroidOffset, centroidLength);
                    }
                    // scale to [0, 1]
                    return (1 + dp) / 2;
                };
            case COSINE:
                float norm1 = VectorUtil.dotProduct(centeredQuery, centeredQuery);
                return (node2) -> {
                    var encodedChunk = getChunk(node2);
                    var encodedOffset = getOffsetInChunk(node2);
                    // compute the dot product of the query and the codebook centroids corresponding to the encoded points
                    float sum = 0;
                    float norm2 = 0;
                    for (int m = 0; m < pq.getSubspaceCount(); m++) {
                        int centroidIndex = Byte.toUnsignedInt(encodedChunk.get(m + encodedOffset));
                        int centroidLength = pq.subvectorSizesAndOffsets[m][0];
                        int centroidOffset = pq.subvectorSizesAndOffsets[m][1];
                        var codebookOffset = centroidIndex * centroidLength;
                        sum += VectorUtil.dotProduct(pq.codebooks[m], codebookOffset, centeredQuery, centroidOffset, centroidLength);
                        norm2 += VectorUtil.dotProduct(pq.codebooks[m], codebookOffset, pq.codebooks[m], codebookOffset, centroidLength);
                    }
                    float cosine = sum / (float) Math.sqrt(norm1 * norm2);
                    // scale to [0, 1]
                    return (1 + cosine) / 2;
                };
            case EUCLIDEAN:
                return (node2) -> {
                    var encodedChunk = getChunk(node2);
                    var encodedOffset = getOffsetInChunk(node2);
                    // compute the euclidean distance between the query and the codebook centroids corresponding to the encoded points
                    float sum = 0;
                    for (int m = 0; m < pq.getSubspaceCount(); m++) {
                        int centroidIndex = Byte.toUnsignedInt(encodedChunk.get(m + encodedOffset));
                        int centroidLength = pq.subvectorSizesAndOffsets[m][0];
                        int centroidOffset = pq.subvectorSizesAndOffsets[m][1];
                        sum += VectorUtil.squareL2Distance(pq.codebooks[m], centroidIndex * centroidLength, centeredQuery, centroidOffset, centroidLength);
                    }
                    // scale to [0, 1]
                    return 1 / (1 + sum);
                };
            default:
                throw new IllegalArgumentException("Unsupported similarity function " + similarityFunction);
        }
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction diversityFunctionFor(int node1, VectorSimilarityFunction similarityFunction) {
        final int subspaceCount = pq.getSubspaceCount();
        var node1Chunk = getChunk(node1);
        var node1Offset = getOffsetInChunk(node1);

        switch (similarityFunction) {
            case DOT_PRODUCT:
                return (node2) -> {
                    var node2Chunk = getChunk(node2);
                    var node2Offset = getOffsetInChunk(node2);
                    // compute the euclidean distance between the query and the codebook centroids corresponding to the encoded points
                    float dp = 0;
                    for (int m = 0; m < subspaceCount; m++) {
                        int centroidIndex1 = Byte.toUnsignedInt(node1Chunk.get(m + node1Offset));
                        int centroidIndex2 = Byte.toUnsignedInt(node2Chunk.get(m + node2Offset));
                        int centroidLength = pq.subvectorSizesAndOffsets[m][0];
                        dp += VectorUtil.dotProduct(pq.codebooks[m], centroidIndex1 * centroidLength, pq.codebooks[m], centroidIndex2 * centroidLength, centroidLength);
                    }
                    // scale to [0, 1]
                    return (1 + dp) / 2;
                };
            case COSINE:
                float norm1 = 0.0f;
                for (int m1 = 0; m1 < subspaceCount; m1++) {
                    int centroidIndex = Byte.toUnsignedInt(node1Chunk.get(m1 + node1Offset));
                    int centroidLength = pq.subvectorSizesAndOffsets[m1][0];
                    var codebookOffset = centroidIndex * centroidLength;
                    norm1 += VectorUtil.dotProduct(pq.codebooks[m1], codebookOffset, pq.codebooks[m1], codebookOffset, centroidLength);
                }
                final float norm1final = norm1;
                return (node2) -> {
                    var node2Chunk = getChunk(node2);
                    var node2Offset = getOffsetInChunk(node2);
                    // compute the dot product of the query and the codebook centroids corresponding to the encoded points
                    float sum = 0;
                    float norm2 = 0;
                    for (int m = 0; m < subspaceCount; m++) {
                        int centroidIndex1 = Byte.toUnsignedInt(node1Chunk.get(m + node1Offset));
                        int centroidIndex2 = Byte.toUnsignedInt(node2Chunk.get(m + node2Offset));
                        int centroidLength = pq.subvectorSizesAndOffsets[m][0];
                        int codebookOffset = centroidIndex2 * centroidLength;
                        sum += VectorUtil.dotProduct(pq.codebooks[m], codebookOffset, pq.codebooks[m], centroidIndex1 * centroidLength, centroidLength);
                        norm2 += VectorUtil.dotProduct(pq.codebooks[m], codebookOffset, pq.codebooks[m], codebookOffset, centroidLength);
                    }
                    float cosine = sum / (float) Math.sqrt(norm1final * norm2);
                    // scale to [0, 1]
                    return (1 + cosine) / 2;
                };
            case EUCLIDEAN:
                return (node2) -> {
                    var node2Chunk = getChunk(node2);
                    var node2Offset = getOffsetInChunk(node2);
                    // compute the euclidean distance between the query and the codebook centroids corresponding to the encoded points
                    float sum = 0;
                    for (int m = 0; m < subspaceCount; m++) {
                        int centroidIndex1 = Byte.toUnsignedInt(node1Chunk.get(m + node1Offset));
                        int centroidIndex2 = Byte.toUnsignedInt(node2Chunk.get(m + node2Offset));
                        int centroidLength = pq.subvectorSizesAndOffsets[m][0];
                        sum += VectorUtil.squareL2Distance(pq.codebooks[m], centroidIndex1 * centroidLength, pq.codebooks[m], centroidIndex2 * centroidLength, centroidLength);
                    }
                    // scale to [0, 1]
                    return 1 / (1 + sum);
                };
            default:
                throw new IllegalArgumentException("Unsupported similarity function " + similarityFunction);
        }
    }

    /**
     * Returns a {@link ByteSequence} for the given ordinal.
     * @param ordinal the vector's ordinal
     * @return the {@link ByteSequence}
     */
    public ByteSequence<?> get(int ordinal) {
        if (ordinal < 0 || ordinal >= count())
            throw new IndexOutOfBoundsException("Ordinal " + ordinal + " out of bounds for vector count " + count());
        return get(compressedDataChunks, ordinal, vectorsPerChunk, pq.getSubspaceCount());
    }

    static ByteSequence<?> get(ByteSequence<?>[] chunks, int ordinal, int vectorsPerChunk, int subspaceCount) {
        int vectorIndexInChunk = ordinal % vectorsPerChunk;
        int start = vectorIndexInChunk * subspaceCount;
        return getChunk(chunks, ordinal, vectorsPerChunk).slice(start, subspaceCount);
    }

    /**
     * Returns a reference to the {@link ByteSequence} containing for the given ordinal. Only intended for use where
     * the caller wants to avoid an allocation for the slice object. After getting the chunk, callers should use the
     * {@link #getOffsetInChunk(int)} method to get the offset of the vector within the chunk and then use the pq's
     * {@link ProductQuantization#getSubspaceCount()} to get the length of the vector.
     * @param ordinal the vector's ordinal
     * @return the {@link ByteSequence} chunk containing the vector
     */
    ByteSequence<?> getChunk(int ordinal) {
        if (ordinal < 0 || ordinal >= count())
            throw new IndexOutOfBoundsException("Ordinal " + ordinal + " out of bounds for vector count " + count());

        return getChunk(compressedDataChunks, ordinal, vectorsPerChunk);
    }

    int getOffsetInChunk(int ordinal) {
        if (ordinal < 0 || ordinal >= count())
            throw new IndexOutOfBoundsException("Ordinal " + ordinal + " out of bounds for vector count " + count());

        int vectorIndexInChunk = ordinal % vectorsPerChunk;
        return vectorIndexInChunk * pq.getSubspaceCount();
    }

    static ByteSequence<?> getChunk(ByteSequence<?>[] chunks, int ordinal, int vectorsPerChunk) {
        int chunkIndex = ordinal / vectorsPerChunk;
        return chunks[chunkIndex];
    }


    VectorFloat<?> reusablePartialSums() {
        return pq.reusablePartialSums();
    }

    AtomicReference<VectorFloat<?>> partialSquaredMagnitudes() {
        return pq.partialSquaredMagnitudes();
    }

    @Override
    public int getOriginalSize() {
        return pq.originalDimension * Float.BYTES;
    }

    @Override
    public int getCompressedSize() {
        return pq.compressedVectorSize();
    }

    @Override
    public ProductQuantization getCompressor() {
        return pq;
    }

    @Override
    public long ramBytesUsed() {
        int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        int OH_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        int AH_BYTES = RamUsageEstimator.NUM_BYTES_ARRAY_HEADER;

        long codebooksSize = pq.ramBytesUsed();
        long chunksArraySize = OH_BYTES + AH_BYTES + (long) validChunkCount() * REF_BYTES;
        long dataSize = 0;
        for (int i = 0; i < validChunkCount(); i++) {
            dataSize += compressedDataChunks[i].ramBytesUsed();
        }
        return codebooksSize + chunksArraySize + dataSize;
    }

    @Override
    public String toString() {
        return "PQVectors{" +
                "pq=" + pq +
                ", count=" + count() +
                '}';
    }
}
