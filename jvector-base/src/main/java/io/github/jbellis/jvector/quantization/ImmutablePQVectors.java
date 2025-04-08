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

import io.github.jbellis.jvector.vector.types.ByteSequence;

public class ImmutablePQVectors extends PQVectors {
    private final int vectorCount;

    /**
     * Construct an immutable PQVectors instance with the given ProductQuantization and compressed data chunks.
     * @param pq the ProductQuantization to use
     * @param compressedDataChunks the compressed data chunks
     * @param vectorCount the number of vectors
     * @param vectorsPerChunk the number of vectors per chunk
     */
    public ImmutablePQVectors(ProductQuantization pq, ByteSequence<?>[] compressedDataChunks, int vectorCount, int vectorsPerChunk) {
        super(pq);
        this.compressedDataChunks = compressedDataChunks;
        this.vectorCount = vectorCount;
        this.vectorsPerChunk = vectorsPerChunk;
    }

    @Override
    protected int validChunkCount() {
        return compressedDataChunks.length;
    }

    @Override
    public int count() {
        return vectorCount;
    }

    // Getters for batch processing in Bench / Grid
    public ProductQuantization getPQ() {
        return pq;
    }

    public int getVectorsPerChunk() {
        return vectorsPerChunk;
    }

    public ByteSequence<?>[] getChunks() {
        return compressedDataChunks;
    }
}
