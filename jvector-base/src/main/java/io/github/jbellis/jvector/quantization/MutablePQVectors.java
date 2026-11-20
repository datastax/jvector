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

import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.util.concurrent.atomic.AtomicInteger;

import static java.lang.Math.max;

/**
 * A threadsafe mutable PQVectors implementation that grows dynamically as needed.
 */
public class MutablePQVectors extends PQVectors implements MutableCompressedVectors<VectorFloat<?>> {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    private static final int VECTORS_PER_CHUNK = 1024;
    private static final int INITIAL_CHUNKS = 10;
    private static final float GROWTH_FACTOR = 1.5f;

    protected AtomicInteger vectorCount;
    // Written under the existing allocation lock, published to lock-free memory-limit readers.
    private volatile long retainedBytes;
    private int allocatedChunks;

    /**
     * Construct a mutable PQVectors instance with the given ProductQuantization.
     * The vectors storage will grow dynamically as needed.
     * @param pq the ProductQuantization to use
     */
    public MutablePQVectors(ProductQuantization pq) {
        super(pq);
        this.vectorCount = new AtomicInteger(0);
        this.vectorsPerChunk = VECTORS_PER_CHUNK;
        this.compressedDataChunks = new ByteSequence<?>[INITIAL_CHUNKS];
        retainedBytes = pq.ramBytesUsed() + RamUsageEstimator.shallowSizeOf(this)
                + RamUsageEstimator.shallowSizeOf(vectorCount)
                + RamUsageEstimator.shallowSizeOf(compressedDataChunks);
    }

    @Override
    public long ramBytesUsed() {
        return retainedBytes;
    }

    @Override
    public void encodeAndSet(int ordinal, VectorFloat<?> vector) {
        ensureChunkCapacity(ordinal);
        // increase count first so get() works
        vectorCount.updateAndGet(current -> max(current, ordinal + 1));
        pq.encodeTo(vector, get(ordinal));
    }

    @Override
    public void setZero(int ordinal) {
        ensureChunkCapacity(ordinal);
        // increase count first so get() works
        vectorCount.updateAndGet(current -> max(current, ordinal + 1));
        get(ordinal).zero();
    }

    private synchronized void ensureChunkCapacity(int ordinal) {
        if (ordinal < 0 || ordinal == Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Ordinal must be between 0 and Integer.MAX_VALUE - 1: " + ordinal);
        }
        int chunkOrdinal = ordinal / vectorsPerChunk;
        if (chunkOrdinal < allocatedChunks) return; // Existing storage: no accounting work.
        
        // Grow backing array if needed
        if (chunkOrdinal >= compressedDataChunks.length) {
            int newLength = max(chunkOrdinal + 1, (int)(compressedDataChunks.length * GROWTH_FACTOR));
            ByteSequence<?>[] newChunks = new ByteSequence<?>[newLength];
            System.arraycopy(compressedDataChunks, 0, newChunks, 0, compressedDataChunks.length);
            long growth = RamUsageEstimator.shallowSizeOf(newChunks)
                    - RamUsageEstimator.shallowSizeOf(compressedDataChunks);
            compressedDataChunks = newChunks;
            retainedBytes += growth;
        }

        // Allocate all chunks up to and including the required one
        int chunkBytes = Math.multiplyExact(VECTORS_PER_CHUNK, pq.compressedVectorSize());
        while (allocatedChunks <= chunkOrdinal) {
            var chunk = vectorTypeSupport.createByteSequence(chunkBytes);
            long bytes = chunk.ramBytesUsed();
            compressedDataChunks[allocatedChunks++] = chunk;
            retainedBytes += bytes;
        }
    }

    @Override
    protected int validChunkCount() {
        if (vectorCount.get() == 0)
            return 0;
        int chunkOrdinal = (vectorCount.get() - 1) / vectorsPerChunk;
        return chunkOrdinal + 1;
    }

    @Override
    public int count() {
        return vectorCount.get();
    }
}
