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

import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * A threadsafe mutable BQVectors implementation that grows dynamically as needed.
 */
@SuppressWarnings("unused")
public class MutableBQVectors extends BQVectors implements MutableCompressedVectors<VectorFloat<?>> {
    private static final int INITIAL_CAPACITY = 1024;
    private static final float GROWTH_FACTOR = 1.5f;
    
    protected final AtomicInteger vectorCount;

    /**
     * Construct a mutable BQVectors instance with the given BinaryQuantization.
     * The vectors storage will grow dynamically as needed.
     * @param bq the BinaryQuantization to use
     */
    public MutableBQVectors(BinaryQuantization bq) {
        super(bq);
        this.compressedVectors = new long[INITIAL_CAPACITY][];
        this.vectorCount = new AtomicInteger(0);
    }

    // Growth copies the array, so the store must happen under the same lock or a concurrent copy can drop it.
    private synchronized void set(int ordinal, long[] encoded) {
        if (ordinal < 0 || ordinal == Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Ordinal must be between 0 and Integer.MAX_VALUE - 1: " + ordinal);
        }
        if (ordinal >= compressedVectors.length) {
            int newCapacity = Math.max(ordinal + 1, (int)(compressedVectors.length * GROWTH_FACTOR));
            long[][] newVectors = new long[newCapacity][];
            System.arraycopy(compressedVectors, 0, newVectors, 0, compressedVectors.length);
            compressedVectors = newVectors;
        }
        compressedVectors[ordinal] = encoded;
        vectorCount.accumulateAndGet(ordinal + 1, Math::max);
    }

    @Override
    public void encodeAndSet(int ordinal, VectorFloat<?> vector) {
        set(ordinal, bq.encode(vector));
    }

    @Override
    public void setZero(int ordinal) {
        set(ordinal, new long[bq.compressedVectorSize() / Long.BYTES]);
    }

    @Override
    public int count() {
        return vectorCount.get();
    }
}
