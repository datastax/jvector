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

/**
 * A mutable implementation of BQVectors that supports dynamic growth as vectors are added.
 * Storage expands automatically using a configurable growth factor.
 */
@SuppressWarnings("unused")
public class MutableBQVectors extends BQVectors implements MutableCompressedVectors<VectorFloat<?>> {
    private static final int INITIAL_CAPACITY = 1024;
    /**
     * Growth factor used when expanding the internal storage arrays.
     */
    private static final float GROWTH_FACTOR = 1.5f;

    /**
     * The current number of vectors stored in this collection.
     */
    protected int vectorCount;

    /**
     * Construct a mutable BQVectors instance with the given BinaryQuantization.
     * The vectors storage will grow dynamically as needed.
     * @param bq the BinaryQuantization to use
     */
    public MutableBQVectors(BinaryQuantization bq) {
        super(bq);
        this.compressedVectors = new long[INITIAL_CAPACITY][];
        this.vectorCount = 0;
    }

    private void ensureCapacity(int ordinal) {
        if (ordinal >= compressedVectors.length) {
            int newCapacity = Math.max(ordinal + 1, (int)(compressedVectors.length * GROWTH_FACTOR));
            long[][] newVectors = new long[newCapacity][];
            System.arraycopy(compressedVectors, 0, newVectors, 0, compressedVectors.length);
            compressedVectors = newVectors;
        }
    }

    @Override
    public void encodeAndSet(int ordinal, VectorFloat<?> vector) {
        ensureCapacity(ordinal);
        compressedVectors[ordinal] = bq.encode(vector);
        vectorCount = Math.max(vectorCount, ordinal + 1);
    }

    @Override
    public void setZero(int ordinal) {
        ensureCapacity(ordinal);
        compressedVectors[ordinal] = new long[bq.compressedVectorSize()];
        vectorCount = Math.max(vectorCount, ordinal + 1);
    }

    @Override
    public int count() {
        return vectorCount;
    }
}
