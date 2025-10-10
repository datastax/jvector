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

package io.github.jbellis.jvector.util;

import java.util.concurrent.atomic.AtomicLongArray;

/**
 * A {@link BitSet} implementation that offers concurrent reads and writes through an {@link
 * AtomicLongArray} as bit storage.
 * <p>
 * For now, only implements the minimum functionality required by BitSet
 * (in contrast to the extra methods in FixedBitSet).
 */
public class AtomicFixedBitSet extends BitSet {
    private static final long BASE_RAM_BYTES_USED =
            RamUsageEstimator.shallowSizeOfInstance(AtomicFixedBitSet.class);

    private final AtomicLongArray storage;

    /**
     * Create an AtomicFixedBitSet with the specified number of bits.
     * @param numBits the number of bits
     */
    public AtomicFixedBitSet(int numBits) {
        int numLongs = (numBits + 63) >>> 6;
        storage = new AtomicLongArray(numLongs);
    }

    private static int index(int bit) {
        return bit >> 6;
    }

    private static long mask(int bit) {
        return 1L << bit;
    }

    @Override
    public int length() {
        return storage.length() << 6;
    }

    @Override
    public void set(int i) {
        int idx = index(i);
        long mask = mask(i);
        storage.getAndAccumulate(idx, mask, (prev, m) -> prev | m);
    }

    @Override
    public boolean get(int i) {
        if (i >= length()) {
            return false;
        }
        int idx = index(i);
        long mask = mask(i);
        long value = storage.get(idx);
        return (value & mask) != 0;
    }

    @Override
    public boolean getAndSet(int i) {
        int idx = index(i);
        long mask = mask(i);
        long prev = storage.getAndAccumulate(idx, mask, (p, m) -> p | m);
        return (prev & mask) != 0;
    }

    @Override
    public void clear() {
        for (int i = 0; i < storage.length(); i++) {
            storage.set(i, 0L);
        }
    }

    @Override
    public void clear(int i) {
        if (i >= length()) {
            return;
        }
        int idx = index(i);
        long mask = mask(i);
        storage.getAndAccumulate(idx, mask, (prev, m) -> prev & ~m);
    }

    @Override
    public void clear(int startIndex, int endIndex) {
        if (endIndex <= startIndex) {
            return;
        }

        int startIdx = index(startIndex);
        int endIdx = index(endIndex - 1);

        long startMask = -1L << (startIndex & 63);
        long endMask = -1L >>> -(endIndex & 63);

        // Invert masks since we are clearing
        startMask = ~startMask;
        endMask = ~endMask;

        if (startIdx == endIdx) {
            storage.getAndAccumulate(startIdx, startMask | endMask, (prev, m) -> prev & m);
            return;
        }

        storage.getAndAccumulate(startIdx, startMask, (prev, m) -> prev & m);
        for (int i = startIdx + 1; i < endIdx; i++) {
            storage.set(i, 0L);
        }
        storage.getAndAccumulate(endIdx, endMask, (prev, m) -> prev & m);
    }

    @Override
    public int cardinality() {
        int count = 0;
        for (int i = 0; i < storage.length(); i++) {
            count += Long.bitCount(storage.get(i));
        }
        return count;
    }

    @Override
    public int approximateCardinality() {
        return cardinality();
    }

    @Override
    public int prevSetBit(int index) {
        assert index >= 0 && index < length() : "index=" + index + " length=" + length();
        int i = index(index);
        final int subIndex = index & 63; // index within the word
        long word = (storage.get(i) << (63 - subIndex)); // skip all the bits to the left of index

        if (word != 0) {
            return (i << 6) + subIndex - Long.numberOfLeadingZeros(word);
        }

        while (--i >= 0) {
            word = storage.get(i);
            if (word != 0) {
                return (i << 6) + 63 - Long.numberOfLeadingZeros(word);
            }
        }

        return -1;
    }

    @Override
    public int nextSetBit(int index) {
        assert index >= 0 && index < length() : "index=" + index + ", length=" + length();

        int i = index(index);

        if (i >= storage.length()) {
            return DocIdSetIterator.NO_MORE_DOCS;
        }

        long word = storage.get(i) & (-1L << (index & 63)); // Mask all the bits to the right of index

        while (true) {
            if (word != 0) {
                return (i << 6) + Long.numberOfTrailingZeros(word);
            }
            if (++i >= storage.length()) {
                return DocIdSetIterator.NO_MORE_DOCS;
            }
            word = storage.get(i);
        }
    }

    @Override
    public long ramBytesUsed() {
        final int longSizeInBytes = Long.BYTES;
        final int arrayOverhead = 16; // Estimated overhead of AtomicLongArray object in bytes
        long storageSize = (long) storage.length() * longSizeInBytes + arrayOverhead;
        return BASE_RAM_BYTES_USED + storageSize;
    }

    /**
     * Create a copy of this AtomicFixedBitSet.
     * @return a copy of this bitset
     */
    public AtomicFixedBitSet copy() {
        AtomicFixedBitSet copy = new AtomicFixedBitSet(length());
        for (int i = 0; i < storage.length(); i++) {
            copy.storage.set(i, storage.get(i));
        }
        return copy;
    }
}

