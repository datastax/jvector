/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.graph;

import org.agrona.collections.IntHashSet;

import java.util.Arrays;

/**
 * Marks visited ordinals with a dense bitset for small id spaces.
 * Large id spaces, including graphs that grow past the dense budget, use a hash set.
 */
final class VisitedSet implements ImmutableGraphIndex.IntMarker {
    // Bound both the per-searcher dense allocation and the work of clearing it to 2 KiB.
    private static final int MAX_DENSE_BITS = 1 << 14;
    private static final int MAX_DENSE_WORDS = MAX_DENSE_BITS >>> 6;

    private long[] words;
    private IntHashSet sparse;
    private int highestWord = -1;

    VisitedSet(int upperBound) {
        if (upperBound >= 0 && upperBound <= MAX_DENSE_BITS) {
            words = new long[(upperBound + 63) >>> 6];
        } else {
            sparse = new IntHashSet();
        }
    }

    @Override
    public boolean mark(int value) {
        return words == null ? sparse.add(value) : markDense(value);
    }

    private boolean markDense(final int value) {
        if (value < 0) {
            return this.switchToSparse().add(value);
        }

        final int word = value >>> 6;
        if (word >= this.words.length) {
            if (word >= MAX_DENSE_WORDS) {
                return this.switchToSparse().add(value);
            }
            this.words = Arrays.copyOf(
                this.words,
                Math.min(MAX_DENSE_WORDS, Math.max(word + 1, this.words.length << 1))
            );
        }

        final long mask = 1L << value;
        if ((this.words[word] & mask) != 0L) {
            return false;
        }
        this.words[word] |= mask;
        if (word > this.highestWord) {
            this.highestWord = word;
        }
        return true;
    }

    void clear() {
        if (this.words != null) {
            Arrays.fill(this.words, 0, this.highestWord + 1, 0L);
            this.highestWord = -1;
        } else {
            this.sparse.clear();
        }
    }

    /** For tests: whether the dense words have been allocated. */
    boolean isDense() {
        return this.words != null;
    }

    private IntHashSet switchToSparse() {
        final IntHashSet set = new IntHashSet();
        for (int word = 0; word <= this.highestWord; word++) {
            long bits = this.words[word];
            while (bits != 0L) {
                final int bit = Long.numberOfTrailingZeros(bits);
                set.add((word << 6) + bit);
                bits &= bits - 1L;
            }
        }
        this.words = null;
        this.sparse = set;
        return set;
    }
}
