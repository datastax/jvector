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

import io.github.jbellis.jvector.annotations.Experimental;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicLongArray;

/**
 * A fixed-size uniform sample of the ordinals offered to it, kept as you go (reservoir sampling), for training
 * a model on a stream without holding any vectors. The sample is of ordinals only: a caller reads the sampled
 * vectors from wherever it keeps them, when it trains.
 *
 * <p>Offering is constant time and lock-free, and may be done from many threads at once. Under concurrent
 * offers the sample stays a sample of offered ordinals, but is not exactly uniform: two offers may replace the
 * same slot. That does not matter for training.</p>
 */
@Experimental
public final class OrdinalReservoir implements Accountable {
    private static final long EMPTY = -1L;

    private final AtomicLongArray slots;
    private final AtomicLong offered = new AtomicLong();

    /** A reservoir that keeps up to {@code capacity} ordinals. */
    public OrdinalReservoir(int capacity) {
        if (capacity <= 0) throw new IllegalArgumentException("capacity must be positive");
        slots = new AtomicLongArray(capacity);
        for (int i = 0; i < capacity; i++) slots.set(i, EMPTY);
    }

    /** Offers one ordinal: kept while the reservoir is filling, then kept with probability capacity / offered. */
    public void offer(int ordinal) {
        if (ordinal < 0) throw new IllegalArgumentException("ordinal must be nonnegative");
        long seen = offered.getAndIncrement();
        int capacity = slots.length();
        if (seen < capacity) {
            slots.set((int) seen, ordinal);
            return;
        }
        long j = ThreadLocalRandom.current().nextLong(seen + 1);
        if (j < capacity) slots.set((int) j, ordinal);
    }

    /** The number of ordinals offered so far. */
    public long offered() {
        return offered.get();
    }

    /** The most ordinals this reservoir keeps. */
    public int capacity() {
        return slots.length();
    }

    /** The sampled ordinals, sorted and without repeats. */
    public int[] sample() {
        int n = (int) Math.min(slots.length(), offered.get());
        int[] out = new int[n];
        int count = 0;
        for (int i = 0; i < n; i++) {
            long v = slots.get(i);
            if (v != EMPTY) out[count++] = (int) v;
        }
        out = Arrays.copyOf(out, count);
        Arrays.sort(out);
        int unique = 0;
        for (int i = 0; i < out.length; i++) if (i == 0 || out[i] != out[i - 1]) out[unique++] = out[i];
        return Arrays.copyOf(out, unique);
    }

    @Override
    public long ramBytesUsed() {
        return RamUsageEstimator.shallowSizeOfInstance(OrdinalReservoir.class)
                + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER + (long) Long.BYTES * slots.length()
                + RamUsageEstimator.shallowSizeOfInstance(AtomicLong.class);
    }
}
