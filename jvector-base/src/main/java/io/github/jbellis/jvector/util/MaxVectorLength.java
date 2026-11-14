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

import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * The largest length of a set of vectors, recorded as they are added, from many threads at once. Dot-product searches
 * that stop adaptively bound their scores with it. It is in one of three states:
 *
 * <ul>
 * <li>none recorded yet: {@link #get} is NaN, and the first recorded length sets it;</li>
 * <li>known: {@link #get} is the largest length recorded, and longer lengths raise it;</li>
 * <li>unknown: {@link #get} is positive infinity, and it stays so. Use it when vectors exist whose lengths were not
 * recorded, such as vectors encoded by a model loaded from a format that did not store their largest length; a
 * length recorded later could not bound them.</li>
 * </ul>
 *
 * <p>{@link #get} and {@link #restore} round-trip all three states through one float, for storage.
 */
public final class MaxVectorLength {
    private static final int NONE = -1;
    private static final int UNKNOWN = Float.floatToIntBits(Float.POSITIVE_INFINITY);

    // The largest squared length as float bits: nonnegative floats order like their bits, NONE orders below every
    // length and UNKNOWN above, so one compare-and-set loop keeps the maximum in every state.
    private final AtomicInteger squaredBits = new AtomicInteger(NONE);

    /** Records a vector's length. */
    public void record(VectorFloat<?> vector) {
        recordSquared(VectorUtil.dotProduct(vector, vector));
    }

    /** Records a squared length; a NaN length makes the largest length unknown. */
    public void recordSquared(float squaredLength) {
        int bits = Float.isNaN(squaredLength) ? UNKNOWN : Float.floatToIntBits(Math.max(0f, squaredLength));
        int current;
        while (bits > (current = squaredBits.get())) {
            if (squaredBits.compareAndSet(current, bits)) return;
        }
    }

    /** Makes the largest length unknown for good. */
    public void markUnknown() {
        squaredBits.set(UNKNOWN);
    }

    /** The largest length recorded: NaN if none was, positive infinity if unknown. */
    public float get() {
        int bits = squaredBits.get();
        return bits == NONE ? Float.NaN : (float) Math.sqrt(Float.intBitsToFloat(bits));
    }

    /** Sets the state from a value {@link #get} returned, as when loading a saved one. */
    public void restore(float stored) {
        squaredBits.set(Float.isNaN(stored) ? NONE : Float.floatToIntBits(Math.abs(stored) * Math.abs(stored)));
    }
}
