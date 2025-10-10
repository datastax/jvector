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

package io.github.jbellis.jvector.util;

/**
 * Interface for Bitset-like structures that provide read-only bit access.
 * <p>
 * This interface is used for filtering operations where bits represent the presence
 * or absence of elements. It provides constant instances for common cases and utility
 * methods for combining Bits instances.
 */
public interface Bits {
    /** A Bits instance where all bits are set. */
    Bits ALL = new MatchAllBits();

    /** A Bits instance where no bits are set. */
    Bits NONE = new MatchNoBits();

    /**
     * Returns the value of the bit with the specified <code>index</code>.
     *
     * @param index index, should be non-negative. The result of passing
     *     negative or out of bounds values is undefined by this interface, <b>just don't do it!</b>
     * @return <code>true</code> if the bit is set, <code>false</code> otherwise.
     */
    boolean get(int index);

    /**
     * Returns a Bits instance that is the inverse of the given Bits.
     * The result is {@code true} when {@code bits} is {@code false}, and vice versa.
     * @param bits the Bits to invert
     * @return a Bits instance representing the inverse
     */
    static Bits inverseOf(Bits bits) {
        return new Bits() {
            @Override
            public boolean get(int index) {
                return !bits.get(index);
            }
        };
    }

    /**
     * Returns a Bits instance representing the intersection of two Bits instances.
     * A bit is set in the result if and only if it is set in both {@code a} and {@code b}.
     * @param a the first Bits instance
     * @param b the second Bits instance
     * @return a Bits instance representing the intersection of {@code a} and {@code b}
     */
    static Bits intersectionOf(Bits a, Bits b) {
        if (a instanceof MatchAllBits) {
            return b;
        }
        if (b instanceof MatchAllBits) {
            return a;
        }

        if (a instanceof MatchNoBits) {
            return a;
        }
        if (b instanceof MatchNoBits) {
            return b;
        }

        return new Bits() {
            @Override
            public boolean get(int index) {
                return a.get(index) && b.get(index);
            }
        };
    }

    /**
     * A Bits implementation where all bits are set.
     */
    class MatchAllBits implements Bits {
        /** Creates a MatchAllBits instance. */
        public MatchAllBits() {}

        @Override
        public boolean get(int index) {
            return true;
        }
    }

    /**
     * A Bits implementation where no bits are set.
     */
    class MatchNoBits implements Bits {
        /** Creates a MatchNoBits instance. */
        public MatchNoBits() {}

        @Override
        public boolean get(int index) {
            return false;
        }
    }
}
