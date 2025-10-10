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
 * Interface for Bitset-like structures.
 */
public interface Bits {
    /** Bits with all bits set. */
    Bits ALL = new MatchAllBits();
    /** Bits with no bits set. */
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
     * Returns a Bits that is true when `bits` is false, and false when `bits` is true
     * @param bits the bits to invert
     * @return the inverted bits
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
     * Return a Bits that is set for a given ordinal iff both it is set in both `a` and `b`.
     * @param a the first bits
     * @param b the second bits
     * @return the intersection of a and b
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

    /** Bits with all bits set. */
    class MatchAllBits implements Bits {
        /** Creates a MatchAllBits instance. */
        MatchAllBits() {}

        @Override
        public boolean get(int index) {
            return true;
        }
    }

    /** Bits with no bits set. */
    class MatchNoBits implements Bits {
        /** Creates a MatchNoBits instance. */
        MatchNoBits() {}

        @Override
        public boolean get(int index) {
            return false;
        }
    }
}
