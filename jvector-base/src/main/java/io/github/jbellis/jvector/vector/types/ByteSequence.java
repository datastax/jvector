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

package io.github.jbellis.jvector.vector.types;

import io.github.jbellis.jvector.util.Accountable;
import java.util.Objects;

/**
 * A sequence of bytes with offset and length.
 * @param <T> the T type parameter
 */
public interface ByteSequence<T> extends Accountable
{
    /**
     * Gets the entire sequence backing storage.
     * @return entire sequence backing storage
     */
    T get();

    /**
     * Gets the offset.
     * @return the offset
     */
    int offset();

    /**
     * Gets the length.
     * @return the length
     */
    int length();

    /**
     * Gets the byte at the specified index.
     * @param i the index
     * @return the byte
     */
    byte get(int i);

    /**
     * Sets the byte at the specified index.
     * @param i the index
     * @param value the value
     */
    void set(int i, byte value);

    /**
     * Sets a short value in little-endian format.
     * @param shortIndex index (as if this was a short array) inside the sequence to set the short value
     * @param value short value to set
     */
    void setLittleEndianShort(int shortIndex, short value);

    /**
     * Zeroes the sequence.
     */
    void zero();

    /**
     * Copies from another sequence.
     * @param src the source sequence
     * @param srcOffset the source offset
     * @param destOffset the destination offset
     * @param length the length
     */
    void copyFrom(ByteSequence<?> src, int srcOffset, int destOffset, int length);

    /**
     * Creates a copy of the sequence.
     * @return the copy
     */
    ByteSequence<T> copy();

    /**
     * Creates a slice of the sequence.
     * @param offset the offset
     * @param length the length
     * @return the slice
     */
    ByteSequence<T>  slice(int offset, int length);

    /**
     * Two ByteSequences are equal if they have the same length and the same bytes at each position.
     * @param o the other object to compare to
     * @return true if the two ByteSequences are equal
     */
    default boolean equalTo(Object o) {
        if (this == o) return true;
        if (!(o instanceof ByteSequence)) return false;
        ByteSequence<?> that = (ByteSequence<?>) o;
        if (length() != that.length()) return false;
        for (int i = 0; i < length(); i++) {
            if (get(i) != that.get(i)) return false;
        }
        return true;
    }

    /**
     * Computes a hash code for this ByteSequence.
     * @return a hash code for this ByteSequence
     */
    default int getHashCode() {
        int result = 1;
        for (int i = 0; i < length(); i++) {
            if (get(i) != 0) {
                result = 31 * result + get(i);
            }
        }
        return result;
    }
}
