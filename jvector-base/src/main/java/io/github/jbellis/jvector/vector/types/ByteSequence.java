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
 * A generic interface for accessing and manipulating byte sequences backed by various storage types.
 * <p>
 * This interface provides a uniform abstraction over different byte storage implementations,
 * allowing efficient access to byte data through a common API. The storage type {@code T}
 * represents the underlying backing storage (e.g., byte arrays, direct memory buffers, etc.).
 * <p>
 * Implementations support:
 * <ul>
 *   <li>Random access to individual bytes via {@link #get(int)} and {@link #set(int, byte)}</li>
 *   <li>Little-endian short operations via {@link #setLittleEndianShort(int, short)}</li>
 *   <li>Bulk operations including {@link #copyFrom(ByteSequence, int, int, int)} and {@link #zero()}</li>
 *   <li>Sequence slicing and copying for efficient memory management</li>
 *   <li>Value-based equality comparison through {@link #equalTo(Object)}</li>
 * </ul>
 * <p>
 * ByteSequence is designed to be used in performance-critical contexts where direct byte
 * manipulation is required, such as vector operations and low-level data processing.
 *
 * @param <T> the type of the backing storage
 */
public interface ByteSequence<T> extends Accountable
{
    /**
     * Returns the entire backing storage for this byte sequence.
     * <p>
     * The returned object represents the underlying storage implementation,
     * which may be a byte array, ByteBuffer, or other storage mechanism
     * depending on the concrete implementation.
     *
     * @return the backing storage object of type {@code T}
     */
    T get();

    /**
     * Returns the offset within the backing storage where this sequence begins.
     * <p>
     * This offset is used in conjunction with {@link #length()} to define the
     * valid range of bytes in the backing storage that belong to this sequence.
     * For a sequence that starts at the beginning of its backing storage, this
     * method returns 0.
     *
     * @return the starting offset in bytes, zero-based
     */
    int offset();

    /**
     * Returns the number of bytes in this sequence.
     * <p>
     * Valid indices for {@link #get(int)} and {@link #set(int, byte)} operations
     * range from 0 (inclusive) to the value returned by this method (exclusive).
     *
     * @return the length of this sequence in bytes
     */
    int length();

    /**
     * Returns the byte value at the specified index within this sequence.
     * <p>
     * The index is relative to the beginning of this sequence, not the underlying
     * backing storage. Valid indices range from 0 to {@link #length()} - 1.
     *
     * @param i the index of the byte to retrieve, zero-based
     * @return the byte value at the specified index
     * @throws IndexOutOfBoundsException if the index is negative or greater than or equal to {@link #length()}
     */
    byte get(int i);

    /**
     * Sets the byte value at the specified index within this sequence.
     * <p>
     * The index is relative to the beginning of this sequence, not the underlying
     * backing storage. Valid indices range from 0 to {@link #length()} - 1.
     *
     * @param i the index where the byte should be set, zero-based
     * @param value the byte value to set
     * @throws IndexOutOfBoundsException if the index is negative or greater than or equal to {@link #length()}
     */
    void set(int i, byte value);

    /**
     * Sets a short value in little-endian byte order at the specified index.
     * <p>
     * This method treats the byte sequence as an array of shorts, where each short
     * occupies 2 bytes. The {@code shortIndex} parameter specifies which short position
     * to write to (e.g., shortIndex=0 writes to bytes 0-1, shortIndex=1 writes to bytes 2-3).
     * The value is stored in little-endian format (least significant byte first).
     *
     * @param shortIndex the index in short positions (not bytes) where the value should be set
     * @param value the short value to set in little-endian byte order
     * @throws IndexOutOfBoundsException if the short position would exceed the sequence bounds
     */
    void setLittleEndianShort(int shortIndex, short value);

    /**
     * Sets all bytes in this sequence to zero.
     * <p>
     * This method efficiently clears the entire byte sequence by writing zero to each
     * position from 0 to {@link #length()} - 1.
     */
    void zero();

    /**
     * Copies bytes from another ByteSequence into this sequence.
     * <p>
     * This method performs a bulk copy operation, transferring {@code length} bytes
     * from the source sequence starting at {@code srcOffset} to this sequence starting
     * at {@code destOffset}. The source and destination may use different backing storage
     * types, as indicated by the wildcard parameter type.
     * <p>
     * The source and destination regions must not overlap if both sequences share the
     * same backing storage. The behavior of overlapping copies is implementation-dependent.
     *
     * @param src the source ByteSequence to copy from
     * @param srcOffset the starting offset in the source sequence
     * @param destOffset the starting offset in this sequence
     * @param length the number of bytes to copy
     * @throws IndexOutOfBoundsException if the copy operation would read beyond the source
     *         sequence bounds or write beyond this sequence bounds
     * @throws NullPointerException if {@code src} is null
     */
    void copyFrom(ByteSequence<?> src, int srcOffset, int destOffset, int length);

    /**
     * Creates an independent copy of this ByteSequence.
     * <p>
     * The returned sequence contains the same byte values as this sequence but uses
     * a separate backing storage. Modifications to the copy will not affect this
     * sequence, and vice versa. The copy has the same length as the original.
     *
     * @return a new ByteSequence containing a copy of this sequence's data
     */
    ByteSequence<T> copy();

    /**
     * Creates a new ByteSequence that represents a subsequence of this sequence.
     * <p>
     * The returned slice shares the same backing storage as this sequence but has
     * different offset and length values. This allows efficient sub-sequence access
     * without copying data. Modifications to the slice will affect the original
     * sequence and vice versa.
     * <p>
     * The slice's valid byte range starts at the specified {@code offset} within
     * this sequence and extends for {@code length} bytes.
     *
     * @param offset the starting position within this sequence for the slice
     * @param length the number of bytes to include in the slice
     * @return a new ByteSequence view representing the specified subsequence
     * @throws IndexOutOfBoundsException if {@code offset} is negative, {@code length}
     *         is negative, or {@code offset + length} exceeds this sequence's length
     */
    ByteSequence<T>  slice(int offset, int length);

    /**
     * Compares this ByteSequence to another object for byte-wise equality.
     * <p>
     * Two ByteSequences are considered equal if and only if:
     * <ul>
     *   <li>They have the same {@link #length()}</li>
     *   <li>They contain the same byte value at each corresponding position</li>
     * </ul>
     * <p>
     * This method performs value-based comparison rather than reference equality.
     * It can compare ByteSequences with different backing storage types, as it only
     * examines the logical byte content.
     * <p>
     * Note: This is a utility method for value comparison. Implementations should
     * not override {@code Object.equals()} with this logic to maintain proper
     * collection behavior if needed.
     *
     * @param o the object to compare to, which may be any type
     * @return {@code true} if {@code o} is a ByteSequence with identical length
     *         and byte content; {@code false} otherwise
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
     * Computes a hash code for this ByteSequence based on its byte content.
     * <p>
     * The hash code is calculated by iterating through all bytes in the sequence
     * and combining their values using a standard polynomial rolling hash algorithm
     * (multiplying by 31 and adding each non-zero byte value). This ensures that:
     * <ul>
     *   <li>ByteSequences with identical content produce the same hash code</li>
     *   <li>Zero bytes are optimized out to improve performance for sparse sequences</li>
     *   <li>The hash code is consistent with {@link #equalTo(Object)}</li>
     * </ul>
     * <p>
     * Note: Like {@link #equalTo(Object)}, this is a utility method. Implementations
     * should not override {@code Object.hashCode()} with this logic to maintain proper
     * collection behavior if needed.
     *
     * @return a hash code value for this ByteSequence based on its content
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
