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

package io.github.jbellis.jvector.disk;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * This is a subset of DataInput, plus seek and readFully methods, which allows implementations
 * to use more efficient options like FloatBuffer for bulk reads.
 * <p>
 * JVector includes production-ready implementations; the recommended way to use these are via
 * `ReaderSupplierFactory.open`.  For custom implementations, e.g. reading from network storage,
 * you should also implement a corresponding `ReaderSupplier`.
 * <p>
 * The general usage pattern is expected to be "seek to a position, then read sequentially from there."
 * Thus, RandomAccessReader implementations are expected to be stateful and NOT threadsafe; JVector
 * uses the ReaderSupplier API to create a RandomAccessReader per thread, as needed.
 */
public interface RandomAccessReader extends AutoCloseable {
    /**
     * Seeks to the specified position in the reader.
     * @param offset the offset position to seek to
     * @throws IOException if an I/O error occurs
     */
    void seek(long offset) throws IOException;

    /**
     * Returns the current position in the reader.
     * @return the current position in the reader
     * @throws IOException if an I/O error occurs
     */
    long getPosition() throws IOException;

    /**
     * Reads and returns a 32-bit integer.
     * @return the integer value read
     * @throws IOException if an I/O error occurs
     */
    int readInt() throws IOException;

    /**
     * Reads and returns a 32-bit float.
     * @return the float value read
     * @throws IOException if an I/O error occurs
     */
    float readFloat() throws IOException;

    /**
     * Reads and returns a 64-bit long.
     * @return the long value read
     * @throws IOException if an I/O error occurs
     */
    long readLong() throws IOException;

    /**
     * Reads bytes to completely fill the specified byte array.
     * @param bytes the byte array to fill
     * @throws IOException if an I/O error occurs
     */
    void readFully(byte[] bytes) throws IOException;

    /**
     * Reads bytes to completely fill the specified ByteBuffer.
     * @param buffer the ByteBuffer to fill
     * @throws IOException if an I/O error occurs
     */
    void readFully(ByteBuffer buffer) throws IOException;

    /**
     * Reads floats to completely fill the specified float array.
     * @param floats the float array to fill
     * @throws IOException if an I/O error occurs
     */
    default void readFully(float[] floats) throws IOException {
        read(floats, 0, floats.length);
    }

    /**
     * Reads longs to completely fill the specified long array.
     * @param vector the long array to fill
     * @throws IOException if an I/O error occurs
     */
    void readFully(long[] vector) throws IOException;

    /**
     * Reads a specified number of integers into an array starting at the given offset.
     * @param ints the array to read integers into
     * @param offset the starting position in the array
     * @param count the number of integers to read
     * @throws IOException if an I/O error occurs
     */
    void read(int[] ints, int offset, int count) throws IOException;

    /**
     * Reads a specified number of floats into an array starting at the given offset.
     * @param floats the array to read floats into
     * @param offset the starting position in the array
     * @param count the number of floats to read
     * @throws IOException if an I/O error occurs
     */
    void read(float[] floats, int offset, int count) throws IOException;

    /**
     * Closes this reader and releases any system resources associated with it.
     * @throws IOException if an I/O error occurs
     */
    void close() throws IOException;

    /**
     * Returns the length of the reader slice.
     * @return the length of the reader slice
     * @throws IOException if an I/O error occurs
     */
    long length() throws IOException;
}
