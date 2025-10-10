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
     * Seek to an offset in the reader.
     * @param offset the offset
     * @throws IOException if an error occurs
     */
    void seek(long offset) throws IOException;

    /**
     * Get the current position in the reader.
     * @return the current position
     * @throws IOException if an error occurs
     */
    long getPosition() throws IOException;

    /**
     * Read an int.
     * @return the int
     * @throws IOException if an error occurs
     */
    int readInt() throws IOException;

    /**
     * Read a float.
     * @return the float
     * @throws IOException if an error occurs
     */
    float readFloat() throws IOException;

    /**
     * Read a long.
     * @return the long
     * @throws IOException if an error occurs
     */
    long readLong() throws IOException;

    /**
     * Read fully into a byte array.
     * @param bytes the byte array
     * @throws IOException if an error occurs
     */
    void readFully(byte[] bytes) throws IOException;

    /**
     * Read fully into a ByteBuffer.
     * @param buffer the ByteBuffer
     * @throws IOException if an error occurs
     */
    void readFully(ByteBuffer buffer) throws IOException;

    /**
     * Read fully into a float array.
     * @param floats the float array
     * @throws IOException if an error occurs
     */
    default void readFully(float[] floats) throws IOException {
        read(floats, 0, floats.length);
    }

    /**
     * Read fully into a long array.
     * @param vector the long array
     * @throws IOException if an error occurs
     */
    void readFully(long[] vector) throws IOException;

    /**
     * Read into an int array.
     * @param ints the int array
     * @param offset the offset
     * @param count the count
     * @throws IOException if an error occurs
     */
    void read(int[] ints, int offset, int count) throws IOException;

    /**
     * Read into a float array.
     * @param floats the float array
     * @param offset the offset
     * @param count the count
     * @throws IOException if an error occurs
     */
    void read(float[] floats, int offset, int count) throws IOException;

    /**
     * Close the reader.
     * @throws IOException if an error occurs
     */
    void close() throws IOException;

    /**
     * Length of the reader slice.
     * @return the length
     * @throws IOException if an error occurs
     */
    long length() throws IOException;
}
