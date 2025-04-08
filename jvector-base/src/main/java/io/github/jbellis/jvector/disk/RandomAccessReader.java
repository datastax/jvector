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
    void seek(long offset) throws IOException;

    long getPosition() throws IOException;

    int readInt() throws IOException;

    float readFloat() throws IOException;

    long readLong() throws IOException;

    void readFully(byte[] bytes) throws IOException;

    void readFully(ByteBuffer buffer) throws IOException;

    default void readFully(float[] floats) throws IOException {
        read(floats, 0, floats.length);
    }

    void readFully(long[] vector) throws IOException;

    void read(int[] ints, int offset, int count) throws IOException;

    void read(float[] floats, int offset, int count) throws IOException;

    void close() throws IOException;
}
