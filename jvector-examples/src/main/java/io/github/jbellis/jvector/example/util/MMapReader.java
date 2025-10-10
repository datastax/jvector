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
package io.github.jbellis.jvector.example.util;

import com.indeed.util.mmap.MMapBuffer;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;

/**
 * Memory-mapped implementation of RandomAccessReader that provides efficient
 * random access to data stored in a file. Uses memory mapping for fast I/O operations.
 */
@SuppressWarnings("unused")
public class MMapReader implements RandomAccessReader {
    /** The memory-mapped buffer for reading data. */
    private final MMapBuffer buffer;
    /** The current read position in the buffer. */
    private long position;
    /** Reusable scratch buffer for bulk read operations. */
    private byte[] scratch = new byte[0];

    /**
     * Constructs a MMapReader wrapping the specified memory-mapped buffer.
     *
     * @param buffer the memory-mapped buffer to read from
     */
    MMapReader(MMapBuffer buffer) {
        this.buffer = buffer;
    }

    @Override
    public void seek(long offset) {
        position = offset;
    }

    @Override
    public long getPosition() {
        return position;
    }

    public int readInt() {
        try {
            return buffer.memory().getInt(position);
        } finally {
            position += Integer.BYTES;
        }
    }

    @Override
    public long readLong() {
        try {
            return buffer.memory().getLong(position);
        } finally {
            position += Long.BYTES;
        }
    }

    @Override
    public float readFloat() {
        try {
            return buffer.memory().getFloat(position);
        } finally {
            position += Float.BYTES;
        }
    }

    public void readFully(byte[] bytes) {
        read(bytes, 0, bytes.length);
    }

    public void readFully(ByteBuffer buffer) {
        var length = buffer.remaining();
        try {
            this.buffer.memory().getBytes(position, buffer);
        } finally {
            position += length;
        }
    }

    private void read(byte[] bytes, int offset, int count) {
        try {
            buffer.memory().getBytes(position, bytes, offset, count);
        } finally {
            position += count;
        }
    }

    @Override
    public void readFully(float[] floats) {
        int bytesToRead = floats.length * Float.BYTES;
        if (scratch.length < bytesToRead) {
            scratch = new byte[bytesToRead];
        }
        read(scratch, 0, bytesToRead);
        ByteBuffer byteBuffer = ByteBuffer.wrap(scratch).order(ByteOrder.BIG_ENDIAN);
        byteBuffer.asFloatBuffer().get(floats);
    }

    @Override
    public void read(int[] ints, int offset, int count) {
        int bytesToRead = count * Integer.BYTES;
        if (scratch.length < bytesToRead) {
            scratch = new byte[bytesToRead];
        }
        read(scratch, 0, bytesToRead);
        ByteBuffer byteBuffer = ByteBuffer.wrap(scratch).order(ByteOrder.BIG_ENDIAN);
        byteBuffer.asIntBuffer().get(ints, offset, count);
    }

    @Override
    public void read(float[] floats, int offset, int count) {
        int bytesToRead = count * Float.BYTES;
        if (scratch.length < bytesToRead) {
            scratch = new byte[bytesToRead];
        }
        read(scratch, 0, bytesToRead);
        ByteBuffer byteBuffer = ByteBuffer.wrap(scratch).order(ByteOrder.BIG_ENDIAN);
        byteBuffer.asFloatBuffer().get(floats, offset, count);
    }

    @Override
    public void readFully(long[] vector) {
        int bytesToRead = vector.length * Long.BYTES;
        if (scratch.length < bytesToRead) {
            scratch = new byte[bytesToRead];
        }
        read(scratch, 0, bytesToRead);
        ByteBuffer byteBuffer = ByteBuffer.wrap(scratch).order(ByteOrder.BIG_ENDIAN);
        byteBuffer.asLongBuffer().get(vector);
    }

    @Override
    public long length() {
        return buffer.memory().length();
    }

    @Override
    public void close() {
        // don't close buffer, let the Supplier handle that
    }

    /**
     * Supplier that creates MMapReader instances from a memory-mapped file.
     * The file is mapped into memory once during construction and shared across all readers.
     */
    public static class Supplier implements ReaderSupplier {
        /** The shared memory-mapped buffer for this file. */
        private final MMapBuffer buffer;

        /**
         * Constructs a Supplier that memory-maps the file at the specified path.
         *
         * @param path the path to the file to map
         * @throws IOException if an I/O error occurs during mapping
         */
        public Supplier(Path path) throws IOException {
            buffer = new MMapBuffer(path, FileChannel.MapMode.READ_ONLY, ByteOrder.BIG_ENDIAN);
        }

        @Override
        public RandomAccessReader get() {
            return new MMapReader(buffer);
        }

        @Override
        public void close() throws IOException {
            buffer.close();
        }
    }
}
