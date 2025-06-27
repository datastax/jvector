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
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * {@code MappedChunkReader} provides random access reading of large files using memory-mapped I/O,
 * supporting files larger than 2GB by mapping them in manageable chunks.
 * <p>
 * This class implements {@link RandomAccessReader} and allows reading primitive types and arrays
 * from a file channel, handling chunk remapping transparently.
 * <p>
 * This class is intended as a replacement for {@link SimpleMappedReader} to provide the capability
 * to handle files larger than 2GB in size regardless of the OS or JDK in use.
 * </p>
 */
public class MappedChunkReader implements RandomAccessReader {
    private static final long CHUNK_SIZE = Integer.MAX_VALUE; // ~2GB
    private final FileChannel channel;
    private final long fileSize;
    private final ByteOrder byteOrder;
    private long position;

    private ByteBuffer currentBuffer;
    private long currentChunkStart;

    /**
     * Constructs a new {@code MappedChunkReader} for the given file channel and byte order.
     *
     * @param channel   the file channel to read from
     * @param byteOrder the byte order to use for reading
     * @throws IOException if an I/O error occurs
     */
    public MappedChunkReader(FileChannel channel, ByteOrder byteOrder) throws IOException {
        this.channel = channel;
        this.byteOrder = byteOrder;
        this.fileSize = channel.size();
        this.position = 0;
        mapChunk(0);
    }

    /**
     * {@code Supplier} is a factory for creating {@link MappedChunkReader} instances
     * from a given file path.
     */
    public static class Supplier implements ReaderSupplier {
        private final FileChannel channel;

        /**
         * Opens a file channel for the specified path in read-only mode.
         *
         * @param path the path to the file
         * @throws IOException if an I/O error occurs
         */
        public Supplier(Path path) throws IOException {
            this.channel = FileChannel.open(path, StandardOpenOption.READ);
        }

        /**
         * Returns a new {@link MappedChunkReader} using the opened file channel.
         *
         * @return a new {@code MappedChunkReader}
         */
        @Override
        public RandomAccessReader get() {
            try {
                return new MappedChunkReader(channel, ByteOrder.BIG_ENDIAN);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        /**
         * Closes the underlying file channel.
         *
         * @throws IOException if an I/O error occurs
         */
        @Override
        public void close() throws IOException {
            channel.close();
        }
    }

    /**
     * Maps a chunk of the file into memory starting at the specified offset.
     *
     * @param chunkStart the start offset of the chunk
     * @throws IOException if an I/O error occurs
     */
    private void mapChunk(long chunkStart) throws IOException {
        long size = Math.min(CHUNK_SIZE, fileSize - chunkStart);
        currentBuffer = channel.map(FileChannel.MapMode.READ_ONLY, chunkStart, size).order(byteOrder);
        currentChunkStart = chunkStart;
    }

    /**
     * Ensures that the specified number of bytes are available in the current buffer,
     * remapping a new chunk if necessary.
     *
     * @param size the number of bytes required
     * @throws IOException if an I/O error occurs
     */
    private void ensureAvailable(int size) throws IOException {
        if (position < currentChunkStart || position + size > currentChunkStart + currentBuffer.capacity()) {
            mapChunk((position / CHUNK_SIZE) * CHUNK_SIZE);
        }
        currentBuffer.position((int)(position - currentChunkStart));
    }

    /**
     * Sets the current read position in the file.
     *
     * @param offset the new position
     */
    @Override
    public void seek(long offset) {
        this.position = offset;
    }

    /**
     * Returns the current read position in the file.
     *
     * @return the current position
     */
    @Override
    public long getPosition() {
        return position;
    }

    /**
     * Reads a 4-byte integer from the current position.
     *
     * @return the integer value read
     * @throws RuntimeException if an I/O error occurs
     */
    @Override
    public int readInt() {
        try {
            ensureAvailable(4);
            int v = currentBuffer.getInt();
            position += 4;
            return v;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Reads an 8-byte long from the current position.
     *
     * @return the long value read
     * @throws RuntimeException if an I/O error occurs
     */
    @Override
    public long readLong() {
        try {
            ensureAvailable(8);
            long v = currentBuffer.getLong();
            position += 8;
            return v;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Reads a 4-byte float from the current position.
     *
     * @return the float value read
     * @throws RuntimeException if an I/O error occurs
     */
    @Override
    public float readFloat() {
        try {
            ensureAvailable(4);
            float v = currentBuffer.getFloat();
            position += 4;
            return v;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Reads bytes into the provided array, filling it completely.
     *
     * @param b the byte array to fill
     * @throws RuntimeException if an I/O error occurs
     */
    @Override
    public void readFully(byte[] b) {
        try {
            int offset = 0;
            while (offset < b.length) {
                ensureAvailable(1);
                int toRead = Math.min(b.length - offset, currentBuffer.remaining());
                currentBuffer.get(b, offset, toRead);
                offset += toRead;
                position += toRead;
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Reads bytes into the provided {@link ByteBuffer}, filling it completely.
     *
     * @param buffer the buffer to fill
     * @throws RuntimeException if an I/O error occurs
     */
    @Override
    public void readFully(ByteBuffer buffer) {
        try {
            while (buffer.hasRemaining()) {
                ensureAvailable(1);
                int toRead = Math.min(buffer.remaining(), currentBuffer.remaining());
                ByteBuffer slice = currentBuffer.slice();
                slice.limit(toRead);
                buffer.put(slice);
                currentBuffer.position(currentBuffer.position() + toRead);
                position += toRead;
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Reads long values into the provided array, filling it completely.
     *
     * @param vector the array to fill with long values
     * @throws RuntimeException if an I/O error occurs
     */
    @Override
    public void readFully(long[] vector) {
        ByteBuffer tmp = ByteBuffer.allocate(vector.length * Long.BYTES).order(byteOrder);
        readFully(tmp);
        tmp.flip().asLongBuffer().get(vector);
    }

    /**
     * Reads integer values into the provided array at the specified offset.
     *
     * @param ints   the array to fill with integer values
     * @param offset the starting offset in the array
     * @param count  the number of integers to read
     * @throws RuntimeException if an I/O error occurs
     */
    @Override
    public void read(int[] ints, int offset, int count) {
        ByteBuffer tmp = ByteBuffer.allocate(count * Integer.BYTES).order(byteOrder);
        readFully(tmp);
        tmp.flip().asIntBuffer().get(ints, offset, count);
    }

    /**
     * Reads float values into the provided array at the specified offset.
     *
     * @param floats the array to fill with float values
     * @param offset the starting offset in the array
     * @param count  the number of floats to read
     * @throws RuntimeException if an I/O error occurs
     */
    @Override
    public void read(float[] floats, int offset, int count) {
        ByteBuffer tmp = ByteBuffer.allocate(count * Float.BYTES).order(byteOrder);
        readFully(tmp);
        tmp.flip().asFloatBuffer().get(floats, offset, count);
    }

    /**
     * Returns the total length of the file.
     *
     * @return the file size in bytes
     */
    @Override
    public long length() {
        return fileSize;
    }

    /**
     * Closes this reader. The underlying channel is managed by {@link Supplier} and is not closed here.
     */
    @Override
    public void close() {
        // Channel is managed by Supplier
    }
}
