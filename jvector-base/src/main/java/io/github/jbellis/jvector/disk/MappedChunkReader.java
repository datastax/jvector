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

public class MappedChunkReader implements RandomAccessReader {
    private static final long CHUNK_SIZE = Integer.MAX_VALUE; // ~2GB
    private final FileChannel channel;
    private final long fileSize;
    private final ByteOrder byteOrder;
    private long position;

    private ByteBuffer currentBuffer;
    private long currentChunkStart;

    public MappedChunkReader(FileChannel channel, ByteOrder byteOrder) throws IOException {
        this.channel = channel;
        this.byteOrder = byteOrder;
        this.fileSize = channel.size();
        this.position = 0;
        mapChunk(0);
    }

    public static class Supplier implements ReaderSupplier {
        private final FileChannel channel;

        public Supplier(Path path) throws IOException {
            this.channel = FileChannel.open(path, StandardOpenOption.READ);
        }

        @Override
        public RandomAccessReader get() {
            try {
                return new MappedChunkReader(channel, ByteOrder.BIG_ENDIAN);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public void close() throws IOException {
            channel.close();
        }
    }

    private void mapChunk(long chunkStart) throws IOException {
        long size = Math.min(CHUNK_SIZE, fileSize - chunkStart);
        currentBuffer = channel.map(FileChannel.MapMode.READ_ONLY, chunkStart, size).order(byteOrder);
        currentChunkStart = chunkStart;
    }

    private void ensureAvailable(int size) throws IOException {
        if (position < currentChunkStart || position + size > currentChunkStart + currentBuffer.capacity()) {
            mapChunk((position / CHUNK_SIZE) * CHUNK_SIZE);
        }
        currentBuffer.position((int)(position - currentChunkStart));
    }

    @Override
    public void seek(long offset) {
        this.position = offset;
    }

    @Override
    public long getPosition() {
        return position;
    }

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

    @Override
    public void readFully(long[] vector) {
        ByteBuffer tmp = ByteBuffer.allocate(vector.length * Long.BYTES).order(byteOrder);
        readFully(tmp);
        tmp.flip().asLongBuffer().get(vector);
    }

    @Override
    public void read(int[] ints, int offset, int count) {
        ByteBuffer tmp = ByteBuffer.allocate(count * Integer.BYTES).order(byteOrder);
        readFully(tmp);
        tmp.flip().asIntBuffer().get(ints, offset, count);
    }

    @Override
    public void read(float[] floats, int offset, int count) {
        ByteBuffer tmp = ByteBuffer.allocate(count * Float.BYTES).order(byteOrder);
        readFully(tmp);
        tmp.flip().asFloatBuffer().get(floats, offset, count);
    }

    @Override
    public long length() {
        return fileSize;
    }

    @Override
    public void close() {
        // Channel is managed by Supplier
    }
}
