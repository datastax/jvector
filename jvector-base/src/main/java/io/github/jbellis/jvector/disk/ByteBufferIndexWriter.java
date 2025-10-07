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
 * An IndexWriter implementation backed by a ByteBuffer for in-memory record building.
 * This allows existing Feature.writeInline() implementations to write to memory buffers
 * that can later be bulk-written to disk.
 * <p>
 * Not thread-safe. Each thread should use its own instance.
 */
public class ByteBufferIndexWriter implements IndexWriter {
    private final ByteBuffer buffer;
    private final int initialPosition;

    /**
     * Creates a writer that writes to the given buffer starting at its current position.
     * The buffer's position will be advanced as data is written.
     */
    public ByteBufferIndexWriter(ByteBuffer buffer) {
        this.buffer = buffer;
        this.initialPosition = buffer.position();
    }

    /**
     * Creates a writer with a new heap ByteBuffer of the given capacity.
     */
    public static ByteBufferIndexWriter allocate(int capacity) {
        return new ByteBufferIndexWriter(ByteBuffer.allocate(capacity));
    }

    /**
     * Creates a writer with a new direct ByteBuffer of the given capacity.
     */
    public static ByteBufferIndexWriter allocateDirect(int capacity) {
        return new ByteBufferIndexWriter(ByteBuffer.allocateDirect(capacity));
    }

    /**
     * Returns the underlying buffer. The buffer's position will be at the end of written data.
     */
    public ByteBuffer getBuffer() {
        return buffer;
    }

    /**
     * Returns a read-only view of the written data (from initial position to current position).
     */
    public ByteBuffer getWrittenData() {
        int currentPos = buffer.position();
        buffer.position(initialPosition);
        ByteBuffer slice = buffer.slice();
        slice.limit(currentPos - initialPosition);
        buffer.position(currentPos);
        return slice.asReadOnlyBuffer();
    }

    /**
     * Resets the buffer position to the initial position, allowing reuse.
     */
    public void reset() {
        buffer.position(initialPosition);
    }

    @Override
    public long position() {
        return buffer.position() - initialPosition;
    }

    @Override
    public void close() {
        // No-op for ByteBuffer
    }

    // DataOutput methods

    @Override
    public void write(int b) {
        buffer.put((byte) b);
    }

    @Override
    public void write(byte[] b) {
        buffer.put(b);
    }

    @Override
    public void write(byte[] b, int off, int len) {
        buffer.put(b, off, len);
    }

    @Override
    public void writeBoolean(boolean v) {
        buffer.put((byte) (v ? 1 : 0));
    }

    @Override
    public void writeByte(int v) {
        buffer.put((byte) v);
    }

    @Override
    public void writeShort(int v) {
        buffer.putShort((short) v);
    }

    @Override
    public void writeChar(int v) {
        buffer.putChar((char) v);
    }

    @Override
    public void writeInt(int v) {
        buffer.putInt(v);
    }

    @Override
    public void writeLong(long v) {
        buffer.putLong(v);
    }

    @Override
    public void writeFloat(float v) {
        buffer.putFloat(v);
    }

    @Override
    public void writeDouble(double v) {
        buffer.putDouble(v);
    }

    @Override
    public void writeBytes(String s) {
        int len = s.length();
        for (int i = 0; i < len; i++) {
            buffer.put((byte) s.charAt(i));
        }
    }

    @Override
    public void writeChars(String s) {
        int len = s.length();
        for (int i = 0; i < len; i++) {
            buffer.putChar(s.charAt(i));
        }
    }

    @Override
    public void writeUTF(String s) throws IOException {
        // Use standard DataOutputStream UTF encoding
        byte[] bytes = s.getBytes("UTF-8");
        int utflen = bytes.length;
        
        if (utflen > 65535) {
            throw new IOException("encoded string too long: " + utflen + " bytes");
        }
        
        buffer.putShort((short) utflen);
        buffer.put(bytes);
    }
}
