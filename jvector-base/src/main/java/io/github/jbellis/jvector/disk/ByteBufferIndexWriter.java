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

/**
 * An IndexWriter implementation backed by a ByteBuffer for in-memory record building.
 * This allows existing Feature.writeInline() implementations to write to memory buffers
 * that can later be bulk-written to disk.
 * <p>
 * Byte order is set to BIG_ENDIAN to match Java's DataOutput specification and ensure
 * cross-platform compatibility.
 * <p>
 * Not thread-safe. Each thread should use its own instance.
 */
public class ByteBufferIndexWriter implements IndexWriter {
    private final ByteBuffer buffer;
    private final int initialPosition;

    /**
     * Creates a writer that writes to the given buffer.
     * The buffer's byte order is set to BIG_ENDIAN to match DataOutput behavior.
     *
     * @param buffer the buffer to write to
     * @param autoClear if true, automatically clears the buffer before writing
     */
    public ByteBufferIndexWriter(ByteBuffer buffer, boolean autoClear) {
        this.buffer = buffer;
        if (autoClear) {
            buffer.clear();
        }
        this.buffer.order(ByteOrder.BIG_ENDIAN);
        this.initialPosition = buffer.position();
    }

    /**
     * Creates a writer that writes to the given buffer, automatically clearing it first.
     * The buffer's byte order is set to BIG_ENDIAN to match DataOutput behavior.
     * This is the most common usage pattern and is equivalent to:
     * {@code new ByteBufferIndexWriter(buffer, true)}
     *
     * @param buffer the buffer to write to (will be cleared)
     */
    public ByteBufferIndexWriter(ByteBuffer buffer) {
        this(buffer, true);
    }

    /**
     * Creates a new {@code ByteBufferIndexWriter} with the specified capacity.
     * <p>
     * If {@code offHeap} is {@code true}, a direct (off-heap) {@link ByteBuffer} is used;
     * otherwise, a heap-based buffer is used.
     *
     * @param capacity the buffer capacity in bytes
     * @param offHeap  if {@code true}, use a direct (off-heap) buffer; otherwise, use a heap buffer
     * @return a new {@code ByteBufferIndexWriter} backed by a buffer of the specified type and capacity
     */
    public static ByteBufferIndexWriter create(int capacity, boolean offHeap) {
        if (offHeap) {
            return allocateDirect(capacity);
        } else {
            return allocate(capacity);
        }
    }

    /**
     * Creates a writer with a new heap ByteBuffer of the given capacity.
     * The buffer uses BIG_ENDIAN byte order.
     */
    private static ByteBufferIndexWriter allocate(int capacity) {
        ByteBuffer buffer = ByteBuffer.allocate(capacity);
        buffer.order(ByteOrder.BIG_ENDIAN);
        return new ByteBufferIndexWriter(buffer);
    }

    /**
     * Creates a writer with a new direct ByteBuffer of the given capacity.
     * The buffer uses BIG_ENDIAN byte order.
     */
    private static ByteBufferIndexWriter allocateDirect(int capacity) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(capacity);
        buffer.order(ByteOrder.BIG_ENDIAN);
        return new ByteBufferIndexWriter(buffer);
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
        // Reset for next use
        buffer.clear();
        buffer.position(initialPosition);
    }

    /**
     * Returns an independent copy of the written data as a new ByteBuffer.
     * The returned buffer is ready to read (position=0, limit=written data length).
     * The writer's buffer is automatically reset and ready for reuse.
     * <p>
     * This method handles all buffer management:
     * <ul>
     *   <li>Flips the buffer to prepare for reading (sets limit=position, position=initialPosition)</li>
     *   <li>Allocates and creates a copy of the data</li>
     *   <li>Resets the buffer for the next write operation</li>
     * </ul>
     * <p>
     * This is the recommended way to extract data from the writer when the buffer
     * will be reused (e.g., in thread-local scenarios).
     *
     * @return a new ByteBuffer containing a copy of the written data
     */
    public ByteBuffer cloneBuffer() {
        // Calculate the amount of data written
        int bytesWritten = buffer.position() - initialPosition;

        // Set limit to current position and position to initial for reading
        int savedPosition = buffer.position();
        buffer.position(initialPosition);
        buffer.limit(savedPosition);

        // Create independent copy
        ByteBuffer copy = ByteBuffer.allocate(bytesWritten);
        copy.put(buffer);
        copy.flip();

        return copy;
    }

    /**
     * Returns the number of bytes written since construction or last reset.
     *
     * @return bytes written
     */
    public int bytesWritten() {
        return buffer.position() - initialPosition;
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
        // UTF format stores the string length as a 2-byte (16-bit) unsigned integer prefix,
        // which has a maximum value of 65535
        if (utflen > 65535) {
            throw new IOException("encoded string too long: " + utflen + " bytes");
        }
        
        buffer.putShort((short) utflen);
        buffer.put(bytes);
    }
}
