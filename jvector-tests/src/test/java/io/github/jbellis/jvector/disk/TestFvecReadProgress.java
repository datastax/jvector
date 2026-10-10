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

import org.junit.Test;
import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.*;
import static org.junit.Assert.*;

public class TestFvecReadProgress {
    @Test(timeout = 1000) public void repeatedZeroReadsFailInsteadOfSpinning() {
        var channel = new TestChannel(-1);
        var failure = assertThrows(IOException.class, () ->
                FvecFileVectorValues.readFully(channel, ByteBuffer.allocate(1), 123));
        assertTrue(failure.getMessage().contains("123"));
        assertEquals(16, channel.calls);
    }

    @Test(timeout = 1000) public void progressResetsTheZeroReadLimitAndPreservesOffsets() throws Exception {
        var channel = new TestChannel(15);
        var buffer = ByteBuffer.allocate(4);
        buffer.position(2);
        FvecFileVectorValues.readFully(channel, buffer, 123);
        assertEquals(32, channel.calls);
        assertEquals(4, buffer.position());
        assertEquals((byte) 123, buffer.get(2));
        assertEquals((byte) 124, buffer.get(3));
    }

    @Test(timeout = 1000) public void eofStillFailsImmediately() {
        var channel = new TestChannel(0) {
            @Override public int read(ByteBuffer buffer, long offset) { return -1; }
        };
        assertThrows(EOFException.class, () ->
                FvecFileVectorValues.readFully(channel, ByteBuffer.allocate(1), 0));
    }

    private static class TestChannel extends FileChannel {
        final int zerosPerByte;
        int calls, zeros;
        TestChannel(int zerosPerByte) { this.zerosPerByte = zerosPerByte; }
        @Override public int read(ByteBuffer buffer, long offset) {
            calls++;
            if (zerosPerByte < 0 || zeros++ < zerosPerByte) return 0;
            zeros = 0;
            buffer.put((byte) offset);
            return 1;
        }
        @Override public int read(ByteBuffer buffer) { throw new UnsupportedOperationException(); }
        @Override public long read(ByteBuffer[] buffers, int offset, int count) { throw new UnsupportedOperationException(); }
        @Override public int write(ByteBuffer buffer) { throw new UnsupportedOperationException(); }
        @Override public int write(ByteBuffer buffer, long offset) { throw new UnsupportedOperationException(); }
        @Override public long write(ByteBuffer[] buffers, int offset, int count) { throw new UnsupportedOperationException(); }
        @Override public long position() { throw new UnsupportedOperationException(); }
        @Override public FileChannel position(long position) { throw new UnsupportedOperationException(); }
        @Override public long size() { throw new UnsupportedOperationException(); }
        @Override public FileChannel truncate(long size) { throw new UnsupportedOperationException(); }
        @Override public void force(boolean metadata) { throw new UnsupportedOperationException(); }
        @Override public long transferTo(long position, long count, WritableByteChannel target) { throw new UnsupportedOperationException(); }
        @Override public long transferFrom(ReadableByteChannel source, long position, long count) { throw new UnsupportedOperationException(); }
        @Override public MappedByteBuffer map(MapMode mode, long position, long size) { throw new UnsupportedOperationException(); }
        @Override public FileLock lock(long position, long size, boolean shared) { throw new UnsupportedOperationException(); }
        @Override public FileLock tryLock(long position, long size, boolean shared) { throw new UnsupportedOperationException(); }
        @Override protected void implCloseChannel() {}
    }
}
