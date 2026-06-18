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

package io.github.jbellis.jvector.vector;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.vector.types.ByteSequence;

import java.io.IOException;
import org.junit.jupiter.api.Test;

public class MemorySegmentVectorProviderTest {

    @Test
    void testWriteByteSequenceSlice() throws IOException {
        MemorySegmentVectorProvider provider = new MemorySegmentVectorProvider();

        byte[] originalBytes = {10, 20, 30, 40, 50};
        ByteSequence<?> original = provider.createByteSequence(originalBytes);

        ByteSequence<?> slice = original.slice(2, 3);

        MockIndexWriter dummyWriter = new MockIndexWriter();

        // Proves sliced writes function perfectly
        provider.writeByteSequence(dummyWriter, slice);

        byte[] expected = {30, 40, 50};
        org.junit.jupiter.api.Assertions.assertArrayEquals(expected, dummyWriter.toByteArray());
    }

    @Test
    void testWriteByteSequenceFull() throws IOException {
        MemorySegmentVectorProvider provider = new MemorySegmentVectorProvider();

        byte[] expectedBytes = {1, 2, 3, 4, 5};
        ByteSequence<?> sequence = provider.createByteSequence(expectedBytes);

        MockIndexWriter dummyWriter = new MockIndexWriter();

        // Proves standard, non-sliced writes function perfectly
        provider.writeByteSequence(dummyWriter, sequence);

        org.junit.jupiter.api.Assertions.assertArrayEquals(expectedBytes, dummyWriter.toByteArray());
    }

    @Test
    void testWriteByteSequenceZeroLength() throws IOException {
        MemorySegmentVectorProvider provider = new MemorySegmentVectorProvider();

        byte[] originalBytes = {10, 20, 30};
        ByteSequence<?> original = provider.createByteSequence(originalBytes);

        // Create a logical empty slice
        ByteSequence<?> emptySlice = original.slice(1, 0);

        MockIndexWriter dummyWriter = new MockIndexWriter();

        // Proves edge cases don't throw IndexOutOfBoundsException
        provider.writeByteSequence(dummyWriter, emptySlice);

        org.junit.jupiter.api.Assertions.assertArrayEquals(new byte[0], dummyWriter.toByteArray());
    }

    /**
     * A lightweight mock to capture IndexWriter output without boilerplate.
     */
    private static class MockIndexWriter implements IndexWriter {
        private final java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
        private final java.io.DataOutputStream out = new java.io.DataOutputStream(bos);

        public byte[] toByteArray() { return bos.toByteArray(); }

        @Override public long position() { return bos.size(); }
        @Override public void close() throws IOException { out.close(); }

        // DataOutput delegation
        @Override public void write(int b) throws IOException { out.write(b); }
        @Override public void write(byte[] b) throws IOException { out.write(b); }
        @Override public void write(byte[] b, int off, int len) throws IOException { out.write(b, off, len); }
        @Override public void writeBoolean(boolean v) throws IOException { out.writeBoolean(v); }
        @Override public void writeByte(int v) throws IOException { out.writeByte(v); }
        @Override public void writeShort(int v) throws IOException { out.writeShort(v); }
        @Override public void writeChar(int v) throws IOException { out.writeChar(v); }
        @Override public void writeInt(int v) throws IOException { out.writeInt(v); }
        @Override public void writeLong(long v) throws IOException { out.writeLong(v); }
        @Override public void writeFloat(float v) throws IOException { out.writeFloat(v); }
        @Override public void writeDouble(double v) throws IOException { out.writeDouble(v); }
        @Override public void writeBytes(String s) throws IOException { out.writeBytes(s); }
        @Override public void writeChars(String s) throws IOException { out.writeChars(s); }
        @Override public void writeUTF(String s) throws IOException { out.writeUTF(s); }
    }
}
