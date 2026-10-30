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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.disk.IndexWriter;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;

/** A strictly sequential in-memory output for the compaction tests, failing once {@code capacity} bytes are written. */
final class MemoryIndexWriter implements IndexWriter {
    private final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    private final DataOutputStream out;
    boolean closed;

    MemoryIndexWriter(long capacity) {
        this.out = new DataOutputStream(new OutputStream() {
            @Override
            public void write(int b) throws IOException {
                if (bytes.size() >= capacity)
                    throw new IOException("output full at " + capacity + " bytes");
                bytes.write(b);
            }

            @Override
            public void write(byte[] b, int off, int len) throws IOException {
                if (bytes.size() + (long) len > capacity)
                    throw new IOException("output full at " + capacity + " bytes");
                bytes.write(b, off, len);
            }
        });
    }

    byte[] toArray() {
        return bytes.toByteArray();
    }

    byte[] bytesFrom(int offset) {
        byte[] all = toArray();
        return Arrays.copyOfRange(all, offset, all.length);
    }

    @Override
    public long position() {
        return bytes.size();
    }

    @Override
    public void close() {
        closed = true;
    }

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
