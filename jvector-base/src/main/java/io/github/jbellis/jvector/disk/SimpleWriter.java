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

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.zip.CRC32;

/**
 * A simple implementation of IndexWriter that writes to a file.
 * This implementation is primarily for testing purposes.
 */
public class SimpleWriter implements IndexWriter {
    private final FileOutputStream fos;
    private final DataOutputStream dos;
    private volatile long bytesWrittenSinceStart = 0;
    private long startPosition = 0;

    public SimpleWriter(Path path) throws IOException {
        fos = new FileOutputStream(path.toFile());
        dos = new DataOutputStream(fos);
    }

    @Override
    public long position() throws IOException {
        dos.flush();
        return fos.getChannel().position();
    }

    @Override
    public void write(int b) throws IOException {
        dos.write(b);
        bytesWrittenSinceStart++;
    }

    @Override
    public void write(byte[] b) throws IOException {
        dos.write(b);
        bytesWrittenSinceStart += b.length;
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
        dos.write(b, off, len);
        bytesWrittenSinceStart += len;
    }

    @Override
    public void writeBoolean(boolean v) throws IOException {
        dos.writeBoolean(v);
        bytesWrittenSinceStart++;
    }

    @Override
    public void writeByte(int v) throws IOException {
        dos.writeByte(v);
        bytesWrittenSinceStart++;
    }

    @Override
    public void writeShort(int v) throws IOException {
        dos.writeShort(v);
        bytesWrittenSinceStart += 2;
    }

    @Override
    public void writeChar(int v) throws IOException {
        dos.writeChar(v);
        bytesWrittenSinceStart += 2;
    }

    @Override
    public void writeInt(int v) throws IOException {
        dos.writeInt(v);
        bytesWrittenSinceStart += 4;
    }

    @Override
    public void writeLong(long v) throws IOException {
        dos.writeLong(v);
        bytesWrittenSinceStart += 8;
    }

    @Override
    public void writeFloat(float v) throws IOException {
        dos.writeFloat(v);
        bytesWrittenSinceStart += 4;
    }

    @Override
    public void writeDouble(double v) throws IOException {
        dos.writeDouble(v);
        bytesWrittenSinceStart += 8;
    }

    @Override
    public void writeBytes(String s) throws IOException {
        dos.writeBytes(s);
        bytesWrittenSinceStart += s.length();
    }

    @Override
    public void writeChars(String s) throws IOException {
        dos.writeChars(s);
        bytesWrittenSinceStart += s.length() * 2;
    }

    @Override
    public void writeUTF(String s) throws IOException {
        dos.writeUTF(s);
        // UTF encoding adds variable length, so we need to recalculate position
        bytesWrittenSinceStart = fos.getChannel().position() - startPosition;
    }

    @Override
    public void close() throws IOException {
        dos.close();
        fos.close();
    }
}