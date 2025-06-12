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

/**
 * A simple implementation of IndexWriter that writes to a file.
 * This implementation is primarily for testing purposes.
 * Primarily, it is a basic sequential writer, unlike {@link BufferedRandomAccessWriter} which is mostly used in production.
 */
public class SimpleWriter implements IndexWriter {
    private final FileOutputStream fos;
    private final DataOutputStream dos;

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
    }

    @Override
    public void write(byte[] b) throws IOException {
        dos.write(b);
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
        dos.write(b, off, len);
    }

    @Override
    public void writeBoolean(boolean v) throws IOException {
        dos.writeBoolean(v);
    }

    @Override
    public void writeByte(int v) throws IOException {
        dos.writeByte(v);
    }

    @Override
    public void writeShort(int v) throws IOException {
        dos.writeShort(v);
    }

    @Override
    public void writeChar(int v) throws IOException {
        dos.writeChar(v);
    }

    @Override
    public void writeInt(int v) throws IOException {
        dos.writeInt(v);
    }

    @Override
    public void writeLong(long v) throws IOException {
        dos.writeLong(v);
    }

    @Override
    public void writeFloat(float v) throws IOException {
        dos.writeFloat(v);
    }

    @Override
    public void writeDouble(double v) throws IOException {
        dos.writeDouble(v);
    }

    @Override
    public void writeBytes(String s) throws IOException {
        dos.writeBytes(s);
    }

    @Override
    public void writeChars(String s) throws IOException {
        dos.writeChars(s);
    }

    @Override
    public void writeUTF(String s) throws IOException {
        dos.writeUTF(s);
    }

    @Override
    public void close() throws IOException {
        dos.flush();
        dos.close();
        fos.close();
    }
}