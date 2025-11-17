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

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStream;
import java.io.RandomAccessFile;
import java.nio.file.Path;
import java.util.zip.CRC32;

/**
 * Implements RandomAccessWriter as a buffered RandomAccessFile.
 */
public class BufferedRandomAccessWriter implements RandomAccessWriter {
    // Our approach is to create an OutputStream that writes to the RAF,
    // and then buffer that and wrap it in a DataOutputStream.  Obviously this means
    // that we need to be careful to flush the buffer when seeking to a new position.
    private final RandomAccessFile raf;
    private final DataOutputStream stream;

    /**
     * Creates a buffered random access writer for the given path.
     * @param path the path to write to
     * @throws FileNotFoundException if the file cannot be created
     */
    public BufferedRandomAccessWriter(Path path) throws FileNotFoundException {
        raf = new RandomAccessFile(path.toFile(), "rw");
        stream = new DataOutputStream(new BufferedOutputStream(new RandomAccessOutputStream(raf)));
    }

    private static class RandomAccessOutputStream extends OutputStream {
        private final RandomAccessFile raf;

        public RandomAccessOutputStream(RandomAccessFile raf) {
            this.raf = raf;
        }

        @Override
        public void write(int b) throws IOException {
            raf.write(b);
        }

        @Override
        public void write(byte[] b, int off, int len) throws IOException {
            raf.write(b, off, len);
        }

        @Override
        public void close() throws IOException {
            raf.close();
        }
    }

    @Override
    public void seek(long position) throws IOException {
        flush();
        raf.seek(position);
    }

    @Override
    public void close() throws IOException {
        flush();
        stream.close();
        raf.close();
    }

    @Override
    public long position() throws IOException {
        flush();
        return raf.getFilePointer();
    }

    @Override
    public void flush() throws IOException {
        stream.flush();
    }

    /**
     * Returns the CRC32 checksum for the range [startOffset .. endOffset)
     *
     * The file pointer will be left at endOffset.
     */
    @Override
    public long checksum(long startOffset, long endOffset) throws IOException {
        flush();

        var crc = new CRC32();
        var a = new byte[4096];
        seek(startOffset);
        for (long p = startOffset; p < endOffset; ) {
            int toRead = (int) Math.min(a.length, endOffset - p);
            int read = raf.read(a, 0, toRead);
            if (read < 0) {
                throw new IOException("EOF reached before endOffset");
            }
            crc.update(a, 0, read);
            p += read;
        }
        return crc.getValue();
    }

    //
    // DataOutput methods
    //

    @Override
    public void write(int b) throws IOException {
        stream.write(b);
    }

    @Override
    public void write(byte[] b) throws IOException {
        stream.write(b);
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
        stream.write(b, off, len);
    }

    @Override
    public void writeBoolean(boolean v) throws IOException {
        stream.writeBoolean(v);
    }

    @Override
    public void writeByte(int v) throws IOException {
        stream.writeByte(v);
    }

    @Override
    public void writeShort(int v) throws IOException {
        stream.writeShort(v);
    }

    @Override
    public void writeChar(int v) throws IOException {
        stream.writeChar(v);
    }

    @Override
    public void writeInt(int v) throws IOException {
        stream.writeInt(v);
    }

    @Override
    public void writeLong(long v) throws IOException {
        stream.writeLong(v);
    }

    @Override
    public void writeFloat(float v) throws IOException {
        stream.writeFloat(v);
    }

    @Override
    public void writeDouble(double v) throws IOException {
        stream.writeDouble(v);
    }

    @Override
    public void writeBytes(String s) throws IOException {
        stream.writeBytes(s);
    }

    @Override
    public void writeChars(String s) throws IOException {
        stream.writeChars(s);
    }

    @Override
    public void writeUTF(String s) throws IOException {
        stream.writeUTF(s);
    }
}
