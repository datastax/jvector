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

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;

// TODO what are the low-hanging optimization options here?
// The requirement that we need to read from a file that is potentially changing in length
// limits our options.
public class SimpleReader implements RandomAccessReader {
    RandomAccessFile raf;

    public SimpleReader(Path path) throws FileNotFoundException {
        raf = new RandomAccessFile(path.toFile(), "r");
    }

    @Override
    public void seek(long offset) throws IOException {
        raf.seek(offset);
    }

    @Override
    public long getPosition() throws IOException {
        return raf.getFilePointer();
    }

    @Override
    public int readInt() throws IOException {
        return raf.readInt();
    }

    @Override
    public long readLong() throws IOException {
        return raf.readLong();
    }

    @Override
    public float readFloat() throws IOException {
        return raf.readFloat();
    }

    @Override
    public void readFully(byte[] bytes) throws IOException {
        raf.readFully(bytes);
    }

    @Override
    public void readFully(ByteBuffer buffer) throws IOException {
        raf.getChannel().read(buffer);
    }

    @Override
    public void readFully(float[] floats) throws IOException {
        read(floats, 0, floats.length);
    }

    @Override
    public void readFully(long[] vector) throws IOException {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = raf.readLong();
        }
    }

    @Override
    public void read(int[] ints, int offset, int count) throws IOException {
        for (int i = 0; i < count; i++) {
            ints[offset + i] = raf.readInt();
        }
    }

    @Override
    public void read(float[] floats, int offset, int count) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(count * Float.BYTES);
        raf.getChannel().read(bb);
        bb.flip().order(ByteOrder.BIG_ENDIAN);
        bb.asFloatBuffer().get(floats, offset, count);
    }

    @Override
    public void close() throws IOException {
        raf.close();
    }

    public static class Supplier implements ReaderSupplier {
        private final Path path;

        public Supplier(Path path) {
            this.path = path;
        }

        @Override
        public RandomAccessReader get() throws FileNotFoundException {
            return new SimpleReader(path);
        }
    }
}
