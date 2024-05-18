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

import sun.misc.Unsafe;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.reflect.Field;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.logging.Logger;

/**
 * Simple sample implementation of RandomAccessReader.
 * It provides a bare minimum to run against disk in reasonable time.
 * Does not handle files above 2 GB.
 */
public class SimpleMappedReader extends ByteBufferReader {
    private static final Logger LOG = Logger.getLogger(SimpleMappedReader.class.getName());

    private static final Unsafe unsafe = getUnsafe();

    private static Unsafe getUnsafe() {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            return (Unsafe) f.get(null);
        } catch (Exception e) {
            LOG.warning("MappedRandomAccessReader can't acquire needed Unsafe access");
            return null;
        }
    }

    public SimpleMappedReader(Path path) throws IOException {
        this(path.toString());
    }

    public SimpleMappedReader(String name) throws IOException {
        this(getMappedByteBuffer(name));
    }

    private SimpleMappedReader(MappedByteBuffer mbb) {
        super(mbb);
    }

    private static MappedByteBuffer getMappedByteBuffer(String name) throws IOException {
        var raf = new RandomAccessFile(name, "r");
        if (raf.length() > Integer.MAX_VALUE) {
            throw new RuntimeException("MappedRandomAccessReader doesn't support large files");
        }
        MappedByteBuffer mbb = raf.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, raf.length());
        mbb.load();
        raf.close();
        return mbb;
    }

    @Override
    public void close() {
        if (unsafe != null) {
            try {
                unsafe.invokeCleaner(bb);
            } catch (IllegalArgumentException e) {
                // empty catch, this was a duplicated/indirect buffer or
                // otherwise not cleanable
            }
        }
    }

    public SimpleMappedReader duplicate() {
        return new SimpleMappedReader((MappedByteBuffer) bb.duplicate());
    }

    public static class Supplier implements ReaderSupplier {
        private final SimpleMappedReader smr;

        public Supplier(Path path) throws IOException {
            smr = new SimpleMappedReader(path);
        }

        @Override
        public RandomAccessReader get() {
            return smr.duplicate();
        }

        @Override
        public void close() {
            smr.close();
        }
    }
}
