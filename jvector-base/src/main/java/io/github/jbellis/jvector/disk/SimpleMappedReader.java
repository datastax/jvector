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

    /**
     * Constructs a SimpleMappedReader wrapping the specified memory-mapped buffer.
     *
     * @param mbb the memory-mapped byte buffer to read from
     */
    SimpleMappedReader(MappedByteBuffer mbb) {
        super(mbb);
    }

    @Override
    public void close() {
        // Individual readers don't close anything
    }

    /**
     * Supplier that creates SimpleMappedReader instances from a memory-mapped file.
     * The file is mapped into memory once during construction and shared across all readers.
     */
    public static class Supplier implements ReaderSupplier {
        /** The shared memory-mapped buffer for this file. */
        private final MappedByteBuffer buffer;
        /** Unsafe instance for invoking the buffer cleaner when closing. */
        private static final Unsafe unsafe = getUnsafe();

        /**
         * Constructs a Supplier that memory-maps the file at the specified path.
         * The entire file is loaded into memory. Files larger than 2GB are not supported.
         *
         * @param path the path to the file to map
         * @throws IOException if an I/O error occurs
         * @throws RuntimeException if the file is larger than 2GB
         */
        public Supplier(Path path) throws IOException {
            try (var raf = new RandomAccessFile(path.toString(), "r")) {
                if (raf.length() > Integer.MAX_VALUE) {
                    throw new RuntimeException("SimpleMappedReader doesn't support files above 2GB");
                }
                this.buffer = raf.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, raf.length());
                this.buffer.load();
            }
        }

        @Override
        public SimpleMappedReader get() {
            return new SimpleMappedReader((MappedByteBuffer) buffer.duplicate());
        }

        @Override
        public void close() {
            if (unsafe != null) {
                try {
                    unsafe.invokeCleaner(buffer);
                } catch (IllegalArgumentException e) {
                    // empty catch, this was a duplicated/indirect buffer or
                    // otherwise not cleanable
                }
            }
        }
    }
}
