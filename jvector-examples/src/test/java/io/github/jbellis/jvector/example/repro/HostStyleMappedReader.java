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

package io.github.jbellis.jvector.example.repro;

import io.github.jbellis.jvector.disk.ByteBufferReader;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import sun.misc.Unsafe;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.reflect.Field;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;

/// A [RandomAccessReader] over a raw NIO `mmap` that models how an embedding host (for example a
/// Cassandra `FileHandle`-based adapter) typically reads a jvector graph file. Two properties are
/// deliberate, because they are the properties the memory-safety theories hinge on:
///
/// - Bulk float reads go through a byte-swapping [FloatBuffer#get(float[],int,int)] view (the file
///   format is big-endian, the hardware little-endian), so an access to invalidated pages faults
///   inside `Unsafe.copySwapMemory0` / `Copy::conjoint_swap` — the exact leaf frame in the
///   production crash. (jvector's own [ByteBufferReader] reads floats one element at a time, which
///   would fault in a different leaf.)
/// - [Supplier#close()] unmaps immediately via `sun.misc.Unsafe.invokeCleaner`, exactly like
///   jvector's shipped [SimpleMappedReader.Supplier#close()] and Cassandra's `FileUtils.clean`.
///   Unlike the `Arena`-managed mapping in jvector-native's `MemorySegmentReader`, there is no
///   liveness handshake: readers vended by [Supplier#get()] are left dangling and any subsequent
///   access faults natively instead of throwing.
public final class HostStyleMappedReader extends ByteBufferReader {
    private final MappedByteBuffer mbb;

    private HostStyleMappedReader(MappedByteBuffer mbb) {
        super(mbb);
        this.mbb = mbb;
    }

    /// Bulk read through the swapped [FloatBuffer] view so a fault lands in `Copy::conjoint_swap`.
    @Override
    public void read(float[] floats, int offset, int count) {
        FloatBuffer fb = mbb.asFloatBuffer();
        fb.get(floats, offset, count);
        mbb.position(mbb.position() + count * Float.BYTES);
    }

    @Override
    public void readFully(float[] floats) {
        read(floats, 0, floats.length);
    }

    /// Vends dangling-capable readers over one shared mapping; [#close()] performs a raw,
    /// immediate munmap with no coordination with outstanding readers.
    public static final class Supplier implements ReaderSupplier {
        private static final Unsafe UNSAFE = loadUnsafe();
        private final MappedByteBuffer buffer;

        public Supplier(Path path) throws IOException {
            try (RandomAccessFile raf = new RandomAccessFile(path.toString(), "r")) {
                if (raf.length() > Integer.MAX_VALUE) {
                    throw new IOException("file too large for a single NIO mapping: " + path);
                }
                this.buffer = raf.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, raf.length());
                this.buffer.order(ByteOrder.BIG_ENDIAN);
            }
        }

        @Override
        public HostStyleMappedReader get() {
            MappedByteBuffer dup = (MappedByteBuffer) buffer.duplicate();
            dup.order(ByteOrder.BIG_ENDIAN);
            return new HostStyleMappedReader(dup);
        }

        @Override
        public void close() {
            if (UNSAFE != null) {
                try {
                    UNSAFE.invokeCleaner(buffer);
                } catch (IllegalArgumentException e) {
                    // not a cleanable direct buffer; nothing to unmap
                }
            }
        }

        private static Unsafe loadUnsafe() {
            try {
                Field f = Unsafe.class.getDeclaredField("theUnsafe");
                f.setAccessible(true);
                return (Unsafe) f.get(null);
            } catch (Exception e) {
                return null;
            }
        }
    }
}
