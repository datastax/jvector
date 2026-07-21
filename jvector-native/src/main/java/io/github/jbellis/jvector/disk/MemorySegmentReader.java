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

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.foreign.ValueLayout.OfFloat;
import java.lang.foreign.ValueLayout.OfInt;
import java.lang.foreign.ValueLayout.OfLong;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.channels.FileChannel.MapMode;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link MemorySegment} based implementation of RandomAccessReader.  This is the recommended
 * RandomAccessReader implementation included with JVector.
 * <p>
 * MemorySegmentReader applies MADV_RANDOM to the backing storage, and doesn't have the 2GB file size limitation
 * of {@link SimpleMappedReader}.
 */
public class MemorySegmentReader implements RandomAccessReader {
    private static final Logger logger = LoggerFactory.getLogger(MemorySegmentReader.class);

    private static final int MADV_RANDOM = 1; // Value for Linux
    private static final OfInt intLayout = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.BIG_ENDIAN);
    private static final OfFloat floatLayout = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.BIG_ENDIAN);
    private static final OfLong longLayout = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.BIG_ENDIAN);

    final MemorySegment memory;
    private long position = 0;

    MemorySegmentReader(MemorySegment memory) {
        this.memory = memory;
    }

    @Override
    public void seek(long offset) {
        this.position = offset;
    }

    @Override
    public long getPosition() {
        return position;
    }

    @Override
    public void readFully(float[] buffer) {
        MemorySegment.copy(memory, floatLayout, position, buffer, 0, buffer.length);
        position += buffer.length * 4L;
    }

    @Override
    public void readFully(byte[] b) {
        MemorySegment.copy(memory, ValueLayout.JAVA_BYTE, position, b, 0, b.length);
        position += b.length;
    }

    @Override
    public void readFully(ByteBuffer buffer) {
        var remaining = buffer.remaining();
        var slice = memory.asSlice(position, remaining).asByteBuffer();
        buffer.put(slice);
        position += remaining;
    }

    @Override
    public void readFully(long[] vector) {
        MemorySegment.copy(memory, longLayout, position, vector, 0, vector.length);
        position += vector.length * 8L;
    }

    @Override
    public int readInt() {
        var k = memory.get(intLayout, position);
        position += 4;
        return k;
    }

    @Override
    public long readLong() {
        var l = memory.get(longLayout, position);
        position += 8;
        return l;
    }

    @Override
    public float readFloat() {
        var f = memory.get(floatLayout, position);
        position += 4;
        return f;
    }

    @Override
    public void read(int[] ints, int offset, int count) {
        MemorySegment.copy(memory, intLayout, position, ints, offset, count);
        position += count * 4L;
    }

    @Override
    public void read(float[] floats, int offset, int count) {
        MemorySegment.copy(memory, floatLayout, position, floats, offset, count);
        position += count * 4L;
    }

    @Override
    public long length() {
        return memory.byteSize();
    }

    /**
     * Loads the contents of the mapped segment into physical memory.
     * This is a best-effort mechanism.
     */
    @SuppressWarnings("unused")
    public void loadMemory() {
        memory.load();
    }

    @Override
    public void close() {
        // Individual readers don't close the shared memory
    }

    public static class Supplier implements ReaderSupplier {
        private final Arena arena;
        private final MemorySegment memory;
        private final Path path;

        public Supplier(Path path) throws IOException {
            this.path = path;
            this.arena = Arena.ofShared();
            try (var ch = FileChannel.open(path, StandardOpenOption.READ)) {
                this.memory = ch.map(MapMode.READ_ONLY, 0L, ch.size(), arena);
                // Search-time access is random; disable kernel readahead by default.
                madvise(memory, MADV_RANDOM, true);
            } catch (Throwable e) {
                arena.close();
                if (e instanceof IOException) {
                    throw (IOException) e;
                }
                throw new RuntimeException(e);
            }
        }

        private static void madvise(MemorySegment memory, int advice, boolean strict) throws IOException {
            var linker = Linker.nativeLinker();
            var maybeMadvise = linker.defaultLookup().find("posix_madvise");
            if (maybeMadvise.isPresent()) {
                var madvise = linker.downcallHandle(maybeMadvise.get(),
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));
                int result;
                try {
                    result = (int) madvise.invokeExact(memory, memory.byteSize(), advice);
                } catch (Throwable t) {
                    throw new RuntimeException(t);
                }
                if (result != 0 && strict) {
                    throw new IOException("posix_madvise failed with error code: " + result);
                }
            } else {
                logger.warn("posix_madvise not found, advice {} not applied", advice);
            }
        }

        // Windowed prefetch streams ranges through a separate read-ahead-enabled file descriptor:
        // the mapping itself advises MADV_RANDOM (disables kernel readahead), and MADV_WILLNEED
        // proved to be a no-op hint for large ranges. The page cache is shared per file, so
        // subsequent random access through the mapping hits the populated pages. Called
        // concurrently from many worker threads; positional reads on a shared channel are
        // thread-safe, and the per-thread buffer avoids allocation churn.
        private static final ThreadLocal<java.nio.ByteBuffer> PREFETCH_BUF =
                ThreadLocal.withInitial(() -> java.nio.ByteBuffer.wrap(new byte[4 << 20]));

        @Override
        public void prefetch(long offset, long length) {
            if (length <= 0) {
                return;
            }
            var buf = PREFETCH_BUF.get();
            long end = Math.min(offset + length, memory.byteSize());
            try (var ch = FileChannel.open(path, StandardOpenOption.READ)) {
                long pos = Math.max(0, offset);
                while (pos < end) {
                    buf.clear().limit((int) Math.min(buf.capacity(), end - pos));
                    int n = ch.read(buf, pos);
                    if (n < 0) {
                        break;
                    }
                    pos += n;
                }
            } catch (IOException e) {
                logger.warn("ranged prefetch of {} [{}, {}) failed; continuing without warm cache",
                            path, offset, end, e);
            }
        }

        @Override
        public MemorySegmentReader get() {
            return new MemorySegmentReader(memory);
        }

        @Override
        public void close() {
            arena.close();
        }
    }
}
