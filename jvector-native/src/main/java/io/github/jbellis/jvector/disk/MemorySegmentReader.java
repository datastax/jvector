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
import java.lang.invoke.MethodHandle;
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
    private static final int MADV_WILLNEED = 3; // POSIX value
    private static final long DEFAULT_PAGE = 4096;   // when getpagesize() is unavailable
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
        private final MethodHandle madvise;   // posix_madvise, or null when the symbol is missing
        private final long page;              // the system page size: posix_madvise wants a page-aligned address
        private volatile boolean announceFailed;   // set by the first failed WILLNEED, which is logged; later ones prefetch instead

        public Supplier(Path path) throws IOException {
            this.path = path;
            this.arena = Arena.ofShared();
            var linker = Linker.nativeLinker();
            this.page = pageSize(linker);
            try (var ch = FileChannel.open(path, StandardOpenOption.READ)) {
                this.memory = ch.map(MapMode.READ_ONLY, 0L, ch.size(), arena);

                // Apply MADV_RANDOM advice
                var maybeMadvise = linker.defaultLookup().find("posix_madvise");
                if (maybeMadvise.isPresent()) {
                    madvise = linker.downcallHandle(maybeMadvise.get(),
                            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));
                    int result = (int) madvise.invokeExact(memory, memory.byteSize(), MADV_RANDOM);
                    if (result != 0) {
                        throw new IOException("posix_madvise failed with error code: " + result);
                    }
                } else {
                    madvise = null;
                    logger.warn("posix_madvise not found, MADV_RANDOM advice not applied");
                }
            } catch (Throwable e) {
                arena.close();
                if (e instanceof IOException) {
                    throw (IOException) e;
                }
                throw new RuntimeException(e);
            }
        }

        /**
         * The page size the kernel aligns advice to, from {@code getpagesize()}: 4 KiB on most x86-64 systems,
         * 16 KiB on Apple silicon and 64 KiB on some arm64 kernels, where a 4 KiB-aligned address is rejected
         * with EINVAL. {@link #DEFAULT_PAGE} when the symbol is missing or returns nonsense.
         */
        private static long pageSize(Linker linker) {
            try {
                var maybeGetpagesize = linker.defaultLookup().find("getpagesize");
                if (maybeGetpagesize.isPresent()) {
                    var getpagesize = linker.downcallHandle(maybeGetpagesize.get(), FunctionDescriptor.of(ValueLayout.JAVA_INT));
                    int size = (int) getpagesize.invokeExact();
                    if (size >= 4096 && Integer.bitCount(size) == 1) {
                        return size;
                    }
                    logger.warn("getpagesize() returned {}; assuming {}-byte pages for madvise", size, DEFAULT_PAGE);
                } else {
                    logger.warn("getpagesize not found; assuming {}-byte pages for madvise", DEFAULT_PAGE);
                }
            } catch (Throwable e) {
                logger.warn("getpagesize() failed; assuming {}-byte pages for madvise", DEFAULT_PAGE, e);
            }
            return DEFAULT_PAGE;
        }

        // Windowed prefetch streams ranges through a separate read-ahead-enabled file descriptor:
        // the mapping itself advises MADV_RANDOM (disables kernel readahead), and MADV_WILLNEED
        // proved to be a no-op hint for large ranges. The page cache is shared per file, so
        // subsequent random access through the mapping hits the populated pages. The bytes read
        // here are discarded, so the buffer is sized to stay L2-resident rather than for syscall
        // amortization (cold throughput is device-bound at any size >= 32KB, and the warm case
        // is within noise of larger buffers at 64KB).
        private static final ThreadLocal<java.nio.ByteBuffer> PREFETCH_BUF =
                ThreadLocal.withInitial(() -> java.nio.ByteBuffer.wrap(new byte[64 << 10]));

        /**
         * {@code POSIX_MADV_WILLNEED} on the pages of {@code [offset, offset + length)}: the kernel
         * starts reading them in and returns; a later access through the mapping finds them resident
         * or already in flight. The positional-read prefetch without the symbol, and after the first
         * failure: posix_madvise returns its error instead of setting errno, so a rejected advice
         * (EINVAL for an address the kernel does not consider page-aligned, ENOMEM for an unmapped
         * range) would otherwise be a silent no-op that leaves every record to be demand-faulted.
         * The first failure is logged once, with the code.
         */
        @Override
        public void announce(long offset, long length) {
            if (length <= 0 || madvise == null || announceFailed) {
                prefetch(offset, length);
                return;
            }
            long from = Math.max(0, offset) & -page;
            long to = Math.min(memory.byteSize(), offset + length);
            if (to <= from) {
                return;
            }
            int result;
            try {
                result = (int) madvise.invokeExact(memory.asSlice(from, to - from), to - from, MADV_WILLNEED);
            } catch (Throwable e) {
                result = -1;
                if (!announceFailed) {
                    logger.warn("posix_madvise(WILLNEED) of {} [{}, {}) threw; announcements fall back to prefetch", path, from, to, e);
                }
            }
            if (result != 0) {
                if (!announceFailed) {
                    announceFailed = true;
                    if (result > 0) {
                        logger.warn("posix_madvise(WILLNEED) of {} [{}, {}) returned error {} (page size {}); announcements fall back to prefetch",
                                    path, from, to, result, page);
                    }
                }
                prefetch(offset, length);
            }
        }

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
