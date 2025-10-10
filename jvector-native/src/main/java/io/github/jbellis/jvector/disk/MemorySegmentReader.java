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

    /**
     * Factory for creating multiple {@link MemorySegmentReader} instances that share the same
     * memory-mapped file.
     * <p>
     * This supplier implementation allows multiple readers to be created for concurrent access
     * to the same underlying memory-mapped file, which is particularly useful for multi-threaded
     * vector search operations. All readers share a single memory-mapped region, reducing memory
     * overhead and improving cache efficiency.
     * <p>
     * The supplier manages the lifecycle of the shared {@link Arena} and {@link MemorySegment},
     * ensuring that the memory mapping remains valid while any reader might be using it. When
     * {@link #close()} is called, the arena is closed and all associated memory mappings are
     * released.
     * <p>
     * <strong>Performance optimizations:</strong>
     * <ul>
     * <li>Uses {@code posix_madvise} with {@code MADV_RANDOM} advice to optimize for random
     *     access patterns typical of vector similarity searches</li>
     * <li>Creates a shared arena that allows the memory segment to be accessed from multiple
     *     threads safely</li>
     * <li>Avoids the 2GB file size limitation of {@code ByteBuffer}-based implementations</li>
     * </ul>
     *
     * @see MemorySegmentReader
     * @see Arena#ofShared()
     */
    public static class Supplier implements ReaderSupplier {
        private final Arena arena;
        private final MemorySegment memory;

        /**
         * Creates a new supplier that memory-maps the specified file for random access.
         * <p>
         * This constructor performs the following operations:
         * <ol>
         * <li>Creates a shared {@link Arena} for managing the memory segment lifecycle</li>
         * <li>Opens the file as a {@link FileChannel} in read-only mode</li>
         * <li>Memory-maps the entire file using {@link FileChannel#map(MapMode, long, long, Arena)}</li>
         * <li>Applies {@code MADV_RANDOM} advice via {@code posix_madvise} to hint that the
         *     file will be accessed in random order, which is typical for vector similarity
         *     search workloads. This advice tells the OS not to perform aggressive read-ahead.</li>
         * </ol>
         * The {@code posix_madvise} call is performed using the Foreign Function &amp; Memory API
         * to directly invoke the native function. If {@code posix_madvise} is not available
         * (e.g., on non-POSIX systems), a warning is logged but construction proceeds normally.
         * <p>
         * <strong>Thread safety:</strong> The created supplier is thread-safe and can be used
         * to create readers from multiple threads. Each call to {@link #get()} returns a new
         * reader instance, but all readers share the same underlying memory mapping.
         * <p>
         * <strong>Resource management:</strong> The caller is responsible for calling
         * {@link #close()} when done with the supplier. Failing to close the supplier will
         * prevent the memory mapping from being released, potentially causing resource leaks.
         *
         * @param path the path to the file to be memory-mapped; must be readable and should
         *             contain vector data in the expected format
         * @throws IOException if an I/O error occurs opening the file, mapping the memory,
         *                     or if {@code posix_madvise} returns a non-zero error code
         * @throws RuntimeException if an unexpected error occurs during native method invocation
         */
        public Supplier(Path path) throws IOException {
            this.arena = Arena.ofShared();
            try (var ch = FileChannel.open(path, StandardOpenOption.READ)) {
                this.memory = ch.map(MapMode.READ_ONLY, 0L, ch.size(), arena);
                
                // Apply MADV_RANDOM advice
                var linker = Linker.nativeLinker();
                var maybeMadvise = linker.defaultLookup().find("posix_madvise");
                if (maybeMadvise.isPresent()) {
                    var madvise = linker.downcallHandle(maybeMadvise.get(),
                            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));
                    int result = (int) madvise.invokeExact(memory, memory.byteSize(), MADV_RANDOM);
                    if (result != 0) {
                        throw new IOException("posix_madvise failed with error code: " + result);
                    }
                } else {
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
