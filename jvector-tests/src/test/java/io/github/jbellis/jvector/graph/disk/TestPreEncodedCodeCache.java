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

package io.github.jbellis.jvector.graph.disk;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Covers the chunked mapping in {@link PreEncodedCodeCache}.
 *
 * <p>Production chunks are 1 GiB, so a graph would need tens of millions of nodes before a second
 * chunk appears — far past what a unit test can build. These tests use the package-private
 * {@code maxChunkBytes} overload to force many small chunks, so the boundary arithmetic is
 * exercised directly rather than only on the single-chunk path that every other compaction test
 * happens to take.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestPreEncodedCodeCache extends RandomizedTest {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    /** Distinct, ordinal-derived bytes so a mis-indexed read is detected rather than masked by zeros. */
    private static ByteSequence<?> codeFor(int ordinal, int codeSize) {
        ByteSequence<?> code = vts.createByteSequence(codeSize);
        for (int i = 0; i < codeSize; i++) {
            code.set(i, (byte) ((ordinal * 31 + i * 7) & 0xff));
        }
        return code;
    }

    private static void assertCodeMatches(int ordinal, int codeSize, byte[] actual) {
        for (int i = 0; i < codeSize; i++) {
            byte expected = (byte) ((ordinal * 31 + i * 7) & 0xff);
            assertEquals("byte " + i + " of ordinal " + ordinal, expected, actual[i]);
        }
    }

    private Path newCacheFile(int count, int codeSize) throws IOException {
        Path p = Files.createTempFile("code-cache-", ".bin");
        p.toFile().deleteOnExit();
        try (FileChannel fc = FileChannel.open(p, StandardOpenOption.WRITE)) {
            long bytes = PreEncodedCodeCache.sectionBytes(count, codeSize);
            fc.write(ByteBuffer.wrap(new byte[]{0}), bytes - 1);
        }
        return p;
    }

    /** Round-trips every ordinal with chunks small enough that most ordinals land in a different one. */
    @Test
    public void testRoundTripAcrossChunkBoundaries() throws IOException {
        int codeSize = 48, count = 500;
        int maxChunkBytes = codeSize * 7;   // 7 codes per chunk => 72 chunks, last one partial
        Path p = newCacheFile(count, codeSize);

        try (FileChannel fc = FileChannel.open(p, StandardOpenOption.READ, StandardOpenOption.WRITE);
             PreEncodedCodeCache cache = PreEncodedCodeCache.map(fc, 0, count, codeSize, maxChunkBytes)) {

            assertEquals("expected a partial trailing chunk", 72, cache.chunkCount());
            assertEquals(codeSize, cache.codeSize());
            assertEquals(count, cache.count());

            for (int i = 0; i < count; i++) {
                cache.put(i, codeFor(i, codeSize));
            }
            byte[] dst = new byte[codeSize];
            for (int i = 0; i < count; i++) {
                cache.get(i, dst);
                assertCodeMatches(i, codeSize, dst);
            }
        }
    }

    /** The single-chunk path must stay byte-identical to the chunked one. */
    @Test
    public void testSingleChunkMatchesChunked() throws IOException {
        int codeSize = 32, count = 200;
        Path p = newCacheFile(count, codeSize);

        try (FileChannel fc = FileChannel.open(p, StandardOpenOption.READ, StandardOpenOption.WRITE);
             PreEncodedCodeCache cache = PreEncodedCodeCache.map(fc, 0, count, codeSize, 1 << 30)) {
            assertEquals(1, cache.chunkCount());
            for (int i = 0; i < count; i++) {
                cache.put(i, codeFor(i, codeSize));
            }
            byte[] dst = new byte[codeSize];
            for (int i = 0; i < count; i++) {
                cache.get(i, dst);
                assertCodeMatches(i, codeSize, dst);
            }
        }
    }

    /** A chunk target that is not a multiple of codeSize must still never split a code. */
    @Test
    public void testChunkTargetNotMultipleOfCodeSize() throws IOException {
        int codeSize = 48, count = 300;
        int maxChunkBytes = 100;            // 2 codes per chunk (96 bytes), 4 bytes wasted per chunk
        Path p = newCacheFile(count, codeSize);

        try (FileChannel fc = FileChannel.open(p, StandardOpenOption.READ, StandardOpenOption.WRITE);
             PreEncodedCodeCache cache = PreEncodedCodeCache.map(fc, 0, count, codeSize, maxChunkBytes)) {
            assertEquals(150, cache.chunkCount());
            for (int i = 0; i < count; i++) {
                cache.put(i, codeFor(i, codeSize));
            }
            byte[] dst = new byte[codeSize];
            for (int i = count - 1; i >= 0; i--) {   // reverse order: no sequential-position luck
                cache.get(i, dst);
                assertCodeMatches(i, codeSize, dst);
            }
        }
    }

    /** The cache is mapped at a non-zero offset in production; indexing must be offset-relative. */
    @Test
    public void testNonZeroFileOffset() throws IOException {
        int codeSize = 16, count = 129;
        long offset = 4096 + 7;             // deliberately not page-aligned
        Path p = Files.createTempFile("code-cache-off-", ".bin");
        p.toFile().deleteOnExit();
        try (FileChannel fc = FileChannel.open(p, StandardOpenOption.WRITE)) {
            fc.write(ByteBuffer.wrap(new byte[]{0}), offset + PreEncodedCodeCache.sectionBytes(count, codeSize) - 1);
        }

        try (FileChannel fc = FileChannel.open(p, StandardOpenOption.READ, StandardOpenOption.WRITE);
             PreEncodedCodeCache cache = PreEncodedCodeCache.map(fc, offset, count, codeSize, codeSize * 5)) {
            for (int i = 0; i < count; i++) {
                cache.put(i, codeFor(i, codeSize));
            }
            byte[] dst = new byte[codeSize];
            for (int i = 0; i < count; i++) {
                cache.get(i, dst);
                assertCodeMatches(i, codeSize, dst);
            }
        }
    }

    /** copyInto is the write-path accessor; it must agree with get() and advance the destination. */
    @Test
    public void testCopyIntoMatchesGet() throws IOException {
        int codeSize = 24, count = 97;
        Path p = newCacheFile(count, codeSize);

        try (FileChannel fc = FileChannel.open(p, StandardOpenOption.READ, StandardOpenOption.WRITE);
             PreEncodedCodeCache cache = PreEncodedCodeCache.map(fc, 0, count, codeSize, codeSize * 3)) {
            for (int i = 0; i < count; i++) {
                cache.put(i, codeFor(i, codeSize));
            }
            ByteBuffer dst = ByteBuffer.allocate(codeSize * count);
            for (int i = 0; i < count; i++) {
                cache.copyInto(i, dst);
            }
            assertEquals("copyInto must advance the destination", codeSize * count, dst.position());
            byte[] all = dst.array();
            byte[] one = new byte[codeSize];
            for (int i = 0; i < count; i++) {
                System.arraycopy(all, i * codeSize, one, 0, codeSize);
                assertCodeMatches(i, codeSize, one);
            }
        }
    }

    /**
     * Reads race across threads in both the write and refine passes. Per-thread views must keep
     * absolute-position seeks from interfering.
     */
    @Test
    public void testConcurrentReadsAreIsolated() throws Exception {
        int codeSize = 48, count = 4000, threads = 8;
        Path p = newCacheFile(count, codeSize);
        ExecutorService pool = Executors.newFixedThreadPool(threads);

        try (FileChannel fc = FileChannel.open(p, StandardOpenOption.READ, StandardOpenOption.WRITE);
             PreEncodedCodeCache cache = PreEncodedCodeCache.map(fc, 0, count, codeSize, codeSize * 11)) {

            for (int i = 0; i < count; i++) {
                cache.put(i, codeFor(i, codeSize));
            }

            List<Callable<Boolean>> tasks = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                final int seed = t;
                tasks.add(() -> {
                    byte[] dst = new byte[codeSize];
                    // Interleave so threads hit the same chunks at overlapping times.
                    for (int pass = 0; pass < 4; pass++) {
                        for (int i = seed; i < count; i += threads) {
                            cache.get(i, dst);
                            for (int b = 0; b < codeSize; b++) {
                                if (dst[b] != (byte) ((i * 31 + b * 7) & 0xff)) {
                                    return false;
                                }
                            }
                        }
                    }
                    return true;
                });
            }
            for (Future<Boolean> f : pool.invokeAll(tasks)) {
                assertTrue("a worker observed a torn or mis-indexed read", f.get());
            }
        } finally {
            pool.shutdown();
            pool.awaitTermination(30, TimeUnit.SECONDS);
        }
    }

    /** Concurrent writes to disjoint ordinals are how precomputeCodes fills the cache. */
    @Test
    public void testConcurrentDisjointWrites() throws Exception {
        int codeSize = 48, count = 4000, threads = 8;
        Path p = newCacheFile(count, codeSize);
        ExecutorService pool = Executors.newFixedThreadPool(threads);

        try (FileChannel fc = FileChannel.open(p, StandardOpenOption.READ, StandardOpenOption.WRITE);
             PreEncodedCodeCache cache = PreEncodedCodeCache.map(fc, 0, count, codeSize, codeSize * 11)) {

            List<Callable<Void>> tasks = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                final int seed = t;
                tasks.add(() -> {
                    for (int i = seed; i < count; i += threads) {
                        cache.put(i, codeFor(i, codeSize));
                    }
                    return null;
                });
            }
            for (Future<Void> f : pool.invokeAll(tasks)) {
                f.get();
            }

            byte[] dst = new byte[codeSize];
            for (int i = 0; i < count; i++) {
                cache.get(i, dst);
                assertCodeMatches(i, codeSize, dst);
            }
        } finally {
            pool.shutdown();
            pool.awaitTermination(30, TimeUnit.SECONDS);
        }
    }

    /** close() unmaps; calling it twice must not blow up (onAfterClose can run after an error path). */
    @Test
    public void testCloseIsIdempotent() throws IOException {
        int codeSize = 8, count = 40;
        Path p = newCacheFile(count, codeSize);
        try (FileChannel fc = FileChannel.open(p, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            PreEncodedCodeCache cache = PreEncodedCodeCache.map(fc, 0, count, codeSize, codeSize * 3);
            cache.put(0, codeFor(0, codeSize));
            cache.close();
            cache.close();
        }
    }
}
