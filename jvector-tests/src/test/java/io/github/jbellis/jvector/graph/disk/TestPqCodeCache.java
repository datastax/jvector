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

import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static java.nio.file.StandardOpenOption.CREATE;
import static java.nio.file.StandardOpenOption.READ;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;
import static java.nio.file.StandardOpenOption.WRITE;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/// Proves the chunked {@link PqCodeCache} primitive that replaces the single-mapping fused-PQ code
/// cache: multi-chunk round-trip, power-of-two chunk sizing, and int-overflow-safe addressing for
/// ordinals whose old {@code ord * codeSize} offset would have overflowed {@code int}.
public class TestPqCodeCache {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    private static Path targetFile(String name) throws IOException {
        Path dir = Paths.get("target", "pqcodecache");
        Files.createDirectories(dir);
        return dir.resolve(name);
    }

    private static ByteSequence<?> codeFor(int ord, int codeSize) {
        ByteSequence<?> code = vts.createByteSequence(codeSize);
        for (int i = 0; i < codeSize; i++) {
            // distinct, ordinal- and position-dependent so a mis-addressed read is detected
            code.set(i, (byte) (ord * 31 + i * 7 + 1));
        }
        return code;
    }

    /// Write a distinct code per ordinal across many chunks, then read every one back through the
    /// reader views and assert exact identity. A wrong chunk index or within-chunk offset — the
    /// failure mode of the old single-mapping cache above ~21M nodes — would corrupt this.
    @Test
    public void roundTripAcrossManyChunks() throws IOException {
        int codeSize = 7;
        long maxChunkBytes = 20;   // budget/codeSize = 2, largest pow2 <= 2 is 2 -> 2 codes/chunk
        int numCodes = 11;         // -> ceil(11/2) = 6 chunks, last chunk partial (1 code)

        Path file = targetFile("roundtrip.bin");
        try (FileChannel fc = FileChannel.open(file, CREATE, READ, WRITE, TRUNCATE_EXISTING)) {
            PqCodeCache cache = PqCodeCache.map(fc, 0, codeSize, numCodes, maxChunkBytes);
            assertEquals("codes per chunk (power of two)", 2, cache.codesPerChunk());
            assertEquals("chunk count", 6, cache.chunkCount());
            assertEquals("shift = log2(codesPerChunk)", 1, cache.codesPerChunkShift());
            assertEquals("mask = codesPerChunk - 1", 1, cache.codesPerChunkMask());
            assertTrue("a mapped cache is active", cache.isActive());

            for (int ord = 0; ord < numCodes; ord++) {
                cache.putCode(ord, codeFor(ord, codeSize));
            }

            ByteBuffer[] views = cache.newViews();
            byte[] out = new byte[codeSize];
            for (int ord = 0; ord < numCodes; ord++) {
                cache.copyCode(views, ord, out);
                ByteSequence<?> expected = codeFor(ord, codeSize);
                for (int i = 0; i < codeSize; i++) {
                    assertEquals("ord " + ord + " byte " + i, expected.get(i), out[i]);
                }
            }
            cache.unmap();
        }
    }

    /// Chunk sizing is the largest power of two of codes that fits the (1 GiB-capped) budget.
    @Test
    public void codesPerChunkForIsPowerOfTwoWithinBudget() {
        assertEquals(8, PqCodeCache.codesPerChunkFor(100, 1000));  // 1000/100=10 -> 8
        assertEquals(1, PqCodeCache.codesPerChunkFor(100, 100));   // 1
        assertEquals(1, PqCodeCache.codesPerChunkFor(100, 50));    // code larger than budget -> 1
        assertEquals(8388608, PqCodeCache.codesPerChunkFor(100, 1L << 30)); // 2^30/100 -> 2^23

        for (long budget : new long[]{1, 63, 256, 999, 1L << 20, 1L << 30, (1L << 30) + 1}) {
            for (int codeSize : new int[]{1, 3, 8, 100, 999}) {
                int perChunk = PqCodeCache.codesPerChunkFor(codeSize, budget);
                assertTrue("power of two: " + perChunk, Integer.bitCount(perChunk) == 1);
                assertTrue("at least one code", perChunk >= 1);
                long effectiveBudget = Math.min(budget, PqCodeCache.DEFAULT_MAX_CHUNK_BYTES);
                assertTrue("fits budget unless a single code exceeds it",
                        (long) perChunk * codeSize <= effectiveBudget || perChunk == 1);
                // within-chunk offsets stay < 2^31 (the whole point of the chunked addressing)
                assertTrue("chunk span fits int", (long) perChunk * codeSize < (1L << 31));
            }
        }
    }

    /// The addressing (shift/mask) is int-overflow-safe for ordinals whose old
    /// {@code int offset = ord * codeSize} would have wrapped past {@code Integer.MAX_VALUE}. This is
    /// the exact cliff that silently disabled the single-mapping cache; here it is structurally gone.
    @Test
    public void addressingIsIntOverflowSafeForLargeOrdinals() throws IOException {
        int codeSize = 100;
        // Map a tiny cache purely to obtain the production shift/mask for this codeSize + 1 GiB budget.
        Path file = targetFile("addressing.bin");
        try (FileChannel fc = FileChannel.open(file, CREATE, READ, WRITE, TRUNCATE_EXISTING)) {
            PqCodeCache cache = PqCodeCache.map(fc, 0, codeSize, 4, PqCodeCache.DEFAULT_MAX_CHUNK_BYTES);
            int shift = cache.codesPerChunkShift();
            int mask = cache.codesPerChunkMask();
            int codesPerChunk = cache.codesPerChunk();

            int ord = 30_000_000; // ord * codeSize = 3.0e9, which overflows a signed int
            assertTrue("precondition: old int offset would overflow",
                    (long) ord * codeSize > Integer.MAX_VALUE);

            int chunkIndex = ord >>> shift;
            int within = (ord & mask) * codeSize;
            assertEquals("chunk index matches ord / codesPerChunk", ord / codesPerChunk, chunkIndex);
            assertTrue("within-chunk offset is non-negative (no overflow)", within >= 0);
            assertTrue("within-chunk offset stays inside the chunk",
                    within < (long) codesPerChunk * codeSize);
            cache.unmap();
        }
    }

    /// Internal dynamic bypass: a mapped cache is active and can be flipped off/on at runtime; the
    /// NONE sentinel is a composed-in-but-off cache holding no mappings.
    @Test
    public void dynamicBypassAndNoneSentinel() throws IOException {
        assertFalse("NONE reports inactive", PqCodeCache.NONE.isActive());
        assertEquals("NONE holds no chunks", 0, PqCodeCache.NONE.chunkCount());

        Path file = targetFile("bypass.bin");
        try (FileChannel fc = FileChannel.open(file, CREATE, READ, WRITE, TRUNCATE_EXISTING)) {
            PqCodeCache cache = PqCodeCache.map(fc, 0, 4, 8, PqCodeCache.DEFAULT_MAX_CHUNK_BYTES);
            assertTrue(cache.isActive());
            cache.bypass();
            assertFalse("bypass() disables", cache.isActive());
            cache.setActive(true);
            assertTrue("setActive(true) re-enables", cache.isActive());
            cache.unmap();
        }
    }

    /// Config: compose-in/leave-off and chunk sizing are independent, immutable settings.
    @Test
    public void configComposition() {
        assertTrue(PqCodeCacheConfig.DEFAULT.enabled());
        assertEquals(PqCodeCache.DEFAULT_MAX_CHUNK_BYTES, PqCodeCacheConfig.DEFAULT.maxChunkBytes());
        assertFalse(PqCodeCacheConfig.DISABLED.enabled());

        assertFalse(PqCodeCacheConfig.DEFAULT.withEnabled(false).enabled());
        assertEquals(256, PqCodeCacheConfig.DEFAULT.withMaxChunkBytes(256).maxChunkBytes());
        // withX preserves the other field
        assertTrue(PqCodeCacheConfig.DEFAULT.withMaxChunkBytes(256).enabled());
    }
}
