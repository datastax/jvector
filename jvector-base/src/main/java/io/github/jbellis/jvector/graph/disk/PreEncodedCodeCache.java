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

import io.github.jbellis.jvector.vector.types.ByteSequence;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.misc.Unsafe;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * A memory-mapped array of pre-encoded fused codes, indexed by new ordinal.
 *
 * <p>Compaction pre-encodes every live node's code once, in parallel, so the write and refine
 * passes can memcpy a neighbor's code instead of re-encoding it. Because a fused graph stores
 * each node's <em>neighbors'</em> codes inline, a code is needed once per edge rather than once
 * per node — so without this cache the encode cost is multiplied by the graph degree.
 *
 * <p>The backing region is a section appended past the projected end of the output graph file,
 * truncated away once compaction completes.
 *
 * <h2>Why chunked</h2>
 * {@link MappedByteBuffer} extends {@link java.nio.Buffer}, whose index is an {@code int}, so a
 * single mapping cannot exceed 2 GiB. That capped the cache at
 * {@code floor(Integer.MAX_VALUE / codeSize)} nodes. Since {@code codeSize} grows with dimension,
 * the cap falls as dimension rises; it also applies to the total node count across all partitions
 * being compacted rather than to any single index. Beyond the cap the pre-encode pass was skipped
 * entirely and compaction fell back to re-encoding per edge, which does not merely cost a constant
 * factor more: it changes the growth rate, because each of a node's {@code degree} neighbors is
 * re-encoded from its full float vector.
 *
 * <p>Splitting the region into several mappings removes the ceiling. Each chunk holds a whole
 * number of codes ({@link #MAX_CHUNK_BYTES} rounded <em>down</em> to a multiple of
 * {@code codeSize}), so no code straddles a mapping boundary — that invariant is what lets every
 * accessor resolve a code with one division and stay within a single chunk.
 *
 * <p>Instances are shared across worker threads. {@link #get} and {@link #put} are safe to call
 * concurrently: reads use per-thread duplicate views so absolute-position seeks cannot race, and
 * writes from the pre-encode pass target disjoint ordinals.
 */
public final class PreEncodedCodeCache implements AutoCloseable {
    private static final Logger log = LoggerFactory.getLogger(PreEncodedCodeCache.class);

    /**
     * Target bytes per mapping. Rounded down to a whole number of codes, so the effective chunk
     * is at most this and always under the 2 GiB single-mapping limit.
     */
    static final int MAX_CHUNK_BYTES = 1 << 30;

    private static final Unsafe UNSAFE = getUnsafe();

    private static Unsafe getUnsafe() {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            return (Unsafe) f.get(null);
        } catch (Exception e) {
            log.warn("PreEncodedCodeCache can't acquire needed Unsafe access; "
                    + "the code cache will not be explicitly unmapped");
            return null;
        }
    }

    private final MappedByteBuffer[] chunks;
    private final int codeSize;
    private final int codesPerChunk;
    private final int count;
    private final ThreadLocal<ByteBuffer[]> viewsPerThread;

    private PreEncodedCodeCache(MappedByteBuffer[] chunks, int codeSize, int codesPerChunk, int count) {
        this.chunks = chunks;
        this.codeSize = codeSize;
        this.codesPerChunk = codesPerChunk;
        this.count = count;
        this.viewsPerThread = ThreadLocal.withInitial(() -> {
            ByteBuffer[] views = new ByteBuffer[chunks.length];
            for (int i = 0; i < chunks.length; i++) {
                views[i] = chunks[i].duplicate();
            }
            return views;
        });
    }

    /**
     * Maps a cache for {@code count} codes of {@code codeSize} bytes each, starting at
     * {@code offset} in {@code fc}. The file must already be at least
     * {@code offset + (long) count * codeSize} bytes long.
     *
     * @param fc       channel opened for read and write
     * @param offset   byte offset of the cache section within the file
     * @param count    number of codes (== maxOrdinal + 1)
     * @param codeSize bytes per code; must be positive
     */
    public static PreEncodedCodeCache map(FileChannel fc, long offset, int count, int codeSize)
            throws IOException {
        return map(fc, offset, count, codeSize, MAX_CHUNK_BYTES);
    }

    /**
     * As {@link #map(FileChannel, long, int, int)}, with an explicit chunk target so tests can
     * exercise the multi-chunk path without allocating gigabytes. Production callers use the
     * four-argument form.
     */
    static PreEncodedCodeCache map(FileChannel fc, long offset, int count, int codeSize, int maxChunkBytes)
            throws IOException {
        if (codeSize <= 0) {
            throw new IllegalArgumentException("codeSize must be positive, got " + codeSize);
        }
        if (count <= 0) {
            throw new IllegalArgumentException("count must be positive, got " + count);
        }
        if (maxChunkBytes <= 0) {
            throw new IllegalArgumentException("maxChunkBytes must be positive, got " + maxChunkBytes);
        }

        int codesPerChunk = maxChunkBytes / codeSize;
        if (codesPerChunk == 0) {
            // A single code larger than the chunk target; give it a chunk of its own.
            codesPerChunk = 1;
        }
        int numChunks = (int) (((long) count + codesPerChunk - 1) / codesPerChunk);

        MappedByteBuffer[] chunks = new MappedByteBuffer[numChunks];
        for (int i = 0; i < numChunks; i++) {
            long firstCode = (long) i * codesPerChunk;
            long codesHere = Math.min(codesPerChunk, count - firstCode);
            chunks[i] = fc.map(FileChannel.MapMode.READ_WRITE,
                               offset + firstCode * codeSize,
                               codesHere * codeSize);
        }
        return new PreEncodedCodeCache(chunks, codeSize, codesPerChunk, count);
    }

    /** Total bytes spanned by the cache section, for sizing and truncation. */
    public static long sectionBytes(int count, int codeSize) {
        return (long) count * codeSize;
    }

    /** Bytes per code. */
    public int codeSize() {
        return codeSize;
    }

    /** Number of codes the cache holds. */
    public int count() {
        return count;
    }

    /** Number of mappings backing this cache; exposed for logging and tests. */
    int chunkCount() {
        return chunks.length;
    }

    /** Writes {@code code} at {@code ordinal}. Callers must not write the same ordinal twice concurrently. */
    public void put(int ordinal, ByteSequence<?> code) {
        int chunk = ordinal / codesPerChunk;
        int offset = (ordinal - chunk * codesPerChunk) * codeSize;
        ByteBuffer buf = viewsPerThread.get()[chunk];
        for (int i = 0; i < codeSize; i++) {
            buf.put(offset + i, code.get(i));
        }
    }

    /** Copies the code at {@code ordinal} into {@code dst}, which must hold at least {@link #codeSize()} bytes. */
    public void get(int ordinal, byte[] dst) {
        int chunk = ordinal / codesPerChunk;
        int offset = (ordinal - chunk * codesPerChunk) * codeSize;
        ByteBuffer view = viewsPerThread.get()[chunk];
        view.position(offset);
        view.get(dst, 0, codeSize);
    }

    /**
     * Copies the code at {@code ordinal} directly into {@code dst} at its current position,
     * advancing it by {@link #codeSize()} bytes. Saves the intermediate byte[] hop on the
     * write path.
     */
    public void copyInto(int ordinal, ByteBuffer dst) {
        int chunk = ordinal / codesPerChunk;
        int offset = (ordinal - chunk * codesPerChunk) * codeSize;
        ByteBuffer view = viewsPerThread.get()[chunk];
        for (int i = 0; i < codeSize; i++) {
            dst.put(view.get(offset + i));
        }
    }

    /** Unmaps every chunk. Safe to call more than once. */
    @Override
    public void close() {
        if (UNSAFE == null) {
            return;
        }
        for (int i = 0; i < chunks.length; i++) {
            if (chunks[i] == null) {
                continue;
            }
            try {
                UNSAFE.invokeCleaner(chunks[i]);
            } catch (IllegalArgumentException ignored) {
                // duplicated/indirect buffer; not cleanable
            }
            chunks[i] = null;
        }
    }
}
