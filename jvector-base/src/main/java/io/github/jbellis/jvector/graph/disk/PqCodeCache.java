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

/// Chunked, memory-mapped pre-encode cache of fused-PQ codes, indexed by new (post-remap) ordinal.
///
/// Replaces the single-[MappedByteBuffer] cache, whose `Integer.MAX_VALUE` (2 GiB) mapping limit
/// silently disabled the cache — forcing roughly `degree ×` encode amplification — once
/// `(maxOrdinal + 1) * codeSize` crossed `2^31`. Codes are split across an array of mappings, each
/// holding a whole number of codes and at most `maxChunkBytes` (≤ 1 GiB), so total capacity is
/// bounded only by disk. A code never straddles a chunk boundary, so every read/write is a single
/// contiguous `codeSize`-byte access.
///
/// **Uniform, power-of-two chunks.** Every chunk spans exactly `codesPerChunk` ordinals, and
/// `codesPerChunk` is a power of two ([#codesPerChunkFor]). The global-ordinal address therefore
/// reduces to register shift/mask rather than integer division:
/// ```
/// chunk  = ord >>> codesPerChunkShift        // ord / codesPerChunk
/// within = (ord & codesPerChunkMask) * codeSize   // (ord % codesPerChunk) * codeSize
/// ```
/// Because `within < codesPerChunk * codeSize <= maxChunkBytes <= 1 GiB`, it always fits an `int` —
/// the old `int offset = ord * codeSize` overflow is structurally impossible.
///
/// Threading mirrors the previous cache: writes use absolute [MappedByteBuffer#put(int, byte)] (the
/// pre-encode tasks cover disjoint ordinals, so concurrent absolute puts never race), and readers
/// each take their own [#newViews()] duplicates so relative-position reads don't share a cursor.
/// jvector-base targets `--release 11`, so this uses NIO mappings rather than a
/// `java.lang.foreign` `MemorySegment`.
public final class PqCodeCache {
    private static final Logger log = LoggerFactory.getLogger(PqCodeCache.class);
    private static final Unsafe UNSAFE = getUnsafe();

    /// Maximum bytes per chunk mapping: 1 GiB. Also the hard ceiling enforced by
    /// [#codesPerChunkFor], which keeps `codesPerChunk * codeSize` (and thus every within-chunk
    /// offset) below `2^31`, comfortably under the 2 GiB single-mapping limit.
    public static final long DEFAULT_MAX_CHUNK_BYTES = 1L << 30;

    /// Null-object sentinel for "no active cache", paralleling [QuantizationCompactionStrategy#NONE].
    /// It holds no mappings, so [#isActive()] is permanently `false` and every consumer takes its
    /// encode fallback. Returned instead of `null` when the cache is composed in but configured off
    /// (see [PqCodeCacheConfig]), keeping the object graph uniform. Safe to share as a singleton: a
    /// chunk-less cache is inactive regardless of the bypass flag, so it cannot be toggled into a
    /// state that would read from absent mappings.
    public static final PqCodeCache NONE = new PqCodeCache(new MappedByteBuffer[0], 0, 1, false);

    private final MappedByteBuffer[] chunks;
    private final int codeSize;
    private final int codesPerChunk;        // power of two: uniform stride across chunks
    private final int codesPerChunkShift;   // log2(codesPerChunk): ord -> chunk index
    private final int codesPerChunkMask;    // codesPerChunk - 1: ord -> in-chunk code index

    /// Internal bypass. When {@code false}, consumers must encode rather than read this cache; a
    /// mapped cache starts active and can be flipped off at runtime via [#setActive] / [#bypass].
    /// Volatile so a mid-run toggle is observed by reader threads. Bypass is always correctness-safe
    /// because a memcpy from the cache and a fresh `encodeTo` produce identical bytes for the same
    /// compressor, so consumers may switch paths per record.
    private volatile boolean active;

    private PqCodeCache(MappedByteBuffer[] chunks, int codeSize, int codesPerChunk, boolean active) {
        assert Integer.bitCount(codesPerChunk) == 1 : "codesPerChunk must be a power of two, got " + codesPerChunk;
        this.chunks = chunks;
        this.codeSize = codeSize;
        this.codesPerChunk = codesPerChunk;
        this.codesPerChunkShift = Integer.numberOfTrailingZeros(codesPerChunk);
        this.codesPerChunkMask = codesPerChunk - 1;
        this.active = active;
    }

    /// Maps a code cache over `[baseOffset, baseOffset + numCodes * codeSize)` of `fc`, growing the
    /// file if needed. The region is split into uniform chunks of `codesPerChunk` codes each (the
    /// last chunk maps only the remaining codes), every chunk a separate [MappedByteBuffer].
    ///
    /// @param fc           the output channel, opened for read and write
    /// @param baseOffset   byte offset at which the cache region begins
    /// @param codeSize     bytes per code (`> 0`)
    /// @param numCodes     number of codes to hold, one per new ordinal (`> 0`)
    /// @param maxChunkBytes soft cap on bytes per chunk; values above 1 GiB are clamped down
    public static PqCodeCache map(FileChannel fc, long baseOffset, int codeSize, long numCodes, long maxChunkBytes) throws IOException {
        if (codeSize <= 0) {
            throw new IllegalArgumentException("codeSize must be > 0, got " + codeSize);
        }
        if (numCodes <= 0) {
            throw new IllegalArgumentException("numCodes must be > 0, got " + numCodes);
        }
        int codesPerChunk = codesPerChunkFor(codeSize, maxChunkBytes);
        long totalBytes = numCodes * codeSize;

        // Ensure the backing file spans the whole cache region before mapping any chunk.
        long end = baseOffset + totalBytes;
        if (fc.size() < end) {
            ByteBuffer pad = ByteBuffer.wrap(new byte[]{0});
            fc.write(pad, end - 1);
        }

        int numChunks = (int) ((numCodes + codesPerChunk - 1) / codesPerChunk);
        MappedByteBuffer[] chunks = new MappedByteBuffer[numChunks];
        for (int c = 0; c < numChunks; c++) {
            long firstCode = (long) c * codesPerChunk;
            long chunkOffset = baseOffset + firstCode * codeSize;
            long thisChunkCodes = Math.min(codesPerChunk, numCodes - firstCode);
            long thisChunkBytes = thisChunkCodes * codeSize;
            chunks[c] = fc.map(FileChannel.MapMode.READ_WRITE, chunkOffset, thisChunkBytes);
        }
        return new PqCodeCache(chunks, codeSize, codesPerChunk, true);
    }

    /// Largest power-of-two number of codes that fits in `maxChunkBytes`, at least 1. Power-of-two
    /// sizing makes the per-chunk stride uniform so ordinal addressing is shift/mask (see class
    /// doc). The budget is capped at [#DEFAULT_MAX_CHUNK_BYTES] so that `codesPerChunk * codeSize` —
    /// the largest within-chunk byte offset — stays below `2^31` and cannot overflow `int`.
    public static int codesPerChunkFor(int codeSize, long maxChunkBytes) {
        long budget = Math.min(maxChunkBytes, DEFAULT_MAX_CHUNK_BYTES);
        long perChunk = budget / codeSize;
        if (perChunk < 1) {
            return 1;   // 2^0: a single code larger than the budget still gets its own chunk
        }
        // Largest power of two <= perChunk. perChunk <= 1 GiB / codeSize <= 2^30, so this fits int.
        return Integer.highestOneBit((int) perChunk);
    }

    /// Bytes per code held in this cache.
    public int codeSize() {
        return codeSize;
    }

    /// Number of codes stored per chunk (a power of two). The last chunk may hold fewer.
    public int codesPerChunk() {
        return codesPerChunk;
    }

    /// `log2(codesPerChunk)`: the shift that maps a new ordinal to its chunk index.
    public int codesPerChunkShift() {
        return codesPerChunkShift;
    }

    /// `codesPerChunk - 1`: the mask that maps a new ordinal to its in-chunk code index.
    public int codesPerChunkMask() {
        return codesPerChunkMask;
    }

    /// Number of chunk mappings backing this cache.
    public int chunkCount() {
        return chunks.length;
    }

    /// Whether consumers should read codes from this cache ({@code true}) or take their encode
    /// fallback ({@code false}). Read once per record on the write/refine hot paths. A chunk-less
    /// cache (the [#NONE] sentinel) is never active regardless of the bypass flag, which keeps the
    /// shared sentinel a safe, immutable-in-effect null-object.
    public boolean isActive() {
        return active && chunks.length > 0;
    }

    /// Dynamically enables or disables this cache. Bypass ({@code false}) is correctness-safe: a
    /// consumer that encodes instead of reading produces identical bytes for the same compressor.
    public void setActive(boolean active) {
        this.active = active;
    }

    /// Convenience for {@code setActive(false)} — dynamically bypasses this cache.
    public void bypass() {
        this.active = false;
    }

    /// Per-reader duplicate views (one per chunk) so the relative-position reads in [#copyCode]
    /// don't share a cursor with other threads. The duplicates are cheap wrappers over the same
    /// mapped memory; hand a distinct array to each reader thread.
    public ByteBuffer[] newViews() {
        ByteBuffer[] views = new ByteBuffer[chunks.length];
        for (int c = 0; c < chunks.length; c++) {
            views[c] = chunks[c].duplicate();
        }
        return views;
    }

    /// Writes `code` (length `codeSize`) at new ordinal `ord` using absolute puts. Thread-safe for
    /// concurrent callers as long as they target disjoint ordinals (as the pre-encode tasks do).
    public void putCode(int ord, ByteSequence<?> code) {
        MappedByteBuffer chunk = chunks[ord >>> codesPerChunkShift];
        int within = (ord & codesPerChunkMask) * codeSize;
        for (int i = 0; i < codeSize; i++) {
            chunk.put(within + i, code.get(i));
        }
    }

    /// Copies the `codeSize` bytes stored at new ordinal `ord` into `out`, using this reader's own
    /// `views` (from [#newViews()]). `out` must be at least `codeSize` long.
    public void copyCode(ByteBuffer[] views, int ord, byte[] out) {
        ByteBuffer view = views[ord >>> codesPerChunkShift];
        view.position((ord & codesPerChunkMask) * codeSize);
        view.get(out, 0, codeSize);
    }

    /// Explicitly unmaps every chunk mapping. Best-effort (no-op if `Unsafe` is unavailable); call
    /// once, before truncating the cache section off the output file.
    public void unmap() {
        if (UNSAFE == null) {
            return;
        }
        for (MappedByteBuffer chunk : chunks) {
            if (chunk == null) {
                continue;
            }
            try {
                UNSAFE.invokeCleaner(chunk);
            } catch (IllegalArgumentException ignored) {
                // duplicated/indirect buffer; not cleanable
            }
        }
    }

    private static Unsafe getUnsafe() {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            return (Unsafe) f.get(null);
        } catch (Exception e) {
            log.warn("PqCodeCache can't acquire needed Unsafe access; mapped chunks will not be explicitly unmapped");
            return null;
        }
    }
}
