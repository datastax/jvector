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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayDeque;
import java.util.Map;
import java.util.concurrent.atomic.LongAdder;

/**
 * Accumulates cross-source reverse candidates ("offers") during L0 compaction with peak memory
 * bounded by a chosen band width instead of the merged graph's node count. A resident per-target
 * slot buffer accumulates for the entire run on the largest (last-processed) source — ~23 GB of
 * live blocks in a 192M-node merge, ~110 GB projected at 1B; this class instead partitions the
 * target ordinal space into fixed-width bands. Offers append to a small per-band write buffer
 * that spills to a per-band file, sequentially. When processing reaches a band (the inter-source
 * barrier guarantees its offers are complete by then), the band's spill is replayed once through
 * dedup-by-candidate and top-{@code slots}-by-score selection, yielding the same per-target
 * candidate sets an offer-time selection would keep. Peak memory is O(write buffers) + O(loaded
 * bands), independent of node count; offers pass through disk once, as pure sequential I/O.
 * Spill files persist until {@link #close()}, so an evicted band can be reloaded — a resurrected
 * already-consumed target is never consumed twice (each target is processed exactly once).
 */
final class BandedReverseCandidateBuffer implements AutoCloseable {
    private static final Logger log = LoggerFactory.getLogger(BandedReverseCandidateBuffer.class);

    /** Offers per band held in RAM before spilling to the band's file (16K quads = 256KB). */
    private static final int BUFFER_QUADS = 16 * 1024;
    /** Loaded bands kept concurrently; batches near a band boundary touch two. */
    private static final int MAX_LOADED_BANDS = 8;

    private final int slots;
    private final int bandWidth;
    private final int bandsPerSource;
    private final int numBands;
    private final Path spillDir;
    private final LongAdder offered = new LongAdder();
    private final LongAdder touched = new LongAdder();
    private final LongAdder spilledBytes = new LongAdder();

    // Per-band spill state, guarded by bandLocks[band]
    private final int[][] buffers;      // [band] -> quads (target, src, oldOrd, scoreBits)
    private final int[] bufferCounts;   // in quads
    private final FileChannel[] channels;
    private final Object[] bandLocks;

    // Loaded (consumable) bands. Reads are lock-free (concurrent map); loads and eviction are
    // serialized by loadLock. Per-target block consumption needs no lock: each target is
    // consumed by exactly one thread.
    private final Object loadLock = new Object();
    private final Map<Integer, int[][]> loadedBands = new java.util.concurrent.ConcurrentHashMap<>();
    private final ArrayDeque<Integer> loadOrder = new ArrayDeque<>();

    BandedReverseCandidateBuffer(int numSources, int numOrdinals, int slots, int bandWidth, Path spillParent) {
        this.slots = slots;
        this.bandWidth = bandWidth;
        // Bands are keyed by (target source, ordinal band): a source's offers are complete
        // exactly when its own group starts (inter-source barrier), so bands must never mix
        // targets from different sources -- an ordinal band straddling two sources could
        // otherwise be loaded while the later source's offers are still arriving, losing them.
        this.bandsPerSource = (int) (((long) numOrdinals + bandWidth - 1) / bandWidth);
        this.numBands = numSources * bandsPerSource;
        this.buffers = new int[numBands][];
        this.bufferCounts = new int[numBands];
        this.channels = new FileChannel[numBands];
        this.bandLocks = new Object[numBands];
        for (int i = 0; i < numBands; i++) {
            bandLocks[i] = new Object();
        }
        try {
            this.spillDir = Files.createTempDirectory(spillParent, "reverse-offers-");
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public void offer(int targetSrc, int targetNewOrd, int src, int oldOrd, float score) {
        offered.increment();
        int band = targetSrc * bandsPerSource + targetNewOrd / bandWidth;
        synchronized (bandLocks[band]) {
            int[] buf = buffers[band];
            if (buf == null) {
                buf = new int[BUFFER_QUADS * 4];
                buffers[band] = buf;
            }
            int c = bufferCounts[band];
            buf[c * 4] = targetNewOrd;
            buf[c * 4 + 1] = src;
            buf[c * 4 + 2] = oldOrd;
            buf[c * 4 + 3] = Float.floatToRawIntBits(score);
            bufferCounts[band] = c + 1;
            if (c + 1 == BUFFER_QUADS) {
                flushBand(band);
            }
        }
    }

    // caller holds bandLocks[band]
    private void flushBand(int band) {
        int quads = bufferCounts[band];
        if (quads == 0) {
            return;
        }
        try {
            FileChannel ch = channels[band];
            if (ch == null) {
                ch = FileChannel.open(spillDir.resolve("band-" + band),
                        StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.APPEND);
                channels[band] = ch;
            }
            ByteBuffer bb = ByteBuffer.allocate(quads * 16).order(ByteOrder.LITTLE_ENDIAN);
            bb.asIntBuffer().put(buffers[band], 0, quads * 4);
            bb.limit(quads * 16);
            while (bb.hasRemaining()) {
                ch.write(bb);
            }
            spilledBytes.add(quads * 16L);
            bufferCounts[band] = 0;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public int countAt(int targetSrc, int targetNewOrd) {
        int[] b = blockFor(targetSrc, targetNewOrd, false);
        return b == null ? 0 : b[0];
    }

    public int appendTo(int targetSrc, int targetNewOrd, int[] candSrc, int[] candNode, float[] candScore,
                        int candSize) {
        int[] b = blockFor(targetSrc, targetNewOrd, true);
        if (b == null) {
            return candSize;
        }
        int n = b[0];
        for (int i = 0; i < n; i++) {
            candSrc[candSize] = b[1 + i];
            candNode[candSize] = b[1 + slots + i];
            candScore[candSize] = Float.intBitsToFloat(b[1 + 2 * slots + i]);
            candSize++;
        }
        return candSize;
    }

    private int[] blockFor(int targetSrc, int targetNewOrd, boolean consume) {
        int band = targetSrc * bandsPerSource + targetNewOrd / bandWidth;
        int[][] blocks = loadedBands.get(band);
        if (blocks == null) {
            blocks = loadBand(band);
        }
        int idx = targetNewOrd % bandWidth;
        int[] b = blocks[idx];
        if (consume) {
            blocks[idx] = null;
        }
        return b;
    }

    private int[][] loadBand(int band) {
        synchronized (loadLock) {
            int[][] existing = loadedBands.get(band);
            if (existing != null) {
                return existing;
            }
            int[][] blocks = new int[bandWidth][];
            synchronized (bandLocks[band]) {
                flushBand(band);
                buffers[band] = null;
            }
            Path file = spillDir.resolve("band-" + band);
            if (Files.exists(file)) {
                try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
                    ByteBuffer bb = ByteBuffer.allocate(BUFFER_QUADS * 16).order(ByteOrder.LITTLE_ENDIAN);
                    while (ch.read(bb) > 0) {
                        bb.flip();
                        while (bb.remaining() >= 16) {
                            replay(blocks, bb.getInt(), bb.getInt(), bb.getInt(), bb.getInt());
                        }
                        bb.compact();
                    }
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
            loadedBands.put(band, blocks);
            loadOrder.addLast(band);
            if (loadedBands.size() > MAX_LOADED_BANDS) {
                loadedBands.remove(loadOrder.pollFirst());
            }
            return blocks;
        }
    }

    /** Identical semantics to the in-RAM buffer's offer-time selection: dedup by (src, oldOrd),
     * keep the top-{@code slots} by score. */
    private void replay(int[][] blocks, int targetNewOrd, int src, int oldOrd, int scoreBits) {
        int idx = targetNewOrd % bandWidth;
        int[] b = blocks[idx];
        if (b == null) {
            b = new int[1 + 3 * slots];
            blocks[idx] = b;
            touched.increment();
        }
        int n = b[0];
        for (int i = 0; i < n; i++) {
            if (b[1 + i] == src && b[1 + slots + i] == oldOrd) {
                return;
            }
        }
        if (n < slots) {
            b[1 + n] = src;
            b[1 + slots + n] = oldOrd;
            b[1 + 2 * slots + n] = scoreBits;
            b[0] = n + 1;
            return;
        }
        float score = Float.intBitsToFloat(scoreBits);
        int minIdx = 0;
        float minScore = Float.intBitsToFloat(b[1 + 2 * slots]);
        for (int i = 1; i < slots; i++) {
            float s = Float.intBitsToFloat(b[1 + 2 * slots + i]);
            if (s < minScore) {
                minScore = s;
                minIdx = i;
            }
        }
        if (score > minScore) {
            b[1 + minIdx] = src;
            b[1 + slots + minIdx] = oldOrd;
            b[1 + 2 * slots + minIdx] = scoreBits;
        }
    }

    public long offered() {
        return offered.sum();
    }

    public long touchedTargets() {
        return touched.sum();
    }

    public long ramBytesUsed() {
        long size = 0;
        for (int i = 0; i < numBands; i++) {
            if (buffers[i] != null) {
                size += 16 + BUFFER_QUADS * 16L;
            }
        }
        // loaded bands: reference array plus a block per touched target, bounded by band width
        size += (long) loadedBands.size() * ((long) bandWidth * 8 + (long) bandWidth / 2 * (16 + (1 + 3L * slots) * 4));
        return size;
    }

    @Override
    public void close() {
        for (int i = 0; i < numBands; i++) {
            try {
                if (channels[i] != null) {
                    channels[i].close();
                }
            } catch (IOException ignored) {
            }
        }
        try (var paths = Files.list(spillDir)) {
            paths.forEach(p -> {
                try {
                    Files.delete(p);
                } catch (IOException ignored) {
                }
            });
            Files.delete(spillDir);
        } catch (IOException e) {
            log.warn("could not fully remove offer spill dir {}", spillDir, e);
        }
        log.info("Banded reverse offers: {} offers, {} touched targets, {} MB spilled",
                offered.sum(), touched.sum(), spilledBytes.sum() >> 20);
    }
}
