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

import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.misc.Unsafe;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.IntUnaryOperator;

/**
 * One source's base-layer records, distributed into bands of the merged ordinal space so the
 * merge's per-node reads of its own record come from a sequential spill instead of a random
 * seek into the source.
 *
 * <p>This is the distribute stage of {@code vector_merge_splat_design.md} §5, per source: the
 * source is swept once in its own ordinal order (the dense-regime scan), and each live node's
 * vector and base-layer adjacency — the two things the merge needs from a node's own record —
 * are appended to the spill of the band its <em>new</em> ordinal falls in. A band is a
 * contiguous range of {@code bandNodes} new ordinals; the records inside it arrive in old-ordinal
 * order, so a slot index ({@code new - start -> slot}) is kept while distributing and the band
 * file is memory-mapped for reads. Because similarity ordinals are assigned source by source,
 * one source's live nodes occupy one contiguous range {@code [start, start + liveCount)} of the
 * merged space and its bands are its own; scratch is one source's records, not the payload's,
 * and is released when the source's batches are done.
 *
 * <p>Spill record: {@code int old | float[dimension] | int count | int[degree]} — the fused codes
 * are not copied, since the output's codes come from the pre-encode cache, so the spill is the
 * vector and edges only (about 40% of a fused record). Appends are buffered per band under a
 * per-band lock, so distribute windows may run on the pool concurrently.
 */
final class BandStore implements AutoCloseable {
    private static final Logger log = LoggerFactory.getLogger(BandStore.class);
    /** Cap on one band file, so a {@link MappedByteBuffer} (int-indexed) always covers it. */
    static final long MAX_BAND_BYTES = 1L << 30;
    /** Per-band append buffer. */
    private static final int WRITE_BUFFER_BYTES = 256 << 10;
    private static final Unsafe UNSAFE = getUnsafe();

    final int sourceIdx;
    final int start;
    final int liveCount;
    final int dimension;
    final int degree;
    final int recordBytes;
    final int bandNodes;
    final int numBands;
    private final IntUnaryOperator oldToNew;
    private final Path dir;
    private final int[] slotOf;             // new - start -> slot within its band; -1 = not distributed
    private final Object[] locks;
    private final FileChannel[] channels;   // append channels while distributing
    private final ByteBuffer[] writeBufs;
    private final int[] counts;
    private final MappedByteBuffer[] maps;  // lazily mapped for reads
    private final LongAdder records = new LongAdder();
    private final LongAdder bytes = new LongAdder();
    private final LongAdder vectorsServed = new LongAdder();
    private final LongAdder neighborsServed = new LongAdder();
    private volatile boolean distributing = true;
    private int bandsMapped;

    /**
     * @param bandNodesRequested band width in new ordinals; clamped so a band file stays under
     *                           {@link #MAX_BAND_BYTES}
     */
    BandStore(Path spillParent, int sourceIdx, int start, int liveCount, int dimension, int degree,
              int bandNodesRequested, IntUnaryOperator oldToNew) throws IOException {
        this.sourceIdx = sourceIdx;
        this.start = start;
        this.liveCount = liveCount;
        this.dimension = dimension;
        this.degree = degree;
        this.recordBytes = Integer.BYTES + dimension * Float.BYTES + Integer.BYTES + degree * Integer.BYTES;
        int maxNodesPerBand = (int) Math.max(1, Math.min(Integer.MAX_VALUE, MAX_BAND_BYTES / recordBytes));
        this.bandNodes = Math.max(1, Math.min(bandNodesRequested, maxNodesPerBand));
        this.numBands = liveCount == 0 ? 0 : (liveCount + bandNodes - 1) / bandNodes;
        this.oldToNew = oldToNew;
        this.slotOf = new int[liveCount];
        Arrays.fill(slotOf, -1);
        this.locks = new Object[numBands];
        this.channels = new FileChannel[numBands];
        this.writeBufs = new ByteBuffer[numBands];
        this.counts = new int[numBands];
        this.maps = new MappedByteBuffer[numBands];
        for (int b = 0; b < numBands; b++) {
            locks[b] = new Object();
        }
        this.dir = Files.createTempDirectory(spillParent, "bands-src" + sourceIdx + "-");
    }

    private Path bandFile(int band) {
        return dir.resolve("band-" + band);
    }

    private int bandOf(int newOrdinal) {
        return (newOrdinal - start) / bandNodes;
    }

    /** Whether {@code old} of this store's source was distributed here. */
    boolean has(int old) {
        int n = oldToNew.applyAsInt(old) - start;
        return n >= 0 && n < liveCount && slotOf[n] >= 0;
    }

    /** Appends one live node's vector and base-layer edges to its band. Safe from concurrent windows. */
    void put(int old, VectorFloat<?> vector, NodesIterator neighbors) throws IOException {
        if (!distributing) {
            throw new IllegalStateException("distribute already finished");
        }
        int n = oldToNew.applyAsInt(old) - start;
        if (n < 0 || n >= liveCount) {
            throw new IllegalStateException("old ordinal " + old + " maps outside this source's range: new-start=" + n);
        }
        int count = neighbors.size();
        if (count > degree) {
            throw new IllegalStateException("node " + old + " has " + count + " neighbours, degree " + degree);
        }
        int band = n / bandNodes;
        synchronized (locks[band]) {
            ByteBuffer buf = writeBufs[band];
            if (buf == null) {
                buf = ByteBuffer.allocate(Math.max(WRITE_BUFFER_BYTES, recordBytes));
                writeBufs[band] = buf;
                channels[band] = FileChannel.open(bandFile(band), StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
            }
            if (buf.remaining() < recordBytes) {
                flush(band);
            }
            buf.putInt(old);
            for (int i = 0; i < dimension; i++) {
                buf.putFloat(vector.get(i));
            }
            buf.putInt(count);
            int k = 0;
            while (neighbors.hasNext()) {
                buf.putInt(neighbors.nextInt());
                k++;
            }
            for (; k < degree; k++) {
                buf.putInt(-1);
            }
            slotOf[n] = counts[band]++;
        }
        records.increment();
        bytes.add(recordBytes);
    }

    private void flush(int band) throws IOException {
        ByteBuffer buf = writeBufs[band];
        buf.flip();
        while (buf.hasRemaining()) {
            channels[band].write(buf);
        }
        buf.clear();
    }

    /** Flushes and closes every band for writing; reads may begin. */
    void finishDistribute() throws IOException {
        for (int b = 0; b < numBands; b++) {
            synchronized (locks[b]) {
                if (writeBufs[b] != null) {
                    flush(b);
                    channels[b].close();
                    channels[b] = null;
                    writeBufs[b] = null;
                }
            }
        }
        distributing = false;
    }

    private MappedByteBuffer map(int band) {
        MappedByteBuffer m = maps[band];
        if (m != null) {
            return m;
        }
        synchronized (locks[band]) {
            m = maps[band];
            if (m == null) {
                if (distributing) {
                    throw new IllegalStateException("reads before finishDistribute()");
                }
                try (FileChannel ch = FileChannel.open(bandFile(band), StandardOpenOption.READ)) {
                    long size = ch.size();
                    if (size != (long) counts[band] * recordBytes) {
                        throw new IllegalStateException("band " + band + " is " + size + " bytes, expected "
                                                        + (long) counts[band] * recordBytes);
                    }
                    m = ch.map(FileChannel.MapMode.READ_ONLY, 0, size);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
                maps[band] = m;
                bandsMapped++;
            }
            return m;
        }
    }

    private int recordOffset(int old) {
        int n = oldToNew.applyAsInt(old) - start;
        int slot = slotOf[n];
        if (slot < 0) {
            throw new IllegalStateException("node " + old + " of source " + sourceIdx + " was not distributed");
        }
        return slot * recordBytes;
    }

    /** The node's own vector, from its band. */
    void vectorInto(int old, VectorFloat<?> out) {
        int n = oldToNew.applyAsInt(old) - start;
        MappedByteBuffer m = map(n / bandNodes);
        int pos = recordOffset(old);
        int storedOld = m.getInt(pos);
        if (storedOld != old) {
            throw new IllegalStateException("band slot for " + old + " holds " + storedOld);
        }
        pos += Integer.BYTES;
        for (int i = 0; i < dimension; i++) {
            out.set(i, m.getFloat(pos + i * Float.BYTES));
        }
        vectorsServed.increment();
    }

    /** The node's base-layer edges, from its band, into {@code out}; returns the count. */
    int neighbors(int old, int[] out) {
        int n = oldToNew.applyAsInt(old) - start;
        MappedByteBuffer m = map(n / bandNodes);
        int pos = recordOffset(old);
        int storedOld = m.getInt(pos);
        if (storedOld != old) {
            throw new IllegalStateException("band slot for " + old + " holds " + storedOld);
        }
        pos += Integer.BYTES + dimension * Float.BYTES;
        int count = m.getInt(pos);
        if (count < 0 || count > degree || count > out.length) {
            throw new IllegalStateException("band record for " + old + " declares " + count + " neighbours");
        }
        pos += Integer.BYTES;
        for (int k = 0; k < count; k++) {
            out[k] = m.getInt(pos + k * Integer.BYTES);
        }
        neighborsServed.increment();
        return count;
    }

    long records() {
        return records.sum();
    }

    long bytes() {
        return bytes.sum();
    }

    long vectorsServed() {
        return vectorsServed.sum();
    }

    long neighborsServed() {
        return neighborsServed.sum();
    }

    int bandsMapped() {
        return bandsMapped;
    }

    Path directory() {
        return dir;
    }

    /** Unmaps and deletes the spill. */
    @Override
    public void close() {
        for (int b = 0; b < numBands; b++) {
            try {
                if (channels[b] != null) {
                    channels[b].close();
                    channels[b] = null;
                }
            } catch (IOException e) {
                log.warn("closing band {} of source {}", b, sourceIdx, e);
            }
            if (maps[b] != null) {
                if (UNSAFE != null) {
                    UNSAFE.invokeCleaner(maps[b]);
                }
                maps[b] = null;
            }
            try {
                Files.deleteIfExists(bandFile(b));
            } catch (IOException e) {
                log.warn("deleting band {} of source {}", b, sourceIdx, e);
            }
        }
        try {
            Files.deleteIfExists(dir);
        } catch (IOException e) {
            log.warn("deleting band directory {}", dir, e);
        }
    }

    private static Unsafe getUnsafe() {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            return (Unsafe) f.get(null);
        } catch (Exception e) {
            log.warn("BandStore can't acquire Unsafe; band files will not be explicitly unmapped");
            return null;
        }
    }
}
