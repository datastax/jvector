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

import io.github.jbellis.jvector.disk.ByteBufferIndexWriter;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.AsynchronousFileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.util.function.IntFunction;

/**
 * A task that writes L0 records for a range of nodes to disk via an AsynchronousFileChannel.
 * <p>
 * Each task processes a contiguous range of ordinals.  Two execution paths exist:
 * <p>
 * <b>Fast path</b> (no pre-written features): all records in the range are built into one
 * contiguous {@link ByteBuffer} and written with a single {@code channel.write()} call.
 * <p>
 * <b>Legacy path</b> (pre-written features present): some feature regions in each node record
 * were written to disk ahead of time via {@code writeFeaturesInline()}.  Those byte ranges must
 * not be overwritten.  {@link #callLegacy()} builds one growing in-memory buffer for the whole
 * task range, but skips writing anything into it for a pre-written (gap) feature — the buffer
 * ends up holding only owned bytes, packed contiguously.  A run of owned bytes is sliced off and
 * flushed as a single non-blocking write only when a gap is reached (or at the end of the task),
 * so a run commonly spans <em>many</em> nodes rather than being flushed per-node: the seam
 * between one node's trailing neighbor section and the next node's leading ordinal is never
 * itself a gap, so runs only break where a {@link FeatureId} is actually pre-written, regardless
 * of node boundaries. All writes for the entire task are submitted before any
 * {@code Future.get()} call, so the OS sees the full I/O workload and can schedule it efficiently.
 * <p>
 * <b>Understanding {@code hasPrewrittenFeatures}</b>: this flag is derived from the
 * {@code featureStateSuppliers} map passed to {@code write()}.  It does <em>not</em> involve
 * any read-before-write.  The mechanism is purely contractual: a client that calls
 * {@code writeFeaturesInline(ordinal, stateMap)} before {@code write(featureStateSuppliers)}
 * simply omits those {@link FeatureId}s from the suppliers map.
 * {@code featureStateSuppliers.get(featureId) == null} is the signal "those bytes are already
 * on disk — do not touch them."  The flag is computed once at construction time so the
 * per-node hot path pays no overhead checking it.
 * <p>
 * FUTURE IMPROVEMENT: when {@code writeFeaturesInline} support is removed, {@code hasPrewrittenFeatures}
 * will always be {@code false}, the legacy path ({@link #callLegacy()}) and all related helpers
 * can be deleted, and this class becomes the clean single-path fast path only.
 */
class NodeRecordTask implements Callable<Void> {
    private final int startOrdinal;
    private final int endOrdinal;
    private final OrdinalMapper ordinalMapper;
    private final ImmutableGraphIndex graph;
    private final ImmutableGraphIndex.View view;
    private final List<Feature> inlineFeatures;
    private final Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers;
    private final int recordSize;
    private final long baseOffset;
    private final AsynchronousFileChannel channel;
    private final boolean useDirectBuffers;

    // FUTURE IMPROVEMENT: when writeFeaturesInline is removed this flag is always false,
    // the callLegacy() branch disappears, and this field can be deleted entirely.
    private final boolean hasPrewrittenFeatures;

    NodeRecordTask(int startOrdinal,
                   int endOrdinal,
                   OrdinalMapper ordinalMapper,
                   ImmutableGraphIndex graph,
                   ImmutableGraphIndex.View view,
                   List<Feature> inlineFeatures,
                   Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers,
                   int recordSize,
                   long baseOffset,
                   AsynchronousFileChannel channel,
                   boolean useDirectBuffers) {
        this.startOrdinal = startOrdinal;
        this.endOrdinal = endOrdinal;
        this.ordinalMapper = ordinalMapper;
        this.graph = graph;
        this.view = view;
        this.inlineFeatures = inlineFeatures;
        this.featureStateSuppliers = featureStateSuppliers;
        this.recordSize = recordSize;
        this.baseOffset = baseOffset;
        this.channel = channel;
        this.useDirectBuffers = useDirectBuffers;
        // Null supplier for any inline feature means the caller omitted it from the write()
        // suppliers map, signalling that writeFeaturesInline() already placed that data on disk.
        // FUTURE IMPROVEMENT: remove this field once writeFeaturesInline support is dropped.
        this.hasPrewrittenFeatures = inlineFeatures.stream()
                .anyMatch(f -> featureStateSuppliers.get(f.id()) == null);
    }

    @Override
    public Void call() throws Exception {
        // FUTURE IMPROVEMENT: once writeFeaturesInline is removed this dispatch goes away —
        // callBatched() becomes the only execution path.
        if (hasPrewrittenFeatures) {
            callLegacy();
        } else {
            callBatched();
        }
        return null;
    }

    // -------------------------------------------------------------------------
    // Fast path: one contiguous buffer for the entire ordinal range, one write.
    // -------------------------------------------------------------------------

    private void callBatched() throws Exception {
        int rangeSize = endOrdinal - startOrdinal;
        ByteBuffer rangeBuffer = useDirectBuffers
                ? ByteBuffer.allocateDirect(rangeSize * recordSize)
                : ByteBuffer.allocate(rangeSize * recordSize);
        rangeBuffer.order(java.nio.ByteOrder.BIG_ENDIAN);

        // ByteBufferIndexWriter clears the buffer on construction; since it was just
        // allocated that is a no-op, but it sets initialPosition = 0 as required.
        var writer = new ByteBufferIndexWriter(rangeBuffer);

        for (int newOrdinal = startOrdinal; newOrdinal < endOrdinal; newOrdinal++) {
            buildFullRecord(writer, newOrdinal);
        }

        // One channel.write() for the entire task range — one syscall, one OS I/O request
        // in the common case. writeAllFully() guarantees the buffer is fully drained even
        // if the OS reports a short write (e.g. disk full).
        rangeBuffer.flip();
        writeAllFully(List.of(new PendingWrite(rangeBuffer, baseOffset + (long) startOrdinal * recordSize)));
    }

    // -------------------------------------------------------------------------
    // Legacy path: handle pre-written feature regions.
    //
    // Pre-written bytes must not be overwritten. A single growing buffer is built for the
    // whole task range, same as callBatched(), except nothing is written into it for a
    // pre-written (gap) feature -- the buffer ends up holding only owned bytes, packed
    // contiguously. A run of owned bytes is sliced off the buffer and queued as a single
    // non-blocking write only when a gap is reached (or at the end of the task), so runs
    // commonly span many nodes rather than being flushed per-node: the seam between one
    // node's trailing neighbor section and the next node's leading ordinal is never itself
    // a gap, so runs only break where a FeatureId is actually pre-written. ALL writes for
    // the entire task are submitted before any Future.get() call, letting the OS pipeline
    // them.
    //
    // FUTURE IMPROVEMENT: delete this method entirely once writeFeaturesInline support is
    // removed. The fast path handles everything.
    // -------------------------------------------------------------------------

    private void callLegacy() throws Exception {
        int rangeSize = endOrdinal - startOrdinal;
        ByteBuffer rangeBuffer = useDirectBuffers
                ? ByteBuffer.allocateDirect(rangeSize * recordSize)
                : ByteBuffer.allocate(rangeSize * recordSize);
        rangeBuffer.order(java.nio.ByteOrder.BIG_ENDIAN);
        var writer = new ByteBufferIndexWriter(rangeBuffer);

        List<PendingWrite> pending = new ArrayList<>();
        // Buffer-relative start of the run currently being accumulated, and the file
        // position that buffer offset corresponds to. Advanced past both the flushed run
        // and the skipped gap every time a gap is hit.
        int runStart = 0;
        long runFilePosition = baseOffset + (long) startOrdinal * recordSize;

        for (int newOrdinal = startOrdinal; newOrdinal < endOrdinal; newOrdinal++) {
            var originalOrdinal = ordinalMapper.newToOld(newOrdinal);

            if (originalOrdinal == OrdinalMapper.OMITTED) {
                // OMITTED nodes are holes in the ordinal space. writeFeaturesInline() is
                // never called for them, so nothing here is a gap: the whole record extends
                // the current run with no flush needed.
                writer.writeInt(newOrdinal);
                for (var feature : inlineFeatures) {
                    for (int i = 0; i < feature.featureSize(); i++) writer.writeByte(0);
                }
                writer.writeInt(0); // neighbor count
                for (int n = 0; n < graph.getDegree(0); n++) writer.writeInt(-1);
                continue;
            }

            if (!graph.containsNode(originalOrdinal)) {
                throw new IllegalStateException(String.format(
                        "Ordinal mapper mapped new ordinal %d to non-existing node %d",
                        newOrdinal, originalOrdinal));
            }

            // Ordinal: always owned.
            writer.writeInt(newOrdinal);

            for (var feature : inlineFeatures) {
                var supplier = featureStateSuppliers.get(feature.id());
                if (supplier != null) {
                    // Owned: extend the current run directly.
                    feature.writeInline(writer, supplier.apply(originalOrdinal));
                } else {
                    // Pre-written gap: flush the run accumulated so far (which may span
                    // multiple earlier nodes), then skip the gap in the file without
                    // writing anything into rangeBuffer for it.
                    int runEnd = rangeBuffer.position();
                    if (runEnd > runStart) {
                        pending.add(new PendingWrite(sliceRange(rangeBuffer, runStart, runEnd), runFilePosition));
                    }
                    runFilePosition += (runEnd - runStart) + feature.featureSize();
                    runStart = runEnd;
                }
            }

            // Neighbor section: always owned — extends the current run.
            var neighbors = view.getNeighborsIterator(0, originalOrdinal);
            if (neighbors.size() > graph.getDegree(0)) {
                throw new IllegalStateException(String.format(
                        "Node %d has more neighbors %d than max degree %d -- run Builder.cleanup()!",
                        originalOrdinal, neighbors.size(), graph.getDegree(0)));
            }
            writer.writeInt(neighbors.size());
            int n = 0;
            for (; n < neighbors.size(); n++) {
                int newNeighbor = ordinalMapper.oldToNew(neighbors.nextInt());
                if (newNeighbor < 0 || newNeighbor > ordinalMapper.maxOrdinal()) {
                    throw new IllegalStateException(String.format(
                            "Neighbor ordinal out of bounds: %d/%d",
                            newNeighbor, ordinalMapper.maxOrdinal()));
                }
                writer.writeInt(newNeighbor);
            }
            for (; n < graph.getDegree(0); n++) writer.writeInt(-1);
        }

        // Final trailing run.
        int runEnd = rangeBuffer.position();
        if (runEnd > runStart) {
            pending.add(new PendingWrite(sliceRange(rangeBuffer, runStart, runEnd), runFilePosition));
        }

        writeAllFully(pending);
    }

    /**
     * Returns an independent, ready-to-read view over {@code buf}'s backing storage covering
     * {@code [start, end)}. The view shares memory with {@code buf} but has its own position
     * and limit, so later writes into {@code buf} at other offsets don't affect it — safe to
     * hand off to a concurrent {@code channel.write()} while {@code buf} keeps being appended to.
     */
    private static ByteBuffer sliceRange(ByteBuffer buf, int start, int end) {
        ByteBuffer dup = buf.duplicate();
        dup.limit(end);
        dup.position(start);
        return dup.slice();
    }

    // -------------------------------------------------------------------------
    // Shared helpers
    // -------------------------------------------------------------------------

    /**
     * Writes a complete node record (ordinal + all features + neighbors) sequentially
     * into {@code writer}.  Called only from {@link #callBatched()}, where all feature
     * suppliers are guaranteed non-null.
     */
    private void buildFullRecord(ByteBufferIndexWriter writer, int newOrdinal) throws Exception {
        var originalOrdinal = ordinalMapper.newToOld(newOrdinal);
        writer.writeInt(newOrdinal);

        if (originalOrdinal == OrdinalMapper.OMITTED) {
            for (var feature : inlineFeatures) {
                for (int i = 0; i < feature.featureSize(); i++) writer.writeByte(0);
            }
            writer.writeInt(0);
            for (int n = 0; n < graph.getDegree(0); n++) writer.writeInt(-1);
        } else {
            if (!graph.containsNode(originalOrdinal)) {
                throw new IllegalStateException(String.format(
                        "Ordinal mapper mapped new ordinal %d to non-existing node %d",
                        newOrdinal, originalOrdinal));
            }
            for (var feature : inlineFeatures) {
                feature.writeInline(writer, featureStateSuppliers.get(feature.id()).apply(originalOrdinal));
            }
            var neighbors = view.getNeighborsIterator(0, originalOrdinal);
            if (neighbors.size() > graph.getDegree(0)) {
                throw new IllegalStateException(String.format(
                        "Node %d has more neighbors %d than max degree %d -- run Builder.cleanup()!",
                        originalOrdinal, neighbors.size(), graph.getDegree(0)));
            }
            writer.writeInt(neighbors.size());
            int n = 0;
            for (; n < neighbors.size(); n++) {
                int newNeighbor = ordinalMapper.oldToNew(neighbors.nextInt());
                if (newNeighbor < 0 || newNeighbor > ordinalMapper.maxOrdinal()) {
                    throw new IllegalStateException(String.format(
                            "Neighbor ordinal out of bounds: %d/%d",
                            newNeighbor, ordinalMapper.maxOrdinal()));
                }
                writer.writeInt(newNeighbor);
            }
            for (; n < graph.getDegree(0); n++) writer.writeInt(-1);
        }
    }

    /** A not-yet-fully-written buffer destined for a fixed file offset. */
    private static final class PendingWrite {
        final ByteBuffer buffer;
        final long position;

        PendingWrite(ByteBuffer buffer, long position) {
            this.buffer = buffer;
            this.position = position;
        }
    }

    /**
     * Caps how many writes this task submits to the channel before joining any of them.
     * <p>
     * On platforms without a native async file-I/O backend wired into NIO2 (macOS and most
     * POSIX systems get {@code sun.nio.ch.SimpleAsynchronousFileChannelImpl}; only Windows gets
     * true IOCP), {@code AsynchronousFileChannel.write()} is emulated by handing the write to an
     * executor that is observed to spin up a fresh native thread per outstanding call rather
     * than reusing a small, bounded pool. {@link #callLegacy()}'s per-task write count can be
     * O(nodes) — unlike {@link #callBatched()}'s O(1) — so submitting an entire task's writes
     * before joining any of them can create thousands of simultaneously outstanding writes per
     * task, times however many tasks are running concurrently. In practice this has been
     * observed to exhaust the OS thread limit ({@code OutOfMemoryError: unable to create native
     * thread}, {@code pthread_create failed (EAGAIN)}) on a large write. This bound keeps the
     * number of writes any single task has outstanding at once small and constant, regardless
     * of how many nodes the task covers.
     */
    private static final int MAX_IN_FLIGHT_WRITES = 32;

    /**
     * Submits {@code wave} to the channel in chunks of at most {@link #MAX_IN_FLIGHT_WRITES},
     * joining each chunk before submitting the next. Within a chunk, every write is submitted
     * before any is joined, preserving pipelining at a bounded scale; see
     * {@link #MAX_IN_FLIGHT_WRITES} for why an unbounded submit-everything-first approach is
     * unsafe on some platforms.
     */
    private void writeAllFully(List<PendingWrite> wave) throws Exception {
        for (int chunkStart = 0; chunkStart < wave.size(); chunkStart += MAX_IN_FLIGHT_WRITES) {
            int chunkEnd = Math.min(chunkStart + MAX_IN_FLIGHT_WRITES, wave.size());
            writeChunkFully(wave.subList(chunkStart, chunkEnd));
        }
    }

    /**
     * Submits every write in {@code chunk} without blocking, then joins them all.
     * {@code AsynchronousFileChannel.write()} is only guaranteed to write "up to" the
     * buffer's remaining bytes in a single call (see
     * {@link AsynchronousFileChannel#write(ByteBuffer, long)}); a short write is rare for a
     * regular file but can happen (disk full, quota limits, an
     * oversized single write). Any write that completes short is resubmitted for its
     * remaining bytes — which the channel has already advanced the buffer's position past —
     * as a follow-up chunk.
     * <p>
     * In the common case (no short writes) this runs exactly one submit-all/join-all round,
     * so the pipelining {@link #callLegacy()} and {@link #callBatched()} rely on is preserved;
     * the retry loop only does extra work on the rare short-write path.
     */
    private void writeChunkFully(List<PendingWrite> chunk) throws Exception {
        while (!chunk.isEmpty()) {
            List<Future<Integer>> futures = new ArrayList<>(chunk.size());
            for (var w : chunk) {
                futures.add(channel.write(w.buffer, w.position));
            }

            List<PendingWrite> retry = new ArrayList<>();
            for (int i = 0; i < chunk.size(); i++) {
                int written = futures.get(i).get();
                if (written < 0) {
                    throw new IOException("Channel closed during write");
                }
                var w = chunk.get(i);
                if (w.buffer.hasRemaining()) {
                    if (written == 0) {
                        throw new IOException("Channel made no progress writing at position " + w.position);
                    }
                    retry.add(new PendingWrite(w.buffer, w.position + written));
                }
            }
            chunk = retry;
        }
    }
}
