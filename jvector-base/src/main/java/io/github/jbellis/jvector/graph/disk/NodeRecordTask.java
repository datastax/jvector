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
 * not be overwritten.  Each node's <em>owned</em> byte spans (ordinal, non-null features,
 * neighbor section) are identified and written as a minimal set of non-blocking channel writes.
 * All writes for the entire task are submitted before any {@code Future.get()} call, so the OS
 * sees the full I/O workload and can schedule it efficiently.
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
 * will always be {@code false}, the legacy path ({@link #callLegacy()}, {@link #collectNodeWrites})
 * and all related helpers can be deleted, and this class becomes the clean single-path fast path only.
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

        // One channel.write() for the entire task range — one syscall, one OS I/O request.
        rangeBuffer.flip();
        channel.write(rangeBuffer, baseOffset + (long) startOrdinal * recordSize).get();
    }

    // -------------------------------------------------------------------------
    // Legacy path: handle pre-written feature regions.
    //
    // Pre-written bytes must not be overwritten.  For each node we identify the
    // contiguous owned spans (ordinal + consecutive non-null features + neighbor
    // section) and fire one non-blocking channel write per span.  ALL writes for
    // the entire task are submitted before any Future.get() call, letting the OS
    // pipeline them.
    //
    // FUTURE IMPROVEMENT: delete this method and collectNodeWrites() entirely once
    // writeFeaturesInline support is removed.  The fast path handles everything.
    // -------------------------------------------------------------------------

    private void callLegacy() throws Exception {
        List<Future<Integer>> pendingWrites = new ArrayList<>();

        for (int newOrdinal = startOrdinal; newOrdinal < endOrdinal; newOrdinal++) {
            collectNodeWrites(newOrdinal, pendingWrites);
        }

        // All writes are in-flight; wait once rather than blocking after each one.
        for (var f : pendingWrites) {
            int written = f.get();
            if (written < 0) {
                throw new IOException("Channel closed during write");
            }
        }
    }

    /**
     * Determines the owned byte spans for one node record and appends non-blocking
     * channel writes for each span to {@code pending}.
     * <p>
     * The record layout is:
     * <pre>
     *   [ordinal: 4] [feature_0: F0] [feature_1: F1] ... [neighbor_count: 4] [neighbors: D×4]
     * </pre>
     * The ordinal and neighbor section are always owned.  Features are owned when their
     * supplier is non-null; null-supplier features are pre-written gaps we skip over.
     * Consecutive owned bytes are merged into a single write to minimise I/O calls.
     * <p>
     * FUTURE IMPROVEMENT: when writeFeaturesInline is removed, null suppliers cannot exist.
     * This method disappears and every node is handled by buildFullRecord() in callBatched().
     */
    private void collectNodeWrites(int newOrdinal, List<Future<Integer>> pending) throws Exception {
        var originalOrdinal = ordinalMapper.newToOld(newOrdinal);
        long recordBase = baseOffset + (long) newOrdinal * recordSize;

        if (originalOrdinal == OrdinalMapper.OMITTED) {
            // OMITTED nodes are holes in the ordinal space.  writeFeaturesInline() is never
            // called for them, so the entire record is ours — write it in one shot.
            ByteBuffer buf = allocRegion(recordSize);
            buf.putInt(newOrdinal);
            for (var feature : inlineFeatures) {
                for (int i = 0; i < feature.featureSize(); i++) buf.put((byte) 0);
            }
            buf.putInt(0); // neighbor count
            for (int n = 0; n < graph.getDegree(0); n++) buf.putInt(-1);
            buf.flip();
            pending.add(channel.write(buf, recordBase));
            return;
        }

        if (!graph.containsNode(originalOrdinal)) {
            throw new IllegalStateException(String.format(
                    "Ordinal mapper mapped new ordinal %d to non-existing node %d",
                    newOrdinal, originalOrdinal));
        }

        // Accumulate contiguous owned bytes into a region buffer.  When a null supplier
        // (pre-written gap) is encountered, flush the current region and start a new one
        // positioned past the gap.
        //
        // FUTURE IMPROVEMENT: this entire accumulation loop simplifies to a straight
        // linear write through all features when null suppliers are impossible.
        ByteBuffer region = allocRegion(recordSize);
        long regionStart = recordBase;

        // Ordinal: always owned.
        region.putInt(newOrdinal);

        for (var feature : inlineFeatures) {
            var supplier = featureStateSuppliers.get(feature.id());
            if (supplier != null) {
                // Owned: extend the current region directly.
                // ByteBufferIndexWriter(region, false) picks up at region.position()
                // without clearing, so it appends to whatever we have already written.
                var fw = new ByteBufferIndexWriter(region, false);
                feature.writeInline(fw, supplier.apply(originalOrdinal));
            } else {
                // Pre-written gap: flush the current region (if any bytes are pending)
                // then advance regionStart past both the flushed bytes and the gap.
                //
                // FUTURE IMPROVEMENT: this branch is only needed because writeFeaturesInline
                // can place data at arbitrary sub-record offsets.  Remove it with that API.
                if (region.position() > 0) {
                    int owned = region.position();
                    region.flip();
                    pending.add(channel.write(region, regionStart));
                    regionStart += owned + feature.featureSize();
                } else {
                    // Nothing accumulated yet in this region — just skip the gap.
                    regionStart += feature.featureSize();
                }
                region = allocRegion(recordSize);
            }
        }

        // Neighbor section: always owned — append to the current region.
        var neighbors = view.getNeighborsIterator(0, originalOrdinal);
        if (neighbors.size() > graph.getDegree(0)) {
            throw new IllegalStateException(String.format(
                    "Node %d has more neighbors %d than max degree %d -- run Builder.cleanup()!",
                    originalOrdinal, neighbors.size(), graph.getDegree(0)));
        }
        var nw = new ByteBufferIndexWriter(region, false);
        nw.writeInt(neighbors.size());
        int n = 0;
        for (; n < neighbors.size(); n++) {
            int newNeighbor = ordinalMapper.oldToNew(neighbors.nextInt());
            if (newNeighbor < 0 || newNeighbor > ordinalMapper.maxOrdinal()) {
                throw new IllegalStateException(String.format(
                        "Neighbor ordinal out of bounds: %d/%d",
                        newNeighbor, ordinalMapper.maxOrdinal()));
            }
            nw.writeInt(newNeighbor);
        }
        for (; n < graph.getDegree(0); n++) nw.writeInt(-1);

        if (region.position() > 0) {
            region.flip();
            pending.add(channel.write(region, regionStart));
        }
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

    /** Allocates a fresh BIG_ENDIAN ByteBuffer for a write region. */
    private ByteBuffer allocRegion(int capacity) {
        var buf = useDirectBuffers
                ? ByteBuffer.allocateDirect(capacity)
                : ByteBuffer.allocate(capacity);
        buf.order(java.nio.ByteOrder.BIG_ENDIAN);
        return buf;
    }
}
