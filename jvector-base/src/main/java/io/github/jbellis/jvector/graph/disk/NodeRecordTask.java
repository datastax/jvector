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
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.function.BooleanSupplier;
import java.util.function.IntFunction;

/**
 * A task that writes L0 records for a range of nodes directly to disk using synchronous position-based writes.
 * <p>
 * This task is designed to be executed in a thread pool, with each worker thread
 * owning its own ImmutableGraphIndex.View for thread-safe neighbor iteration.
 * Each task processes a contiguous range of ordinals to reduce task creation overhead.
 * <p>
 * This writes directly to the AsynchronousFileChannel using position-based writes with writeFully
 * to ensure each write completes before its buffer is reused. Durability is handled by
 * the caller's final channel.force(true).
 */
class NodeRecordTask implements Callable<Void> {
    private final int startOrdinal;  // Inclusive
    private final int endOrdinal;    // Exclusive
    private final OrdinalMapper ordinalMapper;
    private final ImmutableGraphIndex graph;
    private final ImmutableGraphIndex.View view;
    private final List<Feature> inlineFeatures;
    private final Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers;
    private final int recordSize;
    private final long baseOffset;   // Base file offset for L0 (offsets calculated per-ordinal)
    private final AsynchronousFileChannel channel;
    private final ByteBuffer buffer; // Thread-local buffer for building record components
    private final BooleanSupplier stopRequested;

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
                   ByteBuffer buffer) {
        this(startOrdinal, endOrdinal, ordinalMapper, graph, view, inlineFeatures, featureStateSuppliers,
                recordSize, baseOffset, channel, buffer, () -> false);
    }

    /// Create a task that stops before its next record once `stopRequested` returns true, for
    /// example after another task failed or the writing thread was interrupted. A stopped task
    /// returns normally; the writer reports the failure that requested the stop.
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
                   ByteBuffer buffer,
                   BooleanSupplier stopRequested) {
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
        this.buffer = buffer;
        this.stopRequested = stopRequested;
    }

    /**
     * Writes a buffer fully to the channel at the specified position.
     * Ensures all bytes are written by looping until the buffer is empty.
     * This is critical for correctness as AsynchronousFileChannel.write() may not write all bytes in one call.
     *
     * @param channel the channel to write to
     * @param buffer the buffer to write (will be fully consumed)
     * @param position the file position to write at
     * @throws IOException if an I/O error occurs
     * @throws ExecutionException if the write operation fails
     * @throws InterruptedException if interrupted while waiting for write completion
     */
    private static void writeFully(AsynchronousFileChannel channel, ByteBuffer buffer, long position)
            throws IOException, ExecutionException, InterruptedException {
        long currentPosition = position;
        while (buffer.hasRemaining()) {
            int written = channel.write(buffer, currentPosition).get();
            if (written <= 0) {
                throw new IOException("Channel made no progress while writing");
            }
            currentPosition += written;
        }
    }

    @Override
    public Void call() throws Exception {
        // Reuse writer and buffer across all ordinals in this range
        var writer = new ByteBufferIndexWriter(buffer);

        int payloadBytes = Integer.BYTES * (2 + graph.getDegree(0))
                + inlineFeatures.stream().mapToInt(Feature::featureSize).sum();
        if (recordSize != payloadBytes) {
            throw new IllegalArgumentException("Record size must match its serialized payload");
        }
        final boolean completeRecord = inlineFeatures.stream().allMatch(f -> featureStateSuppliers.get(f.id()) != null);
        final int batch = buffer.capacity() / recordSize;
        for (int newOrdinal = startOrdinal; newOrdinal < endOrdinal; newOrdinal++) {
            if (stopRequested.getAsBoolean()) return null;
            var originalOrdinal = ordinalMapper.newToOld(newOrdinal);
            long recordOffset = baseOffset + (long) newOrdinal * recordSize;
            long currentPosition = recordOffset;

            // A complete record can share one bounded buffer and one positioned write.
            if (completeRecord) {
                if (buffer.position() == buffer.limit()) writer.reset();
                int recordBegin = buffer.position();
                writer.writeInt(originalOrdinal == OrdinalMapper.OMITTED ? -1 : newOrdinal);
                if (originalOrdinal == OrdinalMapper.OMITTED) {
                    for (var feature : inlineFeatures)
                        for (int i = 0; i < feature.featureSize(); i++) writer.writeByte(0);
                    writer.writeInt(0);
                    for (int n = 0; n < graph.getDegree(0); n++) writer.writeInt(-1);
                } else {
                    if (!graph.containsNode(originalOrdinal)) {
                        throw new IllegalStateException("Missing graph node " + originalOrdinal);
                    }
                    for (var feature : inlineFeatures)
                        feature.writeInline(writer, featureStateSuppliers.get(feature.id()).apply(originalOrdinal));
                    var neighbors = view.getNeighborsIterator(0, originalOrdinal);
                    if (neighbors.size() > graph.getDegree(0)) throw new IllegalStateException("Too many neighbors for " + originalOrdinal);
                    writer.writeInt(neighbors.size());
                    int n = 0;
                    while (neighbors.hasNext()) {
                        int mapped = ordinalMapper.oldToNew(neighbors.nextInt());
                        if (mapped < 0 || mapped > ordinalMapper.maxOrdinal()) throw new IllegalStateException("Invalid mapped neighbor " + mapped);
                        writer.writeInt(mapped); n++;
                    }
                    for (; n < graph.getDegree(0); n++) writer.writeInt(-1);
                }
                if (writer.position() != recordBegin + recordSize) throw new IllegalStateException("Wrong serialized record length");
                if (buffer.position() < recordSize * batch && newOrdinal != endOrdinal - 1) continue;
                long batchOffset = recordOffset - recordBegin;
                buffer.flip();
                // writeFully waits for completion before this worker reuses its buffer.
                writeFully(channel, buffer, batchOffset);
                buffer.clear();
                continue;
            }

            // Reset buffer for this ordinal
            writer.reset();

            // Write node ordinal
            writer.writeInt(originalOrdinal == OrdinalMapper.OMITTED ? -1 : newOrdinal);
            ByteBuffer ordinalData = writer.cloneBuffer();
            writeFully(channel, ordinalData, currentPosition);
            currentPosition += Integer.BYTES;

            // Handle OMITTED nodes (holes in ordinal space)
            if (originalOrdinal == OrdinalMapper.OMITTED) {
                // Write placeholder: zeros for features and empty neighbor list
                writer.reset();
                for (var feature : inlineFeatures) {
                    // Write zeros for missing features
                    for (int i = 0; i < feature.featureSize(); i++) {
                        writer.writeByte(0);
                    }
                }
                ByteBuffer featureData = writer.cloneBuffer();
                // writeFully consumes the buffer; retain its length before writing.
                int featureBytes = featureData.remaining();
                writeFully(channel, featureData, currentPosition);
                currentPosition += featureBytes;

                // Write empty neighbor list
                writer.reset();
                writer.writeInt(0); // neighbor count
                for (int n = 0; n < graph.getDegree(0); n++) {
                    writer.writeInt(-1); // padding
                }
                ByteBuffer neighborData = writer.cloneBuffer();
                writeFully(channel, neighborData, currentPosition);
            } else {
                // Validate node exists
                if (!graph.containsNode(originalOrdinal)) {
                    throw new IllegalStateException(
                        String.format("Ordinal mapper mapped new ordinal %s to non-existing node %s",
                                      newOrdinal, originalOrdinal));
                }

                // Write inline features (skip if supplier is null - feature was pre-written)
                for (var feature : inlineFeatures) {
                    var supplier = featureStateSuppliers.get(feature.id());
                    if (supplier != null) {
                        // Feature not pre-written, write it now
                        writer.reset();
                        feature.writeInline(writer, supplier.apply(originalOrdinal));
                        ByteBuffer featureData = writer.cloneBuffer();
                        writeFully(channel, featureData, currentPosition);
                    }
                    // Skip to next feature position (whether we wrote it or not)
                    currentPosition += feature.featureSize();
                }

                // Write neighbors
                writer.reset();
                var neighbors = view.getNeighborsIterator(0, originalOrdinal);
                if (neighbors.size() > graph.getDegree(0)) {
                    throw new IllegalStateException(
                        String.format("Node %d has more neighbors %d than the graph's max degree %d -- run Builder.cleanup()!",
                                      originalOrdinal, neighbors.size(), graph.getDegree(0)));
                }

                writer.writeInt(neighbors.size());
                int n = 0;
                for (; n < neighbors.size(); n++) {
                    var newNeighborOrdinal = ordinalMapper.oldToNew(neighbors.nextInt());
                    if (newNeighborOrdinal < 0 || newNeighborOrdinal > ordinalMapper.maxOrdinal()) {
                        throw new IllegalStateException(
                            String.format("Neighbor ordinal out of bounds: %d/%d",
                                          newNeighborOrdinal, ordinalMapper.maxOrdinal()));
                    }
                    writer.writeInt(newNeighborOrdinal);
                }

                // Pad to max degree
                for (; n < graph.getDegree(0); n++) {
                    writer.writeInt(-1);
                }

                ByteBuffer neighborData = writer.cloneBuffer();
                writeFully(channel, neighborData, currentPosition);
            }
        }

        return null;
    }
}

