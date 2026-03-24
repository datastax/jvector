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

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.function.IntFunction;

/**
 * A task that builds L0 records for a range of nodes and submits them for asynchronous writing.
 * <p>
 * This task separates record building (CPU-bound) from I/O operations (I/O-bound), enabling
 * true asynchronous execution. Worker threads build records in memory and immediately submit
 * them for async writing without blocking, allowing the thread to continue building more records
 * while previous writes are in flight.
 * <p>
 * Key differences from NodeRecordTask:
 * - Returns CompletableFutures instead of blocking on writes
 * - Uses AsyncWriteCoordinator for non-blocking write submission
 * - Builds complete records in memory before submitting writes
 * - Enables pipelined execution: build → submit → build → submit...
 * <p>
 * This task is designed to be executed in a thread pool, with each worker thread
 * owning its own ImmutableGraphIndex.View for thread-safe neighbor iteration.
 */
class AsyncNodeRecordTask implements Callable<List<CompletableFuture<Void>>> {
    private final int startOrdinal;  // Inclusive
    private final int endOrdinal;    // Exclusive
    private final OrdinalMapper ordinalMapper;
    private final ImmutableGraphIndex graph;
    private final ImmutableGraphIndex.View view;
    private final List<Feature> inlineFeatures;
    private final Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers;
    private final int recordSize;
    private final long baseOffset;   // Base file offset for L0
    private final AsyncWriteCoordinator writeCoordinator;
    private final ByteBuffer buffer; // Thread-local buffer for building record components

    AsyncNodeRecordTask(int startOrdinal,
                       int endOrdinal,
                       OrdinalMapper ordinalMapper,
                       ImmutableGraphIndex graph,
                       ImmutableGraphIndex.View view,
                       List<Feature> inlineFeatures,
                       Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers,
                       int recordSize,
                       long baseOffset,
                       AsyncWriteCoordinator writeCoordinator,
                       ByteBuffer buffer) {
        this.startOrdinal = startOrdinal;
        this.endOrdinal = endOrdinal;
        this.ordinalMapper = ordinalMapper;
        this.graph = graph;
        this.view = view;
        this.inlineFeatures = inlineFeatures;
        this.featureStateSuppliers = featureStateSuppliers;
        this.recordSize = recordSize;
        this.baseOffset = baseOffset;
        this.writeCoordinator = writeCoordinator;
        this.buffer = buffer;
    }

    @Override
    public List<CompletableFuture<Void>> call() throws Exception {
        List<CompletableFuture<Void>> writeFutures = new ArrayList<>();
        var writer = new ByteBufferIndexWriter(buffer);

        for (int newOrdinal = startOrdinal; newOrdinal < endOrdinal; newOrdinal++) {
            var originalOrdinal = ordinalMapper.newToOld(newOrdinal);
            long recordOffset = baseOffset + (long) newOrdinal * recordSize;

            // Build the complete record in memory
            RecordData record = buildRecord(newOrdinal, originalOrdinal, recordOffset, writer);

            // Submit all components for async writing (non-blocking)
            for (RecordData.RecordComponent component : record.getComponents()) {
                CompletableFuture<Void> writeFuture = 
                    writeCoordinator.submitWrite(component.getBuffer(), component.getFileOffset());
                writeFutures.add(writeFuture);
            }
        }

        return writeFutures;
    }

    /**
     * Builds a complete record in memory without performing any I/O.
     * Returns a RecordData containing all components ready to be written.
     *
     * @param newOrdinal the new ordinal for this node
     * @param originalOrdinal the original ordinal (or OMITTED)
     * @param recordOffset the base file offset for this record
     * @param writer the writer to use for building components
     * @return RecordData containing all components
     */
    private RecordData buildRecord(int newOrdinal, int originalOrdinal, long recordOffset, ByteBufferIndexWriter writer) {
        RecordData record = new RecordData();
        long currentPosition = recordOffset;

        // Build ordinal component
        writer.reset();
        writer.writeInt(newOrdinal);
        record.addComponent(writer.cloneBuffer(), currentPosition);
        currentPosition += Integer.BYTES;

        // Handle OMITTED nodes (holes in ordinal space)
        if (originalOrdinal == OrdinalMapper.OMITTED) {
            // Build placeholder features (zeros)
            writer.reset();
            for (var feature : inlineFeatures) {
                for (int i = 0; i < feature.featureSize(); i++) {
                    writer.writeByte(0);
                }
            }
            record.addComponent(writer.cloneBuffer(), currentPosition);
            currentPosition += writer.cloneBuffer().remaining();

            // Build empty neighbor list
            writer.reset();
            writer.writeInt(0); // neighbor count
            for (int n = 0; n < graph.getDegree(0); n++) {
                writer.writeInt(-1); // padding
            }
            record.addComponent(writer.cloneBuffer(), currentPosition);
        } else {
            // Validate node exists
            if (!graph.containsNode(originalOrdinal)) {
                throw new IllegalStateException(
                    String.format("Ordinal mapper mapped new ordinal %s to non-existing node %s",
                                  newOrdinal, originalOrdinal));
            }

            // Build inline features
            for (var feature : inlineFeatures) {
                var supplier = featureStateSuppliers.get(feature.id());
                if (supplier != null) {
                    // Feature not pre-written, build it now
                    writer.reset();
                    feature.writeInline(writer, supplier.apply(originalOrdinal));
                    record.addComponent(writer.cloneBuffer(), currentPosition);
                }
                // Skip to next feature position (whether we wrote it or not)
                currentPosition += feature.featureSize();
            }

            // Build neighbors
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

            record.addComponent(writer.cloneBuffer(), currentPosition);
        }

        return record;
    }
}

// Made with Bob
