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
import java.util.function.IntFunction;

/**
 * A task that builds L0 records for a range of nodes in memory.
 * <p>
 * This task is designed to be executed in a thread pool, with each worker thread
 * owning its own ImmutableGraphIndex.View for thread-safe neighbor iteration.
 * Each task processes a contiguous range of ordinals to reduce task creation overhead.
 */
class NodeRecordTask implements Callable<List<NodeRecordTask.Result>> {
    private final int startOrdinal;  // Inclusive
    private final int endOrdinal;    // Exclusive
    private final OrdinalMapper ordinalMapper;
    private final ImmutableGraphIndex graph;
    private final ImmutableGraphIndex.View view;
    private final List<Feature> inlineFeatures;
    private final Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers;
    private final int recordSize;
    private final long baseOffset;   // Base file offset for L0 (offsets calculated per-ordinal)
    private final ByteBuffer buffer;

    /**
     * Result of building a node record.
     */
    static class Result {
        final int newOrdinal;
        final long fileOffset;
        final ByteBuffer data;

        Result(int newOrdinal, long fileOffset, ByteBuffer data) {
            this.newOrdinal = newOrdinal;
            this.fileOffset = fileOffset;
            this.data = data;
        }
    }

    NodeRecordTask(int startOrdinal,
                   int endOrdinal,
                   OrdinalMapper ordinalMapper,
                   ImmutableGraphIndex graph,
                   ImmutableGraphIndex.View view,
                   List<Feature> inlineFeatures,
                   Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers,
                   int recordSize,
                   long baseOffset,
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
        this.buffer = buffer;
    }

    @Override
    public List<Result> call() throws Exception {
        List<Result> results = new ArrayList<>(endOrdinal - startOrdinal);

        // Reuse writer and buffer across all ordinals in this range
        var writer = new ByteBufferIndexWriter(buffer);

        for (int newOrdinal = startOrdinal; newOrdinal < endOrdinal; newOrdinal++) {
            // Calculate file offset for this ordinal
            long fileOffset = baseOffset + (long) newOrdinal * recordSize;

            // Reset buffer for this ordinal
            writer.reset();

            var originalOrdinal = ordinalMapper.newToOld(newOrdinal);

            // Write node ordinal
            writer.writeInt(newOrdinal);

            // Handle OMITTED nodes (holes in ordinal space)
            if (originalOrdinal == OrdinalMapper.OMITTED) {
                // Write placeholder: skip inline features and write empty neighbor list
                for (var feature : inlineFeatures) {
                    // Write zeros for missing features
                    for (int i = 0; i < feature.featureSize(); i++) {
                        writer.writeByte(0);
                    }
                }
                writer.writeInt(0); // neighbor count
                for (int n = 0; n < graph.getDegree(0); n++) {
                    writer.writeInt(-1); // padding
                }
            } else {
                // Validate node exists
                if (!graph.containsNode(originalOrdinal)) {
                    throw new IllegalStateException(
                        String.format("Ordinal mapper mapped new ordinal %s to non-existing node %s",
                                      newOrdinal, originalOrdinal));
                }

                // Write inline features
                for (var feature : inlineFeatures) {
                    var supplier = featureStateSuppliers.get(feature.id());
                    if (supplier == null) {
                        // Write zeros for missing supplier
                        for (int i = 0; i < feature.featureSize(); i++) {
                            writer.writeByte(0);
                        }
                    } else {
                        feature.writeInline(writer, supplier.apply(originalOrdinal));
                    }
                }

                // Write neighbors
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
            }

            // Verify we wrote exactly the expected amount
            if (writer.bytesWritten() != recordSize) {
                throw new IllegalStateException(
                    String.format("Record size mismatch for ordinal %d: expected %d bytes, wrote %d bytes",
                                  newOrdinal, recordSize, writer.bytesWritten()));
            }

            // Writer handles flip, copy, and reset internally
            // The copy ensures thread-local buffer can be safely reused for the next ordinal
            ByteBuffer dataCopy = writer.cloneBuffer();
            results.add(new Result(newOrdinal, fileOffset, dataCopy));
        }

        return results;
    }
}
