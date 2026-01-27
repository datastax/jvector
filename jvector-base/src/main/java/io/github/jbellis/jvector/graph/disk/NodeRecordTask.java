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
import java.nio.channels.AsynchronousFileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.util.function.IntFunction;

/**
 * A task that writes L0 records for a range of nodes directly to disk using position-based writes.
 * <p>
 * This task is designed to be executed in a thread pool, with each worker thread
 * owning its own ImmutableGraphIndex.View for thread-safe neighbor iteration.
 * Each task processes a contiguous range of ordinals to reduce task creation overhead.
 * <p>
 * This writes directly to the AsynchronousFileChannel using position-based writes, allowing it
 * to handle pre-written features the same way as the sequential writer: by checking
 * for null suppliers and skipping those features (not writing to those positions).
 */
class NodeRecordTask implements Callable<List<Future<Integer>>> {
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
    }

    @Override
    public List<Future<Integer>> call() throws Exception {
        List<Future<Integer>> writeFutures = new ArrayList<>();

        // Reuse writer and buffer across all ordinals in this range
        var writer = new ByteBufferIndexWriter(buffer);

        for (int newOrdinal = startOrdinal; newOrdinal < endOrdinal; newOrdinal++) {
            var originalOrdinal = ordinalMapper.newToOld(newOrdinal);
            long recordOffset = baseOffset + (long) newOrdinal * recordSize;
            long currentPosition = recordOffset;

            // Reset buffer for this ordinal
            writer.reset();

            // Write node ordinal
            writer.writeInt(newOrdinal);
            ByteBuffer ordinalData = writer.cloneBuffer();
            writeFutures.add(channel.write(ordinalData, currentPosition));
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
                writeFutures.add(channel.write(featureData, currentPosition));
                currentPosition += featureData.remaining();

                // Write empty neighbor list
                writer.reset();
                writer.writeInt(0); // neighbor count
                for (int n = 0; n < graph.getDegree(0); n++) {
                    writer.writeInt(-1); // padding
                }
                ByteBuffer neighborData = writer.cloneBuffer();
                writeFutures.add(channel.write(neighborData, currentPosition));
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
                        writeFutures.add(channel.write(featureData, currentPosition));
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
                writeFutures.add(channel.write(neighborData, currentPosition));
            }
        }

        return writeFutures;
    }
}

