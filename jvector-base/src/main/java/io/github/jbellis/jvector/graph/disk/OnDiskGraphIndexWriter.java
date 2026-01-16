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

import io.github.jbellis.jvector.disk.BufferedRandomAccessWriter;
import io.github.jbellis.jvector.disk.RandomAccessWriter;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.Map;
import java.util.function.IntFunction;

/**
 * Writes a graph index to disk in a format that can be loaded as an OnDiskGraphIndex.
 * <p>
 * The serialization process follows these steps:
 * <p>
 * 1. File Layout:
 *    - CommonHeader: Contains version, dimension, entry node, and layer information
 *    - Header with Features: Contains feature-specific headers
 *    - Layer 0 data: Contains node ordinals, inline features, and edges for all nodes
 *    - Higher layer data (levels 1..N): Contains sparse node ordinals and edges
 *    - Separated features: Contains feature data stored separately from nodes
 * <p>
 * 2. Serialization Process:
 *    - First, a placeholder header is written to reserve space
 *    - For each node in layer 0:
 *      - Write node ordinal
 *      - Write inline features (vectors, quantized data, etc.)
 *      - Write neighbor count and neighbor ordinals
 *    - For each higher layer (1..N):
 *      - Write only nodes that exist in that layer
 *      - For each node: write ordinal, neighbor count, and neighbor ordinals
 *    - For each separated feature:
 *      - Write feature data for all nodes sequentially
 *    - Finally, rewrite the header with correct offsets
 * <p>
 * 3. Ordinal Mapping:
 *    - The writer uses an OrdinalMapper to map between original node IDs and 
 *      the sequential IDs used in the on-disk format
 *    - This allows for compaction (removing "holes" from deleted nodes)
 *    - It also enables custom ID mapping schemes for specific use cases
 * <p>
 * The class supports incremental writing through the writeInline method, which
 * allows writing features for individual nodes without writing the entire graph.
 */
public class OnDiskGraphIndexWriter extends RandomAccessOnDiskGraphIndexWriter {

    /**
     * Constructs an OnDiskGraphIndexWriter with all parameters including optional file path
     * and parallel write configuration.
     *
     * @param randomAccessWriter the writer to use for output
     * @param version the format version to write
     * @param startOffset the starting offset in the file
     * @param graph the graph to write
     * @param oldToNewOrdinals mapper for ordinal renumbering
     * @param dimension the vector dimension
     * @param features the features to include
     */
    OnDiskGraphIndexWriter(RandomAccessWriter randomAccessWriter,
                                   int version,
                                   long startOffset,
                                   ImmutableGraphIndex graph,
                                   OrdinalMapper oldToNewOrdinals,
                                   int dimension,
                                   EnumMap<FeatureId, Feature> features)
    {
        super(randomAccessWriter, version, startOffset, graph, oldToNewOrdinals, dimension, features);
    }

    /**
     * Writes L0 records sequentially
     */
    @Override
    protected void writeL0Records(ImmutableGraphIndex.View view,
                                          Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException {
        // for each graph node, write the associated features, followed by its neighbors at L0
        for (int newOrdinal = 0; newOrdinal <= ordinalMapper.maxOrdinal(); newOrdinal++) {
            var originalOrdinal = ordinalMapper.newToOld(newOrdinal);

            // if no node exists with the given ordinal, write a placeholder
            if (originalOrdinal == OrdinalMapper.OMITTED) {
                out.writeInt(-1);
                for (var feature : inlineFeatures) {
                    out.seek(out.position() + feature.featureSize());
                }
                out.writeInt(0);
                for (int n = 0; n < graph.getDegree(0); n++) {
                    out.writeInt(-1);
                }
                continue;
            }

            if (!graph.containsNode(originalOrdinal)) {
                var msg = String.format("Ordinal mapper mapped new ordinal %s to non-existing node %s", newOrdinal, originalOrdinal);
                throw new IllegalStateException(msg);
            }
            out.writeInt(newOrdinal); // unnecessary, but a reasonable sanity check
            assert out.position() == featureOffsetForOrdinal(newOrdinal) : String.format("%d != %d", out.position(), featureOffsetForOrdinal(newOrdinal));
            for (var feature : inlineFeatures) {
                var supplier = featureStateSuppliers.get(feature.id());
                if (supplier == null) {
                    out.seek(out.position() + feature.featureSize());
                } else {
                    feature.writeInline(out, supplier.apply(originalOrdinal));
                }
            }

            var neighbors = view.getNeighborsIterator(0, originalOrdinal);
            if (neighbors.size() > graph.getDegree(0)) {
                var msg = String.format("Node %d has more neighbors %d than the graph's max degree %d -- run Builder.cleanup()!",
                                        originalOrdinal, neighbors.size(), graph.getDegree(0));
                throw new IllegalStateException(msg);
            }
            // write neighbors list
            out.writeInt(neighbors.size());
            int n = 0;
            for (; n < neighbors.size(); n++) {
                var newNeighborOrdinal = ordinalMapper.oldToNew(neighbors.nextInt());
                if (newNeighborOrdinal < 0 || newNeighborOrdinal > ordinalMapper.maxOrdinal()) {
                    var msg = String.format("Neighbor ordinal out of bounds: %d/%d", newNeighborOrdinal, ordinalMapper.maxOrdinal());
                    throw new IllegalStateException(msg);
                }
                out.writeInt(newNeighborOrdinal);
            }
            assert !neighbors.hasNext();

            // pad out to maxEdgesPerNode
            for (; n < graph.getDegree(0); n++) {
                out.writeInt(-1);
            }
        }
    }

    /**
     * Builder for {@link OnDiskGraphIndexWriter}, with optional features.
     */
    public static class Builder extends AbstractGraphIndexWriter.Builder<OnDiskGraphIndexWriter, RandomAccessWriter> {
        private long startOffset = 0L;

        public Builder(ImmutableGraphIndex graphIndex, Path outPath) throws FileNotFoundException {
            this(graphIndex, new BufferedRandomAccessWriter(outPath));
        }

        public Builder(ImmutableGraphIndex graphIndex, RandomAccessWriter out) {
            super(graphIndex, out);
        }

        /**
         * Set the starting offset for the graph index in the output file.  This is useful if you want to
         * append the index to an existing file.
         */
        public Builder withStartOffset(long startOffset) {
            this.startOffset = startOffset;
            return this;
        }

        @Override
        protected OnDiskGraphIndexWriter reallyBuild(int dimension) {
            return new OnDiskGraphIndexWriter(out, version, startOffset, graphIndex, ordinalMapper, dimension, features);
        }
    }
}