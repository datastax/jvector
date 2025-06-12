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
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.disk.feature.SeparatedFeature;
import io.github.jbellis.jvector.graph.disk.feature.SeparatedNVQ;
import io.github.jbellis.jvector.graph.disk.feature.SeparatedVectors;

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
 * 
 * 1. File Layout:
 *    - CommonHeader: Contains version, dimension, entry node, and layer information
 *    - Header with Features: Contains feature-specific headers
 *    - Layer 0 data: Contains node ordinals, inline features, and edges for all nodes
 *    - Higher layer data (levels 1..N): Contains sparse node ordinals and edges
 *    - Separated features: Contains feature data stored separately from nodes
 * 
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
 * 
 * 3. Ordinal Mapping:
 *    - The writer uses an OrdinalMapper to map between original node IDs and 
 *      the sequential IDs used in the on-disk format
 *    - This allows for compaction (removing "holes" from deleted nodes)
 *    - It also enables custom ID mapping schemes for specific use cases
 * 
 * The class supports incremental writing through the writeInline method, which
 * allows writing features for individual nodes without writing the entire graph.
 */
public class OnDiskGraphIndexWriter extends AbstractGraphIndexWriter<RandomAccessWriter> {
    private final long startOffset;

    OnDiskGraphIndexWriter(RandomAccessWriter randomAccessWriter,
                                   int version,
                                   long startOffset,
                                   GraphIndex graph,
                                   OrdinalMapper oldToNewOrdinals,
                                   int dimension,
                                   EnumMap<FeatureId, Feature> features)
    {
        super(randomAccessWriter, version, graph, oldToNewOrdinals, dimension, features);
        this.startOffset = startOffset;
    }

    /**
     * Close the view and the output stream. Unlike the super method, for backwards compatibility reasons,
     * this method assumes ownership of the output stream.
     */
    @Override
    public synchronized void close() throws IOException {
        view.close();
        out.close();
    }

    /**
     * Caller should synchronize on this OnDiskGraphIndexWriter instance if mixing usage of the
     * output with calls to any of the synchronized methods in this class.
     * <p>
     * Provided for callers (like Cassandra) that want to add their own header/footer to the output.
     */
    public RandomAccessWriter getOutput() {
        return out;
    }

    /**
     * Write the inline features of the given ordinal to the output at the correct offset.
     * Nothing else is written (no headers, no edges).  The output IS NOT flushed.
     * <p>
     * Note: the ordinal given is implicitly a "new" ordinal in the sense of the OrdinalMapper,
     * but since no nodes or edges are involved (we just write the given State to the index file),
     * the mapper is not invoked.
     */
    public synchronized void writeInline(int ordinal, Map<FeatureId, Feature.State> stateMap) throws IOException
    {
        for (var featureId : stateMap.keySet()) {
            if (!featureMap.containsKey(featureId)) {
                throw new IllegalArgumentException(String.format("Feature %s not configured for index", featureId));
            }
        }

        out.seek(featureOffsetForOrdinal(ordinal));

        for (var feature : inlineFeatures) {
            var state = stateMap.get(feature.id());
            if (state == null) {
                out.seek(out.position() + feature.featureSize());
            } else {
                feature.writeInline(out, state);
            }
        }

        maxOrdinalWritten = Math.max(maxOrdinalWritten, ordinal);
    }

    private long featureOffsetForOrdinal(int ordinal) {
        return super.featureOffsetForOrdinal(startOffset, ordinal);
    }

    public synchronized void write(Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException
    {
        if (graph instanceof OnHeapGraphIndex) {
            var ohgi = (OnHeapGraphIndex) graph;
            if (ohgi.getDeletedNodes().cardinality() > 0) {
                throw new IllegalArgumentException("Run builder.cleanup() before writing the graph");
            }
        }
        for (var featureId : featureStateSuppliers.keySet()) {
            if (!featureMap.containsKey(featureId)) {
                throw new IllegalArgumentException(String.format("Feature %s not configured for index", featureId));
            }
        }
        if (ordinalMapper.maxOrdinal() < graph.size(0) - 1) {
            var msg = String.format("Ordinal mapper from [0..%d] does not cover all nodes in the graph of size %d",
                    ordinalMapper.maxOrdinal(), graph.size(0));
            throw new IllegalStateException(msg);
        }

        writeHeader(); // sets position to start writing features

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

        // We will use the abstract method because no random access is needed
        writeSparseLevels();

        // We will use the abstract method because no random access is needed
        writeSeparatedFeatures(featureStateSuppliers);

        // Write the header again with updated offsets
        if (version >= 5) {
            writeFooter(out.position());
        }

        final var endOfGraphPosition = out.position();
        writeHeader();
        out.seek(endOfGraphPosition);
        out.flush();
    }

    /**
     * Write the index header and completed edge lists to the given output.
     * Unlike the super method, this method flushes the output and also assumes it's using a RandomAccessWriter that can
     * seek to the startOffset and re-write the header.
     * @throws IOException if there is an error writing the header
     */
    public synchronized void writeHeader() throws IOException {
        // graph-level properties
        out.seek(startOffset);
        super.writeHeader(startOffset);
        out.flush();
    }

    /** CRC32 checksum of bytes written since the starting offset */
    public synchronized long checksum() throws IOException {
        long endOffset = out.position();
        return out.checksum(startOffset, endOffset);
    }

    /**
     * Builder for {@link OnDiskGraphIndexWriter}, with optional features.
     */
    public static class Builder extends AbstractGraphIndexWriter.Builder<OnDiskGraphIndexWriter, RandomAccessWriter> {
        private long startOffset = 0L;

        public Builder(GraphIndex graphIndex, Path outPath) throws FileNotFoundException {
            this(graphIndex, new BufferedRandomAccessWriter(outPath));
        }

        public Builder(GraphIndex graphIndex, RandomAccessWriter out) {
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
        protected OnDiskGraphIndexWriter reallyBuild(int dimension) throws IOException {
            return new OnDiskGraphIndexWriter(out, version, startOffset, graphIndex, ordinalMapper, dimension, features);
        }
    }
}