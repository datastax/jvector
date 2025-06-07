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
import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessWriter;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.*;
import org.agrona.collections.Int2IntHashMap;

import java.io.Closeable;
import java.io.DataOutput;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.IntFunction;
import java.util.stream.Collectors;

/**
 * Writes a graph index to disk in a format that can be loaded as an OnDiskGraphIndex.
 * <p>
 * Unlike {@link OnDiskGraphIndexWriter}, this class always writes in a sequential order and separates header metadata into a separate file.
 * <p>
 * Assumptions:
 * <l>
 * <li> The graph already exists and is not modified as it is being written, and therefore can be written sequentially in a single pass.
 * <li> This assumption is valid for two common cases in Log Structure Merge Tree-based systems such as Cassandra and Lucene:
 *   <ol>
 *   <li> The graph is being written as part of compaction
 *   <li> The graph is being written for addition of a small immutable segment.
 *   <ol>
 * </l>
 * <p>
 * Goals:
 * <ol>
 * <li> Immutability: Every byte written to the index file is immutable. This allows for running calculation of checksums without needing to re-read the file.
 * <li> Performance: We can take advantage of sequential writes for performance.
 * </ol>
 * <p>
 * The above goals are driven by the following motivations:
 * <ol>
 * <li> When we work with either cloud object storage where random writes are not supported on a single stream
 * <li> When we embed jVector in frameworks such as Lucene that rely on sequential writes for performance and correctness
 * </ol>
 */
public class OnDiskSequentialGraphIndexWriter implements GraphIndexWriter {
    private final int version;
    private final GraphIndex graph;
    private final GraphIndex.View view;
    private final OrdinalMapper ordinalMapper;
    private final int dimension;
    // we don't use Map features but EnumMap is the best way to make sure we don't
    // accidentally introduce an ordering bug in the future
    private final EnumMap<FeatureId, Feature> featureMap;
    private final IndexWriter dataOut; /* output for graph nodes and inline features */
    private final IndexWriter metadataOut; /* output for graph {@link Header} metadata */
    private final int headerSize;
    private volatile int maxOrdinalWritten = -1;
    private final List<Feature> inlineFeatures;

    private OnDiskSequentialGraphIndexWriter(IndexWriter dataOut,
                                             IndexWriter metadataOut,
                                             int version,
                                             GraphIndex graph,
                                             OrdinalMapper oldToNewOrdinals,
                                             int dimension,
                                             EnumMap<FeatureId, Feature> features)
    {
        if (graph.getMaxLevel() > 0 && version < 4) {
            throw new IllegalArgumentException("Multilayer graphs must be written with version 4 or higher");
        }
        this.version = version;
        this.graph = graph;
        this.view = graph instanceof OnHeapGraphIndex ? ((OnHeapGraphIndex) graph).getFrozenView() : graph.getView();
        this.ordinalMapper = oldToNewOrdinals;
        this.dimension = dimension;
        this.featureMap = features;
        this.inlineFeatures = features.values().stream().filter(f -> !(f instanceof SeparatedFeature)).collect(Collectors.toList());
        this.dataOut = dataOut;
        this.metadataOut = metadataOut;

        // create a mock Header to determine the correct size
        var layerInfo = CommonHeader.LayerInfo.fromGraph(graph, ordinalMapper);
        var ch = new CommonHeader(version, dimension, 0, layerInfo, 0);
        var placeholderHeader = new Header(ch, featureMap);
        this.headerSize = placeholderHeader.size();
    }

    public Set<FeatureId> getFeatureSet() {
        return featureMap.keySet();
    }

    @Override
    public synchronized void close() throws IOException {
        view.close();
        // Note: we don't close the output streams since we don't own them in this writer
    }

    /**
     * @return the maximum ordinal written so far, or -1 if no ordinals have been written yet
     */
    public int getMaxOrdinal() {
        return maxOrdinalWritten;
    }

    private long featureOffsetForOrdinal(long startOffset, int ordinal) {
        int edgeSize = Integer.BYTES * (1 + graph.getDegree(0));
        long inlineBytes = ordinal * (long) (Integer.BYTES + inlineFeatures.stream().mapToInt(Feature::featureSize).sum() + edgeSize);
        return startOffset
                + inlineBytes // previous nodes
                + Integer.BYTES; // the ordinal of the node whose features we're about to write
    }

    private boolean isSeparated(Feature feature) {
        return feature instanceof SeparatedFeature;
    }

    /**
     * Note: There are several limitations you should be aware of when using:
     * <li> This method doesn't persist (e.g. flush) the output streams.  The caller is responsible for doing so.
     * <li> This method does not support writing to "holes" in the ordinal space.  If your ordinal mapper
     *      maps a new ordinal to an old ordinal that does not exist in the graph, an exception will be thrown.
     */
    @Override
    public synchronized void write(Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException
    {
        final long startOffset = dataOut.position();
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

        // for each graph node, write the associated features, followed by its neighbors at L0
        for (int newOrdinal = 0; newOrdinal <= ordinalMapper.maxOrdinal(); newOrdinal++) {
            var originalOrdinal = ordinalMapper.newToOld(newOrdinal);

            // if no node exists with the given ordinal, write a placeholder
            if (originalOrdinal == OrdinalMapper.OMITTED) {
                throw new IllegalStateException("Ordinal mapper mapped new ordinal" + newOrdinal + " to non-existing node. This behavior is not supported on OnDiskSequentialGraphIndexWriter. Use OnDiskGraphIndexWriter instead.");
            }

            if (!graph.containsNode(originalOrdinal)) {
                var msg = String.format("Ordinal mapper mapped new ordinal %s to non-existing node %s", newOrdinal, originalOrdinal);
                throw new IllegalStateException(msg);
            }
            dataOut.writeInt(newOrdinal); // unnecessary, but a reasonable sanity check
            assert dataOut.position() == featureOffsetForOrdinal(startOffset, newOrdinal) : String.format("%d != %d", dataOut.position(), featureOffsetForOrdinal(startOffset, newOrdinal));
            for (var feature : inlineFeatures) {
                var supplier = featureStateSuppliers.get(feature.id());
                if (supplier == null) {
                    throw new IllegalStateException("Supplier for feature " + feature.id() + " not found");
                } else {
                    feature.writeInline(dataOut, supplier.apply(originalOrdinal));
                }
            }

            var neighbors = view.getNeighborsIterator(0, originalOrdinal);
            if (neighbors.size() > graph.getDegree(0)) {
                var msg = String.format("Node %d has more neighbors %d than the graph's max degree %d -- run Builder.cleanup()!",
                                        originalOrdinal, neighbors.size(), graph.getDegree(0));
                throw new IllegalStateException(msg);
            }
            // write neighbors list
            dataOut.writeInt(neighbors.size());
            int n = 0;
            for (; n < neighbors.size(); n++) {
                var newNeighborOrdinal = ordinalMapper.oldToNew(neighbors.nextInt());
                if (newNeighborOrdinal < 0 || newNeighborOrdinal > ordinalMapper.maxOrdinal()) {
                    var msg = String.format("Neighbor ordinal out of bounds: %d/%d", newNeighborOrdinal, ordinalMapper.maxOrdinal());
                    throw new IllegalStateException(msg);
                }
                dataOut.writeInt(newNeighborOrdinal);
            }
            assert !neighbors.hasNext();

            // pad out to maxEdgesPerNode
            for (; n < graph.getDegree(0); n++) {
                dataOut.writeInt(-1);
            }
        }

        // write sparse levels
        for (int level = 1; level <= graph.getMaxLevel(); level++) {
            int layerSize = graph.size(level);
            int layerDegree = graph.getDegree(level);
            int nodesWritten = 0;
            for (var it = graph.getNodes(level); it.hasNext(); ) {
                int originalOrdinal = it.nextInt();
                // node id
                dataOut.writeInt(ordinalMapper.oldToNew(originalOrdinal));
                // neighbors
                var neighbors = view.getNeighborsIterator(level, originalOrdinal);
                dataOut.writeInt(neighbors.size());
                int n = 0;
                for ( ; n < neighbors.size(); n++) {
                    dataOut.writeInt(ordinalMapper.oldToNew(neighbors.nextInt()));
                }
                assert !neighbors.hasNext() : "Mismatch between neighbor's reported size and actual size";
                // pad out to degree
                for (; n < layerDegree; n++) {
                    dataOut.writeInt(-1);
                }
                nodesWritten++;
            }
            if (nodesWritten != layerSize) {
                throw new IllegalStateException("Mismatch between layer size and nodes written");
            }
        }

        // Write separated features
        for (var featureEntry : featureMap.entrySet()) {
            if (isSeparated(featureEntry.getValue())) {
                var fid = featureEntry.getKey();
                var supplier = featureStateSuppliers.get(fid);
                if (supplier == null) {
                    throw new IllegalStateException("Supplier for feature " + fid + " not found");
                }

                // Set the offset for this feature
                var feature = (SeparatedFeature) featureEntry.getValue();
                feature.setOffset(dataOut.position());

                // Write separated data for each node
                for (int newOrdinal = 0; newOrdinal <= ordinalMapper.maxOrdinal(); newOrdinal++) {
                    int originalOrdinal = ordinalMapper.newToOld(newOrdinal);
                    if (originalOrdinal != OrdinalMapper.OMITTED) {
                        feature.writeSeparately(dataOut, supplier.apply(originalOrdinal));
                    } else {
                        throw new IllegalStateException("Ordinal mapper mapped new ordinal" + newOrdinal + " to non-existing node. This behavior is not supported on OnDiskSequentialGraphIndexWriter. Use OnDiskGraphIndexWriter instead.");
                    }
                }
            }
        }

        // Write the header with the offsets
        writeHeader(startOffset);
        // Note: flushing the data output is the responsibility of the caller we are not going to make assumptions about further uses of the data outputs
    }

    /**
     * Writes the index header, including the graph size, so that OnDiskGraphIndex can open it.
     * <p>
     * Public so that you can write the index size (and thus usefully open an OnDiskGraphIndex against the index)
     * to read Features from it before writing the edges.
     */
    public synchronized void writeHeader(long startOffset) throws IOException {
        var layerInfo = CommonHeader.LayerInfo.fromGraph(graph, ordinalMapper);
        var commonHeader = new CommonHeader(version,
                dimension,
                ordinalMapper.oldToNew(view.entryNode().node),
                layerInfo,
                ordinalMapper.maxOrdinal() + 1);
        var header = new Header(commonHeader, featureMap);
        header.write(metadataOut);
        assert metadataOut.position() == startOffset + headerSize : String.format("%d != %d", metadataOut.position(), startOffset + headerSize);
    }

    /**
     * @return a Map of old to new graph ordinals where the new ordinals are sequential starting at 0,
     * while preserving the original relative ordering in `graph`.  That is, for all node ids i and j,
     * if i &lt; j in `graph` then map[i] &lt; map[j] in the returned map.  "Holes" left by
     * deleted nodes are filled in by shifting down the new ordinals.
     */
    public static Map<Integer, Integer> sequentialRenumbering(GraphIndex graph) {
        try (var view = graph.getView()) {
            Int2IntHashMap oldToNewMap = new Int2IntHashMap(-1);
            int nextOrdinal = 0;
            for (int i = 0; i < view.getIdUpperBound(); i++) {
                if (graph.containsNode(i)) {
                    oldToNewMap.put(i, nextOrdinal++);
                }
            }
            return oldToNewMap;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Builder for OnDiskGraphIndexWriter, with optional features.
     */
    public static class Builder {
        private final GraphIndex graphIndex;
        private final EnumMap<FeatureId, Feature> features;
        private final IndexWriter dataOut;
        private final IndexWriter metadataOut;
        private OrdinalMapper ordinalMapper;
        private int version;

        public Builder(GraphIndex graphIndex, IndexWriter dataOut, IndexWriter metadataOut) {
            this.graphIndex = graphIndex;
            this.dataOut = dataOut;
            this.metadataOut = metadataOut;
            this.features = new EnumMap<>(FeatureId.class);
            this.version = OnDiskGraphIndex.CURRENT_VERSION;
        }

        public Builder withVersion(int version) {
            if (version > OnDiskGraphIndex.CURRENT_VERSION) {
                throw new IllegalArgumentException("Unsupported version: " + version);
            }

            this.version = version;
            return this;
        }

        public Builder with(Feature feature) {
            features.put(feature.id(), feature);
            return this;
        }

        public Builder withMapper(OrdinalMapper ordinalMapper) {
            this.ordinalMapper = ordinalMapper;
            return this;
        }

        public OnDiskSequentialGraphIndexWriter build() throws IOException {
            if (version < 3 && (!features.containsKey(FeatureId.INLINE_VECTORS) || features.size() > 1)) {
                throw new IllegalArgumentException("Only INLINE_VECTORS is supported until version 3");
            }

            int dimension;
            if (features.containsKey(FeatureId.INLINE_VECTORS)) {
                dimension = ((InlineVectors) features.get(FeatureId.INLINE_VECTORS)).dimension();
            } else if (features.containsKey(FeatureId.NVQ_VECTORS)) {
                dimension = ((NVQ) features.get(FeatureId.NVQ_VECTORS)).dimension();
            } else if (features.containsKey(FeatureId.SEPARATED_VECTORS)) {
                dimension = ((SeparatedVectors) features.get(FeatureId.SEPARATED_VECTORS)).dimension();
            } else if (features.containsKey(FeatureId.SEPARATED_NVQ)) {
                dimension = ((SeparatedNVQ) features.get(FeatureId.SEPARATED_NVQ)).dimension();
            } else {
                throw new IllegalArgumentException("Inline or separated vector feature must be provided");
            }

            if (ordinalMapper == null) {
                ordinalMapper = new OrdinalMapper.MapMapper(sequentialRenumbering(graphIndex));
            }
            return new OnDiskSequentialGraphIndexWriter(dataOut, metadataOut, version, graphIndex, ordinalMapper, dimension, features);
        }

        public Builder withMap(Map<Integer, Integer> oldToNewOrdinals) {
            return withMapper(new OrdinalMapper.MapMapper(oldToNewOrdinals));
        }

        public Feature getFeature(FeatureId featureId) {
            return features.get(featureId);
        }
    }
}