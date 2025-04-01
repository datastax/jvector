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
import org.agrona.collections.Int2IntHashMap;

import java.io.Closeable;
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
 * Write a graph index to disk, for later loading as an OnDiskGraphIndex.
 * <p>
 * Implements `getFeatureSource` to allow incremental construction of a larger-than-memory graph
 * (using the writer as the source of INLINE_VECTORS).
 *
 * Layout:
 * [CommonHeader]
 * [Header with Features]
 * [Edges + inline features for level 0]
 * [Edges for levels 1..N]
 * [Separated features]
 */
public class OnDiskGraphIndexWriter implements Closeable {
    private final int version;
    private final GraphIndex graph;
    private final GraphIndex.View view;
    private final OrdinalMapper ordinalMapper;
    private final int dimension;
    // we don't use Map features but EnumMap is the best way to make sure we don't
    // accidentally introduce an ordering bug in the future
    private final EnumMap<FeatureId, Feature> featureMap;
    private final RandomAccessWriter out;
    private final long startOffset;
    private final int headerSize;
    private volatile int maxOrdinalWritten = -1;
    private final List<Feature> inlineFeatures;

    private OnDiskGraphIndexWriter(RandomAccessWriter out,
                                   int version,
                                   long startOffset,
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
        this.out = out;
        this.startOffset = startOffset;

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

    /**
     * @return the maximum ordinal written so far, or -1 if no ordinals have been written yet
     */
    public int getMaxOrdinal() {
        return maxOrdinalWritten;
    }

    private long featureOffsetForOrdinal(int ordinal) {
        int edgeSize = Integer.BYTES * (1 + graph.getDegree(0));
        long inlineBytes = ordinal * (long) (Integer.BYTES + inlineFeatures.stream().mapToInt(Feature::featureSize).sum() + edgeSize);
        return startOffset
                + headerSize
                + inlineBytes // previous nodes
                + Integer.BYTES; // the ordinal of the node whose features we're about to write
    }

    /**
     * Write the index header and completed edge lists to the given output.  Inline features given in
     * `featureStateSuppliers` will also be written.  (Features that do not have a supplier are assumed
     * to have already been written by calls to writeInline).  The output IS flushed.
     * <p>
     * Each supplier takes a node ordinal and returns a FeatureState suitable for Feature.writeInline.
     */
    private boolean isSeparated(Feature feature) {
        return feature instanceof SeparatedFeature;
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
        if (ordinalMapper.maxOrdinal() < graph.size() - 1) {
            var msg = String.format("Ordinal mapper from [0..%d] does not cover all nodes in the graph of size %d",
                    ordinalMapper.maxOrdinal(), graph.size());
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
                for (int n = 0; n < graph.maxDegree(); n++) {
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

        // write sparse levels
        for (int level = 1; level <= graph.getMaxLevel(); level++) {
            int layerSize = graph.size(level);
            int layerDegree = graph.getDegree(level);
            int nodesWritten = 0;
            for (var it = graph.getNodes(level); it.hasNext(); ) {
                int originalOrdinal = it.nextInt();
                // node id
                out.writeInt(ordinalMapper.oldToNew(originalOrdinal));
                // neighbors
                var neighbors = view.getNeighborsIterator(level, originalOrdinal);
                out.writeInt(neighbors.size());
                int n = 0;
                for ( ; n < neighbors.size(); n++) {
                    out.writeInt(ordinalMapper.oldToNew(neighbors.nextInt()));
                }
                assert !neighbors.hasNext() : "Mismatch between neighbor's reported size and actual size";
                // pad out to degree
                for (; n < layerDegree; n++) {
                    out.writeInt(-1);
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
                feature.setOffset(out.position());

                // Write separated data for each node
                for (int newOrdinal = 0; newOrdinal <= ordinalMapper.maxOrdinal(); newOrdinal++) {
                    int originalOrdinal = ordinalMapper.newToOld(newOrdinal);
                    if (originalOrdinal != OrdinalMapper.OMITTED) {
                        feature.writeSeparately(out, supplier.apply(originalOrdinal));
                    } else {
                        out.seek(out.position() + feature.featureSize());
                    }
                }
            }
        }

        // Write the header again with updated offsets
        long currentPosition = out.position();
        writeHeader();
        out.seek(currentPosition);
        out.flush();
    }

    /**
     * Writes the index header, including the graph size, so that OnDiskGraphIndex can open it.
     * The output IS flushed.
     * <p>
     * Public so that you can write the index size (and thus usefully open an OnDiskGraphIndex against the index)
     * to read Features from it before writing the edges.
     */
    public synchronized void writeHeader() throws IOException {
        // graph-level properties
        out.seek(startOffset);
        var layerInfo = CommonHeader.LayerInfo.fromGraph(graph, ordinalMapper);
        var commonHeader = new CommonHeader(version,
                dimension,
                ordinalMapper.oldToNew(view.entryNode().node),
                layerInfo,
                ordinalMapper.maxOrdinal() + 1);
        var header = new Header(commonHeader, featureMap);
        header.write(out);
        out.flush();
        assert out.position() == startOffset + headerSize : String.format("%d != %d", out.position(), startOffset + headerSize);
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

    /** CRC32 checksum of bytes written since the starting offset */
    public synchronized long checksum() throws IOException {
        long endOffset = out.position();
        return out.checksum(startOffset, endOffset);
    }

    /**
     * Builder for OnDiskGraphIndexWriter, with optional features.
     */
    public static class Builder {
        private final GraphIndex graphIndex;
        private final EnumMap<FeatureId, Feature> features;
        private final RandomAccessWriter out;
        private OrdinalMapper ordinalMapper;
        private long startOffset;
        private int version;

        public Builder(GraphIndex graphIndex, Path outPath) throws FileNotFoundException {
            this(graphIndex, new BufferedRandomAccessWriter(outPath));
        }

        public Builder(GraphIndex graphIndex, RandomAccessWriter out) {
            this.graphIndex = graphIndex;
            this.out = out;
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

        /**
         * Set the starting offset for the graph index in the output file.  This is useful if you want to
         * append the index to an existing file.
         */
        public Builder withStartOffset(long startOffset) {
            this.startOffset = startOffset;
            return this;
        }

        public OnDiskGraphIndexWriter build() throws IOException {
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
            return new OnDiskGraphIndexWriter(out, version, startOffset, graphIndex, ordinalMapper, dimension, features);
        }

        public Builder withMap(Map<Integer, Integer> oldToNewOrdinals) {
            return withMapper(new OrdinalMapper.MapMapper(oldToNewOrdinals));
        }

        public Feature getFeature(FeatureId featureId) {
            return features.get(featureId);
        }
    }
}