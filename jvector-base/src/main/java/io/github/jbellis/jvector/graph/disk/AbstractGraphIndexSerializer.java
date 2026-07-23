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

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.SeparatedFeature;

import java.io.IOException;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.IntFunction;

/**
 * Abstract base class for graph index serializers providing common functionality.
 */
abstract class AbstractGraphIndexSerializer implements GraphIndexSerializer {
    private final int version;
    private final Set<FeatureId> supportedFeatures;
    private final boolean supportsMultiLayer;
    private final boolean usesFooter;
    private final FeatureOrdering featureOrdering;

    /** A magic number to indicate the file footer */
    public static final int FOOTER_MAGIC = 0x4a564244;
    /** The size of the offset in the footer. */
    public static final int FOOTER_OFFSET_SIZE = Long.BYTES;
    /** The size of the magic number in the footer. */
    public static final int FOOTER_MAGIC_SIZE = Integer.BYTES;
    /** The total size of the footer. */
    public static final int FOOTER_SIZE = FOOTER_MAGIC_SIZE + FOOTER_OFFSET_SIZE;

    protected AbstractGraphIndexSerializer(int version, 
                                          Set<FeatureId> supportedFeatures,
                                          boolean supportsMultiLayer,
                                          boolean usesFooter,
                                          FeatureOrdering featureOrdering) {
        this.version = version;
        this.supportedFeatures = supportedFeatures;
        this.supportsMultiLayer = supportsMultiLayer;
        this.usesFooter = usesFooter;
        this.featureOrdering = featureOrdering;
    }

    @Override
    public int getVersion() {
        return version;
    }

    @Override
    public boolean supportsFeature(FeatureId feature) {
        return supportedFeatures.contains(feature);
    }

    @Override
    public Set<FeatureId> getSupportedFeatures() { return supportedFeatures; }

    @Override
    public boolean supportsMultiLayer() {
        return supportsMultiLayer;
    }

    @Override
    public boolean usesFooter() {
        return usesFooter;
    }

    @Override
    public FeatureOrdering getFeatureOrdering() {
        return featureOrdering;
    }

    @Override
    public void writeSparseLevels(ImmutableGraphIndex graph, OrdinalMapper ordinalMapper, IndexWriter out, Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException {
        for (int level = 1; level <= graph.getMaxLevel(); level++) {
            int layerSize = graph.size(level);
            int layerDegree = graph.getDegree(level);
            int nodesWritten = 0;
            for (var it = graph.getNodes(level); it.hasNext(); ) {
                int originalOrdinal = it.nextInt();
                // node id
                final int newOrdinal = ordinalMapper.oldToNew(originalOrdinal);
                out.writeInt(newOrdinal);
                // neighbors
                var view = graph.getView();
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
    }

    @Override
    public synchronized void writeHeader(ImmutableGraphIndex graph, OrdinalMapper ordinalMapper, Map<FeatureId,Feature> featureMap, int dimension, long startOffset, IndexWriter out, long headerSize) throws IOException {
        // graph-level properties
        var layerInfo = CommonHeader.LayerInfo.fromGraph(graph, ordinalMapper);
        var commonHeader = new CommonHeader(version,
                dimension,
                graph.getView().entryNode() == null ? ImmutableGraphIndex.ENTRY_NODE_ABSENT : ordinalMapper.oldToNew(graph.getView().entryNode().node),
                layerInfo,
                ordinalMapper.maxOrdinal() + 1);
        var header = new Header(commonHeader, featureMap);
        header.write(out);
        assert out.position() == startOffset + headerSize : String.format("%d != %d", out.position(), startOffset + headerSize);
    }

    @Override
    public long featureOffsetForOrdinal(ImmutableGraphIndex graph, List<Feature> inlineFeatures, long startOffset, int ordinal, long headerSize) {
        int edgeSize = Integer.BYTES * (1 + graph.getDegree(0));
        long inlineBytes = ordinal * (long) (Integer.BYTES + inlineFeatures.stream().mapToInt(Feature::featureSize).sum() + edgeSize);
        return startOffset
                + headerSize
                + inlineBytes // previous nodes
                + Integer.BYTES; // the ordinal of the node whose features we're about to write
    }

    @Override
    public void writeFooter(ImmutableGraphIndex graph, OrdinalMapper ordinalMapper, long headerOffset, int dimension, IndexWriter out, Map<FeatureId,Feature> featureMap, long headerSize) throws IOException {
        var layerInfo = CommonHeader.LayerInfo.fromGraph(graph, ordinalMapper);
        var commonHeader = new CommonHeader(version,
                dimension,
                graph.getView().entryNode() == null ? ImmutableGraphIndex.ENTRY_NODE_ABSENT : ordinalMapper.oldToNew(graph.getView().entryNode().node),
                layerInfo,
                ordinalMapper.maxOrdinal() + 1);
        var header = new Header(commonHeader, featureMap);
        header.write(out); // write the header
        out.writeLong(headerOffset); // We write the offset of the header at the end of the file
        out.writeInt(FOOTER_MAGIC);
        final long expectedPosition = headerOffset + headerSize + FOOTER_SIZE;
        assert out.position() == expectedPosition : String.format("%d != %d", out.position(), expectedPosition);
    }

    public void writeSeparatedFeatures(Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers, Map<FeatureId,Feature> featureMap, IndexWriter out, OrdinalMapper ordinalMapper) throws IOException {
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
                        // write zeros for missing data as padding
                        for (int i = 0; i < feature.featureSize(); i++) {
                            out.writeByte(0);
                        }
                    }
                }
            }
        }
    }

    boolean isSeparated(Feature feature) {
        return feature instanceof SeparatedFeature;
    }

    /**
     * Helper to create a set of all known features.
     * Only correct for the current maximum version — do not use for older versions.
     * When a new version is introduced with a new feature, the previous version's serializer
     * must be updated to use an explicit set that excludes the new feature.
     */
    protected static Set<FeatureId> allFeatures() {
        return EnumSet.allOf(FeatureId.class);
    }

    /**
     * Helper to create a set of all features except FUSED_PQ.
     * Used by versions 3–5, which support all non-fused features but predate fused PQ hierarchy
     * support introduced in version 6.
     */
    protected static Set<FeatureId> nonFusedFeatures() {
        return EnumSet.complementOf(EnumSet.of(FeatureId.FUSED_PQ));
    }

    /**
     * Helper to create a set with only inline vectors (for version 2).
     */
    protected static Set<FeatureId> inlineVectorsOnly() {
        return EnumSet.of(FeatureId.INLINE_VECTORS);
    }
}
