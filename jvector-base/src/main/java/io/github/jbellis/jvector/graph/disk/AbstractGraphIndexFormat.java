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
import io.github.jbellis.jvector.disk.RandomAccessWriter;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.SeparatedFeature;

import java.io.IOException;
import java.util.EnumSet;
import java.util.Map;
import java.util.Set;
import java.util.function.IntFunction;

/**
 * Abstract base class for graph index formats providing common functionality.
 */
abstract class AbstractGraphIndexFormat implements GraphIndexFormat {
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

    /**
     * Initialises the format with the format characteristics for a specific version.
     *
     * @param version           on-disk format version number reported by {@link #getVersion()}
     * @param supportedFeatures the set of {@link FeatureId}s this version can store
     * @param supportsMultiLayer whether this version supports hierarchical (multi-layer) graphs
     * @param usesFooter        whether metadata is placed in a footer rather than a header
     * @param featureOrdering   the ordering strategy used when laying out feature data
     */
    protected AbstractGraphIndexFormat(int version,
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
    public Set<FeatureId> getSupportedFeatures() {
        return supportedFeatures;
    }

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
    public void writeSparseLevels(WriteContext ctx, IndexWriter out, Map<FeatureId, IntFunction<Feature.State>> suppliers) throws IOException {
        for (int level = 1; level <= ctx.graph.getMaxLevel(); level++) {
            int layerSize = ctx.graph.size(level);
            int layerDegree = ctx.graph.getDegree(level);
            int nodesWritten = 0;
            for (var it = ctx.graph.getNodes(level); it.hasNext(); ) {
                int originalOrdinal = it.nextInt();
                final int newOrdinal = ctx.ordinalMapper.oldToNew(originalOrdinal);
                out.writeInt(newOrdinal);
                var view = ctx.graph.getView();
                var neighbors = view.getNeighborsIterator(level, originalOrdinal);
                out.writeInt(neighbors.size());
                int n = 0;
                for ( ; n < neighbors.size(); n++) {
                    out.writeInt(ctx.ordinalMapper.oldToNew(neighbors.nextInt()));
                }
                assert !neighbors.hasNext() : "Mismatch between neighbor's reported size and actual size";
                for (; n < layerDegree; n++) {
                    out.writeInt(-1);
                }
                nodesWritten++;
            }
            if (nodesWritten != layerSize) {
                throw new IllegalStateException("Mismatch between layer size and nodes written");
            }
        }
        writeAfterSparseLevels(ctx, out, suppliers);
    }

    /**
     * Hook called at the end of {@link #writeSparseLevels} for version-specific additions.
     * The default implementation is a no-op; V6 overrides this to write fused feature data.
     */
    protected void writeAfterSparseLevels(WriteContext ctx, IndexWriter out, Map<FeatureId, IntFunction<Feature.State>> suppliers) throws IOException {}

    @Override
    public void writeHeader(WriteContext ctx, IndexWriter out) throws IOException {
        var layerInfo = CommonHeader.LayerInfo.fromGraph(ctx.graph, ctx.ordinalMapper);
        var entryNode = ctx.graph.getView().entryNode() == null
                ? ImmutableGraphIndex.ENTRY_NODE_ABSENT
                : ctx.ordinalMapper.oldToNew(ctx.graph.getView().entryNode().node);
        var commonHeader = new CommonHeader(getVersion(), ctx.dimension, entryNode, layerInfo, ctx.ordinalMapper.maxOrdinal() + 1);
        var header = new Header(commonHeader, ctx.featureMap);
        header.write(out);
        assert out.position() == ctx.startOffset + ctx.headerSize
                : String.format("%d != %d", out.position(), ctx.startOffset + ctx.headerSize);
    }

    @Override
    public long featureOffsetForOrdinal(WriteContext ctx, int ordinal) {
        int edgeSize = Integer.BYTES * (1 + ctx.graph.getDegree(0));
        long inlineBytes = ordinal * (long) (Integer.BYTES + ctx.inlineFeatures.stream().mapToInt(Feature::featureSize).sum() + edgeSize);
        return ctx.startOffset + ctx.headerSize + inlineBytes + Integer.BYTES;
    }

    @Override
    public void writeFooter(WriteContext ctx, long headerOffset, IndexWriter out) throws IOException {
        var layerInfo = CommonHeader.LayerInfo.fromGraph(ctx.graph, ctx.ordinalMapper);
        var entryNode = ctx.graph.getView().entryNode() == null
                ? ImmutableGraphIndex.ENTRY_NODE_ABSENT
                : ctx.ordinalMapper.oldToNew(ctx.graph.getView().entryNode().node);
        var commonHeader = new CommonHeader(getVersion(), ctx.dimension, entryNode, layerInfo, ctx.ordinalMapper.maxOrdinal() + 1);
        var header = new Header(commonHeader, ctx.featureMap);
        header.write(out);
        out.writeLong(headerOffset);
        out.writeInt(FOOTER_MAGIC);
        final long expectedPosition = headerOffset + ctx.headerSize + FOOTER_SIZE;
        assert out.position() == expectedPosition : String.format("%d != %d", out.position(), expectedPosition);
    }

    @Override
    public void writeSeparatedFeatures(WriteContext ctx, IndexWriter out, Map<FeatureId, IntFunction<Feature.State>> suppliers) throws IOException {
        for (var featureEntry : ctx.featureMap.entrySet()) {
            if (featureEntry.getValue() instanceof SeparatedFeature) {
                var fid = featureEntry.getKey();
                var supplier = suppliers.get(fid);
                if (supplier == null) {
                    throw new IllegalStateException("Supplier for feature " + fid + " not found");
                }
                var feature = (SeparatedFeature) featureEntry.getValue();
                feature.setOffset(out.position());
                for (int newOrdinal = 0; newOrdinal <= ctx.ordinalMapper.maxOrdinal(); newOrdinal++) {
                    int originalOrdinal = ctx.ordinalMapper.newToOld(newOrdinal);
                    if (originalOrdinal != OrdinalMapper.OMITTED) {
                        feature.writeSeparately(out, supplier.apply(originalOrdinal));
                    } else {
                        for (int i = 0; i < feature.featureSize(); i++) {
                            out.writeByte(0);
                        }
                    }
                }
            }
        }
    }

    @Override
    public void writeFeaturesInline(WriteContext ctx, int ordinal, Map<FeatureId, Feature.State> stateMap, RandomAccessWriter out) throws IOException {
        for (var featureId : stateMap.keySet()) {
            if (!ctx.featureMap.containsKey(featureId)) {
                throw new IllegalArgumentException(String.format("Feature %s not configured for index", featureId));
            }
        }
        out.seek(featureOffsetForOrdinal(ctx, ordinal));
        for (var feature : ctx.inlineFeatures) {
            var state = stateMap.get(feature.id());
            if (state == null) {
                out.seek(out.position() + feature.featureSize());
            } else {
                feature.writeInline(out, state);
            }
        }
    }

    @Override
    public void writeOnDiskSequential(WriteContext ctx, IndexWriter out, Map<FeatureId, IntFunction<Feature.State>> suppliers) throws IOException {
        if (ctx.graph instanceof OnHeapGraphIndex) {
            var ohgi = (OnHeapGraphIndex) ctx.graph;
            if (ohgi.getDeletedNodes().cardinality() > 0) {
                throw new IllegalArgumentException("Run builder.cleanup() before writing the graph");
            }
        }
        for (var featureId : suppliers.keySet()) {
            if (!ctx.featureMap.containsKey(featureId)) {
                throw new IllegalArgumentException(String.format("Feature %s not configured for index", featureId));
            }
        }
        if (ctx.ordinalMapper.maxOrdinal() < ctx.graph.size(0) - 1) {
            throw new IllegalStateException(String.format("Ordinal mapper from [0..%d] does not cover all nodes in the graph of size %d",
                    ctx.ordinalMapper.maxOrdinal(), ctx.graph.size(0)));
        }

        var view = ctx.graph.getView();

        writeHeader(ctx, out);

        for (int newOrdinal = 0; newOrdinal <= ctx.ordinalMapper.maxOrdinal(); newOrdinal++) {
            var originalOrdinal = ctx.ordinalMapper.newToOld(newOrdinal);

            if (originalOrdinal == OrdinalMapper.OMITTED) {
                throw new IllegalStateException("Ordinal mapper mapped new ordinal " + newOrdinal
                        + " to non-existing node. This behavior is not supported on OnDiskSequentialGraphIndexWriter. Use OnDiskGraphIndexWriter instead.");
            }
            if (!ctx.graph.containsNode(originalOrdinal)) {
                throw new IllegalStateException(String.format("Ordinal mapper mapped new ordinal %s to non-existing node %s", newOrdinal, originalOrdinal));
            }

            out.writeInt(newOrdinal);
            long featureOffset = featureOffsetForOrdinal(ctx, newOrdinal);
            assert out.position() == featureOffset : String.format("%d != %d", out.position(), featureOffset);

            for (var feature : ctx.inlineFeatures) {
                var supplier = suppliers.get(feature.id());
                if (supplier == null) {
                    throw new IllegalStateException("Supplier for feature " + feature.id() + " not found");
                }
                feature.writeInline(out, supplier.apply(originalOrdinal));
            }

            var neighbors = view.getNeighborsIterator(0, originalOrdinal);
            if (neighbors.size() > ctx.graph.getDegree(0)) {
                throw new IllegalStateException(String.format("Node %d has more neighbors %d than the graph's max degree %d -- run Builder.cleanup()!",
                        originalOrdinal, neighbors.size(), ctx.graph.getDegree(0)));
            }
            out.writeInt(neighbors.size());
            int n = 0;
            for (; n < neighbors.size(); n++) {
                var newNeighborOrdinal = ctx.ordinalMapper.oldToNew(neighbors.nextInt());
                if (newNeighborOrdinal < 0 || newNeighborOrdinal > ctx.ordinalMapper.maxOrdinal()) {
                    throw new IllegalStateException(String.format("Neighbor ordinal out of bounds: %d/%d", newNeighborOrdinal, ctx.ordinalMapper.maxOrdinal()));
                }
                out.writeInt(newNeighborOrdinal);
            }
            assert !neighbors.hasNext();
            for (; n < ctx.graph.getDegree(0); n++) {
                out.writeInt(-1);
            }
        }

        writeSparseLevels(ctx, out, suppliers);
        writeSeparatedFeatures(ctx, out, suppliers);
        writeFooter(ctx, out.position(), out);

        view.close();
    }

    @Override
    public void writeRandomAccess(WriteContext ctx, RandomAccessWriter out, Map<FeatureId, IntFunction<Feature.State>> suppliers, GraphIndexFormat.L0RecordWriter l0Writer) throws IOException {
        if (ctx.graph instanceof OnHeapGraphIndex) {
            var ohgi = (OnHeapGraphIndex) ctx.graph;
            if (ohgi.getDeletedNodes().cardinality() > 0) {
                throw new IllegalArgumentException("Run builder.cleanup() before writing the graph");
            }
        }
        for (var featureId : suppliers.keySet()) {
            if (!ctx.featureMap.containsKey(featureId)) {
                throw new IllegalArgumentException(String.format("Feature %s not configured for index", featureId));
            }
        }
        if (ctx.ordinalMapper.maxOrdinal() < ctx.graph.size(0) - 1) {
            throw new IllegalStateException(String.format("Ordinal mapper from [0..%d] does not cover all nodes in the graph of size %d",
                    ctx.ordinalMapper.maxOrdinal(), ctx.graph.size(0)));
        }

        var view = ctx.graph.getView();

        out.seek(ctx.startOffset);
        writeHeader(ctx, out);
        l0Writer.write(view, suppliers);
        writeSparseLevels(ctx, out, suppliers);
        writeSeparatedFeatures(ctx, out, suppliers);

        final var endOfGraphPosition = out.position();
        out.seek(ctx.startOffset);
        writeHeader(ctx, out);
        out.seek(endOfGraphPosition);
        out.flush();
        view.close();
    }

    /**
     * Helper to create a set of all known features.
     * Only correct for the current maximum version — do not use for older versions.
     */
    protected static Set<FeatureId> allFeatures() {
        return EnumSet.allOf(FeatureId.class);
    }

    /**
     * Helper to create a set of all features except FUSED_PQ.
     * Used by versions 3–5, which predate fused PQ hierarchy support (version 6).
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
