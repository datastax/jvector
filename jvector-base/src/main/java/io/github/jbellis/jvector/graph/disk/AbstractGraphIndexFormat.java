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
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.RandomAccessWriter;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.SeparatedFeature;

import java.io.IOException;
import java.util.*;
import java.util.function.IntFunction;

/**
 * Abstract base class for graph index formats providing common functionality.
 */
abstract class AbstractGraphIndexFormat implements GraphIndexFormat {
    private final int version;
    private final Set<FeatureId> supportedFeatures;
    private final boolean supportsMultiLayer;
    private final boolean usesFooter;

    // Footers are used by format versions 5+ (usesFooter=true; see GraphIndexFormatV5) in place
    // of relying solely on the header written at the start of the file. writeRandomAccess()
    // writes a placeholder header at ctx.startOffset before the graph structure is fully known
    // (e.g. final layer sizes), then appends a footer after all graph data (L0 records, sparse
    // levels, separated features) once everything is final. A footer consists of, in order:
    //   [a full duplicate Header, accurate/final] [8-byte offset back to that Header's own start]
    //   [4-byte FOOTER_MAGIC]
    // A reader locates it independent of header size or position by seeking to the last
    // FOOTER_MAGIC_SIZE bytes of the file to validate the magic, then reading the preceding
    // FOOTER_OFFSET_SIZE bytes to find where the accurate Header copy begins (see
    // OnDiskGraphIndex#loadFromFooter). This also allows the graph index to be embedded as a
    // slice within a larger file/stream, since locating it only requires knowing the end of
    // that slice, not any absolute offset from the start.

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
     */
    protected AbstractGraphIndexFormat(int version,
                                       Set<FeatureId> supportedFeatures,
                                       boolean supportsMultiLayer,
                                       boolean usesFooter) {
        this.version = version;
        this.supportedFeatures = supportedFeatures;
        this.supportsMultiLayer = supportsMultiLayer;
        this.usesFooter = usesFooter;
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

    /**
     * Default ordering: preserves the natural {@link FeatureId} enum ordinal order.
     * Version 6 overrides this to place fused features last.
     */
    @Override
    public Map<FeatureId, Feature> orderFeatures(EnumMap<FeatureId, Feature> features) {
        return new LinkedHashMap<>(features);
    }

    @Override
    public void writeSparseLevels(WriteContext ctx, IndexWriter out, Map<FeatureId, IntFunction<Feature.State>> suppliers) throws IOException {
        try (var view = ctx.graph.getView()) {
            for (int level = 1; level <= ctx.graph.getMaxLevel(); level++) {
                int layerSize = ctx.graph.size(level);
                int layerDegree = ctx.graph.getDegree(level);
                int nodesWritten = 0;
                for (var it = ctx.graph.getNodes(level); it.hasNext(); ) {
                    int originalOrdinal = it.nextInt();
                    final int newOrdinal = ctx.ordinalMapper.oldToNew(originalOrdinal);
                    out.writeInt(newOrdinal);
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
        }
        writeAfterSparseLevels(ctx, out, suppliers);
    }

    /**
     * Hook called at the end of {@link #writeSparseLevels} for version-specific additions.
     * The default implementation is a no-op; V6 overrides this to write fused feature data.
     */
    protected void writeAfterSparseLevels(WriteContext ctx, IndexWriter out, Map<FeatureId, IntFunction<Feature.State>> suppliers) throws IOException {}

    /**
     * Hook called at the point in {@link #writeOnDiskSequential} and {@link #writeRandomAccess}
     * where a footer would be appended. The default implementation is a no-op; {@link GraphIndexFormatV5}
     * overrides it to call {@link #writeFooter} with the current output position as the header
     * offset. Version-specific behavior is selected by overriding this hook rather than by
     * branching on {@link #usesFooter()} at the call site, mirroring {@link #writeAfterSparseLevels}.
     * {@code usesFooter()} itself remains on the interface as a queryable capability for external
     * callers; this hook only replaces its former use as an internal behavior-selection branch.
     */
    protected void maybeWriteFooter(WriteContext ctx, IndexWriter out) throws IOException {}

    @Override
    public void writeHeader(WriteContext ctx, IndexWriter out) throws IOException {
        var layerInfo = CommonHeader.LayerInfo.fromGraph(ctx.graph, ctx.ordinalMapper);
        final int entryNode;
        try (var view = ctx.graph.getView()) {
            var en = view.entryNode();
            entryNode = en == null ? ImmutableGraphIndex.ENTRY_NODE_ABSENT : ctx.ordinalMapper.oldToNew(en.node);
        }
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
        final int entryNode;
        try (var view = ctx.graph.getView()) {
            var en = view.entryNode();
            entryNode = en == null ? ImmutableGraphIndex.ENTRY_NODE_ABSENT : ctx.ordinalMapper.oldToNew(en.node);
        }
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

        writeHeader(ctx, out);

        try (var view = ctx.graph.getView()) {
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
        }

        writeSparseLevels(ctx, out, suppliers);
        writeSeparatedFeatures(ctx, out, suppliers);
        maybeWriteFooter(ctx, out);
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

        out.seek(ctx.startOffset);
        writeHeader(ctx, out);
        try (var view = ctx.graph.getView()) {
            l0Writer.write(view, suppliers);
        }
        writeSparseLevels(ctx, out, suppliers);
        writeSeparatedFeatures(ctx, out, suppliers);
        maybeWriteFooter(ctx, out);

        final var endOfGraphPosition = out.position();
        out.seek(ctx.startOffset);
        writeHeader(ctx, out);
        out.seek(endOfGraphPosition);
        out.flush();
    }

    @Override
    public void writeCommonHeader(IndexWriter out, List<CommonHeader.LayerInfo> layerInfo, int dimension, int entryNode, int idUpperBound) throws IOException {
        out.writeInt(layerInfo.get(0).size);
        out.writeInt(dimension);
        out.writeInt(entryNode);
        out.writeInt(layerInfo.get(0).degree);
        if (layerInfo.size() > 1) {
            throw new IllegalArgumentException("Layer info is not supported in version " + getVersion());
        }
    }

    @Override
    public CommonHeader readCommonHeader(RandomAccessReader in, int size) throws IOException {
        int dimension = in.readInt();
        int entryNode = in.readInt();
        int maxDegree = in.readInt();

        List<CommonHeader.LayerInfo> layerInfo;
        layerInfo = List.of(new CommonHeader.LayerInfo(size, maxDegree));
        logger.debug("Common header finished reading at position {}", in.getPosition());

        return new CommonHeader(version, dimension, entryNode, layerInfo, size);
    }

    @Override
    public int commonHeaderSize() {
        return 4 * Integer.BYTES;
    }

    @Override
    public void writeHeaderFeatures(IndexWriter out, Map<FeatureId,? extends Feature> features) throws IOException {
        // we restrict pre-version-3 writers to INLINE_VECTORS features, so we don't need additional version-handling here
        for (Feature writer : features.values()) {
            writer.writeHeader(out);
        }
    }

    @Override
    public int headerSize(Map<FeatureId,? extends Feature> features) {
        int size = this.commonHeaderSize();

        size += features.values().stream().mapToInt(Feature::headerSize).sum();

        return size;
    }

    @Override
    public Map<FeatureId, Feature> loadHeaderFeatures(RandomAccessReader reader, CommonHeader common) throws IOException {
        Map<FeatureId, Feature> features = new EnumMap<>(FeatureId.class);
        FeatureId featureId = FeatureId.INLINE_VECTORS;
        features.put(featureId, featureId.load(common, reader));
        return features;
    }

    @Override
    public OnDiskGraphIndex loadOnDiskIndex(RandomAccessReader reader, Header header, ReaderSupplier readerSupplier, boolean useFooter) throws IOException {
        return OnDiskGraphIndex.construct(readerSupplier, header, reader.getPosition(), reader);
    }

    /**
     * Helper to create the frozen set of features supported by version 6.
     * Deliberately enumerated rather than {@code EnumSet.allOf(FeatureId.class)}: version 6's
     * supported-feature set is a historical fact about a shipped format and must not change just
     * because a new {@link FeatureId} is added to the enum for some future version.
     */
    protected static Set<FeatureId> allFeatures() {
        return EnumSet.of(FeatureId.INLINE_VECTORS, FeatureId.FUSED_PQ, FeatureId.NVQ_VECTORS,
                FeatureId.SEPARATED_VECTORS, FeatureId.SEPARATED_NVQ);
    }

    /**
     * Helper to create the frozen set of features supported by versions 3–5, which predate fused
     * PQ hierarchy support (version 6). Deliberately enumerated rather than
     * {@code EnumSet.complementOf(EnumSet.of(FUSED_PQ))}, so that adding a new {@link FeatureId}
     * in the future requires an explicit decision about which already-shipped versions support it,
     * instead of silently being included here.
     */
    protected static Set<FeatureId> nonFusedFeatures() {
        return EnumSet.of(FeatureId.INLINE_VECTORS, FeatureId.NVQ_VECTORS,
                FeatureId.SEPARATED_VECTORS, FeatureId.SEPARATED_NVQ);
    }

    /**
     * Helper to create a set with only inline vectors (for version 2).
     */
    protected static Set<FeatureId> inlineVectorsOnly() {
        return EnumSet.of(FeatureId.INLINE_VECTORS);
    }
}
