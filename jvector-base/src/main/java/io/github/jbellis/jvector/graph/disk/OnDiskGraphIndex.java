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

import io.github.jbellis.jvector.annotations.VisibleForTesting;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FeatureSource;
import io.github.jbellis.jvector.graph.disk.feature.FusedADC;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.disk.feature.SeparatedFeature;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.Accountable;
import org.agrona.collections.Int2ObjectHashMap;
import java.util.ArrayList;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A class representing a graph index stored on disk. The base graph contains only graph structure.
 * <p> * The base graph

 * This graph may be extended with additional features, which are stored inline in the graph and in headers.
 * At runtime, this class may choose the best way to use these features.
 */
public class OnDiskGraphIndex implements GraphIndex, AutoCloseable, Accountable
{
    private static final Logger logger = LoggerFactory.getLogger(OnDiskGraphIndex.class);
    public static final int CURRENT_VERSION = 4;
    static final int MAGIC = 0xFFFF0D61; // FFFF to distinguish from old graphs, which should never start with a negative size "ODGI"
    static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    final ReaderSupplier readerSupplier;
    final int version;
    final int dimension;
    final NodeAtLevel entryNode;
    final int idUpperBound;
    final int inlineBlockSize; // total size of all inline elements contributed by features
    final EnumMap<FeatureId, ? extends Feature> features;
    final EnumMap<FeatureId, Integer> inlineOffsets;
    private final List<CommonHeader.LayerInfo> layerInfo;
    // offset of L0 adjacency data
    private final long neighborsOffset;
    /** For layers > 0, store adjacency fully in memory. */
    private volatile AtomicReference<List<Int2ObjectHashMap<int[]>>> inMemoryNeighbors;

    OnDiskGraphIndex(ReaderSupplier readerSupplier, Header header, long neighborsOffset)
    {
        this.readerSupplier = readerSupplier;
        this.version = header.common.version;
        this.layerInfo = header.common.layerInfo;
        this.dimension = header.common.dimension;
        this.entryNode = new NodeAtLevel(header.common.layerInfo.size() - 1, header.common.entryNode);
        this.idUpperBound = header.common.idUpperBound;
        this.features = header.features;
        this.neighborsOffset = neighborsOffset;
        var inlineBlockSize = 0;
        inlineOffsets = new EnumMap<>(FeatureId.class);
        for (var entry : features.entrySet()) {
            var feature = entry.getValue();
            if (!(feature instanceof SeparatedFeature)) {
                inlineOffsets.put(entry.getKey(), inlineBlockSize);
                inlineBlockSize += feature.featureSize();
            }
        }
        this.inlineBlockSize = inlineBlockSize;
        inMemoryNeighbors = new AtomicReference<>(null);
    }

    private List<Int2ObjectHashMap<int[]>> loadInMemoryLayers(RandomAccessReader in) throws IOException {
        var imn = new ArrayList<Int2ObjectHashMap<int[]>>(layerInfo.size());
        // For levels > 0, we load adjacency into memory
        imn.add(null); // L0 placeholder so we don't have to mangle indexing
        long L0size = 0;
        L0size = idUpperBound * (inlineBlockSize + Integer.BYTES * (1L + 1L + layerInfo.get(0).degree));
        in.seek(neighborsOffset + L0size);

        for (int lvl = 1; lvl < layerInfo.size(); lvl++) {
            CommonHeader.LayerInfo info = layerInfo.get(lvl);
            Int2ObjectHashMap<int[]> edges = new Int2ObjectHashMap<>();

            for (int i = 0; i < info.size; i++) {
                int nodeId = in.readInt();
                assert nodeId >= 0 && nodeId < layerInfo.get(0).size :
                        String.format("Node ID %d out of bounds for layer %d", nodeId, lvl);
                int neighborCount = in.readInt();
                assert neighborCount >= 0 && neighborCount <= info.degree
                        : String.format("Node %d neighborCount %d > M %d", nodeId, neighborCount, info.degree);
                int[] neighbors = new int[neighborCount];
                in.read(neighbors, 0, neighborCount);

                // skip any padding up to 'degree' neighbors
                int skip = info.degree - neighborCount;
                if (skip > 0) in.seek(in.getPosition() + ((long) skip * Integer.BYTES));

                edges.put(nodeId, neighbors);
            }
            imn.add(edges);
        }
        return imn;
    }

    /**
     * Load an index from the given reader supplier, where the index starts at `offset`.
     */
    public static OnDiskGraphIndex load(ReaderSupplier readerSupplier, long offset) {
        try (var reader = readerSupplier.get()) {
            logger.debug("Loading OnDiskGraphIndex from offset={}", offset);
            var header = Header.load(reader, offset);
            logger.debug("Header loaded: version={}, dimension={}, entryNode={}, layerInfoCount={}",
                    header.common.version, header.common.dimension, header.common.entryNode, header.common.layerInfo.size());
            logger.debug("Position after reading header={}",
                    reader.getPosition());
            return new OnDiskGraphIndex(readerSupplier, header, reader.getPosition());
        } catch (Exception e) {
            throw new RuntimeException("Error initializing OnDiskGraph at offset " + offset, e);
        }
    }

    /**
     * Load an index from the given reader supplier at offset 0.
     */
    public static OnDiskGraphIndex load(ReaderSupplier readerSupplier) {
        return load(readerSupplier, 0);
    }

    public Set<FeatureId> getFeatureSet() {
        return features.keySet();
    }

    public int getDimension() {
        return dimension;
    }

    @Override
    public int size(int level) {
        return layerInfo.get(level).size;
    }

    @Override
    public int getDegree(int level) {
        return layerInfo.get(level).degree;
    }

    @Override
    public int getIdUpperBound() {
        return idUpperBound;
    }

    @Override
    public NodesIterator getNodes(int level) {
        int size = size(level);
        int maxDegree = getDegree(level);

        long layer0NodeSize = (long) Integer.BYTES // ids
                + inlineBlockSize // inline elements
                + (Integer.BYTES * (long) (maxDegree + 1));
        long layerUpperNodeSize = (long) Integer.BYTES // ids
                + (Integer.BYTES * (long) (maxDegree + 1)); // neighbor count + neighbors)
        long thisLayerNodeSide = level == 0? layer0NodeSize : layerUpperNodeSize;

        long layerOffset = neighborsOffset;
        layerOffset += level > 0? layer0NodeSize * size(0) : 0;
        for (int lvl = 1; lvl < level; lvl++) {
            layerOffset += layerUpperNodeSize * size(lvl);
        }

        try (var reader = readerSupplier.get()) {
            int[] valid_nodes = new int[size(level)];
            int upperBound = level == 0 ? getIdUpperBound() : size(level);
            int pos = 0;
            for (int node = 0; node < upperBound; node++) {
                long node_offset = layerOffset + (node * thisLayerNodeSide);
                reader.seek(node_offset);
                if (reader.readInt() != -1) {
                    valid_nodes[pos++] = node;
                }
            }
            return new NodesIterator.ArrayNodesIterator(valid_nodes, size);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public long ramBytesUsed() {
        return Long.BYTES + 6 * Integer.BYTES + RamUsageEstimator.NUM_BYTES_OBJECT_REF
                + (long) 2 * RamUsageEstimator.NUM_BYTES_OBJECT_REF * FeatureId.values().length;
    }

    public void close() throws IOException {
        // caller is responsible for closing ReaderSupplier
    }

    @Override
    public String toString() {
        return String.format("OnDiskGraphIndex(layers=%s, entryPoint=%s, features=%s)", layerInfo, entryNode,
                features.keySet().stream().map(Enum::name).collect(Collectors.joining(",")));
    }

    @Override
    public int getMaxLevel() {
        return entryNode.level;
    }

    @Override
    public int maxDegree() {
        return layerInfo.stream().mapToInt(li -> li.degree).max().orElseThrow();
    }

    // re-declared to specify type
    @Override
    public View getView() {
        try {
            return new View(readerSupplier.get());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public class View implements FeatureSource, ScoringView, RandomAccessVectorValues {
        protected final RandomAccessReader reader;
        private final int[] neighbors;

        public View(RandomAccessReader reader) {
            this.reader = reader;
            this.neighbors = new int[layerInfo.stream().mapToInt(li -> li.degree).max().orElse(0)];
        }

        @Override
        public int dimension() {
            return dimension;
        }

        // getVector isn't called on the hot path, only getVectorInto, so we don't bother using a shared value
        @Override
        public boolean isValueShared() {
            return false;
        }

        @Override
        public RandomAccessVectorValues copy() {
            throw new UnsupportedOperationException(); // need to copy reader
        }

        protected long offsetFor(int node, FeatureId featureId) {
            Feature feature = features.get(featureId);

            // Separated features are just global offset + node offset
            if (feature instanceof SeparatedFeature) {
                SeparatedFeature sf = (SeparatedFeature) feature;
                return sf.getOffset() + (node * (long) feature.featureSize());
            }

            // Inline features are in layer 0 only
            return neighborsOffset +
                    (node * ((long) Integer.BYTES // ids
                            + inlineBlockSize // inline elements
                            + (Integer.BYTES * (long) (layerInfo.get(0).degree + 1)) // neighbor count + neighbors)
                    )) + Integer.BYTES + // id
                    inlineOffsets.get(featureId);
        }

        private long neighborsOffsetFor(int level, int node) {
            assert level == 0; // higher layers are in memory
            int degree = layerInfo.get(level).degree;

            // skip node ID + inline features
            long skipInline = Integer.BYTES + inlineBlockSize;
            long blockBytes = skipInline + (long) Integer.BYTES * (degree + 1);

            long offsetWithinLayer = blockBytes * node;
            return neighborsOffset + offsetWithinLayer + skipInline;
        }

        @Override
        public RandomAccessReader featureReaderForNode(int node, FeatureId featureId) throws IOException {
            long offset = offsetFor(node, featureId);
            reader.seek(offset);
            return reader;
        }

        @Override
        public VectorFloat<?> getVector(int node) {
            var feature = features.get(FeatureId.INLINE_VECTORS);
            if (feature == null) {
                feature = features.get(FeatureId.SEPARATED_VECTORS);
            }
            if (feature == null) {
                throw new UnsupportedOperationException("No full-resolution vectors in this graph");
            }

            try {
                long offset = offsetFor(node, feature.id());
                reader.seek(offset);
                return vectorTypeSupport.readFloatVector(reader, dimension);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        @Override
        public void getVectorInto(int node, VectorFloat<?> vector, int offset) {
            var feature = features.get(FeatureId.INLINE_VECTORS);
            if (feature == null) {
                feature = features.get(FeatureId.SEPARATED_VECTORS);
            }
            if (feature == null) {
                throw new UnsupportedOperationException("No full-resolution vectors in this graph");
            }

            try {
                long diskOffset = offsetFor(node, feature.id());
                reader.seek(diskOffset);
                vectorTypeSupport.readFloatVector(reader, dimension, vector, offset);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        public NodesIterator getNeighborsIterator(int level, int node) {
            try {
                if (level == 0) {
                    // For layer 0, read from disk
                    reader.seek(neighborsOffsetFor(level, node));
                    int neighborCount = reader.readInt();
                    assert neighborCount <= neighbors.length
                            : String.format("Node %d neighborCount %d > M %d", node, neighborCount, neighbors.length);
                    reader.read(neighbors, 0, neighborCount);
                    return new NodesIterator.ArrayNodesIterator(neighbors, neighborCount);
                } else {
                    // For levels > 0, read from memory
                    var imn = inMemoryNeighbors.updateAndGet(current -> {
                        if (current != null) {
                            return current;
                        }
                        try {
                            return loadInMemoryLayers(reader);
                        } catch (IOException e) {
                            throw new UncheckedIOException(e);
                        }
                    });
                    int[] stored = imn.get(level).get(node);
                    assert stored != null : String.format("No neighbors found for node %d at level %d", node, level);
                    return new NodesIterator.ArrayNodesIterator(stored, stored.length);
                }
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        @Override
        public int size() {
            // For vector operations we only care about layer 0
            return OnDiskGraphIndex.this.size(0);
        }

        @Override
        public NodeAtLevel entryNode() {
            return entryNode;
        }

        @Override
        public int getIdUpperBound() {
            return idUpperBound;
        }

        @Override
        public Bits liveNodes() {
            return Bits.ALL;
        }

        @Override
        public void close() throws IOException {
            reader.close();
        }

        @Override
        public ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf) {
            if (features.containsKey(FeatureId.INLINE_VECTORS)) {
                return RandomAccessVectorValues.super.rerankerFor(queryVector, vsf);
            } else if (features.containsKey(FeatureId.NVQ_VECTORS)) {
                return ((NVQ) features.get(FeatureId.NVQ_VECTORS)).rerankerFor(queryVector, vsf, this);
            } else {
                throw new UnsupportedOperationException("No reranker available for this graph");
            }
        }

        @Override
        public ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf) {
            if (features.containsKey(FeatureId.FUSED_ADC)) {
                return ((FusedADC) features.get(FeatureId.FUSED_ADC)).approximateScoreFunctionFor(queryVector, vsf, this, rerankerFor(queryVector, vsf));
            } else {
                throw new UnsupportedOperationException("No approximate score function available for this graph");
            }
        }
    }

    /** Convenience function for writing a vanilla DiskANN-style index with no extra Features. */
    public static void write(GraphIndex graph, RandomAccessVectorValues vectors, Path path) throws IOException {
        write(graph, vectors, OnDiskGraphIndexWriter.sequentialRenumbering(graph), path);
    }

    /** Convenience function for writing a vanilla DiskANN-style index with no extra Features. */
    public static void write(GraphIndex graph,
                             RandomAccessVectorValues vectors,
                             Map<Integer, Integer> oldToNewOrdinals,
                             Path path)
            throws IOException
    {
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, path).withMap(oldToNewOrdinals)
                .with(new InlineVectors(vectors.dimension()))
                .build())
        {
            var suppliers = Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                    nodeId -> new InlineVectors.State(vectors.getVector(nodeId)));
            writer.write(suppliers);
        }
    }

    @VisibleForTesting
    static boolean areHeadersEqual(OnDiskGraphIndex g1, OnDiskGraphIndex g2) {
        return g1.version == g2.version &&
                g1.dimension == g2.dimension &&
                g1.entryNode.equals(g2.entryNode) &&
                g1.layerInfo.equals(g2.layerInfo);
    }
}