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

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.disk.BufferedRandomAccessWriter;
import io.github.jbellis.jvector.disk.RandomAccessWriter;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.NodeArray;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.diversity.VamanaDiversityProvider;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.*;
import io.github.jbellis.jvector.util.BitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import static java.lang.Math.*;

public final class OnDiskGraphIndexCompactor {

    static final class NodeWrite {
        final int newOrdinal;
        final VectorFloat<?> vec;
        final NodeArray neighbors;

        NodeWrite(int newOrdinal, VectorFloat<?> vec, NodeArray neighbors) {
            this.newOrdinal = newOrdinal;
            this.vec = vec;
            this.neighbors = neighbors;
        }
    };

    private final List<OnDiskGraphIndex> sources;
    private final Map<OnDiskGraphIndex, FixedBitSet> liveNodes;
    private final Map<OnDiskGraphIndex, OrdinalMapper> remappers;
    private final List<Map.Entry<OnDiskGraphIndex, OnDiskGraphIndex.NodeAtLevel>> upperLayerNodeList;
    private final List<Integer> maxDegrees;

    private final float neighborOverflow = 1.2f;
    private final boolean addHierarchy;
    private final int dimension;
    private final Random rng;
    private int maxOrdinal = -1;
    private int numNodes;
    private final ForkJoinPool executor;
    private static final AtomicInteger threadCounter = new AtomicInteger(0);
    private final int beamWidth = 1000;

    public OnDiskGraphIndexCompactor(List<OnDiskGraphIndex> sources, ForkJoinPool executor) {
        if (sources.isEmpty()) {
            throw new IllegalArgumentException("sources must not be empty");
        }
        // check dimensions
        int dim = sources.get(0).getDimension();
        for(OnDiskGraphIndex source : sources) {
            if(source.getDimension() != dim) {
                throw new IllegalArgumentException("sources must have the same dimension");
            }
        }
        this.upperLayerNodeList = new ArrayList<>();
        this.liveNodes = new HashMap<>();
        this.remappers = new HashMap<>();
        this.sources = sources;
        for(OnDiskGraphIndex source : this.sources) {
            var bits = new FixedBitSet(source.size(0));
            bits.set(0, source.size(0));
            liveNodes.put(source, bits);
            numNodes += source.size(0);
        }

        addHierarchy = this.sources.get(0).getMaxLevel() != 1;
        maxDegrees = this.sources.get(0).maxDegrees();
        dimension = this.sources.get(0).getDimension();
        this.rng = new Random(0);
        this.executor = executor;
    }
    public OnDiskGraphIndexCompactor(List<OnDiskGraphIndex> sources) {
        this(sources, ForkJoinPool.commonPool());
    }

    public void setLiveNodes(OnDiskGraphIndex index, FixedBitSet bits) {
        if(bits.length() != index.size(0)) {
            throw new IllegalArgumentException("index " + index + " out of bounds");
        }
        liveNodes.put(index, bits);
    }

    public void setRemapper(OnDiskGraphIndex index, OrdinalMapper mapper) {
        maxOrdinal = max(mapper.maxOrdinal(), maxOrdinal);
        remappers.put(index, mapper);
    }

    private int getRandomGraphLevel() {
        double ml;
        double randDouble;
        if (addHierarchy) {
            ml = maxDegrees.get(0) == 1 ? 1 : 1 / log(1.0 * maxDegrees.get(0));
            do {
                randDouble = this.rng.nextDouble();  // avoid 0 value, as log(0) is undefined
            } while (randDouble == 0.0);
        } else {
            ml = 0;
            randDouble = 0;
        }
        return ((int) (-log(randDouble) * ml));
    }

    public void compact(Path outputPath, VectorSimilarityFunction similarityFunction) throws FileNotFoundException {

        for(OnDiskGraphIndex source : sources) {
            if(source.getDimension() != dimension) {
                throw new IllegalArgumentException("sources must have the same dimension");
            }
            if(!remappers.containsKey(source)) {
                throw new IllegalArgumentException("Each source must set a remapper");
            }
        }

        // first stage: find the nodes for upper layer graph
        for(int s = 0; s < sources.size(); s++) {
            OnDiskGraphIndex source = sources.get(s);
            NodesIterator sourceNodes = source.getNodes(0);
            FixedBitSet sourceAlive = liveNodes.get(sources.get(s));
            while(sourceNodes.hasNext()) {
                int node = sourceNodes.next();
                if(!sourceAlive.get(node)) continue;
                int level = getRandomGraphLevel();
                if(level > 0) {
                    var nodeLevel = new OnDiskGraphIndex.NodeAtLevel(level, node);
                    upperLayerNodeList.add(Map.entry(source, nodeLevel));
                }
            }
        }

        int window = executor.getPoolSize() * 8;
        Deque<Future<NodeWrite>> inFlight = new ArrayDeque<>(window);

        // second stage: construct the upper layer graph without base layer
        OnHeapGraphIndex upperLayerGraph = constructUpperLayerGraph(similarityFunction);

        // third stage: write base layer nodes, then let writer handle upper layers
        try(CompactWriter writer = new CompactWriter.Builder(upperLayerGraph, outputPath, numNodes).withMapper(new OrdinalMapper.IdentityMapper(maxOrdinal)).with(new InlineVectors(dimension)).build()) {
            writer.writeHeader(upperLayerGraph.getView());

            ExplicitThreadLocal<GraphSearcher[]> tlSearchers = ExplicitThreadLocal.withInitial(() -> {
                GraphSearcher[] gs = new GraphSearcher[sources.size()];
                for (int i = 0; i < sources.size(); i++) {
                    gs[i] = new GraphSearcher(sources.get(i));
                }
                return gs;
            });
            CompactVamanaDiversityProvider vdp = new CompactVamanaDiversityProvider(similarityFunction, 1.2f);

            // Write base layer (level 0) nodes
            for (int s = 0; s < sources.size(); s++) {
                NodesIterator sourceNodes = sources.get(s).getNodes(0);
                FixedBitSet sourceAlive = liveNodes.get(sources.get(s));
                OnDiskGraphIndex.View sourceView = sources.get(s).getView();

                while (sourceNodes.hasNext()) {
                    int node = sourceNodes.next();
                    if (!sourceAlive.get(node)) continue;

                    int finalS = s;
                    inFlight.addLast(executor.submit(() -> {
                        VectorFloat<?> vec = sourceView.getVector(node);

                        List<Map.Entry<OnDiskGraphIndex, SearchResult.NodeScore>> candidates =
                                new ArrayList<>();

                        for (int i = 0; i < sources.size(); ++i) {
                            OnDiskGraphIndex idx = sources.get(i);
                            FixedBitSet indexAlive = liveNodes.get(idx);
                            Bits searchBits;

                            if (finalS == i) {
                                FixedBitSet indexAliveCopy = new FixedBitSet(indexAlive.length());
                                indexAliveCopy.or(indexAlive);
                                indexAliveCopy.clear(node);
                                searchBits = indexAliveCopy;
                            } else {
                                searchBits = indexAlive;
                            }

                            SearchScoreProvider ssp =
                                    DefaultSearchScoreProvider.exact(vec, similarityFunction, idx.getView());

                            SearchResult results = tlSearchers.get()[i].search(ssp, maxDegrees.get(0) * 16, beamWidth, 0.0f, 0.0f, searchBits);
                            for (SearchResult.NodeScore re : results.getNodes()) {
                                candidates.add(Map.entry(idx, re));
                            }

                        }

                        candidates.sort((a, b) -> Float.compare(b.getValue().score, a.getValue().score));

                        BitSet selected = new FixedBitSet(candidates.size());
                        vdp.retainDiverse(candidates, maxDegrees.get(0), 0, selected);

                        NodeArray neighbors = new NodeArray(maxDegrees.get(0));
                        Set<Integer> seenTargetOrdinals = new HashSet<>();

                        for (int k = 0; k < candidates.size(); k++) {
                            if (!selected.get(k)) continue;

                            var candidate = candidates.get(k);
                            OnDiskGraphIndex cSource = candidate.getKey();
                            int targetOrdinal =
                                    remappers.get(cSource).oldToNew(candidate.getValue().node);

                            if (seenTargetOrdinals.add(targetOrdinal)) {
                                neighbors.addInOrder(targetOrdinal, candidate.getValue().score);
                            }
                        }

                        int newOrdinal = remappers.get(sources.get(finalS)).oldToNew(node);
                        return new NodeWrite(newOrdinal, vec, neighbors);

                    }));
                    // If window is full, drain one result IN ORDER and write it
                    if (inFlight.size() >= window) {
                        NodeWrite out = inFlight.removeFirst().get();
                        writer.writeInlineNode(
                                out.newOrdinal,
                                Feature.singleState(FeatureId.INLINE_VECTORS, new InlineVectors.State(out.vec)),
                                out.neighbors);
                    }
                }
                // Drain remaining
                while (!inFlight.isEmpty()) {
                    NodeWrite out = inFlight.removeFirst().get();
                    writer.writeInlineNode(
                            out.newOrdinal,
                            Feature.singleState(FeatureId.INLINE_VECTORS, new InlineVectors.State(out.vec)),
                            out.neighbors);
                }
            }
            writer.write(Map.of());
            writer.writeHeader(upperLayerGraph.getView());
            writer.close();
        } catch (IOException | ExecutionException | InterruptedException e) {
            throw new RuntimeException(e);
        }
        finally {
            executor.shutdownNow();
        }
    }
    public OnHeapGraphIndex constructUpperLayerGraphV2(VectorSimilarityFunction similarityFunction) {
        UpperLayerOrdinalMapper upperLayerOrdinalMapper = new UpperLayerOrdinalMapper(upperLayerNodeList);
        var ulravv = new UpperLayerRandomAccessVectorValues(upperLayerOrdinalMapper);
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ulravv, similarityFunction);
        OnHeapGraphIndex upperLayerGraph = new OnHeapGraphIndex(maxDegrees, dimension, neighborOverflow, new VamanaDiversityProvider(bsp, 1.2f), addHierarchy);
        ConcurrentSkipListSet<OnDiskGraphIndex.NodeAtLevel> insertionsInProgress = new ConcurrentSkipListSet<>();
        ExplicitThreadLocal<NodeArray> naturalScratch;
        ExplicitThreadLocal<NodeArray> concurrentScratch;
        naturalScratch = ExplicitThreadLocal.withInitial(() -> new NodeArray(max(beamWidth, maxDegrees.get(0) + 1)));
        concurrentScratch = ExplicitThreadLocal.withInitial(() -> new NodeArray(max(beamWidth, maxDegrees.get(0) + 1)));

        ExplicitThreadLocal<GraphSearcher> searchers = ExplicitThreadLocal.withInitial(() -> {
            var gs = new GraphSearcher(upperLayerGraph);
            gs.usePruning(false);
            return gs;
        });
        var inProgressBefore = insertionsInProgress.clone();

        for(var node: upperLayerNodeList) {
            var nodeLevel = node.getValue();
            int newOrdinal = upperLayerOrdinalMapper.oldToNew(node);
            var newNodeLevel = new OnDiskGraphIndex.NodeAtLevel(nodeLevel.level, newOrdinal);
            upperLayerGraph.addNode(newNodeLevel);
            insertionsInProgress.add(nodeLevel);
            try (var gs = searchers.get()) {
                var view = upperLayerGraph.getView();
                gs.setView(view);
                var naturalScratchPooled = naturalScratch.get();
                var concurrentScratchPooled = concurrentScratch.get();

                VectorFloat<?> vec = node.getKey().getView().getVector(node.getValue().node);
                SearchScoreProvider upperLayerGraphSsp = DefaultSearchScoreProvider.exact(vec, similarityFunction, ulravv);

                var bits = new OnDiskGraphIndexCompactor.ExcludingBits(newNodeLevel.node);

                var entry = upperLayerGraph.entryNode();
                if (entry != null) {
                    gs.initializeInternal(upperLayerGraphSsp, entry, bits);

                    // Move downward from entry.level to 1
                    for (int lvl = entry.level; lvl > 0; lvl--) {
                        if (lvl > newNodeLevel.level) {
                            gs.searchOneLayer(upperLayerGraphSsp, 1, 0.0f, lvl, gs.getView().liveNodes());
                        } else {
                            gs.searchOneLayer(upperLayerGraphSsp, beamWidth, 0.0f, lvl, gs.getView().liveNodes());
                            SearchResult.NodeScore[] neighbors = new SearchResult.NodeScore[gs.approximateResults.size()];
                            AtomicInteger index = new AtomicInteger();
                            gs.approximateResults.foreach((neighbor, score) -> {
                                neighbors[index.getAndIncrement()] = new SearchResult.NodeScore(neighbor, score);
                            });
                            Arrays.sort(neighbors);
                            updateNeighborsOneLayer(upperLayerGraph, lvl, newNodeLevel.node, neighbors);
                        }
                        gs.setEntryPointsFromPreviousLayer();
                    }
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            upperLayerGraph.markComplete(newNodeLevel);

        }

        // Enforce degree limits for all nodes in the upper layer graph
        for (int i = 0; i < upperLayerNodeList.size(); i++) {
            int newOrdinal = upperLayerOrdinalMapper.oldToNew(upperLayerNodeList.get(i));
            upperLayerGraph.enforceDegree(newOrdinal);
        }

        return upperLayerGraph;
    }

    public OnHeapGraphIndex constructUpperLayerGraph(VectorSimilarityFunction similarityFunction) {
        UpperLayerOrdinalMapper upperLayerOrdinalMapper = new UpperLayerOrdinalMapper(upperLayerNodeList);
        var ulravv = new UpperLayerRandomAccessVectorValues(upperLayerOrdinalMapper);
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ulravv, similarityFunction);
        OnHeapGraphIndex upperLayerGraph = new OnHeapGraphIndex(maxDegrees, dimension, neighborOverflow, new VamanaDiversityProvider(bsp, 1.2f), addHierarchy);
        GraphSearcher searchers = new GraphSearcher(upperLayerGraph);
        searchers.usePruning(false);

        for(var node: upperLayerNodeList) {
            var nodeLevel = node.getValue();
            int newOrdinal = upperLayerOrdinalMapper.oldToNew(node);
            var newNodeLevel = new OnDiskGraphIndex.NodeAtLevel(nodeLevel.level, newOrdinal);
            upperLayerGraph.addNode(newNodeLevel);

            VectorFloat<?> vec = node.getKey().getView().getVector(node.getValue().node);
            SearchScoreProvider upperLayerGraphSsp = DefaultSearchScoreProvider.exact(vec, similarityFunction, ulravv);

            var bits = new OnDiskGraphIndexCompactor.ExcludingBits(newNodeLevel.node);

            var entry = upperLayerGraph.entryNode();
            SearchResult result;
            if (entry == null) {
                result = new SearchResult(new SearchResult.NodeScore[] {}, 0, 0, 0, 0, 0);
            } else {
                searchers.initializeInternal(upperLayerGraphSsp, entry, bits);

                // Move downward from entry.level to 1
                for (int lvl = entry.level; lvl > 0; lvl--) {
                    if (lvl > newNodeLevel.level) {
                        searchers.searchOneLayer(upperLayerGraphSsp, 1, 0.0f, lvl, searchers.getView().liveNodes());
                    } else {
                        searchers.searchOneLayer(upperLayerGraphSsp, beamWidth, 0.0f, lvl, searchers.getView().liveNodes());
                        SearchResult.NodeScore[] neighbors = new SearchResult.NodeScore[searchers.approximateResults.size()];
                        final int[] index = {0};
                        searchers.approximateResults.foreach((neighbor, score) -> {
                            neighbors[index[0]++] = new SearchResult.NodeScore(neighbor, score);
                        });
                        Arrays.sort(neighbors);
                        updateNeighborsOneLayer(upperLayerGraph, lvl, newNodeLevel.node, neighbors);
                    }
                    searchers.setEntryPointsFromPreviousLayer();
                }
            }
            upperLayerGraph.markComplete(newNodeLevel);

        }

        // Enforce degree limits for all nodes in the upper layer graph
        for (int i = 0; i < upperLayerNodeList.size(); i++) {
            int newOrdinal = upperLayerOrdinalMapper.oldToNew(upperLayerNodeList.get(i));
            upperLayerGraph.enforceDegree(newOrdinal);
        }

        return upperLayerGraph;
    }
    private static class ExcludingBits implements Bits {
        private final int excluded;

        public ExcludingBits(int excluded) {
            this.excluded = excluded;
        }

        @Override
        public boolean get(int index) {
            return index != excluded;
        }
    }
    private void updateNeighborsOneLayer(OnHeapGraphIndex upperLayerGraph, int level, int nodeId, SearchResult.NodeScore[] neighbors) {
        if (level == 0) throw new IllegalArgumentException("Level should not be zero.");
        var neighborsArray = new NodeArray(neighbors.length);
        Set<Integer> seenNodes = new HashSet<>();
        for (var neighbor : neighbors) {
            // Skip duplicate nodes (keep only the first/highest-scored occurrence)
            if (seenNodes.add(neighbor.node)) {
                neighborsArray.addInOrder(neighbor.node, neighbor.score);
            }
        }
        upperLayerGraph.addEdges(level, nodeId, neighborsArray, neighborOverflow);
    }

    public class UpperLayerOrdinalMapper {
        private final Map<Integer, Map.Entry<OnDiskGraphIndex, OnDiskGraphIndex.NodeAtLevel>> newToOld;

        public UpperLayerOrdinalMapper(List<Map.Entry<OnDiskGraphIndex, OnDiskGraphIndex.NodeAtLevel>> upperLayerNodeList) {
            newToOld = new HashMap<>();
            for (var node: upperLayerNodeList) {
                int remappedOrdinal = remappers.get(node.getKey()).oldToNew(node.getValue().node);
                newToOld.put(remappedOrdinal, node);
            }
        }

        public int oldToNew(Map.Entry<OnDiskGraphIndex, OnDiskGraphIndex.NodeAtLevel> oldOrdinal) {
            return remappers.get(oldOrdinal.getKey()).oldToNew(oldOrdinal.getValue().node);
        }

        public Map.Entry<OnDiskGraphIndex, OnDiskGraphIndex.NodeAtLevel> newToOld(int newOrdinal) {
            return newToOld.get(newOrdinal);
        }
    }

    public class UpperLayerRandomAccessVectorValues implements RandomAccessVectorValues {
        private final UpperLayerOrdinalMapper upperLayerOrdinalMapper;
        private final int size;

        public UpperLayerRandomAccessVectorValues(UpperLayerOrdinalMapper upperLayerOrdinalMapper) {
            this.size = upperLayerNodeList.size();
            this.upperLayerOrdinalMapper = upperLayerOrdinalMapper;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public int dimension() {
            return dimension;
        }

        @Override
        public VectorFloat<?> getVector(int nodeId) {
            var node = upperLayerOrdinalMapper.newToOld(nodeId);
            return node.getKey().getView().getVector(node.getValue().node);
        }

        @Override
        public boolean isValueShared() {
            return false;
        }

        @Override
        public RandomAccessVectorValues copy() {
            return this;
        }
    }

    final class CompactVamanaDiversityProvider {
        /** the diversity threshold; 1.0 is equivalent to HNSW; Vamana uses 1.2 or more */
        public final float alpha;

        /** used to compute diversity */
        public final VectorSimilarityFunction vsf;

        /** Create a new diversity provider */
        public CompactVamanaDiversityProvider(VectorSimilarityFunction vsf, float alpha) {
            this.vsf = vsf;
            this.alpha = alpha;
        }

        /**
         * Update `selected` with the diverse members of `neighbors`.  `neighbors` is not modified
         * It assumes that the i-th neighbor with 0 {@literal <=} i {@literal <} diverseBefore is already diverse.
         * @return the fraction of short edges (neighbors within alpha=1.0)
         */
        public double retainDiverse(List<Map.Entry<OnDiskGraphIndex, SearchResult.NodeScore>> neighbors, int maxDegree, int diverseBefore, BitSet selected) {
            for (int i = 0; i < min(diverseBefore, maxDegree); i++) {
                selected.set(i);
            }

            int nSelected = diverseBefore;
            double shortEdges = Double.NaN;
            // add diverse candidates, gradually increasing alpha to the threshold
            // (so that the nearest candidates are prioritized)
            float currentAlpha = 1.0f;
            while (currentAlpha <= alpha + 1E-6 && nSelected < maxDegree) {
                for (int i = diverseBefore; i < neighbors.size() && nSelected < maxDegree; i++) {
                    if (selected.get(i)) {
                        continue;
                    }

                    OnDiskGraphIndex.View cSourceView = neighbors.get(i).getKey().getView();
                    int cNode = neighbors.get(i).getValue().node;
                    float cScore = neighbors.get(i).getValue().score;
                    if (isDiverse(cNode, cSourceView, cScore, neighbors, selected, currentAlpha)) {
                        selected.set(i);
                        nSelected++;
                    }
                }

                if (currentAlpha == 1.0f) {
                    // this isn't threadsafe, but (for now) we only care about the result after calling cleanup(),
                    // when we don't have to worry about concurrent changes
                    shortEdges = nSelected / (float) maxDegree;
                }

                currentAlpha += 0.2f;
            }
            return shortEdges;
        }

        // is the candidate node with the given score closer to the base node than it is to any of the
        // already-selected neighbors
        private boolean isDiverse(int cNode, OnDiskGraphIndex.View cSourceView, float score, List<Map.Entry<OnDiskGraphIndex, SearchResult.NodeScore>> others, BitSet selected, float alpha) {
            assert others.size() > 0;

            for (int i = selected.nextSetBit(0); i != DocIdSetIterator.NO_MORE_DOCS; i = selected.nextSetBit(i + 1)) {
                OnDiskGraphIndex.View otherView = others.get(i).getKey().getView();
                int otherNode = others.get(i).getValue().node;
                if (cSourceView == otherView && cNode == otherNode) {
                    break;
                }
                if (vsf.compare(cSourceView.getVector(cNode), otherView.getVector(otherNode)) > score * alpha) {
                    return false;
                }
            }
            return true;
        }

    }
}

final class CompactWriter extends OnDiskGraphIndexWriter {
    int numBaseLayerNodes;

    CompactWriter(RandomAccessWriter randomAccessWriter,
                  int numNodes,
                  int version,
                  long startOffset,
                  ImmutableGraphIndex graph,
                  OrdinalMapper oldToNewOrdinals,
                  int dimension,
                  EnumMap<FeatureId, Feature> features)
    {
        super(randomAccessWriter, version, startOffset, graph, oldToNewOrdinals, dimension, features);
        this.numBaseLayerNodes = numNodes;
    }

    @Override
    public void writeHeader(ImmutableGraphIndex.View view) throws IOException {
        out.seek(startOffset);
        // graph-level properties
        var layerInfo = CommonHeader.LayerInfo.fromGraph(graph, ordinalMapper);
        layerInfo.set(0, new CommonHeader.LayerInfo(numBaseLayerNodes, graph.getDegree(0)));

        var commonHeader = new CommonHeader(version,
                dimension,
                ordinalMapper.oldToNew(view.entryNode().node),
                layerInfo,
                ordinalMapper.maxOrdinal() + 1);
        var header = new Header(commonHeader, featureMap);
        headerSize = header.size();
        header.write(out);
        assert out.position() == startOffset + headerSize : String.format("%d != %d", out.position(), startOffset + headerSize);
    }

    @Override
    void writeFooter(ImmutableGraphIndex.View view, long headerOffset) throws IOException {
        var layerInfo = CommonHeader.LayerInfo.fromGraph(graph, ordinalMapper);
        layerInfo.set(0, new CommonHeader.LayerInfo(numBaseLayerNodes, graph.getDegree(0)));
        var commonHeader = new CommonHeader(version,
                dimension,
                ordinalMapper.oldToNew(view.entryNode().node),
                layerInfo,
                ordinalMapper.maxOrdinal() + 1);
        var header = new Header(commonHeader, featureMap);
        header.write(out); // write the header
        out.writeLong(headerOffset); // We write the offset of the header at the end of the file
        out.writeInt(FOOTER_MAGIC);
        final long expectedPosition = headerOffset + headerSize + FOOTER_SIZE;
        assert out.position() == expectedPosition : String.format("%d != %d", out.position(), expectedPosition);
    }

    public static List<CommonHeader.LayerInfo> fromGraph(ImmutableGraphIndex graph, OrdinalMapper mapper) {
        return IntStream.rangeClosed(0, graph.getMaxLevel())
                .mapToObj(i -> new CommonHeader.LayerInfo(graph.size(i), graph.getDegree(i)))
                .collect(Collectors.toList());
    }

    @Override
    public void write(Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException {

        var view = graph.getView();
        writeSparseLevels(view, featureStateSuppliers);
        // We will use the abstract method because no random access is needed
        writeSeparatedFeatures(featureStateSuppliers);

        if (version >= 5) {
            writeFooter(view, out.position());
        }
    }

    public void close() throws IOException {
        var view = graph.getView();
        final var endOfGraphPosition = out.position();
        out.seek(endOfGraphPosition);
        out.flush();
        view.close();
    }

    public void writeInlineNode(int ordinal, Map<FeatureId, Feature.State> stateMap, NodeArray neighbors) throws IOException
    {
        writeInline(ordinal, stateMap);

        // write neighbors list
        out.writeInt(neighbors.size());
        int n = 0;
        for (; n < neighbors.size(); n++) {
            out.writeInt(neighbors.getNode(n));
        }

        // pad out to maxEdgesPerNode
        for (; n < graph.getDegree(0); n++) {
            out.writeInt(-1);
        }
    }
    /**
     * Builder for {@link CompactWriter}, with optional features.
     */
    static class Builder extends AbstractGraphIndexWriter.Builder<CompactWriter, RandomAccessWriter> {
        private long startOffset = 0L;
        private final int numBaseLayerNodes;

        public Builder(ImmutableGraphIndex graphIndex, Path outPath, int numBaseLayerNodes) throws FileNotFoundException {
            this(graphIndex, new BufferedRandomAccessWriter(outPath), numBaseLayerNodes);
        }

        public Builder(ImmutableGraphIndex graphIndex, RandomAccessWriter out, int numBaseLayerNodes) {
            super(graphIndex, out);
            this.numBaseLayerNodes = numBaseLayerNodes;
        }

        /**
         * Set the starting offset for the graph index in the output file.  This is useful if you want to
         * append the index to an existing file.
         */
        public CompactWriter.Builder withStartOffset(long startOffset) {
            this.startOffset = startOffset;
            return this;
        }

        @Override
        protected CompactWriter reallyBuild(int dimension) throws IOException {
            return new CompactWriter(out, numBaseLayerNodes, version, startOffset, graphIndex, ordinalMapper, dimension, features);
        }
    }
}














