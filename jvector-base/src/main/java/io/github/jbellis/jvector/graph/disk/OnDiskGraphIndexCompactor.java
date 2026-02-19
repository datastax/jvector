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
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.nio.ByteBuffer;
//import java.nio.channels.AsynchronousFileChannel;
import java.nio.channels.FileChannel;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.disk.BufferedRandomAccessWriter;
import io.github.jbellis.jvector.disk.RandomAccessWriter;
import io.github.jbellis.jvector.disk.ByteBufferIndexWriter;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.FusedFeature;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.diversity.VamanaDiversityProvider;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.*;
import io.github.jbellis.jvector.util.BitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.github.jbellis.jvector.vector.types.ByteSequence;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static java.lang.Math.*;

final class WriteResult {
    final int newOrdinal;
    final long fileOffset;
    final ByteBuffer data;

    WriteResult(int newOrdinal, long fileOffset, ByteBuffer data) {
        this.newOrdinal = newOrdinal;
        this.fileOffset = fileOffset;
        this.data = data;
    }
};

public final class OnDiskGraphIndexCompactor {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final Logger log = LoggerFactory.getLogger(OnDiskGraphIndexCompactor.class);

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
    private int numTotalNodes = 0;
    private final ForkJoinPool executor;
    private static final AtomicInteger threadCounter = new AtomicInteger(0);
    private final ExplicitThreadLocal<GraphSearcher[]> tlSearchers;

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
        tlSearchers = ExplicitThreadLocal.withInitial(() -> {
            GraphSearcher[] gs = new GraphSearcher[sources.size()];
            for (int i = 0; i < sources.size(); i++) {
                gs[i] = new GraphSearcher(sources.get(i));
                gs[i].usePruning(false);
            }
            return gs;
        });

        this.upperLayerNodeList = new ArrayList<>();
        this.liveNodes = new HashMap<>();
        this.remappers = new HashMap<>();
        this.sources = sources;
        for(OnDiskGraphIndex source : this.sources) {
            var bits = new FixedBitSet(source.size(0));
            bits.set(0, source.size(0));
            liveNodes.put(source, bits);
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
    private void checkBeforeCompact() {
        if(sources.size() <= 1) {
            throw new IllegalArgumentException("Must have at least two sources");
        }
        for(OnDiskGraphIndex source : sources) {
            if(source.getDimension() != dimension) {
                throw new IllegalArgumentException("sources must have the same dimension");
            }
            for(int d = 0; d < maxDegrees.size(); d++) {
                if(!Objects.equals(source.maxDegrees().get(d), maxDegrees.get(d))) {
                    throw new IllegalArgumentException("sources must have the same max degrees");
                }
            }
            if(addHierarchy != source.isHierarchical()) {
                throw new IllegalArgumentException("sources must have the same hierarchical setting");
            }
            if(!remappers.containsKey(source)) {
                throw new IllegalArgumentException("Each source must set a remapper");
            }
            if(!liveNodes.containsKey(source)) {
                throw new IllegalArgumentException("Each source must set live nodes");
            }
        }

        // check features
        Set<FeatureId> refKeys = sources.get(0).getFeatures().keySet();
        boolean sameFeatures = sources.stream()
                .skip(1)
                .map(s -> s.getFeatures().keySet())
                .allMatch(refKeys::equals);

        if(!sameFeatures) {
            throw new IllegalArgumentException("Each source must have the same features");
        }
        if(!refKeys.contains(FeatureId.INLINE_VECTORS)) {
            throw new IllegalArgumentException("Each source must have the INLINE_VECTORS feature");
        }
    }

    public void compact(Path outputPath, VectorSimilarityFunction similarityFunction) throws FileNotFoundException {
        checkBeforeCompact();
        numTotalNodes = 0;
        for(OnDiskGraphIndex source : sources) {
            numTotalNodes += liveNodes.get(source).cardinality();
        }

        // first stage: find the nodes for upper layer graph
        for (OnDiskGraphIndex source : sources) {
            NodesIterator sourceNodes = source.getNodes(0);
            FixedBitSet sourceAlive = liveNodes.get(source);
            while (sourceNodes.hasNext()) {
                int node = sourceNodes.next();
                if (!sourceAlive.get(node)) continue;
                int level = getRandomGraphLevel();
                if (level > 0) {
                    var nodeLevel = new OnDiskGraphIndex.NodeAtLevel(level, node);
                    upperLayerNodeList.add(Map.entry(source, nodeLevel));
                }
            }
        }

        log.info("Upper layer candidates selected: {} nodes across {} sources", upperLayerNodeList.size(), sources.size());

        // second stage: construct the upper layer graph without base layer
        UpperLayerOrdinalMapper upperLayerOrdinalMapper = new UpperLayerOrdinalMapper(upperLayerNodeList);
        var ulravv = new UpperLayerRandomAccessVectorValues(upperLayerOrdinalMapper);
        OnHeapGraphIndex upperLayerGraph = constructUpperLayerGraph(upperLayerOrdinalMapper, ulravv, similarityFunction);
        log.info("Upper layer graph constructed");

        FusedPQ fpq = (FusedPQ) this.sources.get(0).getFeatures().get(FeatureId.FUSED_PQ);
        ProductQuantization pq;
        PQVectors ulpqv;
        int[] ulmap;
        RemappedRandomAccessVectorValues ulmrav;
        Map<Integer, Integer> rulmap;
        InlineVectors iv = new InlineVectors(dimension);
        if(fpq != null) {
            // we utilize the first PQ codebook
            // we cannot directly use encodeAll as it encodes ordinals. The upperLayerRandomAccessValues are map-based and can be non-contiguous.
            // use ordinals mapping/reverse mapping to solve this issue
            int i = 0;
            ulmap = new int[upperLayerNodeList.size()];
            rulmap = new HashMap<>();
            for(Integer key: upperLayerOrdinalMapper.newToOld.keySet()) {
                ulmap[i] = key;
                rulmap.put(key, i++);
            }
            ulmrav = new RemappedRandomAccessVectorValues(ulravv, ulmap);
            pq = fpq.getPQ();
            ulpqv = (PQVectors) pq.encodeAll(ulmrav);
        }
        else {
            pq = null;
            ulpqv = null;
            ulmap = null;
            ulmrav = null;
            rulmap = null;
        }

        // third stage: write base layer nodes, then let writer handle upper layers
        log.info("Writing compacted graph: {} total nodes, maxOrdinal={}, dimension={}, degree={}",
                numTotalNodes, maxOrdinal, dimension, maxDegrees.get(0));
        try(CompactWriter writer = new CompactWriter(outputPath, maxOrdinal, numTotalNodes, 0, upperLayerGraph, dimension, iv, fpq)) {
        writer.writeHeader();

        CompactVamanaDiversityProvider vdp = new CompactVamanaDiversityProvider(similarityFunction, 1.2f);
        AtomicInteger batchesCompleted = new AtomicInteger(0);
        int totalBatches = 0; // counted below
        int submitted = 0;
        int searchTopK = Math.max(2, (maxDegrees.get(0) + sources.size() - 1) / sources.size());
        int beamWidth = searchTopK;
        int maxCandidateSize = searchTopK * sources.size() + maxDegrees.get(0);
        final ThreadLocal<LiveExcludingBits> tlBits =
            ThreadLocal.withInitial(LiveExcludingBits::new);
        final ThreadLocal<SelectedVecCache> tlSelectedCache = 
            ThreadLocal.withInitial(() -> new SelectedVecCache(maxDegrees.get(0)));
        final ThreadLocal<FixedBitSet> tlSelected =
            ThreadLocal.withInitial(() -> new FixedBitSet(maxCandidateSize));
        final ThreadLocal<VectorFloat<?>> tlBaseVec = 
            ThreadLocal.withInitial(() -> vectorTypeSupport.createFloatVector(dimension));
        final ThreadLocal<VectorFloat<?>> tlTmpVec = 
            ThreadLocal.withInitial(() -> vectorTypeSupport.createFloatVector(dimension));
        ExecutorCompletionService<List<WriteResult>> ecs = new ExecutorCompletionService<>(executor);

        // Write base layer (level 0) nodes to buffer
        for (int s = 0; s < sources.size(); s++) {
            NodesIterator sourceNodes = sources.get(s).getNodes(0);
            int numNodes = sourceNodes.size();
            int[] nodes = new int[numNodes];
            // materialize for parallelism
            for(int i = 0; i < numNodes; ++i) nodes[i] = sourceNodes.next();
            FixedBitSet sourceAlive = liveNodes.get(sources.get(s));

            int numBatches = max(40, (numNodes + 128 - 1) / 128);
            if(numBatches > numNodes) {
                numBatches = numNodes;
            }
            totalBatches += numBatches;
            int batchSize = (numNodes + numBatches - 1) / numBatches;

            for(int b = 0; b < numBatches; ++b) {
                final int start = min(numNodes, batchSize * b);
                final int end = min(numNodes, batchSize * (b + 1));
                final int finalS = s;

                ecs.submit(() -> {
                    OnDiskGraphIndex.View sourceView = (OnDiskGraphIndex.View) tlSearchers.get()[finalS].getView();
                    List<WriteResult> wrs = new ArrayList<>(end - start);
                    for(int i = start; i < end; ++i) {
                        int node = nodes[i];
                        if (!sourceAlive.get(node)) continue;
                        VectorFloat<?> vec = tlBaseVec.get();
                        sourceView.getVectorInto(node, vec, 0);
                        VectorFloat<?> tmp = tlTmpVec.get();
                        List<Map.Entry<Integer, SearchResult.NodeScore>> candidates =
                                new ArrayList<>(maxCandidateSize);

                        for (int ss = 0; ss < sources.size(); ++ss) {
                            OnDiskGraphIndex idx = sources.get(ss);
                            FixedBitSet indexAlive = liveNodes.get(idx);
                            var cv = (OnDiskGraphIndex.View) tlSearchers.get()[ss].getView();

                            if (finalS == ss) {
                                // use existing neighbors as candidates
                                var it = cv.getNeighborsIterator(0, node);
                                while(it.hasNext()) {
                                    int nb = it.nextInt();
                                    if(!indexAlive.get(nb)) continue;
                                    cv.getVectorInto(nb, tmp, 0);
                                    float score = similarityFunction.compare(vec, tmp);
                                    candidates.add(Map.entry(ss, new SearchResult.NodeScore(nb, score)));
                                }

                            } else {
                                SearchScoreProvider ssp;
                                if(fpq != null) {
                                   ssp = new DefaultSearchScoreProvider(cv.approximateScoreFunctionFor(vec, similarityFunction));
                                 }
                                else {
                                   ssp = DefaultSearchScoreProvider.exact(vec, similarityFunction, cv);
                                }

                                // PRIORITY
                                // TODO: clarify heuristics approach, identify key questions and verification methods
                                // TODO: parameterize topK and beamWidth in JMH coverage
                                // TODO: validate assumption that recall is stable enough between original contributing graphs and the compacted graph
                                // FUTURE
                                // TODO: ensure that result metrics contain recall first as a correctness measure, and then add performance data
                                //       to include completion time, merge speeds, etc
                                //       to include resource usage variations for compaction and search
                                //SearchResult results = tlSearchers.get()[ss].search(ssp, maxDegrees.get(0), maxDegrees.get(0), 0.0f, 0.0f, indexAlive);
                                SearchResult results = tlSearchers.get()[ss].search(ssp, searchTopK, beamWidth, 0.0f, 0.0f, indexAlive);
                                for (SearchResult.NodeScore re : results.getNodes()) {
                                    candidates.add(Map.entry(ss, re));
                                }
                            }

                        }

                        candidates.sort((a1, a2) -> Float.compare(a2.getValue().score, a1.getValue().score));

                        FixedBitSet selected = tlSelected.get();
                        selected.clear(0, candidates.size());
                        SelectedVecCache selectedCache = tlSelectedCache.get();
                        selectedCache.reset();
                        vdp.retainDiverse(candidates, maxDegrees.get(0), 0, selected, selectedCache);

                        NodeArray neighbors = new NodeArray(maxDegrees.get(0));
                        List<ByteSequence<?>> neighborPQs = null;
                        if(pq != null) {
                          neighborPQs = new ArrayList<>(maxDegrees.get(0));
                        }

                        for (int k = 0; k < selectedCache.size; k++) {
                            int targetOrdinal =
                                    remappers.get(sources.get(selectedCache.idxs[k])).oldToNew(selectedCache.nodes[k]);

                            neighbors.addInOrder(targetOrdinal, selectedCache.scores[k]);
                            if(pq != null) {
                                neighborPQs.add(pq.encode(selectedCache.vecs[k]));
                            }
                        }

                        int newOrdinal = remappers.get(sources.get(finalS)).oldToNew(node);
                        wrs.add(writer.writeInlineNodeRecord(newOrdinal, vec, neighbors, neighborPQs));
                    }
                    int done = batchesCompleted.incrementAndGet();
                    if (done % 10 == 0) {
                        log.info("Compaction progress: {} batches computed so far ({} nodes in this batch)",
                                done, wrs.size());
                    }
                    return wrs;
                });
                submitted++;
            }
        }
        log.info("Submitted {} compute batches across {} sources to thread pool (parallelism={})",
                totalBatches, sources.size(), executor.getParallelism());

        var opts = EnumSet.of(StandardOpenOption.WRITE, StandardOpenOption.READ);
        try(FileChannel fc = FileChannel.open(outputPath, opts)) {
            for(int sb = 0; sb < submitted; ++sb) {
                List<WriteResult> results = ecs.take().get();
                for(WriteResult r: results) {
                    ByteBuffer b = r.data;
                    long pos = r.fileOffset;
                    while (b.hasRemaining()) {
                        int n = fc.write(b, pos);
                        pos += n;
                    }
                }
                if (sb != 0 && sb % 10 == 0) {
                    log.info("Compaction I/O progress: {}/{} batches written to disk", sb, submitted);
                }
            }
        }
        log.info("All {} batches written to disk, writing upper layers and footer", totalBatches);
        writer.offsetAfterInline();
        writer.writeUpperLayers(ulpqv, rulmap);
        writer.writeFooter();
        writer.close();
        log.info("Compaction complete: {}", outputPath);
        } catch (IOException | ExecutionException | InterruptedException e) {
            throw new RuntimeException(e);
        }
        finally {
            executor.shutdownNow();
        }
    }

    public OnHeapGraphIndex constructUpperLayerGraph(UpperLayerOrdinalMapper upperLayerOrdinalMapper, UpperLayerRandomAccessVectorValues ulravv, VectorSimilarityFunction similarityFunction) {

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
                        searchers.searchOneLayer(upperLayerGraphSsp, 100, 0.0f, lvl, searchers.getView().liveNodes());
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

    private static final class SelectedVecCache {
        int[] idxs;
        OnDiskGraphIndex.View[] views;
        int[] nodes;
        float[] scores;
        VectorFloat<?>[] vecs;
        int size;

        SelectedVecCache(int capacity) {
            idxs = new int[capacity];
            views = new OnDiskGraphIndex.View[capacity];
            nodes = new int[capacity];
            scores = new float[capacity];
            vecs = new VectorFloat<?>[capacity];
            size = 0;
        }

        void reset() { size = 0; }

        void add(int idx, OnDiskGraphIndex.View view, int node, float score, VectorFloat<?> vec) {
            idxs[size] = idx;
            views[size] = view;
            nodes[size] = node;
            scores[size] = score;
            vecs[size] = vec;
            size++;
        }
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
    private static final class LiveExcludingBits implements Bits {
        private FixedBitSet live;
        private int excluded;

        public void reset(FixedBitSet live, int excluded) {
            this.live = live;
            this.excluded = excluded;
        }

        @Override
        public boolean get(int index) {
            // Respect live nodes AND exclude the one node
            return index != excluded && live.get(index);
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
        public final Map<Integer, Map.Entry<OnDiskGraphIndex, OnDiskGraphIndex.NodeAtLevel>> newToOld;

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
        public double retainDiverse(List<Map.Entry<Integer, SearchResult.NodeScore>> neighbors, int maxDegree, int diverseBefore, BitSet selected, SelectedVecCache selectedCache) {
            assert neighbors.size() > 0;

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

                    int cIdx = neighbors.get(i).getKey();
                    OnDiskGraphIndex.View cView = (OnDiskGraphIndex.View) tlSearchers.get()[cIdx].getView();
                    int cNode = neighbors.get(i).getValue().node;
                    float cScore = neighbors.get(i).getValue().score;
                    VectorFloat<?> cVec = cView.getVector(cNode);
                    if (isDiverse(cNode, cView, cVec, cScore, neighbors, selected, currentAlpha, selectedCache)) {
                        selected.set(i);
                        nSelected++;
                        selectedCache.add(cIdx, cView, cNode, cScore, cVec);
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
        private boolean isDiverse(int cNode, OnDiskGraphIndex.View cView, VectorFloat<?> cVec, float cScore, List<Map.Entry<Integer, SearchResult.NodeScore>> others, BitSet selected, float alpha, SelectedVecCache selectedCache) {
            for (int j = 0; j < selectedCache.size; j++) {
                if (selectedCache.views[j] == cView && selectedCache.nodes[j] == cNode) {
                    break;
                }
                if (vsf.compare(cVec, selectedCache.vecs[j]) > cScore * alpha) {
                    return false;
                }
            }
            return true;
        }

    }
}

final class CompactWriter implements AutoCloseable {

    private static final int FOOTER_MAGIC = 0x4a564244;
    private static final int FOOTER_OFFSET_SIZE = Long.BYTES;
    private static final int FOOTER_MAGIC_SIZE = Integer.BYTES;
    private static final int FOOTER_SIZE = FOOTER_MAGIC_SIZE + FOOTER_OFFSET_SIZE;

    private final RandomAccessWriter writer;
    private final ImmutableGraphIndex upperLayerGraph;
    private final int numBaseLayerNodes;
    private final int dimension;
    private final int recordSize;
    private final long startOffset;
    private final int headerSize;
    private final Header header;
    private final int version;
    private final FusedPQ fusedPQFeature;
    private final InlineVectors inlineVectorFeature;
    private final int baseDegree;
    private final int maxOrdinal;
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final ThreadLocal<ByteBuffer> bufferPerThread;
    //private final ThreadLocal<ByteSequence<?>> zeroPQ = ThreadLocal.withInitial(() -> {
        //vectorTypeSupport.createByteSequence();
    //});;

    CompactWriter(Path outputPath,
                  int maxOrdinal,
                  int numBaseLayerNodes,
                  long startOffset,
                  ImmutableGraphIndex upperLayerGraph,
                  int dimension,
                  InlineVectors inlineVectorFeature,
                  FusedPQ fusedPQFeature)
    throws IOException {
        this.version = OnDiskGraphIndex.CURRENT_VERSION;
        this.writer = new BufferedRandomAccessWriter(outputPath);
        this.numBaseLayerNodes = numBaseLayerNodes;
        this.startOffset = startOffset;
        this.upperLayerGraph = upperLayerGraph;
        this.baseDegree = upperLayerGraph.getDegree(0);
        this.inlineVectorFeature = inlineVectorFeature;
        this.fusedPQFeature = fusedPQFeature;
        this.dimension = dimension;
        this.maxOrdinal = maxOrdinal;
        int rsize = Integer.BYTES // node ordinal
            + inlineVectorFeature.featureSize()
            + Integer.BYTES // neighbor count
            + baseDegree * Integer.BYTES; // neighbors + padding

        if(fusedPQFeature != null) {
            rsize += fusedPQFeature.featureSize();
        }
        this.recordSize = rsize;

        // header
        List<CommonHeader.LayerInfo> layerInfo = IntStream.rangeClosed(0, upperLayerGraph.getMaxLevel())
                    .mapToObj(i -> new CommonHeader.LayerInfo(upperLayerGraph.size(i), upperLayerGraph.getDegree(i)))
                    .collect(Collectors.toList());
        layerInfo.set(0, new CommonHeader.LayerInfo(numBaseLayerNodes, baseDegree));

        var commonHeader = new CommonHeader(this.version,
                dimension,
                upperLayerGraph.getView().entryNode().node,
                layerInfo,
                this.maxOrdinal + 1);
        Map<FeatureId, Feature> featureMap = new LinkedHashMap<>();
        featureMap.put(FeatureId.INLINE_VECTORS, inlineVectorFeature);
        if(fusedPQFeature != null) {
            featureMap.put(FeatureId.FUSED_PQ, fusedPQFeature);
        }
        this.header = new Header(commonHeader, featureMap);
        this.headerSize = header.size();

        this.bufferPerThread = ThreadLocal.withInitial(() -> {
            ByteBuffer buffer = ByteBuffer.allocate(recordSize);
            buffer.order(ByteOrder.BIG_ENDIAN);
            return buffer;
        });
    }

    public void writeHeader() throws IOException {
        writer.seek(startOffset);
        header.write(writer);
        assert writer.position() == startOffset + headerSize : String.format("%d != %d", writer.position(), startOffset + headerSize);
        writer.flush();
    }

    void writeFooter() throws IOException {
        long headerOffset = writer.position();
        header.write(writer); // write the header
        writer.writeLong(headerOffset); // We write the offset of the header at the end of the file
        writer.writeInt(FOOTER_MAGIC);
        final long expectedPosition = headerOffset + headerSize + FOOTER_SIZE;
        assert writer.position() == expectedPosition : String.format("%d != %d", writer.position(), expectedPosition);
    }

    public void offsetAfterInline() throws IOException {
        long offset = startOffset + headerSize + (long) (maxOrdinal + 1) * recordSize;
        writer.seek(offset);
    }

    public void writeUpperLayers(PQVectors ulpqv, Map<Integer, Integer> rulmap) throws IOException {
        var view = upperLayerGraph.getView();
        // write sparse levels
        for (int level = 1; level <= upperLayerGraph.getMaxLevel(); level++) {
            int layerSize = upperLayerGraph.size(level);
            int layerDegree = upperLayerGraph.getDegree(level);
            int nodesWritten = 0;
            for (var it = upperLayerGraph.getNodes(level); it.hasNext(); ) {
                int ordinal = it.nextInt();
                // node id
                writer.writeInt(ordinal);
                // neighbors
                var neighbors = view.getNeighborsIterator(level, ordinal);
                writer.writeInt(neighbors.size());
                int n = 0;
                for ( ; n < neighbors.size(); n++) {
                    writer.writeInt(neighbors.nextInt());
                }
                assert !neighbors.hasNext() : "Mismatch between neighbor's reported size and actual size";
                // pad out to degree
                for (; n < layerDegree; n++) {
                    writer.writeInt(-1);
                }
                nodesWritten++;
            }
            if (nodesWritten != layerSize) {
                throw new IllegalStateException("Mismatch between layer size and nodes written");
            }
        }

        if (version == 6) {
            if (ulpqv != null && fusedPQFeature != null) {
                IntFunction<Feature.State> fusedPQFeatureStateSupplier = ordinal -> new FusedPQ.State(view, ulpqv, ordinal);

                if (upperLayerGraph.getMaxLevel() >= 1) {
                    int level = 1;
                    int layerSize = upperLayerGraph.size(level);
                    int nodesWritten = 0;
                    for (var it = upperLayerGraph.getNodes(level); it.hasNext(); ) {
                        int ordinal = it.nextInt();

                        writer.writeInt(ordinal);
                        fusedPQFeature.writeSourceFeature(writer, fusedPQFeatureStateSupplier.apply(rulmap.get(ordinal)));
                        nodesWritten++;
                    }
                    if (nodesWritten != layerSize) {
                        throw new IllegalStateException("Mismatch between layer 1 size and features written");
                    }
                } else {
                    // Write the source feature of the entry node
                    final int entryNode = view.entryNode().node;
                    writer.writeInt(entryNode);
                    fusedPQFeature.writeSourceFeature(writer, fusedPQFeatureStateSupplier.apply(rulmap.get(entryNode)));
                }
            }
        }
    }

    public void close() throws IOException {
        var view = upperLayerGraph.getView();
        final var endOfGraphPosition = writer.position();
        writer.seek(endOfGraphPosition);
        writer.flush();
        view.close();
    }


    public WriteResult writeInlineNodeRecord(int ordinal, VectorFloat<?> vec, NodeArray neighbors, List<ByteSequence<?>> neighborPQs) throws IOException
    {
        var bwriter = new ByteBufferIndexWriter(bufferPerThread.get());

        long fileOffset = startOffset + headerSize + ordinal * recordSize;
        bwriter.reset();
        bwriter.writeInt(ordinal);
        //TODO: handle omitted nodes?
        // write inline vector
        //inlineVectorFeature.writeInline(bwriter, new InlineVectors.State(nw.vec));
        //vectorTypeSupport.writeFloatVector(bwriter, nw.vec);

        for(int i = 0; i < vec.length(); ++i) {
            bwriter.writeFloat(vec.get(i));
        }

        // write fused PQ
        // since we build a graph in a streaming way,
        // we cannot use fusedPQfeature.writeInline
        if (fusedPQFeature != null) {
          // TODO: what if no neighbors? lenght will be incorrect
            int length = neighborPQs.get(0).length();
            int i = 0;
            for (; i < neighbors.size(); ++i) {
                vectorTypeSupport.writeByteSequence(bwriter, neighborPQs.get(i));
            }
            ByteSequence<?> zeros = vectorTypeSupport.createByteSequence(length);
            zeros.zero();
            for (; i < baseDegree; i++) {
                vectorTypeSupport.writeByteSequence(bwriter, zeros);
            }
        }

        // write neighbors list
        bwriter.writeInt(neighbors.size());
        int n = 0;
        for (; n < neighbors.size(); n++) {
            bwriter.writeInt(neighbors.getNode(n));
        }

        // pad out to base layer degree
        for (; n < baseDegree; n++) {
            bwriter.writeInt(-1);
        }

        // TODO: verify we wrote exactly the expected amount
        if (bwriter.bytesWritten() != recordSize) {
            throw new IllegalStateException(
                String.format("Record size mismatch for ordinal %d: expected %d bytes, wrote %d bytes, base degree: %d",
                              ordinal, recordSize, bwriter.bytesWritten(), baseDegree));
        }

        ByteBuffer dataCopy = bwriter.cloneBuffer();
        return new WriteResult(ordinal, fileOffset, dataCopy);
    }
}

