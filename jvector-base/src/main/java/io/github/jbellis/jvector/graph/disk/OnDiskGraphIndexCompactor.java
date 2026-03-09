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
import io.github.jbellis.jvector.graph.disk.CompactOptions;
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
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
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
    final ByteSequence<?> pqCode;

    WriteResult(int newOrdinal, long fileOffset, ByteBuffer data, ByteSequence<?> pqCode) {
        this.newOrdinal = newOrdinal;
        this.fileOffset = fileOffset;
        this.data = data;
        this.pqCode = pqCode;
    }
};

public final class OnDiskGraphIndexCompactor implements Accountable {
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
    private static final AtomicInteger threadCounter = new AtomicInteger(0);

    public OnDiskGraphIndexCompactor(List<OnDiskGraphIndex> sources) {

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
        }

        addHierarchy = this.sources.get(0).getMaxLevel() != 1;
        maxDegrees = this.sources.get(0).maxDegrees();
        dimension = this.sources.get(0).getDimension();
        this.rng = new Random(0);
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
        if(sources.size() < 2) {
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

    public void compact(Path outputPath, VectorSimilarityFunction similarityFunction, CompactOptions opts) throws FileNotFoundException {
        checkBeforeCompact();
        opts.validateStatic();
        opts.validateWithRuntime(dimension);

        ForkJoinPool executor;
        boolean ownsExecutor = false;

        if (opts.executor != null) {
            executor = opts.executor;
        } else {
            int threads = opts.taskWindowSize > 0
                ? Math.min(opts.taskWindowSize, Runtime.getRuntime().availableProcessors())
                : Runtime.getRuntime().availableProcessors();

            executor = new ForkJoinPool(threads);
            ownsExecutor = true;
        }
        int taskWindowSize = opts.effectiveTaskWindowSize();

        final boolean fusedPQEnabled = opts.writeFeatures.contains(FeatureId.FUSED_PQ);
        final boolean compressedPrecision = opts.precision == CompactOptions.CompactionPrecision.COMPRESSED;
        final PQVectors providedPQVectors = opts.compressionConfig.pqVectors.orElse(null);
        final boolean writePQSeparate = opts.compressionConfig.pqVectorsOutputPath != null;
        final Path pqVectorsPath = opts.compressionConfig.pqVectorsOutputPath;

        ProductQuantization pqResolved = null;
        if (fusedPQEnabled || writePQSeparate || compressedPrecision) {
            switch (opts.compressionConfig.kind) {
                case PQ_VECTORS:
                    pqResolved = providedPQVectors.getCompressor();
                    break;
                case PQ_CODEBOOK:
                    pqResolved = opts.compressionConfig.pqCodebook.orElseThrow();
                    break;
                case SOURCE_PQ:
                    pqResolved = resolvePQFromSources(opts.compressionConfig.sourcePolicy);
                    break;
                case NONE:
                    throw new IllegalArgumentException("Compression required but compressionConfig is NONE");
            }
        }


        final int pqLength = (pqResolved == null) ? -1 : pqResolved.compressedVectorSize();

        if (compressedPrecision) {
            boolean sourcePQAvailable = hasSourcePQ();
            if (providedPQVectors == null && !sourcePQAvailable) {
                throw new IllegalArgumentException(
                        "CompactionPrecision.COMPRESSED requires caller-provided PQVectors or source-side FUSED_PQ");
            }
        }

        numTotalNodes = 0;
        for(OnDiskGraphIndex source : sources) {
            numTotalNodes += liveNodes.get(source).cardinality();
        }

        // first stage: find the nodes for upper layer graph and prepare nodes and alive
        final FixedBitSet[] aliveBySource = new FixedBitSet[sources.size()];
        final int[][] nodesBySource = new int[sources.size()][];
        for(int s = 0; s < sources.size(); ++s) {
            var source = sources.get(s);
            NodesIterator sourceNodes = source.getNodes(0);
            FixedBitSet sourceAlive = liveNodes.get(source);
            int[] nodes = new int[sourceNodes.size()];
            int i = 0;
            while (sourceNodes.hasNext()) {
                int node = sourceNodes.next();
                nodes[i++] = node;
                if (!sourceAlive.get(node)) continue;
                int level = getRandomGraphLevel();
                if (level > 0) {
                    var nodeLevel = new OnDiskGraphIndex.NodeAtLevel(level, node);
                    upperLayerNodeList.add(Map.entry(source, nodeLevel));
                }
            }
            aliveBySource[s] = sourceAlive;
            nodesBySource[s] = nodes;
        }

        log.info("Upper layer candidates selected: {} nodes across {} sources", upperLayerNodeList.size(), sources.size());

        // second stage: construct the upper layer graph without base layer
        UpperLayerOrdinalMapper upperLayerOrdinalMapper = new UpperLayerOrdinalMapper(upperLayerNodeList);
        var ulravv = new UpperLayerRandomAccessVectorValues(upperLayerOrdinalMapper);
        OnHeapGraphIndex upperLayerGraph = constructUpperLayerGraph(upperLayerOrdinalMapper, ulravv, similarityFunction);
        log.info("Upper layer graph constructed");

        PQVectors ulpqv;
        int[] ulmap;
        RemappedRandomAccessVectorValues ulmrav;
        Map<Integer, Integer> rulmap;
        if (fusedPQEnabled || writePQSeparate) {
            // For upperlayer nodes,
            // we cannot directly use encodeAll as it encodes ordinals. 
            // The upperLayerRandomAccessValues are map-based and can be non-contiguous.
            // Use ordinals mapping/reverse mapping to solve this issue.
            int i = 0;
            ulmap = new int[upperLayerNodeList.size()];
            rulmap = new HashMap<>();
            for(Integer key: upperLayerOrdinalMapper.newToOld.keySet()) {
                ulmap[i] = key;
                rulmap.put(key, i++);
            }
            ulmrav = new RemappedRandomAccessVectorValues(ulravv, ulmap);
            ulpqv = (PQVectors) pqResolved.encodeAll(ulmrav);
        }
        else {
            ulpqv = null;
            ulmap = null;
            ulmrav = null;
            rulmap = null;
        }

        // third stage: write base layer nodes, then let writer handle upper layers
        log.info("Writing compacted graph: {} total nodes, maxOrdinal={}, dimension={}, degree={}",
                numTotalNodes, maxOrdinal, dimension, maxDegrees.get(0));
        try(CompactWriter writer = new CompactWriter(outputPath, maxOrdinal, numTotalNodes, 0, upperLayerGraph, dimension, pqResolved, pqLength, fusedPQEnabled, writePQSeparate)) {
        writer.writeHeader();

        CompactVamanaDiversityProvider vdp = new CompactVamanaDiversityProvider(similarityFunction, 1.2f);
        AtomicInteger batchesCompleted = new AtomicInteger(0);
        int totalBatches = 0; // counted below
        int submitted = 0;
        int searchTopK = Math.max(2, (maxDegrees.get(0) + sources.size() - 1) / sources.size());
        int beamWidth;
        if(compressedPrecision) {
            beamWidth = searchTopK * 2;
        } 
        else {
            beamWidth = searchTopK;
        }
        int maxCandidateSize = searchTopK * sources.size() + maxDegrees.get(0);
        ArrayBlockingQueue<Scratch> scratchPool = new ArrayBlockingQueue<>(taskWindowSize);
        for(int p = 0; p < taskWindowSize; ++p) scratchPool.add(new Scratch(maxCandidateSize, maxDegrees.get(0), dimension, sources, pqResolved));

        ExecutorCompletionService<List<WriteResult>> ecs = new ExecutorCompletionService<>(executor);
        List<BatchSpec> batches = new ArrayList<>();
        for (int s = 0; s < sources.size(); s++) {
            int[] nodes = nodesBySource[s];
            FixedBitSet sourceAlive = aliveBySource[s];
            int numNodes = nodes.length;

            int numBatches = max(40, (numNodes + 128 - 1) / 128);
            if (numBatches > numNodes) numBatches = numNodes;
            totalBatches += numBatches;

            int batchSize = (numNodes + numBatches - 1) / numBatches;
            for (int b = 0; b < numBatches; ++b) {
                int start = min(numNodes, batchSize * b);
                int end   = min(numNodes, batchSize * (b + 1));
                batches.add(new BatchSpec(s, nodes, start, end, sourceAlive));
            }
        }
        final int total = batches.size();
        log.info("Prepared {} compute batches across {} sources (parallelism={})",
            total, sources.size(), executor.getParallelism());
        java.util.function.Consumer<BatchSpec> submitOne = (BatchSpec bs) -> {
            ecs.submit(() -> {
                List<WriteResult> wrs = new ArrayList<>(bs.end - bs.start);
                Scratch scratch = scratchPool.take();
                OnDiskGraphIndex.View sourceView = (OnDiskGraphIndex.View) scratch.gs[bs.sourceIdx].getView();
                VectorFloat<?> tmpVec = scratch.tmpVec;
                for(int i = bs.start; i < bs.end; ++i) {
                    int node = bs.nodes[i];
                    if (!bs.sourceAlive.get(node)) continue;
                    VectorFloat<?> baseVec = scratch.baseVec;
                    int[] candSrc = scratch.candSrc;
                    int[] candNode = scratch.candNode;
                    float[] candScore = scratch.candScore;
                    sourceView.getVectorInto(node, baseVec, 0);
                    int candSize = 0;

                    for (int ss = 0; ss < sources.size(); ++ss) {
                        OnDiskGraphIndex idx = sources.get(ss);
                        FixedBitSet indexAlive = liveNodes.get(idx);
                        var searchView = (OnDiskGraphIndex.View) scratch.gs[ss].getView();

                        if (bs.sourceIdx == ss) {
                            // use existing neighbors as candidates
                            var it = searchView.getNeighborsIterator(0, node);
                            while(it.hasNext()) {
                                int nb = it.nextInt();
                                if(!indexAlive.get(nb)) continue;
                                searchView.getVectorInto(nb, tmpVec, 0);
                                float score = similarityFunction.compare(baseVec, tmpVec);
                                candSrc[candSize] = ss;
                                candNode[candSize] = nb;
                                candScore[candSize] = score;
                                candSize++;
                            }

                        } else {
                            SearchScoreProvider ssp = buildCrossSourceScoreProvider(
                                    compressedPrecision,
                                    providedPQVectors,
                                    remappers.get(idx),
                                    searchView,
                                    baseVec,
                                    tmpVec,
                                    similarityFunction);

                            // PRIORITY
                            // TODO: clarify heuristics approach, identify key questions and verification methods
                            // TODO: parameterize topK and beamWidth in JMH coverage
                            // TODO: validate assumption that recall is stable enough between original contributing graphs and the compacted graph
                            // FUTURE
                            // TODO: ensure that result metrics contain recall first as a correctness measure, and then add performance data
                            //       to include completion time, merge speeds, etc
                            //       to include resource usage variations for compaction and search
                            //SearchResult results = tlSearchers.get()[ss].search(ssp, maxDegrees.get(0), maxDegrees.get(0), 0.0f, 0.0f, indexAlive);
                            SearchResult results = scratch.gs[ss].search(ssp, searchTopK, beamWidth, 0.f, 0.0f, indexAlive);
                            for (SearchResult.NodeScore re : results.getNodes()) {
                                candSrc[candSize] = ss;
                                candNode[candSize] = re.node;
                                candScore[candSize] = re.score;
                                candSize++;
                            }
                        }

                    }

                    int[] order = IntStream.range(0, candSize).toArray();
                    sortOrderByScoreDesc(order, candScore, candSize);

                    SelectedVecCache selectedCache = scratch.selectedCache;
                    vdp.retainDiverse(candSrc, candNode, candScore, order, candSize, maxDegrees.get(0), selectedCache, tmpVec, scratch.gs);

                    for (int k = 0; k < selectedCache.size; k++) {
                        selectedCache.nodes[k] = remappers.get(sources.get(selectedCache.sourceIdx[k])).oldToNew(selectedCache.nodes[k]);
                    }

                    int newOrdinal = remappers.get(sources.get(bs.sourceIdx)).oldToNew(node);
                    wrs.add(writer.writeInlineNodeRecord(newOrdinal, baseVec, selectedCache, scratch.pqCode));
                }
                int done = batchesCompleted.incrementAndGet();
                if (done % 10 == 0) {
                    log.info("Compaction progress: {} batches computed so far ({} nodes in this batch)",
                            done, wrs.size());
                }
                scratchPool.put(scratch);
                return wrs;
            });
        };

        int nextToSubmit = 0;
        int inFlight = 0;
        while (inFlight < taskWindowSize && nextToSubmit < total) {
            submitOne.accept(batches.get(nextToSubmit++));
            inFlight++;
        }

        var wropts = EnumSet.of(StandardOpenOption.WRITE, StandardOpenOption.READ);
        // Create PQ writer if needed (same thread that does IO writes)
        PQVectorsWriter pqWriter = null;
        if (writePQSeparate) {
            if (pqResolved == null) {
              throw new IllegalStateException("pqVectorsOutput enabled but pqResolved is null");
            }
            int vectorCount = maxOrdinal + 1; // must match PQVectors.load expectations
            pqWriter = new PQVectorsWriter(
                opts.compressionConfig.pqVectorsOutputPath,
                pqResolved,
                vectorCount,
                OnDiskGraphIndex.CURRENT_VERSION
            );
        }
        try(FileChannel fc = FileChannel.open(outputPath, wropts)) {
            int completed = 0;
            while(completed < total) {
                List<WriteResult> results = ecs.take().get();
                for(WriteResult r: results) {
                    // 1) write index record to outputPath
                    ByteBuffer b = r.data;
                    long pos = r.fileOffset;
                    while (b.hasRemaining()) {
                        int n = fc.write(b, pos);
                        pos += n;
                    }
                    // 2) write PQ code to pqOutputPath (PQVectors format) if enabled
                    if (pqWriter != null) {
                        if (r.pqCode != null) {
                            pqWriter.writeCode(r.newOrdinal, r.pqCode);
                        }
                    }
                }
                completed++;
                inFlight--;

                // refill the window by submitting one more batch
                if (nextToSubmit < total) {
                    submitOne.accept(batches.get(nextToSubmit++));
                    inFlight++;
                }

                if (completed % 10 == 0) {
                    log.info("Compaction I/O progress: {}/{} batches written to disk", completed, total);
                }
            }
        }
        log.info("All {} batches written to disk, writing upper layers and footer", totalBatches);
        writer.offsetAfterInline();
        writer.writeUpperLayers(ulpqv, rulmap);
        writer.writeFooter();
        writer.close();
        for (Scratch s : scratchPool) {
            s.close();
        }
        log.info("Compaction complete: {}", outputPath);
        } catch (IOException | ExecutionException | InterruptedException e) {
            throw new RuntimeException(e);
        }
        finally {
            if(ownsExecutor) executor.shutdown();
        }
    }

    private ProductQuantization resolvePQFromSources(CompactOptions.CompressionConfig.PQSourcePolicy policy) {
        switch (policy) {
          case FIRST: {
              FusedPQ fpq = (FusedPQ) sources.get(0).getFeatures().get(FeatureId.FUSED_PQ);
              if (fpq == null) {
                throw new IllegalArgumentException("FIRST policy but source[0] has no FUSED_PQ");
              }
              return fpq.getPQ();
          }

          case LARGEST_LIVE:
          case AUTO: {
              int best = 0;
              int bestLive = -1;

              for (int i = 0; i < sources.size(); i++) {
                  int c = liveNodes.get(sources.get(i)).cardinality();
                  if (c > bestLive) {
                      bestLive = c;
                      best = i;
                  }
              }

              FusedPQ fpq = (FusedPQ) sources.get(best).getFeatures().get(FeatureId.FUSED_PQ);
              if (fpq == null) {
                  throw new IllegalArgumentException("LARGEST_LIVE policy but best source has no FUSED_PQ");
              }

              return fpq.getPQ();
          }

          default:
              throw new IllegalArgumentException("Unhandled policy: " + policy);
        }
    }

    private boolean hasSourcePQ() {
        for (OnDiskGraphIndex source : sources) {
            if (source.getFeatures().containsKey(FeatureId.FUSED_PQ)) {
                return true;
            }
        }
        return false;
    }

    private SearchScoreProvider buildCrossSourceScoreProvider(boolean compressedPrecision,
                                                              PQVectors providedPQVectors,
                                                              OrdinalMapper remapper,
                                                              OnDiskGraphIndex.View searchView,
                                                              VectorFloat<?> baseVec,
                                                              VectorFloat<?> tmpVec,
                                                              VectorSimilarityFunction similarityFunction) {
        if (compressedPrecision) {
            if (providedPQVectors != null) {
                var approx = providedPQVectors.scoreFunctionFor(baseVec, similarityFunction);
                var remapped = new ScoreFunction.ApproximateScoreFunction() {
                    @Override
                    public float similarityTo(int node2) {
                        int globalOrdinal = remapper.oldToNew(node2);
                        return approx.similarityTo(globalOrdinal);
                    }
                };
                return new DefaultSearchScoreProvider(remapped);
            }

            return new DefaultSearchScoreProvider(
                    searchView.approximateScoreFunctionFor(baseVec, similarityFunction));
        }

        var sf = new ScoreFunction.ExactScoreFunction() {
            @Override
            public float similarityTo(int node2) {
                searchView.getVectorInto(node2, tmpVec, 0);
                return similarityFunction.compare(baseVec, tmpVec);
            }
        };
        return new DefaultSearchScoreProvider(sf);
    }

    public OnHeapGraphIndex constructUpperLayerGraph(UpperLayerOrdinalMapper upperLayerOrdinalMapper, UpperLayerRandomAccessVectorValues ulravv, VectorSimilarityFunction similarityFunction) {

        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ulravv, similarityFunction);
        OnHeapGraphIndex upperLayerGraph = new OnHeapGraphIndex(maxDegrees, dimension, neighborOverflow, new VamanaDiversityProvider(bsp, 1.2f), addHierarchy);
        GraphSearcher searchers = new GraphSearcher(upperLayerGraph);
        searchers.usePruning(false);

        VectorFloat<?> searchVec = vectorTypeSupport.createFloatVector(dimension);
        VectorFloat<?> tmpVec = vectorTypeSupport.createFloatVector(dimension);
        var view = searchers.getView();
        for(var node: upperLayerNodeList) {
            var nodeLevel = node.getValue();
            int newOrdinal = upperLayerOrdinalMapper.oldToNew(node);
            var newNodeLevel = new OnDiskGraphIndex.NodeAtLevel(nodeLevel.level, newOrdinal);
            upperLayerGraph.addNode(newNodeLevel);

            node.getKey().getView().getVectorInto(node.getValue().node, searchVec, 0);

            //SearchScoreProvider upperLayerGraphSsp = DefaultSearchScoreProvider.exact(searchVec, similarityFunction, ulravv);
            var sf = new ScoreFunction.ExactScoreFunction() {
                 @Override
                 public float similarityTo(int node2) {
                     ulravv.getVectorInto(node2, tmpVec, 0);
                     return similarityFunction.compare(searchVec, tmpVec);
                 }
            };
            SearchScoreProvider upperLayerGraphSsp = new DefaultSearchScoreProvider(sf);

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
                        searchers.searchOneLayer(upperLayerGraphSsp, 1, 0.0f, lvl, view.liveNodes());
                    } else {
                        searchers.searchOneLayer(upperLayerGraphSsp, 100, 0.0f, lvl, view.liveNodes());
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

    // TODO: outdated
    @Override
    public long ramBytesUsed() {
        int OH = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        int REF = RamUsageEstimator.NUM_BYTES_OBJECT_REF;

        // Shallow size of this object (header + fields)
        // fields: sources, liveNodes, remappers, upperLayerNodeList, maxDegrees,
        //         neighborOverflow(float), addHierarchy(boolean), dimension(int), rng,
        //         maxOrdinal(int), numTotalNodes(int), executor, tlSearchers
        long size = OH + 13L * REF + Integer.BYTES * 3 + Float.BYTES + 1;

        // liveNodes: FixedBitSet per source
        for (var entry : liveNodes.values()) {
            size += entry.ramBytesUsed();
        }

        // remappers: each MapMapper holds an oldToNew HashMap and newToOld Int2IntHashMap
        // Estimate based on the number of mappings
        for (var mapper : remappers.values()) {
            // Object overhead + two maps with int key/value pairs
            // HashMap entry: ~32 bytes each; Int2IntHashMap: ~16 bytes per entry
            if (mapper instanceof OrdinalMapper.MapMapper) {
                // rough estimate: the mapper stores two maps over all mapped ordinals
                size += OH + (long) (maxOrdinal + 1) * 48;
            }
        }

        // upperLayerNodeList: List of Map.Entry<OnDiskGraphIndex, NodeAtLevel>
        // Each entry: OH + 2*REF, NodeAtLevel: OH + int + int
        long entrySize = OH + 2L * REF + OH + Integer.BYTES + Integer.BYTES;
        size += OH + REF + (long) upperLayerNodeList.size() * entrySize;

        // maxDegrees: small list of integers
        size += OH + REF + (long) maxDegrees.size() * (OH + Integer.BYTES);

        // tlSearchers: ConcurrentHashMap of GraphSearcher[] per thread
        // Cannot measure precisely since it depends on the number of active threads;
        // account for the container overhead only
        size += OH + REF;

        return size;
    }
    static void sortOrderByScoreDesc(int[] order, float[] score, int size) {
        quicksort(order, score, 0, size - 1);
    }

    private static void quicksort(int[] order, float[] score, int lo, int hi) {
        while (lo < hi) {
            int p = partition(order, score, lo, hi);
            // recurse smaller side first (limits stack)
            if (p - lo < hi - p) {
                quicksort(order, score, lo, p - 1);
                lo = p + 1;
            } else {
                quicksort(order, score, p + 1, hi);
                hi = p - 1;
            }
        }
    }

    private static int partition(int[] order, float[] score, int lo, int hi) {
        float pivot = score[order[hi]];
        int i = lo;
        for (int j = lo; j < hi; j++) {
            if (score[order[j]] > pivot) { // DESC
                int t = order[i]; order[i] = order[j]; order[j] = t;
                i++;
            }
        }
        int t = order[i]; order[i] = order[hi]; order[hi] = t;
        return i;
    }

    final class Scratch implements AutoCloseable {

      final int[] candSrc, candNode;
      final float[] candScore;
      final SelectedVecCache selectedCache;
      final VectorFloat<?> tmpVec, baseVec;
      final GraphSearcher[] gs;
      final ByteSequence<?> pqCode;

      Scratch(int maxCandidateSize, int maxDegree, int dimension, List<OnDiskGraphIndex> sources, ProductQuantization pq) {
          this.candSrc = new int[maxCandidateSize];
          this.candNode = new int[maxCandidateSize];
          this.candScore = new float[maxCandidateSize];
          this.selectedCache = new SelectedVecCache(maxDegree);
          this.tmpVec = vectorTypeSupport.createFloatVector(dimension);
          this.baseVec = vectorTypeSupport.createFloatVector(dimension);
          this.pqCode = (pq == null) ? null : vectorTypeSupport.createByteSequence(pq.getSubspaceCount());

          this.gs = new GraphSearcher[sources.size()];
          for (int i = 0; i < sources.size(); i++) {
              gs[i] = new GraphSearcher(sources.get(i));
              gs[i].usePruning(false);
          }
      }

      @Override
      public void close() throws IOException {
          for (var s : gs) s.close();
          selectedCache.reset();
      }
    }


    private static final class BatchSpec {
        final int sourceIdx;
        final int[] nodes;              // materialized node ids for this source
        final int start;
        final int end;
        final FixedBitSet sourceAlive;

        BatchSpec(int sourceIdx, int[] nodes, int start, int end, FixedBitSet sourceAlive) {
            this.sourceIdx = sourceIdx;
            this.nodes = nodes;
            this.start = start;
            this.end = end;
            this.sourceAlive = sourceAlive;
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
        public void getVectorInto(int nodeId, VectorFloat<?> vector, int offset) {
            var node = upperLayerOrdinalMapper.newToOld(nodeId);
            node.getKey().getView().getVectorInto(node.getValue().node, vector, offset);
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
        public double retainDiverse(int[] candSrc, int[] candNode, float[] candScore, int[] order, int orderSize, int maxDegree, SelectedVecCache selectedCache, VectorFloat<?> tmp, GraphSearcher[] gs) {
            selectedCache.reset();
            if (orderSize == 0) return Double.NaN;
            int nSelected = 0;
            double shortEdges = Double.NaN;

            // add diverse candidates, gradually increasing alpha to the threshold
            // (so that the nearest candidates are prioritized)
            float currentAlpha = 1.0f;
            while (currentAlpha <= alpha + 1E-6 && nSelected < maxDegree) {
                for (int i = 0; i < orderSize && nSelected < maxDegree; i++) {
                    int ci = order[i];
                    int cSrc = candSrc[ci];
                    int cNode = candNode[ci];
                    float cScore = candScore[ci];

                    OnDiskGraphIndex.View cView = (OnDiskGraphIndex.View) gs[cSrc].getView();
                    cView.getVectorInto(cNode, tmp, 0);
                    if (isDiverse(cView, cNode, tmp, cScore, currentAlpha, selectedCache)) {
                        selectedCache.add(cSrc, cView, cNode, cScore, tmp);
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
        private boolean isDiverse(OnDiskGraphIndex.View cView, int cNode, VectorFloat<?> cVec, float cScore, float alpha, SelectedVecCache selectedCache) {
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
    private final int recordSize;
    private final long startOffset;
    private final int headerSize;
    private final Header header;
    private final int version;
    private final FusedPQ fusedPQFeature;
    private final ProductQuantization pq;
    private final int baseDegree;
    private final int maxOrdinal;
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final ThreadLocal<ByteBuffer> bufferPerThread;
    private final ThreadLocal<ByteSequence<?>> zeroPQ;
    private final boolean fusedPQEnabled;
    private final boolean writePQSeparate;

    CompactWriter(Path outputPath,
                  int maxOrdinal,
                  int numBaseLayerNodes,
                  long startOffset,
                  ImmutableGraphIndex upperLayerGraph,
                  int dimension,
                  ProductQuantization pq,
                  int pqLength,
                  boolean fusedPQEnabled,
                  boolean writePQSeparate)
    throws IOException {
        this.fusedPQEnabled = fusedPQEnabled;
        this.writePQSeparate = writePQSeparate;
        this.version = OnDiskGraphIndex.CURRENT_VERSION;
        this.writer = new BufferedRandomAccessWriter(outputPath);
        this.startOffset = startOffset;
        this.upperLayerGraph = upperLayerGraph;
        this.baseDegree = upperLayerGraph.getDegree(0);
        this.pq = pq;
        this.maxOrdinal = maxOrdinal;

        Map<FeatureId, Feature> featureMap = new LinkedHashMap<>();
        InlineVectors inlineVectorFeature = new InlineVectors(dimension);
        featureMap.put(FeatureId.INLINE_VECTORS, inlineVectorFeature);
        if(fusedPQEnabled) {
            this.fusedPQFeature = new FusedPQ(upperLayerGraph.maxDegree(), pq);
            featureMap.put(FeatureId.FUSED_PQ, this.fusedPQFeature);
        }
        else {
            this.fusedPQFeature = null;
        }
        int rsize = Integer.BYTES // node ordinal
            + inlineVectorFeature.featureSize()
            + Integer.BYTES // neighbor count
            + baseDegree * Integer.BYTES; // neighbors + padding

        if(fusedPQEnabled) {
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
        this.header = new Header(commonHeader, featureMap);
        this.headerSize = header.size();

        this.bufferPerThread = ThreadLocal.withInitial(() -> {
            ByteBuffer buffer = ByteBuffer.allocate(recordSize);
            buffer.order(ByteOrder.BIG_ENDIAN);
            return buffer;
        });
        this.zeroPQ = ThreadLocal.withInitial(() -> {
            var vec = vectorTypeSupport.createByteSequence(pqLength);
            vec.zero();
            return vec;
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

        if (fusedPQEnabled && version == 6) {
            if (ulpqv != null) {
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


    public WriteResult writeInlineNodeRecord(int ordinal, VectorFloat<?> vec, SelectedVecCache selectedCache, ByteSequence<?> pqCode) throws IOException
    {
        var bwriter = new ByteBufferIndexWriter(bufferPerThread.get());

        long fileOffset = startOffset + headerSize + (long) ordinal * recordSize;
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
        if (fusedPQEnabled) {
            int k = 0;
            for (; k < selectedCache.size; k++) {
                pqCode.zero();
                pq.encodeTo(selectedCache.vecs[k], pqCode);
                vectorTypeSupport.writeByteSequence(bwriter, pqCode);
            }
            for (; k < baseDegree; k++) {
                vectorTypeSupport.writeByteSequence(bwriter, zeroPQ.get());
            }
        }

        // write neighbors list
        bwriter.writeInt(selectedCache.size);
        int n = 0;
        for (; n < selectedCache.size; n++) {
            bwriter.writeInt(selectedCache.nodes[n]);
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

        ByteSequence pqCopy = null;
        if (writePQSeparate) {
            // encode into scratch ByteSequence to avoid allocations inside pq.encode()
            pqCode.zero();
            pq.encodeTo(vec, pqCode);

            pqCopy = pqCode.copy();
        }

        return new WriteResult(ordinal, fileOffset, dataCopy, pqCopy);
    }
}

final class SelectedVecCache {
    int[] sourceIdx;
    OnDiskGraphIndex.View[] views;
    int[] nodes;
    float[] scores;
    VectorFloat<?>[] vecs;
    int size;

    SelectedVecCache(int capacity) {
        sourceIdx = new int[capacity];
        views = new OnDiskGraphIndex.View[capacity];
        nodes = new int[capacity];
        scores = new float[capacity];
        vecs = new VectorFloat<?>[capacity];
        size = 0;
    }

    void reset() { 
        Arrays.fill(vecs, 0, size, null);
        Arrays.fill(views, 0, size, null);
        size = 0; 
    }

    void add(int source, OnDiskGraphIndex.View view, int node, float score, VectorFloat<?> vec) {
        sourceIdx[size] = source;
        views[size] = view;
        nodes[size] = node;
        scores[size] = score;
        vecs[size] = vec.copy();
        size++;
    }
}

final class PQVectorsWriter implements AutoCloseable {

    private static final VectorTypeSupport vts =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    private final RandomAccessWriter out;
    private final ProductQuantization pq;
    private final int vectorCount;
    private final int subspaceCount;
    private final PQVectors.PQLayout layout;
    private final long dataStartOffset;

    /**
     * Creates a PQVectors separate file that is readable by {@link PQVectors#load(RandomAccessReader)}.
     *
     * File layout (exactly PQVectors.write/load):
     *   pq.write(out, version)
     *   out.writeInt(vectorCount)
     *   out.writeInt(subspaceCount)   // NOTE: PQVectors.write uses pq.getSubspaceCount()
     *   raw chunk bytes (full chunks, then last chunk)
     *
     * This constructor also pre-allocates/writes zeros for the chunk region so load() can read it fully.
     */
    public PQVectorsWriter(Path path, ProductQuantization pq, int vectorCount, int version) throws IOException {
        this.out = new BufferedRandomAccessWriter(path);
        this.pq = pq;
        this.vectorCount = vectorCount;
        this.subspaceCount = pq.getSubspaceCount();

        // PQLayout takes (vectorCount, "compressedDimension"), and PQVectors.write stores subspaceCount there.
        this.layout = new PQVectors.PQLayout(vectorCount, subspaceCount);

        // 1) write codebook
        pq.write(out, version);

        // 2) write vectorCount and compressedDimension (subspaceCount)
        out.writeInt(vectorCount);
        out.writeInt(subspaceCount);

        // Remember where the chunk region begins
        this.dataStartOffset = out.position();

        // 3) pre-write zero chunks so PQVectors.load can read all bytes.
        //    We must write exactly the same chunk sizes PQVectors.write emits.
        writeZeroChunkRegion();

        out.flush();
    }

    private void writeZeroChunkRegion() throws IOException {
        // We can reuse a single ByteSequence for full chunks and one for last chunk to avoid big allocations.
        ByteSequence<?> fullZero = vts.createByteSequence(layout.fullChunkBytes);
        fullZero.zero();

        for (int i = 0; i < layout.fullSizeChunks; i++) {
            vts.writeByteSequence(out, fullZero);
        }

        if (layout.totalChunks > layout.fullSizeChunks) {
            ByteSequence<?> lastZero = vts.createByteSequence(layout.lastChunkBytes);
            lastZero.zero();
            vts.writeByteSequence(out, lastZero);
        }
    }

    /**
     * Random-access write the PQ code for a given ordinal into the chunk region.
     * This does NOT write the codebook; codebook is written once in the constructor.
     *
     * @param ordinal in [0, vectorCount)
     * @param code a ByteSequence of length == pq.getSubspaceCount()
     */
    public void writeCode(int ordinal, ByteSequence<?> code) throws IOException {
        if (ordinal < 0 || ordinal >= vectorCount) {
            throw new IndexOutOfBoundsException("ordinal " + ordinal + " out of bounds [0," + vectorCount + ")");
        }
        if (code.length() != subspaceCount) {
            throw new IllegalArgumentException("PQ code length " + code.length()
                    + " != subspaceCount " + subspaceCount);
        }

        final int chunkIndex = ordinal / layout.fullChunkVectors;
        final int indexInChunk = ordinal % layout.fullChunkVectors;
        final long chunkBase = dataStartOffset + (long) chunkIndex * layout.fullChunkBytes;
        final long offset = chunkBase + (long) indexInChunk * subspaceCount;

        out.seek(offset);
        // IMPORTANT: write raw bytes only (no length prefix) to match PQVectors.write/load.
        vts.writeByteSequence(out, code);
    }

    public long dataStartOffset() {
        return dataStartOffset;
    }

    public int vectorCount() {
        return vectorCount;
    }

    public int subspaceCount() {
        return subspaceCount;
    }

    @Override
    public void close() throws IOException {
        out.flush();
        out.close();
    }
}
