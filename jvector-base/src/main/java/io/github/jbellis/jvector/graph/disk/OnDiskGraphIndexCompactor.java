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
import java.nio.channels.FileChannel;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.disk.BufferedRandomAccessWriter;
import io.github.jbellis.jvector.disk.RandomAccessWriter;
import io.github.jbellis.jvector.disk.ByteBufferIndexWriter;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.*;
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

final class UpperLayerWriteResult {
    final int ordinal;
    final int[] neighbors;
    final int size;
    final ByteSequence<?> pqCode;

    UpperLayerWriteResult(int ordinal, SelectedVecCache cache, ByteSequence<?> pqCode) {
        this.ordinal = ordinal;
        this.neighbors = Arrays.copyOf(cache.nodes, cache.size);
        this.size = cache.size;
        this.pqCode = pqCode == null ? null : pqCode.copy();
    }
};

public final class OnDiskGraphIndexCompactor implements Accountable {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final Logger log = LoggerFactory.getLogger(OnDiskGraphIndexCompactor.class);

    private final List<OnDiskGraphIndex> sources;
    private final List<FixedBitSet> liveNodes;
    private final List<OrdinalMapper> remappers;
    private final List<Map.Entry<Integer, OnDiskGraphIndex.NodeAtLevel>> upperLayerNodeList;
    private final List<Integer> maxDegrees;

    private final float neighborOverflow = 1.2f;
    private final int dimension;
    private int maxOrdinal = -1;
    private int numTotalNodes = 0;
    private boolean ownsExecutor = false;
    private final ForkJoinPool executor;
    private final int taskWindowSize;
    private final VectorSimilarityFunction similarityFunction;

    public OnDiskGraphIndexCompactor(
            List<OnDiskGraphIndex> sources,
            List<FixedBitSet> liveNodes,
            List<OrdinalMapper> remappers,
            List<Path> fpVectorsPaths,
            VectorSimilarityFunction similarityFunction,
            ForkJoinPool executor) {
        checkBeforeCompact(sources, liveNodes, remappers, fpVectorsPaths);

        int threads = Runtime.getRuntime().availableProcessors();
        if (executor != null) {
            this.executor = executor;
        } else {
            this.executor = new ForkJoinPool(threads);
            this.ownsExecutor = true;
        }
        this.taskWindowSize = threads * 2;

        this.sources = sources;
        this.remappers = remappers;
        this.liveNodes = liveNodes;
        for (int s = 0; s < this.sources.size(); s++) {
            this.numTotalNodes += this.liveNodes.get(s).cardinality();
        }

        this.upperLayerNodeList = new ArrayList<>();
        maxDegrees = this.sources.get(0).maxDegrees();
        dimension = this.sources.get(0).getDimension();
        for (var mapper : remappers) {
            maxOrdinal = max(mapper.maxOrdinal(), maxOrdinal);
        }
        this.similarityFunction = similarityFunction;
    }

    private void checkBeforeCompact(
            List<OnDiskGraphIndex> sources,
            List<FixedBitSet> liveNodes,
            List<OrdinalMapper> remappers,
            List<Path> fpVectorsPaths) {
        if (sources.size() < 2) {
            throw new IllegalArgumentException("Must have at least two sources");
        }
        Objects.requireNonNull(liveNodes, "liveNodes");
        Objects.requireNonNull(remappers, "remappers");
        if (sources.size() != liveNodes.size()) {
            throw new IllegalArgumentException("sources and liveNodes must have the same size");
        }
        if (sources.size() != remappers.size()) {
            throw new IllegalArgumentException("sources and remappers must have the same size");
        }
        if (fpVectorsPaths != null && sources.size() != fpVectorsPaths.size()) {
            throw new IllegalArgumentException("fpInputPaths must be null or match the number of sources");
        }

        for (int s = 0; s < sources.size(); ++s) {
            if (liveNodes.get(s).length() != sources.get(s).size(0)) {
                throw new IllegalArgumentException("source " + s + " out of bounds");
            }
        }

        // check dimensions
        int dimension = sources.get(0).getDimension();
        var maxDegrees = sources.get(0).maxDegrees();
        var addHierarchy = sources.get(0).isHierarchical();

        for (OnDiskGraphIndex source : sources) {
            if (source.getDimension() != dimension) {
                throw new IllegalArgumentException("sources must have the same dimension");
            }
            for (int d = 0; d < maxDegrees.size(); d++) {
                if (!Objects.equals(source.maxDegrees().get(d), maxDegrees.get(d))) {
                    throw new IllegalArgumentException("sources must have the same max degrees");
                }
            }
            if (addHierarchy != source.isHierarchical()) {
                throw new IllegalArgumentException("sources must have the same hierarchical setting");
            }
        }

        // check features
        Set<FeatureId> refKeys = sources.get(0).getFeatures().keySet();
        boolean sameFeatures = sources.stream()
                .skip(1)
                .map(s -> s.getFeatures().keySet())
                .allMatch(refKeys::equals);

        if (!sameFeatures) {
            throw new IllegalArgumentException("Each source must have the same features");
        }
        if (!refKeys.contains(FeatureId.INLINE_VECTORS) && (fpVectorsPaths == null || fpVectorsPaths.size() != sources.size())) {
            throw new IllegalArgumentException("Each source must have the INLINE_VECTORS feature or corresponding fp vectors path");
        }
        if (!refKeys.contains(FeatureId.INLINE_VECTORS) && !refKeys.contains(FeatureId.FUSED_PQ)) {
            throw new IllegalArgumentException("Current compaction support INLINE_VECTORS and FUSED_PQ only");
        }
    }

    public void compact(Path outputPath) throws FileNotFoundException {
        boolean fusedPQEnabled = hasFusedPQ();
        boolean compressedPrecision = fusedPQEnabled;

        ProductQuantization pq;
        int pqLength;
        if (fusedPQEnabled) {
            pq = resolvePQFromSources(similarityFunction);
            pqLength = pq.compressedVectorSize();
        } else {
            pq = null;
            pqLength = -1;
        }

        List<CommonHeader.LayerInfo> layerInfo = computeLayerInfoFromSources();
        int entryNode = findEntryNodeFromSources();

        log.info("Writing compacted graph : {} total nodes, maxOrdinal={}, dimension={}, degree={}",
                numTotalNodes, maxOrdinal, dimension, maxDegrees.get(0));
        try (CompactWriter writer = new CompactWriter(outputPath, maxOrdinal, numTotalNodes, 0, layerInfo, entryNode, dimension, maxDegrees, pq, pqLength, fusedPQEnabled)) {
            writer.writeHeader();
            compactLevels(writer, similarityFunction, fusedPQEnabled, compressedPrecision, pq);
            writer.writeFooter();
            log.info("Compaction complete: {}", outputPath);
        } catch (IOException | ExecutionException | InterruptedException e) {
            throw new RuntimeException(e);
        } finally {
            if (ownsExecutor) executor.shutdown();
        }
    }

    private void compactLevels(CompactWriter writer,
                                 VectorSimilarityFunction similarityFunction,
                                 boolean fusedPQEnabled,
                                 boolean compressedPrecision,
                                 ProductQuantization pq)
            throws IOException, ExecutionException, InterruptedException {

        int maxUpperDegree = 0;
        for (int level = 1; level < maxDegrees.size(); level++) {
            maxUpperDegree = Math.max(maxUpperDegree, maxDegrees.get(level));
        }

        int baseSearchTopK = Math.max(2, ((maxDegrees.get(0) + sources.size() - 1) / sources.size()) * 2);
        int baseMaxCandidateSize = baseSearchTopK * (sources.size() - 1) + maxDegrees.get(0);
        int upperMaxPerSourceTopK = maxUpperDegree == 0 ? 0 : Math.max(2, ((maxUpperDegree + sources.size() - 1) / sources.size()) * 2);
        int upperMaxCandidateSize = upperMaxPerSourceTopK * sources.size();
        int maxCandidateSize = Math.max(baseMaxCandidateSize, upperMaxCandidateSize);
        int scratchDegree = Math.max(maxDegrees.get(0), Math.max(1, maxUpperDegree));
        final ThreadLocal<Scratch> threadLocalScratch = ThreadLocal.withInitial(() ->
            new Scratch(maxCandidateSize, scratchDegree, dimension, sources, pq)
        );

        for (int level = 0; level < maxDegrees.size(); level++) {
            List<BatchSpec> batches = buildBatches(level);
            int searchTopK = Math.max(2, ((maxDegrees.get(level) + sources.size() - 1) / sources.size()) * 2);
            int beamWidth = searchTopK * 2;

            if (level == 0) {
                log.info("Compacting level 0 (base layer)");

                ExecutorCompletionService<List<WriteResult>> ecs =
                        new ExecutorCompletionService<>(executor);

                java.util.function.Consumer<BatchSpec> submitOne = (bs) -> {
                    ecs.submit(() -> {
                        Scratch scratch = threadLocalScratch.get();
                        return computeBaseBatch(
                                writer, bs, scratch,
                                similarityFunction,
                                fusedPQEnabled,
                                compressedPrecision,
                                searchTopK,
                                beamWidth,
                                pq
                        );
                    });
                };

                var wropts = EnumSet.of(StandardOpenOption.WRITE, StandardOpenOption.READ);
                try (FileChannel fc = FileChannel.open(writer.getOutputPath(), wropts)) {

                    runBatchesWithBackpressure(
                            batches,
                            ecs,
                            submitOne,
                            (results) -> {
                                try {
                                    for (WriteResult r : results) {
                                        ByteBuffer b = r.data;
                                        long pos = r.fileOffset;
                                        while (b.hasRemaining()) {
                                            int n = fc.write(b, pos);
                                            pos += n;
                                        }
                                    }
                                } catch (IOException e) {
                                    throw new RuntimeException(e);
                                }
                            }
                    );
                }

                writer.offsetAfterInline();

            } else {
                final int lvl = level;
                log.info("Compacting upper layer {}", level);

                ExecutorCompletionService<List<UpperLayerWriteResult>> ecs =
                        new ExecutorCompletionService<>(executor);

                java.util.function.Consumer<BatchSpec> submitOne = (bs) -> {
                    ecs.submit(() -> {
                        Scratch scratch = threadLocalScratch.get();
                        return computeUpperBatchForLevel(
                                bs,
                                lvl,
                                scratch,
                                similarityFunction,
                                fusedPQEnabled,
                                compressedPrecision,
                                searchTopK,
                                beamWidth,
                                pq
                        );
                    });
                };

                runBatchesWithBackpressure(
                        batches,
                        ecs,
                        submitOne,
                        (results) -> {
                            try {
                                for (UpperLayerWriteResult r : results) {
                                    writer.writeUpperLayerNode(
                                            lvl,
                                            r.ordinal,
                                            r.neighbors,
                                            r.pqCode
                                    );
                                }
                            } catch (IOException e) {
                                throw new RuntimeException(e);
                            }
                        }
                );
            }
        }

        Scratch s = threadLocalScratch.get();
        s.close();
        threadLocalScratch.remove();
    }

    private List<BatchSpec> buildBatches(int level) {
        List<BatchSpec> batches = new ArrayList<>();

        for (int s = 0; s < sources.size(); ++s) {
            var source = sources.get(s);
            NodesIterator sourceNodes = source.getNodes(level);
            int numNodes = sourceNodes.size();
            int[] nodes = new int[numNodes];
            int i = 0;
            while (sourceNodes.hasNext()) {
                nodes[i++] = sourceNodes.next();
            }

            int numBatches = max(40, (numNodes + 128 - 1) / 128);
            if (numBatches > numNodes) numBatches = numNodes;
            int batchSize = (numNodes + numBatches - 1) / numBatches;
            for (int b = 0; b < numBatches; ++b) {
                int start = min(numNodes, batchSize * b);
                int end = min(numNodes, batchSize * (b + 1));
                batches.add(new BatchSpec(s, nodes, start, end));
            }
        }

        return batches;
    }

    private List<WriteResult> computeBaseBatch(CompactWriter writer,
                                               BatchSpec bs,
                                               Scratch scratch,
                                               VectorSimilarityFunction similarityFunction,
                                               boolean fusedPQEnabled,
                                               boolean compressedPrecision,
                                               int searchTopK,
                                               int beamWidth,
                                               ProductQuantization pq) throws IOException {
        CompactVamanaDiversityProvider vdp = new CompactVamanaDiversityProvider(similarityFunction, 1.2f);
        List<WriteResult> wrs = new ArrayList<>(bs.end - bs.start);
        OnDiskGraphIndex.View sourceView = (OnDiskGraphIndex.View) scratch.gs[bs.sourceIdx].getView();
        VectorFloat<?> tmpVec = scratch.tmpVec;
        for (int i = bs.start; i < bs.end; ++i) {
            int node = bs.nodes[i];
            if (!liveNodes.get(bs.sourceIdx).get(node)) continue;
            VectorFloat<?> baseVec = scratch.baseVec;
            int[] candSrc = scratch.candSrc;
            int[] candNode = scratch.candNode;
            float[] candScore = scratch.candScore;
            sourceView.getVectorInto(node, baseVec, 0);
            int candSize = 0;

            for (int ss = 0; ss < sources.size(); ++ss) {
                var indexAlive = liveNodes.get(ss);
                var remapper = remappers.get(ss);
                var searchView = (OnDiskGraphIndex.View) scratch.gs[ss].getView();

                if (bs.sourceIdx == ss) {
                    var it = searchView.getNeighborsIterator(0, node);
                    while (it.hasNext()) {
                        int nb = it.nextInt();
                        if (!indexAlive.get(nb)) continue;
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
                            searchView,
                            remapper,
                            baseVec,
                            tmpVec,
                            similarityFunction);
                    SearchResult results = scratch.gs[ss].search(ssp, searchTopK, beamWidth, 0.f, 0.0f, indexAlive);
                    for (SearchResult.NodeScore re : results.getNodes()) {
                        candSrc[candSize] = ss;
                        candNode[candSize] = re.node;
                        if (fusedPQEnabled) {
                            searchView.getVectorInto(re.node, tmpVec, 0);
                            candScore[candSize] = similarityFunction.compare(baseVec, tmpVec);
                        } else {
                            candScore[candSize] = re.score;
                        }
                        candSize++;
                    }
                }
            }

            int[] order = IntStream.range(0, candSize).toArray();
            sortOrderByScoreDesc(order, candScore, candSize);

            SelectedVecCache selectedCache = scratch.selectedCache;
            vdp.retainDiverse(candSrc, candNode, candScore, order, candSize, maxDegrees.get(0), selectedCache, tmpVec, scratch.gs);

            for (int k = 0; k < selectedCache.size; k++) {
                selectedCache.nodes[k] = remappers.get(selectedCache.sourceIdx[k]).oldToNew(selectedCache.nodes[k]);
            }

            int newOrdinal = remappers.get(bs.sourceIdx).oldToNew(node);
            wrs.add(writer.writeInlineNodeRecord(newOrdinal, baseVec, selectedCache, scratch.pqCode));
        }
        return wrs;
    }

    private List<UpperLayerWriteResult> computeUpperBatchForLevel(BatchSpec bs,
                                                                  int level,
                                                                  Scratch scratch,
                                                                  VectorSimilarityFunction similarityFunction,
                                                                  boolean fusedPQEnabled,
                                                                  boolean compressedPrecision,
                                                                  int searchTopK,
                                                                  int beamWidth,
                                                                  ProductQuantization pq) {
        CompactVamanaDiversityProvider vdp = new CompactVamanaDiversityProvider(similarityFunction, 1.2f);
        List<UpperLayerWriteResult> results = new ArrayList<>(bs.end - bs.start);
        var sourceView = (OnDiskGraphIndex.View) scratch.gs[bs.sourceIdx].getView();
        int degree = maxDegrees.get(level);

        for (int i = bs.start; i < bs.end; i++) {
            int node = bs.nodes[i];
            if (!liveNodes.get(bs.sourceIdx).get(node)) continue;
            var baseVec = scratch.baseVec;
            var tmpVec = scratch.tmpVec;
            var candSrc = scratch.candSrc;
            var candNode = scratch.candNode;
            var candScore = scratch.candScore;

            sourceView.getVectorInto(node, baseVec, 0);
            int candSize = 0;

            for (int ss = 0; ss < sources.size(); ss++) {
                var searchView = (OnDiskGraphIndex.View) scratch.gs[ss].getView();
                var indexAlive = liveNodes.get(ss);

                if (bs.sourceIdx == ss) {
                    var it = searchView.getNeighborsIterator(level, node);
                    while (it.hasNext()) {
                        int nb = it.nextInt();
                        if (!indexAlive.get(nb)) continue;
                        searchView.getVectorInto(nb, tmpVec, 0);
                        float score = similarityFunction.compare(baseVec, tmpVec);
                        candSrc[candSize] = ss;
                        candNode[candSize] = nb;
                        candScore[candSize] = score;
                        candSize++;
                    }
                }
                else {

                    SearchScoreProvider ssp = buildCrossSourceScoreProvider(
                            compressedPrecision,
                            searchView,
                            remappers.get(ss),
                            baseVec,
                            tmpVec,
                            similarityFunction
                    );

                    scratch.gs[ss].initializeInternal(
                            ssp,
                            searchView.entryNode(),
                            Bits.ALL
                    );

                    scratch.gs[ss].searchOneLayer(
                            ssp,
                            searchTopK,
                            0.0f,
                            level,
                            indexAlive
                    );

                    candSize = appendApproximateResults(
                            scratch.gs[ss].approximateResults,
                            ss,
                            scratch,
                            candSize
                    );

                    // rescore
                    for(int c = 0; c < candSize; c++) {
                        if (fusedPQEnabled) {
                            searchView.getVectorInto(candNode[c], tmpVec, 0);
                            candScore[c] = similarityFunction.compare(baseVec, tmpVec);
                        }
                    }
                }
            }

            int[] order = IntStream.range(0, candSize).toArray();
            sortOrderByScoreDesc(order, candScore, candSize);

            SelectedVecCache selected = scratch.selectedCache;
            vdp.retainDiverse(
                    candSrc,
                    candNode,
                    candScore,
                    order,
                    candSize,
                    degree,
                    selected,
                    tmpVec,
                    scratch.gs
            );

            for (int k = 0; k < selected.size; k++) {
                selected.nodes[k] = remappers.get(selected.sourceIdx[k]).oldToNew(selected.nodes[k]);
            }

            int newOrdinal = remappers.get(bs.sourceIdx).oldToNew(node);
            ByteSequence<?> pqCode = null;
            if (fusedPQEnabled && level == 1) {
                scratch.pqCode.zero();
                pq.encodeTo(scratch.baseVec, scratch.pqCode);
                pqCode = scratch.pqCode;
            }

            results.add(new UpperLayerWriteResult(newOrdinal, selected, pqCode));
        }
        return results;
    }

    private <T> void runBatchesWithBackpressure(
            List<BatchSpec> batches,
            ExecutorCompletionService<List<T>> ecs,
            java.util.function.Consumer<BatchSpec> submitOne,
            java.util.function.Consumer<List<T>> onComplete
    ) throws InterruptedException, ExecutionException {

        final int total = batches.size();
        int nextToSubmit = 0;
        int inFlight = 0;

        // initial window
        while (inFlight < taskWindowSize && nextToSubmit < total) {
            submitOne.accept(batches.get(nextToSubmit++));
            inFlight++;
        }

        int completed = 0;
        while (completed < total) {
            List<T> results = ecs.take().get();
            onComplete.accept(results);

            completed++;
            inFlight--;

            if (nextToSubmit < total) {
                submitOne.accept(batches.get(nextToSubmit++));
                inFlight++;
            }
            if (completed % 10 == 0) {
                log.info("Compaction I/O progress: {}/{} batches written to disk", completed, total);
            }
        }
    }

    private int appendApproximateResults(NodeQueue queue,
                                         int sourceIdx,
                                         Scratch scratch,
                                         int candSize) {
        final int ss = sourceIdx;
        final int[] idx = new int[] { candSize };

        queue.foreach((nb, score) -> {
            scratch.candSrc[idx[0]] = ss;
            scratch.candNode[idx[0]] = nb;
            scratch.candScore[idx[0]] = score;
            idx[0]++;
        });

        return idx[0];
    }

    private List<CommonHeader.LayerInfo> computeLayerInfoFromSources() {
        int maxLevel = sources.get(0).getMaxLevel();
        List<CommonHeader.LayerInfo> layerInfo = new ArrayList<>(maxLevel + 1);
        for (int level = 0; level <= maxLevel; level++) {
            int count = 0;
            for (int s = 0; s < sources.size(); s++) {
                NodesIterator it = sources.get(s).getNodes(level);
                FixedBitSet alive = liveNodes.get(s);
                while (it.hasNext()) {
                    int node = it.next();
                    if (alive.get(node)) count++;
                }
            }
            layerInfo.add(new CommonHeader.LayerInfo(count, maxDegrees.get(level)));
        }
        return layerInfo;
    }

    private int findEntryNodeFromSources() {
        int maxLevel = sources.get(0).getMaxLevel();
        for (int level = maxLevel; level >= 1; level--) {
            for (int s = 0; s < sources.size(); s++) {
                NodesIterator it = sources.get(s).getNodes(level);
                FixedBitSet alive = liveNodes.get(s);
                while (it.hasNext()) {
                    int node = it.next();
                    if (alive.get(node)) {
                        return remappers.get(s).oldToNew(node);
                    }
                }
            }
        }
        for (int s = 0; s < sources.size(); s++) {
            NodesIterator it = sources.get(s).getNodes(0);
            FixedBitSet alive = liveNodes.get(s);
            while (it.hasNext()) {
                int node = it.next();
                if (alive.get(node)) {
                    return remappers.get(s).oldToNew(node);
                }
            }
        }
        return 0;
    }

    private ProductQuantization resolvePQFromSources(VectorSimilarityFunction similarityFunction) {
        ProductQuantization pq;

        // method #1: get PQ codebook from the largest source
//         int best = 0;
//         int bestLive = -1;
//         for (int i = 0; i < sources.size(); i++) {
//             int c = liveNodes.get(i).cardinality();
//             if (c > bestLive) {
//                 bestLive = c;
//                 best = i;
//             }
//         }
//         FusedPQ fpq = (FusedPQ) sources.get(best).getFeatures().get(FeatureId.FUSED_PQ);
//         pq = fpq.getPQ();

        // method #2: retrain PQ codebook
        //if (upperLayerNodeList.size() > 10000) {
        //    log.info("Retraining PQ codebook using upper layer nodes");
        //    boolean center = similarityFunction == VectorSimilarityFunction.EUCLIDEAN;
        //    pq = ProductQuantization.compute(ulravv, pq.compressedVectorSize(), pq.getClusterCount(), center);
        //}

        // method #3: refining
        log.info("Refining PQ codebook");
        FusedPQ fpq = (FusedPQ) sources.get(0).getFeatures().get(FeatureId.FUSED_PQ);
        pq = fpq.getPQ();
        for (int i = 1; i < sources.size(); i++) {
            pq.refine(sources.get(i).getView());
        }
        return pq;
    }

    private boolean hasFusedPQ() {
        return sources.get(0).getFeatures().containsKey(FeatureId.FUSED_PQ);
    }

    private SearchScoreProvider buildCrossSourceScoreProvider(boolean compressedPrecision,
                                                              OnDiskGraphIndex.View searchView,
                                                              OrdinalMapper remapper,
                                                              VectorFloat<?> baseVec,
                                                              VectorFloat<?> tmpVec,
                                                              VectorSimilarityFunction similarityFunction) {
        if (compressedPrecision) {
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
        for (var entry : liveNodes) {
            size += entry.ramBytesUsed();
        }

        // remappers: each MapMapper holds an oldToNew HashMap and newToOld Int2IntHashMap
        // Estimate based on the number of mappings
        for (var mapper : remappers) {
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
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
                i++;
            }
        }
        int t = order[i];
        order[i] = order[hi];
        order[hi] = t;
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
            this.selectedCache = new SelectedVecCache(maxDegree, dimension);
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

        BatchSpec(int sourceIdx, int[] nodes, int start, int end) {
            this.sourceIdx = sourceIdx;
            this.nodes = nodes;
            this.start = start;
            this.end = end;
        }
    }

    final class CompactVamanaDiversityProvider {
        /**
         * the diversity threshold; 1.0 is equivalent to HNSW; Vamana uses 1.2 or more
         */
        public final float alpha;

        /**
         * used to compute diversity
         */
        public final VectorSimilarityFunction vsf;

        /**
         * Create a new diversity provider
         */
        public CompactVamanaDiversityProvider(VectorSimilarityFunction vsf, float alpha) {
            this.vsf = vsf;
            this.alpha = alpha;
        }

        /**
         * Update `selected` with the diverse members of `neighbors`.  `neighbors` is not modified
         * It assumes that the i-th neighbor with 0 {@literal <=} i {@literal <} diverseBefore is already diverse.
         *
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
    private final Path outputPath;
    private final List<CommonHeader.LayerInfo> configuredLayerInfo;
    private final List<Integer> configuredLayerDegrees;
    private final List<UpperLayerFeatureRecord> level1FeatureRecords;

    CompactWriter(Path outputPath,
                  int maxOrdinal,
                  int numBaseLayerNodes,
                  long startOffset,
                  ImmutableGraphIndex upperLayerGraph,
                  int dimension,
                  ProductQuantization pq,
                  int pqLength,
                  boolean fusedPQEnabled)
            throws IOException {
        this.fusedPQEnabled = fusedPQEnabled;
        this.version = OnDiskGraphIndex.CURRENT_VERSION;
        this.outputPath = outputPath;
        this.writer = new BufferedRandomAccessWriter(outputPath);
        this.startOffset = startOffset;
        this.upperLayerGraph = upperLayerGraph;
        this.baseDegree = upperLayerGraph.getDegree(0);
        this.pq = pq;
        this.maxOrdinal = maxOrdinal;
        this.level1FeatureRecords = new ArrayList<>();

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
        int rsize = Integer.BYTES
                + inlineVectorFeature.featureSize()
                + Integer.BYTES
                + baseDegree * Integer.BYTES;

        if(fusedPQEnabled) {
            rsize += fusedPQFeature.featureSize();
        }
        this.recordSize = rsize;

        this.configuredLayerInfo = IntStream.rangeClosed(0, upperLayerGraph.getMaxLevel())
                .mapToObj(i -> new CommonHeader.LayerInfo(upperLayerGraph.size(i), upperLayerGraph.getDegree(i)))
                .collect(Collectors.toList());
        this.configuredLayerInfo.set(0, new CommonHeader.LayerInfo(numBaseLayerNodes, baseDegree));
        this.configuredLayerDegrees = IntStream.rangeClosed(0, upperLayerGraph.getMaxLevel())
                .mapToObj(upperLayerGraph::getDegree)
                .collect(Collectors.toList());

        var commonHeader = new CommonHeader(this.version,
                dimension,
                upperLayerGraph.getView().entryNode().node,
                this.configuredLayerInfo,
                this.maxOrdinal + 1);
        this.header = new Header(commonHeader, featureMap);
        this.headerSize = header.size();

        this.bufferPerThread = ThreadLocal.withInitial(() -> {
            ByteBuffer buffer = ByteBuffer.allocate(recordSize);
            buffer.order(ByteOrder.BIG_ENDIAN);
            return buffer;
        });
        this.zeroPQ = ThreadLocal.withInitial(() -> {
            var vec = vectorTypeSupport.createByteSequence(pqLength > 0 ? pqLength : 1);
            vec.zero();
            return vec;
        });
    }

    CompactWriter(Path outputPath,
                  int maxOrdinal,
                  int numBaseLayerNodes,
                  long startOffset,
                  List<CommonHeader.LayerInfo> layerInfo,
                  int entryNode,
                  int dimension,
                  List<Integer> layerDegrees,
                  ProductQuantization pq,
                  int pqLength,
                  boolean fusedPQEnabled)
            throws IOException {
        this.fusedPQEnabled = fusedPQEnabled;
        this.version = OnDiskGraphIndex.CURRENT_VERSION;
        this.outputPath = outputPath;
        this.writer = new BufferedRandomAccessWriter(outputPath);
        this.startOffset = startOffset;
        this.upperLayerGraph = null;
        this.configuredLayerInfo = new ArrayList<>(layerInfo);
        this.configuredLayerDegrees = new ArrayList<>(layerDegrees);
        this.baseDegree = layerDegrees.get(0);
        this.pq = pq;
        this.maxOrdinal = maxOrdinal;
        this.level1FeatureRecords = new ArrayList<>();

        Map<FeatureId, Feature> featureMap = new LinkedHashMap<>();
        InlineVectors inlineVectorFeature = new InlineVectors(dimension);
        featureMap.put(FeatureId.INLINE_VECTORS, inlineVectorFeature);
        if (fusedPQEnabled) {
            this.fusedPQFeature = new FusedPQ(Collections.max(layerDegrees), pq);
            featureMap.put(FeatureId.FUSED_PQ, this.fusedPQFeature);
        } else {
            this.fusedPQFeature = null;
        }

        int rsize = Integer.BYTES + inlineVectorFeature.featureSize() + Integer.BYTES + baseDegree * Integer.BYTES;
        if (fusedPQEnabled) {
            rsize += fusedPQFeature.featureSize();
        }
        this.recordSize = rsize;

        this.configuredLayerInfo.set(0, new CommonHeader.LayerInfo(numBaseLayerNodes, baseDegree));
        var commonHeader = new CommonHeader(this.version, dimension, entryNode, this.configuredLayerInfo, this.maxOrdinal + 1);
        this.header = new Header(commonHeader, featureMap);
        this.headerSize = header.size();

        this.bufferPerThread = ThreadLocal.withInitial(() -> {
            ByteBuffer buffer = ByteBuffer.allocate(recordSize);
            buffer.order(ByteOrder.BIG_ENDIAN);
            return buffer;
        });
        this.zeroPQ = ThreadLocal.withInitial(() -> {
            var vec = vectorTypeSupport.createByteSequence(pqLength > 0 ? pqLength : 1);
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
        if (fusedPQEnabled && version == 6 && !level1FeatureRecords.isEmpty()) {
            for (UpperLayerFeatureRecord record : level1FeatureRecords) {
                writer.writeInt(record.ordinal);
                vectorTypeSupport.writeByteSequence(writer, record.pqCode);
            }
        }
        long headerOffset = writer.position();
        header.write(writer);
        writer.writeLong(headerOffset);
        writer.writeInt(FOOTER_MAGIC);
        final long expectedPosition = headerOffset + headerSize + FOOTER_SIZE;
        assert writer.position() == expectedPosition : String.format("%d != %d", writer.position(), expectedPosition);
    }

    public void offsetAfterInline() throws IOException {
        long offset = startOffset + headerSize + (long) (maxOrdinal + 1) * recordSize;
        writer.seek(offset);
    }

    public Path getOutputPath() {
        return outputPath;
    }

    public void writeUpperLayerNode(int level, int ordinal, int[] neighbors, ByteSequence<?> level1PqCode) throws IOException {
        writer.writeInt(ordinal);
        writer.writeInt(neighbors.length);
        int degree = configuredLayerDegrees.get(level);
        int n = 0;
        for (; n < neighbors.length; n++) {
            writer.writeInt(neighbors[n]);
        }
        for (; n < degree; n++) {
            writer.writeInt(-1);
        }
        if (fusedPQEnabled && version == 6 && level == 1 && level1PqCode != null) {
            level1FeatureRecords.add(new UpperLayerFeatureRecord(ordinal, level1PqCode.copy()));
        }
    }

    public void close() throws IOException {
        final var endOfGraphPosition = writer.position();
        writer.seek(endOfGraphPosition);
        writer.flush();
        if (upperLayerGraph != null) {
            var view = upperLayerGraph.getView();
            view.close();
        }
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

        if (bwriter.bytesWritten() != recordSize) {
            throw new IllegalStateException(
                    String.format("Record size mismatch for ordinal %d: expected %d bytes, wrote %d bytes, base degree: %d",
                            ordinal, recordSize, bwriter.bytesWritten(), baseDegree));
        }

        ByteBuffer dataCopy = bwriter.cloneBuffer();

        ByteSequence pqCopy = null;

        // Note: writePQSeparate is for future work that supports writing PQ separately
        //if (writePQSeparate) {
        //    // encode into scratch ByteSequence to avoid allocations inside pq.encode()
        //    pqCode.zero();
        //    pq.encodeTo(vec, pqCode);
        //
        //    pqCopy = pqCode.copy();
        // }

        return new WriteResult(ordinal, fileOffset, dataCopy, pqCopy);
    }
}

final class UpperLayerFeatureRecord {
    final int ordinal;
    final ByteSequence<?> pqCode;

    UpperLayerFeatureRecord(int ordinal, ByteSequence<?> pqCode) {
        this.ordinal = ordinal;
        this.pqCode = pqCode;
    }
}

final class SelectedVecCache {
    int[] sourceIdx;
    OnDiskGraphIndex.View[] views;
    int[] nodes;
    float[] scores;
    VectorFloat<?>[] vecs;
    int size;
    private static final VectorTypeSupport vts =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    SelectedVecCache(int capacity, int dimension) {
        sourceIdx = new int[capacity];
        views = new OnDiskGraphIndex.View[capacity];
        nodes = new int[capacity];
        scores = new float[capacity];
        vecs = new VectorFloat<?>[capacity];
        for(int c = 0; c < capacity; ++c) {
            vecs[c] = vts.createFloatVector(dimension);
        }
        size = 0;
    }

    void reset() {
        Arrays.fill(views, 0, size, null);
        for (VectorFloat<?> vec : vecs) {
            vec.zero();
        }
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