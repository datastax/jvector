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

package io.github.jbellis.jvector.example;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.IntFunction;

import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

/**
 * A/B harness for PQ-bucketed vs search-based cross-source candidate acquisition during
 * compaction, on the bundled siftsmall dataset (10k vectors, dim 128, Euclidean).
 * <p>
 * Round-robin partitions the base vectors, builds a fused-PQ on-disk graph per partition, then
 * compacts the partitions twice — once per acquisition mode — reporting wall-clock time and
 * recall@10 against the dataset ground truth for each.
 * <p>
 * Run from the repo root: {@code mvn -pl jvector-examples compile exec:java
 * -Dexec.mainClass=io.github.jbellis.jvector.example.BucketedCompactionAB}
 */
public class BucketedCompactionAB {
    private static final int NUM_PARTITIONS = 4;
    private static final int GRAPH_DEGREE = 32;
    private static final int BEAM_WIDTH = 100;
    private static final int PQ_SUBSPACES = 16;
    private static final int TOP_K = 10;
    private static final VectorSimilarityFunction VSF = VectorSimilarityFunction.EUCLIDEAN;

    public static void main(String[] args) throws Exception {
        var base = SiftLoader.readFvecs("siftsmall/siftsmall_base.fvecs");
        var queries = SiftLoader.readFvecs("siftsmall/siftsmall_query.fvecs");
        var gt = SiftLoader.readIvecs("siftsmall/siftsmall_groundtruth.ivecs");
        int dimension = base.get(0).length();
        System.out.printf("siftsmall: %d base vectors, %d queries, dim %d%n",
                base.size(), queries.size(), dimension);

        Path dir = Files.createTempDirectory("bucketed_ab");
        // Round-robin split: partition p holds original ids p, p+P, p+2P, ...
        List<List<VectorFloat<?>>> partitions = new ArrayList<>();
        for (int p = 0; p < NUM_PARTITIONS; p++) {
            partitions.add(new ArrayList<>());
        }
        for (int i = 0; i < base.size(); i++) {
            partitions.get(i % NUM_PARTITIONS).add(base.get(i));
        }

        System.out.println("Building fused-PQ partition graphs...");
        long tBuild = System.currentTimeMillis();
        for (int p = 0; p < NUM_PARTITIONS; p++) {
            buildPartitionGraph(partitions.get(p), dimension, dir.resolve("partition_" + p));
        }
        System.out.printf("Built %d partition graphs in %,d ms%n",
                NUM_PARTITIONS, System.currentTimeMillis() - tBuild);

        // globalOrdinal -> original base id, per the OffsetMapper layout used below.
        int[] globalToOriginal = new int[base.size()];
        int globalOffset = 0;
        for (int p = 0; p < NUM_PARTITIONS; p++) {
            for (int i = 0; i < partitions.get(p).size(); i++) {
                globalToOriginal[globalOffset + i] = i * NUM_PARTITIONS + p;
            }
            globalOffset += partitions.get(p).size();
        }

        // Mode per JVM keeps JIT warmup from favoring whichever arm runs second; the warmup
        // pass (discarded, always search-based) levels the playing field within a JVM.
        String mode = args.length > 0 ? args[0] : "both";
        if (!mode.equals("both")) {
            System.out.println("Warmup compaction (discarded)...");
            runOne(false, dir, base, queries, gt, globalToOriginal);
            System.out.println("Measured run:");
        }
        if (mode.equals("search") || mode.equals("both")) {
            runOne(false, dir, base, queries, gt, globalToOriginal);
        }
        if (mode.equals("bucketed") || mode.equals("both")) {
            runOne(true, dir, base, queries, gt, globalToOriginal);
        }
    }

    private static void runOne(boolean bucketed,
                               Path dir,
                               List<VectorFloat<?>> base,
                               List<VectorFloat<?>> queries,
                               List<List<Integer>> gt,
                               int[] globalToOriginal) throws Exception {
        List<OnDiskGraphIndex> graphs = new ArrayList<>();
        List<ReaderSupplier> rss = new ArrayList<>();
        List<FixedBitSet> liveNodes = new ArrayList<>();
        List<OrdinalMapper> remappers = new ArrayList<>();
        int globalOffset = 0;
        for (int p = 0; p < NUM_PARTITIONS; p++) {
            ReaderSupplier rs = ReaderSupplierFactory.open(dir.resolve("partition_" + p));
            rss.add(rs);
            OnDiskGraphIndex g = OnDiskGraphIndex.load(rs);
            graphs.add(g);
            int size = g.size(0);
            remappers.add(new OrdinalMapper.OffsetMapper(globalOffset, size));
            var live = new FixedBitSet(size);
            live.set(0, size);
            liveNodes.add(live);
            globalOffset += size;
        }

        Path out = dir.resolve(bucketed ? "compacted_bucketed" : "compacted_search");
        var compactor = new OnDiskGraphIndexCompactor(graphs, liveNodes, remappers, VSF, null);
        compactor.setBucketedCandidateAcquisition(bucketed);

        long t0 = System.currentTimeMillis();
        compactor.compact(out);
        long elapsed = System.currentTimeMillis() - t0;

        for (var g : graphs) g.close();
        for (var rs : rss) rs.close();

        double recall = computeRecall(out, base, queries, gt, globalToOriginal);
        System.out.printf("%n=== %s acquisition: compaction %,d ms, recall@%d %.4f ===%n%n",
                bucketed ? "BUCKETED" : "SEARCH", elapsed, TOP_K, recall);
    }

    private static void buildPartitionGraph(List<VectorFloat<?>> vecs, int dimension, Path outputPath)
            throws IOException {
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vecs, dimension);
        ProductQuantization pq = ProductQuantization.compute(ravv, PQ_SUBSPACES, 256, true, UNWEIGHTED,
                PhysicalCoreExecutor.pool(), PhysicalCoreExecutor.pool());
        PQVectors pqv = (PQVectors) pq.encodeAll(ravv, PhysicalCoreExecutor.pool());
        var bsp = BuildScoreProvider.pqBuildScoreProvider(VSF, pqv);
        var builder = new GraphIndexBuilder(bsp, dimension, GRAPH_DEGREE, BEAM_WIDTH, 1.2f, 1.2f,
                false, true, PhysicalCoreExecutor.pool(), PhysicalCoreExecutor.pool());
        var graph = builder.getGraph();

        Map<FeatureId, IntFunction<Feature.State>> writeSuppliers = new EnumMap<>(FeatureId.class);
        writeSuppliers.put(FeatureId.INLINE_VECTORS, ordinal -> new InlineVectors.State(ravv.getVector(ordinal)));

        var writerBuilder = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
                .withMapper(new OrdinalMapper.IdentityMapper(ravv.size() - 1))
                .with(new InlineVectors(dimension))
                .with(new FusedPQ(graph.maxDegree(), pq));
        try (var writer = writerBuilder.build()) {
            for (int node = 0; node < ravv.size(); node++) {
                var stateMap = new EnumMap<FeatureId, Feature.State>(FeatureId.class);
                stateMap.put(FeatureId.INLINE_VECTORS, writeSuppliers.get(FeatureId.INLINE_VECTORS).apply(node));
                writer.writeInline(node, stateMap);
                builder.addGraphNode(node, ravv.getVector(node));
            }
            builder.cleanup();
            writeSuppliers.put(FeatureId.FUSED_PQ, ordinal -> new FusedPQ.State(graph.getView(), pqv, ordinal));
            writer.write(writeSuppliers);
        }
    }

    private static double computeRecall(Path graphPath,
                                        List<VectorFloat<?>> base,
                                        List<VectorFloat<?>> queries,
                                        List<List<Integer>> gt,
                                        int[] globalToOriginal) throws Exception {
        // RAVV over base vectors permuted into compacted-ordinal order, for exact scoring.
        List<VectorFloat<?>> permuted = new ArrayList<>(base.size());
        for (int ord = 0; ord < base.size(); ord++) {
            permuted.add(base.get(globalToOriginal[ord]));
        }
        var ravv = new ListRandomAccessVectorValues(permuted, base.get(0).length());

        try (ReaderSupplier rs = ReaderSupplierFactory.open(graphPath)) {
            var graph = OnDiskGraphIndex.load(rs);
            long hits = 0;
            try (var searcher = new GraphSearcher(graph)) {
                for (int q = 0; q < queries.size(); q++) {
                    var ssp = DefaultSearchScoreProvider.exact(queries.get(q), VSF, ravv);
                    SearchResult sr = searcher.search(ssp, TOP_K, Bits.ALL);
                    Set<Integer> expected = new HashSet<>(gt.get(q).subList(0, TOP_K));
                    for (var ns : sr.getNodes()) {
                        if (expected.contains(globalToOriginal[ns.node])) {
                            hits++;
                        }
                    }
                }
            }
            return (double) hits / ((long) queries.size() * TOP_K);
        }
    }
}
