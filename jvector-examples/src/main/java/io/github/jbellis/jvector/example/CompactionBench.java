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

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.example.util.DataSetPartitioner;
import io.github.jbellis.jvector.example.yaml.TestDataPartition.Distribution;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.IntFunction;

/**
 * Runs a partition-and-compact regression pass against a single dataset and returns a BenchResult.
 *
 * The pipeline mirrors the PARTITION_AND_COMPACT workload in CompactorBenchmark:
 *   1. Split base vectors into NUM_PARTITIONS equal segments.
 *   2. Build one on-disk graph per segment (full-precision, no PQ).
 *   3. Compact all segments into one graph via OnDiskGraphIndexCompactor.
 *   4. Search the compacted graph and compute recall@TOP_K against the dataset ground truth.
 *
 * All temp files are written to a system temp directory and deleted on completion.
 */
public final class CompactionBench {
    private static final Logger logger = LoggerFactory.getLogger(CompactionBench.class);

    static final int NUM_PARTITIONS = 4;
    static final int GRAPH_DEGREE = 32;
    static final int BEAM_WIDTH = 100;
    static final int TOP_K = 10;
    static final Distribution DISTRIBUTION = Distribution.UNIFORM;

    private CompactionBench() {}

    /**
     * Runs the full compaction regression for {@code ds} and returns the result.
     * Throws if the dataset has no query vectors or ground truth.
     */
    public static BenchResult run(DataSet ds) throws Exception {
        var queryVectors = ds.getQueryVectors();
        var groundTruth = ds.getGroundTruth();
        if (queryVectors == null || queryVectors.isEmpty()) {
            throw new IllegalArgumentException("Dataset " + ds.getName() + " has no query vectors");
        }
        if (groundTruth == null || groundTruth.isEmpty()) {
            throw new IllegalArgumentException("Dataset " + ds.getName() + " has no ground truth");
        }

        Path tempDir = Files.createTempDirectory("autobench-compact");
        try {
            return runInDir(ds, tempDir);
        } finally {
            deleteRecursively(tempDir);
        }
    }

    private static BenchResult runInDir(DataSet ds, Path tempDir) throws Exception {
        List<VectorFloat<?>> baseVectors = ds.getBaseVectors();
        int dimension = ds.getDimension();
        VectorSimilarityFunction vsf = ds.getSimilarityFunction();
        String datasetName = ds.getName();

        logger.info("Compaction bench [{}]: {} vectors, {} partitions, degree={}, bw={}",
                datasetName, baseVectors.size(), NUM_PARTITIONS, GRAPH_DEGREE, BEAM_WIDTH);

        // 1. Partition
        var partitioned = DataSetPartitioner.partition(baseVectors, NUM_PARTITIONS, DISTRIBUTION);

        // 2. Build one on-disk graph per partition
        List<Path> partitionPaths = new ArrayList<>(NUM_PARTITIONS);
        for (int i = 0; i < NUM_PARTITIONS; i++) {
            Path partPath = tempDir.resolve("partition-" + i);
            buildPartition(partitioned.vectors.get(i), dimension, vsf, partPath);
            partitionPaths.add(partPath);
            logger.info("Built partition {}/{}", i + 1, NUM_PARTITIONS);
        }

        // 3. Load graphs and set up ordinal mapping: partition i's local ordinals shift by
        //    the sum of all prior partition sizes, preserving the original base-vector ordering.
        List<ReaderSupplier> rss = new ArrayList<>(NUM_PARTITIONS);
        List<OnDiskGraphIndex> graphs = new ArrayList<>(NUM_PARTITIONS);
        List<OrdinalMapper> remappers = new ArrayList<>(NUM_PARTITIONS);
        List<FixedBitSet> liveNodes = new ArrayList<>(NUM_PARTITIONS);
        int globalOffset = 0;
        for (int i = 0; i < NUM_PARTITIONS; i++) {
            ReaderSupplier rs = ReaderSupplierFactory.open(partitionPaths.get(i));
            rss.add(rs);
            OnDiskGraphIndex g = OnDiskGraphIndex.load(rs);
            graphs.add(g);
            int size = g.size(0);
            remappers.add(new OrdinalMapper.OffsetMapper(globalOffset, size));
            FixedBitSet live = new FixedBitSet(size);
            live.set(0, size);  // all nodes live
            liveNodes.add(live);
            globalOffset += size;
        }

        // 4. Compact
        Path compactPath = tempDir.resolve("compacted");
        var compactor = new OnDiskGraphIndexCompactor(graphs, liveNodes, remappers, vsf, null);
        long t0 = System.currentTimeMillis();
        compactor.compact(compactPath);
        long compactionMs = System.currentTimeMillis() - t0;
        logger.info("Compaction finished in {} ms", compactionMs);

        for (var g : graphs) g.close();
        for (var rs : rss) rs.close();

        // 5. Measure recall — global ordinal k maps back to baseVectors.get(k) by construction
        double recall = measureRecall(compactPath, ds, baseVectors, dimension, vsf);
        logger.info("Compaction recall@{} for {}: {}", TOP_K, datasetName, recall);

        Map<String, Object> params = new LinkedHashMap<>();
        params.put("numPartitions", NUM_PARTITIONS);
        params.put("distribution", DISTRIBUTION.name());
        params.put("graphDegree", GRAPH_DEGREE);
        params.put("beamWidth", BEAM_WIDTH);

        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("compactionTimeMs", compactionMs);
        metrics.put("recall@" + TOP_K, recall);
        metrics.put("numVectors", baseVectors.size());

        return new BenchResult(datasetName + " (compaction)", params, metrics);
    }

    private static void buildPartition(List<VectorFloat<?>> vectors, int dimension,
                                       VectorSimilarityFunction vsf, Path outputPath) throws IOException {
        var ravv = new ListRandomAccessVectorValues(vectors, dimension);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, vsf);
        var builder = new GraphIndexBuilder(bsp, dimension, GRAPH_DEGREE, BEAM_WIDTH, 1.2f, 1.2f, true);
        var graph = builder.build(ravv);

        var writerBuilder = new OnDiskGraphIndexWriter.Builder(graph, outputPath);
        writerBuilder.with(new InlineVectors(dimension));
        try (var writer = writerBuilder.build()) {
            Map<FeatureId, IntFunction<Feature.State>> suppliers = new EnumMap<>(FeatureId.class);
            suppliers.put(FeatureId.INLINE_VECTORS, ord -> new InlineVectors.State(ravv.getVector(ord)));
            writer.write(suppliers);
        }
    }

    private static double measureRecall(Path indexPath, DataSet ds,
                                        List<VectorFloat<?>> baseVectors,
                                        int dimension, VectorSimilarityFunction vsf) throws Exception {
        var queryVectors = ds.getQueryVectors();
        var groundTruth = ds.getGroundTruth();
        var ravv = new ListRandomAccessVectorValues(baseVectors, dimension);

        try (var rs = ReaderSupplierFactory.open(indexPath)) {
            var graph = OnDiskGraphIndex.load(rs);
            try (var searcher = new GraphSearcher(graph)) {
                searcher.usePruning(false);
                List<SearchResult> results = new ArrayList<>(queryVectors.size());
                for (var q : queryVectors) {
                    var ssp = DefaultSearchScoreProvider.exact(q, vsf, ravv);
                    results.add(searcher.search(ssp, TOP_K, TOP_K, 0f, 0f, Bits.ALL));
                }
                return AccuracyMetrics.recallFromSearchResults(groundTruth, results, TOP_K, TOP_K);
            }
        }
    }

    private static void deleteRecursively(Path dir) {
        try {
            if (!Files.exists(dir)) return;
            Files.walk(dir)
                 .sorted(Comparator.reverseOrder())
                 .forEach(p -> { try { Files.delete(p); } catch (IOException ignored) {} });
        } catch (IOException ignored) {}
    }
}
