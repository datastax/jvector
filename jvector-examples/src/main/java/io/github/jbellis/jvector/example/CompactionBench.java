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
import io.github.jbellis.jvector.example.util.CompactionPartitionSource;
import io.github.jbellis.jvector.example.yaml.TestDataPartition.Distribution;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
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
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Runs a partition-and-compact regression pass against a single dataset and returns one
 * {@link BenchResult} per partition configuration.
 *
 * The pipeline reuses pre-built partitions stored in S3 rather than rebuilding them each run:
 *   1. For each configured layout, download its per-source partition graphs from
 *      {@link CompactionPartitionSource} (cached locally across runs).
 *   2. Compact all partitions into one graph via {@link OnDiskGraphIndexCompactor}.
 *   3. Search the compacted graph and compute recall@TOP_K against the dataset ground truth.
 *
 * Partitions are laid out in S3 as {@code <datasetName>/<numPartitions>-<distribution>-<precision>/per-source-graph-<i>}.
 * The compacted graph is written to a system temp directory and deleted on completion.
 */
public final class CompactionBench {
    private static final Logger logger = LoggerFactory.getLogger(CompactionBench.class);

    static final int TOP_K = 10;

    /** Pre-built partition configurations to compact for every dataset. */
    static final List<PartitionConfig> CONFIGS = Arrays.asList(
            new PartitionConfig(2, Distribution.TIERED_10_90, "FUSEDPQ"),
            new PartitionConfig(2, Distribution.UNIFORM, "FUSEDPQ"),
            new PartitionConfig(4, Distribution.UNIFORM, "FUSEDPQ")
    );

    /** A single pre-built partition layout, identified by its S3 directory name. */
    static final class PartitionConfig {
        final int numPartitions;
        final Distribution distribution;
        final String precision;

        PartitionConfig(int numPartitions, Distribution distribution, String precision) {
            this.numPartitions = numPartitions;
            this.distribution = distribution;
            this.precision = precision;
        }

        /** The S3 directory name for this config, e.g. {@code 4-UNIFORM-FUSEDPQ}. */
        String dirName() {
            return numPartitions + "-" + distribution.name() + "-" + precision;
        }
    }

    private CompactionBench() {}

    /**
     * Runs the compaction regression for {@code ds} across every {@link #CONFIGS} layout and returns
     * one result per config. A config that fails (e.g. missing partitions) is logged and skipped so
     * the remaining configs still run. Throws if the dataset has no query vectors or ground truth.
     */
    public static List<BenchResult> run(DataSet ds) throws Exception {
        var queryVectors = ds.getQueryVectors();
        var groundTruth = ds.getGroundTruth();
        if (queryVectors == null || queryVectors.isEmpty()) {
            throw new IllegalArgumentException("Dataset " + ds.getName() + " has no query vectors");
        }
        if (groundTruth == null || groundTruth.isEmpty()) {
            throw new IllegalArgumentException("Dataset " + ds.getName() + " has no ground truth");
        }

        List<BenchResult> results = new ArrayList<>(CONFIGS.size());
        for (PartitionConfig cfg : CONFIGS) {
            try {
                results.add(runConfig(ds, cfg));
            } catch (Exception e) {
                logger.error("Compaction config {} failed for dataset {}", cfg.dirName(), ds.getName(), e);
            }
        }
        return results;
    }

    private static BenchResult runConfig(DataSet ds, PartitionConfig cfg) throws Exception {
        String datasetName = ds.getName();
        logger.info("Compaction bench [{}] config {}: {} vectors",
                datasetName, cfg.dirName(), ds.getBaseVectors().size());

        // 1. Fetch pre-built partitions from S3 (cached locally).
        List<Path> partitionPaths = CompactionPartitionSource.ensurePartitions(
                datasetName, cfg.dirName(), cfg.numPartitions);

        Path tempDir = Files.createTempDirectory("autobench-compact");
        try {
            return compactAndMeasure(ds, cfg, partitionPaths, tempDir);
        } finally {
            deleteRecursively(tempDir);
        }
    }

    private static BenchResult compactAndMeasure(DataSet ds, PartitionConfig cfg,
                                                 List<Path> partitionPaths, Path tempDir) throws Exception {
        List<VectorFloat<?>> baseVectors = ds.getBaseVectors();
        int dimension = ds.getDimension();
        VectorSimilarityFunction vsf = ds.getSimilarityFunction();
        String datasetName = ds.getName();
        int numPartitions = cfg.numPartitions;

        // Load graphs and set up ordinal mapping: partition i's local ordinals shift by the sum of
        // all prior partition sizes, preserving the original base-vector ordering so global ordinal
        // k maps back to baseVectors.get(k) by construction.
        List<ReaderSupplier> rss = new ArrayList<>(numPartitions);
        List<OnDiskGraphIndex> graphs = new ArrayList<>(numPartitions);
        List<OrdinalMapper> remappers = new ArrayList<>(numPartitions);
        List<FixedBitSet> liveNodes = new ArrayList<>(numPartitions);
        try {
            int globalOffset = 0;
            for (int i = 0; i < numPartitions; i++) {
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

            // Recover graphDegree and precision from the actual partition graphs.
            int graphDegree = graphs.get(0).maxDegree();
            String precision = graphs.get(0).getFeatures().containsKey(FeatureId.FUSED_PQ)
                    ? "FUSEDPQ" : "FULLPRECISION";

            // Compact
            Path compactPath = tempDir.resolve("compacted");
            var compactor = new OnDiskGraphIndexCompactor(graphs, liveNodes, remappers, vsf, null);
            // A/B switch for PQ-bucketed cross-source candidate acquisition (fused-PQ only):
            // -Djvector.compaction.bucketed=true
            if (Boolean.getBoolean("jvector.compaction.bucketed")) {
                logger.info("Bucketed candidate acquisition ENABLED");
                compactor.setBucketedCandidateAcquisition(true);
            }
            long t0 = System.currentTimeMillis();
            compactor.compact(compactPath);
            long compactionMs = System.currentTimeMillis() - t0;
            logger.info("Compaction [{} {}] finished in {} ms", datasetName, cfg.dirName(), compactionMs);

            for (var g : graphs) g.close();
            for (var rs : rss) rs.close();
            graphs.clear();
            rss.clear();

            // Search the compacted graph: measure recall and search latency in one pass.
            // Global ordinal k maps back to baseVectors.get(k) by construction.
            SearchStats search = searchCompacted(compactPath, ds, baseVectors, dimension, vsf);
            logger.info(String.format(
                    "%n" +
                    "  ┌─ Compaction result: %s [%s]%n" +
                    "  │  compaction time : %,d ms%n" +
                    "  │  recall@%-2d       : %.4f%n" +
                    "  │  mean latency    : %.3f ms%n" +
                    "  │  p99 latency     : %.3f ms%n" +
                    "  │  throughput      : %,.1f qps%n" +
                    "  └─",
                    datasetName, cfg.dirName(), compactionMs,
                    TOP_K, search.recall, search.meanLatencyMs, search.p99LatencyMs, search.qps));

            // numPartitions and distribution come from the config iteration (the S3 dir name);
            // graphDegree and precision are read back from the partition graphs.
            Map<String, Object> params = new LinkedHashMap<>();
            params.put("numPartitions", numPartitions);
            params.put("distribution", cfg.distribution.name());
            params.put("graphDegree", graphDegree);
            params.put("precision", precision);

            Map<String, Object> metrics = new LinkedHashMap<>();
            metrics.put("compactionTimeMs", compactionMs);
            metrics.put("recall@" + TOP_K, search.recall);
            metrics.put("meanLatencyMs", search.meanLatencyMs);
            metrics.put("qps", search.qps);
            metrics.put("numVectors", baseVectors.size());

            return new BenchResult(datasetName + " (compaction " + cfg.dirName() + ")", params, metrics);
        } finally {
            for (var g : graphs) {
                try { g.close(); } catch (Exception ignored) {}
            }
            for (var rs : rss) {
                try { rs.close(); } catch (Exception ignored) {}
            }
        }
    }

    /** Recall and search-latency stats from searching the compacted graph. */
    static final class SearchStats {
        final double recall;
        final double meanLatencyMs;
        final double p99LatencyMs;
        final double qps;

        SearchStats(double recall, double meanLatencyMs, double p99LatencyMs, double qps) {
            this.recall = recall;
            this.meanLatencyMs = meanLatencyMs;
            this.p99LatencyMs = p99LatencyMs;
            this.qps = qps;
        }
    }

    /**
     * Searches every query against the compacted graph, timing each search, and returns recall plus
     * mean and p99 per-query latency (ms) and throughput (queries/sec, single-threaded sequential).
     */
    private static SearchStats searchCompacted(Path indexPath, DataSet ds,
                                               List<VectorFloat<?>> baseVectors,
                                               int dimension, VectorSimilarityFunction vsf) throws Exception {
        var queryVectors = ds.getQueryVectors();
        var groundTruth = ds.getGroundTruth();
        var ravv = new ListRandomAccessVectorValues(baseVectors, dimension);

        try (var rs = ReaderSupplierFactory.open(indexPath)) {
            var graph = OnDiskGraphIndex.load(rs);
            try (var searcher = new GraphSearcher(graph)) {
                searcher.usePruning(false);
                int n = queryVectors.size();
                List<SearchResult> results = new ArrayList<>(n);
                long[] latenciesNanos = new long[n];
                long totalNanos = 0;
                for (int i = 0; i < n; i++) {
                    var ssp = DefaultSearchScoreProvider.exact(queryVectors.get(i), vsf, ravv);
                    long t0 = System.nanoTime();
                    SearchResult result = searcher.search(ssp, TOP_K, TOP_K, 0f, 0f, Bits.ALL);
                    long elapsed = System.nanoTime() - t0;
                    latenciesNanos[i] = elapsed;
                    totalNanos += elapsed;
                    results.add(result);
                }
                double recall = AccuracyMetrics.recallFromSearchResults(groundTruth, results, TOP_K, TOP_K);
                double meanLatencyMs = (totalNanos / (double) n) / 1_000_000.0;
                double qps = totalNanos > 0 ? n / (totalNanos / 1_000_000_000.0) : 0.0;

                Arrays.sort(latenciesNanos);
                // nearest-rank p99: index of the smallest value with at least 99% of samples at or below it
                int p99Index = (int) Math.ceil(0.99 * n) - 1;
                p99Index = Math.min(Math.max(p99Index, 0), n - 1);
                double p99LatencyMs = latenciesNanos[p99Index] / 1_000_000.0;

                return new SearchStats(recall, meanLatencyMs, p99LatencyMs, qps);
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
