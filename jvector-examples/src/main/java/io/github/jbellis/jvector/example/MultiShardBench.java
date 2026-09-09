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
import io.github.jbellis.jvector.example.benchmarks.datasets.SimpleDataSet;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.DataSetPartitioner;
import io.github.jbellis.jvector.example.yaml.TestDataPartition.Distribution;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.MultiGraphSearcher;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.ShardedSearchResult;
import io.github.jbellis.jvector.graph.disk.GraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.GraphIndexWriterTypes;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.function.IntFunction;

/**
 * Compares {@link MultiGraphSearcher} against a traditional, unsharded single-index search over the
 * same data, to detect any recall loss and measure the performance impact of searching across
 * multiple independent shards versus one merged index.
 * <p>
 * {@link #runAll} mirrors {@link Grid#runAll}'s parameter shape exactly -- the same construction grid
 * (M/outDegree, efConstruction, neighborOverflow, addHierarchy, refineFinalGraph, feature sets,
 * build-time compression) and the same search grid (search-time compression, pruning, topK/overquery)
 * -- with one additional axis, {@code shardCounts}, layered on top. This means a caller (see
 * {@link BenchYAML}) can drive this benchmark from the very same {@code MultiConfig}-derived lists it
 * already builds for the single-index {@link Grid} run, so every shard is built and searched using
 * the <b>same configuration</b> the legacy single-index benchmark actually uses for this dataset. That
 * match matters: an earlier version of this benchmark always built full-precision shards and searched
 * them with exact scoring regardless of the dataset's real config, which is a fundamentally different
 * (much more expensive, much higher-recall) search mode -- making "numShards=1" incomparable to the
 * legacy baseline it was meant to sanity-check against. Search-time compressors, when configured, are
 * trained independently per shard (mirroring how a real sharded system like Cassandra or OpenSearch
 * trains per-segment codebooks rather than one global codebook, since segments are built independently
 * over time).
 * <p>
 * For each configuration, {@link DataSet#getBaseVectors()} is split into {@code numShards} independent
 * on-disk graphs, every query is searched across all of them via {@link MultiGraphSearcher}, and
 * shard-local ordinals are translated back to dataset-global ordinals so the existing, unmodified
 * {@link AccuracyMetrics#recallFromSearchResults} can be reused directly against
 * {@link DataSet#getGroundTruth()}. A shard count of 1 is included deliberately: since it goes through
 * the exact same build/search code as every other configuration, it isolates {@link MultiGraphSearcher}'s
 * own overhead from the effect of sharding itself, and serves as the apples-to-apples single-index
 * baseline for comparison.
 * <p>
 * Two things {@link Grid} supports are deliberately out of scope here: index caching
 * ({@code ConstructionParameters.useSavedIndexIfExists}) -- every configuration builds its shards fresh
 * -- and {@code usePruningGrid}'s value has no effect, since {@link MultiGraphSearcher} doesn't expose
 * a way to configure per-shard pruning and the underlying heuristic is permanently disabled anyway
 * (see {@code GraphSearcher#usePruning}); it's still accepted and recorded per result so a caller
 * driving both benchmarks from the same config doesn't need special-case handling.
 */
public final class MultiShardBench {
    private static final Logger logger = LoggerFactory.getLogger(MultiShardBench.class);

    /** Shard counts swept when the caller doesn't specify its own; 1 is the traditional single-index baseline. */
    static final List<Integer> DEFAULT_SHARD_COUNTS = Arrays.asList(1, 4, 16);
    static final Distribution DISTRIBUTION = Distribution.UNIFORM;

    private MultiShardBench() {}

    /**
     * Full grid sweep: for every combination of the construction parameters, feature set, and
     * build-time compressor, builds {@code numShards} shards once (for each {@code numShards} in
     * {@code shardCounts}) and reuses them across every combination of search-time compressor,
     * pruning flag, and topK/overquery -- exactly mirroring how {@link Grid#runOneGraph} builds an
     * index once and reuses it across {@link Grid}'s own search-side grid. A configuration that fails
     * is logged and skipped so the rest of the sweep still runs.
     *
     * @param mGrid                 construction.outDegree
     * @param efConstructionGrid    construction.efConstruction
     * @param neighborOverflowGrid  construction.neighborOverflow
     * @param addHierarchyGrid      construction.addHierarchy
     * @param refineFinalGraphGrid  construction.refineFinalGraph
     * @param featureSets           construction.getFeatureSets()
     * @param buildCompressors      construction.getCompressorParameters()
     * @param compressionGrid       search.getCompressorParameters()
     * @param topKGrid              search.topKOverquery (topK -> list of overquery factors)
     * @param usePruningGrid        search.useSearchPruning (accepted and recorded; see class javadoc)
     * @param shardCounts           shard counts to sweep; 1 is the single-index baseline
     * @return one result per (construction config x feature set x build compressor x shard count x
     * search compressor x pruning x topK x overquery) leaf
     */
    public static List<BenchResult> runAll(DataSet ds,
                                            List<Integer> mGrid,
                                            List<Integer> efConstructionGrid,
                                            List<Float> neighborOverflowGrid,
                                            List<Boolean> addHierarchyGrid,
                                            List<Boolean> refineFinalGraphGrid,
                                            List<? extends Set<FeatureId>> featureSets,
                                            List<Function<DataSet, CompressorParameters>> buildCompressors,
                                            List<Function<DataSet, CompressorParameters>> compressionGrid,
                                            Map<Integer, List<Double>> topKGrid,
                                            List<Boolean> usePruningGrid,
                                            List<Integer> shardCounts) throws Exception
    {
        var queryVectors = ds.getQueryVectors();
        var groundTruth = ds.getGroundTruth();
        if (queryVectors == null || queryVectors.isEmpty()) {
            throw new IllegalArgumentException("Dataset " + ds.getName() + " has no query vectors");
        }
        if (groundTruth == null || groundTruth.isEmpty()) {
            throw new IllegalArgumentException("Dataset " + ds.getName() + " has no ground truth");
        }

        List<BenchResult> results = new ArrayList<>();
        for (boolean addHierarchy : addHierarchyGrid) {
            for (boolean refineFinalGraph : refineFinalGraphGrid) {
                for (int M : mGrid) {
                    for (float neighborOverflow : neighborOverflowGrid) {
                        for (int efConstruction : efConstructionGrid) {
                            for (var buildCompressorFn : buildCompressors) {
                                for (Set<FeatureId> features : featureSets) {
                                    for (int numShards : shardCounts) {
                                        try {
                                            results.addAll(runOneConstructionConfig(ds, M, efConstruction, neighborOverflow,
                                                    addHierarchy, refineFinalGraph, features, buildCompressorFn,
                                                    compressionGrid, topKGrid, usePruningGrid, numShards));
                                        } catch (Exception e) {
                                            logger.error("Multi-shard config failed for dataset {} [M={} efConstruction={} " +
                                                    "neighborOverflow={} addHierarchy={} refineFinalGraph={} features={} numShards={}]",
                                                    ds.getName(), M, efConstruction, neighborOverflow, addHierarchy,
                                                    refineFinalGraph, features, numShards, e);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return results;
    }

    /** Convenience overload sweeping {@link #DEFAULT_SHARD_COUNTS} ({1, 4, 16}). */
    public static List<BenchResult> runAll(DataSet ds,
                                            List<Integer> mGrid,
                                            List<Integer> efConstructionGrid,
                                            List<Float> neighborOverflowGrid,
                                            List<Boolean> addHierarchyGrid,
                                            List<Boolean> refineFinalGraphGrid,
                                            List<? extends Set<FeatureId>> featureSets,
                                            List<Function<DataSet, CompressorParameters>> buildCompressors,
                                            List<Function<DataSet, CompressorParameters>> compressionGrid,
                                            Map<Integer, List<Double>> topKGrid,
                                            List<Boolean> usePruningGrid) throws Exception
    {
        return runAll(ds, mGrid, efConstructionGrid, neighborOverflowGrid, addHierarchyGrid, refineFinalGraphGrid,
                featureSets, buildCompressors, compressionGrid, topKGrid, usePruningGrid, DEFAULT_SHARD_COUNTS);
    }

    /**
     * Builds {@code numShards} shards once for this construction configuration, then reuses them
     * across every (search compressor x pruning x topK x overquery) combination.
     */
    private static List<BenchResult> runOneConstructionConfig(DataSet ds, int M, int efConstruction, float neighborOverflow,
                                                                boolean addHierarchy, boolean refineFinalGraph,
                                                                Set<FeatureId> features,
                                                                Function<DataSet, CompressorParameters> buildCompressorFn,
                                                                List<Function<DataSet, CompressorParameters>> compressionGrid,
                                                                Map<Integer, List<Double>> topKGrid,
                                                                List<Boolean> usePruningGrid,
                                                                int numShards) throws Exception
    {
        String datasetName = ds.getName();
        VectorSimilarityFunction vsf = ds.getSimilarityFunction();
        CompressorParameters buildCompressor = buildCompressorFn == null ? CompressorParameters.NONE : buildCompressorFn.apply(ds);
        String buildCompressorLabel = buildCompressor.getClass().getSimpleName();

        logger.info("Multi-shard bench [{}] numShards={} features={} M={} efConstruction={} neighborOverflow={} " +
                        "addHierarchy={} refineFinalGraph={} buildCompression={}: {} vectors",
                datasetName, numShards, features, M, efConstruction, neighborOverflow, addHierarchy, refineFinalGraph,
                buildCompressorLabel, ds.getBaseVectors().size());

        var partitioned = DataSetPartitioner.partition(ds, numShards, DISTRIBUTION);
        List<ShardHandle> shardHandles = new ArrayList<>(numShards);
        try {
            // Track each shard's offset into the dataset's global ordinal space so shard-local
            // ordinals can be translated back for recall computation against ds.getGroundTruth().
            int[] shardOffsets = new int[numShards];
            int runningOffset = 0;
            for (int s = 0; s < numShards; s++) {
                shardOffsets[s] = runningOffset;
                runningOffset += partitioned.sizes.get(s);
                String shardName = datasetName + "-shard" + s + "-of-" + numShards;
                shardHandles.add(buildAndWriteShard(shardName, partitioned.vectors.get(s), vsf, features, buildCompressor,
                        M, efConstruction, neighborOverflow, addHierarchy, refineFinalGraph));
            }

            List<BenchResult> results = new ArrayList<>();
            for (var searchCompressorFn : compressionGrid) {
                CompressorParameters searchCompressor = searchCompressorFn == null ? CompressorParameters.NONE : searchCompressorFn.apply(ds);
                CompressedVectors[] searchCvs = trainSearchCompressors(shardHandles, features, searchCompressor);
                String searchCompressorLabel = searchCompressor.getClass().getSimpleName();

                for (boolean usePruning : usePruningGrid) {
                    for (int topK : topKGrid.keySet()) {
                        for (double overquery : topKGrid.get(topK)) {
                            int rerankK = (int) (topK * overquery);
                            results.add(searchAndReport(ds, shardHandles, shardOffsets, searchCvs, features, vsf, numShards,
                                    M, efConstruction, neighborOverflow, addHierarchy, refineFinalGraph, buildCompressorLabel,
                                    searchCompressorLabel, usePruning, topK, overquery, rerankK));
                        }
                    }
                }
            }
            return results;
        } finally {
            for (ShardHandle handle : shardHandles) {
                try { handle.close(); } catch (Exception ignored) {}
            }
        }
    }

    /**
     * Trains one {@link CompressedVectors} per shard for {@code searchCompressor}, on that shard's own
     * vectors (not the whole dataset) -- matching how a real sharded system trains codebooks
     * independently per segment. Returns an all-null array (meaning "score exactly") if
     * {@code searchCompressor} is {@link CompressorParameters#NONE} or {@code features} contains
     * {@link FeatureId#FUSED_PQ} (fused shards score from the graph's own view; a separate
     * {@link CompressedVectors} would never be consulted).
     */
    private static CompressedVectors[] trainSearchCompressors(List<ShardHandle> shardHandles, Set<FeatureId> features,
                                                                CompressorParameters searchCompressor) {
        CompressedVectors[] cvs = new CompressedVectors[shardHandles.size()];
        if (features.contains(FeatureId.FUSED_PQ)) {
            return cvs;
        }
        for (int i = 0; i < shardHandles.size(); i++) {
            ShardHandle h = shardHandles.get(i);
            VectorCompressor<?> compressorObj = searchCompressor.computeCompressor(h.shardDs);
            cvs[i] = compressorObj == null ? null : compressorObj.encodeAll(h.shardDs.getBaseRavv());
        }
        return cvs;
    }

    private static BenchResult searchAndReport(DataSet ds, List<ShardHandle> shardHandles, int[] shardOffsets,
                                                CompressedVectors[] searchCvs, Set<FeatureId> features, VectorSimilarityFunction vsf,
                                                int numShards, int M, int efConstruction, float neighborOverflow,
                                                boolean addHierarchy, boolean refineFinalGraph, String buildCompressorLabel,
                                                String searchCompressorLabel, boolean usePruning, int topK,
                                                double overquery, int rerankK) throws IOException
    {
        SearchStats stats = search(ds, shardHandles, shardOffsets, searchCvs, features, vsf, topK, rerankK);

        logger.info(String.format(
                "%n" +
                "  ┌─ Multi-shard result: %s [numShards=%d, features=%s, M=%d, efC=%d, overflow=%.2f, " +
                "addHierarchy=%s, refine=%s, buildCompression=%s, searchCompression=%s, usePruning=%s, topK=%d, overquery=%.2f]%n" +
                "  │  recall@%-2d       : %.4f%n" +
                "  │  mean latency    : %.3f ms%n" +
                "  │  p99 latency     : %.3f ms%n" +
                "  │  throughput      : %,.1f qps%n" +
                "  │  avg rounds used : %.2f%n" +
                "  │  avg visited     : %.1f%n" +
                "  └─",
                ds.getName(), numShards, features, M, efConstruction, neighborOverflow, addHierarchy, refineFinalGraph,
                buildCompressorLabel, searchCompressorLabel, usePruning, topK, overquery,
                topK, stats.recall, stats.meanLatencyMs, stats.p99LatencyMs, stats.qps, stats.avgRoundsUsed, stats.avgVisitedCount));

        Map<String, Object> params = new LinkedHashMap<>();
        params.put("numShards", numShards);
        params.put("distribution", DISTRIBUTION.name());
        params.put("features", features.toString());
        params.put("M", M);
        params.put("efConstruction", efConstruction);
        params.put("neighborOverflow", neighborOverflow);
        params.put("addHierarchy", addHierarchy);
        params.put("refineFinalGraph", refineFinalGraph);
        params.put("buildCompression", buildCompressorLabel);
        params.put("searchCompression", searchCompressorLabel);
        params.put("usePruning", usePruning);
        params.put("topK", topK);
        params.put("overquery", overquery);
        params.put("rerankK", rerankK);

        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("recall@" + topK, stats.recall);
        metrics.put("meanLatencyMs", stats.meanLatencyMs);
        metrics.put("p99LatencyMs", stats.p99LatencyMs);
        metrics.put("qps", stats.qps);
        metrics.put("avgRoundsUsed", stats.avgRoundsUsed);
        metrics.put("avgVisitedCount", stats.avgVisitedCount);
        metrics.put("numVectors", ds.getBaseVectors().size());

        return new BenchResult(ds.getName() + " (multishard numShards=" + numShards + ")", params, metrics);
    }

    /** Recall, search-latency, and internal adaptive-resume stats from searching across all shards. */
    static final class SearchStats {
        final double recall;
        final double meanLatencyMs;
        final double p99LatencyMs;
        final double qps;
        final double avgRoundsUsed;
        final double avgVisitedCount;

        SearchStats(double recall, double meanLatencyMs, double p99LatencyMs, double qps,
                    double avgRoundsUsed, double avgVisitedCount)
        {
            this.recall = recall;
            this.meanLatencyMs = meanLatencyMs;
            this.p99LatencyMs = p99LatencyMs;
            this.qps = qps;
            this.avgRoundsUsed = avgRoundsUsed;
            this.avgVisitedCount = avgVisitedCount;
        }
    }

    /**
     * Searches every query across all shards via {@link MultiGraphSearcher}, timing each call and
     * translating shard-local ordinals back to dataset-global ordinals so recall can be computed with
     * the existing {@link AccuracyMetrics} machinery unchanged.
     */
    private static SearchStats search(DataSet ds, List<ShardHandle> shardHandles, int[] shardOffsets,
                                       CompressedVectors[] searchCvs, Set<FeatureId> features, VectorSimilarityFunction vsf,
                                       int topK, int rerankK) throws IOException
    {
        var queryVectors = ds.getQueryVectors();
        var groundTruth = ds.getGroundTruth();
        List<ImmutableGraphIndex> shards = new ArrayList<>(shardHandles.size());
        for (ShardHandle h : shardHandles) {
            shards.add(h.graph);
        }

        try (MultiGraphSearcher searcher = MultiGraphSearcher.builder(shards).build()) {
            int n = queryVectors.size();
            List<SearchResult> results = new ArrayList<>(n);
            long[] latenciesNanos = new long[n];
            long totalNanos = 0;
            long totalRounds = 0;
            long totalVisited = 0;

            for (int i = 0; i < n; i++) {
                VectorFloat<?> query = queryVectors.get(i);
                List<SearchScoreProvider> providers = new ArrayList<>(shardHandles.size());
                for (int s = 0; s < shardHandles.size(); s++) {
                    providers.add(scoreProviderFor(query, shardHandles.get(s), searcher.getView(s), searchCvs[s], features, vsf));
                }

                long t0 = System.nanoTime();
                ShardedSearchResult sr = searcher.search(providers, topK, rerankK);
                long elapsed = System.nanoTime() - t0;
                latenciesNanos[i] = elapsed;
                totalNanos += elapsed;
                totalRounds += sr.getRoundsUsed();
                totalVisited += sr.getVisitedCount();

                results.add(toGlobalSearchResult(sr, shardOffsets));
            }

            double recall = AccuracyMetrics.recallFromSearchResults(groundTruth, results, topK, topK);
            double meanLatencyMs = (totalNanos / (double) n) / 1_000_000.0;
            double qps = totalNanos > 0 ? n / (totalNanos / 1_000_000_000.0) : 0.0;

            Arrays.sort(latenciesNanos);
            int p99Index = (int) Math.ceil(0.99 * n) - 1;
            p99Index = Math.min(Math.max(p99Index, 0), n - 1);
            double p99LatencyMs = latenciesNanos[p99Index] / 1_000_000.0;

            return new SearchStats(recall, meanLatencyMs, p99LatencyMs, qps,
                    totalRounds / (double) n, totalVisited / (double) n);
        }
    }

    /**
     * Builds the query-time {@link SearchScoreProvider} for one shard, mirroring
     * {@code Grid.ConfiguredSystem#scoreProviderFor} exactly: {@code FUSED_PQ} shards score
     * approximately straight from the graph's own {@link ImmutableGraphIndex.ScoringView}; otherwise
     * a per-shard {@link CompressedVectors} (if one was trained) provides the approximate score and
     * the view's exact reranker finishes the job; with no compression at all, scoring is exact
     * throughout, same as the legacy path's uncompressed configuration.
     * <p>
     * {@code searcherView} must be the view {@link MultiGraphSearcher}'s own internal searcher for
     * this shard traverses with (obtained via {@link MultiGraphSearcher#getView}), not
     * {@code shard.view} (independently opened when the shard was built). For {@code FUSED_PQ}, whose
     * approximate scoring reads neighbor codes through the view's own reader position as the search
     * walks neighbors, building the score provider from a different view than the one actually driving
     * the traversal silently reads garbage -- confirmed by a targeted repro: recall collapses from
     * ~0.65 to ~0.03, matching exactly what this method used to produce before it took the searcher's
     * view instead of the shard's own.
     */
    private static SearchScoreProvider scoreProviderFor(VectorFloat<?> query, ShardHandle shard,
                                                          ImmutableGraphIndex.View searcherView, CompressedVectors searchCv,
                                                          Set<FeatureId> features, VectorSimilarityFunction vsf) {
        var scoringView = (ImmutableGraphIndex.ScoringView) searcherView;
        ScoreFunction.ApproximateScoreFunction asf;
        if (features.contains(FeatureId.FUSED_PQ)) {
            asf = scoringView.approximateScoreFunctionFor(query, vsf);
        } else if (searchCv == null) {
            return DefaultSearchScoreProvider.exact(query, vsf, shard.shardDs.getBaseRavv());
        } else {
            asf = searchCv.precomputedScoreFunctionFor(query, vsf);
        }
        var rr = scoringView.rerankerFor(query, vsf);
        return new DefaultSearchScoreProvider(asf, rr);
    }

    /**
     * Translates a {@link ShardedSearchResult} (shard-local ordinals) into a {@link SearchResult}
     * (dataset-global ordinals), the type {@link AccuracyMetrics} expects. Only the node identities
     * matter for recall, so the other {@link SearchResult} statistics are left at zero.
     */
    private static SearchResult toGlobalSearchResult(ShardedSearchResult sr, int[] shardOffsets) {
        var shardNodes = sr.getNodes();
        var globalNodes = new SearchResult.NodeScore[shardNodes.length];
        for (int i = 0; i < shardNodes.length; i++) {
            var n = shardNodes[i];
            globalNodes[i] = new SearchResult.NodeScore(shardOffsets[n.shardIndex] + n.node, n.score);
        }
        return new SearchResult(globalNodes, 0, 0, 0, 0, Float.POSITIVE_INFINITY);
    }

    /**
     * Builds one shard's graph in memory from its slice of the dataset's base vectors, using the given
     * construction parameters/feature set/build-time compressor (normally whatever the legacy
     * single-index path uses for this dataset), then writes it to a temporary on-disk file and
     * re-opens it as a read-only {@link OnDiskGraphIndex}. Search-time compression is trained
     * separately (see {@link #trainSearchCompressors}) so the same built shards can be reused across
     * every search-compressor grid entry without rebuilding.
     */
    private static ShardHandle buildAndWriteShard(String shardName, List<VectorFloat<?>> vectors, VectorSimilarityFunction vsf,
                                                    Set<FeatureId> features, CompressorParameters buildCompressor,
                                                    int M, int efConstruction, float neighborOverflow,
                                                    boolean addHierarchy, boolean refineFinalGraph) throws IOException
    {
        int dimension = vectors.get(0).length();
        // Dummy 1-element query/ground-truth: this DataSet exists only to drive CompressorParameters'
        // computeCompressor(DataSet), which reads getBaseRavv()/getDimension()/getSimilarityFunction()
        // and never touches queries or ground truth.
        DataSet shardDs = new SimpleDataSet(shardName, vsf, vectors, List.of(vectors.get(0)), List.of(List.of(0)));
        RandomAccessVectorValues ravv = shardDs.getBaseRavv();

        VectorCompressor<?> buildCompressorObj = buildCompressor.computeCompressor(shardDs);
        BuildScoreProvider bsp;
        PQVectors buildPq = null;
        if (buildCompressorObj != null) {
            buildPq = (PQVectors) buildCompressorObj.encodeAll(ravv);
            bsp = BuildScoreProvider.pqBuildScoreProvider(vsf, buildPq);
        } else {
            if (features.contains(FeatureId.FUSED_PQ)) {
                throw new IllegalArgumentException(
                        "FUSED_PQ requires a build-time PQ compressor, but buildCompressor resolved to no compression");
            }
            bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, vsf);
        }

        ImmutableGraphIndex heapGraph;
        try (GraphIndexBuilder graphBuilder = new GraphIndexBuilder(
                bsp, dimension, M, efConstruction, neighborOverflow, 1.2f, addHierarchy, refineFinalGraph)) {
            heapGraph = graphBuilder.build(ravv);
        }

        Path path = Files.createTempFile("multishard-bench-shard", null);
        // RANDOM_ACCESS (sequential), not RANDOM_ACCESS_PARALLEL: matches how Grid itself writes shards
        // (via RandomAccessOnDiskGraphIndexWriter.Builder with no parallel worker threads configured).
        // The parallel writer invokes feature-state suppliers concurrently across worker threads (see
        // ParallelGraphWriter.writeL0Records), which is fine for suppliers that only read from a shared
        // RandomAccessVectorValues, but FUSED_PQ's supplier needs an ImmutableGraphIndex.View for
        // per-node neighbor lookups, and View is explicitly documented as not thread-safe.
        var writerBuilder = GraphIndexWriter.getBuilderFor(GraphIndexWriterTypes.RANDOM_ACCESS, heapGraph, path);
        Map<FeatureId, IntFunction<Feature.State>> suppliers = new EnumMap<>(FeatureId.class);
        for (FeatureId featureId : features) {
            switch (featureId) {
                case INLINE_VECTORS:
                    writerBuilder.with(new InlineVectors(dimension));
                    suppliers.put(FeatureId.INLINE_VECTORS, ordinal -> new InlineVectors.State(ravv.getVector(ordinal)));
                    break;
                case NVQ_VECTORS:
                    int nSubVectors = dimension == 2 ? 1 : 2;
                    var nvq = NVQuantization.compute(ravv, nSubVectors);
                    writerBuilder.with(new NVQ(nvq));
                    suppliers.put(FeatureId.NVQ_VECTORS, ordinal -> new NVQ.State(nvq.encode(ravv.getVector(ordinal))));
                    break;
                case FUSED_PQ:
                    // buildPq is non-null here: checked above (features.contains(FUSED_PQ) => buildCompressorObj != null).
                    final PQVectors fusedPq = buildPq;
                    final ImmutableGraphIndex finalHeapGraph = heapGraph;
                    // getDegree(0), not maxDegree(): FusedPQ's per-node record is sized around the
                    // level-0 adjacency list specifically (state.view.getNeighborsIterator(0, ...) is
                    // hardcoded to level 0), and with addHierarchy=true, maxDegree() (the max across
                    // any layer) can differ from level 0's degree -- see the reference usage at
                    // TestOnDiskGraphIndex.java:504 (new FusedPQ(graph.getDegree(0), pq)).
                    writerBuilder.with(new FusedPQ(finalHeapGraph.getDegree(0), fusedPq.getCompressor()));
                    suppliers.put(FeatureId.FUSED_PQ, ordinal -> new FusedPQ.State(finalHeapGraph.getView(), fusedPq, ordinal));
                    break;
                default:
                    throw new IllegalArgumentException("Unsupported feature for shard construction: " + featureId);
            }
        }
        try (GraphIndexWriter writer = writerBuilder.build()) {
            writer.write(suppliers);
        }

        ReaderSupplier readerSupplier = ReaderSupplierFactory.open(path);
        OnDiskGraphIndex onDiskGraph = OnDiskGraphIndex.load(readerSupplier);
        var view = onDiskGraph.getView();

        return new ShardHandle(onDiskGraph, view, shardDs, readerSupplier, path);
    }

    /** One shard's on-disk graph plus everything needed to search and close it cleanly. */
    private static final class ShardHandle implements Closeable {
        final OnDiskGraphIndex graph;
        /**
         * The shard's own in-memory dataset, retained so search-time compressors can be (re)trained
         * per search-compressor grid entry, and so exact scoring always has full-resolution vectors to
         * read -- {@link #shardDs}'s {@code getBaseRavv()}, never the on-disk graph's own view. A
         * graph built with only {@code NVQ_VECTORS} (no {@code INLINE_VECTORS}) stores no
         * full-resolution vectors at all, so casting its view to a {@code RandomAccessVectorValues}
         * and calling {@code getVector} throws {@code UnsupportedOperationException("No
         * full-resolution vectors in this graph")} -- exactly {@link Grid}'s own reason for always
         * using {@code ds.getBaseRavv()}, never the graph's view, for both compressor training and the
         * exact-scoring fallback (see {@code Grid.ConfiguredSystem#scoreProviderFor}).
         */
        final DataSet shardDs;
        private final ImmutableGraphIndex.View view;
        private final ReaderSupplier readerSupplier;
        private final Path path;

        ShardHandle(OnDiskGraphIndex graph, ImmutableGraphIndex.View view, DataSet shardDs,
                    ReaderSupplier readerSupplier, Path path)
        {
            this.graph = graph;
            this.view = view;
            this.shardDs = shardDs;
            this.readerSupplier = readerSupplier;
            this.path = path;
        }

        @Override
        public void close() throws IOException {
            view.close();
            readerSupplier.close();
            Files.deleteIfExists(path);
        }
    }
}
