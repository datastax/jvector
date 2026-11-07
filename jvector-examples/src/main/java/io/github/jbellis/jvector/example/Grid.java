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

import io.github.jbellis.jvector.disk.BufferedRandomAccessWriter;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.benchmarks.AccuracyBenchmark;
import io.github.jbellis.jvector.example.benchmarks.CountBenchmark;
import io.github.jbellis.jvector.example.benchmarks.LatencyBenchmark;
import io.github.jbellis.jvector.example.benchmarks.QueryBenchmark;
import io.github.jbellis.jvector.example.benchmarks.QueryTester;
import io.github.jbellis.jvector.example.benchmarks.ThroughputBenchmark;
import io.github.jbellis.jvector.example.benchmarks.diagnostics.BenchmarkDiagnostics;
import io.github.jbellis.jvector.example.benchmarks.diagnostics.DiagnosticLevel;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.reporting.RunArtifacts;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.OnDiskGraphIndexCache;
import io.github.jbellis.jvector.management.GraphIndexBuilderConfig;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.*;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.ASHVectors;
import io.github.jbellis.jvector.graph.disk.feature.FusedASH;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Tests a grid of configurations against a dataset
 */
public class Grid {

    private static final String pqCacheDir = "pq_cache";

    private static final String indexCacheDir = "index_cache";

    private static final String dirPrefix = "BenchGraphDir";

    private enum Phase { INDEX, SEARCH }

    private static final Map<String,Double> indexBuildTimes = new HashMap<>();

    private static int diagnostic_level;

    /**
     * Get the index build time for a dataset
     *
     * @deprecated Returns the most recent build for the dataset, which is not the current build config's
     * when its indexes came from the index cache. Grid no longer reads it.
     */
    @Deprecated
    public static Double getIndexBuildTimeSeconds(String datasetName) {
        return indexBuildTimes.get(datasetName);
    }

    /**
     * Runs {@code spec} against {@code ds}, printing console tables and writing experiments.csv rows as
     * selected by {@code artifacts}.
     */
    public static void run(DataSet ds, GridSpec spec, RunArtifacts artifacts) throws IOException {
        runGrid(ds, spec, () -> setupBenchmarks(artifacts.benchmarksToCompute()), reportingListeners(artifacts), artifacts);
    }

    /**
     * Runs {@code spec} against {@code ds} with the fixed regression benchmarks, printing console tables
     * and returning one result per measurement.
     */
    public static List<BenchResult> runAndCollectResults(DataSet ds, GridSpec spec) throws IOException {
        var collector = new BenchResultCollector();
        runGrid(ds, spec, Grid::regressionBenchmarks,
                List.of(new ConsoleReportListener(RunArtifacts.disabled()), collector), RunArtifacts.disabled());
        return collector.results();
    }

    /** Console tables and experiments.csv rows, as selected by run-config.yml. */
    private static List<GridListener> reportingListeners(RunArtifacts artifacts) {
        return List.of(new ConsoleReportListener(artifacts), new ExperimentsLogListener(artifacts));
    }

    /**
     * Shared sweep used by every entry point.
     *
     * @param benchmarks creates the query benchmarks for each index and search compressor
     * @param listeners receive every measurement, in order
     */
    private static void runGrid(DataSet ds,
                                GridSpec spec,
                                Supplier<List<QueryBenchmark>> benchmarks,
                                List<GridListener> listeners, RunArtifacts artifacts) throws IOException
    {
        // Resolve every point first, so an incomplete sweep fails before any build.
        final List<BuildConfig> buildPoints = spec.buildPoints();
        boolean success = false;

        // Always use a fresh temp directory for per-run artifacts
        final Path workDir = Files.createTempDirectory(dirPrefix);

        // Initialize index cache (creates stable directory when enabled, cleans up stale temp files; no-op otherwise)
        final OnDiskGraphIndexCache cache =
                spec.enableIndexCache
                        ? OnDiskGraphIndexCache.initialize(Paths.get(indexCacheDir))
                        : OnDiskGraphIndexCache.disabled();

        try {
            for (var build : buildPoints) {
                runOneGraph(cache, spec.featureSets, build, spec.search, ds, workDir, benchmarks, listeners, artifacts);
            }
            success = true; // Helps with Exception attribution
        } finally {
            try {
                // Final attempt to wipe workDir and its children
                wipeDirectory(workDir);
            } catch (IOException e) {
                if (success) {
                    // The run was fine, but cleanup failed
                    throw new IOException("Run succeeded, but cleanup failed: " + workDir, e);
                } else {
                    // If crashing, don't mask another Exception.
                    System.err.println("Warning: Cleanup failed during crash: " + e.getMessage());
                }
            }

            if (spec.enableIndexCache) {
                System.out.println("Preserving index cache directory: " + indexCacheDir);
            }
            cachedCompressors.clear();
        }
    }

    /**
     * Iterates through the path and deletes everything inside, then the path itself.
     */
    private static void wipeDirectory(Path path) throws IOException {
        if (path == null || !Files.exists(path)) return;

        final IOException[] lastError = { null };

        try (var stream = Files.walk(path)) {
            stream.sorted(Comparator.reverseOrder())
                    .forEach(p -> {
                        try {
                            Files.deleteIfExists(p);
                        } catch (IOException e) {
                            lastError[0] = e; // Capture the last reason for failure
                        }
                    });
        }
        // Final Check: If the directory still exists, the wipe failed.
        if (Files.exists(path)) {
            throw new IOException("Failed to completely delete index directory: " + path, lastError[0]);
        }
    }

    static void runOneGraph(OnDiskGraphIndexCache cache,
                            List<? extends Set<FeatureId>> featureSets,
                            int M,
                            int efConstruction,
                            float neighborOverflow,
                            boolean addHierarchy,
                            boolean refineFinalGraph,
                            Function<DataSet, CompressorParameters> buildCompressor,
                            List<Function<DataSet, CompressorParameters>> compressionGrid,
                            Map<Integer, List<Double>> topKGrid,
                            List<Boolean> usePruningGrid,
                            RunArtifacts artifacts,
                            DataSet ds,
                            Path workDirectory) throws IOException
    {
        var build = BuildConfig.builder()
                .M(M)
                .efConstruction(efConstruction)
                .neighborOverflow(neighborOverflow)
                .addHierarchy(addHierarchy)
                .refineFinalGraph(refineFinalGraph)
                .buildCompressor(buildCompressor)
                .build();
        var search = SearchConfig.builder()
                .searchCompressors(compressionGrid)
                .topKGrid(topKGrid)
                .usePruningGrid(usePruningGrid)
                .build();
        runOneGraph(cache, featureSets, build, search, ds, workDirectory,
                () -> setupBenchmarks(artifacts.benchmarksToCompute()), reportingListeners(artifacts), artifacts);
    }

    private static void runOneGraph(OnDiskGraphIndexCache cache,
                                    List<? extends Set<FeatureId>> featureSets,
                                    BuildConfig build,
                                    SearchConfig search,
                                    DataSet ds,
                                    Path workDirectory,
                                    Supplier<List<QueryBenchmark>> benchmarks,
                                    List<GridListener> listeners, RunArtifacts artifacts) throws IOException
    {
        // Prepare to collect index construction metrics for reporting....
        var constructionMetrics = new ConstructionMetrics();

        // Capture initial memory and disk state
        try (var diagnostics = new BenchmarkDiagnostics(getDiagnosticLevel())) {
            diagnostics.startMonitoring("testDirectory", workDirectory);
            diagnostics.startMonitoring("indexCache", Paths.get(indexCacheDir));
            diagnostics.capturePrePhaseSnapshot("Graph Build");
            System.out.printf("%s: Dataset similarity function is %s%n", ds.getName(), ds.getSimilarityFunction());

            // Resolve build compressor (and label quant type) so we can record compute time
            VectorCompressor<?> buildCompressorObj = null;
            String buildQuantType = null;

            if (build.buildCompressor != null) {
                var buildParams = build.buildCompressor.apply(ds);
                buildQuantType = quantTypeOf(buildParams); // "PQ", "BQ", or null
                buildCompressorObj = getCompressor(build.buildCompressor, ds, constructionMetrics, Phase.INDEX, buildQuantType);
            }

            // Used in logging
            String buildCompressorString =
                    (buildCompressorObj == null) ? "None" : String.valueOf(buildCompressorObj);

            IndexBuildStrategy strategy = (buildCompressorObj == null)
                    ? new InMemoryBuild(workDirectory)
                    : new CompressedOnDiskBuild(cache, workDirectory, buildCompressorObj, constructionMetrics);
            var built = strategy.build(featureSets, build, ds);
            // Null when every index came from the index cache. Read it from here, not the per-dataset map,
            // which still holds the previous build config's time in that case.
            Double indexBuildTimeS = built.buildTimeS;
            if (indexBuildTimeS != null) {
                indexBuildTimes.put(ds.getName(), indexBuildTimeS);
            }
            var indexes = inConfiguredOrder(built.indexes, featureSets);
            var indexFileSizes = built.fileSizes;

            // Capture post-build memory and disk state
            diagnostics.capturePostPhaseSnapshot("Graph Build");
            var buildSnapshot = diagnostics.getLatestSystemSnapshot();

            diagnostics.printDiskStatistics("Graph Index Build");
            if (indexBuildTimeS != null) {
                System.out.printf("Index build time: %f seconds%n%n", indexBuildTimeS);
            } else {
                System.out.printf("Index build time: not measured, all indexes loaded from the index cache%n%n");
            }
            constructionMetrics.indexBuildTimeS = indexBuildTimeS;

            try {
                for (var cpSupplier : search.searchCompressors) {
                    indexes.forEach((features, index) -> {
                        constructionMetrics.indexFileSizeBytes = indexFileSizes.get(features);
                        final Set<FeatureId> featureSetForIndex = index instanceof OnDiskGraphIndex ? ((OnDiskGraphIndex) index).getFeatureSet() : Set.of();

                        // Per (index, cpSupplier) config, so an index without search compression never reports
                        // the previous index's search timings.
                        constructionMetrics.resetSearch();

                        CompressedVectors cv;
                        if (FeatureProviders.anyMatch(featureSetForIndex, FeatureProvider::providesApproximateScoring)) {
                            cv = null;
                            System.out.format("%s: configured to use FUSED PQ, skipping vector compression%n", ds.getName());
                        } else {
                            var searchParams = cpSupplier.apply(ds);
                            String searchQuantType = quantTypeOf(searchParams); // "PQ", "BQ", or null
                            var compressor = getCompressor(cpSupplier, ds, constructionMetrics, Phase.SEARCH, searchQuantType);

                            if (compressor == null) {
                                cv = null;
                                System.out.format("%s: No search compressor configured, FULL PRECISION vectors will be used for search%n", ds.getName());
                            } else {
                                long start = System.nanoTime();
                                cv = constructionMetrics.search(searchQuantType)
                                        .timeEncode(() -> compressor.encodeAll(ds.getBaseRavv()));
                                double encodingTimeS = (System.nanoTime() - start) / 1_000_000_000.0;
                                if (cv == null) {
                                    throw new IllegalStateException(String.format(
                                            "Compressor '%s' was provided but failed to encode vectors for dataset '%s'. " +
                                                    "Aborting to prevent false recall results.", compressor, ds.getName()));
                                }
                                System.out.format("%s: %s encoded %d vectors [%.2f MB] in %.2fs%n", ds.getName(), compressor, ds.getBaseVectors().size(), (cv.ramBytesUsed() / 1024f / 1024f), encodingTimeS);
                            }
                        }

                        try (var cs = new ConfiguredSystem(ds, index, cv, featureSetForIndex)) {
                            cs.configureQueryExecution(artifacts.queryExecutionMode(), artifacts.queryBatchSize());
                            var ctx = new SearchContext(ds, build, search, features, featureSetForIndex,
                                    buildCompressorString, cpSupplier,
                                    (cv == null) ? "None" : String.valueOf(cv.getCompressor()),
                                    indexBuildTimeS, indexFileSizes.get(features), buildSnapshot,
                                    queryParameters(cs), cs.ashQueryProjectionModel());
                            testConfiguration(cs, ctx, constructionMetrics, workDirectory, benchmarks, listeners);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    });
                }

                // Log final diagnostics summary
                if (diagnostic_level > 0) {
                    diagnostics.logSummary();
                }
            } finally {
                // Close before deleting: a failed search must not leave files open or mapped.
                for (var index : indexes.values()) {
                    try {
                        index.close();
                    } catch (Exception e) {
                        // Log and move on; don't let a close-fail mask a real exception
                        System.err.println("Cleanup Failed: Could not close index -> " + e.getMessage());
                    }
                }
                // Clean up the tmp files for this run
                for (int n = 0; n < featureSets.size(); n++) {
                    Path p = workDirectory.resolve("graph" + n);
                    try {
                        Files.deleteIfExists(p);
                    } catch (IOException e) {
                        // Log and move on; don't let a delete-fail mask a real exception
                        System.err.println("Cleanup Failed: Could not delete " + p.getFileName() + " -> " + e.getMessage());
                    }
                }
            }
        }
    }

    private static Map<String, Object> queryParameters(ConfiguredSystem cs) {
        Map<String, Object> queryParams = new LinkedHashMap<>();
        queryParams.put("executionMode", cs.usesQueuedExecution() ? "queued" : "direct");
        if (cs.usesQueuedExecution()) {
            queryParams.put("workers", cs.queryWorkerCount());
            queryParams.put("queueCapacity", Math.multiplyExact(Math.max(8, cs.queryBatchSize()), cs.queryWorkerCount()));
            queryParams.put("maxBatchSize", cs.queryBatchSize());
            var preparer = cs.queryPreparer();
            queryParams.put("batchPreparation", preparer != null && preparer.supportsBatchPreparation());
        } else {
            int threads = Integer.getInteger("jvector.bench.queryThreads", 0);
            queryParams.put("callerThreads", threads == 0 ? "common pool" : threads);
        }

        return queryParams;
    }


    // Entry-point policy only; the library's writer selection remains unchanged.
    static void configureWriterForBenchmark() {
        String value = System.getProperty("jvector.parallel_writer", "true").trim();
        if (!value.equalsIgnoreCase("true") && !value.equalsIgnoreCase("false"))
            throw new IllegalArgumentException("jvector.parallel_writer must be true or false");
        GraphIndexBuilderConfig.getInstance().setParallelBuild(Boolean.parseBoolean(value));
    }

    public static void setDiagnosticLevel(int diagLevel) {
        diagnostic_level = diagLevel;
    }

    private static DiagnosticLevel getDiagnosticLevel() {
        switch (diagnostic_level) {
            case 0:
                return DiagnosticLevel.NONE;
            case 1:
                return DiagnosticLevel.BASIC;
            case 2:
                return DiagnosticLevel.DETAILED;
            case 3:
                return DiagnosticLevel.VERBOSE;
            default:
                return DiagnosticLevel.NONE; // fallback for invalid values
        }
    }

    // avoid recomputing the compressor repeatedly (this is a relatively small memory footprint)
    static final Map<String, VectorCompressor<?>> cachedCompressors = new HashMap<>();

    /**
     * The built indexes in the order their feature sets were configured, so output order is the same on
     * every run. EnumSet hash codes come from identity hashes, so HashMap order over them varies by JVM.
     */
    private static Map<Set<FeatureId>, ImmutableGraphIndex> inConfiguredOrder(
            Map<Set<FeatureId>, ImmutableGraphIndex> built, List<? extends Set<FeatureId>> featureSets) {
        Map<Set<FeatureId>, ImmutableGraphIndex> ordered = new LinkedHashMap<>();
        for (var features : featureSets) {
            var index = built.get(features);
            if (index != null) {
                ordered.putIfAbsent(features, index);
            }
        }
        // Keep anything else a strategy returned so it is still searched and closed.
        built.forEach(ordered::putIfAbsent);
        return ordered;
    }

    /** Runs every query configuration against one index and search compressor, reporting each row to the listeners. */
    private static void testConfiguration(ConfiguredSystem cs,
                                          SearchContext ctx,
                                          ConstructionMetrics constructionMetrics,
                                          Path testDirectory,
                                          Supplier<List<QueryBenchmark>> benchmarks,
                                          List<GridListener> listeners) {
        int queryRuns = 2;
        System.out.format("%s: Using %s:%n", cs.ds.getName(), cs.index);

        // Listeners validate their selections before any benchmark runs.
        listeners.forEach(l -> l.onSearchStart(ctx));
        QueryTester tester = new QueryTester(benchmarks.get(), testDirectory, cs.ds.getName());

        for (var usePruning : ctx.search.usePruningGrid) {
            listeners.forEach(l -> l.onPruningStart(ctx, usePruning));
            for (var topK : ctx.search.topKGrid.keySet()) {
                listeners.forEach(l -> l.onTopKStart(ctx, usePruning, topK));
                for (var overquery : ctx.search.topKGrid.get(topK)) {
                    int rerankK = (int) (topK * overquery);

                    var results = tester.run(cs, topK, rerankK, usePruning, queryRuns);
                    // Merge construction-phase metrics into the runtime results so selections can see them.
                    constructionMetrics.appendTo(results);

                    var row = new QueryRow(usePruning, topK, overquery, rerankK, results);
                    listeners.forEach(l -> l.onQueryResult(ctx, row));
                }
                listeners.forEach(l -> l.onTopKEnd(ctx, usePruning, topK));
            }
        }
    }

    private static List<QueryBenchmark> setupBenchmarks(Map<String, List<String>> benchmarkSpec) {
        if (benchmarkSpec == null || benchmarkSpec.isEmpty()) {
            return List.of(
                    ThroughputBenchmark.createEmpty(3, 3)
                            .displayAvgQps(),
                    LatencyBenchmark.createDefault(),
                    CountBenchmark.createDefault(),
                    AccuracyBenchmark.createDefault()
            );
        }

        List<QueryBenchmark> benchmarks = new ArrayList<>();

        // Ensure a deterministic ordering of the benchmarks regardless of how they are provided (by YAML, etc.)
        List<String> orderedTypes = List.of("throughput", "latency", "count", "accuracy");
        for (var benchType : orderedTypes) {
            if (!benchmarkSpec.containsKey(benchType)) {
                continue;
            }
            if (benchType.equals("throughput")) {
                var bench = ThroughputBenchmark.createEmpty(3, 3);
                for (var stat : benchmarkSpec.get(benchType)) {
                    if (stat.equals("AVG")) {
                        bench = bench.displayAvgQps();
                    }
                    if (stat.equals("MEDIAN")) {
                        bench = bench.displayMedianQps();
                    }
                    if (stat.equals("MAX")) {
                        bench = bench.displayMaxQps();
                    }
                }
                benchmarks.add(bench);
            }

            if (benchType.equals("latency")) {
                var bench = LatencyBenchmark.createEmpty();
                for (var stat : benchmarkSpec.get(benchType)) {
                    if (stat.equals("AVG")) {
                        bench = bench.displayAvgLatency();
                    }
                    if (stat.equals("STD")) {
                        bench = bench.displayLatencySTD();
                    }
                    if (stat.equals("P999")) {
                        bench = bench.displayP999Latency();
                    }
                }
                benchmarks.add(bench);
            }

            if (benchType.equals("count")) {
                var bench = CountBenchmark.createEmpty();
                for (var stat : benchmarkSpec.get(benchType)) {
                    if (stat.equals("visited")) {
                        bench = bench.displayAvgNodesVisited();
                    }
                    if (stat.equals("expanded")) {
                        bench = bench.displayAvgNodesExpanded();
                    }
                    if (stat.equals("expanded base layer")) {
                        bench = bench.displayAvgNodesExpandedBaseLayer();
                    }
                }
                benchmarks.add(bench);
            }

            if (benchType.equals("accuracy")) {
                var bench = AccuracyBenchmark.createEmpty();
                for (var stat : benchmarkSpec.get(benchType)) {
                    if (stat.equals("recall")) {
                        bench = bench.displayRecall();
                    }
                    if (stat.equals("MAP")) {
                        bench = bench.displayMAP();
                    }
                }
                benchmarks.add(bench);
            }
        }

        return benchmarks;
    }

    /** @deprecated Use {@link #runAndCollectResults(DataSet, GridSpec)}, e.g. with {@link GridSpec#fromConfig}. */
    @Deprecated
    public static List<BenchResult> runAllAndCollectResults(
            DataSet ds,
            boolean enableIndexCache,
            List<Integer> mGrid,
            List<Integer> efConstructionGrid,
            List<Float> neighborOverflowGrid,
            List<Boolean> addHierarchyGrid,
            List<Boolean> refineFinalGraphGrid,
            List<? extends Set<FeatureId>> featureSets,
            List<Function<DataSet, CompressorParameters>> buildCompressors,
            List<Function<DataSet, CompressorParameters>> compressionGrid,
            Map<Integer, List<Double>> topKGrid,
            List<Boolean> usePruningGrid) throws IOException {
        var spec = GridSpec.fromGrids(addHierarchyGrid, refineFinalGraphGrid, mGrid, neighborOverflowGrid,
                efConstructionGrid, buildCompressors, featureSets,
                SearchConfig.builder()
                        .searchCompressors(compressionGrid)
                        .topKGrid(topKGrid)
                        .usePruningGrid(usePruningGrid)
                        .build(),
                enableIndexCache);
        return runAndCollectResults(ds, spec);
    }

    /** Fixed benchmark set for regression runs; the summarizer and CI comparisons depend on its metric headers. */
    private static List<QueryBenchmark> regressionBenchmarks() {
        return List.of(
                (diagnostic_level > 0 ?
                        ThroughputBenchmark.createDefault().withDiagnostics(getDiagnosticLevel()) :
                        ThroughputBenchmark.createDefault()),
                LatencyBenchmark.createDefault(),
                CountBenchmark.createDefault(),
                AccuracyBenchmark.createDefault(),
                CountBenchmark.createDefault()
        );
    }

    /**
     * Resolve a {@link VectorCompressor} from YAML {@link CompressorParameters}, with optional instrumentation.
     *
     * <p>This overload exists for reporting: when the compressor is constructed (i.e., cache miss), it records
     * the compressor build time as a quantization <em>compute</em> metric under either
     * {@code construction.index_quant_time_s.*} or {@code search.search_quant_time_s.*}, keyed by {@code quantType}
     * (e.g., {@code PQ}, {@code BQ}).</p>
     *
     * <p>We do <b>not</b> record load times. If a cached compressor is loaded, no compute metric is emitted
     * (value remains {@code null} and will be blank in CSV/console).</p>
     *
     * @param cpSupplier supplies dataset-specific compressor parameters (YAML-derived)
     * @param ds dataset being benchmarked
     * @param metrics per-run construction metrics sink (may be null)
     * @param phase which phase to attribute compute time to (index construction vs search-time setup)
     * @param quantType YAML quantization type label (e.g., PQ/BQ); if null, no quant metric is recorded
     */
    private static VectorCompressor<?> getCompressor(Function<DataSet, CompressorParameters> cpSupplier,
                                                     DataSet ds,
                                                     ConstructionMetrics metrics,
                                                     Phase phase,
                                                     String quantType) {
        var cp = cpSupplier.apply(ds);

        // Core compute and save logic
        Supplier<VectorCompressor<?>> computeAndSaveAction = () -> {
            var start = System.nanoTime();
            var compressor = cp.computeCompressor(ds);
            System.out.format("%s: %s codebooks computed in %.2fs,%n",
                    ds.getName(), compressor, (System.nanoTime() - start) / 1_000_000_000.0);

            if (cp.supportsCaching()) {
                saveToCache(cp.idStringFor(ds), compressor);
            }
            return compressor;
        };

        // Execute (with or without metrics)
        Supplier<VectorCompressor<?>> executionWrapper = () -> {
            if (metrics != null && quantType != null) {
                var h = (phase == Phase.INDEX) ? metrics.index(quantType) : metrics.search(quantType);
                return h.timeCompute(computeAndSaveAction::get);
            }
            return computeAndSaveAction.get();
        };

        // No-cache path
        if (!cp.supportsCaching()) {
            return executionWrapper.get();
        }

        // Cache path
        var fname = cp.idStringFor(ds);
        return cachedCompressors.computeIfAbsent(fname, __ -> {
            Path path = Paths.get(pqCacheDir).resolve(fname);
            if (Files.exists(path)) {
                return loadFromCache(ds, cp, fname, path);
            }
            return executionWrapper.get();
        });
    }

    // Load/save helpers for getCompressor
    private static VectorCompressor<?> loadFromCache(DataSet ds, CompressorParameters cp, String fname, Path path) {
        try (var readerSupplier = ReaderSupplierFactory.open(path);
             var rar = readerSupplier.get()) {
            // Each compressor type has its own on-disk format, so the parameters that saved it read it back.
            var compressor = cp.loadCompressor(rar);
            System.out.format("%s: %s loaded from %s%n", ds.getName(), compressor, fname);
            return compressor;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static void saveToCache(String fname, VectorCompressor<?> compressor) {
        try {
            Path path = Paths.get(pqCacheDir).resolve(fname);
            Files.createDirectories(path.getParent());
            try (var writer = new BufferedRandomAccessWriter(path)) {
                compressor.write(writer, OnDiskGraphIndex.CURRENT_VERSION);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public static class ConfiguredSystem implements AutoCloseable {
        DataSet ds;
        ImmutableGraphIndex index;
        CompressedVectors cv;
        Set<FeatureId> features;
        private final Supplier<RandomAccessVectorValues> exactVectors;
        private final boolean scoresFromIndex;
        private boolean queuedExecution;
        private int queryBatchSize = 8;
        private io.github.jbellis.jvector.graph.similarity.QueryPreparer queryPreparer;
        private java.util.concurrent.ExecutorService queuedQueryWorkers;
        private boolean closed;

        public int queryWorkerCount() {
            return Math.max(1, Math.min(Runtime.getRuntime().availableProcessors(), PhysicalCoreExecutor.getPhysicalCoreCount()));
        }

        /** Shared across this configured index's warmups and all query measurements. */
        public synchronized java.util.concurrent.ExecutorService queryWorkers() {
            if (closed) throw new IllegalStateException("Configured system is closed");
            if (queuedQueryWorkers == null) {
                int count = queryWorkerCount();
                // Drain tasks are reusable; an array queue avoids a linked queue
                // node allocation on every isolated query activation.
                queuedQueryWorkers = new java.util.concurrent.ThreadPoolExecutor(count, count,
                        0L, java.util.concurrent.TimeUnit.MILLISECONDS,
                        new java.util.concurrent.ArrayBlockingQueue<Runnable>(count),
                        new java.util.concurrent.ThreadPoolExecutor.AbortPolicy());
            }
            return queuedQueryWorkers;
        }

        /** Select before launching queries; workers and search-owned state are reused. */
        public synchronized void configureQueryExecution(String mode, int batchSize) {
            if (closed) throw new IllegalStateException("Configured system is closed");
            if (!"direct".equals(mode) && !"queued".equals(mode))
                throw new IllegalArgumentException("Query execution must be direct or queued");
            if (batchSize <= 0 || batchSize == Integer.MAX_VALUE)
                throw new IllegalArgumentException("Query batch size must be positive and fit its histogram");
            queuedExecution = "queued".equals(mode);
            queryBatchSize = batchSize;
            if (queuedExecution && queryPreparer == null)
                queryPreparer = index.queryPreparer(ds.getSimilarityFunction()).orElse(null);
        }
        public boolean usesQueuedExecution() { return queuedExecution; }
        public int queryBatchSize() { return queryBatchSize; }
        public io.github.jbellis.jvector.graph.similarity.QueryPreparer queryPreparer() { return queryPreparer; }


        private final ExplicitThreadLocal<GraphSearcher> searchers = ExplicitThreadLocal.withInitial(() -> {
            return new GraphSearcher(index);
        });

        ConfiguredSystem(DataSet ds, ImmutableGraphIndex index, CompressedVectors cv, Set<FeatureId> features) {
            this.ds = ds;
            this.index = index;
            this.cv = cv;
            this.features = features;
            this.scoresFromIndex = FeatureProviders.anyMatch(features, FeatureProvider::providesApproximateScoring);
            // File-backed sources borrow scratch storage; concurrent scorers need separate views.
            this.exactVectors = ds.getBaseRavv().threadLocalSupplier();
        }

        /** Metadata from the same ASH model used for query scoring. */
        public AsymmetricHashing ashQueryProjectionModel() {
            if (features.contains(FeatureId.FUSED_ASH) && index instanceof OnDiskGraphIndex) {
                var feature = ((OnDiskGraphIndex) index).getFeatures().get(FeatureId.FUSED_ASH);
                return feature instanceof FusedASH ? ((FusedASH) feature).getASH() : null;
            }
            return cv instanceof ASHVectors ? ((ASHVectors) cv).getCompressor() : null;
        }

        public SearchScoreProvider scoreProviderFor(VectorFloat<?> queryVector, ImmutableGraphIndex.View view) {
            var scoringView = (ImmutableGraphIndex.ScoringView) view;
            ScoreFunction.ApproximateScoreFunction asf;
            if (scoresFromIndex) {
                asf = scoringView.approximateScoreFunctionFor(queryVector, ds.getSimilarityFunction());
            } else {
                // if we're not compressing then just use the exact score function
                if (cv == null) {
                    return DefaultSearchScoreProvider.exact(queryVector, ds.getSimilarityFunction(), exactVectors.get());
                }

                asf = cv.precomputedScoreFunctionFor(queryVector, ds.getSimilarityFunction());
            }
            var rr = scoringView.rerankerFor(queryVector, ds.getSimilarityFunction());
            return new DefaultSearchScoreProvider(asf, rr, features.contains(FeatureId.FUSED_ASH) || cv instanceof ASHVectors);
        }

        public GraphSearcher getSearcher() {
            return searchers.get();
        }

        public DataSet getDataSet() {
            return ds;
        }

        @Override
        public synchronized void close() throws Exception {
            closed = true;
            try {
                if (queuedQueryWorkers != null) {
                    queuedQueryWorkers.shutdown();
                    boolean interrupted = false;
                    try {
                        while (!queuedQueryWorkers.isTerminated()) {
                            try { queuedQueryWorkers.awaitTermination(Long.MAX_VALUE, java.util.concurrent.TimeUnit.NANOSECONDS); }
                            catch (InterruptedException e) { interrupted = true; }
                        }
                    } finally { if (interrupted) Thread.currentThread().interrupt(); }
                }
            } finally {
                searchers.close();
            }
        }
    }

    // Helper that maps YAML-derived CompressorParameters to a stable label
    private static String quantTypeOf(CompressorParameters cp) {
        if (cp == null || cp == CompressorParameters.NONE) return null;
        if (cp instanceof CompressorParameters.PQParameters) return "PQ";
        if (cp instanceof CompressorParameters.BQParameters) return "BQ";
        if (cp instanceof CompressorParameters.ASHParameters) return "ASH";
        if (cp instanceof CompressorParameters.NVQParameters) return "NVQ";
        // Unknown/unsupported type for quant timing purposes
        return null;
    }
}
