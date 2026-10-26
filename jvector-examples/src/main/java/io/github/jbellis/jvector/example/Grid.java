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
import io.github.jbellis.jvector.example.benchmarks.BenchmarkTablePrinter;
import io.github.jbellis.jvector.example.benchmarks.CountBenchmark;
import io.github.jbellis.jvector.example.benchmarks.LatencyBenchmark;
import io.github.jbellis.jvector.example.benchmarks.Metric;
import io.github.jbellis.jvector.example.benchmarks.QueryBenchmark;
import io.github.jbellis.jvector.example.benchmarks.QueryTester;
import io.github.jbellis.jvector.example.benchmarks.ThroughputBenchmark;
import io.github.jbellis.jvector.example.benchmarks.diagnostics.BenchmarkDiagnostics;
import io.github.jbellis.jvector.example.benchmarks.diagnostics.DiagnosticLevel;
import io.github.jbellis.jvector.example.benchmarks.datasets.BaseDataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.ByteDataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.reporting.*;
import io.github.jbellis.jvector.example.reporting.RunArtifacts;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.FilteredForkJoinPool;
import io.github.jbellis.jvector.example.util.OnDiskGraphIndexCache;
import io.github.jbellis.jvector.example.yaml.MetricSelection;
import io.github.jbellis.jvector.management.GraphIndexBuilderConfig;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.RandomAccessByteVectorValues;
import io.github.jbellis.jvector.graph.VectorAccess;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.*;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineByteVectors;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.BQVectors;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.ByteVectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.stream.IntStream;
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
     */
    public static Double getIndexBuildTimeSeconds(String datasetName) {
        return indexBuildTimes.get(datasetName);
    }

    static void runAll(BaseDataSet ds,
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
                       List<Boolean> usePruningGrid,
                       RunArtifacts artifacts
                       ) throws IOException
    {
        boolean success = false;

        // Always use a fresh temp directory for per-run artifacts
        final Path workDir = Files.createTempDirectory(dirPrefix);

        // Initialize index cache (creates stable directory when enabled, cleans up stale temp files; no-op otherwise)
        final OnDiskGraphIndexCache cache =
                enableIndexCache
                        ? OnDiskGraphIndexCache.initialize(Paths.get(indexCacheDir))
                        : OnDiskGraphIndexCache.disabled();

        try {
            for (var addHierarchy :  addHierarchyGrid) {
                for (var refineFinalGraph : refineFinalGraphGrid) {
                    for (int M : mGrid) {
                        for (float neighborOverflow : neighborOverflowGrid) {
                            for (int efC : efConstructionGrid) {
                                for (var bc : buildCompressors) {
                                    runOneGraph(cache, featureSets, M, efC, neighborOverflow, addHierarchy, refineFinalGraph,
                                            bc, compressionGrid, topKGrid, usePruningGrid, artifacts, ds, workDir);
                                }
                            }
                        }
                    }
                }
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

            if (enableIndexCache) {
                System.out.println("Preserving index cache directory: " + indexCacheDir);
            }
            cachedCompressors.clear();
        }
    }

    // Overload for legacy callers that do not use yaml-configs
    static void runAll(BaseDataSet ds,
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
                       List<Boolean> usePruningGrid) throws IOException
    {
        runAll(ds,
                enableIndexCache,
                mGrid,
                efConstructionGrid,
                neighborOverflowGrid,
                addHierarchyGrid,
                refineFinalGraphGrid,
                featureSets,
                buildCompressors,
                compressionGrid,
                topKGrid,
                usePruningGrid,
                RunArtifacts.disabled() // legacy callers do not use reporting
        );
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
                            BaseDataSet ds,
                            Path workDirectory) throws IOException
    {
        // Prepare to collect index construction metrics for reporting....
        var constructionMetrics = new ConstructionMetrics();

        // Capture initial memory and disk state
        try (var diagnostics = new BenchmarkDiagnostics(getDiagnosticLevel())) {
            diagnostics.startMonitoring("testDirectory", workDirectory);
            diagnostics.startMonitoring("indexCache", Paths.get(indexCacheDir));
            diagnostics.capturePrePhaseSnapshot("Graph Build");
            String sfLabel = (ds instanceof ByteDataSet)
                    ? ((ByteDataSet) ds).getSimilarityFunction().toString()
                    : ((DataSet) ds).getSimilarityFunction().toString();
            System.out.printf("%s: Dataset similarity function is %s%n", ds.getName(), sfLabel);

            // Resolve build compressor (and label quant type) so we can record compute time
            VectorCompressor<?> buildCompressorObj = null;
            String buildQuantType = null;

            // Int8 datasets only support compression type "None" (enforced by MultiConfig.validateForInt8Dataset),
            // so there is no build compressor for them.
            if (buildCompressor != null && !(ds instanceof ByteDataSet)) {
                var floatDs = (DataSet) ds;
                var buildParams = buildCompressor.apply(floatDs);
                buildQuantType = quantTypeOf(buildParams); // "PQ", "BQ", or null
                buildCompressorObj = getCompressor(buildCompressor, floatDs, constructionMetrics, Phase.INDEX, buildQuantType);
            }

            // Used in logging
            String buildCompressorString =
                    (buildCompressorObj == null) ? "None" : String.valueOf(buildCompressorObj);

            Map<Set<FeatureId>, ImmutableGraphIndex> indexes = new HashMap<>();
            Map<Set<FeatureId>, Long> indexFileSizes = new HashMap<>();
            if (buildCompressorObj == null) {
                indexes = buildInMemory(featureSets, M, efConstruction, neighborOverflow, addHierarchy, refineFinalGraph, ds, workDirectory);
            } else {
                // If cache is disabled, we use the (tmp) workDirectory as the output
                Path outputDir = cache.isEnabled() ? cache.cacheDir().toAbsolutePath() : workDirectory;

                List<Set<FeatureId>> missing = new ArrayList<>();
                // Map feature sets to their cache handles for the build phase
                Map<Set<FeatureId>, OnDiskGraphIndexCache.WriteHandle> handles = new HashMap<>();

                for (Set<FeatureId> fs : featureSets) {
                    var key = cache.key(ds.getName(), fs, M, efConstruction, neighborOverflow, 1.2f, addHierarchy, refineFinalGraph, buildCompressorObj);
                    var cached = cache.tryLoad(key);

                    if (cached.isPresent()) {
                        System.out.printf("%s: Using cached graph index for %s%n", key.datasetName, fs);
                        indexes.put(fs, cached.get());
                        try { indexFileSizes.put(fs, Files.size(cache.resolve(key).finalPath)); } catch (IOException ignored) {}
                    } else {
                        missing.add(fs);
                        if (cache.isEnabled()) {
                            var handle = cache.beginWrite(key, OnDiskGraphIndexCache.Overwrite.ALLOW);
                            // Log cache miss / build start
                            System.out.printf("%s: Building graph index (cached enabled) for %s%n", key.datasetName, fs);
                            // Prepare the atomic write handle immediately
                            handles.put(fs, handle);
                        } else {
                            System.out.printf("%s: Building graph index (cache disabled) for %s%n", key.datasetName, fs);
                        }
                    }
                }

                if (!missing.isEmpty()) {
                    // At least one index needs to be built (b/c not in cache or cache is disabled)
                    // We pass the handles map so buildOnDisk knows exactly where to write
                    var result = buildOnDisk(missing, M, efConstruction, neighborOverflow, addHierarchy, refineFinalGraph,
                            ds, outputDir, buildCompressorObj, handles, constructionMetrics);
                    indexes.putAll(result.indexes);
                    indexFileSizes.putAll(result.fileSizes);
                }
            }

            // Capture post-build memory and disk state
            diagnostics.capturePostPhaseSnapshot("Graph Build");

            diagnostics.printDiskStatistics("Graph Index Build");
            System.out.printf("Index build time: %f seconds%n%n", Grid.getIndexBuildTimeSeconds(ds.getName()));
            constructionMetrics.indexBuildTimeS = Grid.getIndexBuildTimeSeconds(ds.getName());

            try {
                for (var cpSupplier : compressionGrid) {
                    indexes.forEach((features, index) -> {
                        constructionMetrics.indexFileSizeBytes = indexFileSizes.get(features);
                        final Set<FeatureId> featureSetForIndex = index instanceof OnDiskGraphIndex ? ((OnDiskGraphIndex) index).getFeatureSet() : Set.of();

                        CompressedVectors cv;
                        if (featureSetForIndex.contains(FeatureId.FUSED_PQ)) {
                            cv = null;
                            System.out.format("%s: configured to use FUSED PQ, skipping vector compression%n", ds.getName());
                        } else {
                            constructionMetrics.resetSearch(); // per (index, cpSupplier) config

                            if (ds instanceof ByteDataSet) {
                                cv = null;
                                System.out.format("%s: No search compressor configured, FULL PRECISION vectors will be used for search%n", ds.getName());
                            } else {
                                var floatDs = (DataSet) ds;
                                var searchParams = cpSupplier.apply(floatDs);
                                String searchQuantType = quantTypeOf(searchParams); // "PQ", "BQ", or null
                                var compressor = getCompressor(cpSupplier, floatDs, constructionMetrics, Phase.SEARCH, searchQuantType);

                                if (compressor == null) {
                                    cv = null;
                                    System.out.format("%s: No search compressor configured, FULL PRECISION vectors will be used for search%n", ds.getName());
                                } else {
                                    long start = System.nanoTime();
                                    cv = constructionMetrics.search(searchQuantType)
                                            .timeEncode(() -> compressor.encodeAll(floatDs.getBaseRavv()));
                                    double encodingTimeS = (System.nanoTime() - start) / 1_000_000_000.0;
                                    if (cv == null) {
                                        throw new IllegalStateException(String.format(
                                                "Compressor '%s' was provided but failed to encode vectors for dataset '%s'. " +
                                                        "Aborting to prevent false recall results.", compressor, ds.getName()));
                                    }
                                    System.out.format("%s: %s encoded %d vectors [%.2f MB] in %.2fs%n", ds.getName(), compressor, ds.size(), (cv.ramBytesUsed() / 1024f / 1024f), encodingTimeS);
                                }
                            }
                        }

                        try (var cs = new ConfiguredSystem(ds, index, cv, featureSetForIndex)) {
                            testConfiguration(cs, topKGrid, usePruningGrid, M, efConstruction, neighborOverflow, addHierarchy, refineFinalGraph, featureSetForIndex,
                                    buildCompressorString, artifacts, constructionMetrics, workDirectory);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    });
                }
                for (var index : indexes.values()) {
                    index.close();
                }

                // Log final diagnostics summary
                if (diagnostic_level > 0) {
                    diagnostics.logSummary();
                }
            } finally {
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

    /**
     * Result of {@link #buildOnDisk}: for each feature set built, the graph index re-opened
     * from the written file and the size in bytes of that file on disk.
     */
    private static final class BuildOnDiskResult {
        final Map<Set<FeatureId>, ImmutableGraphIndex> indexes;
        final Map<Set<FeatureId>, Long> fileSizes;
        BuildOnDiskResult(Map<Set<FeatureId>, ImmutableGraphIndex> indexes, Map<Set<FeatureId>, Long> fileSizes) {
            this.indexes = indexes;
            this.fileSizes = fileSizes;
        }
    }

    private static BuildOnDiskResult buildOnDisk(List<? extends Set<FeatureId>> featureSets,
                                                                        int M,
                                                                        int efConstruction,
                                                                        float neighborOverflow,
                                                                        boolean addHierarchy,
                                                                        boolean refineFinalGraph,
                                                                        BaseDataSet ds,
                                                                        Path outputDir,
                                                                        VectorCompressor<?> buildCompressor,
                                                                        Map<Set<FeatureId>, OnDiskGraphIndexCache.WriteHandle> handles,
                                                                        ConstructionMetrics constructionMetrics) throws IOException
    {
        Files.createDirectories(outputDir);

        var floatDs = (DataSet) ds;
        var floatVectors = floatDs.getBaseRavv();

        // Record the encoding time if asked for....
        var encoded = (constructionMetrics != null)
                ? constructionMetrics.index(buildCompressor instanceof ProductQuantization ? "PQ" : "BQ")
                        .timeEncode(() -> buildCompressor.encodeAll(floatVectors))
                : buildCompressor.encodeAll(floatVectors);
        if (!(encoded instanceof PQVectors) && !(encoded instanceof BQVectors))
            throw new IllegalArgumentException("Disk construction supports PQ or BQ build compression");
        PQVectors pq = encoded instanceof PQVectors ? (PQVectors) encoded : null;
        if (pq == null && featureSets.stream().anyMatch(fs -> fs.contains(FeatureId.FUSED_PQ)))
            throw new IllegalArgumentException("FusedPQ requires PQ build compression with 256 clusters");
        var bsp = pq != null
                ? BuildScoreProvider.pqBuildScoreProvider(floatDs.getSimilarityFunction(), pq)
                : BuildScoreProvider.bqBuildScoreProvider((BQVectors) encoded);
        GraphIndexBuilder builder = new GraphIndexBuilder(bsp, floatVectors.dimension(), M, efConstruction, neighborOverflow, 1.2f, addHierarchy, refineFinalGraph);

        // use the inline vectors index as the score provider for graph construction
        Map<Set<FeatureId>, RandomAccessOnDiskGraphIndexWriter> writers = new HashMap<>();
        Map<Set<FeatureId>, Map<FeatureId, Function<VectorFloat<?>, Feature.State>>> inputSuppliers = new HashMap<>();
        RandomAccessOnDiskGraphIndexWriter scoringWriter = null;
        int n = 0;
        for (var features : featureSets) {
            // if we are using index caching, use cache names instead of tmp names for index files....
            Path graphPath;
            if (handles.containsKey(features)) {
                graphPath = handles.get(features).writePath();
            } else {
                graphPath = outputDir.resolve("graph" + n++);
            }
            var bws = builderWithSuppliers(features, builder.getGraph(), graphPath, floatVectors, pq == null ? null : pq.getCompressor(), constructionMetrics);
            var writer = bws.builder.build();
            writers.put(features, writer);
            inputSuppliers.put(features, bws.inputSuppliers);
            if (features.contains(FeatureId.INLINE_VECTORS) || features.contains(FeatureId.NVQ_VECTORS)) {
                scoringWriter = writer;
            }
        }
        if (scoringWriter == null) {
            throw new IllegalStateException("Bench looks for either NVQ_VECTORS or INLINE_VECTORS feature set for scoring compressed builds.");
        }

        // The parallel writer calls final suppliers from its workers, so original-vector features
        // can be encoded after cleanup. The sequential writer calls them on one thread, so it keeps
        // encoding on the insertion threads and writing each node's features inline.
        boolean deferFeatures = writers.values().stream()
                .allMatch(writer -> writer instanceof OnDiskParallelGraphIndexWriter);
        long startTime = System.nanoTime();
        if (ds.useDatasetPrefetching()) {
            VectorAccess.forEach(floatVectors, PhysicalCoreExecutor.pool(), (node, vector) -> {
                if (!deferFeatures) writeInlineStates(writers, inputSuppliers, node, vector);
                builder.addGraphNode(node, vector);
            });
        } else {
            var vv = floatVectors.threadLocalSupplier();
            PhysicalCoreExecutor.pool().submit(() -> {
                IntStream.range(0, floatVectors.size()).parallel().forEach(node -> {
                    var vector = vv.get().getVector(node);
                    if (!deferFeatures) writeInlineStates(writers, inputSuppliers, node, vector);
                    builder.addGraphNode(node, vector);
                });
            }).join();
        }
        builder.cleanup();

        // With deferred features, supply every feature so the parallel writer emits complete records.
        writers.entrySet().stream().parallel().forEach(entry -> {
            var writer = entry.getValue();
            var features = entry.getKey();
            try (var vectors = new FinalWriteVectors(floatVectors);
                 var view = builder.getGraph().getView();
                 var closeWriter = writer) {
                Map<FeatureId, IntFunction<Feature.State>> writeSuppliers = new EnumMap<>(FeatureId.class);
                if (deferFeatures) {
                    inputSuppliers.get(features).forEach((id, supplier) ->
                            writeSuppliers.put(id, ordinal -> supplier.apply(vectors.get(ordinal))));
                }
                if (features.contains(FeatureId.FUSED_PQ)) {
                    writeSuppliers.put(FeatureId.FUSED_PQ, ordinal -> new FusedPQ.State(view, pq, ordinal));
                }
                writer.write(writeSuppliers);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
        builder.close();
        double totalTime = (System.nanoTime() - startTime) / 1_000_000_000.0;
        System.out.format("Build and write %s in %ss%n", featureSets, totalTime);
        indexBuildTimes.put(ds.getName(), totalTime);

        // Commit the index to cache atomically
        for (var entry : handles.entrySet()) {
            System.out.printf("Committing %s build to graph index cache...%n", entry.getKey());
            entry.getValue().commit();
        }

        // open indexes and capture per-feature-set file sizes
        Map<Set<FeatureId>, ImmutableGraphIndex> indexes = new HashMap<>();
        Map<Set<FeatureId>, Long> fileSizes = new HashMap<>();
        n = 0;
        for (var features : featureSets) {
            Path loadPath = handles.containsKey(features)
                    ? handles.get(features).finalPath()
                    : outputDir.resolve("graph" + n++);

            indexes.put(features, OnDiskGraphIndex.load(ReaderSupplierFactory.open(loadPath)));
            fileSizes.put(features, Files.size(loadPath));
        }

        return new BuildOnDiskResult(indexes, fileSizes);
    }

    /// Write one node's original-vector features to every writer during insertion. The borrowed
    /// vector is encoded and written before the caller advances to another vector.
    private static void writeInlineStates(
            Map<Set<FeatureId>, RandomAccessOnDiskGraphIndexWriter> writers,
            Map<Set<FeatureId>, Map<FeatureId, Function<VectorFloat<?>, Feature.State>>> inputSuppliers,
            int node, VectorFloat<?> vector) {
        writers.forEach((features, writer) -> {
            var states = new EnumMap<FeatureId, Feature.State>(FeatureId.class);
            inputSuppliers.get(features).forEach((id, supplier) -> states.put(id, supplier.apply(vector)));
            try { writer.writeFeaturesInline(node, states); }
            catch (IOException e) { throw new UncheckedIOException(e); }
        });
    }

    /** Worker-local bounded input ranges; a borrowed vector is consumed before advancing. */
    static final class FinalWriteVectors implements AutoCloseable {
        private final RandomAccessVectorValues source;
        private final java.util.concurrent.ConcurrentLinkedQueue<Range> ranges =
                new java.util.concurrent.ConcurrentLinkedQueue<>();
        private final ThreadLocal<Range> local = ThreadLocal.withInitial(() -> {
            var range = new Range();
            ranges.add(range);
            return range;
        });

        FinalWriteVectors(RandomAccessVectorValues source) { this.source = source; }

        VectorFloat<?> get(int ordinal) {
            Objects.checkIndex(ordinal, source.size());
            var range = local.get();
            if (range.cursor == null || ordinal != range.ordinal) {
                if (range.cursor == null || ordinal != range.ordinal + 1 || ordinal == range.end) {
                    range.close();
                    range.end = (int) Math.min(source.size(), (long) ordinal + 64);
                    range.cursor = VectorAccess.openRange(source, ordinal, range.end);
                }
                if (!range.cursor.next() || range.cursor.ordinal() != ordinal)
                    throw new IllegalStateException("Incomplete final-write input range");
                range.ordinal = ordinal;
            }
            return range.cursor.vector();
        }

        @Override public void close() {
            // writer.write joins its workers before cursors and their read-ahead are released.
            RuntimeException failure = null;
            try {
                for (var range : ranges) {
                    try { range.close(); }
                    catch (RuntimeException e) {
                        if (failure == null) failure = e;
                        else if (failure != e) failure.addSuppressed(e);
                    }
                }
            } finally { ranges.clear(); local.remove(); }
            if (failure != null) throw failure;
        }

        private static final class Range {
            io.github.jbellis.jvector.graph.VectorCursor cursor;
            int ordinal = -1, end;
            void close() {
                if (cursor != null) {
                    try { cursor.close(); }
                    finally { cursor = null; ordinal = -1; }
                }
            }
        }
    }

    private static BuilderWithSuppliers builderWithSuppliers(Set<FeatureId> features,
                                                             ImmutableGraphIndex onHeapGraph,
                                                             Path outPath,
                                                             RandomAccessVectorValues floatVectors,
                                                             ProductQuantization pq,
                                                             ConstructionMetrics constructionMetrics)
            throws FileNotFoundException
    {
        var identityMapper = new OrdinalMapper.IdentityMapper(floatVectors.size() - 1);
        var builder = new RandomAccessOnDiskGraphIndexWriter.Builder(onHeapGraph, outPath);
        builder.withMapper(identityMapper);
        builder.withParallelWorkerThreads(PhysicalCoreExecutor.getPhysicalCoreCount());

        Map<FeatureId, IntFunction<Feature.State>> suppliers = new EnumMap<>(FeatureId.class);
        Map<FeatureId, Function<VectorFloat<?>, Feature.State>> inputSuppliers = new EnumMap<>(FeatureId.class);
        // File-backed sources reuse scratch vectors; inline writes need a view per worker.
        var writerVectors = floatVectors.threadLocalSupplier();
        for (var featureId : features) {
            switch (featureId) {
                case INLINE_VECTORS:
                    builder.with(new InlineVectors(floatVectors.dimension()));
                    inputSuppliers.put(FeatureId.INLINE_VECTORS, InlineVectors.State::new);
                    suppliers.put(FeatureId.INLINE_VECTORS, ordinal -> new InlineVectors.State(writerVectors.get().getVector(ordinal)));
                    break;
                case FUSED_PQ:
                    if (pq == null) {
                        System.out.println("Skipping Fused ADC feature due to null ProductQuantization");
                        continue;
                    }
                    // no supplier as these will be used for writeInline, when we don't have enough information to fuse neighbors
                    builder.with(new FusedPQ(onHeapGraph.maxDegree(), pq));
                    break;
                case NVQ_VECTORS:
                    int nSubVectors = floatVectors.dimension() == 2 ? 1 : 2;
                    var nvq = (constructionMetrics != null)
                            ? constructionMetrics.index("NVQ").timeCompute(() -> NVQuantization.compute(floatVectors, nSubVectors))
                            : NVQuantization.compute(floatVectors, nSubVectors);
                    builder.with(new NVQ(nvq));
                    inputSuppliers.put(FeatureId.NVQ_VECTORS, vector -> new NVQ.State(nvq.encode(vector)));
                    suppliers.put(FeatureId.NVQ_VECTORS, ordinal -> new NVQ.State(nvq.encode(writerVectors.get().getVector(ordinal))));
                    break;

            }
        }
        return new BuilderWithSuppliers(builder, suppliers, inputSuppliers);
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

    private static class BuilderWithSuppliers {
        public final RandomAccessOnDiskGraphIndexWriter.Builder builder;
        public final Map<FeatureId, IntFunction<Feature.State>> suppliers;

        public final Map<FeatureId, Function<VectorFloat<?>, Feature.State>> inputSuppliers;

        public BuilderWithSuppliers(RandomAccessOnDiskGraphIndexWriter.Builder builder, Map<FeatureId, IntFunction<Feature.State>> suppliers) {
            this(builder, suppliers, Map.of());
        }

        public BuilderWithSuppliers(RandomAccessOnDiskGraphIndexWriter.Builder builder, Map<FeatureId, IntFunction<Feature.State>> suppliers,
                                    Map<FeatureId, Function<VectorFloat<?>, Feature.State>> inputSuppliers) {
            this.builder = builder;
            this.suppliers = suppliers;
            this.inputSuppliers = inputSuppliers;
        }
    }

    private static Map<Set<FeatureId>, ImmutableGraphIndex> buildInMemory(List<? extends Set<FeatureId>> featureSets,
                                                                          int M,
                                                                          int efConstruction,
                                                                          float neighborOverflow,
                                                                          boolean addHierarchy,
                                                                          boolean refineFinalGraph,
                                                                          BaseDataSet ds,
                                                                          Path testDirectory)
            throws IOException
    {
        Map<Set<FeatureId>, ImmutableGraphIndex> indexes = new HashMap<>();
        long start;

        // ── Int8 path ────────────────────────────────────────────────────────
        if (ds instanceof ByteDataSet) {
            ByteDataSet byteDs = (ByteDataSet) ds;
            RandomAccessByteVectorValues byteVectors = byteDs.getBaseRavv();
            ByteVectorSimilarityFunction bvsf = byteDs.getSimilarityFunction();
            var bsp = BuildScoreProvider.byteVectorScoreProvider(byteVectors, bvsf);
            GraphIndexBuilder builder = new GraphIndexBuilder(bsp,
                                                              byteVectors.dimension(),
                                                              M,
                                                              efConstruction,
                                                              neighborOverflow,
                                                              1.2f,
                                                              addHierarchy,
                                                              refineFinalGraph,
                                                              PhysicalCoreExecutor.pool(),
                                                              FilteredForkJoinPool.createFilteredPool());
            start = System.nanoTime();
            var onHeapGraph = builder.build(byteVectors);
            double buildTimeS = (System.nanoTime() - start) / 1_000_000_000.0;
            System.out.format("Build (int8) M=%d overflow=%.2f ef=%d in %.2fs%n",
                              M, neighborOverflow, efConstruction, buildTimeS);
            for (int i = 0; i <= onHeapGraph.getMaxLevel(); i++) {
                System.out.format("  L%d: %d nodes, %.2f avg degree%n",
                                  i, onHeapGraph.size(i), onHeapGraph.getAverageDegree(i));
            }
            int n = 0;
            for (var features : featureSets) {
                if (features.contains(FeatureId.FUSED_PQ)) {
                    System.out.println("Skipping Fused ADC feature when building in memory");
                    continue;
                }
                var graphPath = testDirectory.resolve("graph" + n++);
                var identityMapper = new OrdinalMapper.IdentityMapper(byteVectors.size() - 1);
                var writerBuilder = new RandomAccessOnDiskGraphIndexWriter.Builder(onHeapGraph, graphPath)
                        .withParallelWorkerThreads(PhysicalCoreExecutor.getPhysicalCoreCount())
                        .withMapper(identityMapper)
                        .with(new InlineByteVectors(byteVectors.dimension()));
                try (var writer = writerBuilder.build()) {
                    start = System.nanoTime();
                    writer.write(Map.of(
                            FeatureId.INLINE_BYTE_VECTORS,
                            ordinal -> new InlineByteVectors.State(byteVectors.getVector(ordinal))));
                    System.out.format("Wrote %s in %.2fs%n", features, (System.nanoTime() - start) / 1_000_000_000.0);
                }
                indexes.put(features, OnDiskGraphIndex.load(ReaderSupplierFactory.open(graphPath)));
            }
            indexBuildTimes.put(ds.getName(), buildTimeS);
            return indexes;
        }

        // ── Float path ───────────────────────────────────────────────────────
        var floatDs = (DataSet) ds;
        var floatVectors = floatDs.getBaseRavv();
        var bsp = BuildScoreProvider.randomAccessScoreProvider(floatVectors, floatDs.getSimilarityFunction());
        GraphIndexBuilder builder = new GraphIndexBuilder(bsp,
                                                          floatVectors.dimension(),
                                                          M,
                                                          efConstruction,
                                                          neighborOverflow,
                                                          1.2f,
                                                          addHierarchy,
                                                          refineFinalGraph,
                                                          PhysicalCoreExecutor.pool(),
                                                          FilteredForkJoinPool.createFilteredPool());
        start = System.nanoTime();
        var onHeapGraph = builder.build(floatVectors);
        double buildTimeS = (System.nanoTime() - start) / 1_000_000_000.0;
        System.out.format("Build (%s) M=%d overflow=%.2f ef=%d in %.2fs%n",
                          "full res",
                          M,
                          neighborOverflow,
                          efConstruction,
                          buildTimeS);
        for (int i = 0; i <= onHeapGraph.getMaxLevel(); i++) {
            System.out.format("  L%d: %d nodes, %.2f avg degree%n",
                              i,
                              onHeapGraph.size(i),
                              onHeapGraph.getAverageDegree(i));
        }
        int n = 0;
        for (var features : featureSets) {
            if (features.contains(FeatureId.FUSED_PQ)) {
                System.out.println("Skipping Fused ADC feature when building in memory");
                continue;
            }
            var graphPath = testDirectory.resolve("graph" + n++);
            var bws = builderWithSuppliers(features, onHeapGraph, graphPath, floatVectors, null, null);
            try (var writer = bws.builder.build()) {
                start = System.nanoTime();
                writer.write(bws.suppliers);
                System.out.format("Wrote %s in %.2fs%n", features, (System.nanoTime() - start) / 1_000_000_000.0);
            }

            var index = OnDiskGraphIndex.load(ReaderSupplierFactory.open(graphPath));
            indexes.put(features, index);
        }
        indexBuildTimes.put(ds.getName(), buildTimeS);
        return indexes;
    }

    // avoid recomputing the compressor repeatedly (this is a relatively small memory footprint)
    static final Map<String, VectorCompressor<?>> cachedCompressors = new IdentityHashMap<>();

    private static void testConfiguration(ConfiguredSystem cs,
                                          Map<Integer, List<Double>> topKGrid,
                                          List<Boolean> usePruningGrid,
                                          int M,
                                          int efConstruction,
                                          float neighborOverflow,
                                          boolean addHierarchy,
                                          boolean refineFinalGraph,
                                          Set<FeatureId> featureSetForIndex,
                                          String buildCompressorString,
                                          RunArtifacts artifacts,
                                          ConstructionMetrics constructionMetrics,
                                          Path testDirectory) {
        int queryRuns = 2;
        System.out.format("%s: Using %s:%n", cs.ds.getName(), cs.index);

        Map<String, List<String>> benchmarksToCompute = artifacts.benchmarksToCompute();
        Map<String, List<String>> benchmarksToDisplay = artifacts.benchmarksToDisplay();
        MetricSelection metricsToDisplay = artifacts.metricsToDisplay();

        Map<String, List<String>> benchmarksToLog = artifacts.benchmarksToLog();
        MetricSelection metricsToLog = artifacts.metricsToLog();

        // 1) Select what to report (console + logging)
        SearchSelection consoleSel = SearchSelection.forConsole(benchmarksToDisplay, metricsToDisplay);
        SearchSelection logSel = SearchSelection.forLogging(benchmarksToLog, metricsToLog);

        // Fail-fast validations once per configuration
        consoleSel.validate(benchmarksToCompute);
        logSel.validate(benchmarksToCompute);

        // 2) Select benchmarks to run (compute spec)
        var benchmarks = setupBenchmarks(benchmarksToCompute);
        QueryTester tester = new QueryTester(benchmarks, testDirectory, cs.ds.getName());

        // 3) Setup benchmark table for printing
        for (var usePruning : usePruningGrid) {
            BenchmarkTablePrinter printer = new BenchmarkTablePrinter();

            // Print the config once per index and query configuration (not per topK)
            printer.printConfig(
                    ordered( // Index configuration
                        "featureSetForIndex", featureSetForIndex,
                        "M",                  M,
                        "efConstruction",     efConstruction,
                        "neighborOverflow",   neighborOverflow,
                        "addHierarchy",       addHierarchy,
                        "refineFinalGraph",   refineFinalGraph
                    ),
                    ordered( // Query configuration
                            "usePruning",         usePruning
                    )
            );

            for (var topK : topKGrid.keySet()) {
                // Resolving selections depends on topK
                var consoleResolved = consoleSel.resolveForTopK(topK);
                var logResolved = logSel.resolveForTopK(topK);

                // Force header to re-print for this topK (columns change)
                printer.resetTable();

                for (var overquery : topKGrid.get(topK)) {
                    int rerankK = (int) (topK * overquery);

                    var results = tester.run(cs, topK, rerankK, usePruning, queryRuns);
                    // Merge construction-phase metrics into the runtime results so selections can see them.
                    constructionMetrics.appendTo(results);

                    // Best-effort runtime availability warnings (warn once per key)
                    consoleSel.warnMissing(results, consoleResolved);
                    logSel.warnMissing(results, logResolved);

                    // Apply console selection and print
                    var displayResults = consoleSel.apply(results, consoleResolved);
                    printer.printRow(overquery, displayResults);

                    // Apply log selection and write CSV row
                    String searchCompressorString = (cs.cv == null) ? "None" : String.valueOf(cs.cv.getCompressor());
                    var logOutputs = logSel.apply(results, logResolved);
                    artifacts.logRow(
                            cs.ds.getName(),
                            M,
                            efConstruction,
                            neighborOverflow,
                            addHierarchy,
                            refineFinalGraph,
                            featureSetForIndex,
                            buildCompressorString,
                            searchCompressorString,
                            usePruning,
                            topK,
                            overquery,
                            rerankK,
                            logOutputs
                    );
                }
                printer.printFooter();
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

    // Helper for printConfig (preserves insertion order)
    private static Map<String, Object> ordered(Object... kv) {
        if ((kv.length & 1) != 0)
            throw new IllegalArgumentException("Need even number of args (k,v)*");

        var m = new LinkedHashMap<String, Object>(kv.length / 2);
        for (int i = 0; i < kv.length; i += 2) {
            m.put((String) kv[i], kv[i + 1]);
        }
        return m; // insertion-ordered
    }

    public static List<BenchResult> runAllAndCollectResults(
            BaseDataSet ds,
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

        // Initialize index caching (if enabled)
        final OnDiskGraphIndexCache cache =
                enableIndexCache
                        ? OnDiskGraphIndexCache.initialize(Paths.get(indexCacheDir))
                        : OnDiskGraphIndexCache.disabled();

        List<BenchResult> results = new ArrayList<>();
        for (int m : mGrid) {
            for (int ef : efConstructionGrid) {
                for (float neighborOverflow : neighborOverflowGrid) {
                    for (boolean addHierarchy : addHierarchyGrid) {
                        for (boolean refineFinalGraph : refineFinalGraphGrid) {
                            for (Set<FeatureId> features : featureSets) {
                                for (Function<DataSet, CompressorParameters> buildCompressor : buildCompressors) {
                                    for (Function<DataSet, CompressorParameters> searchCompressor : compressionGrid) {
                                        Path testDirectory = Files.createTempDirectory("bench");
                                        try (var diagnostics = new BenchmarkDiagnostics(getDiagnosticLevel())) {
                                            // Capture initial state
                                            diagnostics.startMonitoring("testDirectory", testDirectory);
                                            diagnostics.startMonitoring("indexCache", Paths.get(indexCacheDir));
                                            diagnostics.capturePrePhaseSnapshot("Build");
                                            Map<Set<FeatureId>, ImmutableGraphIndex> indexes = new HashMap<>();
                                            Map<Set<FeatureId>, Long> indexFileSizes = new HashMap<>();

                                            VectorCompressor<?> compressor = null;
                                            VectorCompressor<?> searchCompressorObj = null;
                                            if (ds instanceof DataSet) {
                                                var floatDs = (DataSet) ds;
                                                compressor = getCompressor(buildCompressor, floatDs);
                                                searchCompressorObj = getCompressor(searchCompressor, floatDs);
                                            }
                                            // Encode vectors for reranking if a compressor is provided
                                            CompressedVectors cvArg;
                                            if (features.contains(FeatureId.FUSED_PQ)) {
                                                cvArg = null;
                                                System.out.format("%s: configured to use FUSED PQ, skipping vector compression%n", ds.getName());
                                            } else {
                                                if (searchCompressorObj == null) {
                                                    cvArg = null;
                                                    System.out.format("%s: No search compressor configured, " +
                                                            "FULL PRECISION vectors will be used for search%n", ds.getName());
                                                } else {
                                                    var floatDs = (DataSet) ds;
                                                    cvArg = searchCompressorObj.encodeAll(floatDs.getBaseRavv());
                                                    if (cvArg == null) {
                                                        throw new IllegalStateException(String.format(
                                                                "Compressor '%s' was provided but failed to encode vectors for dataset '%s'. " +
                                                                        "Aborting to prevent false recall results.",
                                                                searchCompressorObj, ds.getName()));
                                                    }
                                                    System.out.format("%s: %s encoded %d vectors [%.2f MB] for search%n",
                                                            ds.getName(), searchCompressorObj, ds.size(),
                                                            (cvArg.ramBytesUsed() / 1024f / 1024f));
                                                }
                                            }
                                            // If cache is disabled, we use the (tmp) testDirectory as the output
                                            Path outputDir = cache.isEnabled() ? cache.cacheDir().toAbsolutePath() : testDirectory;

                                            List<Set<FeatureId>> missing = new ArrayList<>();
                                            // Map feature sets to their cache handles for the build phase
                                            Map<Set<FeatureId>, OnDiskGraphIndexCache.WriteHandle> handles = new HashMap<>();

                                            for (Set<FeatureId> fs : featureSets) {
                                                var key = cache.key(ds.getName(), fs, m, ef, neighborOverflow, 1.2f, addHierarchy, refineFinalGraph, compressor);
                                                var cached = cache.tryLoad(key);

                                                if (cached.isPresent()) {
                                                    System.out.printf("%s: Using cached graph index for %s%n", key.datasetName, fs);
                                                    indexes.put(fs, cached.get());
                                                    try { indexFileSizes.put(fs, Files.size(cache.resolve(key).finalPath)); } catch (IOException ignored) {}
                                                } else {
                                                    missing.add(fs);
                                                    if (cache.isEnabled()) {
                                                        var handle = cache.beginWrite(key, OnDiskGraphIndexCache.Overwrite.ALLOW);
                                                        // Log cache miss / build start
                                                        System.out.printf("%s: Building graph index (cached enabled) for %s%n",
                                                                key.datasetName, fs);
                                                        // Prepare the atomic write handle immediately
                                                        handles.put(fs, handle);
                                                    } else {
                                                        System.out.printf("%s: Building graph index (cache disabled) for %s%n", key.datasetName, fs);
                                                    }
                                                }
                                            }

                                            if (!missing.isEmpty()) {
                                                // At least one index needs to be built (b/c not in cache or cache is disabled)
                                                // We pass the handles map so buildOnDisk knows exactly where to write
                                                var result = buildOnDisk(missing, m, ef, neighborOverflow, addHierarchy, refineFinalGraph,
                                                        ds, outputDir, compressor, handles, null);
                                                indexes.putAll(result.indexes);
                                                indexFileSizes.putAll(result.fileSizes);
                                            }

                                            ImmutableGraphIndex index = indexes.get(features);

                                            // Capture post-build state
                                            diagnostics.capturePostPhaseSnapshot("Build");
                                            diagnostics.printDiskStatistics("Graph Index Build");
                                            var buildSnapshot = diagnostics.getLatestSystemSnapshot();

                                            try (ConfiguredSystem cs = new ConfiguredSystem(ds, index, cvArg, features)) {
                                                int queryRuns = 2;
                                                List<QueryBenchmark> benchmarks = List.of(
                                                        (diagnostic_level > 0 ?
                                                                ThroughputBenchmark.createDefault().withDiagnostics(getDiagnosticLevel()) :
                                                                ThroughputBenchmark.createDefault()),
                                                        LatencyBenchmark.createDefault(),
                                                        CountBenchmark.createDefault(),
                                                        AccuracyBenchmark.createDefault(),
                                                        CountBenchmark.createDefault()
                                                );
                                                QueryTester tester = new QueryTester(benchmarks, testDirectory, ds.getName());
                                                for (int topK : topKGrid.keySet()) {
                                                    for (boolean usePruning : usePruningGrid) {
                                                        for (double overquery : topKGrid.get(topK)) {
                                                            int rerankK = (int) (topK * overquery);
                                                            List<Metric> metricsList = tester.run(cs, topK, rerankK, usePruning, queryRuns);
                                                            Map<String, Object> params = new LinkedHashMap<>();
                                                            params.put("M", m);
                                                            params.put("efConstruction", ef);
                                                            params.put("neighborOverflow", neighborOverflow);
                                                            params.put("addHierarchy", addHierarchy);
                                                            params.put("refineFinalGraph", refineFinalGraph);
                                                            params.put("features", features.toString());
                                                            params.put("buildCompressor", buildCompressor.toString());
                                                            params.put("searchCompressor", searchCompressor.toString());
                                                            params.put("topK", topK);
                                                            params.put("overquery", overquery);
                                                            params.put("usePruning", usePruning);

                                                            // Collect all metrics including memory and disk usage
                                                            Map<String, Object> allMetrics = new HashMap<>();
                                                            for (Metric metric : metricsList) {
                                                                allMetrics.put(metric.getHeader(), metric.getValue());
                                                            }

                                                            // Add build time if available
                                                            if (indexBuildTimes.containsKey(ds.getName())) {
                                                                allMetrics.put("Index Build Time", indexBuildTimes.get(ds.getName()));
                                                            }

                                                            // Add memory metrics if available
                                                            if (buildSnapshot != null) {
                                                                allMetrics.put("Heap Memory Used (MB)", buildSnapshot.memoryStats.heapUsed / 1024.0 / 1024.0);
                                                                allMetrics.put("Heap Memory Max (MB)", buildSnapshot.memoryStats.heapMax / 1024.0 / 1024.0);
                                                                allMetrics.put("Off-Heap Direct (MB)", buildSnapshot.memoryStats.directBufferMemory / 1024.0 / 1024.0);
                                                                allMetrics.put("Off-Heap Mapped (MB)", buildSnapshot.memoryStats.mappedBufferMemory / 1024.0 / 1024.0);
                                                                allMetrics.put("Total Off-Heap (MB)", buildSnapshot.memoryStats.getTotalOffHeapMemory() / 1024.0 / 1024.0);
                                                            }

                                                            // Add per-feature-set index file size
                                                            Long fileSizeBytes = indexFileSizes.get(features);
                                                            if (fileSizeBytes != null) {
                                                                allMetrics.put("Disk Usage (MB)", fileSizeBytes / 1024.0 / 1024.0);
                                                            }

                                                            results.add(new BenchResult(ds.getName(), params, allMetrics));
                                                        }
                                                    }
                                                }
                                            } catch (Exception e) {
                                                throw new RuntimeException(e);
                                            }
                                        } finally {
                                            for (int n = 0; n < 1; n++) {
                                                try {
                                                    Files.deleteIfExists(testDirectory.resolve("graph" + n));
                                                } catch (IOException e) { /* ignore */ }
                                            }
                                            try {
                                                Files.deleteIfExists(testDirectory);
                                            } catch (IOException e) { /* ignore */ }

                                            if (enableIndexCache) {
                                                System.out.println("Preserving index cache directory: " + indexCacheDir);
                                            }
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

    /** Overload for non-reporting use */
    private static VectorCompressor<?> getCompressor(Function<DataSet, CompressorParameters> cpSupplier, DataSet ds) {
        return getCompressor(cpSupplier, ds, null, null, null);
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
                return loadFromCache(ds, fname, path);
            }
            return executionWrapper.get();
        });
    }

    // Load/save helpers for getCompressor
    private static VectorCompressor<?> loadFromCache(DataSet ds, String fname, Path path) {
        try (var readerSupplier = ReaderSupplierFactory.open(path);
             var rar = readerSupplier.get()) {
            var pq = ProductQuantization.load(rar);
            System.out.format("%s: %s codebooks loaded from %s%n", ds.getName(), pq, fname);
            return pq;
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
        BaseDataSet ds;
        ImmutableGraphIndex index;
        CompressedVectors cv;
        Set<FeatureId> features;
        private final Supplier<RandomAccessVectorValues> exactVectors;

        private final ExplicitThreadLocal<GraphSearcher> searchers = ExplicitThreadLocal.withInitial(() -> {
            return new GraphSearcher(index);
        });

        ConfiguredSystem(BaseDataSet ds, ImmutableGraphIndex index, CompressedVectors cv, Set<FeatureId> features) {
            this.ds = ds;
            this.index = index;
            this.cv = cv;
            this.features = features;
            // File-backed sources borrow scratch storage; concurrent scorers need separate views.
            // Int8 datasets do not have float base vectors — leave exactVectors null.
            this.exactVectors = ds instanceof DataSet ? ((DataSet) ds).getBaseRavv().threadLocalSupplier() : null;
        }

        /** Score provider for float query vectors (float datasets). */
        public SearchScoreProvider scoreProviderFor(VectorFloat<?> queryVector, ImmutableGraphIndex.View view) {
            var floatDs = (DataSet) ds;
            var scoringView = (ImmutableGraphIndex.ScoringView) view;
            ScoreFunction.ApproximateScoreFunction asf;
            if (features.contains(FeatureId.FUSED_PQ)) {
                asf = scoringView.approximateScoreFunctionFor(queryVector, floatDs.getSimilarityFunction());
            } else {
                // if we're not compressing then just use the exact score function
                if (cv == null) {
                    return DefaultSearchScoreProvider.exact(queryVector, floatDs.getSimilarityFunction(), exactVectors.get());
                }

                asf = cv.precomputedScoreFunctionFor(queryVector, floatDs.getSimilarityFunction());
            }
            var rr = scoringView.rerankerFor(queryVector, floatDs.getSimilarityFunction());
            return new DefaultSearchScoreProvider(asf, rr);
        }

        /** Score provider for int8 byte query vectors (int8 datasets). */
        public SearchScoreProvider scoreProviderFor(ByteSequence<?> queryBytes, ImmutableGraphIndex.View view) {
            var byteDs = (ByteDataSet) ds;
            ByteVectorSimilarityFunction bvsf = byteDs.getSimilarityFunction();
            var diskView = (OnDiskGraphIndex.View) view;
            ScoreFunction.ExactScoreFunction sf = diskView.byteVectorRerankerFor(queryBytes, bvsf);
            return new DefaultSearchScoreProvider(sf);
        }

        public GraphSearcher getSearcher() {
            return searchers.get();
        }

        public BaseDataSet getDataSet() {
            return ds;
        }

        @Override
        public void close() throws Exception {
            searchers.close();
        }
    }

    // Helper that maps YAML-derived CompressorParameters to a stable label
    private static String quantTypeOf(CompressorParameters cp) {
        if (cp == null || cp == CompressorParameters.NONE) return null;
        if (cp instanceof CompressorParameters.PQParameters) return "PQ";
        if (cp instanceof CompressorParameters.BQParameters) return "BQ";
        // Unknown/unsupported type for quant timing purposes
        return null;
    }

    /**
     * Construction/search quantization timing metrics captured while building an index and while encoding
     * vectors for search-time evaluation.
     *
     * <p>Values may be {@code null} when not measured or not applicable (e.g., cache hit).</p>
     *
     * <p>Call {@link #appendTo(List)} to emit these as {@link Metric} entries so they can be selected for
     * console/CSV reporting.</p>
     */
    static final class ConstructionMetrics {
        // null means “not applicable / not measured”
        public Double indexBuildTimeS;
        public Long indexFileSizeBytes;

        // Index-construction phase (single pass per run)
        private final Map<String, QuantStats> indexByType = new HashMap<>();

        // Search phase (reset per evaluated config)
        private final Map<String, QuantStats> searchByType = new HashMap<>();

        ConstructionMetrics() {}

        /** Clear search-phase quant timings for the next evaluated (index, compression) configuration. */
        void resetSearch() {
            searchByType.clear();
        }

        QuantHandle index(String quantType)  { return handle(indexByType, quantType); }
        QuantHandle search(String quantType) { return handle(searchByType, quantType); }

        private static QuantHandle handle(Map<String, QuantStats> map, String quantType) {
            if (quantType == null) return QuantHandle.NOOP;
            QuantStats qs = map.computeIfAbsent(quantType, __ -> new QuantStats());
            return new QuantHandle(qs);
        }

        /** Adds construction + quant metrics as Metric entries (null => blank in CSV). */
        void appendTo(List<Metric> out) {
            if (indexBuildTimeS != null) {
                out.add(Metric.of("construction.index_build_time_s",
                        "Index build time (s)", ".2f", indexBuildTimeS));
            }
            if (indexFileSizeBytes != null) {
                out.add(Metric.of("construction.index_file_size_mb",
                        "Index file size (MB)", ".2f", indexFileSizeBytes / 1024.0 / 1024.0));
            }

            // Index-construction quant timings
            appendQuant(out, "construction.index_quant_time_s", "Idx", indexByType);

            // Search-phase quant timings (note: search.* namespace matches other query-time metrics)
            appendQuant(out, "search.search_quant_time_s", "Search", searchByType);
        }

        private static void appendQuant(List<Metric> out,
                                        String prefix,
                                        String phaseLabel,
                                        Map<String, QuantStats> byType) {
            final String hPrefix = (phaseLabel == null || phaseLabel.isBlank()) ? "" : (phaseLabel + " ");

            for (var e : byType.entrySet()) {
                String qt = e.getKey();
                QuantStats qs = e.getValue();

                // Compute
                if (qs.computeTimeS != null) {
                    out.add(Metric.of(prefix + "." + qt + ".compute_time_s",
                            hPrefix + qt + " compute (s)", ".2f", qs.computeTimeS));
                }

                // Encoding (usually single; suffix only if multiple)
                int n = qs.encodeTimesS.size();
                if (n == 1) {
                    out.add(Metric.of(prefix + "." + qt + ".encoding_time_s",
                            hPrefix + qt + " encoding (s)", ".2f", qs.encodeTimesS.get(0)));
                } else if (n > 1) {
                    for (int i = 0; i < n; i++) {
                        out.add(Metric.of(prefix + "." + qt + ".encoding_time_s." + i,
                                hPrefix + qt + " encoding (s) #" + i, ".2f", qs.encodeTimesS.get(i)));
                    }
                }
            }
        }

        private static final class QuantStats {
            Double computeTimeS;
            final List<Double> encodeTimesS = new ArrayList<>();
        }

        static final class QuantHandle {
            static final QuantHandle NOOP = new QuantHandle(null);

            private final QuantStats qs;

            private QuantHandle(QuantStats qs) {
                this.qs = qs;
            }

            <T> T timeCompute(java.util.function.Supplier<T> f) {
                if (qs == null) return f.get();
                long t0 = System.nanoTime();
                try { return f.get(); }
                finally { qs.computeTimeS = (System.nanoTime() - t0) / 1_000_000_000.0; }
            }

            <T> T timeEncode(java.util.function.Supplier<T> f) {
                if (qs == null) return f.get();
                long t0 = System.nanoTime();
                try { return f.get(); }
                finally { qs.encodeTimesS.add((System.nanoTime() - t0) / 1_000_000_000.0); }
            }
        }
    }
}
