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

package io.github.jbellis.jvector.bench;

import io.github.jbellis.jvector.example.reporting.GitInfo;
import io.github.jbellis.jvector.example.reporting.JfrRecorder;
import io.github.jbellis.jvector.example.reporting.JsonlWriter;
import io.github.jbellis.jvector.example.reporting.RunReporting;
import io.github.jbellis.jvector.example.reporting.SystemStatsCollector;
import io.github.jbellis.jvector.example.util.DataSetPartitioner;
import io.github.jbellis.jvector.example.util.storage.CloudStorageLayoutUtil;
import io.github.jbellis.jvector.example.yaml.TestDataPartition;
import io.github.jbellis.jvector.example.yaml.RunConfig;
import io.github.jbellis.jvector.bench.benchtools.BenchmarkParamCounter;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OnDiskParallelGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.AbstractGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.results.format.ResultFormatType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.OverlappingFileLockException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.IntFunction;
import java.util.stream.Stream;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(1)
@Warmup(iterations = 0)
@Measurement(iterations = 1)
@Threads(1)
public class CompactorBenchmark {

    // RUN_DIR must be initialized before the Logger so log4j2's File appender
    // can resolve ${sys:jvector.internal.runDir}
    private static final Path RUN_DIR;
    static {
        String runDir = System.getProperty("jvector.internal.runDir");
        if (runDir == null) {
            runDir = Path.of("target", "benchmark-results", "compactor-" + Instant.now().getEpochSecond()).toString();
            System.setProperty("jvector.internal.runDir", runDir);
        }
        RUN_DIR = Path.of(runDir);
        try {
            Files.createDirectories(RUN_DIR);
        } catch (IOException e) {
            throw new RuntimeException("Failed to create run directory: " + RUN_DIR, e);
        }
    }

    private static final Logger log = LoggerFactory.getLogger(CompactorBenchmark.class);

    public enum IndexPrecision {
        FULLPRECISION,
        FUSEDPQ
    }

    private static final Path RESULTS_FILE = RUN_DIR.resolve("compactor-results.jsonl");
    private static final Path JFR_DIR = RUN_DIR.resolve("jfrs");
    private static final Path SYSTEM_DIR = RUN_DIR.resolve("system");
    private static final JsonlWriter jsonlWriter = new JsonlWriter(RESULTS_FILE);
    // In the forked JVM, main() passes the computed total via this internal property
    private static final int TOTAL_TESTS = Integer.getInteger("jvector.internal.totalTests",
            BenchmarkParamCounter.computeTotalTests(CompactorBenchmark.class, null));
    private static final AtomicLong LAST_TEST_ID = new AtomicLong(0);
    private static final String TEST_ID = generateTestId();

    /**
     * Generates a lexicographically sortable test ID: base36-encoded milliseconds
     * followed by 2 base36 suffix chars (starting at "00").  Uses an atomic counter
     * so that IDs generated within the same millisecond auto-increment instead of colliding.
     */
    static String generateTestId() {
        long candidate = System.currentTimeMillis() * 1296; // suffix starts at 00
        long actual = LAST_TEST_ID.updateAndGet(last -> Math.max(candidate, last + 1));
        return Long.toString(actual, 36);
    }
    private static final Path COUNTER_FILE = RUN_DIR.resolve("completed-count");
    private static final AtomicInteger completedTests = new AtomicInteger(readCompletedCount());

    /**
     * Read the completed test count from a dedicated counter file.
     * Each JMH fork is a fresh JVM, so this file provides cross-fork continuity.
     * Acquires an exclusive file lock and throws if another process holds it,
     * since concurrent benchmark runs against the same RUN_DIR are not supported.
     */
    private static int readCompletedCount() {
        if (!Files.exists(COUNTER_FILE)) {
            return 0;
        }
        try (var ch = FileChannel.open(COUNTER_FILE, StandardOpenOption.READ)) {
            var lock = ch.tryLock(0, Long.MAX_VALUE, true);
            if (lock == null) {
                throw new IllegalStateException(
                        "Counter file is locked by another process — concurrent benchmark runs sharing "
                        + RUN_DIR + " are not supported");
            }
            try {
                return Integer.parseInt(Files.readString(COUNTER_FILE).trim());
            } finally {
                lock.release();
            }
        } catch (OverlappingFileLockException e) {
            throw new IllegalStateException(
                    "Counter file is locked by another thread — concurrent benchmark runs sharing "
                    + RUN_DIR + " are not supported", e);
        } catch (IllegalStateException e) {
            throw e;
        } catch (Exception e) {
            // Fall back to 0 for parse errors, etc.
            return 0;
        }
    }

    private static void writeCompletedCount(int count) {
        try (var ch = FileChannel.open(COUNTER_FILE,
                StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            var lock = ch.tryLock();
            if (lock == null) {
                throw new IllegalStateException(
                        "Counter file is locked by another process — concurrent benchmark runs sharing "
                        + RUN_DIR + " are not supported");
            }
            try {
                ch.write(ByteBuffer.wrap(String.valueOf(count).getBytes(StandardCharsets.UTF_8)));
            } finally {
                lock.release();
            }
        } catch (OverlappingFileLockException e) {
            throw new IllegalStateException(
                    "Counter file is locked by another thread — concurrent benchmark runs sharing "
                    + RUN_DIR + " are not supported", e);
        } catch (IllegalStateException e) {
            throw e;
        } catch (IOException e) {
            log.error("Failed to write completed count", e);
        }
    }
    private static final AtomicInteger workerCounter = new AtomicInteger(0);

    private RandomAccessVectorValues ravv;
    private List<VectorFloat<?>> queryVectors;
    private List<? extends List<Integer>> groundTruth;
    private DataSet ds;
    private final List<OnDiskGraphIndex> graphs = new ArrayList<>();
    private final List<ReaderSupplier> rss = new ArrayList<>();
    private VectorSimilarityFunction similarityFunction;
    private Path tempDir;
    private List<Integer> vectorsPerSourceCount;
    private String resolvedVectorizationProvider;

    @Param({"glove-100-angular"})
    public String datasetNames;

    @Param({"0", "1", "4"}) // Default value, can be overridden via command line
    public int numSources;

    @Param({"32"}) // Default value
    public int graphDegree;

    @Param({"100"}) // Default value
    public int beamWidth;

    @Param({""})
    public String storageDirectories;

    @Param({""})
    public String storageClasses;

    @Param({"UNIFORM"})
    public TestDataPartition.Distribution splitDistribution;

    @Param({"FULLPRECISION"})
    public IndexPrecision indexPrecision;

    @Param({"1"})
    public int parallelWriteThreads;

    @Param({""})
    public String vectorizationProvider;

    @Param({"1.0"})
    public double datasetPortion;

    @Param({"false"})
    public boolean jfrPartitioning;

    @Param({"true"})
    public boolean jfrCompacting;

    @Param({"false"})
    public boolean jfrObjectCount;

    @Param({"true"})
    public boolean sysStatsEnabled;

    private final JfrRecorder jfrPartitioningRecorder = new JfrRecorder();
    private final JfrRecorder jfrCompactingRecorder = new JfrRecorder();
    private final SystemStatsCollector sysStatsCollector = new SystemStatsCollector();

    private volatile boolean resultPersisted;
    private List<Path> storagePaths;

    @State(Scope.Thread)
    @AuxCounters(AuxCounters.Type.EVENTS)
    public static class RecallResult {
        public double recall;
    }

    private String jfrParamSuffix() {
        return String.format("%s-n%d-d%d-bw%d-%s-%s-pw%d-%s-dp%.2f",
                datasetNames, numSources, graphDegree, beamWidth,
                splitDistribution, indexPrecision, parallelWriteThreads, resolvedVectorizationProvider, datasetPortion);
    }

    @Setup(Level.Iteration)
    public void setup() throws Exception {
        try {
            resultPersisted = false;
            Thread.currentThread().setName("compactor-" + workerCounter.incrementAndGet());

            if (vectorizationProvider != null && !vectorizationProvider.isBlank()) {
                System.setProperty("jvector.vectorization_provider", vectorizationProvider);
            }
            resolvedVectorizationProvider = VectorizationProvider.getInstance().getClass().getSimpleName();

            if (sysStatsEnabled) {
                String sysStatsFileName = String.format("sysstats-%s-n%d-d%d-bw%d-%s-%s-pw%d-%s-dp%.2f.jsonl",
                        datasetNames, numSources, graphDegree, beamWidth,
                        splitDistribution, indexPrecision, parallelWriteThreads, resolvedVectorizationProvider, datasetPortion);
                try {
                    sysStatsCollector.start(SYSTEM_DIR, sysStatsFileName);
                } catch (Exception e) {
                    log.warn("Failed to start system stats collection", e);
                }
            }

            persistStarted();

            if (numSources < 0) {
                throw new IllegalArgumentException("numSources must be non-negative");
            }
            if (graphDegree <= 0) {
                throw new IllegalArgumentException("graphDegree must be positive");
            }
            if (beamWidth <= 0) {
                throw new IllegalArgumentException("beamWidth must be positive");
            }
            if (datasetPortion <= 0.0 || datasetPortion > 1.0) {
                throw new IllegalArgumentException("datasetPortion must be in (0.0, 1.0]");
            }

            DataSet ds = DataSets.loadDataSet(datasetNames).orElseThrow(() -> new RuntimeException("Dataset not found: " + datasetNames));

            List<VectorFloat<?>> baseVectors;
            if (datasetPortion == 1.0) {
                ravv = ds.getBaseRavv();
                baseVectors = ds.getBaseVectors();
            } else {
                int totalVectors = ds.getBaseRavv().size();
                int portionedSize = (int)(totalVectors * datasetPortion);
                if (portionedSize < numSources) {
                    throw new IllegalArgumentException(
                        "datasetPortion=" + datasetPortion + " yields " + portionedSize
                        + " vectors, fewer than numSources=" + numSources);
                }
                baseVectors = ds.getBaseVectors().subList(0, portionedSize);
                ravv = new ListRandomAccessVectorValues(baseVectors, ds.getDimension());
            }

            queryVectors = ds.getQueryVectors();
            groundTruth = ds.getGroundTruth();
            similarityFunction = ds.getSimilarityFunction();

            log.info("Dataset {} loaded. Base vectors: {} (portion {}), Query vectors: {}, Dimensions: {}, Similarity: {}",
                    datasetNames, ravv.size(), datasetPortion, queryVectors.size(), ds.getDimension(), similarityFunction);

            // Handle storage directories
            storagePaths = new ArrayList<>();
            if (storageDirectories != null && !storageDirectories.isBlank()) {
                for (String dir : storageDirectories.split(",")) {
                    Path path = Path.of(dir);
                    if (!Files.exists(path)) {
                        Files.createDirectories(path);
                    }
                    if (!Files.isDirectory(path) || !Files.isWritable(path)) {
                        throw new IllegalArgumentException("Path is not a writable directory: " + dir);
                    }
                    storagePaths.add(path);
                }
            } else {
                tempDir = Files.createTempDirectory("compact-bench");
                storagePaths.add(tempDir);
            }

            // Handle storage classes validation
            if (storageClasses != null && !storageClasses.isBlank()) {
                String[] classes = storageClasses.split(",");
                if (classes.length != storagePaths.size()) {
                    throw new IllegalArgumentException(String.format(
                            "Mismatch between number of storage classes (%d) and storage directories (%d). They must be pairwise 1:1.",
                            classes.length, storagePaths.size()));
                }

                var actualStorageClasses = CloudStorageLayoutUtil.storageClassByMountPoint();
                for (int i = 0; i < storagePaths.size(); i++) {
                    Path path = storagePaths.get(i).toAbsolutePath();
                    CloudStorageLayoutUtil.StorageClass expected;
                    try {
                        expected = CloudStorageLayoutUtil.StorageClass.valueOf(classes[i]);
                    } catch (IllegalArgumentException e) {
                        throw new IllegalArgumentException("Invalid StorageClass: " + classes[i], e);
                    }

                    // Find best matching mount point
                    String bestMount = null;
                    for (String mountPoint : actualStorageClasses.keySet()) {
                        if (path.toString().startsWith(mountPoint)) {
                            if (bestMount == null || mountPoint.length() > bestMount.length()) {
                                bestMount = mountPoint;
                            }
                        }
                    }

                    if (bestMount != null) {
                        CloudStorageLayoutUtil.StorageClass actual = actualStorageClasses.get(bestMount);
                        if (actual != expected) {
                            throw new IllegalStateException(String.format(
                                    "Storage class mismatch for path %s: expected %s, found %s (mount: %s)",
                                    path, expected, actual, bestMount));
                        }
                    } else {
                        log.warn("Could not determine storage class for path {}. Skipping validation.", path);
                    }
                }
            }

            // Clean up stale segment files from prior runs to avoid reading corrupt/mismatched headers
            for (Path dir : storagePaths) {
                try (var entries = Files.newDirectoryStream(dir, entry -> {
                    String name = entry.getFileName().toString();
                    return name.matches("per-source-graph-\\d+") || name.equals("compact-graph");
                })) {
                    for (Path stale : entries) {
                        log.info("Removing stale segment file: {}", stale.toAbsolutePath());
                        Files.delete(stale);
                    }
                }
            }

            int numParts = (numSources == 0) ? 1 : numSources;
            var partitionedData = DataSetPartitioner.partition(baseVectors, numParts, splitDistribution);
            vectorsPerSourceCount = partitionedData.sizes;
            log.info("Splitting dataset into {} segments (degree {}, beamWidth {}, splitDistribution {}, splitSizes {}, indexPrecision {}, parallelWriteThreads {}, vectorizationProvider {}, datasetPortion {}, jfrPartitioning {}, jfrCompacting {}, sysStatsEnabled {}) and building graphs...",
                    numParts, graphDegree, beamWidth, splitDistribution, vectorsPerSourceCount, indexPrecision, parallelWriteThreads, resolvedVectorizationProvider, datasetPortion, jfrPartitioning, jfrCompacting, sysStatsEnabled);

            if (jfrPartitioning) {
                jfrPartitioningRecorder.start(JFR_DIR, "partitioning-" + jfrParamSuffix() + ".jfr", jfrObjectCount);
            }

            for (int i = 0; i < numParts; i++) {
                List<VectorFloat<?>> vectorsPerSource = partitionedData.vectors.get(i);

                // Round-robin assignment of sources to storage paths
                Path baseDir = storagePaths.get(i % storagePaths.size());
                Path outputPath = baseDir.resolve("per-source-graph-" + i);
                log.info("Building and writing segment {}/{} to {}", i + 1, numParts, outputPath.toAbsolutePath());
                var ravvPerSource = new ListRandomAccessVectorValues(vectorsPerSource, ds.getDimension());
                BuildScoreProvider bspPerSource = BuildScoreProvider.randomAccessScoreProvider(ravvPerSource, similarityFunction);
                var builder = new GraphIndexBuilder(bspPerSource,
                        ds.getDimension(),
                        graphDegree, beamWidth, 1.2f, 1.2f, true);
                var graph = builder.build(ravvPerSource);

                // Build the on-disk writer with configurable features
                AbstractGraphIndexWriter.Builder<?, ?> writerBuilder;
                if (parallelWriteThreads > 1) {
                    writerBuilder = new OnDiskParallelGraphIndexWriter.Builder(graph, outputPath)
                            .withParallelWorkerThreads(parallelWriteThreads);
                } else {
                    writerBuilder = new OnDiskGraphIndexWriter.Builder(graph, outputPath);
                }

                writerBuilder.with(new InlineVectors(ds.getDimension()));

                ProductQuantization pq = null;
                PQVectors pqVectors = null;
                if (indexPrecision == IndexPrecision.FUSEDPQ) {
                    boolean centerData = similarityFunction == VectorSimilarityFunction.EUCLIDEAN;
                    pq = ProductQuantization.compute(ravvPerSource, ds.getDimension() / 8, 256, centerData);
                    pqVectors = (PQVectors) pq.encodeAll(ravvPerSource);
                    writerBuilder.with(new FusedPQ(graph.maxDegree(), pq));
                }

                try (var writer = writerBuilder.build()) {
                    var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
                    suppliers.put(FeatureId.INLINE_VECTORS, ordinal -> new InlineVectors.State(ravvPerSource.getVector(ordinal)));

                    if (indexPrecision == IndexPrecision.FUSEDPQ) {
                        var view = graph.getView();
                        var finalPqVectors = pqVectors;
                        suppliers.put(FeatureId.FUSED_PQ, ordinal -> new FusedPQ.State(view, finalPqVectors, ordinal));
                    }

                    writer.write(suppliers);
                }
            }
            log.info("Done building segments.");

            if (jfrPartitioningRecorder.isActive()) {
                jfrPartitioningRecorder.stop();
            }
        } catch (Exception e) {
            persistError(e);
            throw e;
        }
    }

    @TearDown(Level.Iteration)
    public void tearDown() throws IOException, InterruptedException {
        if (sysStatsCollector.isActive()) {
            sysStatsCollector.stop(SYSTEM_DIR);
        }

        if (jfrPartitioningRecorder.isActive()) {
            jfrPartitioningRecorder.stop();
        }
        if (jfrCompactingRecorder.isActive()) {
            jfrCompactingRecorder.stop();
        }

        for (var graph : graphs) {
            try {
                graph.close();
            } catch (Exception e) {
                log.error("Failed to close graph", e);
            }
        }
        graphs.clear();
        for (var rs : rss) {
            try {
                rs.close();
            } catch (Exception e) {
                log.error("Failed to close ReaderSupplier", e);
            }
        }
        rss.clear();

        // If we created a temp dir, clean it up entirely
        if (tempDir != null && Files.exists(tempDir)) {
            try (Stream<Path> walk = Files.walk(tempDir)) {
                walk.sorted(Comparator.reverseOrder())
                        .forEach(p -> {
                            try {
                                Files.delete(p);
                            } catch (IOException e) {
                                log.error("Failed to delete " + p, e);
                            }
                        });
            }
        } else if (storagePaths != null) {
            // Clean up artifacts from user-provided directories
            for (int i = 0; i < numSources; i++) {
                Path baseDir = storagePaths.get(i % storagePaths.size());
                Path sourcePath = baseDir.resolve("per-source-graph-" + i);
                if (Files.exists(sourcePath)) {
                    Files.delete(sourcePath);
                }
            }
        }
    }

    @Benchmark
    public void testCompactWithRandomQueryVectors(Blackhole blackhole, RecallResult recallResult) throws Exception {
        try {
            if (jfrCompacting) {
                try {
                    jfrCompactingRecorder.start(JFR_DIR, "compacting-" + jfrParamSuffix() + ".jfr", jfrObjectCount);
                } catch (Exception e) {
                    log.warn("Failed to start compacting JFR recording", e);
                }
            }

            int numParts = (numSources == 0) ? 1 : numSources;
            for (int i = 0; i < numParts; ++i) {
                Path baseDir = storagePaths.get(i % storagePaths.size());
                var outputPathPerSource = baseDir.resolve("per-source-graph-" + i);
                log.info("Reading segment {}/{} from {}", i + 1, numParts, outputPathPerSource.toAbsolutePath());
                rss.add(ReaderSupplierFactory.open(outputPathPerSource.toAbsolutePath()));
                var onDiskGraph = OnDiskGraphIndex.load(rss.get(i));
                graphs.add(onDiskGraph);
            }

            // Use the first storage path for the output compacted graph
            var outputPath = storagePaths.get(0).resolve("compact-graph");

            long durationMs = 0;
            if (numSources >= 1) {
                var compactor = new OnDiskGraphIndexCompactor(graphs);
                int globalOrdinal = 0;
                for (int n = 0; n < numSources; ++n) {
                    Map<Integer, Integer> map = new HashMap<>();
                    for (int i = 0; i < vectorsPerSourceCount.get(n); ++i) {
                        map.put(i, globalOrdinal++);
                    }
                    var remapper = new OrdinalMapper.MapMapper(map);
                    compactor.setRemapper(graphs.get(n), remapper);
                }
                log.info("Compacting {} segments into {}", numSources, outputPath.toAbsolutePath());
                long startNanos = System.nanoTime();
                compactor.compact(outputPath, similarityFunction);
                durationMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNanos);
            }

            Path searchPath = (numSources >= 1) ? outputPath : storagePaths.get(0).resolve("per-source-graph-0");
            log.info("Loading and searching index at {}", searchPath.toAbsolutePath());

            try (var rs = ReaderSupplierFactory.open(searchPath)) {
                var compactGraph = OnDiskGraphIndex.load(rs);
                List<SearchResult> compactedRetrieved = new ArrayList<>();
                for (int n = 0; n < queryVectors.size(); ++n) {
                    compactedRetrieved.add(GraphSearcher.search(queryVectors.get(n),
                            10,
                            ravv,
                            similarityFunction,
                            compactGraph,
                            Bits.ALL));
                }
                recallResult.recall = AccuracyMetrics.recallFromSearchResults(groundTruth, compactedRetrieved, 10, 10);
                log.info("Recall [dataset={}, numSources={}, graphDegree={}, beamWidth={}, splitDistribution={}, splitSizes={}, indexPrecision={}, parallelWriteThreads={}, vectorizationProvider={}, datasetPortion={}, jfrPartitioning={}, jfrCompacting={}, sysStatsEnabled={}]: {}",
                        datasetNames, numSources, graphDegree, beamWidth, splitDistribution, vectorsPerSourceCount, indexPrecision, parallelWriteThreads, resolvedVectorizationProvider, datasetPortion, jfrPartitioning, jfrCompacting, sysStatsEnabled, recallResult.recall);
                persistResult(recallResult.recall, durationMs);
                blackhole.consume(compactedRetrieved);
            } finally {
                if (jfrCompactingRecorder.isActive()) {
                    jfrCompactingRecorder.stop();
                }

                for (var graph : graphs) {
                    try {
                        graph.close();
                    } catch (Exception ignored) {
                    }
                }
                graphs.clear();
                for (var rs : rss) {
                    try {
                        rs.close();
                    } catch (Exception ignored) {
                    }
                }
                rss.clear();
                // Cleanup the output file
                if (numSources > 1 && Files.exists(outputPath)) {
                    Files.delete(outputPath);
                }
            }
        } catch (Exception e) {
            persistError(e);
            throw e;
        }
    }

    private LinkedHashMap<String, Object> buildParams() {
        var params = new LinkedHashMap<String, Object>();
        params.put("dataset", datasetNames);
        params.put("numSources", numSources);
        params.put("graphDegree", graphDegree);
        params.put("beamWidth", beamWidth);
        params.put("storageDirectories", storageDirectories);
        params.put("storageClasses", storageClasses);
        params.put("splitDistribution", splitDistribution.name());
        params.put("indexPrecision", indexPrecision.name());
        params.put("parallelWriteThreads", parallelWriteThreads);
        params.put("vectorizationProvider", resolvedVectorizationProvider);
        params.put("datasetPortion", datasetPortion);
        params.put("jfrPartitioning", jfrPartitioning);
        params.put("jfrCompacting", jfrCompacting);
        params.put("jfrObjectCount", jfrObjectCount);
        params.put("sysStatsEnabled", sysStatsEnabled);
        return params;
    }

    private LinkedHashMap<String, Object> baseResult(String event) {
        var result = new LinkedHashMap<String, Object>();
        result.put("testId", TEST_ID);
        result.put("gitHash", GitInfo.getShortHash());
        result.put("timestamp", Instant.now().toString());
        result.put("event", event);
        result.put("benchmark", "testCompactWithRandomQueryVectors");
        result.put("params", buildParams());
        return result;
    }

    private void persistStarted() {
        var result = baseResult("started");
        result.put("completedTests", completedTests.get());
        result.put("totalTests", TOTAL_TESTS);
        jsonlWriter.writeLine(result);
        log.info("Starting test {}/{}", completedTests.get() + 1, TOTAL_TESTS);
    }

    private void persistResult(double recall, long durationMs) {
        if (resultPersisted) return;
        resultPersisted = true;
        int completed = completedTests.incrementAndGet();
        writeCompletedCount(completed);
        var result = baseResult("completed");
        var results = new LinkedHashMap<String, Object>();
        results.put("recall", recall);
        results.put("durationMs", durationMs);
        if (vectorsPerSourceCount != null) {
            results.put("splitSizes", vectorsPerSourceCount.toString());
        }
        if (jfrPartitioningRecorder.getFileName() != null) {
            results.put("jfrPartitioningFile", jfrPartitioningRecorder.getFileName());
        }
        if (jfrCompactingRecorder.getFileName() != null) {
            results.put("jfrCompactingFile", jfrCompactingRecorder.getFileName());
        }
        if (sysStatsCollector.getFileName() != null) {
            results.put("sysStatsFile", sysStatsCollector.getFileName());
        }
        result.put("results", results);
        result.put("completedTests", completed);
        result.put("totalTests", TOTAL_TESTS);
        jsonlWriter.writeLine(result);
        log.info("Completed test {}/{}", completed, TOTAL_TESTS);
    }

    private void persistError(Exception e) {
        try {
            var result = baseResult("error");
            var results = new LinkedHashMap<String, Object>();
            results.put("errorMessage", e.getMessage() != null ? e.getMessage() : e.getClass().getName());
            result.put("results", results);
            result.put("completedTests", completedTests.get());
            result.put("totalTests", TOTAL_TESTS);
            jsonlWriter.writeLine(result);
        } catch (Exception inner) {
            log.error("Failed to persist error event", inner);
        }
    }

    public static void main(String[] args) throws Exception {
        // Initialize RunReporting if not already in a run directory
        // This produces sys_info.json and sets up the standard directory structure
        if (System.getProperty("jvector.internal.runDir") == null) {
            RunConfig runCfg = new RunConfig();
            runCfg.logging = new RunConfig.RunLogging();
            runCfg.logging.dir = "target/benchmark-results";
            runCfg.logging.runId = "compactor-{ts}";
            runCfg.logging.type = "csv"; // Enable artifacts

            var reporting = RunReporting.open(runCfg);
            System.setProperty("jvector.internal.runDir", reporting.run().runDir().toString());
        }

        Path runDir = Path.of(System.getProperty("jvector.internal.runDir"));
        Files.createDirectories(runDir);
        String jmhResultFile = runDir.resolve("compactor-jmh.json").toString();
        log.info("Benchmark run directory: {}", runDir.toAbsolutePath());
        log.info("Progressive results will be written to: {}", runDir.resolve("compactor-results.jsonl").toAbsolutePath());
        log.info("JMH results will be written to: {}", Path.of(jmhResultFile).toAbsolutePath());

        org.openjdk.jmh.runner.options.CommandLineOptions cmdOptions = new org.openjdk.jmh.runner.options.CommandLineOptions(args);
        int totalTests = BenchmarkParamCounter.computeTotalTests(CompactorBenchmark.class, cmdOptions);
        log.info("Total test combinations: {}", totalTests);

        // Resolve the log4j2 config so the forked JVM picks it up explicitly
        var log4j2Config = CompactorBenchmark.class.getClassLoader().getResource("log4j2.xml");
        String log4j2Arg = log4j2Config != null
                ? "-Dlog4j2.configurationFile=" + log4j2Config
                : "-Dlog4j2.configurationFile=classpath:log4j2.xml";

        // The forked JVM's stdout is piped through JMH, so System.console() returns null
        // and Log4j2 suppresses ANSI. Propagate the parent's TTY detection to the child.
        String disableAnsi = System.console() == null ? "true" : "false";

        // Collect all JVM args for the forked process in one list,
        // because jvmArgsAppend() replaces (not appends) on each call.
        var jvmArgs = new ArrayList<String>();
        jvmArgs.add("-Djvector.internal.runDir=" + runDir);
        jvmArgs.add("-Djvector.internal.totalTests=" + totalTests);
        jvmArgs.add(log4j2Arg);
        jvmArgs.add("-Dcompactor.disableAnsi=" + disableAnsi);

        // Pass the vectorization provider if specified in command line options
        var vpParam = cmdOptions.getParameter("vectorizationProvider");
        if (vpParam.hasValue()) {
            var vpValues = vpParam.get();
            if (!vpValues.isEmpty()) {
                jvmArgs.add("-Djvector.vectorization_provider=" + vpValues.iterator().next());
            }
        }

        var optBuilder = new org.openjdk.jmh.runner.options.OptionsBuilder();
        optBuilder.include(CompactorBenchmark.class.getSimpleName())
                .parent(cmdOptions)
                .forks(1)
                .threads(1)
                .shouldFailOnError(true)
                .jvmArgsAppend(jvmArgs.toArray(new String[0]))
                .resultFormat(ResultFormatType.JSON)
                .result(jmhResultFile);

        new org.openjdk.jmh.runner.Runner(optBuilder.build()).run();
    }
}
