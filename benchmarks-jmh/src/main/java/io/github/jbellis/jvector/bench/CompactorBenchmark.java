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

import io.github.jbellis.jvector.bench.benchtools.BenchmarkParamCounter;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.example.reporting.GitInfo;
import io.github.jbellis.jvector.example.reporting.JfrRecorder;
import io.github.jbellis.jvector.example.reporting.JsonlWriter;
import io.github.jbellis.jvector.example.reporting.SystemStatsCollector;
import io.github.jbellis.jvector.example.reporting.ThreadAllocTracker;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.example.util.DataSetPartitioner;
import io.github.jbellis.jvector.example.util.storage.CloudStorageLayoutUtil;
import io.github.jbellis.jvector.example.yaml.TestDataPartition;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.disk.AbstractGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OnDiskParallelGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
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
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.IntFunction;

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

    public enum WorkloadMode {
        /**
         * Build per-source partitions and stop. (No compaction, no recall.)
         */
        PARTITION_ONLY,

        /**
         * Assume partitions exist on disk; compact them.
         */
        COMPACT_ONLY,

        /**
         * Assume partitions exist on disk; compact them, then run recall.
         */
        COMPACT_AND_RECALL,

        /**
         * Build a single graph for the whole dataset and write it. Then run recall.
         */
        BUILD_FROM_SCRATCH,
        /**
         * (Default) Build partitions, compact them, then run recall.
         */
        PARTITION_AND_COMPACT
    }

    private static final Path RESULTS_FILE = RUN_DIR.resolve("compactor-results.jsonl");
    private static final Path JFR_DIR = RUN_DIR.resolve("jfrs");
    private static final Path SYSTEM_DIR = RUN_DIR.resolve("system");
    private static final JsonlWriter jsonlWriter = new JsonlWriter(RESULTS_FILE);

    // In the forked JVM, main() passes the computed total via this internal property
    private static final int TOTAL_TESTS = Integer.getInteger(
            "jvector.internal.totalTests",
            BenchmarkParamCounter.computeTotalTests(CompactorBenchmark.class, null)
    );

    private static final AtomicLong LAST_TEST_ID = new AtomicLong(0);
    private static final String TEST_ID = generateTestId();

    /**
     * Generates a lexicographically sortable test ID: base36-encoded milliseconds
     * followed by 2 base36 suffix chars (starting at "00"). Uses an atomic counter
     * so that IDs generated within the same millisecond auto-increment instead of colliding.
     */
    static String generateTestId() {
        long candidate = System.currentTimeMillis() * 1296; // suffix starts at 00
        long actual = LAST_TEST_ID.updateAndGet(last -> Math.max(candidate, last + 1));
        return Long.toString(actual, 36);
    }

    /**
    * Returns a Bits instance representing randomly selected live nodes.
    *
    */
    private static FixedBitSet randomLiveNodes(int size, double liveRate, long seed) {
        FixedBitSet live = new FixedBitSet(size);

        if (liveRate >= 1.0) {
            live.set(0, size); // all nodes live
            return live;
        }

        var rnd = new java.util.SplittableRandom(seed);
        int liveCount = 0;

        for (int i = 0; i < size; i++) {
            if (rnd.nextDouble() < liveRate) {
                live.set(i);
                liveCount++;
            }
        }

        // avoid degenerate case (all dead)
        if (liveCount == 0 && size > 0) {
            live.set(rnd.nextInt(size));
        }

        return live;
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

    // ---------- Benchmark state ----------
    private RandomAccessVectorValues ravv;
    private List<VectorFloat<?>> queryVectors;
    private List<VectorFloat<?>> baseVectors;
    private List<? extends List<Integer>> groundTruth;
    private DataSet ds;
    private VectorSimilarityFunction similarityFunction;

    private final List<OnDiskGraphIndex> graphs = new ArrayList<>();
    private final List<ReaderSupplier> rss = new ArrayList<>();

    private Path tempDir;
    private List<Path> storagePaths;
    private List<Integer> vectorsPerSourceCount;
    private String resolvedVectorizationProvider;

    // Paths used during execution
    private Path partitionsBaseDir;       // where per-source partitions are placed (or found)
    private Path compactOutputPath;     // where compacted graph is written
    private Path scratchOutputPath;     // where build-from-scratch graph is written

    // ---------- Params ----------
    @Param({"glove-100-angular"})
    public String datasetNames;

    @Param({"PARTITION_AND_COMPACT"})
    public WorkloadMode workloadMode;

    @Param({"4"}) // Default value, can be overridden via command line
    public int numPartitions;

    @Param({"32"})
    public int graphDegree;

    @Param({"100"})
    public int beamWidth;

    /**
     * liveNodesRate controls how many nodes are considered "live" per source partition
     * when calling compactor.setLiveNodes(...).
     *
     * - 1.0 => all nodes live (default; behaves like no deletions)
     * - 0.8 => ~80% live, ~20% deleted (randomly selected)
    */
    @Param({"1.0"})
    public double liveNodesRate;

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

    @Param({"false"})
    public boolean threadAllocTracking;

    private final JfrRecorder jfrPartitioningRecorder = new JfrRecorder();
    private final JfrRecorder jfrCompactingRecorder = new JfrRecorder();
    private final SystemStatsCollector sysStatsCollector = new SystemStatsCollector();
    private final ThreadAllocTracker threadAllocTracker = new ThreadAllocTracker();

    private volatile boolean resultPersisted;

    @State(Scope.Thread)
    @AuxCounters(AuxCounters.Type.EVENTS)
    public static class RecallResult {
        public double recall;
    }

    private String jfrParamSuffix() {
        return String.format("%s-w%s-n%d-d%d-bw%d-%s-%s-pw%d-%s-dp%.2f-live%.2f",
                datasetNames, workloadMode, numPartitions, graphDegree, beamWidth,
                splitDistribution, indexPrecision, parallelWriteThreads, resolvedVectorizationProvider, datasetPortion, liveNodesRate);
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
                String sysStatsFileName = String.format("sysstats-%s.jsonl", jfrParamSuffix());
                try {
                    sysStatsCollector.start(SYSTEM_DIR, sysStatsFileName);
                } catch (Exception e) {
                    log.warn("Failed to start system stats collection", e);
                }
            }

            if (threadAllocTracking) {
                String threadAllocFileName = String.format("threadalloc-%s.jsonl", jfrParamSuffix());
                try {
                    threadAllocTracker.start(SYSTEM_DIR, threadAllocFileName);
                } catch (Exception e) {
                    log.warn("Failed to start thread allocation tracking", e);
                }
            }

            persistStarted();

            validateParams();

            int dimension;

            if (workloadMode == WorkloadMode.COMPACT_ONLY) {
                ds = null;
                queryVectors = null;
                groundTruth = null;
                ravv = null;
                similarityFunction = VectorSimilarityFunction.COSINE;
                baseVectors = null;
                dimension = -1;

                log.info("Skipping dataset load for COMPACT_ONLY mode without recall. Workload: {}, Live nodes rate: {}",
                        workloadMode, liveNodesRate);
            } else {
                ds = DataSets.loadDataSet(datasetNames)
                        .orElseThrow(() -> new RuntimeException("Dataset not found: " + datasetNames))
                        .getDataSet();

                if (datasetPortion == 1.0) {
                    ravv = ds.getBaseRavv();
                    baseVectors = ds.getBaseVectors();
                } else {
                    int totalVectors = ds.getBaseRavv().size();
                    int portionedSize = (int) (totalVectors * datasetPortion);
                    if (portionedSize < Math.max(1, numPartitions)) {
                        throw new IllegalArgumentException(
                                "datasetPortion=" + datasetPortion + " yields " + portionedSize
                                        + " vectors, fewer than numPartitions=" + numPartitions);
                    }
                    baseVectors = ds.getBaseVectors().subList(0, portionedSize);
                    ravv = new ListRandomAccessVectorValues(baseVectors, ds.getDimension());
                }

                queryVectors = ds.getQueryVectors();
                groundTruth = ds.getGroundTruth();
                similarityFunction = ds.getSimilarityFunction();
                dimension = ds.getDimension();

                log.info("Dataset {} loaded with recall data. Base vectors: {} (portion {}), Query vectors: {}, Dim: {}, Similarity: {}, Workload: {}, Live nodes rate: {}",
                        datasetNames, ravv.size(), datasetPortion, queryVectors.size(), dimension, similarityFunction, workloadMode, liveNodesRate);
            }

            // Resolve storagePaths + partitionsDir
            storagePaths = resolveStoragePaths();
            partitionsBaseDir = resolvePartitionsBaseDir(storagePaths);
            compactOutputPath = resolveCompactOutputPath(partitionsBaseDir);
            scratchOutputPath = resolveScratchOutputPath(partitionsBaseDir);

            // Clean stale artifacts only if we're going to rebuild them.
            if (workloadMode == WorkloadMode.COMPACT_ONLY || workloadMode == WorkloadMode.COMPACT_AND_RECALL) {
                // For compact-only and compact-and-recall, ensure the partition files exist.
                verifyPartitionsExist(partitionsBaseDir, numPartitions);
            }

            // Partition metadata for remapping (needed for compaction)
            if (workloadMode == WorkloadMode.PARTITION_ONLY || workloadMode == WorkloadMode.PARTITION_AND_COMPACT) {
                var partitionedData = DataSetPartitioner.partition(baseVectors, numPartitions, splitDistribution);
                vectorsPerSourceCount = partitionedData.sizes;
            } else {
                vectorsPerSourceCount = null;
            }

            // Build partitions during setup for SEGMENTS_* (matches original benchmark structure)
            if (workloadMode == WorkloadMode.PARTITION_ONLY || workloadMode == WorkloadMode.PARTITION_AND_COMPACT) {
                if (jfrPartitioning) {
                    jfrPartitioningRecorder.start(JFR_DIR, "partitioning-" + jfrParamSuffix() + ".jfr", jfrObjectCount);
                }
                buildPartitions(ds, baseVectors);
                if (jfrPartitioningRecorder.isActive()) {
                    jfrPartitioningRecorder.stop();
                }
            }

        } catch (Exception e) {
            persistError(e);
            throw e;
        }
    }

    private void validateParams() {
        if (workloadMode == WorkloadMode.BUILD_FROM_SCRATCH) {
            log.warn("numPartitions={} ignored in BUILD_FROM_SCRATCH mode", numPartitions);
        }
        else {
           if (numPartitions <= 1) throw new IllegalArgumentException("numPartitions must be larger than one");
        }
        if (graphDegree <= 0) throw new IllegalArgumentException("graphDegree must be positive");
        if (beamWidth <= 0) throw new IllegalArgumentException("beamWidth must be positive");
        if (datasetPortion <= 0.0 || datasetPortion > 1.0) {
            throw new IllegalArgumentException("datasetPortion must be in (0.0, 1.0]");
        }
        if (liveNodesRate <= 0.0 || liveNodesRate > 1.0) {
            throw new IllegalArgumentException("liveNodesRate must be in (0.0, 1.0]");
        }
    }

    private List<Path> resolveStoragePaths() throws IOException {
        // Priority:
        // 1) storageDirectories (comma-separated)
        // 3) temp dir
        var paths = new ArrayList<Path>();

        if (storageDirectories != null && !storageDirectories.isBlank()) {
            for (String dir : storageDirectories.split(",")) {
                Path path = Path.of(dir.trim());
                if (!Files.exists(path)) Files.createDirectories(path);
                if (!Files.isDirectory(path) || !Files.isWritable(path)) {
                    throw new IllegalArgumentException("Path is not a writable directory: " + dir);
                }
                paths.add(path);
            }
        } else {
            tempDir = Files.createTempDirectory("compact-bench");
            paths.add(tempDir);
        }

        // Handle storage class validation
        if (storageClasses != null && !storageClasses.isBlank()) {
            String[] classes = storageClasses.split(",");
            if (classes.length != paths.size()) {
                throw new IllegalArgumentException(String.format(
                        "Mismatch between number of storage classes (%d) and storage directories (%d). They must be pairwise 1:1.",
                        classes.length, paths.size()));
            }

            var actualStorageClasses = CloudStorageLayoutUtil.storageClassByMountPoint();
            for (int i = 0; i < paths.size(); i++) {
                Path path = paths.get(i).toAbsolutePath();
                CloudStorageLayoutUtil.StorageClass expected;
                try {
                    expected = CloudStorageLayoutUtil.StorageClass.valueOf(classes[i].trim());
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException("Invalid StorageClass: " + classes[i], e);
                }

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

        return paths;
    }

    private Path resolvePartitionsBaseDir(List<Path> storagePaths) throws IOException {
        Path p = storagePaths.get(0);
        Files.createDirectories(p);
        return p;
    }

    private Path resolveCompactOutputPath(Path baseDir) {
        return baseDir.resolve("compact-graph");
    }

    private Path resolveScratchOutputPath(Path baseDir) {
        return baseDir.resolve("scratch-graph");
    }

    private void verifyPartitionsExist(Path partitionsDir, int numPartitions) {
        for (int i = 0; i < numPartitions; i++) {
            Path seg = partitionsDir.resolve("per-source-graph-" + i);
            if (!Files.exists(seg)) {
                throw new IllegalStateException("Missing partition file for COMPACT_ONLY or COMPACT_AND_RECALL: " + seg.toAbsolutePath());
            }
        }
    }

    private void buildPartitions(DataSet ds, List<VectorFloat<?>> baseVectors) throws Exception {

        var partitionedData = DataSetPartitioner.partition(baseVectors, numPartitions, splitDistribution);
        vectorsPerSourceCount = partitionedData.sizes;

        log.info("Building {} partitions into {} (deg={}, bw={}, split={}, splitSizes={}, precision={}, pwThreads={}, vp={})",
                numPartitions, partitionsBaseDir.toAbsolutePath(), graphDegree, beamWidth, splitDistribution, vectorsPerSourceCount,
                indexPrecision, parallelWriteThreads, resolvedVectorizationProvider);

        int dimension = baseVectors.get(0).length();
        for (int i = 0; i < numPartitions; i++) {
            List<VectorFloat<?>> vectorsPerSource = partitionedData.vectors.get(i);

            // Round-robin assignment of partition files to storage paths, but still keep canonical base dir name stable.
            Path baseDirForThisSegment = storagePaths.get(i % storagePaths.size());
            Path outputPath = baseDirForThisSegment.resolve("per-source-graph-" + i);
            if (Files.exists(outputPath)) {
                Files.delete(outputPath);
            }

            log.info("Building partition {}/{}: vectors={} -> {}",
                    i + 1, numPartitions, vectorsPerSource.size(), outputPath.toAbsolutePath());

            var ravvPerSource = new ListRandomAccessVectorValues(vectorsPerSource, dimension);
            BuildScoreProvider bspPerSource;
            ProductQuantization pq = null;
            PQVectors pqVectors = null;
            // TODO: should we build partitions by FUSEDPQ?
            if (indexPrecision == IndexPrecision.FUSEDPQ) {
                boolean centerData = similarityFunction == VectorSimilarityFunction.EUCLIDEAN;
                pq = ProductQuantization.compute(ravvPerSource, dimension / 8, 256, centerData);
                pqVectors = (PQVectors) pq.encodeAll(ravvPerSource);
                bspPerSource = BuildScoreProvider.pqBuildScoreProvider(similarityFunction, pqVectors);
            }
            else {
                bspPerSource = BuildScoreProvider.randomAccessScoreProvider(ravvPerSource, similarityFunction);
            }

            var builder = new GraphIndexBuilder(bspPerSource,
                    dimension,
                    graphDegree, beamWidth, 1.2f, 1.2f, true);
            var graph = builder.build(ravvPerSource);

            AbstractGraphIndexWriter.Builder<?, ?> writerBuilder;
            if (parallelWriteThreads > 1) {
                writerBuilder = new OnDiskParallelGraphIndexWriter.Builder(graph, outputPath)
                        .withParallelWorkerThreads(parallelWriteThreads);
            } else {
                writerBuilder = new OnDiskGraphIndexWriter.Builder(graph, outputPath);
            }

            writerBuilder.with(new InlineVectors(dimension));


            if (indexPrecision == IndexPrecision.FUSEDPQ) {
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

        log.info("Done building partitions.");
    }

    private long compactPartitions() throws Exception {

        // Load partitions (from round-robin storage paths, same naming)
        for (int i = 0; i < numPartitions; i++) {
            Path baseDir = storagePaths.get(i % storagePaths.size());
            Path segPath = baseDir.resolve("per-source-graph-" + i);
            log.info("Loading partition {}/{} from {}", i + 1, numPartitions, segPath.toAbsolutePath());
            rss.add(ReaderSupplierFactory.open(segPath.toAbsolutePath()));
            graphs.add(OnDiskGraphIndex.load(rss.get(i)));
        }

        // Ensure output dir exists
        if (compactOutputPath.getParent() != null) {
            Files.createDirectories(compactOutputPath.getParent());
        }

        if (Files.exists(compactOutputPath)) {
            Files.delete(compactOutputPath);
        }

        log.info("Compacting {} partitions into {}", numPartitions, compactOutputPath.toAbsolutePath());


        List<OrdinalMapper> remappers = new ArrayList<>(numPartitions);
        List<FixedBitSet> liveNodes = new ArrayList<>(numPartitions);
        // Remap ordinals: local [0..size-1] -> global increasing in partition order
        int globalOrdinal = 0;
        for (int n = 0; n < numPartitions; n++) {
            int size = graphs.get(n).size();
            var remapper = new OrdinalMapper.OffsetMapper(globalOrdinal, size);
            remappers.add(remapper);
            liveNodes.add(randomLiveNodes(size, liveNodesRate, n));
            globalOrdinal += size;
        }
        var compactor = new OnDiskGraphIndexCompactor(graphs, liveNodes, remappers, similarityFunction, null);

        long startNanos = System.nanoTime();
        compactor.compact(compactOutputPath);
        return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNanos);
    }

    private long buildFromScratch(List<VectorFloat<?>> baseVectors) throws Exception {
        if (scratchOutputPath.getParent() != null) {
            Files.createDirectories(scratchOutputPath.getParent());
        }
        if (Files.exists(scratchOutputPath)) {
            Files.delete(scratchOutputPath);
        }

        int dimension = baseVectors.get(0).length();
        var full = new ListRandomAccessVectorValues(baseVectors, dimension);
        ProductQuantization pq = null;
        PQVectors pqVectors = null;
        BuildScoreProvider bsp;
        if (indexPrecision == IndexPrecision.FUSEDPQ) {
            boolean centerData = similarityFunction == VectorSimilarityFunction.EUCLIDEAN;
            pq = ProductQuantization.compute(full, dimension / 8, 256, centerData);
            pqVectors = (PQVectors) pq.encodeAll(full);
            bsp = BuildScoreProvider.pqBuildScoreProvider(similarityFunction, pqVectors);
        }
        else {
            bsp = BuildScoreProvider.randomAccessScoreProvider(full, similarityFunction);
        }

        log.info("Building from scratch: vectors={} dim={} sim={} deg={} bw={} precision={} pwThreads={} vp={} -> {}",
                full.size(), dimension, similarityFunction,
                graphDegree, beamWidth, indexPrecision, parallelWriteThreads, resolvedVectorizationProvider,
                scratchOutputPath.toAbsolutePath());

        var builder = new GraphIndexBuilder(bsp, dimension, graphDegree, beamWidth, 1.2f, 1.2f, true);
        var graph = builder.build(full);

        AbstractGraphIndexWriter.Builder<?, ?> writerBuilder =
                (parallelWriteThreads > 1)
                        ? new OnDiskParallelGraphIndexWriter.Builder(graph, scratchOutputPath)
                        .withParallelWorkerThreads(parallelWriteThreads)
                        : new OnDiskGraphIndexWriter.Builder(graph, scratchOutputPath);

        writerBuilder.with(new InlineVectors(dimension));

//        ProductQuantization pq = null;
//        PQVectors pqVectors = null;
//        if (indexPrecision == IndexPrecision.FUSEDPQ) {
//            boolean centerData = similarityFunction == VectorSimilarityFunction.EUCLIDEAN;
//            pq = ProductQuantization.compute(full, dimension / 8, 256, centerData);
//            pqVectors = (PQVectors) pq.encodeAll(full);
//            writerBuilder.with(new FusedPQ(graph.maxDegree(), pq));
//        }
        if (indexPrecision == IndexPrecision.FUSEDPQ) {
            writerBuilder.with(new FusedPQ(graph.maxDegree(), pq));
        }

        long startNanos = System.nanoTime();
        try (var writer = writerBuilder.build()) {
            var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
            suppliers.put(FeatureId.INLINE_VECTORS, ord -> new InlineVectors.State(full.getVector(ord)));

            if (indexPrecision == IndexPrecision.FUSEDPQ) {
                var view = graph.getView();
                var finalPQ = pqVectors;
                suppliers.put(FeatureId.FUSED_PQ, ord -> new FusedPQ.State(view, finalPQ, ord));
            }

            writer.write(suppliers);
        }
        return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNanos);
    }

    @TearDown(Level.Iteration)
    public void tearDown() throws IOException, InterruptedException {
        if (threadAllocTracker.isActive()) {
            threadAllocTracker.stop();
        }

        if (sysStatsCollector.isActive()) {
            sysStatsCollector.stop(SYSTEM_DIR);
        }

        if (jfrPartitioningRecorder.isActive()) {
            jfrPartitioningRecorder.stop();
        }
        if (jfrCompactingRecorder.isActive()) {
            jfrCompactingRecorder.stop();
        }

        closeLoadedGraphs();
    }

    private void closeLoadedGraphs() {
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
    }

    @Benchmark
    public void run(Blackhole blackhole, RecallResult recallResult) throws Exception {
        long durationMs = 0;
        double recall = -1;

        try {
            if (jfrCompacting) {
                try {
                    jfrCompactingRecorder.start(JFR_DIR, "workload-" + jfrParamSuffix() + ".jfr", jfrObjectCount);
                } catch (Exception e) {
                    log.warn("Failed to start workload JFR recording", e);
                }
            }

            // Execute workload
            switch (workloadMode) {
                case PARTITION_ONLY:
                    break;

                case COMPACT_ONLY:
                    durationMs = compactPartitions();
                    break;

                case COMPACT_AND_RECALL:
                    durationMs = compactPartitions();
                    recall = runRecall(compactOutputPath);
                    break;

                case BUILD_FROM_SCRATCH: {
                    durationMs = buildFromScratch(baseVectors);
                    recall = runRecall(scratchOutputPath);
                    break;
                }

                case PARTITION_AND_COMPACT:
                    durationMs = compactPartitions();
                    recall = runRecall(compactOutputPath);
                    break;

                default:
                    throw new IllegalStateException("Unknown workloadMode: " + workloadMode);
            }

            recallResult.recall = recall;
            persistResult(recall, durationMs);
            blackhole.consume(durationMs);

        } catch (Exception e) {
            persistError(e);
            throw e;
        } finally {
            if (jfrCompactingRecorder.isActive()) {
                jfrCompactingRecorder.stop();
            }
            closeLoadedGraphs();
        }
    }

    private double runRecall(Path indexPath) throws Exception {

        log.info("Loading and searching index at {}", indexPath.toAbsolutePath());
        try (var rs = ReaderSupplierFactory.open(indexPath)) {
            var graph = OnDiskGraphIndex.load(rs);
            GraphSearcher searcher = new GraphSearcher(graph);
            var view = (ImmutableGraphIndex.ScoringView) searcher.getView();
            searcher.usePruning(false);
            List<SearchResult> retrieved = new ArrayList<>(queryVectors.size());
            for (int n = 0; n < queryVectors.size(); ++n) {
                SearchResult result;
                if(indexPrecision == IndexPrecision.FUSEDPQ) {
                    var asf = view.approximateScoreFunctionFor(queryVectors.get(n), similarityFunction);
                    var rerank = view.rerankerFor(queryVectors.get(n), similarityFunction);
                    SearchScoreProvider ssp = new DefaultSearchScoreProvider(asf, rerank);
                    result = searcher.search(ssp, 10, 10, 0.0f, 0.0f, Bits.ALL);
                }
                else {
                    var ssp = DefaultSearchScoreProvider.exact(queryVectors.get(n), similarityFunction, ravv);
                    result = searcher.search(ssp, 10, 10, 0.0f, 0.0f, Bits.ALL);
                }
                retrieved.add(result);
            }

            double recall = AccuracyMetrics.recallFromSearchResults(groundTruth, retrieved, 10, 10);
            log.info("Recall [dataset={}, workloadMode={}, numPartitions={}, graphDegree={}, beamWidth={}, splitDistribution={}, indexPrecision={}, parallelWriteThreads={}, vectorizationProvider={}, datasetPortion={}]: {}",
                    datasetNames, workloadMode, numPartitions, graphDegree, beamWidth, splitDistribution, indexPrecision, parallelWriteThreads, resolvedVectorizationProvider, datasetPortion, recall);
            return recall;
        }
    }

    // ---------- result persistence ----------
    private LinkedHashMap<String, Object> buildParams() {
        var params = new LinkedHashMap<String, Object>();
        params.put("dataset", datasetNames);
        params.put("workloadMode", workloadMode.name());
        params.put("numPartitions", numPartitions);
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
        params.put("threadAllocTracking", threadAllocTracking);
        params.put("liveNodesRate", liveNodesRate);
        return params;
    }

    private LinkedHashMap<String, Object> baseResult(String event) {
        var result = new LinkedHashMap<String, Object>();
        result.put("testId", TEST_ID);
        result.put("gitHash", GitInfo.getShortHash());
        result.put("timestamp", Instant.now().toString());
        result.put("event", event);
        result.put("benchmark", "run");
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
        results.put("durationMs", durationMs);

        // Only meaningful for recall-enabled workloads; else NaN
        results.put("recall", recall);

        if (vectorsPerSourceCount != null) {
            results.put("splitSizes", vectorsPerSourceCount.toString());
        }
        if (jfrPartitioningRecorder.getFileName() != null) {
            results.put("jfrPartitioningFile", jfrPartitioningRecorder.getFileName());
        }
        if (jfrCompactingRecorder.getFileName() != null) {
            results.put("jfrWorkloadFile", jfrCompactingRecorder.getFileName());
        }
        if (sysStatsCollector.getFileName() != null) {
            results.put("sysStatsFile", sysStatsCollector.getFileName());
        }
        if (threadAllocTracker.getFileName() != null) {
            results.put("threadAllocFile", threadAllocTracker.getFileName());
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
        Files.createDirectories(RUN_DIR);
        String jmhResultFile = RUN_DIR.resolve("compactor-jmh.json").toString();
        log.info("Benchmark run directory: {}", RUN_DIR.toAbsolutePath());
        log.info("Progressive results will be written to: {}", RESULTS_FILE.toAbsolutePath());
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
        jvmArgs.add("-Djvector.internal.runDir=" + RUN_DIR);
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
