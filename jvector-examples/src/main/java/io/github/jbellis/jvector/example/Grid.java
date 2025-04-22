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

import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.benchmarks.*;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.SubsetRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedADC;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.ImmutablePQVectors;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.DirectoryNotEmptyException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.LinkedBlockingQueue;


import java.time.LocalTime;
import java.time.temporal.ChronoUnit;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.TreeMap;


/**
 * Tests a grid of configurations against a dataset
 */
public class Grid {

    private static final String pqCacheDir = "pq_cache";

    private static final String dirPrefix = "BenchGraphDir";

    static void runAll(DataSet ds,
                       List<Integer> mGrid,
                       List<Integer> efConstructionGrid,
                       List<Float> neighborOverflowGrid,
                       List<Boolean> addHierarchyGrid,
                       List<? extends Set<FeatureId>> featureSets,
                       List<Function<DataSet, CompressorParameters>> buildCompressors,
                       List<Function<DataSet, CompressorParameters>> compressionGrid,
                       List<Integer> topKGrid,
                       List<Double> efSearchFactor,
                       List<Boolean> usePruningGrid) throws IOException
    {
        var testDirectory = Files.createTempDirectory(dirPrefix);
        try {
            for (var addHierarchy :  addHierarchyGrid) {
                for (int M : mGrid) {
                    for (float neighborOverflow: neighborOverflowGrid) {
                        for (int efC : efConstructionGrid) {
                            for (var bc : buildCompressors) {
                                var compressor = getCompressor(bc, ds);
                                runOneGraph(featureSets, M, efC, neighborOverflow, addHierarchy, compressor, compressionGrid, topKGrid, efSearchFactor, usePruningGrid, ds, testDirectory);
                            }
                        }
                    }
                }
            }
        } finally {
            try
            {
                Files.delete(testDirectory);
            } catch (DirectoryNotEmptyException e) {
                // something broke, we're almost certainly in the middle of another exception being thrown,
                // so if we don't swallow this one it will mask the original exception
            }

            cachedCompressors.clear();
        }
    }

    static void runOneGraph(List<? extends Set<FeatureId>> featureSets,
                            int M,
                            int efConstruction,
                            float neighborOverflow,
                            boolean addHierarchy,
                            VectorCompressor<?> buildCompressor,
                            List<Function<DataSet, CompressorParameters>> compressionGrid,
                            List<Integer> topKGrid,
                            List<Double> efSearchOptions,
                            List<Boolean> usePruningGrid,
                            DataSet ds,
                            Path testDirectory) throws IOException
    {
        Map<Set<FeatureId>, GraphIndex> indexes;
        if (buildCompressor == null) {
            indexes = buildInMemory(featureSets, M, efConstruction, neighborOverflow, addHierarchy, ds, testDirectory);
        } else {
            indexes = buildOnDisk(featureSets, M, efConstruction, neighborOverflow, addHierarchy, ds, testDirectory, buildCompressor);
        }

        try {
            for (var cpSupplier : compressionGrid) {
                var compressor = getCompressor(cpSupplier, ds);
                CompressedVectors cv;
                if (compressor == null) {
                    cv = null;
                    System.out.format("Uncompressed vectors%n");
                } else {
                    long start = System.nanoTime();
                    cv = compressor.encodeAll(ds.getBaseRavv());
                    System.out.format("%s encoded %d vectors [%.2f MB] in %.2fs%n", compressor, ds.baseVectors.size(), (cv.ramBytesUsed() / 1024f / 1024f), (System.nanoTime() - start) / 1_000_000_000.0);
                }

                indexes.forEach((features, index) -> {
                    try (var cs = new ConfiguredSystem(ds, index, cv,
                                                       index instanceof OnDiskGraphIndex ? ((OnDiskGraphIndex) index).getFeatureSet() : Set.of())) {
                        testConfiguration(cs, topKGrid, efSearchOptions, usePruningGrid, M, efConstruction, neighborOverflow, addHierarchy);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                });
            }
            for (var index : indexes.values()) {
                index.close();
            }
        } finally {
            for (int n = 0; n < featureSets.size(); n++) {
                Files.deleteIfExists(testDirectory.resolve("graph" + n));
            }
        }
    }

    private static Map<Set<FeatureId>, GraphIndex> buildOnDisk(List<? extends Set<FeatureId>> featureSets,
                                                               int M,
                                                               int efConstruction,
                                                               float neighborOverflow,
                                                               boolean addHierarchy,
                                                               DataSet ds,
                                                               Path testDirectory,
                                                               VectorCompressor<?> buildCompressor)
            throws IOException
    {
        var floatVectors = ds.getBaseRavv();
        ForkJoinPool simdExecutor = PhysicalCoreExecutor.pool();
        int encodingBatchSize = 300_000;
        ImmutablePQVectors pqVectors = processPQInBatches(floatVectors, (ProductQuantization) buildCompressor, simdExecutor, encodingBatchSize);
        System.out.println("PQ encoding complete.");

       //  var pq = (PQVectors) buildCompressor.encodeAll(floatVectors);
        var bsp = BuildScoreProvider.pqBuildScoreProvider(ds.similarityFunction, pqVectors);
        GraphIndexBuilder builder = new GraphIndexBuilder(bsp, floatVectors.dimension(), M, efConstruction, neighborOverflow, 1.2f, addHierarchy);

        // use the inline vectors index as the score provider for graph construction
        Map<Set<FeatureId>, OnDiskGraphIndexWriter> writers = new HashMap<>();
        Map<Set<FeatureId>, Map<FeatureId, IntFunction<Feature.State>>> suppliers = new HashMap<>();
        OnDiskGraphIndexWriter scoringWriter = null;
        int n = 0;
        for (var features : featureSets) {
            var graphPath = testDirectory.resolve("graph" + n++);
            var bws = builderWithSuppliers(features, builder.getGraph(), graphPath, floatVectors, pqVectors.getCompressor());
            var writer = bws.builder.build();
            writers.put(features, writer);
            suppliers.put(features, bws.suppliers);
            if (features.contains(FeatureId.INLINE_VECTORS) || features.contains(FeatureId.NVQ_VECTORS)) {
                scoringWriter = writer;
            }
        }
        if (scoringWriter == null) {
            throw new IllegalStateException("Bench looks for either NVQ_VECTORS or INLINE_VECTORS feature set for scoring compressed builds.");
        }

//        // build the graph incrementally
//        long start = System.nanoTime();
//        var vv = floatVectors.threadLocalSupplier();
//        PhysicalCoreExecutor.pool().submit(() -> {
//            IntStream.range(0, floatVectors.size()).parallel().forEach(node -> {
//                writers.forEach((features, writer) -> {
//                    try {
//                        var stateMap = new EnumMap<FeatureId, Feature.State>(FeatureId.class);
//                        suppliers.get(features).forEach((featureId, supplier) -> {
//                            stateMap.put(featureId, supplier.apply(node));
//                        });
//                        writer.writeInline(node, stateMap);
//                    } catch (IOException e) {
//                        throw new UncheckedIOException(e);
//                    }
//                });
//                builder.addGraphNode(node, vv.get().getVector(node));
//            });
//        }).join();
//        builder.cleanup();

        long startTime = System.nanoTime();
        var vv = floatVectors.threadLocalSupplier(); // ThreadLocal supplier for vector access

        // Determine the maximum number of nodes to process concurrently based on memory.
        int maxConcurrentNodes = 50;
        // Create a queue with a bounded capacity.
        // Tune based on memory vs. keeping threads busy. Start smaller.
        int queueCapacity = maxConcurrentNodes * 4; // e.g., 200 tasks waiting in queue
        LinkedBlockingQueue<Runnable> workQueue = new LinkedBlockingQueue<>(queueCapacity);

        System.out.printf("Using ThreadPoolExecutor with %d threads and queue capacity %d%n",
                maxConcurrentNodes, queueCapacity);

        // Create the ThreadPoolExecutor directly
        // corePoolSize = maxConcurrentNodes
        // maximumPoolSize = maxConcurrentNodes (fixed size pool)
        // keepAliveTime = 0L (doesn't matter for fixed size)
        // workQueue = the bounded queue
        // RejectedExecutionHandler = CallerRunsPolicy()
        ExecutorService nodeProcessingPool = new ThreadPoolExecutor(
                maxConcurrentNodes,
                maxConcurrentNodes,
                0L, TimeUnit.MILLISECONDS,
                workQueue,
                new ThreadPoolExecutor.CallerRunsPolicy() // Crucial for throttling!
        );

        List<Future<?>> futures = new ArrayList<>();
        int vectorCount = floatVectors.size();

        final AtomicInteger nodesProcessedCounter = new AtomicInteger(0);
        final int reportingInterval = 100_000;

        System.out.printf("Submitting %d nodes for processing (will block if queue is full)...%n", vectorCount);

        // Submit one task per node to the limited pool
        for (int i = 0; i < vectorCount; i++) {
            final int nodeOrdinal = i; // Effectively final variable for lambda

            Future<?> future = nodeProcessingPool.submit(() -> {
                try {
                    // 1. Load the vector (only for this node)
                    var vector = vv.get().getVector(nodeOrdinal);

                    // 2. Compute features (this part remains the same)
                    var stateMap = new EnumMap<FeatureId, Feature.State>(FeatureId.class);
                    writers.forEach((features, writer) -> {
                        // This inner part doesn't need explicit try-catch if
                        // UncheckedIOException is acceptable for writer errors.
                        // If writer errors should allow other nodes to proceed, add try-catch here.
                        suppliers.get(features).forEach((featureId, supplier) -> {
                            stateMap.put(featureId, supplier.apply(nodeOrdinal));
                        });
                        try {
                            writer.writeInline(nodeOrdinal, stateMap);
                        } catch (IOException e) {
                            // Consider logging instead of throwing to allow completion?
                            throw new UncheckedIOException("Error writing features for node " + nodeOrdinal, e);
                        }
                    });

                    // 3. Add the node to the graph builder
                    // Assuming builder.addGraphNode is thread-safe AND handles ordinal correctly
                    builder.addGraphNode(nodeOrdinal, vector);

                    // At this point, 'vector' and 'stateMap' go out of scope for this task
                    // and become eligible for garbage collection, helping manage memory.

                    // Increment counter *after* main work for this node is done
                    int count = nodesProcessedCounter.incrementAndGet();

                    if (count % reportingInterval == 0) {
                        // Simple timestamped progress message
                        System.out.printf("[%s] Processed %d / %d nodes...%n",
                                LocalTime.now().truncatedTo(ChronoUnit.SECONDS),
                                count,
                                vectorCount);
                    }

                } catch (Exception e) {
                    // Catch exceptions from vector loading or addGraphNode if they occur
                    // Log or wrap them to be checked later via the Future
                    System.err.printf("!!! Exception processing node %d: %s%n", nodeOrdinal, e.getMessage());
                    // Wrap in a runtime exception to be retrieved by Future.get()
                    throw new RuntimeException("Error processing node " + nodeOrdinal, e);
                }
            });
            futures.add(future);
        }

        // Shutdown the pool gracefully - stop accepting new tasks
        System.out.println("All node processing tasks submitted. Shutting down pool and awaiting completion...");
        nodeProcessingPool.shutdown();

        // Wait for all submitted tasks to finish
        try {
            // Wait a very long time for tasks to complete
            if (!nodeProcessingPool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS)) {
                System.err.println("Node processing pool termination timed out. Forcing shutdown.");
                nodeProcessingPool.shutdownNow();
            }
        } catch (InterruptedException e) {
            System.err.println("Node processing pool awaiting termination interrupted. Forcing shutdown.");
            nodeProcessingPool.shutdownNow();
            Thread.currentThread().interrupt(); // Preserve interrupt status
        }

        System.out.println("Node processing pool finished.");

        // Crucial: Check for exceptions that occurred during task execution
        for (Future<?> f : futures) {
            try {
                f.get(); // This will re-throw any exception caught and wrapped inside the task's lambda
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Graph building interrupted while checking results", e);
            } catch (java.util.concurrent.ExecutionException e) {
                // Unwrap the actual cause thrown inside the task
                Throwable cause = e.getCause();
                System.err.println("Error occurred during node processing task: " + cause.getMessage());
                if (cause instanceof RuntimeException) {
                    throw (RuntimeException) cause;
                } else if (cause instanceof Error) {
                    throw (Error) cause;
                } else {
                    throw new RuntimeException("Unhandled error during graph building execution", cause);
                }
            }
        }

        System.out.println("All nodes processed successfully. Running cleanup...");

        // Finally, call cleanup (assuming this needs to happen after all nodes are added)
        builder.cleanup();

        long endTime = System.nanoTime();
        System.out.printf("Graph building with controlled concurrency took %.2f seconds%n", (endTime - startTime) / 1e9);

        // write the edge lists and close the writers
        // if our feature set contains Fused ADC, we need a Fused ADC write-time supplier (as we don't have neighbor information during writeInline)
        writers.entrySet().stream().parallel().forEach(entry -> {
            var writer = entry.getValue();
            var features = entry.getKey();
            Map<FeatureId, IntFunction<Feature.State>> writeSuppliers;
            if (features.contains(FeatureId.FUSED_ADC)) {
                writeSuppliers = new EnumMap<>(FeatureId.class);
                var view = builder.getGraph().getView();
                writeSuppliers.put(FeatureId.FUSED_ADC, ordinal -> new FusedADC.State(view, pqVectors, ordinal));
            } else {
                writeSuppliers = Map.of();
            }
            try {
                writer.write(writeSuppliers);
                writer.close();
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
        builder.close();
        System.out.format("Build and write %s in %ss%n", featureSets, (System.nanoTime() - startTime) / 1_000_000_000.0);

        // open indexes
        Map<Set<FeatureId>, GraphIndex> indexes = new HashMap<>();
        n = 0;
        for (var features : featureSets) {
            var graphPath = testDirectory.resolve("graph" + n++);
            var index = OnDiskGraphIndex.load(ReaderSupplierFactory.open(graphPath));
            indexes.put(features, index);
        }
        return indexes;
    }

    private static BuilderWithSuppliers builderWithSuppliers(Set<FeatureId> features,
                                                             OnHeapGraphIndex onHeapGraph,
                                                             Path outPath,
                                                             RandomAccessVectorValues floatVectors,
                                                             ProductQuantization pq)
            throws FileNotFoundException
    {
        var identityMapper = new OrdinalMapper.IdentityMapper(floatVectors.size() - 1);
        var builder = new OnDiskGraphIndexWriter.Builder(onHeapGraph, outPath)
                .withMapper(identityMapper);
        Map<FeatureId, IntFunction<Feature.State>> suppliers = new EnumMap<>(FeatureId.class);
        for (var featureId : features) {
            switch (featureId) {
                case INLINE_VECTORS:
                    builder.with(new InlineVectors(floatVectors.dimension()));
                    suppliers.put(FeatureId.INLINE_VECTORS, ordinal -> new InlineVectors.State(floatVectors.getVector(ordinal)));
                    break;
                case FUSED_ADC:
                    if (pq == null) {
                        System.out.println("Skipping Fused ADC feature due to null ProductQuantization");
                        continue;
                    }
                    // no supplier as these will be used for writeInline, when we don't have enough information to fuse neighbors
                    builder.with(new FusedADC(onHeapGraph.maxDegree(), pq));
                    break;
                case NVQ_VECTORS:
                    var nvq = NVQuantization.compute(floatVectors, 2);
                    builder.with(new NVQ(nvq));
                    suppliers.put(FeatureId.NVQ_VECTORS, ordinal -> new NVQ.State(nvq.encode(floatVectors.getVector(ordinal))));
                    break;

            }
        }
        return new BuilderWithSuppliers(builder, suppliers);
    }

    private static class BuilderWithSuppliers {
        public final OnDiskGraphIndexWriter.Builder builder;
        public final Map<FeatureId, IntFunction<Feature.State>> suppliers;

        public BuilderWithSuppliers(OnDiskGraphIndexWriter.Builder builder, Map<FeatureId, IntFunction<Feature.State>> suppliers) {
            this.builder = builder;
            this.suppliers = suppliers;
        }
    }

    private static Map<Set<FeatureId>, GraphIndex> buildInMemory(List<? extends Set<FeatureId>> featureSets,
                                                                 int M,
                                                                 int efConstruction,
                                                                 float neighborOverflow,
                                                                 boolean addHierarchy,
                                                                 DataSet ds,
                                                                 Path testDirectory)
            throws IOException
    {
        var floatVectors = ds.getBaseRavv();
        Map<Set<FeatureId>, GraphIndex> indexes = new HashMap<>();
        long start;
        var bsp = BuildScoreProvider.randomAccessScoreProvider(floatVectors, ds.similarityFunction);
        GraphIndexBuilder builder = new GraphIndexBuilder(bsp,
                                                          floatVectors.dimension(),
                                                          M,
                                                          efConstruction,
                                                          neighborOverflow,
                                                          1.2f,
                                                          addHierarchy,
                                                          PhysicalCoreExecutor.pool(),
                                                          ForkJoinPool.commonPool());
        start = System.nanoTime();
        var onHeapGraph = builder.build(floatVectors);
        System.out.format("Build (%s) M=%d overflow=%.2f ef=%d in %.2fs%n",
                          "full res",
                          M,
                          neighborOverflow,
                          efConstruction,
                          (System.nanoTime() - start) / 1_000_000_000.0);
        for (int i = 0; i <= onHeapGraph.getMaxLevel(); i++) {
            System.out.format("  L%d: %d nodes, %.2f avg degree%n",
                              i,
                              onHeapGraph.getLayerSize(i),
                              onHeapGraph.getAverageDegree(i));
        }
        int n = 0;
        for (var features : featureSets) {
            if (features.contains(FeatureId.FUSED_ADC)) {
                System.out.println("Skipping Fused ADC feature when building in memory");
                continue;
            }
            var graphPath = testDirectory.resolve("graph" + n++);
            var bws = builderWithSuppliers(features, onHeapGraph, graphPath, floatVectors, null);
            try (var writer = bws.builder.build()) {
                start = System.nanoTime();
                writer.write(bws.suppliers);
                System.out.format("Wrote %s in %.2fs%n", features, (System.nanoTime() - start) / 1_000_000_000.0);
            }

            var index = OnDiskGraphIndex.load(ReaderSupplierFactory.open(graphPath));
            indexes.put(features, index);
        }
        return indexes;
    }

    // avoid recomputing the compressor repeatedly (this is a relatively small memory footprint)
    static final Map<String, VectorCompressor<?>> cachedCompressors = new IdentityHashMap<>();

    private static void testConfiguration(ConfiguredSystem cs,
                                          List<Integer> topKGrid,
                                          List<Double> efSearchOptions,
                                          List<Boolean> usePruningGrid,
                                          int M,
                                          int efConstruction,
                                          float neighborOverflow,
                                          boolean addHierarchy) {
        int queryRuns = 2;
        System.out.format("Using %s:%n", cs.index);
        // 1) Select benchmarks to run
        List<QueryBenchmark> benchmarks = List.of(
                new ThroughputBenchmark(2, 0.1),
                new LatencyBenchmark(),
                new CountBenchmark(),
                new AccuracyBenchmark()
        );
        QueryTester tester = new QueryTester(benchmarks);

        for (var topK : topKGrid) {
            for (var usePruning : usePruningGrid) {
                BenchmarkTablePrinter printer = new BenchmarkTablePrinter();
                printer.printConfig(Map.of(
                        "M",                  M,
                        "efConstruction",     efConstruction,
                        "neighborOverflow",   neighborOverflow,
                        "addHierarchy",       addHierarchy,
                        "usePruning",         usePruning
                ));
                for (var overquery : efSearchOptions) {
                    int rerankK = (int) (topK * overquery);

                    var results = tester.run(cs, topK, rerankK, usePruning, queryRuns);
                    printer.printRow(overquery, results);
                }
                printer.printFooter();
            }
        }
    }

    private static VectorCompressor<?> getCompressor(Function<DataSet, CompressorParameters> cpSupplier, DataSet ds) {
        var cp = cpSupplier.apply(ds);
        if (!cp.supportsCaching()) {
            return cp.computeCompressor(ds);
        }

        var fname = cp.idStringFor(ds);
        return cachedCompressors.computeIfAbsent(fname, __ -> {
            var path = Paths.get(pqCacheDir).resolve(fname);
            if (path.toFile().exists()) {
                try {
                    try (var readerSupplier = ReaderSupplierFactory.open(path)) {
                        try (var rar = readerSupplier.get()) {
                            var pq = ProductQuantization.load(rar);
                            System.out.format("%s loaded from %s%n", pq, fname);
                            return pq;
                        }
                    }
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }

            var start = System.nanoTime();
            var compressor = cp.computeCompressor(ds);
            System.out.format("%s build in %.2fs,%n", compressor, (System.nanoTime() - start) / 1_000_000_000.0);
            if (cp.supportsCaching()) {
                try {
                    Files.createDirectories(path.getParent());
                    try (var writer = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
                        compressor.write(writer, OnDiskGraphIndex.CURRENT_VERSION);
                    }
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
            return compressor;
        });
    }

    public static class ConfiguredSystem implements AutoCloseable {
        DataSet ds;
        GraphIndex index;
        CompressedVectors cv;
        Set<FeatureId> features;

        private final ExplicitThreadLocal<GraphSearcher> searchers = ExplicitThreadLocal.withInitial(() -> {
            return new GraphSearcher(index);
        });

        ConfiguredSystem(DataSet ds, GraphIndex index, CompressedVectors cv, Set<FeatureId> features) {
            this.ds = ds;
            this.index = index;
            this.cv = cv;
            this.features = features;
        }

        public SearchScoreProvider scoreProviderFor(VectorFloat<?> queryVector, GraphIndex.View view) {
            // if we're not compressing then just use the exact score function
            if (cv == null) {
                return SearchScoreProvider.exact(queryVector, ds.similarityFunction, ds.getBaseRavv());
            }

            var scoringView = (GraphIndex.ScoringView) view;
            ScoreFunction.ApproximateScoreFunction asf;
            if (features.contains(FeatureId.FUSED_ADC)) {
                asf = scoringView.approximateScoreFunctionFor(queryVector, ds.similarityFunction);
            } else {
                asf = cv.precomputedScoreFunctionFor(queryVector, ds.similarityFunction);
            }
            var rr = scoringView.rerankerFor(queryVector, ds.similarityFunction);
            return new SearchScoreProvider(asf, rr);
        }

        public GraphSearcher getSearcher() {
            return searchers.get();
        }

        public DataSet getDataSet() {
            return ds;
        }

        @Override
        public void close() throws Exception {
            searchers.close();
        }
    }

    public void processInBatches(RandomAccessVectorValues ravv, int batchSize) {
        // Process ravvs in batches to deal with LTM datasets.
        int totalVectors = ravv.size();
        for (int batchStart = 0; batchStart < totalVectors; batchStart += batchSize) {
            int batchEnd = Math.min(batchStart + batchSize, totalVectors);
            System.out.format("Processing batch: %d to %d%n", batchStart, batchEnd);
        }
    }

    private static ImmutablePQVectors processPQInBatches(RandomAccessVectorValues ravv,
                                                         ProductQuantization pq,
                                                         ForkJoinPool simdExecutor,
                                                         int batchSize) {
        // List to store PQ vectors for each batch.
        List<ImmutablePQVectors> batchPQList = new ArrayList<>();
        int totalVectors = ravv.size();

        // Loop over the dataset in batches.
        for (int batchStart = 0; batchStart < totalVectors; batchStart += batchSize) {
            int batchEnd = Math.min(batchStart + batchSize, totalVectors);
            System.out.format("Processing batch: %d to %d%n", batchStart, batchEnd);

            // Create a subset view for the current batch.
            RandomAccessVectorValues batchRavv = new SubsetRandomAccessVectorValues(ravv, batchStart, batchEnd);

            // Process the batch by encoding its vectors using PQVectors.encodeAndBuild.
            ImmutablePQVectors batchPQ = PQVectors.encodeAndBuild(pq, batchRavv.size(), batchRavv, simdExecutor);

            // Store the batch result.
            batchPQList.add(batchPQ);
        }

        // Merge the batch-level PQ vectors into one final instance.
        ImmutablePQVectors finalPQ = mergeBatchPQVectors(batchPQList);
        return finalPQ;
    }

    /**
     * Merge multiple ImmutablePQVectors (from different batches) into a single final instance.
     * This method concatenates the ByteSequence chunks from each batch in the order of processing.
     */
    private static ImmutablePQVectors mergeBatchPQVectors(List<ImmutablePQVectors> batchPQList) {
        // Sum total vector count from all batches.
        int totalVectors = batchPQList.stream().mapToInt(bpq -> bpq.count()).sum();

        // Assume all batches use the same ProductQuantization instance and chunk size.
        ProductQuantization pq = batchPQList.get(0).getPQ();
        int vectorsPerChunk = batchPQList.get(0).getVectorsPerChunk();

        // Accumulate the chunks in order.
        List<ByteSequence<?>> allChunks = new ArrayList<>();
        for (ImmutablePQVectors bpq : batchPQList) {
            for (ByteSequence<?> chunk : bpq.getChunks()) {
                allChunks.add(chunk);
            }
        }
        ByteSequence<?>[] finalChunks = allChunks.toArray(new ByteSequence<?>[0]);

        return new ImmutablePQVectors(pq, finalChunks, totalVectors, vectorsPerChunk);
    }
}
