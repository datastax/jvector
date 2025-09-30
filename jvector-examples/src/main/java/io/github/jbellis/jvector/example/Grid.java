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
import io.github.jbellis.jvector.example.benchmarks.AccuracyBenchmark;
import io.github.jbellis.jvector.example.benchmarks.BenchmarkTablePrinter;
import io.github.jbellis.jvector.example.benchmarks.CountBenchmark;
import io.github.jbellis.jvector.example.benchmarks.LatencyBenchmark;
import io.github.jbellis.jvector.example.benchmarks.QueryBenchmark;
import io.github.jbellis.jvector.example.benchmarks.QueryTester;
import io.github.jbellis.jvector.example.benchmarks.ThroughputBenchmark;
import io.github.jbellis.jvector.example.benchmarks.*;
import io.github.jbellis.jvector.example.benchmarks.diagnostics.DiagnosticLevel;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.FilteredForkJoinPool;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedADC;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.status.StatusTracker;
import io.github.jbellis.jvector.status.StatusUpdate;
import io.github.jbellis.jvector.status.TrackerScope;
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
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.function.BooleanSupplier;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

/**
 * Tests a grid of configurations against a dataset
 */
public class Grid {

    private static final String pqCacheDir = "pq_cache";

    private static final String dirPrefix = "BenchGraphDir";

    private static final Map<String,Double> indexBuildTimes = new HashMap<>();

    private static int diagnostic_level;

    // Backward compatibility method without TrackerScope (with benchmarks)
    static void runAll(DataSet ds,
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
                       Map<String, List<String>> benchmarks) throws IOException
    {
        // Create a simple TrackerScope with no sinks for backward compatibility
        TrackerScope defaultScope = new TrackerScope("Grid");
        runAll(ds, defaultScope, null, mGrid, efConstructionGrid, neighborOverflowGrid, addHierarchyGrid, refineFinalGraphGrid, featureSets, buildCompressors, compressionGrid, topKGrid, usePruningGrid, benchmarks, null);
    }

    static void runAll(DataSet ds,
                       TrackerScope datasetScope,
                       StatusTracker<DatasetTask> datasetTracker,
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
                       Map<String, List<String>> benchmarks,
                       BooleanSupplier shouldAbort) throws IOException
    {
        checkAbort(shouldAbort);

        var testDirectory = Files.createTempDirectory(dirPrefix);
        try {
            for (var addHierarchy :  addHierarchyGrid) {
                checkAbort(shouldAbort);
                for (var refineFinalGraph : refineFinalGraphGrid) {
                    checkAbort(shouldAbort);
                    for (int M : mGrid) {
                        checkAbort(shouldAbort);
                        for (float neighborOverflow : neighborOverflowGrid) {
                            checkAbort(shouldAbort);
                            for (int efC : efConstructionGrid) {
                                checkAbort(shouldAbort);
                                for (var bc : buildCompressors) {
                                    checkAbort(shouldAbort);
                                    var compressor = getCompressor(bc, ds);
                                    // Create a child scope for this specific graph configuration
                                    String configName = String.format("M%d_efC%d_no%.1f", M, efC, neighborOverflow);
                                    TrackerScope graphScope = datasetScope.createChildScope(configName);
                                    GraphConfigTask configTask = new GraphConfigTask(configName);

                                    StatusTracker<GraphConfigTask> configTracker =
                                            datasetTracker != null
                                                    ? datasetTracker.executeWithContext(() -> graphScope.track(configTask))
                                                    : graphScope.track(configTask);

                                    try (configTracker) {
                                        configTask.start();
                                        runOneGraph(featureSets,
                                                graphScope,
                                                configTracker,
                                                M,
                                                efC,
                                                neighborOverflow,
                                                addHierarchy,
                                                refineFinalGraph,
                                                compressor,
                                                compressionGrid,
                                                topKGrid,
                                                usePruningGrid,
                                                benchmarks,
                                                shouldAbort,
                                                ds,
                                                testDirectory);
                                        configTask.complete();
                                    } catch (Exception e) {
                                        configTask.fail();
                                        throw e;
                                    }
                                }
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

    // Backward compatibility method without TrackerScope
    static void runAll(DataSet ds,
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
        // Create a simple TrackerScope with no sinks for backward compatibility
        TrackerScope defaultScope = new TrackerScope("Grid");
        runAll(ds, defaultScope, null, mGrid, efConstructionGrid, neighborOverflowGrid, addHierarchyGrid, refineFinalGraphGrid, featureSets, buildCompressors, compressionGrid, topKGrid, usePruningGrid, null, null);
    }

    static void runAll(DataSet ds,
                       TrackerScope datasetScope,
                       StatusTracker<DatasetTask> datasetTracker,
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
        runAll(ds, datasetScope, datasetTracker, mGrid, efConstructionGrid, neighborOverflowGrid, addHierarchyGrid, refineFinalGraphGrid, featureSets, buildCompressors, compressionGrid, topKGrid, usePruningGrid, null, null);
    }

    // Backward compatibility method without TrackerScope
    static void runOneGraph(List<? extends Set<FeatureId>> featureSets,
                            int M,
                            int efConstruction,
                            float neighborOverflow,
                            boolean addHierarchy,
                            boolean refineFinalGraph,
                            VectorCompressor<?> buildCompressor,
                            List<Function<DataSet, CompressorParameters>> compressionGrid,
                            Map<Integer, List<Double>> topKGrid,
                            List<Boolean> usePruningGrid,
                            Map<String, List<String>> benchmarks,
                            DataSet ds,
                            Path testDirectory) throws IOException
    {
        TrackerScope defaultScope = new TrackerScope("RunOneGraph");
        runOneGraph(featureSets, defaultScope, null, M, efConstruction, neighborOverflow, addHierarchy, refineFinalGraph, buildCompressor, compressionGrid, topKGrid, usePruningGrid, benchmarks, null, ds, testDirectory);
    }

    static void runOneGraph(List<? extends Set<FeatureId>> featureSets,
                            TrackerScope configScope,
                            StatusTracker<GraphConfigTask> configTracker,
                            int M,
                            int efConstruction,
                            float neighborOverflow,
                            boolean addHierarchy,
                            boolean refineFinalGraph,
                            VectorCompressor<?> buildCompressor,
                            List<Function<DataSet, CompressorParameters>> compressionGrid,
                            Map<Integer, List<Double>> topKGrid,
                            List<Boolean> usePruningGrid,
                            Map<String, List<String>> benchmarks,
                            BooleanSupplier shouldAbort,
                            DataSet ds,
                            Path testDirectory) throws IOException
    {
        checkAbort(shouldAbort);

        Map<Set<FeatureId>, GraphIndex> indexes;
        if (buildCompressor == null) {
            TrackerScope buildScope = configScope.createChildScope("BuildInMemory");
            indexes = buildInMemory(featureSets, buildScope, configTracker, M, efConstruction, neighborOverflow, addHierarchy, refineFinalGraph, ds, testDirectory, shouldAbort);
        } else {
            // Create a child scope for building the index
            TrackerScope buildScope = configScope.createChildScope("BuildOnDisk");
            indexes = buildOnDisk(featureSets, buildScope, configTracker, M, efConstruction, neighborOverflow, addHierarchy, refineFinalGraph, ds, testDirectory, buildCompressor, shouldAbort);
        }

        try {
            for (var cpSupplier : compressionGrid) {
                checkAbort(shouldAbort);
                var compressor = getCompressor(cpSupplier, ds);
                CompressedVectors cv;
                if (compressor == null) {
                    cv = null;
                    System.out.format("Uncompressed vectors%n");
                } else {
                    checkAbort(shouldAbort);
                    // Create a task for vector compression/encoding
                    class CompressionTask implements StatusUpdate.Provider<CompressionTask> {
                        private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;
                        private volatile double progress = 0.0;
                        private final String compressorName;

                        CompressionTask(String compressorName) {
                            this.compressorName = compressorName;
                        }

                        @Override
                        public StatusUpdate<CompressionTask> getTaskStatus() {
                            return new StatusUpdate<>(progress, state, this);
                        }

                        @Override
                        public String toString() {
                            return "Vector Compression (" + compressorName + ")";
                        }
                    }

                    CompressionTask compressionTask = new CompressionTask(compressor.toString());

                    // Create tracker as child of configTracker if available
                    final StatusTracker<CompressionTask> compressionTracker;
                    if (configTracker != null && configScope != null) {
                        compressionTracker = configTracker.executeWithContext(() -> configScope.track(compressionTask));
                    } else if (configTracker != null) {
                        compressionTracker = configTracker.createChild(compressionTask);
                    } else if (configScope != null) {
                        compressionTracker = configScope.track(compressionTask);
                    } else {
                        compressionTracker = null;
                    }

                    if (compressionTracker != null) {
                        try (compressionTracker) {
                            compressionTask.state = StatusUpdate.RunState.RUNNING;
                            long start = System.nanoTime();
                            cv = compressor.encodeAll(ds.getBaseRavv());
                            compressionTask.progress = 1.0;
                            compressionTask.state = StatusUpdate.RunState.SUCCESS;
                            System.out.format("%s encoded %d vectors [%.2f MB] in %.2fs%n", compressor, ds.baseVectors.size(), (cv.ramBytesUsed() / 1024f / 1024f), (System.nanoTime() - start) / 1_000_000_000.0);
                        }
                    } else {
                        // No tracking available, just do the compression
                        long start = System.nanoTime();
                        cv = compressor.encodeAll(ds.getBaseRavv());
                        System.out.format("%s encoded %d vectors [%.2f MB] in %.2fs%n", compressor, ds.baseVectors.size(), (cv.ramBytesUsed() / 1024f / 1024f), (System.nanoTime() - start) / 1_000_000_000.0);
                    }
                }

                indexes.forEach((features, index) -> {
                    try (var cs = new ConfiguredSystem(ds, index, cv,
                                                       index instanceof OnDiskGraphIndex ? ((OnDiskGraphIndex) index).getFeatureSet() : Set.of())) {
                        checkAbort(shouldAbort);
                        testConfiguration(configScope, configTracker, cs, topKGrid, usePruningGrid, M, efConstruction, neighborOverflow, addHierarchy, benchmarks, shouldAbort);
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
                                                               TrackerScope buildScope,
                                                               StatusTracker<GraphConfigTask> configTracker,
                                                               int M,
                                                               int efConstruction,
                                                               float neighborOverflow,
                                                               boolean addHierarchy,
                                                               boolean refineFinalGraph,
                                                               DataSet ds,
                                                               Path testDirectory,
                                                               VectorCompressor<?> buildCompressor,
                                                               BooleanSupplier shouldAbort)
            throws IOException
    {
        checkAbort(shouldAbort);

        class OnDiskBuildTask implements StatusUpdate.Provider<OnDiskBuildTask> {
            private final int totalNodes;
            private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;
            private volatile double pqProgress = 0.0;
            private volatile double graphProgress = 0.0;

            OnDiskBuildTask(int totalNodes) {
                this.totalNodes = totalNodes;
            }

            void start() {
                state = StatusUpdate.RunState.RUNNING;
                if (totalNodes == 0) {
                    pqProgress = 1.0;
                    graphProgress = 1.0;
                }
            }

            void updatePQProgress(double progress) {
                pqProgress = clamp(progress);
            }

            void updateGraphProgress(double progress) {
                graphProgress = clamp(progress);
            }

            void complete() {
                pqProgress = 1.0;
                graphProgress = 1.0;
                state = StatusUpdate.RunState.SUCCESS;
            }

            void fail() {
                state = StatusUpdate.RunState.FAILED;
            }

            private double clamp(double value) {
                return Math.max(0.0, Math.min(1.0, value));
            }

            @Override
            public StatusUpdate<OnDiskBuildTask> getTaskStatus() {
                double combined = (pqProgress + graphProgress) / 2.0;
                return new StatusUpdate<>(combined, state, this);
            }

            @Override
            public String toString() {
                return String.format("On-disk Build (%d nodes)", totalNodes);
            }
        }

        class PQTask implements StatusUpdate.Provider<PQTask> {
            private final OnDiskBuildTask parent;
            private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;
            private volatile double progress = 0.0;

            PQTask(OnDiskBuildTask parent) {
                this.parent = parent;
            }

            void start() {
                state = StatusUpdate.RunState.RUNNING;
                progress = 0.0;
                parent.updatePQProgress(progress);
            }

            void complete() {
                progress = 1.0;
                state = StatusUpdate.RunState.SUCCESS;
                parent.updatePQProgress(progress);
            }

            void fail() {
                state = StatusUpdate.RunState.FAILED;
            }

            @Override
            public StatusUpdate<PQTask> getTaskStatus() {
                return new StatusUpdate<>(progress, state, this);
            }

            @Override
            public String toString() {
                return "PQ Compression";
            }
        }

        class GraphConstructionTask implements StatusUpdate.Provider<GraphConstructionTask> {
            private final int totalNodes;
            private final OnDiskBuildTask parent;
            private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;
            private volatile double progress = 0.0;

            GraphConstructionTask(int totalNodes, OnDiskBuildTask parent) {
                this.totalNodes = totalNodes;
                this.parent = parent;
            }

            void start() {
                state = StatusUpdate.RunState.RUNNING;
                if (totalNodes == 0) {
                    progress = 1.0;
                    parent.updateGraphProgress(progress);
                }
            }

            void updateNodesProcessed(int completed) {
                if (totalNodes > 0) {
                    progress = Math.min(1.0, (double) completed / totalNodes);
                } else {
                    progress = 1.0;
                }
                parent.updateGraphProgress(progress);
            }

            void complete() {
                progress = 1.0;
                state = StatusUpdate.RunState.SUCCESS;
                parent.updateGraphProgress(progress);
            }

            void fail() {
                state = StatusUpdate.RunState.FAILED;
            }

            @Override
            public StatusUpdate<GraphConstructionTask> getTaskStatus() {
                return new StatusUpdate<>(progress, state, this);
            }

            @Override
            public String toString() {
                return "Graph Construction & Write";
            }
        }

        var floatVectors = ds.getBaseRavv();
        OnDiskBuildTask buildTask = new OnDiskBuildTask(floatVectors.size());

        StatusTracker<OnDiskBuildTask> tracker =
                configTracker != null && buildScope != null
                        ? configTracker.executeWithContext(() -> buildScope.track(buildTask))
                        : configTracker != null
                            ? configTracker.createChild(buildTask)
                            : buildScope.track(buildTask);

        try (tracker) {
            buildTask.start();

            PQTask pqTask = new PQTask(buildTask);
            PQVectors pq;

            try (StatusTracker<PQTask> pqTracker = tracker.createChild(pqTask)) {
                checkAbort(shouldAbort);
                pqTask.start();
                pq = (PQVectors) buildCompressor.encodeAll(floatVectors);
                pqTask.complete();
            } catch (Exception e) {
                pqTask.fail();
                buildTask.fail();
                throw e;
            }

            var bsp = BuildScoreProvider.pqBuildScoreProvider(ds.similarityFunction, pq);
            GraphIndexBuilder builder = new GraphIndexBuilder(bsp, floatVectors.dimension(), M, efConstruction, neighborOverflow, 1.2f, addHierarchy, refineFinalGraph);

            Map<Set<FeatureId>, OnDiskGraphIndexWriter> writers = new HashMap<>();
            Map<Set<FeatureId>, Map<FeatureId, IntFunction<Feature.State>>> suppliers = new HashMap<>();
            OnDiskGraphIndexWriter scoringWriter = null;
            int n = 0;
            for (var features : featureSets) {
                checkAbort(shouldAbort);
                var graphPath = testDirectory.resolve("graph" + n++);
                var bws = builderWithSuppliers(features, builder.getGraph(), graphPath, floatVectors, pq.getCompressor());
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

            GraphConstructionTask graphTask = new GraphConstructionTask(floatVectors.size(), buildTask);
            long startTime = System.nanoTime();

            try (StatusTracker<GraphConstructionTask> graphTracker = tracker.createChild(graphTask)) {
                graphTask.start();

                var vv = floatVectors.threadLocalSupplier();
                var nodeCounter = new java.util.concurrent.atomic.AtomicInteger(0);

                PhysicalCoreExecutor.pool().submit(() -> {
                    IntStream.range(0, floatVectors.size()).parallel().forEach(node -> {
                        checkAbort(shouldAbort);
                        writers.forEach((features, writer) -> {
                            try {
                                var stateMap = new EnumMap<FeatureId, Feature.State>(FeatureId.class);
                                suppliers.get(features).forEach((featureId, supplier) -> {
                                    stateMap.put(featureId, supplier.apply(node));
                                });
                                writer.writeInline(node, stateMap);
                            } catch (IOException e) {
                                throw new UncheckedIOException(e);
                            }
                        });
                        builder.addGraphNode(node, vv.get().getVector(node));

                        int completed = nodeCounter.incrementAndGet();
                        if (completed % 1000 == 0 || completed == floatVectors.size()) {
                            graphTask.updateNodesProcessed(completed);
                        }
                    });
                }).join();
                builder.cleanup();

                writers.entrySet().stream().parallel().forEach(entry -> {
                    checkAbort(shouldAbort);
                    var writer = entry.getValue();
                    var features = entry.getKey();
                    Map<FeatureId, IntFunction<Feature.State>> writeSuppliers;
                    if (features.contains(FeatureId.FUSED_ADC)) {
                        writeSuppliers = new EnumMap<>(FeatureId.class);
                        var view = builder.getGraph().getView();
                        writeSuppliers.put(FeatureId.FUSED_ADC, ordinal -> new FusedADC.State(view, pq, ordinal));
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

                graphTask.complete();
            } catch (Exception e) {
                graphTask.fail();
                buildTask.fail();
                throw e;
            }

            buildTask.complete();

            double totalTime = (System.nanoTime() - startTime) / 1_000_000_000.0;
            System.out.format("Build and write %s in %ss%n", featureSets, totalTime);
            indexBuildTimes.put(ds.name, totalTime);
        } catch (Exception e) {
            buildTask.fail();
            throw e;
        }

        // open indexes
        Map<Set<FeatureId>, GraphIndex> indexes = new HashMap<>();
        int n = 0;
       for (var features : featureSets) {
            checkAbort(shouldAbort);
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
        var builder = new OnDiskGraphIndexWriter.Builder(onHeapGraph, outPath);
        builder.withMapper(identityMapper);
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
                    int nSubVectors = floatVectors.dimension() == 2 ? 1 : 2;
                    var nvq = NVQuantization.compute(floatVectors, nSubVectors);
                    builder.with(new NVQ(nvq));
                    suppliers.put(FeatureId.NVQ_VECTORS, ordinal -> new NVQ.State(nvq.encode(floatVectors.getVector(ordinal))));
                    break;

            }
        }
        return new BuilderWithSuppliers(builder, suppliers);
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
        public final OnDiskGraphIndexWriter.Builder builder;
        public final Map<FeatureId, IntFunction<Feature.State>> suppliers;

        public BuilderWithSuppliers(OnDiskGraphIndexWriter.Builder builder, Map<FeatureId, IntFunction<Feature.State>> suppliers) {
            this.builder = builder;
            this.suppliers = suppliers;
        }
    }

    private static Map<Set<FeatureId>, GraphIndex> buildInMemory(List<? extends Set<FeatureId>> featureSets,
                                                                 TrackerScope buildScope,
                                                                 StatusTracker<GraphConfigTask> configTracker,
                                                                 int M,
                                                                 int efConstruction,
                                                                 float neighborOverflow,
                                                                 boolean addHierarchy,
                                                                 boolean refineFinalGraph,
                                                                 DataSet ds,
                                                                 Path testDirectory,
                                                                 BooleanSupplier shouldAbort)
            throws IOException
    {
        checkAbort(shouldAbort);
        var floatVectors = ds.getBaseRavv();
        Map<Set<FeatureId>, GraphIndex> indexes = new HashMap<>();
        long start;

        int writableFeatureSets = 0;
        for (var features : featureSets) {
            if (!features.contains(FeatureId.FUSED_ADC)) {
                writableFeatureSets++;
            }
        }

        class InMemoryBuildTask implements StatusUpdate.Provider<InMemoryBuildTask> {
            private final int totalOutputs;
            private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;
            private volatile double progress = 0.0;
            private int completedOutputs = 0;

            InMemoryBuildTask(int totalOutputs) {
                this.totalOutputs = totalOutputs;
            }

            void start() {
                state = StatusUpdate.RunState.RUNNING;
                progress = totalOutputs == 0 ? 1.0 : 0.0;
            }

            void graphBuilt() {
                if (totalOutputs == 0) {
                    progress = 1.0;
                } else {
                    progress = Math.max(progress, 0.5);
                }
            }

            void featureWritten() {
                if (totalOutputs == 0) {
                    progress = 1.0;
                    return;
                }
                completedOutputs++;
                progress = 0.5 + 0.5 * ((double) completedOutputs / totalOutputs);
            }

            void complete() {
                progress = 1.0;
                state = StatusUpdate.RunState.SUCCESS;
            }

            void fail() {
                state = StatusUpdate.RunState.FAILED;
            }

            @Override
            public StatusUpdate<InMemoryBuildTask> getTaskStatus() {
                return new StatusUpdate<>(progress, state, this);
            }

            @Override
            public String toString() {
                return "Build (in-memory index)";
            }
        }

        InMemoryBuildTask buildTask = new InMemoryBuildTask(writableFeatureSets);
        StatusTracker<InMemoryBuildTask> buildTracker;
        if (configTracker != null && buildScope != null) {
            buildTracker = configTracker.executeWithContext(() -> buildScope.track(buildTask));
        } else if (configTracker != null) {
            buildTracker = configTracker.createChild(buildTask);
        } else if (buildScope != null) {
            buildTracker = buildScope.track(buildTask);
        } else {
            buildTracker = null;
        }

        try (StatusTracker<InMemoryBuildTask> ignored = buildTracker) {
            if (buildTracker != null) {
                buildTask.start();
            }

            var bsp = BuildScoreProvider.randomAccessScoreProvider(floatVectors, ds.similarityFunction);
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
            checkAbort(shouldAbort);
            var onHeapGraph = builder.build(floatVectors);
            if (buildTracker != null) {
                buildTask.graphBuilt();
            }
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
                checkAbort(shouldAbort);
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

                if (buildTracker != null) {
                    buildTask.featureWritten();
                }

                var index = OnDiskGraphIndex.load(ReaderSupplierFactory.open(graphPath));
                indexes.put(features, index);
            }
            if (buildTracker != null) {
                buildTask.complete();
            }
            return indexes;
        } catch (Exception e) {
            if (buildTracker != null) {
                buildTask.fail();
            }
            throw e;
        }
    }

    // avoid recomputing the compressor repeatedly (this is a relatively small memory footprint)
    static final Map<String, VectorCompressor<?>> cachedCompressors = new IdentityHashMap<>();

    private static void testConfiguration(TrackerScope parentScope,
                                          StatusTracker<GraphConfigTask> parentTracker,
                                          ConfiguredSystem cs,
                                          Map<Integer, List<Double>> topKGrid,
                                          List<Boolean> usePruningGrid,
                                          int M,
                                          int efConstruction,
                                          float neighborOverflow,
                                          boolean addHierarchy,
                                          Map<String, List<String>> benchmarkSpec,
                                          BooleanSupplier shouldAbort) {
        int queryRuns = 2;
        System.out.format("Using %s:%n", cs.index);
        // 1) Select benchmarks to run.  Use .createDefault or .createEmpty (for other options)

        var benchmarks = setupBenchmarks(benchmarkSpec);
        QueryTester tester = new QueryTester(benchmarks);

        // Create a child scope for this test configuration
        TrackerScope testConfigScope = parentScope != null ?
            parentScope.createChildScope(String.format("TestConfig_M%d_efC%d", M, efConstruction)) :
            null;

        int totalRuns = 0;
        for (var topK : topKGrid.keySet()) {
            checkAbort(shouldAbort);
            int overqueryCount = topKGrid.get(topK).size();
            totalRuns += overqueryCount * usePruningGrid.size();
        }

        QuerySuiteTask suiteTask = new QuerySuiteTask(totalRuns, M, efConstruction);
        StatusTracker<QuerySuiteTask> suiteTracker;
        if (parentTracker != null && testConfigScope != null) {
            suiteTracker = parentTracker.executeWithContext(() -> testConfigScope.track(suiteTask));
        } else if (parentTracker != null) {
            suiteTracker = parentTracker.createChild(suiteTask);
        } else if (testConfigScope != null) {
            suiteTracker = testConfigScope.track(suiteTask);
        } else {
            suiteTracker = null;
        }

        try (StatusTracker<QuerySuiteTask> ignored = suiteTracker) {
            if (suiteTracker != null) {
                suiteTask.start();
            }

            // 2) Setup benchmark table for printing
            for (var topK : topKGrid.keySet()) {
                checkAbort(shouldAbort);
                for (var usePruning : usePruningGrid) {
                    checkAbort(shouldAbort);
                    BenchmarkTablePrinter printer = new BenchmarkTablePrinter();
                    printer.printConfig(Map.of(
                            "M",                  M,
                            "efConstruction",     efConstruction,
                            "neighborOverflow",   neighborOverflow,
                            "addHierarchy",       addHierarchy,
                            "usePruning",         usePruning
                    ));
                    for (var overquery : topKGrid.get(topK)) {
                        checkAbort(shouldAbort);
                        int rerankK = (int) (topK * overquery);

                        TrackerScope groupScope = testConfigScope != null ?
                                testConfigScope.createChildScope(String.format("Benchmarks_topK%d_oq%.2f_prune_%s",
                                        topK, overquery, usePruning ? "on" : "off")) :
                                null;

                        BenchmarkPhaseGroupTask groupTask = new BenchmarkPhaseGroupTask(topK, rerankK, usePruning, overquery);
                        StatusTracker<BenchmarkPhaseGroupTask> groupTracker;
                        if (suiteTracker != null && groupScope != null) {
                            groupTracker = suiteTracker.executeWithContext(() -> groupScope.track(groupTask));
                        } else if (suiteTracker != null) {
                            groupTracker = suiteTracker.createChild(groupTask);
                        } else if (groupScope != null) {
                            groupTracker = groupScope.track(groupTask);
                        } else {
                            groupTracker = null;
                        }

                        try (StatusTracker<BenchmarkPhaseGroupTask> ignoredGroup = groupTracker) {
                            if (groupTracker != null) {
                                groupTask.start();
                            }

                            var results = groupScope != null ?
                                    tester.run(groupScope, groupTracker, cs, topK, rerankK, usePruning, queryRuns) :
                                    tester.run(cs, topK, rerankK, usePruning, queryRuns);
                            printer.printRow(overquery, results);

                            if (suiteTracker != null) {
                                suiteTask.advance();
                            }

                            if (groupTracker != null) {
                                groupTask.complete();
                            }
                        } catch (Exception e) {
                            if (groupTracker != null) {
                                groupTask.fail();
                            }
                            throw e;
                        } finally {
                            if (groupScope != null) {
                                groupScope.close();
                            }
                        }
                    }
                    printer.printFooter();
                }
            }

            if (suiteTracker != null) {
                suiteTask.complete();
            }
        } catch (Exception e) {
            if (suiteTracker != null) {
                suiteTask.fail();
            }
            throw e;
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

        for (var benchType : benchmarkSpec.keySet()) {
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

    public static List<BenchResult> runAllAndCollectResults(
            DataSet ds,
            List<Integer> mGrid,
            List<Integer> efConstructionGrid,
            List<Float> neighborOverflowGrid,
            List<Boolean> addHierarchyGrid,
            List<? extends Set<FeatureId>> featureSets,
            List<Function<DataSet, CompressorParameters>> buildCompressors,
            List<Function<DataSet, CompressorParameters>> compressionGrid,
            Map<Integer, List<Double>> topKGrid,
            List<Boolean> usePruningGrid) throws IOException {

        List<BenchResult> results = new ArrayList<>();
        for (int m : mGrid) {
            for (int ef : efConstructionGrid) {
                for (float neighborOverflow : neighborOverflowGrid) {
                    for (boolean addHierarchy : addHierarchyGrid) {
                        for (Set<FeatureId> features : featureSets) {
                            for (Function<DataSet, CompressorParameters> buildCompressor : buildCompressors) {
                                for (Function<DataSet, CompressorParameters> searchCompressor : compressionGrid) {
                                    Path testDirectory = Files.createTempDirectory("bench");
                                    try {
                                        var compressor = getCompressor(buildCompressor, ds);
                                        var searchCompressorObj = getCompressor(searchCompressor, ds);
                                        CompressedVectors cvArg = (searchCompressorObj instanceof CompressedVectors) ? (CompressedVectors) searchCompressorObj : null;
                                        // Create a simple TrackerScope for this standalone build
                                        TrackerScope buildScope = new TrackerScope("StandaloneBuild");
                                        var indexes = buildOnDisk(List.of(features), buildScope, null, m, ef, neighborOverflow, addHierarchy, false, ds, testDirectory, compressor, null);
                                        GraphIndex index = indexes.get(features);
                                        try (ConfiguredSystem cs = new ConfiguredSystem(ds, index, cvArg, features)) {
                                            int queryRuns = 2;
                                            List<QueryBenchmark> benchmarks = List.of(
                                                    (diagnostic_level > 0 ?
                                                            ThroughputBenchmark.createDefault().withDiagnostics(getDiagnosticLevel()) :
                                                            ThroughputBenchmark.createDefault()),
                                                    LatencyBenchmark.createDefault(),
                                                    CountBenchmark.createDefault(),
                                                    AccuracyBenchmark.createDefault()
                                            );
                                            QueryTester tester = new QueryTester(benchmarks);
                                            for (int topK : topKGrid.keySet()) {
                                                for (boolean usePruning : usePruningGrid) {
                                                    for (double overquery : topKGrid.get(topK)) {
                                                        int rerankK = (int) (topK * overquery);
                                                        List<Metric> metricsList = tester.run(cs, topK, rerankK, usePruning, queryRuns);
                                                        Map<String, Object> params = Map.of(
                                                                "M", m,
                                                                "efConstruction", ef,
                                                                "neighborOverflow", neighborOverflow,
                                                                "addHierarchy", addHierarchy,
                                                                "features", features.toString(),
                                                                "buildCompressor", buildCompressor.toString(),
                                                                "searchCompressor", searchCompressor.toString(),
                                                                "topK", topK,
                                                                "overquery", overquery,
                                                                "usePruning", usePruning
                                                        );
                                                        for (Metric metric : metricsList) {
                                                            Map<String, Object> metrics = java.util.Map.of(metric.getHeader(), metric.getValue());
                                                            results.add(new BenchResult(ds.name, params, metrics));
                                                        }
                                                       results.add(new BenchResult(ds.name, params, Map.of("Index Build Time", indexBuildTimes.get(ds.name))));
                                                    }
                                                }
                                            }
                                        } catch (Exception e) {
                                            throw new RuntimeException(e);
                                        }
                                    } finally {
                                        for (int n = 0; n < 1; n++) {
                                            try { Files.deleteIfExists(testDirectory.resolve("graph" + n)); } catch (IOException e) { /* ignore */ }
                                        }
                                        try { Files.deleteIfExists(testDirectory); } catch (IOException e) { /* ignore */ }
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
                return DefaultSearchScoreProvider.exact(queryVector, ds.similarityFunction, ds.getBaseRavv());
            }

            var scoringView = (GraphIndex.ScoringView) view;
            ScoreFunction.ApproximateScoreFunction asf;
            if (features.contains(FeatureId.FUSED_ADC)) {
                asf = scoringView.approximateScoreFunctionFor(queryVector, ds.similarityFunction);
            } else {
                asf = cv.precomputedScoreFunctionFor(queryVector, ds.similarityFunction);
            }
            var rr = scoringView.rerankerFor(queryVector, ds.similarityFunction);
            return new DefaultSearchScoreProvider(asf, rr);
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

    /**
     * Task representing the evaluation of a single graph configuration.
     */
    static class GraphConfigTask implements StatusUpdate.Provider<GraphConfigTask> {
        private final String configName;
        private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;
        private volatile double progress = 0.0;

        GraphConfigTask(String configName) {
            this.configName = configName;
        }

        void start() {
            state = StatusUpdate.RunState.RUNNING;
        }

        void complete() {
            progress = 1.0;
            state = StatusUpdate.RunState.SUCCESS;
        }

        void fail() {
            state = StatusUpdate.RunState.FAILED;
        }

        @Override
        public StatusUpdate<GraphConfigTask> getTaskStatus() {
            return new StatusUpdate<>(progress, state, this);
        }

        @Override
        public String toString() {
            return "Configuration: " + configName;
        }
    }

    /**
     * Task representing all query benchmarks executed for a single configuration.
     */
    static class QuerySuiteTask implements StatusUpdate.Provider<QuerySuiteTask> {
        private final int totalRuns;
        private final int m;
        private final int efConstruction;
        private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;
        private volatile double progress = 0.0;
        private int completedRuns = 0;

        QuerySuiteTask(int totalRuns, int m, int efConstruction) {
            this.totalRuns = totalRuns;
            this.m = m;
            this.efConstruction = efConstruction;
        }

        void start() {
            state = StatusUpdate.RunState.RUNNING;
            progress = totalRuns == 0 ? 1.0 : 0.0;
        }

        void advance() {
            if (totalRuns == 0) {
                progress = 1.0;
                return;
            }
            completedRuns++;
            progress = Math.min(1.0, (double) completedRuns / totalRuns);
        }

        void complete() {
            progress = 1.0;
            state = StatusUpdate.RunState.SUCCESS;
        }

        void fail() {
            state = StatusUpdate.RunState.FAILED;
        }

        @Override
        public StatusUpdate<QuerySuiteTask> getTaskStatus() {
            return new StatusUpdate<>(progress, state, this);
        }

        @Override
        public String toString() {
            return String.format("Search Benchmarks (M=%d, efC=%d) %d/%d", m, efConstruction, completedRuns, Math.max(totalRuns, 1));
        }
    }

    /**
     * Groups all benchmark variants (throughput, latency, count, accuracy) executed for a specific
     * combination of query parameters.
     */
    static class BenchmarkPhaseGroupTask implements StatusUpdate.Provider<BenchmarkPhaseGroupTask> {
        private final int topK;
        private final int rerankK;
        private final boolean usePruning;
        private final double overquery;
        private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;
        private volatile double progress = 0.0;

        BenchmarkPhaseGroupTask(int topK, int rerankK, boolean usePruning, double overquery) {
            this.topK = topK;
            this.rerankK = rerankK;
            this.usePruning = usePruning;
            this.overquery = overquery;
        }

        void start() {
            state = StatusUpdate.RunState.RUNNING;
            progress = 0.0;
        }

        void complete() {
            progress = 1.0;
            state = StatusUpdate.RunState.SUCCESS;
        }

        void fail() {
            state = StatusUpdate.RunState.FAILED;
        }

        @Override
        public StatusUpdate<BenchmarkPhaseGroupTask> getTaskStatus() {
            return new StatusUpdate<>(progress, state, this);
        }

        @Override
        public String toString() {
            return String.format("Benchmark Set (topK=%d, rerankK=%d, oq=%.2f, pruning=%s)",
                    topK, rerankK, overquery, usePruning ? "on" : "off");
        }
    }

    private static void checkAbort(BooleanSupplier shouldAbort) {
        if (shouldAbort != null && shouldAbort.getAsBoolean()) {
            throw new BenchmarkAbortedException();
        }
    }

    static final class BenchmarkAbortedException extends RuntimeException {
        BenchmarkAbortedException() {
            super("Benchmark aborted by user");
        }
    }

    /**
     * Task representing processing of a single dataset
     */
    static class DatasetTask implements StatusUpdate.Provider<DatasetTask> {
        private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;
        private volatile double progress = 0.0;
        private final String datasetName;

        public DatasetTask(String datasetName) {
            this.datasetName = datasetName;
        }

        public void start() {
            this.state = StatusUpdate.RunState.RUNNING;
        }

        public void updateProgress(double p) {
            this.progress = p;
        }

        public void complete() {
            this.progress = 1.0;
            this.state = StatusUpdate.RunState.SUCCESS;
        }

        public void fail() {
            this.state = StatusUpdate.RunState.FAILED;
        }

        @Override
        public StatusUpdate<DatasetTask> getTaskStatus() {
            return new StatusUpdate<>(progress, state, this);
        }

        @Override
        public String toString() {
            return "Dataset: " + datasetName;
        }
    }
}
