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
import io.github.jbellis.jvector.example.dynamic.DynamicDataset;
import io.github.jbellis.jvector.example.util.AbstractDataset;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.ConfiguredSystem;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Tests a grid of configurations against a dataset
 */
public class GridDynamic {

    private static final String pqCacheDir = "pq_cache";

    static void runAll(DynamicDataset ds,
                       List<Integer> mGrid,
                       List<Integer> efConstructionGrid,
                       List<Float> neighborOverflowGrid,
                       List<Boolean> addHierarchyGrid,
                       List<Boolean> refineFinalGraphGrid,
                       List<? extends Set<FeatureId>> featureSets,
                       List<Function<AbstractDataset, CompressorParameters>> buildCompressors,
                       List<Function<AbstractDataset, CompressorParameters>> compressionGrid,
                       Map<Integer, List<Double>> topKGrid,
                       List<Boolean> usePruningGrid) throws IOException
    {
        try {
            for (var addHierarchy :  addHierarchyGrid) {
                for (var refineFinalGraph : refineFinalGraphGrid) {
                    for (int M : mGrid) {
                        for (float neighborOverflow : neighborOverflowGrid) {
                            for (int efC : efConstructionGrid) {
                                for (var bc : buildCompressors) {
                                    runOneGraph(featureSets, M, efC, neighborOverflow, addHierarchy, refineFinalGraph, bc, compressionGrid, topKGrid, usePruningGrid, ds);
                                }
                            }
                        }
                    }
                }
            }
        } finally {
            cachedCompressors.clear();
        }
    }

    static void runOneGraph(List<? extends Set<FeatureId>> featureSets,
                            int M,
                            int efConstruction,
                            float neighborOverflow,
                            boolean addHierarchy,
                            boolean refineFinalGraph,
                            Function<AbstractDataset, CompressorParameters> buildCompressorFunc,
                            List<Function<AbstractDataset, CompressorParameters>> compressionGrid,
                            Map<Integer, List<Double>> topKGrid,
                            List<Boolean> usePruningGrid,
                            DynamicDataset ds) throws IOException
    {
        var buildCompressor = getCompressor(buildCompressorFunc, ds);

        // TODO This is a temporary hack. We load ALL vectors in memory and create a BuildScoreProvider with them.
        //  In the future, we should support a mutable BuildScoreProvider.
        var floatVectors = ds.getBaseRavv();
        BuildScoreProvider bsp;
        if (buildCompressor != null) {
            var pq = (PQVectors) buildCompressor.encodeAll(floatVectors);
            bsp = BuildScoreProvider.pqBuildScoreProvider(ds.getSimilarityFunction(), pq);
        } else {
            bsp = BuildScoreProvider.randomAccessScoreProvider(floatVectors, ds.getSimilarityFunction());
        }
        GraphIndexBuilder builder = new GraphIndexBuilder(bsp, ds.getDimension(), M, efConstruction, neighborOverflow, 1.2f, addHierarchy, refineFinalGraph);


        for (int epoch = 0; epoch < ds.epochs(); epoch++) {
            var insertions = ds.insertions(epoch);
            var deletions = ds.deletions(epoch);

            // build the graph incrementally
            long startTime = System.nanoTime();
            var vv = floatVectors.threadLocalSupplier();
            // Process insertions
            PhysicalCoreExecutor.pool().submit(() -> {
                insertions.parallelStream().forEach(node -> {
                    builder.addGraphNode(node, vv.get().getVector(node));
                });
            }).join();
            // Process deletions
            PhysicalCoreExecutor.pool().submit(() -> {
                deletions.parallelStream().forEach(node -> {
                    builder.markNodeDeleted(node);
                });
            }).join();
            builder.cleanup();

            System.out.format("Epoch %d built in %ss%n", epoch, (System.nanoTime() - startTime) / 1_000_000_000.0);

            var index = builder.getGraph();

            for (var cpSupplier : compressionGrid) {
                var searchCompressor = getCompressor(cpSupplier, ds);
                CompressedVectors cv;
                if (searchCompressor == null) {
                    cv = null;
                    System.out.format("Uncompressed vectors%n");
                } else {
                    long start = System.nanoTime();
                    cv = searchCompressor.encodeAll(ds.getBaseRavv());
                    System.out.format("%s encoded %d vectors [%.2f MB] in %.2fs%n", searchCompressor, ds.getBaseRavv().size(), (cv.ramBytesUsed() / 1024f / 1024f), (System.nanoTime() - start) / 1_000_000_000.0);
                }

                try (var cs = new ConfiguredSystem.DynamicConfiguredSystem(ds, index, cv, Set.of())) {
                    cs.setEpoch(epoch);
                    testConfiguration(cs, topKGrid, usePruningGrid, M, efConstruction, neighborOverflow, addHierarchy);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
            index.close();
        }
    }

    // avoid recomputing the compressor repeatedly (this is a relatively small memory footprint)
    static final Map<String, VectorCompressor<?>> cachedCompressors = new IdentityHashMap<>();

    private static void testConfiguration(ConfiguredSystem cs,
                                          Map<Integer, List<Double>> topKGrid,
                                          List<Boolean> usePruningGrid,
                                          int M,
                                          int efConstruction,
                                          float neighborOverflow,
                                          boolean addHierarchy) {
        int queryRuns = 2;
        System.out.format("Using %s:%n", cs.getIndex());
        // 1) Select benchmarks to run
        List<QueryBenchmark> benchmarks = List.of(
                ThroughputBenchmark.createDefault(2, 0.1),
                LatencyBenchmark.createDefault(),
                CountBenchmark.createDefault(),
                AccuracyBenchmark.createDefault()
        );
        QueryTester tester = new QueryTester(benchmarks);

        for (var topK : topKGrid.keySet()) {
            for (var usePruning : usePruningGrid) {
                BenchmarkTablePrinter printer = new BenchmarkTablePrinter();
                printer.printConfig(Map.of(
                        "M",                  M,
                        "efConstruction",     efConstruction,
                        "neighborOverflow",   neighborOverflow,
                        "addHierarchy",       addHierarchy,
                        "usePruning",         usePruning
                ));
                for (var overquery : topKGrid.get(topK)) {
                    int rerankK = (int) (topK * overquery);

                    var results = tester.run(cs, topK, rerankK, usePruning, queryRuns);
                    printer.printRow(overquery, results);
                }
                printer.printFooter();
            }
        }
    }

    private static VectorCompressor<?> getCompressor(Function<AbstractDataset, CompressorParameters> cpSupplier, DynamicDataset ds) {
        var cp = cpSupplier.apply(ds);
        if (!cp.supportsCaching()) {
            List<VectorFloat<?>> vectorsEpoch0 = ds.insertions(0).stream().map(ds::getBaseVector).collect(Collectors.toList());
            var ravvEpoch0 = new ListRandomAccessVectorValues(vectorsEpoch0, ds.getDimension());
            return cp.computeCompressor(ravvEpoch0);
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
            List<VectorFloat<?>> vectorsEpoch0 = ds.insertions(0).stream().map(ds::getBaseVector).collect(Collectors.toList());
            var ravvEpoch0 = new ListRandomAccessVectorValues(vectorsEpoch0, ds.getDimension());
            var compressor = cp.computeCompressor(ravvEpoch0);
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
}
