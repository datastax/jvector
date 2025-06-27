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
import io.github.jbellis.jvector.example.util.FilteredForkJoinPool;
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
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
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
import java.util.EnumMap;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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
                       List<Boolean> refineFinalGraphGrid,
                       List<? extends Set<FeatureId>> featureSets,
                       List<Function<DataSet, CompressorParameters>> buildCompressors,
                       List<Function<DataSet, CompressorParameters>> compressionGrid,
                       Map<Integer, List<Double>> topKGrid,
                       List<Boolean> usePruningGrid) throws IOException
    {
        var testDirectory = Files.createTempDirectory(dirPrefix);
        try {
            for (var addHierarchy :  addHierarchyGrid) {
                for (var refineFinalGraph : refineFinalGraphGrid) {
                    for (int M : mGrid) {
                        for (float neighborOverflow : neighborOverflowGrid) {
                            for (int efC : efConstructionGrid) {
                                for (var bc : buildCompressors) {
                                    var compressor = getCompressor(bc, ds);
                                    runOneGraph(featureSets, M, efC, neighborOverflow, addHierarchy, refineFinalGraph, compressor, compressionGrid, topKGrid, usePruningGrid, ds, testDirectory);
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
                            DataSet ds,
                            Path testDirectory) throws IOException
    {
        Map<Set<FeatureId>, GraphIndex> indexes;
        if (buildCompressor == null) {
            indexes = buildInMemory(featureSets, M, efConstruction, neighborOverflow, addHierarchy, refineFinalGraph, ds, testDirectory);
        } else {
            indexes = buildOnDisk(featureSets, M, efConstruction, neighborOverflow, addHierarchy, refineFinalGraph, ds, testDirectory, buildCompressor);
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
                        testConfiguration(cs, topKGrid, usePruningGrid, M, efConstruction, neighborOverflow, addHierarchy);
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
                                                               boolean refineFinalGraph,
                                                               DataSet ds,
                                                               Path testDirectory,
                                                               VectorCompressor<?> buildCompressor)
            throws IOException
    {
        var floatVectors = ds.getBaseRavv();

        var pq = (PQVectors) buildCompressor.encodeAll(floatVectors);
        var bsp = BuildScoreProvider.pqBuildScoreProvider(ds.similarityFunction, pq);
        GraphIndexBuilder builder = new GraphIndexBuilder(bsp, floatVectors.dimension(), M, efConstruction, neighborOverflow, 1.2f, addHierarchy, refineFinalGraph);

        // use the inline vectors index as the score provider for graph construction
        Map<Set<FeatureId>, OnDiskGraphIndexWriter> writers = new HashMap<>();
        Map<Set<FeatureId>, Map<FeatureId, IntFunction<Feature.State>>> suppliers = new HashMap<>();
        OnDiskGraphIndexWriter scoringWriter = null;
        int n = 0;
        for (var features : featureSets) {
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

        // build the graph incrementally
        long startTime = System.nanoTime();
        var vv = floatVectors.threadLocalSupplier();
        PhysicalCoreExecutor.pool().submit(() -> {
            IntStream.range(0, floatVectors.size()).parallel().forEach(node -> {
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
            });
        }).join();
        builder.cleanup();

        // write the edge lists and close the writers
        // if our feature set contains Fused ADC, we need a Fused ADC write-time supplier (as we don't have neighbor information during writeInline)
        writers.entrySet().stream().parallel().forEach(entry -> {
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
                                                                 boolean refineFinalGraph,
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
                                                          refineFinalGraph,
                                                          PhysicalCoreExecutor.pool(),
                                                          FilteredForkJoinPool.createFilteredPool());
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
                                          Map<Integer, List<Double>> topKGrid,
                                          List<Boolean> usePruningGrid,
                                          int M,
                                          int efConstruction,
                                          float neighborOverflow,
                                          boolean addHierarchy) {
        int queryRuns = 2;
        System.out.format("Using %s:%n", cs.index);
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
}
