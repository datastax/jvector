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
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.util.OnDiskGraphIndexCache;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.VectorAccess;
import io.github.jbellis.jvector.graph.VectorMean;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskParallelGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.RandomAccessOnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.quantization.BQVectors;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.ASHVectors;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 * Builds the graph from PQ-, BQ- or ASH-encoded vectors while writing each feature set to disk. Feature sets
 * already in the index cache are loaded, and only the rest are built.
 */
final class CompressedOnDiskBuild implements IndexBuildStrategy {
    private final OnDiskGraphIndexCache cache;
    private final Path workDirectory;
    private VectorCompressor<?> buildCompressor;
    private final Supplier<VectorCompressor<?>> buildCodec;
    private final ConstructionMetrics constructionMetrics;
    private final Supplier<VectorCompressor<?>> fusedCompressor;
    private final String cacheIdentity;

    /** @param constructionMetrics construction metrics sink, or null when not reporting */
    CompressedOnDiskBuild(OnDiskGraphIndexCache cache,
                          Path workDirectory,
                          VectorCompressor<?> buildCompressor,
                          ConstructionMetrics constructionMetrics) {
        this(cache, workDirectory, buildCompressor, constructionMetrics, null, null, null);
    }

    CompressedOnDiskBuild(OnDiskGraphIndexCache cache, Path workDirectory, Supplier<VectorCompressor<?>> buildCodec,
                          ConstructionMetrics constructionMetrics, Supplier<VectorCompressor<?>> fusedCompressor,
                          String cacheIdentity) {
        this(cache, workDirectory, null, constructionMetrics, buildCodec, fusedCompressor, cacheIdentity);
    }

    private CompressedOnDiskBuild(OnDiskGraphIndexCache cache, Path workDirectory, VectorCompressor<?> buildCompressor,
                                  ConstructionMetrics constructionMetrics, Supplier<VectorCompressor<?>> buildCodec,
                                  Supplier<VectorCompressor<?>> fusedCompressor, String cacheIdentity) {
        this.buildCodec = buildCodec;
        this.fusedCompressor = fusedCompressor;
        this.cacheIdentity = cacheIdentity;
        this.cache = cache;
        this.workDirectory = workDirectory;
        this.buildCompressor = buildCompressor;
        this.constructionMetrics = constructionMetrics;
    }

    VectorCompressor<?> buildCompressor() {
        return buildCompressor;
    }

    @Override
    public BuiltIndexes build(List<? extends Set<FeatureId>> featureSets, BuildConfig build, DataSet ds) throws IOException {
        Map<Set<FeatureId>, ImmutableGraphIndex> indexes = new LinkedHashMap<>();
        Map<Set<FeatureId>, Long> fileSizes = new LinkedHashMap<>();

        // If cache is disabled, we use the (tmp) workDirectory as the output
        Path outputDir = cache.isEnabled() ? cache.cacheDir().toAbsolutePath() : workDirectory;

        List<Set<FeatureId>> missing = new ArrayList<>();
        // Map feature sets to their cache handles for the build phase
        Map<Set<FeatureId>, OnDiskGraphIndexCache.WriteHandle> handles = new LinkedHashMap<>();

        try {
            for (Set<FeatureId> fs : featureSets) {
                var key = cacheIdentity == null
                        ? cache.key(ds.getName(), fs, build.M, build.efConstruction, build.neighborOverflow, build.alpha,
                                build.addHierarchy, build.refineFinalGraph, buildCompressor)
                        : cache.keyWithCompressorId(ds.getName(), fs, build.M, build.efConstruction, build.neighborOverflow,
                                build.alpha, build.addHierarchy, build.refineFinalGraph, cacheIdentity);
                var cached = cache.tryLoad(key);

                if (cached.isPresent()) {
                    System.out.printf("%s: Using cached graph index for %s%n", key.datasetName, fs);
                    indexes.put(fs, cached.get());
                    try { fileSizes.put(fs, Files.size(cache.resolve(key).finalPath)); } catch (IOException ignored) {}
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

            Double buildTimeS = missing.isEmpty() ? null
                    : buildMissing(missing, build, ds, outputDir, handles, indexes, fileSizes);
            return new BuiltIndexes(indexes, fileSizes, buildTimeS);
        } catch (IOException | RuntimeException e) {
            // A failed build must release cached readers and discard uncommitted writes.
            for (var handle : handles.values()) {
                try { handle.close(); }
                catch (IOException suppressed) { e.addSuppressed(suppressed); }
            }
            for (var index : indexes.values()) {
                try { index.close(); }
                catch (IOException suppressed) { e.addSuppressed(suppressed); }
            }
            throw e;
        }
    }

    /** Builds one graph for {@code featureSets}, loads each written index into {@code indexes}, and returns the build time. */
    private double buildMissing(List<Set<FeatureId>> featureSets,
                                BuildConfig build,
                                DataSet ds,
                                Path outputDir,
                                Map<Set<FeatureId>, OnDiskGraphIndexCache.WriteHandle> handles,
                                Map<Set<FeatureId>, ImmutableGraphIndex> indexes,
                                Map<Set<FeatureId>, Long> fileSizes) throws IOException
    {
        Files.createDirectories(outputDir);

        if (build.indexType != null) {
            buildCompressor = buildCodec == null ? null : GraphBuilderFactory.inConstructionPool(build, buildCodec::get);
            double seconds = buildAndFlush(featureSets, build, ds, outputDir, handles);
            long reopenStart = System.nanoTime();
            reopen(featureSets, outputDir, handles, indexes, fileSizes);
            if (constructionMetrics != null) constructionMetrics.reopenTimeS = (System.nanoTime() - reopenStart) / 1e9;
            return seconds;
        }

        var floatVectors = ds.getBaseRavv();

        String quantType = buildCompressor instanceof ProductQuantization ? "PQ"
                : buildCompressor instanceof AsymmetricHashing ? "ASH" : "BQ";
        VectorMean mean = featureSets.stream().anyMatch(f -> f.contains(FeatureId.NVQ_VECTORS))
                ? new VectorMean(floatVectors.size(), floatVectors.dimension()) : null;
        var encodingVectors = mean == null ? floatVectors : mean.duringScan(floatVectors);
        var encoded = constructionMetrics == null ? buildCompressor.encodeAll(encodingVectors)
                : constructionMetrics.index(quantType).timeEncode(() -> buildCompressor.encodeAll(encodingVectors));
        if (!(encoded instanceof PQVectors) && !(encoded instanceof BQVectors) && !(encoded instanceof ASHVectors))
            throw new IllegalArgumentException("Disk construction supports PQ, BQ or ASH build compression");
        var inputMean = mean != null && mean.isComplete() ? mean.mean() : null;
        PQVectors pq = encoded instanceof PQVectors ? (PQVectors) encoded : null;
        ASHVectors ash = encoded instanceof ASHVectors ? (ASHVectors) encoded : null;
        if (ash != null && Boolean.getBoolean("jvector.ash.debugSymmetric")) {
            io.github.jbellis.jvector.example.diagnostics.ASHScoreDebug.run(floatVectors, ash,
                    Integer.getInteger("jvector.ash.debugSymmetric.pairs", 2000),
                    Integer.getInteger("jvector.ash.debugSymmetric.subsetN", 50000),
                    Integer.getInteger("jvector.ash.debugSymmetric.topK", 100), new java.util.Random(123));
        }
        for (var features : featureSets) for (var id : features) FeatureProviders.require(id).checkCompressedBuild(encoded);
        boolean allSymmetric = ash != null && "symmetric-all".equals(
                System.getProperty("jvector.bench.ashConstruction", "symmetric-all"));
        // Compressed scores only guide construction; searches rerank with a feature's stored vectors.
        if (featureSets.stream().noneMatch(fs -> FeatureProviders.anyMatch(fs, FeatureProvider::providesRerankVectors))) {
            throw new IllegalStateException("Bench looks for either NVQ_VECTORS or INLINE_VECTORS feature set for scoring compressed builds.");
        }
        long providerStart = System.nanoTime();
        var bsp = GraphBuilderFactory.compressedScores(encoded, ds.getSimilarityFunction(), floatVectors);
        var builder = GraphBuilderFactory.forCompressed(bsp, floatVectors.dimension(), build);
        System.out.printf("Build score provider setup: %.6f seconds%n", (System.nanoTime() - providerStart) / 1e9);

        Map<Set<FeatureId>, RandomAccessOnDiskGraphIndexWriter> writers = new LinkedHashMap<>();
        Map<Set<FeatureId>, FeatureWriteContext> contexts = new LinkedHashMap<>();
        int n = 0;
        try {
            for (var features : featureSets) {
                // if we are using index caching, use cache names instead of tmp names for index files....
                Path graphPath;
                if (handles.containsKey(features)) {
                    graphPath = handles.get(features).writePath();
                } else {
                    graphPath = outputDir.resolve("graph" + n++);
                }
                var ctx = FeatureProviders.configureWriter(features, builder.getGraph(), graphPath, floatVectors, pq, ash, inputMean, constructionMetrics);
                writers.put(features, ctx.buildWriter());
                contexts.put(features, ctx);
            }
        } catch (IOException | RuntimeException e) {
            // Close the writers opened so far rather than leak their files when a later one fails.
            for (var writer : writers.values()) {
                try { writer.close(); }
                catch (IOException suppressed) { e.addSuppressed(suppressed); }
            }
            throw e;
        }

        // The parallel writer calls final suppliers from its workers, so original-vector features
        // can be encoded after cleanup. The sequential writer calls them on one thread, so it keeps
        // encoding on the insertion threads and writing each node's features inline.
        boolean deferFeatures = writers.values().stream()
                .allMatch(writer -> writer instanceof OnDiskParallelGraphIndexWriter);
        long startTime = System.nanoTime();
        if (ds.useDatasetPrefetching()) {
            VectorAccess.forEach(floatVectors, PhysicalCoreExecutor.pool(), (node, vector) -> {
                if (!deferFeatures) writeInlineStates(writers, contexts, node, vector);
                if (allSymmetric) builder.addGraphNode(node, bsp.searchProviderFor(node));
                else builder.addGraphNode(node, vector);
            });
        } else {
            var vv = floatVectors.threadLocalSupplier();
            PhysicalCoreExecutor.pool().submit(() -> {
                IntStream.range(0, floatVectors.size()).parallel().forEach(node -> {
                    var vector = vv.get().getVector(node);
                    if (!deferFeatures) writeInlineStates(writers, contexts, node, vector);
                    if (allSymmetric) builder.addGraphNode(node, bsp.searchProviderFor(node));
                    else builder.addGraphNode(node, vector);
                });
            }).join();
        }
        long insertionEnd = System.nanoTime();
        builder.cleanup();
        long cleanupEnd = System.nanoTime();

        // With deferred features, supply every feature so the parallel writer emits complete records.
        writers.entrySet().stream().parallel().forEach(entry -> {
            var writer = entry.getValue();
            var ctx = contexts.get(entry.getKey());
            try (var vectors = new FinalWriteVectors(floatVectors);
                 var view = builder.getGraph().getView();
                 var closeWriter = writer) {
                writer.write(ctx.completedSuppliers(view, deferFeatures ? vectors::get : null));
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
        builder.close();
        double totalTime = (System.nanoTime() - startTime) / 1_000_000_000.0;
        System.out.printf("Graph phases: insertion=%.6f cleanup=%.6f flush=%.6f seconds%n",
                (insertionEnd-startTime)/1e9, (cleanupEnd-insertionEnd)/1e9,
                (System.nanoTime()-cleanupEnd)/1e9);
        System.out.format("Build and write %s in %ss%n", featureSets, totalTime);

        reopen(featureSets, outputDir, handles, indexes, fileSizes);
        return totalTime;
    }

    /** Builds the topology before preparing search features and writing the disk index. */
    private double buildAndFlush(List<Set<FeatureId>> featureSets, BuildConfig build, DataSet ds, Path outputDir,
                                 Map<Set<FeatureId>, OnDiskGraphIndexCache.WriteHandle> handles) throws IOException {
        var vectors = ds.getBaseRavv();
        var mean = featureSets.stream().anyMatch(features -> features.contains(FeatureId.NVQ_VECTORS))
                ? new VectorMean(vectors.size(), vectors.dimension()) : null;
        var buildVectors = mean == null ? vectors : mean.duringScan(vectors);
        var encoded = Grid.encodeForConstruction(buildCompressor, buildVectors, constructionMetrics,
                GraphBuilderFactory.constructionPool(build));

        long start = System.nanoTime();
        try (var graph = GraphBuilderFactory.buildGraph(ds, build, encoded)) {
            double buildSeconds = (System.nanoTime() - start) / 1e9;
            var codec = fusedCompressor == null ? null
                    : GraphBuilderFactory.inConstructionPool(build, fusedCompressor::get);
            CompressedVectors searchCodes = codec == null ? null : codec == buildCompressor
                    ? encoded
                    : Grid.encodeForConstruction(codec, mean != null && !mean.isComplete() ? buildVectors : vectors,
                            constructionMetrics, GraphBuilderFactory.constructionPool(build));
            var inputMean = mean != null && mean.isComplete() ? mean.mean() : null;
            double flushSeconds = 0;
            int n = 0;
            for (var features : featureSets) {
                Path path = handles.containsKey(features) ? handles.get(features).writePath() : outputDir.resolve("graph" + n++);
                var ctx = FeatureProviders.configureWriter(features, graph, path, vectors,
                        searchCodes instanceof PQVectors ? (PQVectors) searchCodes : null,
                        searchCodes instanceof ASHVectors ? (ASHVectors) searchCodes : null,
                        inputMean, constructionMetrics, build.writerMode, GraphBuilderFactory.constructionThreads(build));
                start = System.nanoTime();
                try (var progress = PhaseProgress.timed("Index write " + features);
                     var input = new FinalWriteVectors(vectors);
                     var view = graph.getView();
                     var writer = ctx.buildWriter()) {
                    writer.write(ctx.completedSuppliers(view, input::get));
                }
                flushSeconds += (System.nanoTime() - start) / 1e9;
            }
            double totalTime = buildSeconds + flushSeconds;
            if (constructionMetrics != null) {
                constructionMetrics.graphBuildTimeS = buildSeconds;
                constructionMetrics.flushTimeS = flushSeconds;
            }
            return totalTime;
        }
    }

    private static void reopen(List<Set<FeatureId>> featureSets, Path outputDir,
                               Map<Set<FeatureId>, OnDiskGraphIndexCache.WriteHandle> handles,
                               Map<Set<FeatureId>, ImmutableGraphIndex> indexes,
                               Map<Set<FeatureId>, Long> fileSizes) throws IOException {
        // Commit the index to cache atomically
        for (var entry : handles.entrySet()) {
            System.out.printf("Committing %s build to graph index cache...%n", entry.getKey());
            entry.getValue().commit();
        }

        // open indexes and capture per-feature-set file sizes
        int n = 0;
        for (var features : featureSets) {
            Path loadPath = handles.containsKey(features)
                    ? handles.get(features).finalPath()
                    : outputDir.resolve("graph" + n++);

            indexes.put(features, OnDiskGraphIndex.load(ReaderSupplierFactory.open(loadPath)));
            fileSizes.put(features, Files.size(loadPath));
        }
    }

    /// Write one node's original-vector features to every writer during insertion. The borrowed
    /// vector is encoded and written before the caller advances to another vector.
    private static void writeInlineStates(
            Map<Set<FeatureId>, RandomAccessOnDiskGraphIndexWriter> writers,
            Map<Set<FeatureId>, FeatureWriteContext> contexts,
            int node, VectorFloat<?> vector) {
        writers.forEach((features, writer) -> {
            var states = new EnumMap<FeatureId, Feature.State>(FeatureId.class);
            contexts.get(features).vectorSuppliers().forEach((id, supplier) -> states.put(id, supplier.apply(vector)));
            try { writer.writeFeaturesInline(node, states); }
            catch (IOException e) { throw new UncheckedIOException(e); }
        });
    }
}
