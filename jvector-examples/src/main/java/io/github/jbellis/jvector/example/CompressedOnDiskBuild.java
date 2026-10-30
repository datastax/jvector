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
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskParallelGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.RandomAccessOnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.quantization.BQVectors;
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
import java.util.function.IntFunction;
import java.util.stream.IntStream;

/**
 * Builds the graph from PQ- or BQ-encoded vectors while writing each feature set to disk. Feature sets
 * already in the index cache are loaded, and only the rest are built.
 */
final class CompressedOnDiskBuild implements IndexBuildStrategy {
    private final OnDiskGraphIndexCache cache;
    private final Path workDirectory;
    private final VectorCompressor<?> buildCompressor;
    private final ConstructionMetrics constructionMetrics;

    /** @param constructionMetrics construction metrics sink, or null when not reporting */
    CompressedOnDiskBuild(OnDiskGraphIndexCache cache,
                          Path workDirectory,
                          VectorCompressor<?> buildCompressor,
                          ConstructionMetrics constructionMetrics) {
        this.cache = cache;
        this.workDirectory = workDirectory;
        this.buildCompressor = buildCompressor;
        this.constructionMetrics = constructionMetrics;
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

        for (Set<FeatureId> fs : featureSets) {
            var key = cache.key(ds.getName(), fs, build.M, build.efConstruction, build.neighborOverflow, build.alpha,
                    build.addHierarchy, build.refineFinalGraph, buildCompressor);
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

        Double buildTimeS = null;
        if (!missing.isEmpty()) {
            // At least one index needs to be built (b/c not in cache or cache is disabled)
            // We pass the handles map so buildMissing knows exactly where to write
            buildTimeS = buildMissing(missing, build, ds, outputDir, handles, indexes, fileSizes);
        }
        return new BuiltIndexes(indexes, fileSizes, buildTimeS);
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

        var floatVectors = ds.getBaseRavv();

        // Record the encoding time if asked for....
        var encoded = (constructionMetrics != null)
                ? constructionMetrics.index(buildCompressor instanceof ProductQuantization ? "PQ" : "BQ")
                        .timeEncode(() -> buildCompressor.encodeAll(floatVectors))
                : buildCompressor.encodeAll(floatVectors);
        if (!(encoded instanceof PQVectors) && !(encoded instanceof BQVectors))
            throw new IllegalArgumentException("Disk construction supports PQ or BQ build compression");
        PQVectors pq = encoded instanceof PQVectors ? (PQVectors) encoded : null;
        for (var features : featureSets) {
            for (var id : features) {
                var provider = FeatureProviders.get(id);
                if (provider != null) provider.checkCompressedBuild(pq);
            }
        }
        // Compressed scores only guide construction; searches rerank with a feature's stored vectors.
        if (featureSets.stream().noneMatch(fs -> FeatureProviders.anyMatch(fs, FeatureProvider::providesRerankVectors))) {
            throw new IllegalStateException("Bench looks for either NVQ_VECTORS or INLINE_VECTORS feature set for scoring compressed builds.");
        }
        var builder = GraphBuilderFactory.forCompressed(encoded, ds.getSimilarityFunction(), floatVectors.dimension(), build);

        Map<Set<FeatureId>, RandomAccessOnDiskGraphIndexWriter> writers = new LinkedHashMap<>();
        Map<Set<FeatureId>, FeatureWriteContext> contexts = new LinkedHashMap<>();
        int n = 0;
        for (var features : featureSets) {
            // if we are using index caching, use cache names instead of tmp names for index files....
            Path graphPath;
            if (handles.containsKey(features)) {
                graphPath = handles.get(features).writePath();
            } else {
                graphPath = outputDir.resolve("graph" + n++);
            }
            var ctx = FeatureProviders.configureWriter(features, builder.getGraph(), graphPath, floatVectors, pq, constructionMetrics);
            writers.put(features, ctx.buildWriter());
            contexts.put(features, ctx);
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
                builder.addGraphNode(node, vector);
            });
        } else {
            var vv = floatVectors.threadLocalSupplier();
            PhysicalCoreExecutor.pool().submit(() -> {
                IntStream.range(0, floatVectors.size()).parallel().forEach(node -> {
                    var vector = vv.get().getVector(node);
                    if (!deferFeatures) writeInlineStates(writers, contexts, node, vector);
                    builder.addGraphNode(node, vector);
                });
            }).join();
        }
        builder.cleanup();

        // With deferred features, supply every feature so the parallel writer emits complete records.
        writers.entrySet().stream().parallel().forEach(entry -> {
            var writer = entry.getValue();
            var ctx = contexts.get(entry.getKey());
            try (var vectors = new FinalWriteVectors(floatVectors);
                 var view = builder.getGraph().getView();
                 var closeWriter = writer) {
                Map<FeatureId, IntFunction<Feature.State>> writeSuppliers = new EnumMap<>(FeatureId.class);
                if (deferFeatures) {
                    ctx.vectorSuppliers().forEach((id, supplier) ->
                            writeSuppliers.put(id, ordinal -> supplier.apply(vectors.get(ordinal))));
                }
                ctx.afterBuildSuppliers().forEach((id, supplier) -> writeSuppliers.put(id, supplier.apply(view)));
                writer.write(writeSuppliers);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
        builder.close();
        double totalTime = (System.nanoTime() - startTime) / 1_000_000_000.0;
        System.out.format("Build and write %s in %ss%n", featureSets, totalTime);

        // Commit the index to cache atomically
        for (var entry : handles.entrySet()) {
            System.out.printf("Committing %s build to graph index cache...%n", entry.getKey());
            entry.getValue().commit();
        }

        // open indexes and capture per-feature-set file sizes
        n = 0;
        for (var features : featureSets) {
            Path loadPath = handles.containsKey(features)
                    ? handles.get(features).finalPath()
                    : outputDir.resolve("graph" + n++);

            indexes.put(features, OnDiskGraphIndex.load(ReaderSupplierFactory.open(loadPath)));
            fileSizes.put(features, Files.size(loadPath));
        }
        return totalTime;
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
