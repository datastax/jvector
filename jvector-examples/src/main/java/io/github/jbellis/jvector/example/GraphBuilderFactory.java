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

import io.github.jbellis.jvector.example.util.FilteredForkJoinPool;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.diagnostics.ASHBuildScoreExperiment;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.VectorAccess;
import io.github.jbellis.jvector.graph.disk.AbstractGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OnDiskParallelGraphIndexWriter;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.RandomAccessOnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.BQVectors;
import io.github.jbellis.jvector.quantization.ASHVectors;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 * The only place the benchmark harness constructs core graph builders and writers, so a change to
 * those constructors is absorbed here.
 */
final class GraphBuilderFactory {
    private GraphBuilderFactory() {}

    /** A builder scoring with exact similarity over full-precision vectors. */
    static GraphIndexBuilder forFullPrecision(RandomAccessVectorValues vectors,
                                              VectorSimilarityFunction similarity,
                                              BuildConfig build) {
        var bsp = BuildScoreProvider.randomAccessScoreProvider(vectors, similarity);
        return new GraphIndexBuilder(bsp,
                                     vectors.dimension(),
                                     build.M,
                                     build.efConstruction,
                                     build.neighborOverflow,
                                     build.alpha,
                                     build.addHierarchy,
                                     build.refineFinalGraph,
                                     constructionPool(build),
                                     hasCustomWorkers(build) ? FilteredForkJoinPool.createFilteredPool(build.workers)
                                             : FilteredForkJoinPool.createFilteredPool());
    }

    private static final java.util.concurrent.ConcurrentHashMap<Integer, java.util.concurrent.ForkJoinPool> CONSTRUCTION_POOLS =
            new java.util.concurrent.ConcurrentHashMap<>();

    /** True when construction.workers asks for a thread count other than the platform's physical cores. */
    static boolean hasCustomWorkers(BuildConfig build) {
        return build.workers != null && build.workers != PhysicalCoreExecutor.getPhysicalCoreCount();
    }

    /** Pool for construction work: the platform pool, or one sized to construction.workers. */
    static java.util.concurrent.ForkJoinPool constructionPool(BuildConfig build) {
        return hasCustomWorkers(build)
                ? CONSTRUCTION_POOLS.computeIfAbsent(build.workers, java.util.concurrent.ForkJoinPool::new)
                : PhysicalCoreExecutor.pool();
    }

    /** Thread count for construction work, including the parallel writer. */
    static int constructionThreads(BuildConfig build) {
        return build.workers != null ? build.workers : PhysicalCoreExecutor.getPhysicalCoreCount();
    }

    /** Runs training or encoding in the construction pool, so its parallel work uses those threads. */
    static <T> T inConstructionPool(BuildConfig build, Supplier<T> work) {
        return hasCustomWorkers(build) ? constructionPool(build).submit(work::get).join() : work.get();
    }

    /** A builder scoring with PQ-, BQ- or ASH-encoded vectors. */
    static GraphIndexBuilder forCompressed(CompressedVectors encoded,
                                           VectorSimilarityFunction similarity,
                                           int dimension,
                                           BuildConfig build) {
        return forCompressed(compressedScores(encoded, similarity, null), dimension, build);
    }

    static BuildScoreProvider compressedScores(CompressedVectors encoded, VectorSimilarityFunction similarity,
                                               RandomAccessVectorValues raw) {
        return compressedScores(encoded, similarity, raw, ASHBuildScoreExperiment.defaultMode());
    }

    private static BuildScoreProvider compressedScores(CompressedVectors encoded, VectorSimilarityFunction similarity,
                                                        RandomAccessVectorValues raw, String ashMode) {
        BuildScoreProvider bsp;
        if (encoded instanceof PQVectors) {
            bsp = BuildScoreProvider.pqBuildScoreProvider(similarity, (PQVectors) encoded);
        } else if (encoded instanceof BQVectors) {
            bsp = BuildScoreProvider.bqBuildScoreProvider((BQVectors) encoded);
        } else if (encoded instanceof ASHVectors) {
            bsp = ASHBuildScoreExperiment.create(similarity, (ASHVectors) encoded, raw, ashMode);
        } else {
            throw new IllegalArgumentException("Construction supports PQ, BQ or ASH build compression");
        }
        return bsp;
    }

    static GraphIndexBuilder forCompressed(BuildScoreProvider bsp, int dimension, BuildConfig build) {
        if (hasCustomWorkers(build)) {
            return new GraphIndexBuilder(bsp, dimension, build.M, build.efConstruction, build.neighborOverflow,
                                         build.alpha, build.addHierarchy, build.refineFinalGraph,
                                         constructionPool(build), FilteredForkJoinPool.createFilteredPool(build.workers));
        }
        return new GraphIndexBuilder(bsp, dimension, build.M, build.efConstruction, build.neighborOverflow,
                                     build.alpha, build.addHierarchy, build.refineFinalGraph);
    }

    static String ashConstructionMode(BuildConfig build) {
        return build.ashConstructionMode == null ? ASHBuildScoreExperiment.defaultMode() : build.ashConstructionMode;
    }

    /** Builds and cleans up the topology before any disk features are written. */
    static ImmutableGraphIndex buildGraph(DataSet ds, BuildConfig build, CompressedVectors encoded) throws IOException {
        var vectors = ds.getBaseRavv();
        var similarity = ds.getSimilarityFunction();
        if (encoded == null) {
            try (var builder = forFullPrecision(vectors, similarity, build);
                 var progress = PhaseProgress.counting("Graph construction", vectors.size(),
                         () -> builder.getGraph().size(0))) {
                if (!ds.useDatasetPrefetching()) return builder.build(vectors);
                VectorAccess.forEach(vectors, constructionPool(build), builder::addGraphNode);
                builder.cleanup();
                return builder.getGraph();
            }
        }
        String ashMode = ashConstructionMode(build);
        var scores = compressedScores(encoded, similarity, vectors, ashMode);
        boolean symmetricInsertion = encoded instanceof ASHVectors && "symmetric-all".equals(ashMode);
        try (var builder = forCompressed(scores, vectors.dimension(), build)) {
            try (var progress = PhaseProgress.counting("Graph insertion", vectors.size(),
                    () -> builder.getGraph().size(0))) {
                if (ds.useDatasetPrefetching()) {
                    VectorAccess.forEach(vectors, constructionPool(build), (node, vector) -> {
                        if (symmetricInsertion) builder.addGraphNode(node, scores.searchProviderFor(node));
                        else builder.addGraphNode(node, vector);
                    });
                } else {
                    // Insertion borrows the vector, so each thread reads rows into one reused vector.
                    var views = vectors.threadLocalSupplier();
                    var rows = ThreadLocal.withInitial(() -> VectorizationProvider.getInstance()
                            .getVectorTypeSupport().createFloatVector(vectors.dimension()));
                    constructionPool(build).submit(() ->
                            IntStream.range(0, vectors.size()).parallel().forEach(node -> {
                                if (symmetricInsertion) {
                                    builder.addGraphNode(node, scores.searchProviderFor(node));
                                } else {
                                    var row = rows.get();
                                    views.get().getVectorInto(node, row, 0);
                                    builder.addGraphNode(node, row);
                                }
                            })).join();
                }
            }
            try (var progress = PhaseProgress.timed("Graph cleanup")) {
                builder.cleanup();
            }
            return builder.getGraph();
        }
    }

    static AbstractGraphIndexWriter.Builder<? extends RandomAccessOnDiskGraphIndexWriter, ?> writerBuilder(
            ImmutableGraphIndex graph, Path outPath, int size, String mode) throws FileNotFoundException {
        return writerBuilder(graph, outPath, size, mode, PhysicalCoreExecutor.getPhysicalCoreCount());
    }

    /** As above, with {@code threads} parallel writer workers. */
    static AbstractGraphIndexWriter.Builder<? extends RandomAccessOnDiskGraphIndexWriter, ?> writerBuilder(
            ImmutableGraphIndex graph, Path outPath, int size, String mode, int threads) throws FileNotFoundException {
        if (mode == null) return writerBuilder(graph, outPath, size);
        AbstractGraphIndexWriter.Builder<? extends RandomAccessOnDiskGraphIndexWriter, ?> builder;
        if ("sequential".equals(mode)) {
            builder = new OnDiskGraphIndexWriter.Builder(graph, outPath);
        } else if ("parallel".equals(mode)) {
            builder = new OnDiskParallelGraphIndexWriter.Builder(graph, outPath)
                    .withParallelWorkerThreads(threads);
        } else {
            throw new IllegalArgumentException("Unknown writer mode: " + mode);
        }
        builder.withMapper(new OrdinalMapper.IdentityMapper(size - 1));
        return builder;
    }

    /** A writer builder with identity ordinals and one parallel worker per physical core. */
    static RandomAccessOnDiskGraphIndexWriter.Builder writerBuilder(ImmutableGraphIndex graph, Path outPath, int size)
            throws FileNotFoundException {
        var builder = new RandomAccessOnDiskGraphIndexWriter.Builder(graph, outPath);
        builder.withMapper(new OrdinalMapper.IdentityMapper(size - 1));
        builder.withParallelWorkerThreads(PhysicalCoreExecutor.getPhysicalCoreCount());
        return builder;
    }
}
