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

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
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
                                     PhysicalCoreExecutor.pool(),
                                     FilteredForkJoinPool.createFilteredPool());
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
            try (var builder = forFullPrecision(vectors, similarity, build)) {
                if (!ds.useDatasetPrefetching()) return builder.build(vectors);
                VectorAccess.forEach(vectors, PhysicalCoreExecutor.pool(), builder::addGraphNode);
                builder.cleanup();
                return builder.getGraph();
            }
        }
        String ashMode = ashConstructionMode(build);
        var scores = compressedScores(encoded, similarity, vectors, ashMode);
        boolean symmetricInsertion = encoded instanceof ASHVectors && "symmetric-all".equals(ashMode);
        try (var builder = forCompressed(scores, vectors.dimension(), build)) {
            if (ds.useDatasetPrefetching()) {
                VectorAccess.forEach(vectors, PhysicalCoreExecutor.pool(), (node, vector) -> {
                    if (symmetricInsertion) builder.addGraphNode(node, scores.searchProviderFor(node));
                    else builder.addGraphNode(node, vector);
                });
            } else {
                var views = vectors.threadLocalSupplier();
                PhysicalCoreExecutor.pool().submit(() ->
                        IntStream.range(0, vectors.size()).parallel().forEach(node -> {
                            if (symmetricInsertion) builder.addGraphNode(node, scores.searchProviderFor(node));
                            else builder.addGraphNode(node, views.get().getVector(node));
                        })).join();
            }
            builder.cleanup();
            return builder.getGraph();
        }
    }

    static AbstractGraphIndexWriter.Builder<? extends RandomAccessOnDiskGraphIndexWriter, ?> writerBuilder(
            ImmutableGraphIndex graph, Path outPath, int size, String mode) throws FileNotFoundException {
        if (mode == null) return writerBuilder(graph, outPath, size);
        AbstractGraphIndexWriter.Builder<? extends RandomAccessOnDiskGraphIndexWriter, ?> builder;
        if ("sequential".equals(mode)) {
            builder = new OnDiskGraphIndexWriter.Builder(graph, outPath);
        } else if ("parallel".equals(mode)) {
            builder = new OnDiskParallelGraphIndexWriter.Builder(graph, outPath)
                    .withParallelWorkerThreads(PhysicalCoreExecutor.getPhysicalCoreCount());
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
