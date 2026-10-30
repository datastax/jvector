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
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.RandomAccessOnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.BQVectors;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.FileNotFoundException;
import java.nio.file.Path;

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

    /** A builder scoring with PQ- or BQ-encoded vectors. */
    static GraphIndexBuilder forCompressed(CompressedVectors encoded,
                                           VectorSimilarityFunction similarity,
                                           int dimension,
                                           BuildConfig build) {
        BuildScoreProvider bsp;
        if (encoded instanceof PQVectors) {
            bsp = BuildScoreProvider.pqBuildScoreProvider(similarity, (PQVectors) encoded);
        } else if (encoded instanceof BQVectors) {
            bsp = BuildScoreProvider.bqBuildScoreProvider((BQVectors) encoded);
        } else {
            throw new IllegalArgumentException("Disk construction supports PQ or BQ build compression");
        }
        return new GraphIndexBuilder(bsp, dimension, build.M, build.efConstruction, build.neighborOverflow,
                                     build.alpha, build.addHierarchy, build.refineFinalGraph);
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
