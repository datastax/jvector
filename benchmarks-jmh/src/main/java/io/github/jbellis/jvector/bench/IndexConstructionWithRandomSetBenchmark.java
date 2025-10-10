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

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/**
 * JMH benchmark for measuring graph index construction performance using randomly generated vectors.
 * This benchmark evaluates the time required to build a graph index with configurable parameters
 * including vector dimensionality, dataset size, and optional Product Quantization (PQ) compression.
 *
 * <p>The benchmark tests various configurations to assess how different factors affect index
 * construction time, including the impact of using PQ compression during the build process.</p>
 *
 * <p>Key parameters:</p>
 * <ul>
 *   <li>Vector dimensionality: 768 or 1536 dimensions</li>
 *   <li>Dataset size: 100,000 vectors</li>
 *   <li>PQ subspaces: 0 (no compression) or 16 subspaces</li>
 *   <li>Graph degree (M): 32 neighbors per node</li>
 *   <li>Beam width: 100 for construction search</li>
 * </ul>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-preview", "-Djvector.experimental.enable_native_vectorization=false"})
@Warmup(iterations = 2)
@Measurement(iterations = 3)
@Threads(1)
public class IndexConstructionWithRandomSetBenchmark {
    private static final Logger log = LoggerFactory.getLogger(IndexConstructionWithRandomSetBenchmark.class);
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

    /** The vector values to be indexed, initialized during setup. */
    private RandomAccessVectorValues ravv;

    /** The score provider used during graph construction, either exact or PQ-based. */
    private BuildScoreProvider buildScoreProvider;

    /** The maximum degree of the graph (number of neighbors per node). */
    private int M = 32;

    /** The beam width used during graph construction searches. */
    private int beamWidth = 100;

    /** The dimensionality of vectors being indexed. */
    @Param({"768", "1536"})
    private int originalDimension;

    /** The number of vectors in the dataset to be indexed. */
    @Param({/*"10000",*/ "100000"/*, "1000000"*/})
    int numBaseVectors;

    /** The number of PQ subspaces to use, or 0 for no compression. */
    @Param({"0", "16"})
    private int numberOfPQSubspaces;

    /**
     * Constructs a new benchmark instance. JMH will instantiate this class
     * and populate the @Param fields before calling setup methods.
     */
    public IndexConstructionWithRandomSetBenchmark() {
        // JMH-managed lifecycle
    }

    /**
     * Initializes the benchmark state by generating random vectors and configuring
     * the appropriate score provider based on whether PQ compression is enabled.
     *
     * @throws IOException if an error occurs during setup
     */
    @Setup(Level.Trial)
    public void setup() throws IOException {

        final var baseVectors = new ArrayList<VectorFloat<?>>(numBaseVectors);
        for (int i = 0; i < numBaseVectors; i++) {
            VectorFloat<?> vector = createRandomVector(originalDimension);
            baseVectors.add(vector);
        }
        // wrap the raw vectors in a RandomAccessVectorValues
        ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        if (numberOfPQSubspaces > 0) {
            log.info("Using PQ build score provider with original dimension: {}, M: {}, beam width: {}", originalDimension, M, beamWidth);
            final ProductQuantization pq = ProductQuantization.compute(ravv,
                    numberOfPQSubspaces,
                    256,
                    true);
            final PQVectors pqVectors = (PQVectors) pq.encodeAll(ravv);
            buildScoreProvider = BuildScoreProvider.pqBuildScoreProvider(VectorSimilarityFunction.EUCLIDEAN, pqVectors);
        } else {
            log.info("Using Exact build score provider with original dimension: {}, M: {}, beam width: {}", originalDimension, M, beamWidth);
            // score provider using the raw, in-memory vectors
            buildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
        }

    }

    /**
     * Tears down resources after each benchmark invocation.
     * Currently performs no operations but is included for future resource cleanup needs.
     *
     * @throws IOException if an error occurs during teardown
     */
    @TearDown(Level.Invocation)
    public void tearDown() throws IOException {

    }

    /**
     * The main benchmark method that measures the time to build a graph index.
     * Constructs a complete graph index from the configured vectors using the
     * specified parameters and score provider.
     *
     * @param blackhole JMH blackhole to prevent dead code elimination
     * @throws IOException if an error occurs during index construction
     */
    @Benchmark
    public void buildIndexBenchmark(Blackhole blackhole) throws IOException {
        // score provider using the raw, in-memory vectors
        try (final var graphIndexBuilder = new GraphIndexBuilder(buildScoreProvider, ravv.dimension(), M, beamWidth, 1.2f, 1.2f, true)) {
            final var graphIndex = graphIndexBuilder.build(ravv);
            blackhole.consume(graphIndex);
        }
    }

    /**
     * Creates a random vector with the specified dimensionality.
     * Each component is randomly generated using {@link Math#random()}.
     *
     * @param dimension the number of dimensions in the vector
     * @return a newly created random vector
     */
    private VectorFloat<?> createRandomVector(int dimension) {
        VectorFloat<?> vector = VECTOR_TYPE_SUPPORT.createFloatVector(dimension);
        for (int i = 0; i < dimension; i++) {
            vector.set(i, (float) Math.random());
        }
        return vector;
    }
}
