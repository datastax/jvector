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

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Benchmark that compares the distance calculation of Product Quantized vectors vs full precision vectors.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-preview", "-Djvector.experimental.enable_native_vectorization=false"})
@Warmup(iterations = 2)
@Measurement(iterations = 3)
@Threads(1)
public class PQDistanceCalculationBenchmark {
    private static final Logger log = LoggerFactory.getLogger(PQDistanceCalculationBenchmark.class);

    /**
     * Creates a new benchmark instance.
     * <p>
     * This constructor is invoked by JMH and should not be called directly.
     */
    public PQDistanceCalculationBenchmark() {
    }
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;

    /** The base vectors used for distance calculations. */
    private List<VectorFloat<?>> vectors;

    /** Product-quantized versions of the base vectors, or null if M=0. */
    private PQVectors pqVectors;

    /** Query vectors used to test distance calculations. */
    private List<VectorFloat<?>> queryVectors;

    /** The Product Quantization model, or null if M=0. */
    private ProductQuantization pq;

    /** Score provider configured for either full precision or PQ-based scoring. */
    private BuildScoreProvider buildScoreProvider;

    /**
     * The dimensionality of the vectors.
     * <p>
     * Default value: 1536 (typical for modern embedding models).
     */
    @Param({"1536"})
    private int dimension;

    /**
     * The number of base vectors to create for the dataset.
     * <p>
     * Default value: 10000
     */
    @Param({"10000"})
    private int vectorCount;

    /**
     * The number of query vectors to test against the dataset.
     * <p>
     * Default value: 100
     */
    @Param({"100"})
    private int queryCount;

    /**
     * The number of subspaces for Product Quantization.
     * <p>
     * When M=0, uses full precision vectors without quantization.
     * When M&gt;0, splits each vector into M subspaces for compression.
     * Values: 0 (no PQ), 16, 64, 192
     */
    @Param({"0", "16", "64", "192"})
    private int M;
    

    /**
     * Sets up the benchmark by creating random vectors and configuring score providers.
     * <p>
     * This method creates the specified number of base vectors and query vectors with random
     * values. If M&gt;0, it also computes Product Quantization and creates PQ-encoded vectors.
     * The appropriate score provider is then configured based on whether PQ is used.
     *
     * @throws IOException if there is an error during setup
     */
    @Setup
    public void setup() throws IOException {
        log.info("Creating dataset with dimension: {}, vector count: {}, query count: {}", dimension, vectorCount, queryCount);
        
        // Create random vectors
        vectors = new ArrayList<>(vectorCount);
        for (int i = 0; i < vectorCount; i++) {
            vectors.add(createRandomVector(dimension));
        }
        
        // Create query vectors
        queryVectors = new ArrayList<>(queryCount);
        for (int i = 0; i < queryCount; i++) {
            queryVectors.add(createRandomVector(dimension));
        }
        
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vectors, dimension);
        if (M == 0) {
            buildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(ravv, vsf);
        } else {
            // Create PQ vectors
            pq = ProductQuantization.compute(ravv, M, 256, true);
            pqVectors = (PQVectors) pq.encodeAll(ravv);
            buildScoreProvider = BuildScoreProvider.pqBuildScoreProvider(vsf, pqVectors);
        }
        log.info("Created dataset with dimension: {}, vector count: {}, query count: {}", dimension, vectorCount, queryCount);
    }

    /**
     * Benchmarks distance calculation using cached search score providers.
     * <p>
     * This benchmark measures the performance of calculating distances between query vectors
     * and all base vectors using a search score provider that caches precomputed values for
     * the query vector. This represents the typical search scenario where a query is compared
     * against many candidates.
     *
     * @param blackhole JMH blackhole to prevent dead code elimination
     */
    @Benchmark
    public void cachedDistanceCalculation(Blackhole blackhole) {
        float totalSimilarity = 0;

        for (VectorFloat<?> query : queryVectors) {
            final SearchScoreProvider searchScoreProvider = buildScoreProvider.searchProviderFor(query);
            for (int i = 0; i < vectorCount; i++) {
                float similarity = searchScoreProvider.scoreFunction().similarityTo(i);
                totalSimilarity += similarity;
            }
        }

        blackhole.consume(totalSimilarity);
    }

    /**
     * Benchmarks distance calculation for diversity scoring.
     * <p>
     * This benchmark measures the performance of calculating distances between base vectors
     * using diversity score providers. This represents the scenario where vectors in the
     * dataset are compared against each other to assess diversity, such as during graph
     * construction or result reranking.
     *
     * @param blackhole JMH blackhole to prevent dead code elimination
     */
    @Benchmark
    public void diversityCalculation(Blackhole blackhole) {
        float totalSimilarity = 0;

        for (int q = 0; q < queryCount; q++) {
            for (int i = 0; i < vectorCount; i++) {
                final ScoreFunction sf = buildScoreProvider.diversityProviderFor(i).scoreFunction();
                float similarity = sf.similarityTo(q);
                totalSimilarity += similarity;
            }
        }

        blackhole.consume(totalSimilarity);
    }

    /**
     * Creates a random vector with the specified dimension.
     * <p>
     * Each component of the vector is assigned a random floating-point value
     * between 0.0 (inclusive) and 1.0 (exclusive).
     *
     * @param dimension the number of dimensions for the vector
     * @return a new random vector
     */
    private VectorFloat<?> createRandomVector(int dimension) {
        VectorFloat<?> vector = VECTOR_TYPE_SUPPORT.createFloatVector(dimension);
        for (int i = 0; i < dimension; i++) {
            vector.set(i, (float) Math.random());
        }
        return vector;
    }
}