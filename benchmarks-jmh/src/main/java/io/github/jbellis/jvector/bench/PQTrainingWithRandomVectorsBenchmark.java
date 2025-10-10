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
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

/**
 * Benchmark for measuring the performance of Product Quantization training on randomly generated vectors.
 * <p>
 * This benchmark evaluates the time required to compute Product Quantization (PQ) codebooks from
 * a dataset of random vectors. PQ training involves clustering vectors in each subspace using k-means,
 * which is a computationally intensive operation. The benchmark tests various configurations of
 * subspace counts (M) to understand the trade-off between compression ratio and training time.
 * <p>
 * Key aspects measured:
 * <ul>
 *   <li>K-means clustering performance across multiple subspaces</li>
 *   <li>Impact of increasing M (number of subspaces) on training time</li>
 *   <li>Scalability with dataset size</li>
 * </ul>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(1)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Threads(1)
public class PQTrainingWithRandomVectorsBenchmark {
    private static final Logger log = LoggerFactory.getLogger(PQTrainingWithRandomVectorsBenchmark.class);

    /**
     * Creates a new benchmark instance.
     * <p>
     * This constructor is invoked by JMH and should not be called directly.
     */
    public PQTrainingWithRandomVectorsBenchmark() {
    }
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

    /** Random access wrapper for the pre-created vector dataset. */
    private RandomAccessVectorValues ravv;

    /**
     * The number of subspaces for Product Quantization.
     * <p>
     * Higher values of M provide more accurate quantization but increase training time
     * and memory usage. Values: 16, 32, 64
     */
    @Param({"16", "32", "64"})
    private int M;

    /**
     * The dimensionality of the vectors.
     * <p>
     * Default value: 768 (common for many embedding models).
     */
    @Param({"768"})
    int originalDimension;

    /**
     * The number of vectors in the training dataset.
     * <p>
     * Default value: 100000
     */
    @Param({"100000"})
    int vectorCount;

    /**
     * Sets up the benchmark by pre-creating a dataset of random vectors.
     * <p>
     * This method generates the specified number of random vectors with the configured
     * dimensionality. The vectors are wrapped in a RandomAccessVectorValues instance
     * for use during PQ training. Pre-creating all vectors ensures the benchmark
     * measures only the PQ training time, not vector generation.
     *
     * @throws IOException if there is an error during setup
     */
    @Setup
    public void setup() throws IOException {
        log.info("Pre-creating vector dataset with original dimension: {}, vector count: {}", originalDimension, vectorCount);
        final List<VectorFloat<?>> vectors = new ArrayList<>(vectorCount);
        for (int i = 0; i < vectorCount; i++) {
            float[] vector = new float[originalDimension];
            for (int j = 0; j < originalDimension; j++) {
                vector[j] = (float) Math.random();
            }
            VectorFloat<?> floatVector = VECTOR_TYPE_SUPPORT.createFloatVector(vector);
            vectors.add(floatVector);
        }
        // wrap the raw vectors in a RandomAccessVectorValues
        ravv = new ListRandomAccessVectorValues(vectors, originalDimension);
        log.info("Pre-created vector dataset with original dimension: {}, vector count: {}", originalDimension, vectorCount);
    }

    /**
     * Tears down the benchmark state.
     * <p>
     * This method is a placeholder for any cleanup operations that may be needed
     * in future implementations.
     *
     * @throws IOException if there is an error during teardown
     * @throws InterruptedException if the thread is interrupted during teardown
     */
    @TearDown
    public void tearDown() throws IOException, InterruptedException {

    }

    /**
     * Benchmarks the computation of Product Quantization codebooks.
     * <p>
     * This benchmark measures the time required to train a Product Quantization model
     * on the pre-created vector dataset. The training process involves:
     * <ul>
     *   <li>Splitting each vector into M subspaces</li>
     *   <li>Running k-means clustering (256 centroids) in each subspace</li>
     *   <li>Centering the dataset to improve quantization accuracy</li>
     * </ul>
     * The resulting PQ model provides a compression ratio based on M and the original dimension.
     *
     * @param blackhole JMH blackhole to prevent dead code elimination
     * @throws IOException if there is an error during PQ computation
     */
    @Benchmark
    public void productQuantizationComputeBenchmark(Blackhole blackhole) throws IOException {
        // Compress the original vectors using PQ. this represents a compression ratio of 128 * 4 / 16 = 32x
        ProductQuantization pq = ProductQuantization.compute(ravv,
                M, // number of subspaces
                256, // number of centroids per subspace
                true // center the dataset
        );

        blackhole.consume(pq);
    }
}
