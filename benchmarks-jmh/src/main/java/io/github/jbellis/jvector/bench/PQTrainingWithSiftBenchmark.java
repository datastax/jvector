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

import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Benchmark for measuring the performance of Product Quantization training on the SIFT dataset.
 * <p>
 * This benchmark evaluates the time required to compute Product Quantization (PQ) codebooks from
 * the SIFT Small dataset, which consists of real-world image feature vectors. Unlike random vectors,
 * SIFT vectors have realistic distributions and correlations, making this benchmark more representative
 * of actual production workloads.
 * <p>
 * The SIFT Small dataset contains:
 * <ul>
 *   <li>10,000 base vectors (128-dimensional)</li>
 *   <li>100 query vectors</li>
 *   <li>Ground truth nearest neighbors for evaluation</li>
 * </ul>
 * <p>
 * Key aspects measured:
 * <ul>
 *   <li>PQ training performance on real-world data with natural clustering</li>
 *   <li>Impact of different M values on training time with realistic vectors</li>
 *   <li>Comparison with random vector training to understand data distribution effects</li>
 * </ul>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(1)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Threads(1)
public class PQTrainingWithSiftBenchmark {
    private static final Logger log = LoggerFactory.getLogger(PQTrainingWithSiftBenchmark.class);

    /**
     * Creates a new benchmark instance.
     * <p>
     * This constructor is invoked by JMH and should not be called directly.
     */
    public PQTrainingWithSiftBenchmark() {
    }

    /** Random access wrapper for the SIFT base vectors. */
    private RandomAccessVectorValues ravv;

    /** The SIFT base vectors used for training. */
    private List<VectorFloat<?>> baseVectors;

    /** The SIFT query vectors (loaded but not used in this benchmark). */
    private List<VectorFloat<?>> queryVectors;

    /** Ground truth nearest neighbors (loaded but not used in this benchmark). */
    private List<List<Integer>> groundTruth;

    /**
     * The number of subspaces for Product Quantization.
     * <p>
     * Higher values of M provide more accurate quantization but increase training time.
     * Values: 16, 32, 64
     */
    @Param({"16", "32", "64"})
    private int M;

    /** The dimensionality of the SIFT vectors (128). */
    int originalDimension;

    /**
     * Sets up the benchmark by loading the SIFT Small dataset.
     * <p>
     * This method loads the SIFT base vectors, query vectors, and ground truth from the
     * local filesystem. The base vectors are wrapped in a RandomAccessVectorValues instance
     * for use during PQ training.
     *
     * @throws IOException if there is an error loading the SIFT dataset files
     */
    @Setup
    public void setup() throws IOException {
        var siftPath = "siftsmall";
        baseVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_base.fvecs", siftPath));
        queryVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_query.fvecs", siftPath));
        groundTruth = SiftLoader.readIvecs(String.format("%s/siftsmall_groundtruth.ivecs", siftPath));
        log.info("base vectors size: {}, query vectors size: {}, loaded, dimensions {}",
                baseVectors.size(), queryVectors.size(), baseVectors.get(0).length());
        originalDimension = baseVectors.get(0).length();
        // wrap the raw vectors in a RandomAccessVectorValues
        ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);
    }

    /**
     * Tears down the benchmark by clearing the loaded vectors.
     * <p>
     * This method releases memory by clearing all loaded vectors and ground truth data.
     *
     * @throws IOException if there is an error during teardown
     */
    @TearDown
    public void tearDown() throws IOException {
        baseVectors.clear();
        queryVectors.clear();
        groundTruth.clear();
    }

    /**
     * Benchmarks the computation of Product Quantization codebooks on SIFT vectors.
     * <p>
     * This benchmark measures the time required to train a Product Quantization model
     * on the SIFT Small dataset. The training process involves:
     * <ul>
     *   <li>Splitting each 128-dimensional SIFT vector into M subspaces</li>
     *   <li>Running k-means clustering (256 centroids) in each subspace</li>
     *   <li>Centering the dataset to improve quantization accuracy</li>
     * </ul>
     * The resulting PQ model provides a compression ratio of 128 * 4 / M bytes per vector.
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
                true); // center the dataset

        blackhole.consume(pq);
    }
}
