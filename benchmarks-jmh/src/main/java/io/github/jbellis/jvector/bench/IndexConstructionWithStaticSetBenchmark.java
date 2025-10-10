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

import io.github.jbellis.jvector.example.SiftSmall;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * JMH benchmark for measuring graph index construction performance using the SIFT dataset.
 * This benchmark evaluates index construction time with a fixed, real-world dataset,
 * testing various combinations of graph degree (M) and beam width parameters.
 *
 * <p>Unlike {@link IndexConstructionWithRandomSetBenchmark}, this benchmark uses the
 * actual SIFT dataset loaded from disk, providing more realistic performance measurements
 * that account for real data characteristics.</p>
 *
 * <p>Key parameters:</p>
 * <ul>
 *   <li>Graph degree (M): 16, 32, or 64 neighbors per node</li>
 *   <li>Beam width: 10 or 100 for construction search</li>
 *   <li>Dataset: SIFT small dataset (10,000 vectors, 128 dimensions)</li>
 *   <li>Similarity function: Euclidean distance</li>
 * </ul>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(1)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Threads(1)
public class IndexConstructionWithStaticSetBenchmark {
    private static final Logger log = LoggerFactory.getLogger(IndexConstructionWithStaticSetBenchmark.class);

    /** The vector values to be indexed, loaded from the SIFT dataset. */
    private RandomAccessVectorValues ravv;

    /** The base vectors from the SIFT dataset. */
    private List<VectorFloat<?>> baseVectors;

    /** The query vectors from the SIFT dataset (loaded but not used in this benchmark). */
    private List<VectorFloat<?>> queryVectors;

    /** The ground truth nearest neighbors (loaded but not used in this benchmark). */
    private List<List<Integer>> groundTruth;

    /** The score provider used during graph construction. */
    private BuildScoreProvider bsp;

    /** The maximum degree of the graph (number of neighbors per node). */
    @Param({"16", "32", "64"})
    private int M;

    /** The beam width used during graph construction searches. */
    @Param({"10", "100"})
    private int beamWidth;

    /** The dimensionality of vectors in the dataset. */
    int originalDimension;

    /**
     * Constructs a new benchmark instance. JMH will instantiate this class
     * and populate the @Param fields before calling setup methods.
     */
    public IndexConstructionWithStaticSetBenchmark() {
        // JMH-managed lifecycle
    }

    /**
     * Initializes the benchmark state by loading the SIFT dataset from disk
     * and configuring the score provider.
     *
     * @throws IOException if an error occurs loading the dataset files
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

        // score provider using the raw, in-memory vectors
        bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
    }

    /**
     * Cleans up resources after the benchmark completes by clearing all vector collections.
     *
     * @throws IOException if an error occurs during teardown
     */
    @TearDown
    public void tearDown() throws IOException {
        baseVectors.clear();
        queryVectors.clear();
        groundTruth.clear();
    }

    /**
     * The main benchmark method that measures the time to build a graph index
     * from the loaded SIFT dataset using the configured parameters.
     *
     * @param blackhole JMH blackhole to prevent dead code elimination
     * @throws IOException if an error occurs during index construction
     */
    @Benchmark
    public void buildIndexBenchmark(Blackhole blackhole) throws IOException {
        // score provider using the raw, in-memory vectors
        try (final var graphIndexBuilder = new GraphIndexBuilder(bsp, ravv.dimension(), M, beamWidth, 1.2f, 1.2f, true)) {
            final var graphIndex = graphIndexBuilder.build(ravv);
            blackhole.consume(graphIndex);
        }
    }
}
