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
 * Benchmark for measuring graph search performance on the SIFT dataset.
 * <p>
 * This benchmark evaluates the performance of approximate nearest neighbor (ANN) search
 * using a hierarchical navigable small world (HNSW) graph index on the SIFT Small dataset.
 * Unlike the random vector benchmarks, this uses real-world image feature vectors with
 * realistic distributions and correlations.
 * <p>
 * The benchmark builds a graph index once during setup using the SIFT base vectors,
 * then measures search time using random query vectors. This focuses purely on search
 * performance without recall measurement.
 * <p>
 * Key characteristics:
 * <ul>
 *   <li>Uses SIFT Small dataset (10,000 base vectors, 128-dimensional)</li>
 *   <li>Full precision vectors (no quantization)</li>
 *   <li>Random query vectors generated at search time</li>
 *   <li>Measures pure search throughput</li>
 * </ul>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(1)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Threads(1)
public class StaticSetVectorsBenchmark {
    private static final Logger log = LoggerFactory.getLogger(StaticSetVectorsBenchmark.class);

    /**
     * Creates a new benchmark instance.
     * <p>
     * This constructor is invoked by JMH and should not be called directly.
     */
    public StaticSetVectorsBenchmark() {
    }

    /** Random access wrapper for the SIFT base vectors. */
    private RandomAccessVectorValues ravv;

    /** The SIFT base vectors in the searchable dataset. */
    private List<VectorFloat<?>> baseVectors;

    /** The SIFT query vectors (loaded but not used in this benchmark). */
    private List<VectorFloat<?>> queryVectors;

    /** Ground truth nearest neighbors (loaded but not used in this benchmark). */
    private List<List<Integer>> groundTruth;

    /** Builder used to construct the graph index. */
    private GraphIndexBuilder graphIndexBuilder;

    /** The constructed graph index used for ANN search. */
    private ImmutableGraphIndex graphIndex;

    /** The dimensionality of the SIFT vectors (128). */
    int originalDimension;

    /**
     * Sets up the benchmark by loading the SIFT dataset and building the graph index.
     * <p>
     * This method performs the following steps:
     * <ol>
     *   <li>Loads SIFT base vectors, query vectors, and ground truth from the filesystem</li>
     *   <li>Wraps the base vectors in a RandomAccessVectorValues instance</li>
     *   <li>Creates a BuildScoreProvider for exact distance calculations</li>
     *   <li>Builds an HNSW graph index with the following configuration:
     *     <ul>
     *       <li>Degree: 16 (max edges per node)</li>
     *       <li>Construction depth: 100 (beam width during construction)</li>
     *       <li>Alpha: 1.2 (degree overflow allowance)</li>
     *       <li>Diversity alpha: 1.2 (neighbor diversity requirement)</li>
     *       <li>Hierarchy: enabled</li>
     *     </ul>
     *   </li>
     * </ol>
     *
     * @throws IOException if there is an error loading the SIFT dataset or building the index
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
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);

        graphIndexBuilder = new GraphIndexBuilder(bsp,
                ravv.dimension(),
                16, // graph degree
                100, // construction search depth
                1.2f, // allow degree overflow during construction by this factor
                1.2f, // relax neighbor diversity requirement by this factor
                true); // add the hierarchy
        graphIndex = graphIndexBuilder.build(ravv);
    }

    /**
     * Tears down the benchmark by releasing resources.
     * <p>
     * This method clears all loaded vectors and closes the graph index builder to release
     * any associated resources.
     *
     * @throws IOException if there is an error during teardown
     */
    @TearDown
    public void tearDown() throws IOException {
        baseVectors.clear();
        queryVectors.clear();
        groundTruth.clear();
        graphIndexBuilder.close();
    }

    /**
     * Benchmarks graph search performance using random query vectors.
     * <p>
     * This benchmark measures the time to perform a single ANN search using a randomly
     * generated query vector. The search uses exact distance calculations (no quantization)
     * and returns the 10 nearest neighbors from the SIFT base vectors.
     * <p>
     * Each benchmark iteration generates a new random query vector with the same dimensionality
     * as the SIFT vectors (128), ensuring that the search operates on fresh data each time.
     *
     * @param blackhole JMH blackhole to prevent dead code elimination
     * @throws IOException if there is an error during search
     */
    @Benchmark
    public void testOnHeapWithRandomQueryVectors(Blackhole blackhole) throws IOException {
        var queryVector = SiftSmall.randomVector(originalDimension);
        // Your benchmark code here
        var searchResult = GraphSearcher.search(queryVector,
                10, // number of results
                ravv, // vectors we're searching, used for scoring
                VectorSimilarityFunction.EUCLIDEAN, // how to score
                graphIndex,
                Bits.ALL); // valid ordinals to consider
        blackhole.consume(searchResult);
    }
}
