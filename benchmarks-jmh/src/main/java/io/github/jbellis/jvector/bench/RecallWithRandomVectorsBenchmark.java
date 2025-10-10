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

import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * Benchmark for measuring graph search recall with and without Product Quantization on random vectors.
 * <p>
 * This benchmark evaluates the quality and performance of approximate nearest neighbor (ANN) search
 * using a hierarchical navigable small world (HNSW) graph index. It measures recall by comparing
 * search results against exact nearest neighbors computed via brute force. The benchmark tests both
 * full-precision vectors and Product Quantized (PQ) vectors to understand the accuracy-speed trade-off.
 * <p>
 * Key metrics tracked via auxiliary counters:
 * <ul>
 *   <li>avgRecall: The fraction of true nearest neighbors found in the search results</li>
 *   <li>avgReRankedCount: Number of candidates re-ranked with exact distances</li>
 *   <li>avgVisitedCount: Number of graph nodes visited during search</li>
 *   <li>avgExpandedCount: Number of graph nodes expanded (neighbors examined)</li>
 *   <li>avgExpandedCountBaseLayer: Number of nodes expanded in the base layer</li>
 * </ul>
 * <p>
 * The benchmark builds a graph index once during setup and then performs searches with multiple
 * query vectors, measuring both search time and recall quality.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(1)
@Warmup(iterations = 0)
@Measurement(iterations = 1)
@Threads(1)
public class RecallWithRandomVectorsBenchmark {
    private static final Logger log = LoggerFactory.getLogger(RecallWithRandomVectorsBenchmark.class);

    /**
     * Creates a new benchmark instance.
     * <p>
     * This constructor is invoked by JMH and should not be called directly.
     */
    public RecallWithRandomVectorsBenchmark() {
    }
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

    /** Random access wrapper for the base vectors. */
    private RandomAccessVectorValues ravv;

    /** The base vectors in the searchable dataset. */
    private ArrayList<VectorFloat<?>> baseVectors;

    /** Query vectors used to test search recall. */
    private ArrayList<VectorFloat<?>> queryVectors;

    /** Builder used to construct the graph index. */
    private GraphIndexBuilder graphIndexBuilder;

    /** The constructed graph index used for ANN search. */
    private ImmutableGraphIndex graphIndex;

    /** Product-quantized versions of the base vectors, or null if numberOfPQSubspaces=0. */
    private PQVectors pqVectors;

    /** Ground truth nearest neighbors for each query, computed via brute force. */
    private ArrayList<int[]> groundTruth;

    /**
     * The dimensionality of the vectors.
     * <p>
     * Default value: 1536 (typical for modern embedding models).
     */
    @Param({"1536"})
    int originalDimension;

    /**
     * The number of base vectors in the dataset.
     * <p>
     * Default value: 100000
     */
    @Param({"100000"})
    int numBaseVectors;

    /**
     * The number of query vectors to test.
     * <p>
     * Default value: 10
     */
    @Param({"10"})
    int numQueryVectors;

    /**
     * The number of subspaces for Product Quantization.
     * <p>
     * When numberOfPQSubspaces=0, uses full precision vectors without quantization.
     * When numberOfPQSubspaces&gt;0, applies PQ compression with approximate scoring.
     * Values: 0 (no PQ), 16, 32, 64, 96, 192
     */
    @Param({"0", "16", "32", "64", "96", "192"})
    int numberOfPQSubspaces;

    /**
     * The number of nearest neighbors to retrieve (k).
     * <p>
     * Default value: 50
     */
    @Param({/*"10",*/ "50"})
    int k;

    /**
     * The over-query factor for PQ searches.
     * <p>
     * When using PQ, the search retrieves k * overQueryFactor candidates using approximate
     * distances, then re-ranks them with exact distances to select the final k results.
     * Only applies when numberOfPQSubspaces &gt; 0.
     * <p>
     * Default value: 5
     */
    @Param({"5"})
    int overQueryFactor;

    /**
     * Sets up the benchmark by creating random vectors, building the graph index, and computing ground truth.
     * <p>
     * This method performs the following steps:
     * <ol>
     *   <li>Generates random base vectors and query vectors</li>
     *   <li>Optionally computes Product Quantization if numberOfPQSubspaces &gt; 0</li>
     *   <li>Builds an HNSW graph index for ANN search</li>
     *   <li>Computes exact nearest neighbors via brute force for recall measurement</li>
     * </ol>
     * The graph is configured with:
     * <ul>
     *   <li>Degree: 16 (max edges per node)</li>
     *   <li>Construction depth: 100 (beam width during construction)</li>
     *   <li>Alpha: 1.2 (degree overflow allowance)</li>
     *   <li>Diversity alpha: 1.2 (neighbor diversity requirement)</li>
     *   <li>Hierarchy: enabled</li>
     * </ul>
     *
     * @throws IOException if there is an error during setup
     */
    @Setup
    public void setup() throws IOException {
        baseVectors = new ArrayList<>(numBaseVectors);
        queryVectors = new ArrayList<>(numQueryVectors);

        for (int i = 0; i < numBaseVectors; i++) {
            VectorFloat<?> vector = createRandomVector(originalDimension);
            baseVectors.add(vector);
        }

        for (int i = 0; i < numQueryVectors; i++) {
            VectorFloat<?> vector = createRandomVector(originalDimension);
            queryVectors.add(vector);
        }

        // wrap the raw vectors in a RandomAccessVectorValues
        ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);
        final BuildScoreProvider buildScoreProvider;
        if (numberOfPQSubspaces > 0) {
            ProductQuantization productQuantization = ProductQuantization.compute(ravv, numberOfPQSubspaces, 256, true);
            pqVectors = (PQVectors) productQuantization.encodeAll(ravv);
            buildScoreProvider = BuildScoreProvider.pqBuildScoreProvider(VectorSimilarityFunction.EUCLIDEAN, pqVectors);
        } else {
            // score provider using the raw, in-memory vectors
            buildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
            pqVectors = null;
        }

        graphIndexBuilder = new GraphIndexBuilder(buildScoreProvider,
                ravv.dimension(),
                16, // graph degree
                100, // construction search depth
                1.2f, // allow degree overflow during construction by this factor
                1.2f, // relax neighbor diversity requirement by this factor
                true); // add the hierarchy
        graphIndex = graphIndexBuilder.build(ravv);

        // Calculate ground truth for recall computation
        calculateGroundTruth();
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

    /**
     * Tears down the benchmark by releasing resources.
     * <p>
     * This method clears all vectors and closes the graph index builder to release
     * any associated resources.
     *
     * @throws IOException if there is an error during teardown
     */
    @TearDown
    public void tearDown() throws IOException {
        baseVectors.clear();
        queryVectors.clear();
        graphIndexBuilder.close();
    }

    /**
     * Auxiliary counters for tracking recall and search statistics across benchmark iterations.
     * <p>
     * These counters accumulate metrics from each benchmark iteration and compute running averages.
     * JMH reports these values as additional benchmark results alongside timing measurements.
     */
    @AuxCounters(AuxCounters.Type.EVENTS)
    @State(Scope.Thread)
    public static class RecallCounters {
        /**
         * Creates a new counter instance.
         * <p>
         * This constructor is invoked by JMH and should not be called directly.
         */
        public RecallCounters() {
        }

        /** The average recall across all iterations. */
        public double avgRecall = 0;

        /** The average number of candidates re-ranked per query across all iterations. */
        public double avgReRankedCount = 0;

        /** The average number of graph nodes visited per query across all iterations. */
        public double avgVisitedCount = 0;

        /** The average number of graph nodes expanded per query across all iterations. */
        public double avgExpandedCount = 0;

        /** The average number of base layer nodes expanded per query across all iterations. */
        public double avgExpandedCountBaseLayer = 0;

        /** The number of benchmark iterations completed. */
        private int iterations = 0;

        /** The cumulative recall across all iterations. */
        private double totalRecall = 0;

        /** The cumulative re-ranked count across all iterations. */
        private double totalReRankedCount = 0;

        /** The cumulative visited count across all iterations. */
        private double totalVisitedCount = 0;

        /** The cumulative expanded count across all iterations. */
        private double totalExpandedCount = 0;

        /** The cumulative base layer expanded count across all iterations. */
        private double totalExpandedCountBaseLayer = 0;

        /**
         * Adds results from a single benchmark iteration and updates running averages.
         * <p>
         * This method is called after each benchmark iteration to accumulate statistics
         * and compute new average values.
         *
         * @param avgIterationRecall the average recall for this iteration
         * @param avgIterationReRankedCount the average re-ranked count for this iteration
         * @param avgIterationVisitedCount the average visited count for this iteration
         * @param avgIterationExpandedCount the average expanded count for this iteration
         * @param avgIterationExpandedCountBaseLayer the average base layer expanded count for this iteration
         */
        public void addResults(double avgIterationRecall, double avgIterationReRankedCount, double avgIterationVisitedCount, double avgIterationExpandedCount, double avgIterationExpandedCountBaseLayer) {
            log.info("adding results avgIterationRecall: {}, avgIterationReRankedCount: {}, avgIterationVisitedCount: {}, avgIterationExpandedCount: {}, avgIterationExpandedCountBaseLayer: {}", avgIterationRecall, avgIterationReRankedCount, avgIterationVisitedCount, avgIterationExpandedCount, avgIterationExpandedCountBaseLayer);
            totalRecall += avgIterationRecall;
            totalReRankedCount += avgIterationReRankedCount;
            totalVisitedCount += avgIterationVisitedCount;
            totalExpandedCount += avgIterationExpandedCount;
            totalExpandedCountBaseLayer += avgIterationExpandedCountBaseLayer;
            iterations++;
            avgRecall = totalRecall / (double) iterations;
            avgReRankedCount = totalReRankedCount / (double)  iterations;
            avgVisitedCount = totalVisitedCount / (double)  iterations;
            avgExpandedCount = totalExpandedCount / (double)  iterations;
            avgExpandedCountBaseLayer = totalExpandedCountBaseLayer / (double)  iterations;
        }
    }


    /**
     * Benchmarks ANN search with recall measurement on random vectors.
     * <p>
     * This benchmark performs graph searches for all query vectors and measures:
     * <ul>
     *   <li>Search time (via JMH timing)</li>
     *   <li>Recall quality (fraction of true nearest neighbors found)</li>
     *   <li>Search statistics (nodes visited, expanded, re-ranked)</li>
     * </ul>
     * <p>
     * The search behavior depends on the numberOfPQSubspaces parameter:
     * <ul>
     *   <li>When numberOfPQSubspaces=0: Uses exact distance calculations throughout</li>
     *   <li>When numberOfPQSubspaces&gt;0: Uses PQ approximate distances for initial search,
     *       then re-ranks top candidates with exact distances</li>
     * </ul>
     * <p>
     * Recall is computed by comparing search results against ground truth exact nearest neighbors.
     *
     * @param blackhole JMH blackhole to prevent dead code elimination
     * @param counters auxiliary counters for accumulating recall and search statistics
     * @throws IOException if there is an error during search
     */
    @Benchmark
    public void testOnHeapRandomVectorsWithRecall(Blackhole blackhole, RecallCounters counters) throws IOException {
        double totalRecall = 0.0;
        int numQueries = queryVectors.size();
        int totalReRankedCount = 0;
        int totalVisitedCount = 0;
        int totalExpandedCount = 0;
        int totalExpandedCountBaseLayer = 0;

        for (int i = 0; i < numQueries; i++) {
            var queryVector = queryVectors.get(i);
            final SearchResult searchResult;
            try (GraphSearcher graphSearcher = new GraphSearcher(graphIndex)) {
                final SearchScoreProvider ssp;
                if (pqVectors != null) { // Quantized, use the precomputed score function
                    // SearchScoreProvider that does a first pass with the loaded-in-memory PQVectors,
                    // then reranks with the exact vectors that are stored on disk in the index
                    ScoreFunction.ApproximateScoreFunction asf = pqVectors.precomputedScoreFunctionFor(
                            queryVector,
                            VectorSimilarityFunction.EUCLIDEAN
                    );
                    ScoreFunction.ExactScoreFunction reranker = ravv.rerankerFor(queryVector, VectorSimilarityFunction.EUCLIDEAN);
                    ssp = new DefaultSearchScoreProvider(asf, reranker);
                    searchResult = graphSearcher.search(ssp, k, overQueryFactor * k, 0.0f, 0.0f, Bits.ALL);
                } else { // Not quantized, used typical searcher
                    ssp = DefaultSearchScoreProvider.exact(queryVector, VectorSimilarityFunction.EUCLIDEAN, ravv);
                    searchResult = graphSearcher.search(ssp, k, Bits.ALL);
                }
            }

            // Extract result node IDs
            Set<Integer> resultIds = new HashSet<>(searchResult.getNodes().length);
            for (int j = 0; j < searchResult.getNodes().length; j++) {
                resultIds.add(searchResult.getNodes()[j].node);
            }

            // Calculate recall for this query
            double recall = calculateRecall(resultIds, groundTruth.get(i), k);
            totalRecall += recall;
            totalReRankedCount += searchResult.getRerankedCount();
            totalVisitedCount += searchResult.getVisitedCount();
            totalExpandedCount += searchResult.getExpandedCount();
            totalExpandedCountBaseLayer += searchResult.getExpandedCountBaseLayer();
            blackhole.consume(searchResult);
        }

        double avgRecall = totalRecall / (double) numQueries;
        double avgReRankedCount = totalReRankedCount / (double) numQueries;
        double avgVisitedCount = totalVisitedCount / (double) numQueries;
        double avgExpandedCount = totalExpandedCount / (double) numQueries;
        double avgExpandedCountBaseLayer = totalExpandedCountBaseLayer / (double) numQueries;

        // Store metrics in aux counters - these will appear in JMH output
        counters.addResults(avgRecall, avgReRankedCount, avgVisitedCount, avgExpandedCount, avgExpandedCountBaseLayer);

        blackhole.consume(avgRecall);
    }


    /**
     * Calculates exact nearest neighbors for all query vectors via brute force.
     * <p>
     * This method computes ground truth by performing exhaustive distance calculations
     * between each query vector and all base vectors. The top-k nearest neighbors for
     * each query are stored for later recall computation. This is computationally expensive
     * but provides the true nearest neighbors needed to measure search quality.
     */
    private void calculateGroundTruth() {
        groundTruth = new ArrayList<>(queryVectors.size());

        for (VectorFloat<?> queryVector : queryVectors) {
            // Calculate exact nearest neighbors for ground truth
            var exactResults = new ArrayList<SearchResult.NodeScore>();

            for (int i = 0; i < baseVectors.size(); i++) {
                float similarityScore = VectorSimilarityFunction.EUCLIDEAN.compare(queryVector, baseVectors.get(i));
                exactResults.add(new SearchResult.NodeScore(i, similarityScore));
            }

            // Sort by score (descending)
            exactResults.sort((a, b) -> Float.compare(b.score, a.score));

            // Store top-k ground truth
            int[] trueNearest = new int[Math.min(k, exactResults.size())];
            for (int i = 0; i < trueNearest.length; i++) {
                trueNearest[i] = exactResults.get(i).node;
            }
            groundTruth.add(trueNearest);
        }
    }

    /**
     * Calculates recall by comparing predicted results against ground truth.
     * <p>
     * Recall is the fraction of true nearest neighbors that appear in the search results.
     * This method compares the node IDs from the search results against the ground truth
     * nearest neighbors and counts how many matches are found.
     *
     * @param predicted the set of node IDs returned by the search
     * @param groundTruth the array of true nearest neighbor node IDs
     * @param k the number of neighbors to consider
     * @return the recall value between 0.0 and 1.0
     */
    private double calculateRecall(Set<Integer> predicted, int[] groundTruth, int k) {
        int hits = 0;
        int actualK = Math.min(k, Math.min(predicted.size(), groundTruth.length));

        for (int i = 0; i < actualK; i++) {
            for (int j = 0; j < actualK; j++) {
                if (predicted.contains(groundTruth[j])) {
                    hits++;
                    break;
                }
            }
        }

        return (double) hits / actualK;
    }
}
