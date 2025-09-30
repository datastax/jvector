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

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(1)
@Warmup(iterations = 0)
@Measurement(iterations = 1)
@Threads(1)
public class RecallWithRandomVectorsBenchmark {
    private static final Logger log = LoggerFactory.getLogger(RecallWithRandomVectorsBenchmark.class);
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
    private RandomAccessVectorValues ravv;
    private ArrayList<VectorFloat<?>> baseVectors;
    private ArrayList<VectorFloat<?>> queryVectors;
    private GraphIndexBuilder graphIndexBuilder;
    private ImmutableGraphIndex graphIndex;
    private PQVectors pqVectors;

    // Add ground truth storage
    private ArrayList<int[]> groundTruth;

    @Param({"1536"})
    int originalDimension;
    @Param({"100000"})
    int numBaseVectors;
    @Param({"10"})
    int numQueryVectors;
    @Param({"0", "16", "32", "64", "96", "192"})
    int numberOfPQSubspaces;
    @Param({/*"10",*/ "50"}) // Add different k values for recall calculation
    int k;
    @Param({"5"})
    int overQueryFactor;

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

    private VectorFloat<?> createRandomVector(int dimension) {
        VectorFloat<?> vector = VECTOR_TYPE_SUPPORT.createFloatVector(dimension);
        for (int i = 0; i < dimension; i++) {
            vector.set(i, (float) Math.random());
        }
        return vector;
    }

    @TearDown
    public void tearDown() throws IOException {
        baseVectors.clear();
        queryVectors.clear();
        graphIndexBuilder.close();
    }

    @AuxCounters(AuxCounters.Type.EVENTS)
    @State(Scope.Thread)
    public static class RecallCounters {
        public double avgRecall = 0;
        public double avgReRankedCount = 0;
        public double avgVisitedCount = 0;
        public double avgExpandedCount = 0;
        public double avgExpandedCountBaseLayer = 0;
        private int iterations = 0;
        private double totalRecall = 0;
        private double totalReRankedCount = 0;
        private double totalVisitedCount = 0;
        private double totalExpandedCount = 0;
        private double totalExpandedCountBaseLayer = 0;

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
