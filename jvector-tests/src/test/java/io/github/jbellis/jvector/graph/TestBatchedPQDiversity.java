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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.diversity.VamanaDiversityProvider;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.Random;
import java.util.concurrent.ForkJoinPool;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Scoring the selected neighbours of a diversity check in batches must give the scores, and so the selected
 * neighbours, that scoring them one at a time gives.
 */
public class TestBatchedPQDiversity {
    private static final int NODES = 600;
    private static final int DIMENSION = 64;

    /** Delegates to a provider but hides batch scoring, so diversity checks run the one-at-a-time loop. */
    private static BuildScoreProvider oneAtATime(BuildScoreProvider inner) {
        return new BuildScoreProvider() {
            @Override public boolean isExact() { return inner.isExact(); }
            @Override public VectorFloat<?> approximateCentroid() { return inner.approximateCentroid(); }
            @Override public SearchScoreProvider searchProviderFor(VectorFloat<?> vector) { return inner.searchProviderFor(vector); }
            @Override public SearchScoreProvider searchProviderFor(int node1) { return inner.searchProviderFor(node1); }
            @Override public SearchScoreProvider diversityProviderFor(int node1) { return inner.diversityProviderFor(node1); }

            @Override
            public ScoreFunction diversityScoreFunctionFor(int node1) {
                ScoreFunction batched = inner.diversityScoreFunctionFor(node1);
                return (ScoreFunction.ApproximateScoreFunction) batched::similarityTo;
            }
        };
    }

    private static PQVectors encode(int subspaces) {
        var random = new Random(3);
        var vectors = new java.util.ArrayList<VectorFloat<?>>();
        for (int i = 0; i < NODES; i++) vectors.add(TestUtil.randomVector(random, DIMENSION));
        var ravv = new ListRandomAccessVectorValues(vectors, DIMENSION);
        var pq = ProductQuantization.compute(ravv, subspaces, 32, false);
        return (PQVectors) pq.encodeAll(ravv, ForkJoinPool.commonPool());
    }

    @Test
    public void batchScoresEqualSingleScores() {
        // 64 dimensions in 64, 16 and 4 subspaces: 1-, 4- and 16-float wide
        for (int subspaces : new int[] {64, 16, 4}) {
            var pqv = encode(subspaces);
            for (var vsf : new VectorSimilarityFunction[] {VectorSimilarityFunction.DOT_PRODUCT,
                                                           VectorSimilarityFunction.COSINE,
                                                           VectorSimilarityFunction.EUCLIDEAN}) {
                var random = new Random(5);
                for (int trial = 0; trial < 20; trial++) {
                    var sf = pqv.diversityFunctionFor(random.nextInt(NODES), vsf);
                    assertTrue(sf.supportsBatchScoring());
                    int count = 1 + random.nextInt(40);
                    int[] nodes = new int[count];
                    for (int i = 0; i < count; i++) nodes[i] = random.nextInt(NODES);
                    float[] scores = new float[count];
                    sf.similarityTo(nodes, count, scores);
                    for (int i = 0; i < count; i++) {
                        assertEquals(vsf + " " + subspaces + " subspaces, node " + nodes[i],
                                sf.similarityTo(nodes[i]), scores[i], 0f);
                    }
                }
            }
        }
    }

    @Test
    public void batchedDiversitySelectsTheSameNeighbours() {
        for (int subspaces : new int[] {64, 16, 4}) {
            var pqv = encode(subspaces);
            for (var vsf : new VectorSimilarityFunction[] {VectorSimilarityFunction.DOT_PRODUCT,
                                                           VectorSimilarityFunction.COSINE,
                                                           VectorSimilarityFunction.EUCLIDEAN}) {
                var batched = BuildScoreProvider.pqBuildScoreProvider(vsf, pqv);
                var batchedProvider = new VamanaDiversityProvider(batched, 1.2f);
                var singleProvider = new VamanaDiversityProvider(oneAtATime(batched), 1.2f);
                assertTrue(batched.diversityScoreFunctionFor(0).supportsBatchScoring());
                assertFalse(oneAtATime(batched).diversityScoreFunctionFor(0).supportsBatchScoring());

                var random = new Random(9);
                int selectedSomething = 0;
                for (int trial = 0; trial < 30; trial++) {
                    // candidates of a random node, best first, as a build hands them over
                    int base = random.nextInt(NODES);
                    var scoreFunction = batched.searchProviderFor(base).scoreFunction();
                    var scored = new java.util.LinkedHashMap<Integer, Float>();
                    for (int i = 0; i < 80; i++) {
                        int node = random.nextInt(NODES);
                        if (node != base) scored.put(node, scoreFunction.similarityTo(node));
                    }
                    var order = new java.util.ArrayList<>(scored.entrySet());
                    order.sort((x, y) -> Float.compare(y.getValue(), x.getValue()));
                    var neighbors = new NodeArray(order.size());
                    for (var e : order) neighbors.addInOrder(e.getKey(), e.getValue());

                    for (int maxDegree : new int[] {8, 32}) {
                        var expected = new FixedBitSet(neighbors.size());
                        var actual = new FixedBitSet(neighbors.size());
                        double expectedShort = singleProvider.retainDiverse(neighbors, maxDegree, 0, expected);
                        double actualShort = batchedProvider.retainDiverse(neighbors, maxDegree, 0, actual);
                        String where = vsf + " " + subspaces + " subspaces, degree " + maxDegree + ", trial " + trial;
                        assertEquals(where, expectedShort, actualShort, 0.0);
                        for (int i = 0; i < neighbors.size(); i++) {
                            assertEquals(where + ", neighbour " + i, expected.get(i), actual.get(i));
                        }
                        selectedSomething += actual.cardinality();
                    }
                }
                assertTrue("the test selected no neighbours at all", selectedSomething > 0);
            }
        }
    }
}
