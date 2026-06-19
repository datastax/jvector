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

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestPruningCompatibility extends LuceneTestCase {
    private static final int N_VECTORS = 10_000;
    private static final int N_QUERIES = 10;
    private static final int DIMENSIONS = 16;
    private static final int TOP_K = 10;
    private static final int RERANK_K = 100;
    private static final int THRESHOLD_TOP_K_CAP = 1_000;
    private static final int MAX_DEGREE = 32;
    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.COSINE;

    @Test
    public void testScoreTrackerFactoryPolicy() {
        var factory = new ScoreTracker.ScoreTrackerFactory();

        assertTrue(factory.getScoreTracker(false, RERANK_K, 0.0f) instanceof ScoreTracker.NoOpTracker);
        assertTrue(factory.getScoreTracker(true, RERANK_K, 0.0f) instanceof ScoreTracker.NoOpTracker);

        // Preserve legacy threshold behavior: threshold searches still use TwoPhaseTracker.
        assertTrue(factory.getScoreTracker(false, RERANK_K, 0.5f) instanceof ScoreTracker.TwoPhaseTracker);
        assertTrue(factory.getScoreTracker(true, RERANK_K, 0.5f) instanceof ScoreTracker.TwoPhaseTracker);
    }

    @Test
    @SuppressWarnings("deprecation")
    public void testUsePruningIgnoredForTopKAndFilteredTopK() throws IOException {
        for (boolean addHierarchy : List.of(false, true)) {
            Fixture fixture = buildFixture(addHierarchy);

            for (VectorFloat<?> query : fixture.queries) {
                assertSameWithPruningOffAndOn(fixture, query, Bits.ALL, 0.0f, TOP_K, RERANK_K);
                assertSameWithPruningOffAndOn(fixture, query, fixture.evenOrds, 0.0f, TOP_K, RERANK_K);
            }
        }
    }

    @Test
    @SuppressWarnings("deprecation")
    public void testUsePruningIgnoredForThresholdSearch() throws IOException {
        for (boolean addHierarchy : List.of(false, true)) {
            Fixture fixture = buildFixture(addHierarchy);

            for (VectorFloat<?> query : fixture.queries) {
                float threshold = exactThreshold(fixture.ravv, query, 100);

                assertSameWithPruningOffAndOn(
                        fixture,
                        query,
                        Bits.ALL,
                        threshold,
                        THRESHOLD_TOP_K_CAP,
                        THRESHOLD_TOP_K_CAP);
            }
        }
    }

    private void assertSameWithPruningOffAndOn(Fixture fixture,
                                               VectorFloat<?> query,
                                               Bits acceptOrds,
                                               float threshold,
                                               int topK,
                                               int rerankK) {
        SearchResult pruningOff = search(fixture, query, acceptOrds, threshold, topK, rerankK, false);
        SearchResult pruningOn = search(fixture, query, acceptOrds, threshold, topK, rerankK, true);

        assertEquals(pruningOff.getVisitedCount(), pruningOn.getVisitedCount());
        assertEquals(pruningOff.getNodes().length, pruningOn.getNodes().length);
        assertArrayEquals(sortedNodes(pruningOff), sortedNodes(pruningOn));

        if (threshold > 0.0f) {
            assertAllAtOrAboveThreshold(fixture, query, threshold, pruningOff);
            assertAllAtOrAboveThreshold(fixture, query, threshold, pruningOn);
        }
    }

    @SuppressWarnings("deprecation")
    private SearchResult search(Fixture fixture,
                                VectorFloat<?> query,
                                Bits acceptOrds,
                                float threshold,
                                int topK,
                                int rerankK,
                                boolean usePruning) {
        var searcher = new GraphSearcher(fixture.graph);
        searcher.usePruning(usePruning);

        var sf = fixture.ravv.rerankerFor(query, SIMILARITY);
        return searcher.search(
                new DefaultSearchScoreProvider(sf),
                topK,
                rerankK,
                threshold,
                0.0f,
                acceptOrds);
    }

    private Fixture buildFixture(boolean addHierarchy) throws IOException {
        var random = getRandom();

        VectorFloat<?>[] vectors = TestVectorGraph.createRandomFloatVectors(N_VECTORS, DIMENSIONS, random);
        var ravv = new ListRandomAccessVectorValues(List.of(vectors), DIMENSIONS);

        var builder = new GraphIndexBuilder(
                ravv,
                SIMILARITY,
                MAX_DEGREE,
                2 * MAX_DEGREE,
                1.2f,
                1.2f,
                addHierarchy);
        var graph = builder.build(ravv);

        FixedBitSet evenOrds = new FixedBitSet(N_VECTORS);
        for (int i = 0; i < N_VECTORS; i += 2) {
            evenOrds.set(i);
        }

        VectorFloat<?>[] queries = new VectorFloat<?>[N_QUERIES];
        for (int i = 0; i < N_QUERIES; i++) {
            queries[i] = TestUtil.randomVector(random, DIMENSIONS);
        }

        return new Fixture(ravv, graph, evenOrds, queries);
    }

    private float exactThreshold(RandomAccessVectorValues ravv,
                                 VectorFloat<?> query,
                                 int targetMatches) {
        float[] scores = new float[ravv.size()];
        for (int i = 0; i < ravv.size(); i++) {
            scores[i] = SIMILARITY.compare(query, ravv.getVector(i));
        }

        Arrays.sort(scores);
        return scores[scores.length - targetMatches];
    }

    private void assertAllAtOrAboveThreshold(Fixture fixture,
                                             VectorFloat<?> query,
                                             float threshold,
                                             SearchResult result) {
        for (var nodeScore : result.getNodes()) {
            float score = SIMILARITY.compare(query, fixture.ravv.getVector(nodeScore.node));
            assertTrue(
                    "returned node below threshold: node=" + nodeScore.node
                            + ", score=" + score
                            + ", threshold=" + threshold,
                    score + 1e-6f >= threshold);
        }
    }

    private static int[] sortedNodes(SearchResult result) {
        int[] nodes = Arrays.stream(result.getNodes())
                .mapToInt(nodeScore -> nodeScore.node)
                .toArray();
        Arrays.sort(nodes);
        return nodes;
    }

    private static class Fixture {
        final RandomAccessVectorValues ravv;
        final ImmutableGraphIndex graph;
        final FixedBitSet evenOrds;
        final VectorFloat<?>[] queries;

        Fixture(RandomAccessVectorValues ravv,
                ImmutableGraphIndex graph,
                FixedBitSet evenOrds,
                VectorFloat<?>[] queries) {
            this.ravv = ravv;
            this.graph = graph;
            this.evenOrds = evenOrds;
            this.queries = queries;
        }
    }
}
