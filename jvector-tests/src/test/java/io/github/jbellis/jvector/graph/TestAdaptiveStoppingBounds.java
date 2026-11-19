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
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Adaptive stopping only ever extends a search: it expands at least the nodes the standard rule would, and, capped,
 * at most the nodes a standard search with twice the beam would, whatever the margin, the data and the similarity.
 */
public class TestAdaptiveStoppingBounds {
    private static final int N = 2000, DIM = 32, QUERIES = 30;

    @Test
    public void adaptiveStoppingStaysBetweenTheStandardRuleAndTwiceItsBeam() throws Exception {
        var random = new Random(31);
        for (boolean unitLength : new boolean[] {true, false}) {
            List<VectorFloat<?>> base = new ArrayList<>();
            float maxNorm = 0;
            for (int i = 0; i < N; i++) {
                var v = TestUtil.randomVector(random, DIM);
                VectorUtil.l2normalize(v);
                if (!unitLength) VectorUtil.scale(v, (float) Math.pow(10, -1 + 2 * random.nextDouble()));
                base.add(v);
                maxNorm = Math.max(maxNorm, (float) Math.sqrt(VectorUtil.dotProduct(v, v)));
            }
            var vectors = new ListRandomAccessVectorValues(base, DIM);
            for (var similarity : VectorSimilarityFunction.values()) {
                if (similarity == VectorSimilarityFunction.COSINE && !unitLength) continue;
                ImmutableGraphIndex graph;
                try (var builder = new GraphIndexBuilder(BuildScoreProvider.randomAccessScoreProvider(vectors, similarity),
                        DIM, 16, 64, 1.2f, 1.2f, false)) {
                    graph = builder.build(vectors);
                }
                try (var standard = new GraphSearcher(graph); var adaptive = new GraphSearcher(graph)) {
                    standard.useAdaptiveTermination(false);
                    for (int q = 0; q < QUERIES; q++) {
                        var query = TestUtil.randomVector(random, DIM);
                        float bound = (float) Math.sqrt(VectorUtil.dotProduct(query, query)) * maxNorm;
                        var sf = DefaultSearchScoreProvider.exact(query, similarity, vectors).scoreFunction();
                        var provider = new DefaultSearchScoreProvider(sf, null, similarity, bound);
                        for (int k : new int[] {5, 10}) {
                            int least = standard.search(provider, k, k, 0f, 0f, Bits.ALL).getExpandedCountBaseLayer();
                            int most = standard.search(provider, k, 2 * k, 0f, 0f, Bits.ALL).getExpandedCountBaseLayer();
                            // The default margin, and one wide enough to search the whole graph without the cap.
                            for (float gamma : new float[] {0.005f, 1f}) {
                                adaptive.useAdaptiveTermination(true, gamma);
                                int expanded = adaptive.search(provider, k, k, 0f, 0f, Bits.ALL).getExpandedCountBaseLayer();
                                String where = similarity + (unitLength ? " unit" : " mixed") + " query " + q + " k " + k
                                        + " gamma " + gamma + ": expanded " + expanded;
                                assertTrue(where + " < standard " + least, expanded >= least);
                                assertTrue(where + " > standard at twice the beam " + most, expanded <= most);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Every top-K search stops adaptively whatever "no threshold" value it passes (0, a negative value, or the
     * NEGATIVE_INFINITY that {@code search(provider, topK, acceptOrds)} passes), while a threshold search
     * (threshold > 0) keeps its own early termination.
     */
    @Test
    public void adaptiveStoppingAppliesToTopKSearchesButNotToThresholdSearches() throws Exception {
        var random = new Random(37);
        List<VectorFloat<?>> base = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            var v = TestUtil.randomVector(random, DIM);
            VectorUtil.l2normalize(v);
            base.add(v);
        }
        var vectors = new ListRandomAccessVectorValues(base, DIM);
        var similarity = VectorSimilarityFunction.DOT_PRODUCT;
        ImmutableGraphIndex graph;
        try (var builder = new GraphIndexBuilder(BuildScoreProvider.randomAccessScoreProvider(vectors, similarity),
                DIM, 16, 64, 1.2f, 1.2f, false)) {
            graph = builder.build(vectors);
        }
        try (var standard = new GraphSearcher(graph); var adaptive = new GraphSearcher(graph)) {
            standard.useAdaptiveTermination(false);
            // A margin wide enough that adaptive stopping expands visibly more than the standard rule.
            adaptive.useAdaptiveTermination(true, 1f);
            int k = 10;
            boolean adaptiveTookEffect = false;
            for (int q = 0; q < QUERIES; q++) {
                var query = TestUtil.randomVector(random, DIM);
                VectorUtil.l2normalize(query);
                var sf = DefaultSearchScoreProvider.exact(query, similarity, vectors).scoreFunction();
                var provider = new DefaultSearchScoreProvider(sf, null, similarity, 1f);

                int zero = adaptive.search(provider, k, k, 0f, 0f, Bits.ALL).getExpandedCountBaseLayer();
                int negInf = adaptive.search(provider, k, Bits.ALL).getExpandedCountBaseLayer();
                int negative = adaptive.search(provider, k, k, -0.5f, 0f, Bits.ALL).getExpandedCountBaseLayer();
                assertEquals("query " + q + ": NEGATIVE_INFINITY stopped differently from 0", zero, negInf);
                assertEquals("query " + q + ": a negative threshold stopped differently from 0", zero, negative);
                int least = standard.search(provider, k, k, 0f, 0f, Bits.ALL).getExpandedCountBaseLayer();
                adaptiveTookEffect |= negInf > least;

                // Unit vectors score (1 + dot) / 2, so 0.6 keeps roughly the best tenth of the graph.
                float threshold = 0.6f;
                var expected = standard.search(provider, N, N, threshold, 0f, Bits.ALL);
                var actual = adaptive.search(provider, N, N, threshold, 0f, Bits.ALL);
                assertEquals("query " + q + ": a threshold search stopped adaptively",
                        expected.getExpandedCountBaseLayer(), actual.getExpandedCountBaseLayer());
                assertEquals("query " + q + ": threshold results differ",
                        expected.getNodes().length, actual.getNodes().length);
            }
            assertTrue("adaptive stopping never took effect", adaptiveTookEffect);
        }
    }
}
