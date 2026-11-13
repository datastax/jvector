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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.MockVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** ASH searches stop adaptively whatever provider wraps their scores; only the searcher's switch turns it off. */
public class TestASHAdaptiveStopping {
    @Test
    public void ashScoresAlwaysStopAdaptivelyUnlessTheSearcherSaysOtherwise() throws Exception {
        var random = new Random(17);
        int count = 3000, dimension = 64;
        var input = new VectorFloat<?>[count];
        for (int i = 0; i < count; i++) {
            input[i] = TestUtil.randomVector(random, dimension);
            VectorUtil.l2normalize(input[i]);
        }
        var values = MockVectorValues.fromValues(input);
        var ash = AsymmetricHashing.initialize(values, AsymmetricHashing.RANDOM,
                AsymmetricHashing.HEADER_BITS + 32 * 2, 1, 2);
        var encoded = ash.encodeAll(values, ForkJoinPool.commonPool());
        var bsp = BuildScoreProvider.ashBuildScoreProvider(VectorSimilarityFunction.DOT_PRODUCT, encoded);
        try (var builder = new GraphIndexBuilder(bsp, dimension, 16, 50, 1.2f, 1.2f, true, true)) {
            for (int i = 0; i < count; i++) builder.addGraphNode(i, input[i]);
            builder.cleanup();
            ImmutableGraphIndex graph = builder.getGraph();
            for (int q = 0; q < 20; q++) {
                var query = TestUtil.randomVector(random, dimension);
                VectorUtil.l2normalize(query);
                var asf = encoded.precomputedScoreFunctionFor(query, VectorSimilarityFunction.DOT_PRODUCT);
                assertTrue(asf.requiresAdaptiveTermination());
                var requested = search(graph, new DefaultSearchScoreProvider(asf, null, VectorSimilarityFunction.DOT_PRODUCT), true);
                // Plain wrappers that do not ask for adaptive stopping still get it.
                assertSame(requested, search(graph, new DefaultSearchScoreProvider(asf), true));
                assertSame(requested, search(graph, new DefaultSearchScoreProvider(asf, null), true));
                assertSame(requested, search(graph, customProvider(asf), true));
                // The searcher's switch is the override: the search is then the standard one.
                ScoreFunction.ApproximateScoreFunction unmarked = asf::similarityTo;
                var standard = search(graph, new DefaultSearchScoreProvider(unmarked), true);
                assertSame(standard, search(graph, new DefaultSearchScoreProvider(asf, null, VectorSimilarityFunction.DOT_PRODUCT), false));
            }
        }
    }

    private static SearchResult search(ImmutableGraphIndex graph, SearchScoreProvider provider, boolean adaptive)
            throws Exception {
        try (var searcher = new GraphSearcher(graph)) {
            searcher.useAdaptiveTermination(adaptive);
            return searcher.search(provider, 10, 10, 0.0f, 0.0f, Bits.ALL);
        }
    }

    /** A provider written by an application, which knows nothing about adaptive stopping. */
    private static SearchScoreProvider customProvider(ScoreFunction scoreFunction) {
        return new SearchScoreProvider() {
            @Override public ScoreFunction scoreFunction() { return scoreFunction; }
            @Override public ScoreFunction.ExactScoreFunction reranker() { return null; }
            @Override public ScoreFunction.ExactScoreFunction exactScoreFunction() { return null; }
        };
    }

    private static void assertSame(SearchResult expected, SearchResult actual) {
        assertArrayEquals(Arrays.stream(expected.getNodes()).mapToInt(n -> n.node).toArray(),
                Arrays.stream(actual.getNodes()).mapToInt(n -> n.node).toArray());
        assertEquals(expected.getVisitedCount(), actual.getVisitedCount());
        assertEquals(expected.getExpandedCount(), actual.getExpandedCount());
    }
}
