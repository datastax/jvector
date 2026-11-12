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
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Deferring candidates that a layer search can no longer expand must not change any search: results, scores,
 * visited and expanded counts match a searcher that keeps every candidate, for standard and adaptive stopping,
 * filters, hierarchical graphs and repeated resume() calls.
 */
public class TestBoundedCandidates {
    @Test
    public void boundedSearchesMatchUnboundedOnes() throws IOException {
        var random = new Random(42);
        int dimension = 24, count = 3000;
        var list = new ArrayList<VectorFloat<?>>();
        for (int i = 0; i < count; i++) list.add(TestUtil.randomVector(random, dimension));
        var vectors = new ListRandomAccessVectorValues(list, dimension);
        for (var similarity : VectorSimilarityFunction.values()) {
            for (boolean hierarchy : new boolean[] {false, true}) {
                var builder = new GraphIndexBuilder(BuildScoreProvider.randomAccessScoreProvider(vectors, similarity),
                        dimension, 16, 60, 1.2f, 1.2f, hierarchy);
                var graph = builder.build(vectors);
                try (var bounded = new GraphSearcher(graph); var unbounded = new GraphSearcher(graph)) {
                    unbounded.boundCandidates(false);
                    for (int q = 0; q < 40; q++) {
                        var query = TestUtil.randomVector(random, dimension);
                        Bits accept = q % 3 == 0 ? Bits.ALL : randomFilter(random, count, q % 3 == 1 ? 0.7 : 0.1);
                        int topK = new int[] {1, 5, 10}[q % 3];
                        int rerankK = topK * new int[] {1, 2, 5}[(q / 3) % 3];
                        boolean adaptive = q % 2 == 0;
                        String where = similarity + " hierarchy " + hierarchy + " query " + q;
                        var b = bounded.search(provider(query, similarity, vectors, adaptive), topK, rerankK, 0f, 0f, accept);
                        var u = unbounded.search(provider(query, similarity, vectors, adaptive), topK, rerankK, 0f, 0f, accept);
                        assertSame(where, u, b);
                        for (int r = 0; r < 3; r++) {
                            assertSame(where + " resume " + r, unbounded.resume(topK, rerankK), bounded.resume(topK, rerankK));
                        }
                    }
                }
            }
        }
    }

    private static DefaultSearchScoreProvider provider(VectorFloat<?> query, VectorSimilarityFunction similarity,
                                                       RandomAccessVectorValues vectors, boolean adaptive) {
        var exact = DefaultSearchScoreProvider.exact(query, similarity, vectors);
        return new DefaultSearchScoreProvider(exact.scoreFunction(), null, adaptive);
    }

    private static Bits randomFilter(Random random, int count, double density) {
        var bits = new FixedBitSet(count);
        for (int i = 0; i < count; i++) if (random.nextDouble() < density) bits.set(i);
        return bits;
    }

    private static void assertSame(String where, SearchResult expected, SearchResult actual) {
        assertEquals(where + " visited", expected.getVisitedCount(), actual.getVisitedCount());
        assertEquals(where + " expanded", expected.getExpandedCount(), actual.getExpandedCount());
        assertEquals(where + " expanded base layer", expected.getExpandedCountBaseLayer(), actual.getExpandedCountBaseLayer());
        assertArrayEquals(where + " nodes", nodes(expected), nodes(actual));
        for (int i = 0; i < expected.getNodes().length; i++) {
            assertEquals(where + " score " + i, expected.getNodes()[i].score, actual.getNodes()[i].score, 0f);
        }
    }

    private static int[] nodes(SearchResult result) {
        int[] nodes = new int[result.getNodes().length];
        for (int i = 0; i < nodes.length; i++) nodes[i] = result.getNodes()[i].node;
        return nodes;
    }
}
