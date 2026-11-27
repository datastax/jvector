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
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * DOT_PRODUCT scores are {@code (1 + dot) / 2}, negative when the dot product is below -1, which happens with
 * non-unit vectors. Searches whose threshold or rerank floor defaults to 0.0f dropped those nodes.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestNegativeDotProductScores extends LuceneTestCase {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final VectorSimilarityFunction DOT = VectorSimilarityFunction.DOT_PRODUCT;

    /** Vectors (x, 0) for the given x values, and the query (1, 0), so that each score is (1 + x) / 2. */
    private static MockVectorValues alongXAxis(float... xs) {
        return MockVectorValues.fromValues(IntStream.range(0, xs.length)
                                                    .mapToObj(i -> vts.createFloatVector(new float[] {xs[i], 0f}))
                                                    .toArray(VectorFloat<?>[]::new));
    }

    private static VectorFloat<?> query() {
        return vts.createFloatVector(new float[] {1f, 0f});
    }

    @Test
    // resume(topK, rerankK) returns the nodes that score below 0
    public void testResumeReturnsNegativeScores() throws IOException {
        // scores from -4.5 to 5.0: nine of the twenty are negative
        var vectors = alongXAxis(-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        var ssp = DefaultSearchScoreProvider.exact(query(), DOT, vectors);
        try (var searcher = new GraphSearcher(new TestUtil.FullyConnectedGraphIndex(0, vectors.size()))) {
            Set<Integer> returned = new HashSet<>();
            var result = searcher.search(ssp, 1, 1, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, Bits.ALL);
            while (result.getNodes().length > 0) {
                for (var ns : result.getNodes()) {
                    assertTrue("returned twice: " + ns.node, returned.add(ns.node));
                }
                result = searcher.resume(1, 1);
            }
            assertEquals(vectors.size(), returned.size());
        }
    }

    @Test
    // with a reranker, search(scoreProvider, topK, threshold, acceptOrds) reranks and returns negative scores
    public void testDefaultRerankFloorKeepsNegativeScores() throws IOException {
        // every score is negative, from -0.5 to -4.5
        var vectors = alongXAxis(-2, -3, -4, -5, -6, -7, -8, -9, -10);
        var exact = DefaultSearchScoreProvider.exact(query(), DOT, vectors);
        var ssp = new DefaultSearchScoreProvider(exact.scoreFunction(), vectors.rerankerFor(query(), DOT));
        try (var searcher = new GraphSearcher(new TestUtil.FullyConnectedGraphIndex(0, vectors.size()))) {
            var result = searcher.search(ssp, 5, Float.NEGATIVE_INFINITY, Bits.ALL);
            assertEquals(5, result.getNodes().length);
            for (int i = 0; i < 5; i++) {
                assertEquals(i, result.getNodes()[i].node);
            }
        }
    }

    @Test
    // the static search(queryVector, topK, rerankK, ...) returns negative scores
    public void testStaticSearchReturnsNegativeScores() {
        var vectors = alongXAxis(-2, -3, -4, -5, -6, -7, -8, -9, -10);
        var graph = new TestUtil.FullyConnectedGraphIndex(0, vectors.size());
        var result = GraphSearcher.search(query(), 5, 5, vectors, DOT, graph, Bits.ALL);
        assertEquals(5, result.getNodes().length);
        for (int i = 0; i < 5; i++) {
            assertEquals(i, result.getNodes()[i].node);
        }
    }
}
