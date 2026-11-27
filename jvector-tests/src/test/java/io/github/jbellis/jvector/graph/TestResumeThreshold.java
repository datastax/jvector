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

/** resume() applies the threshold of the search it continues. */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestResumeThreshold extends LuceneTestCase {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    @Test
    public void testResumeReturnsOnlyNodesAboveThreshold() throws IOException {
        // vectors (x, 0) for x from -10 to 9 and the query (1, 0): DOT_PRODUCT scores are (1 + x) / 2
        var vectors = MockVectorValues.fromValues(IntStream.range(-10, 10)
                                                           .mapToObj(x -> vts.createFloatVector(new float[] {x, 0f}))
                                                           .toArray(VectorFloat<?>[]::new));
        var query = vts.createFloatVector(new float[] {1f, 0f});
        var ssp = DefaultSearchScoreProvider.exact(query, VectorSimilarityFunction.DOT_PRODUCT, vectors);
        for (float threshold : new float[] {-2.0f, 2.0f}) {
            try (var searcher = new GraphSearcher(new TestUtil.FullyConnectedGraphIndex(0, vectors.size()))) {
                Set<Integer> returned = new HashSet<>();
                var result = searcher.search(ssp, 1, 1, threshold, Float.NEGATIVE_INFINITY, Bits.ALL);
                while (result.getNodes().length > 0) {
                    for (var ns : result.getNodes()) {
                        assertTrue("score " + ns.score + " is below the threshold " + threshold, ns.score >= threshold);
                        assertTrue("returned twice: " + ns.node, returned.add(ns.node));
                    }
                    result = searcher.resume(1, 1);
                }
                // every node scoring at least the threshold is returned: x >= 2 * threshold - 1
                long expected = IntStream.range(-10, 10).filter(x -> (1 + x) / 2.0f >= threshold).count();
                assertEquals(expected, returned.size());
            }
        }
    }
}
