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
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import java.util.stream.IntStream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * An expansion that scores more neighbors than the searcher's initial neighbor buffer holds must hand every one of
 * them to the search, including on later searches that reuse the grown buffer.
 */
public class TestWideExpansions {
    @Test
    public void expansionsWiderThanTheBufferFindExactResults() throws IOException {
        var random = new Random(7);
        int dimension = 8, count = 300, topK = 10;
        var list = new ArrayList<VectorFloat<?>>();
        for (int i = 0; i < count; i++) list.add(TestUtil.randomVector(random, dimension));
        var vectors = new ListRandomAccessVectorValues(list, dimension);
        // every node is a neighbor of every other, so the first expansion scores count - 1 neighbors at once
        var graph = new TestUtil.FullyConnectedGraphIndex(0, count);
        for (var similarity : VectorSimilarityFunction.values()) {
            try (var searcher = new GraphSearcher(graph)) {
                for (int q = 0; q < 5; q++) {
                    var query = TestUtil.randomVector(random, dimension);
                    var result = searcher.search(DefaultSearchScoreProvider.exact(query, similarity, vectors),
                                                 topK, Bits.ALL);
                    String where = similarity + " query " + q;
                    assertEquals(where, count - 1, result.getVisitedCount());
                    int[] expected = IntStream.range(0, count).boxed()
                            .sorted(Comparator.comparingDouble((Integer i) -> -similarity.compare(query, vectors.getVector(i))))
                            .limit(topK).mapToInt(i -> i).toArray();
                    int[] actual = Arrays.stream(result.getNodes()).mapToInt(ns -> ns.node).toArray();
                    assertArrayEquals(where, expected, actual);
                }
            }
        }
    }
}
