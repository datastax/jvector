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
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public class TestFlatVectorValues {
    private static final float TOLERANCE = 1e-6f;

    private static List<VectorFloat<?>> vectors(Random random, int count, int dimension) {
        var list = new ArrayList<VectorFloat<?>>(count);
        for (int i = 0; i < count; i++) list.add(TestUtil.randomVector(random, dimension));
        return list;
    }

    private static float[] values(VectorFloat<?> v) {
        float[] a = new float[v.length()];
        for (int i = 0; i < a.length; i++) a[i] = v.get(i);
        return a;
    }

    @Test
    public void rowsRoundTripAcrossPagesAndPadding() {
        var random = new Random(1);
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        for (int dimension : new int[] {1, 7, 16, 33, 384}) {
            var source = new ListRandomAccessVectorValues(vectors(random, 101, dimension), dimension);
            // Small pages force many pages, one row per page for the widest rows.
            for (int pageFloats : new int[] {64, 1 << 12, FlatVectorValues.DEFAULT_PAGE_FLOATS}) {
                var flat = FlatVectorValues.copyOf(source, pageFloats);
                assertEquals(source.size(), flat.size());
                assertEquals(dimension, flat.dimension());
                var into = vts.createFloatVector(dimension + 3);
                for (int node = 0; node < source.size(); node++) {
                    float[] expected = values(source.getVector(node));
                    assertArrayEquals(expected, values(flat.getVector(node)), 0f);
                    flat.getVectorInto(node, into, 3);
                    for (int i = 0; i < dimension; i++) assertEquals(expected[i], into.get(i + 3), 0f);
                }
            }
        }
    }

    @Test
    public void everySimilarityScoresRowsInPlace() {
        var random = new Random(2);
        for (int dimension : new int[] {1, 7, 33, 384}) {
            var source = new ListRandomAccessVectorValues(vectors(random, 64, dimension), dimension);
            var flat = FlatVectorValues.copyOf(source, 1 << 10);
            for (var similarity : VectorSimilarityFunction.values()) {
                var query = TestUtil.randomVector(random, dimension);
                var queryScores = flat.scoreFunctionFor(query, similarity);
                var routed = DefaultSearchScoreProvider.exact(query, similarity, flat).scoreFunction();
                var reranker = flat.rerankerFor(query, similarity);
                var pairScores = flat.scoreFunctionFor(5, similarity);
                for (int node = 0; node < source.size(); node++) {
                    float expected = similarity.compare(query, source.getVector(node));
                    assertEquals(similarity + " dim " + dimension, expected, queryScores.similarityTo(node), TOLERANCE);
                    assertEquals(queryScores.similarityTo(node), routed.similarityTo(node), 0f);
                    assertEquals(queryScores.similarityTo(node), reranker.similarityTo(node), 0f);
                    float expectedPair = similarity.compare(source.getVector(5), source.getVector(node));
                    assertEquals(similarity + " pair dim " + dimension, expectedPair, pairScores.similarityTo(node), TOLERANCE);
                }
            }
        }
    }

    @Test
    public void graphSearchFindsTheSameResults() {
        var random = new Random(3);
        int dimension = 48;
        var list = vectors(random, 2000, dimension);
        var source = new ListRandomAccessVectorValues(list, dimension);
        var flat = FlatVectorValues.copyOf(source);
        for (var similarity : VectorSimilarityFunction.values()) {
            var builder = new GraphIndexBuilder(BuildScoreProvider.randomAccessScoreProvider(source, similarity),
                    dimension, 16, 50, 1.2f, 1.2f, false);
            var graph = builder.build(source);
            try (var searcher = new GraphSearcher(graph)) {
                for (int q = 0; q < 20; q++) {
                    var query = TestUtil.randomVector(random, dimension);
                    var expected = searcher.search(DefaultSearchScoreProvider.exact(query, similarity, source), 10, 0f, Bits.ALL);
                    var actual = searcher.search(DefaultSearchScoreProvider.exact(query, similarity, flat), 10, 0f, Bits.ALL);
                    assertEquals(expected.getNodes().length, actual.getNodes().length);
                    for (int i = 0; i < expected.getNodes().length; i++) {
                        assertEquals(similarity + " query " + q + " rank " + i, expected.getNodes()[i].node, actual.getNodes()[i].node);
                        assertEquals(expected.getNodes()[i].score, actual.getNodes()[i].score, TOLERANCE);
                    }
                }
            } catch (java.io.IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }
    }

    @Test
    public void rejectsMismatchedQueries() {
        var random = new Random(4);
        var flat = FlatVectorValues.copyOf(new ListRandomAccessVectorValues(vectors(random, 4, 8), 8));
        assertThrows(IllegalArgumentException.class,
                () -> flat.scoreFunctionFor(TestUtil.randomVector(random, 9), VectorSimilarityFunction.DOT_PRODUCT));
        assertTrue(flat.copy() == flat);
    }
}
