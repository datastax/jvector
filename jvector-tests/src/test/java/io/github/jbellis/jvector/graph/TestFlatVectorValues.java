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
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
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
    public void reanchoredPairScoresMatchFreshOnes() {
        var random = new Random(5);
        int dimension = 33;
        var source = new ListRandomAccessVectorValues(vectors(random, 50, dimension), dimension);
        var flat = FlatVectorValues.copyOf(source, 1 << 10);
        for (var similarity : VectorSimilarityFunction.values()) {
            var bsp = BuildScoreProvider.randomAccessScoreProvider(flat, similarity);
            var reused = bsp.diversityScoreFunctionFor(0, null);
            for (int anchor = 0; anchor < source.size(); anchor++) {
                var next = bsp.diversityScoreFunctionFor(anchor, reused);
                assertTrue("re-aimed in place", next == reused);
                for (int node = 0; node < source.size(); node++) {
                    float expected = similarity.compare(source.getVector(anchor), source.getVector(node));
                    assertEquals(similarity + " anchor " + anchor, expected, next.similarityTo(node), TOLERANCE);
                }
            }
            // A score function for another similarity is not re-aimed.
            var other = VectorSimilarityFunction.values()[(similarity.ordinal() + 1) % VectorSimilarityFunction.values().length];
            var foreign = flat.scoreFunctionFor(0, other);
            assertTrue(bsp.diversityScoreFunctionFor(1, foreign) != foreign);
        }
    }

    /** Cursors and VectorAccess lend the stored rows, in request order with duplicates, for every page layout. */
    @Test
    public void cursorsLendStoredRows() {
        var random = new Random(9);
        int dimension = 33;
        var source = new ListRandomAccessVectorValues(vectors(random, 300, dimension), dimension);
        for (var flat : new FlatVectorValues[] {FlatVectorValues.copyOf(source, 1 << 10), FlatVectorValues.copyOf(source)}) {
            try (var cursor = flat.openRange(10, 250)) {
                int expected = 10;
                while (cursor.next()) {
                    assertTrue(cursor.isValueShared());
                    assertEquals(expected, cursor.ordinal());
                    assertArrayEquals(values(source.getVector(expected)), values(cursor.vector()), 0f);
                    expected++;
                }
                assertEquals(250, expected);
            }
            int[] ordinals = {5, 299, 5, 0, 120};
            try (var cursor = flat.openSelection(ordinals, 1, 3)) {
                for (int i = 1; i < 4; i++) {
                    assertTrue(cursor.next());
                    assertEquals(ordinals[i], cursor.ordinal());
                    assertArrayEquals(values(source.getVector(ordinals[i])), values(cursor.vector()), 0f);
                }
                assertTrue(!cursor.next());
            }
            assertThrows(IndexOutOfBoundsException.class, () -> flat.openRange(0, 301));
            assertThrows(IndexOutOfBoundsException.class, () -> flat.openSelection(new int[] {300}, 0, 1));
            var copies = VectorAccess.copySelected(flat, new int[] {7, 3, 7}, java.util.concurrent.ForkJoinPool.commonPool());
            assertArrayEquals(values(source.getVector(7)), values(copies.get(0)), 0f);
            assertArrayEquals(values(source.getVector(3)), values(copies.get(1)), 0f);
            assertTrue(copies.get(0) != copies.get(2)); // retained values are copies, not the cursor's vector
            var seen = new float[source.size()][];
            VectorAccess.forEach(flat, java.util.concurrent.ForkJoinPool.commonPool(),
                    (ordinal, vector) -> seen[ordinal] = values(vector));
            for (int node = 0; node < source.size(); node++) {
                assertArrayEquals(values(source.getVector(node)), seen[node], 0f);
            }
        }
    }

    @Test
    public void flatBuildMatchesListBuild() {
        var random = new Random(6);
        int dimension = 24;
        var source = new ListRandomAccessVectorValues(vectors(random, 1500, dimension), dimension);
        var flat = FlatVectorValues.copyOf(source);
        for (var similarity : VectorSimilarityFunction.values()) {
            for (boolean hierarchy : new boolean[] {false, true}) {
                // One thread, so both builds insert in the same order and see the same neighbors.
                var pool = new java.util.concurrent.ForkJoinPool(1);
                try (var expected = new GraphIndexBuilder(BuildScoreProvider.randomAccessScoreProvider(source, similarity),
                             dimension, 12, 40, 1.2f, 1.2f, hierarchy, true, pool, pool);
                     var actual = new GraphIndexBuilder(BuildScoreProvider.randomAccessScoreProvider(flat, similarity),
                             dimension, 12, 40, 1.2f, 1.2f, hierarchy, true, pool, pool)) {
                    var g1 = expected.build(source);
                    var g2 = actual.build(flat); // reads the flat rows through per-thread vectors
                    assertEquals(g1.size(), g2.size());
                    var v1 = g1.getView();
                    var v2 = g2.getView();
                    for (int node = 0; node < source.size(); node++) {
                        assertEquals(similarity + " hierarchy " + hierarchy + " node " + node,
                                neighbors(v1, node), neighbors(v2, node));
                    }
                } catch (java.io.IOException e) {
                    throw new java.io.UncheckedIOException(e);
                } finally {
                    pool.shutdown();
                }
            }
        }
    }

    private static List<Integer> neighbors(ImmutableGraphIndex.View view, int node) {
        var list = new ArrayList<Integer>();
        for (var it = view.getNeighborsIterator(0, node); it.hasNext(); ) list.add(it.nextInt());
        return list;
    }

    @Test
    public void batchScoresEqualSingleScoresAcrossPages() {
        var random = new Random(7);
        for (int dimension : new int[] {1, 7, 33, 384}) {
            var source = new ListRandomAccessVectorValues(vectors(random, 300, dimension), dimension);
            // One page, then pages small enough that a batch spans many of them.
            for (int pageFloats : new int[] {FlatVectorValues.DEFAULT_PAGE_FLOATS, 16 * 64}) {
                var flat = FlatVectorValues.copyOf(source, pageFloats);
                for (var similarity : VectorSimilarityFunction.values()) {
                    var queryScores = flat.scoreFunctionFor(TestUtil.randomVector(random, dimension), similarity);
                    var pairScores = flat.scoreFunctionFor(11, similarity);
                    assertTrue(queryScores.supportsBatchScoring() && pairScores.supportsBatchScoring());
                    for (int count : new int[] {0, 1, 5, 64, 300}) {
                        int[] nodes = new int[count + 2];
                        for (int i = 0; i < count; i++) nodes[i] = random.nextInt(source.size());
                        for (var sf : new ScoreFunction[] {queryScores, pairScores.anchorAt(random.nextInt(source.size()))}) {
                            float[] scores = new float[count + 2];
                            sf.similarityTo(nodes, count, scores);
                            for (int i = 0; i < count; i++) {
                                assertEquals(similarity + " dim " + dimension + " count " + count,
                                        sf.similarityTo(nodes[i]), scores[i], 0f);
                            }
                            assertEquals(0f, scores[count], 0f); // nothing written past count
                        }
                    }
                }
            }
        }
    }

    @Test
    public void rowScoringRejectsRowsOutsideTheBase() {
        var random = new Random(8);
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var base = vts.createFloatVector(4 * 16);
        var query = TestUtil.randomVector(random, 16);
        float[] out = new float[2];
        assertThrows(IndexOutOfBoundsException.class,
                () -> VectorUtil.dotProductRows(query, base, 16, new int[] {1, 4}, 2, 16, out));
        assertThrows(IndexOutOfBoundsException.class,
                () -> VectorUtil.squareL2DistanceRows(query, base, 16, new int[] {-1}, 1, 16, out));
        assertThrows(IndexOutOfBoundsException.class,
                () -> VectorUtil.cosineRows(query, base, 16, new int[] {3}, 1, 17, out));
    }

    @Test
    public void appendableRowsRoundTripAcrossGrowingPages() {
        var random = new Random(9);
        for (int dimension : new int[] {1, 7, 33}) {
            int stride = (dimension + 15) / 16 * 16;
            int count = 1000;
            var list = vectors(random, count, dimension);
            // First page of 2 rows, doubling to full pages of 32 rows; ordinals arrive shuffled.
            var flat = FlatVectorValues.appendable(dimension, 2 * stride, 32 * stride);
            var order = new ArrayList<Integer>();
            for (int i = 0; i < count; i++) order.add(i);
            java.util.Collections.shuffle(order, random);
            long allocated = 0;
            for (int ordinal : order) allocated += flat.add(ordinal, list.get(ordinal));
            assertEquals(count, flat.size());
            assertEquals(allocated, flat.ramBytesUsed());
            // Spare rows stay below both the rows stored and one full page.
            long spareFloats = flat.ramBytesUsed() / Float.BYTES - (long) count * stride;
            assertTrue(spareFloats >= 0 && spareFloats < 32L * stride && spareFloats <= (long) count * stride);
            var into = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(dimension);
            for (int node = 0; node < count; node++) {
                float[] expected = values(list.get(node));
                assertArrayEquals(expected, values(flat.getVector(node)), 0f);
                flat.getVectorInto(node, into, 0);
                assertArrayEquals(expected, values(into), 0f);
            }
            for (var similarity : VectorSimilarityFunction.values()) {
                var query = TestUtil.randomVector(random, dimension);
                var sf = flat.scoreFunctionFor(query, similarity);
                int[] nodes = new int[count];
                for (int i = 0; i < count; i++) nodes[i] = order.get(i);
                float[] scores = new float[count];
                sf.similarityTo(nodes, count, scores);
                for (int i = 0; i < count; i++) {
                    float expected = similarity.compare(query, list.get(nodes[i]));
                    assertEquals(similarity + " dim " + dimension, expected, sf.similarityTo(nodes[i]), TOLERANCE);
                    assertEquals(sf.similarityTo(nodes[i]), scores[i], 0f);
                }
            }
        }
    }

    @Test
    public void appendableStoreRejectsMisuse() {
        var random = new Random(10);
        var flat = FlatVectorValues.appendable(8);
        assertThrows(IllegalArgumentException.class, () -> flat.add(-1, TestUtil.randomVector(random, 8)));
        assertThrows(IllegalArgumentException.class, () -> flat.add(0, TestUtil.randomVector(random, 9)));
        var fixed = FlatVectorValues.copyOf(new ListRandomAccessVectorValues(vectors(random, 4, 8), 8));
        assertThrows(UnsupportedOperationException.class, () -> fixed.add(4, TestUtil.randomVector(random, 8)));
        assertEquals(0, FlatVectorValues.appendable(8).size());
    }

    /**
     * Writers append rows and insert them into a graph while readers search it, as Cassandra's memtable index does:
     * each writer takes the next ordinal, stores its row, then adds the node, which publishes the row. The finished
     * graph must find the exact best matches as often as the same live build over a list of separate vectors.
     */
    @Test
    public void liveInsertsAndSearchesOverAnAppendableStore() throws Exception {
        var random = new Random(11);
        int dimension = 32, count = 4000;
        var list = vectors(random, count, dimension);
        var queries = vectors(random, 100, dimension);
        for (var similarity : VectorSimilarityFunction.values()) {
            // Small pages, so inserts keep allocating pages while searches run.
            var flat = FlatVectorValues.appendable(dimension, 4 * 32, 256 * 32);
            int flatFound = liveBuildAndCount(flat, list, queries, similarity, (ordinal, v) -> flat.add(ordinal, v));
            assertEquals(count, flat.size());
            var separate = new ListRandomAccessVectorValues(list, dimension);
            int listFound = liveBuildAndCount(separate, list, queries, similarity, (ordinal, v) -> 0L);
            // Concurrent builds differ run to run; allow the spread of two such builds, not a systematic loss.
            assertTrue(similarity + ": flat store found " + flatFound + ", separate vectors " + listFound,
                    flatFound >= listFound - queries.size() / 20);
        }
    }

    private interface RowSink {
        long add(int ordinal, VectorFloat<?> vector);
    }

    /** Live-builds a graph over {@code vectors} with 4 writers and 2 searchers; returns queries whose best match is in the top 10. */
    private static int liveBuildAndCount(RandomAccessVectorValues vectors, List<VectorFloat<?>> list, List<VectorFloat<?>> queries,
                                         VectorSimilarityFunction similarity, RowSink sink) throws Exception {
        int count = list.size(), dimension = vectors.dimension(), writers = 4, readers = 2;
        var next = new java.util.concurrent.atomic.AtomicInteger();
        var published = new java.util.concurrent.atomic.AtomicInteger();
        var failure = new java.util.concurrent.atomic.AtomicReference<Throwable>();
        try (var builder = new GraphIndexBuilder(BuildScoreProvider.randomAccessScoreProvider(vectors, similarity),
                dimension, 16, 50, 1.2f, 1.2f, false)) {
            var threads = new ArrayList<Thread>();
            for (int w = 0; w < writers; w++) {
                threads.add(new Thread(() -> {
                    try {
                        for (int ordinal; (ordinal = next.getAndIncrement()) < count; ) {
                            sink.add(ordinal, list.get(ordinal));
                            builder.addGraphNode(ordinal, list.get(ordinal));
                            published.incrementAndGet();
                        }
                    } catch (Throwable t) {
                        failure.compareAndSet(null, t);
                    }
                }));
            }
            for (int r = 0; r < readers; r++) {
                int first = r;
                threads.add(new Thread(() -> {
                    try (var searcher = new GraphSearcher(builder.getGraph())) {
                        for (int q = first; published.get() < count && failure.get() == null; q = (q + 1) % queries.size()) {
                            var query = queries.get(q);
                            var result = searcher.search(DefaultSearchScoreProvider.exact(query, similarity, vectors), 5, Bits.ALL);
                            for (var ns : result.getNodes()) {
                                // Every node a search can reach has its row stored.
                                assertEquals(similarity.compare(query, list.get(ns.node)), ns.score, TOLERANCE);
                            }
                        }
                    } catch (Throwable t) {
                        failure.compareAndSet(null, t);
                    }
                }));
            }
            for (var t : threads) t.start();
            for (var t : threads) t.join();
            if (failure.get() != null) throw new AssertionError(similarity + " live build failed", failure.get());
            builder.cleanup();
            int found = 0;
            try (var searcher = new GraphSearcher(builder.getGraph())) {
                for (var query : queries) {
                    int best = 0;
                    for (int node = 1; node < count; node++) {
                        if (similarity.compare(query, list.get(node)) > similarity.compare(query, list.get(best))) best = node;
                    }
                    var result = searcher.search(DefaultSearchScoreProvider.exact(query, similarity, vectors), 10, Bits.ALL);
                    for (var ns : result.getNodes()) if (ns.node == best) found++;
                }
            }
            return found;
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
