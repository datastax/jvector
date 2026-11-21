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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.nio.file.Files;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

import static io.github.jbellis.jvector.vector.VectorSimilarityFunction.COSINE;
import static io.github.jbellis.jvector.vector.VectorSimilarityFunction.DOT_PRODUCT;
import static io.github.jbellis.jvector.vector.VectorSimilarityFunction.EUCLIDEAN;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * GraphSearcher with {@link AdaptiveTermination}. The central check compares it with a straightforward implementation
 * of Algorithm 1 of the paper; the others check properties that follow from the rule (gamma = 0 is beam search of
 * width k, a larger gamma only extends the search, an unbounded gamma is exhaustive) and its interaction with
 * filters, thresholds, reranking, resume, searcher reuse, hierarchy and on-disk graphs.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestAdaptiveSearch extends RandomizedTest {

    /** a graph with the vectors it was built from */
    private static final class Fixture {
        final List<VectorFloat<?>> vectors;
        final RandomAccessVectorValues ravv;
        final VectorSimilarityFunction vsf;
        final ImmutableGraphIndex graph;

        Fixture(int n, int dim, VectorSimilarityFunction vsf, boolean hierarchy, Random random) {
            this.vsf = vsf;
            vectors = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                // unit vectors for COSINE and DOT_PRODUCT, unnormalized Gaussian ones for EUCLIDEAN
                vectors.add(vsf == EUCLIDEAN ? TestUtil.normalRandomVector(random, dim) : TestUtil.randomVector(random, dim));
            }
            ravv = new ListRandomAccessVectorValues(vectors, dim);
            graph = new GraphIndexBuilder(ravv, vsf, 12, 40, 1.2f, 1.2f, hierarchy).build(ravv);
        }

        VectorFloat<?> query(Random random) {
            int dim = ravv.dimension();
            return vsf == EUCLIDEAN ? TestUtil.normalRandomVector(random, dim) : TestUtil.randomVector(random, dim);
        }

        SearchScoreProvider exact(VectorFloat<?> q) {
            return DefaultSearchScoreProvider.exact(q, vsf, ravv);
        }
    }

    private static Bits randomFilter(Random random, int n) {
        switch (random.nextInt(3)) {
            case 0:
                return Bits.ALL;
            default:
                double p = random.nextBoolean() ? 0.5 : 0.1;
                var allowed = new BitSet(n);
                for (int i = 0; i < n; i++) {
                    if (random.nextDouble() < p) allowed.set(i);
                }
                return allowed::get;
        }
    }

    // ---- the reference: Algorithm 1 of the paper with termination condition (3) ----

    private static final class Reference {
        final int[] nodes;
        final float[] scores;
        final int visited;
        final int expanded;

        Reference(int[] nodes, float[] scores, int visited, int expanded) {
            this.nodes = nodes;
            this.scores = scores;
            this.visited = visited;
            this.expanded = expanded;
        }
    }

    /** distance from score, written independently of AdaptiveTermination (unsquared, as in the paper) */
    private static double distance(VectorSimilarityFunction vsf, float score) {
        switch (vsf) {
            case EUCLIDEAN: return Math.sqrt(1.0 / score - 1.0);
            default: return 2 * Math.sqrt(Math.max(0, 1.0 - score)); // chord between unit vectors, from (1 + cos) / 2
        }
    }

    /**
     * Expands the nearest discovered node until k accepted nodes are closer than it by a factor of 1 + gamma; returns
     * the k nearest accepted expanded nodes. Ties are broken by node id, smaller first, as NodeQueue does.
     */
    private static Reference reference(ImmutableGraphIndex graph, ScoreFunction sf, VectorSimilarityFunction vsf,
                                       int k, float gamma, Bits accept) throws Exception {
        try (var view = graph.getView()) {
            Comparator<float[]> best = (a, b) -> a[1] != b[1] ? Float.compare(b[1], a[1]) : Float.compare(a[0], b[0]);
            var candidates = new PriorityQueue<float[]>(best); // {node, score}
            var results = new PriorityQueue<float[]>(best.reversed()); // worst on top, at most k
            var discovered = new HashSet<Integer>();
            int entry = view.entryNode().node;
            discovered.add(entry);
            candidates.add(new float[] {entry, sf.similarityTo(entry)});
            int visited = 0, expanded = 0;
            while (!candidates.isEmpty()) {
                var x = candidates.peek();
                if (results.size() >= k
                    && (k == 0 || (1 + (double) gamma) * distance(vsf, results.peek()[1]) <= distance(vsf, x[1]))) {
                    break;
                }
                candidates.poll();
                int node = (int) x[0];
                if (accept.get(node)) {
                    results.add(x);
                    if (results.size() > k) results.poll();
                }
                expanded++;
                for (var it = view.getNeighborsIterator(0, node); it.hasNext(); ) {
                    int neighbor = it.nextInt();
                    if (discovered.add(neighbor)) {
                        visited++;
                        candidates.add(new float[] {neighbor, sf.similarityTo(neighbor)});
                    }
                }
            }
            var sorted = new ArrayList<>(results);
            sorted.sort(best);
            return new Reference(sorted.stream().mapToInt(r -> (int) r[0]).toArray(), scores(sorted), visited, expanded);
        }
    }

    private static float[] scores(List<float[]> rows) {
        float[] s = new float[rows.size()];
        for (int i = 0; i < s.length; i++) s[i] = rows.get(i)[1];
        return s;
    }

    /** result nodes ordered by score, then node id (the search does not order ties) */
    private static int[] nodes(SearchResult r) {
        return Arrays.stream(r.getNodes())
                     .sorted(Comparator.<SearchResult.NodeScore>comparingDouble(ns -> -ns.score).thenComparingInt(ns -> ns.node))
                     .mapToInt(ns -> ns.node).toArray();
    }

    private static float[] sortedScores(SearchResult r) {
        float[] s = new float[r.getNodes().length];
        for (int i = 0; i < s.length; i++) s[i] = r.getNodes()[i].score;
        // results are returned best first
        return s;
    }

    @Test
    public void matchesReferenceImplementation() throws Exception {
        Random random = getRandom();
        for (var vsf : new VectorSimilarityFunction[] {COSINE, EUCLIDEAN, DOT_PRODUCT}) {
            var f = new Fixture(1500, 24, vsf, false, random);
            try (var searcher = new GraphSearcher(f.graph)) {
                for (int qi = 0; qi < 25; qi++) {
                    var q = f.query(random);
                    var ssp = f.exact(q);
                    for (float gamma : new float[] {0f, 0.05f, 0.2f, 0.5f, 1f, random.nextFloat() * 3}) {
                        int k = new int[] {1, 10, 37}[random.nextInt(3)];
                        var accept = randomFilter(random, f.vectors.size());
                        var expected = reference(f.graph, ssp.scoreFunction(), vsf, k, gamma, accept);
                        var actual = searcher.search(ssp, k, k, Float.NEGATIVE_INFINITY, 0f, accept,
                                                     AdaptiveTermination.of(vsf, gamma));
                        String where = vsf + " q=" + qi + " gamma=" + gamma + " k=" + k;
                        assertArrayEquals(where, expected.nodes, nodes(actual));
                        assertArrayEquals(where, expected.scores, sortedScores(actual), 0f);
                        assertEquals(where, expected.visited, actual.getVisitedCount());
                        assertEquals(where, expected.expanded, actual.getExpandedCountBaseLayer());
                    }
                }
            }
        }
    }

    /** gamma = 0 stops when k results are no farther than the best candidate: beam search of width k */
    @Test
    public void zeroGammaIsBeamSearchOfWidthK() throws Exception {
        Random random = getRandom();
        for (boolean hierarchy : new boolean[] {false, true}) {
            var f = new Fixture(2000, 16, COSINE, hierarchy, random);
            try (var adaptive = new GraphSearcher(f.graph); var beam = new GraphSearcher(f.graph)) {
                for (int qi = 0; qi < 40; qi++) {
                    var q = f.query(random);
                    int k = 1 + random.nextInt(40);
                    var accept = randomFilter(random, f.vectors.size());
                    var a = adaptive.search(f.exact(q), k, k, Float.NEGATIVE_INFINITY, 0f, accept, AdaptiveTermination.of(COSINE, 0f));
                    var b = beam.search(f.exact(q), k, k, Float.NEGATIVE_INFINITY, 0f, accept);
                    String where = "hierarchy=" + hierarchy + " q=" + qi + " k=" + k;
                    assertArrayEquals(where, nodes(b), nodes(a));
                    assertEquals(where, b.getVisitedCount(), a.getVisitedCount());
                    assertEquals(where, b.getExpandedCount(), a.getExpandedCount());
                }
            }
        }
    }

    /**
     * Search order does not depend on gamma, so a larger gamma expands what a smaller one did and then more: it
     * visits at least as many nodes, and each of its results is at least as good as the one of the same rank.
     */
    @Test
    public void largerGammaOnlyExtendsTheSearch() throws Exception {
        Random random = getRandom();
        float[] gammas = {0f, 0.02f, 0.1f, 0.25f, 0.5f, 1f, 2f};
        for (boolean hierarchy : new boolean[] {false, true}) {
            for (var vsf : new VectorSimilarityFunction[] {COSINE, EUCLIDEAN}) {
                var f = new Fixture(2000, 20, vsf, hierarchy, random);
                try (var searcher = new GraphSearcher(f.graph)) {
                    for (int qi = 0; qi < 20; qi++) {
                        var q = f.query(random);
                        int k = 1 + random.nextInt(25);
                        var accept = randomFilter(random, f.vectors.size());
                        SearchResult previous = null;
                        int growths = 0;
                        for (float gamma : gammas) {
                            var r = searcher.search(f.exact(q), k, k, Float.NEGATIVE_INFINITY, 0f, accept, AdaptiveTermination.of(vsf, gamma));
                            if (previous != null) {
                                String where = vsf + " hierarchy=" + hierarchy + " q=" + qi + " gamma=" + gamma;
                                assertTrue(where, r.getVisitedCount() >= previous.getVisitedCount());
                                assertTrue(where, r.getNodes().length >= previous.getNodes().length);
                                for (int i = 0; i < previous.getNodes().length; i++) {
                                    assertTrue(where + " rank " + i, r.getNodes()[i].score >= previous.getNodes()[i].score);
                                }
                                if (r.getVisitedCount() > previous.getVisitedCount()) growths++;
                            }
                            previous = r;
                        }
                        // gamma actually matters
                        assertTrue(vsf + " q=" + qi, growths > 0);
                    }
                }
            }
        }
    }

    /** With an unbounded gamma the search never stops early: it finds the true top k of everything reachable. */
    @Test
    public void unboundedGammaIsExhaustive() throws Exception {
        Random random = getRandom();
        for (boolean hierarchy : new boolean[] {false, true}) {
            var f = new Fixture(800, 12, EUCLIDEAN, hierarchy, random);
            BitSet reachable = reachableFromEntry(f.graph);
            // with a hierarchy, the base layer also starts from nodes found in the upper layers; the expected set below
            // is only right if those are all reachable from the entry node
            assumeTrue("base layer not connected", reachable.equals(reachableFromAllUpperLevelNodes(f.graph)));
            try (var searcher = new GraphSearcher(f.graph)) {
                for (int qi = 0; qi < 10; qi++) {
                    var q = f.query(random);
                    int k = 1 + random.nextInt(30);
                    var accept = randomFilter(random, f.vectors.size());
                    var r = searcher.search(f.exact(q), k, k, Float.NEGATIVE_INFINITY, 0f, accept,
                                            AdaptiveTermination.of(EUCLIDEAN, 1e30f));
                    assertArrayEquals(bruteForce(f, q, reachable, accept, 0, k), nodes(r));
                    if (!hierarchy) {
                        assertEquals(reachable.cardinality() - 1, r.getVisitedCount());
                    }
                }
            }
        }
    }

    /** resume() keeps the rule: with an unbounded gamma it returns the next k of everything reachable */
    @Test
    public void resumeUsesTheSameRule() throws Exception {
        Random random = getRandom();
        var f = new Fixture(800, 12, COSINE, false, random);
        BitSet reachable = reachableFromEntry(f.graph);
        try (var searcher = new GraphSearcher(f.graph)) {
            for (int qi = 0; qi < 10; qi++) {
                var q = f.query(random);
                int k = 1 + random.nextInt(20);
                var accept = randomFilter(random, f.vectors.size());
                var first = searcher.search(f.exact(q), k, k, Float.NEGATIVE_INFINITY, 0f, accept,
                                            AdaptiveTermination.of(COSINE, 1e30f));
                var second = searcher.resume(k, k);
                assertArrayEquals(bruteForce(f, q, reachable, accept, 0, k), nodes(first));
                assertArrayEquals(bruteForce(f, q, reachable, accept, k, k), nodes(second));
            }
        }
        // and with an ordinary gamma, resumed results are new, accepted and correctly scored
        try (var searcher = new GraphSearcher(f.graph)) {
            for (int qi = 0; qi < 20; qi++) {
                var q = f.query(random);
                var accept = randomFilter(random, f.vectors.size());
                var first = searcher.search(f.exact(q), 10, 10, Float.NEGATIVE_INFINITY, 0f, accept, AdaptiveTermination.of(COSINE, 0.3f));
                var second = searcher.resume(10, 10);
                Set<Integer> seen = new HashSet<>();
                for (var ns : first.getNodes()) seen.add(ns.node);
                for (var ns : second.getNodes()) {
                    assertTrue(seen.add(ns.node));
                    assertTrue(accept.get(ns.node));
                    assertEquals(COSINE.compare(q, f.vectors.get(ns.node)), ns.score, 0f);
                }
            }
        }
    }

    /** An adaptive search must not change what later searches on the same searcher do. */
    @Test
    public void searcherReuseAfterAdaptiveSearch() throws Exception {
        Random random = getRandom();
        var f = new Fixture(1500, 16, COSINE, true, random);
        try (var reused = new GraphSearcher(f.graph)) {
            for (int qi = 0; qi < 20; qi++) {
                var q = f.query(random);
                reused.search(f.exact(f.query(random)), 5, 5, Float.NEGATIVE_INFINITY, 0f, Bits.ALL, AdaptiveTermination.of(COSINE, 0.01f));

                // a beam search through the public API
                var afterAdaptive = reused.search(f.exact(q), 10, 50, Float.NEGATIVE_INFINITY, 0f, Bits.ALL);
                try (var fresh = new GraphSearcher(f.graph)) {
                    var expected = fresh.search(f.exact(q), 10, 50, Float.NEGATIVE_INFINITY, 0f, Bits.ALL);
                    assertArrayEquals(nodes(expected), nodes(afterAdaptive));
                    assertEquals(expected.getVisitedCount(), afterAdaptive.getVisitedCount());
                }

                // and the lower-level calls GraphIndexBuilder makes
                reused.search(f.exact(f.query(random)), 5, 5, Float.NEGATIVE_INFINITY, 0f, Bits.ALL, AdaptiveTermination.of(COSINE, 0.01f));
                try (var fresh = new GraphSearcher(f.graph)) {
                    assertEquals(layer0Search(fresh, f, q), layer0Search(reused, f, q));
                }
            }
        }
    }

    private static List<Integer> layer0Search(GraphSearcher searcher, Fixture f, VectorFloat<?> q) {
        var ssp = f.exact(q);
        var entry = searcher.getView().entryNode();
        searcher.initializeInternal(ssp, new ImmutableGraphIndex.NodeAtLevel(0, entry.node), Bits.ALL);
        searcher.searchOneLayer(ssp, 30, 0f, 0, Bits.ALL);
        var out = new ArrayList<Integer>();
        out.add(searcher.getVisitedCount());
        searcher.approximateResults.foreach((node, score) -> out.add(node));
        searcher.approximateResults.clear();
        return out;
    }

    @Test
    public void thresholdStillApplies() throws Exception {
        Random random = getRandom();
        var f = new Fixture(1000, 8, COSINE, false, random);
        try (var searcher = new GraphSearcher(f.graph)) {
            for (int qi = 0; qi < 30; qi++) {
                var q = f.query(random);
                float threshold = 0.6f + 0.3f * random.nextFloat();
                var r = searcher.search(f.exact(q), 20, 20, threshold, 0f, Bits.ALL, AdaptiveTermination.of(COSINE, 0.5f));
                for (var ns : r.getNodes()) {
                    assertTrue(ns.score >= threshold);
                }
            }
        }
    }

    /** approximate scores walk the graph; rerankK of them are reranked exactly and the best topK returned */
    @Test
    public void approximateScoresWithReranking() throws Exception {
        Random random = getRandom();
        var f = new Fixture(2000, 32, COSINE, false, random);
        var pq = ProductQuantization.compute(f.ravv, 8, 256, false);
        var pqv = pq.encodeAll(f.ravv);
        int topK = 10, rerankK = 30;
        double adaptiveHits = 0, beamHits = 0;
        try (var searcher = new GraphSearcher(f.graph)) {
            for (int qi = 0; qi < 40; qi++) {
                var q = f.query(random);
                var ssp = new DefaultSearchScoreProvider(pqv.scoreFunctionFor(q, COSINE), f.ravv.rerankerFor(q, COSINE));
                var r = searcher.search(ssp, topK, rerankK, 0f, 0f, Bits.ALL, AdaptiveTermination.of(COSINE, 0.3f));
                assertEquals(topK, r.getNodes().length);
                assertEquals(rerankK, r.getRerankedCount());
                for (var ns : r.getNodes()) {
                    assertEquals(COSINE.compare(q, f.vectors.get(ns.node)), ns.score, 1e-6f);
                }
                var truth = new HashSet<Integer>();
                for (int node : bruteForce(f, q, null, Bits.ALL, 0, topK)) truth.add(node);
                for (var ns : r.getNodes()) if (truth.contains(ns.node)) adaptiveHits++;
                var b = searcher.search(ssp, topK, rerankK, 0f, 0f, Bits.ALL);
                for (var ns : b.getNodes()) if (truth.contains(ns.node)) beamHits++;
            }
        }
        // the approximate walk with the rule finds results of the same quality as the beam walk (PQ limits both)
        assertTrue("adaptive " + adaptiveHits + " beam " + beamHits, adaptiveHits >= 0.8 * beamHits);
    }

    @Test
    public void onDiskGraphSearchesTheSame() throws Exception {
        Random random = getRandom();
        var f = new Fixture(1000, 16, EUCLIDEAN, false, random);
        var dir = Files.createTempDirectory("adaptive");
        var path = dir.resolve("graph");
        try {
            TestUtil.writeGraph(f.graph, f.ravv, path);
            try (var supplier = new SimpleMappedReader.Supplier(path);
                 var onDisk = OnDiskGraphIndex.load(supplier);
                 var heapSearcher = new GraphSearcher(f.graph);
                 var diskSearcher = new GraphSearcher(onDisk)) {
                for (int qi = 0; qi < 30; qi++) {
                    var q = f.query(random);
                    var rule = AdaptiveTermination.of(EUCLIDEAN, random.nextFloat());
                    var accept = randomFilter(random, f.vectors.size());
                    var a = heapSearcher.search(f.exact(q), 10, 10, Float.NEGATIVE_INFINITY, 0f, accept, rule);
                    var b = diskSearcher.search(f.exact(q), 10, 10, Float.NEGATIVE_INFINITY, 0f, accept, rule);
                    assertArrayEquals(nodes(a), nodes(b));
                    assertEquals(a.getVisitedCount(), b.getVisitedCount());
                }
            }
        } finally {
            TestUtil.deleteQuietly(dir);
        }
    }

    @Test
    public void edgeCases() throws Exception {
        Random random = getRandom();
        var rule = AdaptiveTermination.of(COSINE, 0.2f);
        // empty graph
        var vectors = new ArrayList<VectorFloat<?>>();
        vectors.add(TestUtil.randomVector(random, 4));
        var ravv = new ListRandomAccessVectorValues(vectors, 4);
        try (var builder = new GraphIndexBuilder(ravv, COSINE, 4, 10, 1.2f, 1.2f, false);
             var searcher = new GraphSearcher(builder.getGraph())) {
            var r = searcher.search(DefaultSearchScoreProvider.exact(vectors.get(0), COSINE, ravv), 5, 5, 0f, 0f, Bits.ALL, rule);
            assertEquals(0, r.getNodes().length);
        }
        // one node
        var single = new Fixture(1, 4, COSINE, false, random);
        try (var searcher = new GraphSearcher(single.graph)) {
            var r = searcher.search(single.exact(single.query(random)), 5, 5, 0f, 0f, Bits.ALL, rule);
            assertEquals(1, r.getNodes().length);
            assertEquals(0, r.getNodes()[0].node);
        }
        var f = new Fixture(300, 8, COSINE, false, random);
        try (var searcher = new GraphSearcher(f.graph)) {
            var q = f.query(random);
            // nothing wanted
            var none = searcher.search(f.exact(q), 0, 0, 0f, 0f, Bits.ALL, rule);
            assertEquals(0, none.getNodes().length);
            // nothing accepted: the whole reachable graph is searched and nothing returned
            var rejected = searcher.search(f.exact(q), 10, 10, 0f, 0f, Bits.NONE, rule);
            assertEquals(0, rejected.getNodes().length);
            assertEquals(reachableFromEntry(f.graph).cardinality() - 1, rejected.getVisitedCount());
            // more wanted than exist
            var all = searcher.search(f.exact(q), 500, 500, 0f, 0f, Bits.ALL, rule);
            assertEquals(reachableFromEntry(f.graph).cardinality(), all.getNodes().length);
            // argument checks are unchanged
            assertThrows(IllegalArgumentException.class, () -> searcher.search(f.exact(q), 10, 5, 0f, 0f, Bits.ALL, rule));
            assertThrows(IllegalArgumentException.class, () -> searcher.search(f.exact(q), 10, 10, 0f, 0f, null, rule));
            // null means beam search
            var viaNull = searcher.search(f.exact(q), 10, 40, 0f, 0f, Bits.ALL, null);
            var beam = searcher.search(f.exact(q), 10, 40, 0f, 0f, Bits.ALL);
            assertArrayEquals(nodes(beam), nodes(viaNull));
            assertEquals(beam.getVisitedCount(), viaNull.getVisitedCount());
        }
    }

    // ---- helpers ----

    private static BitSet reachableFromEntry(ImmutableGraphIndex graph) throws Exception {
        var seen = new BitSet();
        try (var view = graph.getView()) {
            var stack = new ArrayDeque<Integer>();
            int entry = view.entryNode().node;
            seen.set(entry);
            stack.push(entry);
            while (!stack.isEmpty()) {
                for (var it = view.getNeighborsIterator(0, stack.pop()); it.hasNext(); ) {
                    int n = it.nextInt();
                    if (!seen.get(n)) {
                        seen.set(n);
                        stack.push(n);
                    }
                }
            }
        }
        return seen;
    }

    /** nodes ranked [from, from + count) by exact score among the accepted ones in {@code within} (null: all) */
    private static int[] bruteForce(Fixture f, VectorFloat<?> q, BitSet within, Bits accept, int from, int count) {
        var rows = new ArrayList<float[]>();
        for (int i = 0; i < f.vectors.size(); i++) {
            if ((within == null || within.get(i)) && accept.get(i)) {
                rows.add(new float[] {i, f.vsf.compare(q, f.vectors.get(i))});
            }
        }
        rows.sort((a, b) -> a[1] != b[1] ? Float.compare(b[1], a[1]) : Float.compare(a[0], b[0]));
        return rows.subList(Math.min(from, rows.size()), Math.min(from + count, rows.size())).stream()
                   .mapToInt(r -> (int) r[0]).toArray();
    }

    /** reachable in the base layer from the entry node or from any node of an upper layer */
    private static BitSet reachableFromAllUpperLevelNodes(ImmutableGraphIndex graph) throws Exception {
        var seen = new BitSet();
        try (var view = graph.getView()) {
            var stack = new ArrayDeque<Integer>();
            stack.push(view.entryNode().node);
            for (int level = 1; level <= graph.getMaxLevel(); level++) {
                for (var it = graph.getNodes(level); it.hasNext(); ) {
                    stack.push(it.nextInt());
                }
            }
            for (int node : stack) seen.set(node);
            while (!stack.isEmpty()) {
                for (var it = view.getNeighborsIterator(0, stack.pop()); it.hasNext(); ) {
                    int n = it.nextInt();
                    if (!seen.get(n)) {
                        seen.set(n);
                        stack.push(n);
                    }
                }
            }
        }
        return seen;
    }
}
