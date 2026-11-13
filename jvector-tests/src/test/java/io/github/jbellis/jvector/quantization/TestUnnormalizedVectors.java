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

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
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
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntFunction;

import static org.junit.Assert.assertTrue;

/**
 * Dot-product search over vectors whose lengths vary 100x must rank by the full dot product, norms included, for exact,
 * PQ and ASH scoring, both by brute force and through a graph built with that scoring. The data is chosen so that a
 * scorer that ignored norms (ranking by direction alone) would miss most of the true neighbors. Each scorer is judged
 * against itself on the same directions at unit length, so its accuracy at a given budget does not decide the result:
 * a scorer that keeps norm information is at least as accurate on the mixed lengths and clearly above norm-blind ranking.
 */
public class TestUnnormalizedVectors {
    private static final int N = 1000, DIM = 64, QUERIES = 100, K = 10;
    // Recall@10 over 100 queries moves by about 0.01 between backends and graph builds; allow several times that.
    private static final double NOISE = 0.05;
    // How far above norm-blind ranking a norm-aware scorer must be.
    private static final double MARGIN = 0.1;

    @Test
    public void graphSearchRanksByTheFullDotProduct() throws Exception {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var random = new Random(20261112);
        List<VectorFloat<?>> base = new ArrayList<>(N), unitBase = new ArrayList<>(N);
        for (int i = 0; i < N; i++) {
            var v = randomVector(vts, random);
            base.add(v);
            unitBase.add(unit(v));
        }
        List<VectorFloat<?>> queries = new ArrayList<>(QUERIES), unitQueries = new ArrayList<>(QUERIES);
        for (int i = 0; i < QUERIES; i++) {
            var q = randomVector(vts, random);
            queries.add(q);
            unitQueries.add(unit(q));
        }

        // The norm-blind ranking (cosine) against the dot-product truth: what a scorer ignoring norms would score.
        List<Set<Integer>> dotTruth = truth(queries, base), cosineTruth = new ArrayList<>();
        for (var q : queries) cosineTruth.add(topK(i -> VectorUtil.cosine(q, base.get(i))));
        double normBlind = recall(cosineTruth, dotTruth);
        System.out.format("norm-blind (cosine) recall@%d against the dot-product truth: %.3f%n", K, normBlind);

        // The same directions at unit length are the control: a scorer as accurate on mixed lengths as there loses
        // nothing to the norms; one less accurate on mixed lengths does.
        List<String> failures = new ArrayList<>();
        var mixed = runAll("mixed lengths", base, queries, dotTruth);
        var unitResults = runAll("unit length  ", unitBase, unitQueries, truth(unitQueries, unitBase));
        for (var name : mixed.keySet()) {
            double[] m = mixed.get(name), u = unitResults.get(name);
            System.out.format("%-22s mixed scan %.3f graph %.3f | unit scan %.3f graph %.3f | mixed - unit %+.3f%n",
                    name, m[0], m[1], u[0], u[1], m[0] - u[0]);
            for (int i = 0; i < 2; i++) {
                String path = i == 0 ? "scan" : "graph";
                if (!(m[i] >= u[i] - NOISE)) failures.add(name + " " + path + " loses to norms: " + m[i] + " < unit " + u[i]);
                if (!(m[i] >= normBlind + MARGIN)) failures.add(name + " " + path + " near norm-blind: " + m[i]);
            }
        }
        assertTrue("the data must separate norm-aware from norm-blind ranking: " + normBlind, normBlind < 0.5);
        assertTrue("must rank by the full dot product: " + failures, failures.isEmpty());
    }

    /** Recall of every scorer configuration on one dataset; budgets are matched at 16 and 32 bytes per vector. */
    private static java.util.Map<String, double[]> runAll(String label, List<VectorFloat<?>> base,
                                                          List<VectorFloat<?>> queries, List<Set<Integer>> truth)
            throws java.io.IOException {
        var ravv = new ListRandomAccessVectorValues(base, DIM);
        var vsf = VectorSimilarityFunction.DOT_PRODUCT;
        var results = new java.util.LinkedHashMap<String, double[]>();
        results.put("exact (256 B)", check(label, "exact (256 B)", BuildScoreProvider.randomAccessScoreProvider(ravv, vsf),
                ravv, queries, truth, q -> DefaultSearchScoreProvider.exact(q, vsf, ravv).scoreFunction()));
        // PQ: 16 B = 16 subspaces of 4 dimensions; 32 B = 32 subspaces of 2; 256 centroids each.
        for (int m : new int[] {16, 32}) {
            var pqv = ProductQuantization.compute(ravv, m, 256, false).encodeAll(ravv, ForkJoinPool.commonPool());
            String name = "PQ " + m + "x8 (" + m + " B)";
            results.put(name, check(label, name, BuildScoreProvider.pqBuildScoreProvider(vsf, pqv), ravv, queries, truth,
                    q -> pqv.precomputedScoreFunctionFor(q, vsf)));
        }
        // ASH, one landmark, 5-byte header: 16 B = 44 dims x 2 bits or 22 x 4; 32 B = 54 x 4 (or 64 x 2 = 21 B).
        int[][] ash = {{2, 44}, {4, 22}, {4, 54}, {2, 64}};
        for (int[] config : ash) {
            int bits = config[0], dims = config[1];
            int totalBits = AsymmetricHashing.HEADER_BITS + dims * bits;
            var ashv = AsymmetricHashing.initialize(ravv, AsymmetricHashing.RANDOM, totalBits, 1, bits)
                    .encodeAll(ravv, ForkJoinPool.commonPool());
            String name = "ASH " + dims + "x" + bits + " (" + (totalBits / 8) + " B)";
            results.put(name, check(label, name, BuildScoreProvider.ashBuildScoreProvider(vsf, ashv), ravv, queries,
                    truth, q -> ashv.precomputedScoreFunctionFor(q, vsf)));
        }
        return results;
    }

    private static List<Set<Integer>> truth(List<VectorFloat<?>> queries, List<VectorFloat<?>> base) {
        List<Set<Integer>> truth = new ArrayList<>();
        for (var q : queries) truth.add(topK(i -> VectorUtil.dotProduct(q, base.get(i))));
        return truth;
    }

    private static VectorFloat<?> unit(VectorFloat<?> v) {
        var u = v.copy();
        VectorUtil.l2normalize(u);
        return u;
    }

    /** Random direction, length log-uniform in [0.1, 10]. */
    private static VectorFloat<?> randomVector(io.github.jbellis.jvector.vector.types.VectorTypeSupport vts, Random r) {
        var v = vts.createFloatVector(DIM);
        for (int d = 0; d < DIM; d++) v.set(d, (float) r.nextGaussian());
        VectorUtil.l2normalize(v);
        VectorUtil.scale(v, (float) Math.pow(10, -1 + 2 * r.nextDouble()));
        return v;
    }

    /**
     * Brute-force and graph recall@K against {@code truth}, as {scan, graph}. The graph is built with the same scoring
     * and searched with adaptive stopping off.
     */
    private static double[] check(String label, String name, BuildScoreProvider bsp, ListRandomAccessVectorValues ravv,
                                  List<VectorFloat<?>> queries, List<Set<Integer>> truth,
                                  java.util.function.Function<VectorFloat<?>, ScoreFunction> scorerFor) {
        List<Set<Integer>> scan = new ArrayList<>(), graphResults = new ArrayList<>();
        ImmutableGraphIndex graph;
        try (var builder = new GraphIndexBuilder(bsp, DIM, 16, 100, 1.2f, 1.2f, false)) {
            graph = builder.build(ravv);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        try (var searcher = new GraphSearcher(graph)) {
            searcher.useAdaptiveTermination(false);
            for (var q : queries) {
                var sf = scorerFor.apply(q);
                scan.add(topK(sf::similarityTo));
                var result = searcher.search(new DefaultSearchScoreProvider(sf), K, 100, 0f, 0f, Bits.ALL);
                var found = new HashSet<Integer>();
                for (var ns : result.getNodes()) found.add(ns.node);
                graphResults.add(found);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        double scanRecall = recall(scan, truth), graphRecall = recall(graphResults, truth);
        System.out.format("%s %-22s brute-force recall@%d %.3f, graph recall@%d %.3f%n",
                label, name, K, scanRecall, K, graphRecall);
        return new double[] {scanRecall, graphRecall};
    }

    private static Set<Integer> topK(IntFunction<Float> score) {
        Integer[] ids = new Integer[N];
        float[] scores = new float[N];
        for (int i = 0; i < N; i++) {
            ids[i] = i;
            scores[i] = score.apply(i);
        }
        Arrays.sort(ids, (a, b) -> Float.compare(scores[b], scores[a]));
        return new HashSet<>(Arrays.asList(ids).subList(0, K));
    }

    private static double recall(List<Set<Integer>> found, List<Set<Integer>> truth) {
        int hits = 0;
        for (int i = 0; i < truth.size(); i++) {
            for (int id : found.get(i)) if (truth.get(i).contains(id)) hits++;
        }
        return hits / (double) (truth.size() * K);
    }
}
