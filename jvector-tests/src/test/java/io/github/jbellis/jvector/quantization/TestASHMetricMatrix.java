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

import io.github.jbellis.jvector.disk.ByteBufferIndexWriter;
import io.github.jbellis.jvector.disk.ByteBufferReader;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.MockVectorValues;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.StreamingASHIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedASH;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.AfterClass;
import org.junit.Test;

import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntFunction;

import static org.junit.Assert.*;

/**
 * Every ASH setting supports DOT_PRODUCT, COSINE and EUCLIDEAN: bit widths 1 to 9, one and several landmarks, ITQ and
 * RANDOM, in-memory, symmetric, fused, separated and streaming scoring, and every kernel mode. Each path must match the
 * scalar reference, and the reference must estimate the exact similarity, better as the width grows. Run under each
 * vectorization provider to cover the native, Panama and Java kernels.
 */
public class TestASHMetricMatrix {
    private static final VectorSimilarityFunction[] METRICS = {
            VectorSimilarityFunction.DOT_PRODUCT, VectorSimilarityFunction.COSINE, VectorSimilarityFunction.EUCLIDEAN};
    private static final int DIM = 64;
    // Not a multiple of 8 or 64, so every layout has a partial tail.
    private static final int PROJECTED = 37;
    private static final int[] LANDMARKS = {1, 3};
    private static final int[] OPTIMIZERS = {AsymmetricHashing.ITQ, AsymmetricHashing.RANDOM};
    private static final int[] BLOCK_SIZES = {8, 16, 32};
    private static final String SINGLE = "jvector.ash.singleKernel", BLOCK = "jvector.ash.blockKernel",
            GATHERED = "jvector.ash.gatheredScoring";
    /** Kernel modes: single kernel, block kernel, gathered scoring. */
    private static final String[][] MODES = {
            {"scalar", "scalar", "false"}, {"simd", "simd", "false"}, {"auto", "auto", "false"}, {"auto", "auto", "true"}};

    /** Modes a backend lacks, reported once so a run shows what it did not exercise. */
    private static final java.util.TreeMap<String, TreeSet<Integer>> SKIPPED = new java.util.TreeMap<>();
    /** Largest fused field-block error seen, in standard deviations of the standalone scores. */
    private static final java.util.TreeMap<String, Double> FIELD_BLOCK_ERRORS = new java.util.TreeMap<>();
    /** INT8 query rounding: measured up to about 0.03 standard deviations at 3 to 9 bits. */
    private static final double FIELD_BLOCK_ERROR = 0.05;

    @AfterClass
    public static void reportSkipped() {
        System.out.println("TestASHMetricMatrix on " + VectorizationProvider.getInstance().getClass().getSimpleName()
                + ", unsupported SIMD modes: " + (SKIPPED.isEmpty() ? "none" : SKIPPED));
        var errors = new StringBuilder();
        FIELD_BLOCK_ERRORS.forEach((k, v) -> errors.append(String.format(" %s: %.4f;", k, v)));
        System.out.println("TestASHMetricMatrix largest fused field-block errors (standard deviations):" + errors);
    }

    /**
     * Embedding-like vectors around a shared offset, with spread that decays across components and lengths from 0.5
     * to 4, so that the three similarities rank them differently.
     */
    static VectorFloat<?>[] vectors(int count, long seed) {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        Random random = new Random(seed);
        Random shared = new Random(1234);
        float[] offset = new float[DIM];
        for (int d = 0; d < DIM; d++) offset[d] = 0.4f * (float) shared.nextGaussian();
        VectorFloat<?>[] out = new VectorFloat<?>[count];
        for (int i = 0; i < count; i++) {
            out[i] = vts.createFloatVector(DIM);
            float length = 0.5f + 3.5f * random.nextFloat();
            for (int d = 0; d < DIM; d++) {
                out[i].set(d, length * (offset[d] + (float) random.nextGaussian() / (float) Math.sqrt(1 + d / 4.0)));
            }
        }
        return out;
    }

    private static AsymmetricHashing train(VectorFloat<?>[] data, VectorSimilarityFunction vsf, int optimizer,
                                           int projected, int bits, int landmarks) {
        try {
            return AsymmetricHashing.initialize(MockVectorValues.fromValues(data), vsf, optimizer,
                AsymmetricHashing.HEADER_BITS + projected * bits, landmarks, bits);
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }

    private static String name(VectorSimilarityFunction vsf, int bits, int landmarks, int optimizer) {
        return vsf + " bits=" + bits + " C=" + landmarks + " " + (optimizer == AsymmetricHashing.ITQ ? "ITQ" : "RANDOM");
    }

    /** The metric's own value of a similarity: a dot product or cosine, or a squared distance. */
    private static double raw(VectorSimilarityFunction vsf, float similarity) {
        return vsf == VectorSimilarityFunction.EUCLIDEAN ? 1.0 / similarity - 1.0 : 2.0 * similarity - 1.0;
    }

    /** RMS error of estimates against exact values, over the exact values' standard deviation. */
    private static double normalizedError(VectorSimilarityFunction vsf, float[] estimates, float[] exact) {
        double mean = 0, error = 0, variance = 0;
        for (float e : exact) mean += raw(vsf, e);
        mean /= exact.length;
        for (int i = 0; i < exact.length; i++) {
            double x = raw(vsf, exact[i]);
            double delta = raw(vsf, estimates[i]) - x;
            error += delta * delta;
            variance += (x - mean) * (x - mean);
        }
        return Math.sqrt(error / variance);
    }

    /**
     * The scalar reference estimates every metric at every width, and its error falls as the width grows. Bounds are
     * the measured errors with margin; at full projection the error is the code's, not the projection's.
     */
    @Test
    public void theReferenceEstimatesEveryMetricAtEveryWidth() {
        VectorFloat<?>[] data = vectors(800, 11);
        VectorFloat<?>[] queries = vectors(8, 12);
        var errors = new StringBuilder();
        runWithModes(MODES[0], () -> {
            for (var vsf : METRICS) {
                for (int optimizer : OPTIMIZERS) {
                    for (int landmarks : LANDMARKS) {
                        double previous = Double.POSITIVE_INFINITY;
                        for (int bits = 1; bits <= 9; bits++) {
                            var ash = train(data, vsf, optimizer, DIM, bits, landmarks);
                            assertEquals(vsf, ash.similarityFunction());
                            var encoded = ash.encodeAll(MockVectorValues.fromValues(data), ForkJoinPool.commonPool());
                            float[] estimates = new float[queries.length * data.length];
                            float[] exact = new float[estimates.length];
                            for (int q = 0; q < queries.length; q++) {
                                var f = encoded.scoreFunctionFor(queries[q], vsf);
                                for (int i = 0; i < data.length; i++) {
                                    estimates[q * data.length + i] = f.similarityTo(i);
                                    exact[q * data.length + i] = vsf.compare(queries[q], data[i]);
                                }
                            }
                            double error = normalizedError(vsf, estimates, exact);
                            String where = name(vsf, bits, landmarks, optimizer);
                            errors.append(String.format("%s error %.4f%n", where, error));
                            assertTrue(where + " error " + error, error < referenceBound(bits));
                            // Each extra bit at least halves the code's error, until the training noise dominates.
                            assertTrue(where + " error " + error + " after " + previous, error < previous * 1.05 + 0.002);
                            previous = error;
                        }
                    }
                }
            }
        });
        System.out.print(errors);
    }

    private static double referenceBound(int bits) {
        return bits == 1 ? 0.6 : bits == 2 ? 0.35 : 0.45 / (1 << (bits - 2)) + 0.02;
    }

    /**
     * In-memory asymmetric scoring: node, batch, precomputed, prepared, block, growable and reloaded (separated) codes
     * score like the scalar reference in every kernel mode.
     */
    @Test
    public void everyAsymmetricPathMatchesTheReference() throws Exception {
        VectorFloat<?>[] data = vectors(300, 21);
        VectorFloat<?>[] queries = vectors(3, 22);
        int[] nodes = new int[data.length];
        for (int i = 0; i < nodes.length; i++) nodes[i] = i;
        for (var vsf : METRICS) {
            for (int optimizer : OPTIMIZERS) {
                for (int landmarks : LANDMARKS) {
                    for (int bits = 1; bits <= 9; bits++) {
                        var ash = train(data, vsf, optimizer, PROJECTED, bits, landmarks);
                        var encoded = ash.encodeAll(MockVectorValues.fromValues(data), ForkJoinPool.commonPool());
                        var separated = reload(encoded);
                        // Growable codes have one landmark and encode one vector at a time; compare them with codes
                        // encoded the same way (batched projection can differ in the last bit).
                        var singly = new ASHVectors(ash, encodeEach(ash, data));
                        MutableASHVectors mutable = null;
                        if (landmarks == 1) {
                            mutable = new MutableASHVectors(ash);
                            for (int i = 0; i < data.length; i++) mutable.encodeAndSet(i, data[i]);
                        }
                        String model = name(vsf, bits, landmarks, optimizer);
                        for (var q : queries) {
                            float[] expected = withModes(MODES[0], () -> scores(encoded.scoreFunctionFor(q, vsf), data.length));
                            float[] expectedSingly = withModes(MODES[0], () -> scores(singly.scoreFunctionFor(q, vsf), data.length));
                            for (String[] mode : MODES) {
                                String where = model + " mode=" + String.join("/", mode);
                                var finalMutable = mutable;
                                supported(where, () -> withModes(mode, () -> {
                                    check(where + " node", vsf, expected, encoded.scoreFunctionFor(q, vsf), nodes);
                                    check(where + " separated", vsf, expected, separated.scoreFunctionFor(q, vsf), nodes);
                                    check(where + " precomputed", vsf, expected,
                                            encoded.precomputedScoreFunctionFor(q, vsf), nodes);
                                    var preparer = encoded.queryPreparer();
                                    var prepared = preparer.newState();
                                    preparer.prepare(q, prepared);
                                    check(where + " prepared", vsf, expected, encoded.scoreFunctionFor(prepared), nodes);
                                    check(where + " block", expected, encoded.blockScorerFor(q));
                                    for (int bs : BLOCK_SIZES) {
                                        check(where + " block" + bs, expected, encoded.blockScorerFor(q, bs));
                                    }
                                    if (finalMutable != null) {
                                        check(where + " mutable", vsf, expectedSingly,
                                                finalMutable.scoreFunctionFor(q, vsf), nodes);
                                    }
                                    return null;
                                }));
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Symmetric construction scoring: every kernel scores like the scalar kernel, diversity scoring matches it, and
     * scalar symmetric scores estimate the exact similarity between stored vectors.
     */
    @Test
    public void symmetricScoringMatchesTheReferenceAndEstimatesEveryMetric() {
        VectorFloat<?>[] data = vectors(400, 31);
        for (var vsf : METRICS) {
            for (int optimizer : OPTIMIZERS) {
                for (int landmarks : LANDMARKS) {
                    for (int bits = 1; bits <= 9; bits++) {
                        var ash = train(data, vsf, optimizer, DIM, bits, landmarks);
                        var encoded = new ASHVectors(ash, encodeEach(ash, data));
                        String model = name(vsf, bits, landmarks, optimizer);
                        var scalar = new ASHSymmetricScorer(encoded, ASHSymmetricScorer.Kernel.SCALAR);
                        MutableASHVectors mutable = null;
                        if (landmarks == 1) {
                            mutable = new MutableASHVectors(ash);
                            for (int i = 0; i < data.length; i++) mutable.encodeAndSet(i, data[i]);
                        }
                        int[] others = new int[data.length - 20];
                        for (int i = 0; i < others.length; i++) others[i] = 20 + i;
                        float[] estimates = new float[20 * others.length], exact = new float[estimates.length];
                        for (int a = 0; a < 20; a++) {
                            float[] expected = new float[others.length];
                            var f = scalar.scoreFunctionFor(a);
                            assertEquals(model, vsf, f.similarityFunction());
                            for (int i = 0; i < others.length; i++) {
                                expected[i] = f.similarityTo(others[i]);
                                estimates[a * others.length + i] = expected[i];
                                exact[a * others.length + i] = vsf.compare(data[a], data[others[i]]);
                            }
                            check(model + " diversity a=" + a, vsf, expected, encoded.diversityFunctionFor(a, vsf), others);
                            if (mutable != null) {
                                // Growable codes score through their own path, whose float order can differ.
                                checkWithin(model + " mutable diversity a=" + a, expected,
                                        mutable.diversityFunctionFor(a, vsf), others, 1e-3f);
                            }
                            for (var kernel : new ASHSymmetricScorer.Kernel[]{
                                    ASHSymmetricScorer.Kernel.SIMD, ASHSymmetricScorer.Kernel.AUTO}) {
                                String where = model + " symmetric " + kernel + " a=" + a;
                                final int origin = a;
                                supported(model + " symmetric " + kernel, () -> {
                                    var scorer = new ASHSymmetricScorer(encoded, kernel);
                                    check(where, vsf, expected, scorer.scoreFunctionFor(origin), others);
                                    float[] all = new float[data.length];
                                    scorer.blockScorerFor(encoded.get(origin), 32, kernel).scoreRange(0, data.length, all);
                                    for (int i = 0; i < others.length; i++) {
                                        assertClose(where + " block " + others[i], expected[i], all[others[i]],
                                                tolerance(expected, 1e-4f));
                                    }
                                    return null;
                                });
                            }
                        }
                        double error = normalizedError(vsf, estimates, exact);
                        assertTrue(model + " symmetric error " + error, error < symmetricBound(bits));
                    }
                }
            }
        }
    }

    /** Both vectors carry their codes' error, so symmetric estimates are looser than asymmetric ones. */
    private static double symmetricBound(int bits) {
        return bits == 1 ? 0.9 : bits == 2 ? 0.5 : 0.6 / (1 << (bits - 2)) + 0.03;
    }

    /**
     * Fused on-disk scoring, in every LUT and field-block layout and block size, scores stored neighborhoods and
     * direct nodes like standalone codes, and a graph search with fused or separated codes finds the exact neighbors.
     */
    @Test
    public void fusedAndSeparatedOnDiskScoringMatchStandaloneCodes() throws Exception {
        VectorFloat<?>[] data = vectors(500, 41);
        VectorFloat<?>[] queries = vectors(6, 42);
        var values = MockVectorValues.fromValues(data);
        for (var vsf : METRICS) {
            // The graph does not depend on the codes: build it once from exact scores.
            ImmutableGraphIndex graph = new GraphIndexBuilder(BuildScoreProvider.randomAccessScoreProvider(values, vsf),
                    DIM, 12, 48, 1.2f, 1.2f, true, true).build(values);
            for (int optimizer : OPTIMIZERS) {
                for (int landmarks : LANDMARKS) {
                    for (int bits = 1; bits <= 9; bits++) {
                        var ash = train(data, vsf, optimizer, PROJECTED, bits, landmarks);
                        var encoded = ash.encodeAll(values, ForkJoinPool.commonPool());
                        var separated = reload(encoded);
                        String model = name(vsf, bits, landmarks, optimizer);
                        checkSeparatedSearch(model, vsf, graph, data, queries, separated);
                        for (int bs : BLOCK_SIZES) {
                            checkFused(model + " block=" + bs, vsf, graph, data, queries, ash, encoded, bs);
                        }
                    }
                }
            }
        }
    }

    private static void checkFused(String model, VectorSimilarityFunction vsf, ImmutableGraphIndex graph,
                                   VectorFloat<?>[] data, VectorFloat<?>[] queries, AsymmetricHashing ash,
                                   ASHVectors encoded, int blockSize) throws Exception {
        var path = Files.createTempFile("ash-metric-matrix-", ".index");
        try {
            var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
            try (var view = graph.getView(); var writer = new OnDiskGraphIndexWriter.Builder(graph, path)
                    .with(new FusedASH(graph.maxDegree(), ash, blockSize)).with(new InlineVectors(DIM)).build()) {
                suppliers.put(FeatureId.FUSED_ASH, n -> new FusedASH.State(view, encoded, n));
                suppliers.put(FeatureId.INLINE_VECTORS, n -> new InlineVectors.State(data[n]));
                writer.write(suppliers);
            }
            try (var reader = new SimpleMappedReader.Supplier(path); var disk = OnDiskGraphIndex.load(reader, 0)) {
                for (String[] mode : new String[][]{MODES[0], MODES[2]}) {
                    String where = model + " mode=" + String.join("/", mode);
                    withModes(mode, () -> {
                        int hits = 0;
                        for (var q : queries) {
                            // Score and search through the searcher's own view, as a search does.
                            try (var searcher = new GraphSearcher(disk)) {
                                var view = (OnDiskGraphIndex.View) searcher.getView();
                                var standalone = encoded.scoreFunctionFor(q, vsf);
                                float[] expected = scores(standalone, data.length);
                                float tolerance = fusedTolerance(vsf, ash.bitsPerDimension, expected);
                                var fused = view.approximateScoreFunctionFor(q, vsf);
                                assertEquals(where, vsf, fused.similarityFunction());
                                // Upper-layer nodes are scored directly, from their stored source codes.
                                assertTrue(where + " needs upper layers", graph.getMaxLevel() > 0);
                                for (var it = graph.getNodes(1); it.hasNext(); ) {
                                    int node = it.nextInt();
                                    assertEquals(where + " direct " + node, expected[node], fused.similarityTo(node),
                                            tolerance(expected, 1e-4f));
                                }
                                for (int origin = 0; origin < data.length; origin += 23) {
                                    fused.enableSimilarityToNeighbors(origin);
                                    var neighbors = view.getNeighborsIterator(0, origin);
                                    for (int lane = 0; neighbors.hasNext(); lane++) {
                                        int n = neighbors.nextInt();
                                        float got = fused.similarityToNeighbor(origin, lane);
                                        String at = where + " origin " + origin + " lane " + lane;
                                        if (FusedASHLayout.usesFieldBlocks(ash.bitsPerDimension)) {
                                            // In the metric's own units: a squared distance's error does not
                                            // depend on the distance, its similarity's does.
                                            double error = Math.abs(raw(vsf, got) - raw(vsf, expected[n]));
                                            FIELD_BLOCK_ERRORS.merge(vsf + " bits=" + ash.bitsPerDimension,
                                                    FIELD_BLOCK_ERROR * error / tolerance, Math::max);
                                            assertTrue(at + " expected " + expected[n] + " but was " + got,
                                                    error <= tolerance);
                                        } else {
                                            assertEquals(at, expected[n], got, tolerance);
                                        }
                                    }
                                }
                                var scores = new DefaultSearchScoreProvider(fused, view.rerankerFor(q, vsf));
                                hits += hits(vsf, data, q, searcher.search(scores, 10, 40, 0f, 0f, Bits.ALL));
                            }
                        }
                        checkRecall(where + " fused search", hits, queries.length, ash.bitsPerDimension);
                        return null;
                    });
                }
            }
        } finally {
            Files.deleteIfExists(path);
        }
    }

    /**
     * 1, 2 and 4-bit neighborhoods use the FP32 LUT and agree to rounding, in similarity units. Field blocks score
     * with an INT8-rounded query, whose error is a fraction of the spread of the metric's own values (dot products,
     * cosines or squared distances).
     */
    private static float fusedTolerance(VectorSimilarityFunction vsf, int bits, float[] expected) {
        if (!FusedASHLayout.usesFieldBlocks(bits)) {
            float max = 0;
            for (float e : expected) max = Math.max(max, Math.abs(e));
            return 2e-3f * max + 2e-3f;
        }
        double mean = 0, variance = 0;
        for (float e : expected) mean += raw(vsf, e);
        mean /= expected.length;
        for (float e : expected) variance += (raw(vsf, e) - mean) * (raw(vsf, e) - mean);
        return (float) (FIELD_BLOCK_ERROR * Math.sqrt(variance / expected.length));
    }

    /** Separated codes: an on-disk graph with codes held beside it, scored by the reloaded codes. */
    private static void checkSeparatedSearch(String model, VectorSimilarityFunction vsf, ImmutableGraphIndex graph,
                                             VectorFloat<?>[] data, VectorFloat<?>[] queries, ASHVectors separated)
            throws Exception {
        var path = Files.createTempFile("ash-metric-matrix-separated-", ".index");
        try {
            OnDiskGraphIndex.write(graph, MockVectorValues.fromValues(data), path);
            try (var reader = new SimpleMappedReader.Supplier(path); var disk = OnDiskGraphIndex.load(reader, 0);
                 var view = disk.getView(); var searcher = new GraphSearcher(disk)) {
                int hits = 0;
                for (var q : queries) {
                    var scores = new DefaultSearchScoreProvider(separated.scoreFunctionFor(q, vsf), view.rerankerFor(q, vsf));
                    hits += hits(vsf, data, q, searcher.search(scores, 10, 40, 0f, 0f, Bits.ALL));
                }
                checkRecall(model + " separated search", hits, queries.length, separated.getCompressor().bitsPerDimension);
            }
        } finally {
            Files.deleteIfExists(path);
        }
    }

    /** How many of the exact top 10 a search found. */
    private static int hits(VectorSimilarityFunction vsf, VectorFloat<?>[] data, VectorFloat<?> q,
                            io.github.jbellis.jvector.graph.SearchResult result) {
        var expected = exactTop(vsf, data, q, 10);
        int hits = 0;
        for (var ns : result.getNodes()) if (expected.contains(ns.node)) hits++;
        return hits;
    }

    /** A reranked search of a 500-node graph with beam 40 finds most of the exact top 10; 37-dimension 1 and 2-bit codes less. */
    private static void checkRecall(String where, int hits, int queries, int bits) {
        double recall = hits / (10.0 * queries);
        assertTrue(where + " recall " + recall, recall >= (bits == 1 ? 0.6 : bits == 2 ? 0.7 : 0.8));
    }

    /** A streaming index on a model of every width and metric scores and searches like that model's static codes. */
    @Test
    public void streamingIndexScoresAndSearchesEveryWidth() throws Exception {
        VectorFloat<?>[] data = vectors(800, 51);
        VectorFloat<?>[] queries = vectors(10, 52);
        var values = new ListRandomAccessVectorValues(Arrays.asList(data), DIM);
        int[] nodes = new int[data.length];
        for (int i = 0; i < nodes.length; i++) nodes[i] = i;
        for (var vsf : METRICS) {
            for (int optimizer : OPTIMIZERS) {
                for (int bits = 1; bits <= 9; bits++) {
                    var ash = train(data, vsf, optimizer, PROJECTED, bits, 1);
                    String model = name(vsf, bits, 1, optimizer);
                    var index = StreamingASHIndex.withModel(values, ash,
                            scores -> new GraphIndexBuilder(scores, DIM, 12, 48, 1.2f, 1.2f, false, false),
                            new StreamingASHIndex.Options());
                    for (int i = 0; i < data.length; i++) index.add(i);
                    assertEquals(model, vsf, index.model().similarityFunction());
                    var reference = new ASHVectors(ash, encodeEach(ash, data));
                    // The same codes built into a graph in one pass, searched the same way.
                    var batchGraph = new GraphIndexBuilder(BuildScoreProvider.ashBuildScoreProvider(reference), DIM, 12,
                            48, 1.2f, 1.2f, false, false).build(MockVectorValues.fromValues(data));
                    int hits = 0, batchHits = 0;
                    for (var q : queries) {
                        float[] expected = scores(reference.scoreFunctionFor(q, vsf), data.length);
                        check(model + " streaming codes", vsf, expected, index.codes().scoreFunctionFor(q, vsf), nodes);
                        var top = exactTop(vsf, data, q, 10);
                        for (var ns : index.search(q, 10, 40, Bits.ALL).getNodes()) if (top.contains(ns.node)) hits++;
                        try (var searcher = new GraphSearcher(batchGraph)) {
                            var scores = new DefaultSearchScoreProvider(reference.scoreFunctionFor(q, vsf),
                                    n -> vsf.compare(q, data[n]));
                            for (var ns : searcher.search(scores, 10, 40, 0f, 0f, Bits.ALL).getNodes())
                                if (top.contains(ns.node)) batchHits++;
                        }
                    }
                    double recall = hits / (10.0 * queries.length), batch = batchHits / (10.0 * queries.length);
                    assertTrue(model + " streaming recall " + recall + " vs one-pass " + batch, recall >= batch - 0.1);
                    assertTrue(model + " streaming recall " + recall, recall >= 0.5);
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------------

    private interface Body<T> {
        T run() throws Exception;
    }

    /** Runs {@code body} with the given kernel properties, restoring the previous ones after it. */
    private static <T> T withModes(String[] mode, Body<T> body) {
        String single = System.getProperty(SINGLE), block = System.getProperty(BLOCK), gathered = System.getProperty(GATHERED);
        try {
            System.setProperty(SINGLE, mode[0]);
            System.setProperty(BLOCK, mode[1]);
            System.setProperty(GATHERED, mode[2]);
            return body.run();
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            restore(SINGLE, single);
            restore(BLOCK, block);
            restore(GATHERED, gathered);
        }
    }

    private static void runWithModes(String[] mode, Runnable body) {
        withModes(mode, () -> {
            body.run();
            return null;
        });
    }

    /** Runs {@code body}, recording a forced-SIMD mode the backend lacks instead of failing. */
    private static void supported(String where, Body<?> body) {
        try {
            body.run();
        } catch (IllegalStateException | UnsupportedOperationException e) {
            String message = String.valueOf(e.getMessage());
            if (!message.contains("SIMD") && !message.contains("unavailable")) throw e;
            // One entry per path and message, listing the widths: "symmetric SIMD (message): bits [3, 5]"
            String path = where.replaceAll("^\\S+ bits=\\d+ C=\\d+ (ITQ|RANDOM) ?", "").replaceAll(" a=\\d+", "");
            int bits = Integer.parseInt(where.replaceAll("^\\S+ bits=(\\d+).*", "$1"));
            SKIPPED.computeIfAbsent(path + " (" + message + ")", k -> new TreeSet<>()).add(bits);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static ASHVectors reload(ASHVectors encoded) throws Exception {
        try (var writer = ByteBufferIndexWriter.create(1 << 20, false)) {
            encoded.write(writer);
            return ASHVectors.load(new ByteBufferReader(writer.getWrittenData()));
        }
    }

    private static AsymmetricHashing.QuantizedVector[] encodeEach(AsymmetricHashing ash, VectorFloat<?>[] data) {
        var codes = new AsymmetricHashing.QuantizedVector[data.length];
        for (int i = 0; i < data.length; i++) codes[i] = ash.encode(data[i]);
        return codes;
    }

    private static float[] scores(ScoreFunction f, int count) {
        float[] out = new float[count];
        for (int i = 0; i < count; i++) out[i] = f.similarityTo(i);
        return out;
    }

    /** Rounding differs between kernels in proportion to the scores' magnitude, not each score's. */
    private static float tolerance(float[] expected, float relative) {
        float max = 0f;
        for (float e : expected) max = Math.max(max, Math.abs(e));
        return relative * max + 1e-7f;
    }

    private static void assertClose(String where, float expected, float actual, float tolerance) {
        assertTrue(where + " not finite: " + actual, Float.isFinite(actual));
        assertEquals(where, expected, actual, tolerance);
    }

    /** Single and batch scores of {@code nodes} match {@code expected}, which is indexed like {@code nodes}. */
    private static void check(String where, VectorSimilarityFunction vsf, float[] expected, ScoreFunction f, int[] nodes) {
        assertEquals(where, vsf, f.similarityFunction());
        checkWithin(where, expected, f, nodes, 1e-4f);
    }

    private static void checkWithin(String where, float[] expected, ScoreFunction f, int[] nodes, float relative) {
        float[] batch = new float[nodes.length];
        f.similarityTo(nodes, nodes.length, batch);
        float tolerance = tolerance(expected, relative);
        for (int i = 0; i < nodes.length; i++) {
            assertClose(where + " node " + nodes[i], expected[i], f.similarityTo(nodes[i]), tolerance);
            assertClose(where + " batch " + nodes[i], expected[i], batch[i], tolerance);
        }
    }

    private static void check(String where, float[] expected, ASHBlockScorer f) {
        float[] out = new float[expected.length];
        f.scoreRange(0, expected.length, out);
        float tolerance = tolerance(expected, 1e-4f);
        for (int i = 0; i < expected.length; i++) assertClose(where + " " + i, expected[i], out[i], tolerance);
    }

    private static HashSet<Integer> exactTop(VectorSimilarityFunction vsf, VectorFloat<?>[] data, VectorFloat<?> q, int k) {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < data.length; i++) order.add(i);
        order.sort((a, b) -> Float.compare(vsf.compare(q, data[b]), vsf.compare(q, data[a])));
        return new HashSet<>(order.subList(0, k));
    }

    private static void restore(String key, String value) {
        if (value == null) System.clearProperty(key);
        else System.setProperty(key, value);
    }
}
