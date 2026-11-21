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
import org.junit.Test;

import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntFunction;

import static org.junit.Assert.*;

/**
 * EUCLIDEAN ASH (paper, Appendix A): the model owns its similarity, every scorer returns {@code 1 / (1 + d^2)}, and
 * squared distances are right on vectors far from unit length.
 */
public class TestASHEuclidean {
    private static final VectorSimilarityFunction L2 = VectorSimilarityFunction.EUCLIDEAN;
    private static final VectorSimilarityFunction DOT = VectorSimilarityFunction.DOT_PRODUCT;
    private static final int DIM = 64;

    /**
     * SIFT-like vectors: non-negative integer components around a few cluster centers, with spread that decays across
     * components as real descriptors' and embeddings' spectra do. Lengths are in the hundreds and squared distances in
     * the tens of thousands, past half precision's range unscaled.
     */
    static VectorFloat<?>[] siftLike(int count, long seed) {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        Random random = new Random(seed);
        float[][] centers = new float[12][DIM];
        for (float[] center : centers) for (int d = 0; d < DIM; d++) center[d] = 100f * random.nextFloat();
        VectorFloat<?>[] out = new VectorFloat<?>[count];
        for (int i = 0; i < count; i++) {
            float[] center = centers[random.nextInt(centers.length)];
            out[i] = vts.createFloatVector(DIM);
            for (int d = 0; d < DIM; d++) {
                float spread = 45f / (float) Math.sqrt(1 + d / 2.0);
                out[i].set(d, Math.max(0f, Math.round(center[d] + spread * (float) random.nextGaussian())));
            }
        }
        return out;
    }

    private static int encodedBits(int projected, int bits) {
        return AsymmetricHashing.HEADER_BITS + projected * bits;
    }

    private static AsymmetricHashing train(VectorFloat<?>[] data, int projected, int bits, int landmarks)
            throws Exception {
        return AsymmetricHashing.initialize(MockVectorValues.fromValues(data), L2, AsymmetricHashing.ITQ,
                encodedBits(projected, bits), landmarks, bits);
    }

    private static float exactSquaredDistance(VectorFloat<?> a, VectorFloat<?> b) {
        double sum = 0;
        for (int d = 0; d < a.length(); d++) {
            double delta = a.get(d) - b.get(d);
            sum += delta * delta;
        }
        return (float) sum;
    }

    private static float squaredDistance(float similarity) {
        return 1f / similarity - 1f;
    }

    @Test
    public void squaredDistancesApproachExactAtFullResolution() throws Exception {
        VectorFloat<?>[] data = siftLike(1500, 7);
        VectorFloat<?>[] queries = siftLike(20, 8);
        var ash = train(data, DIM, 8, 1);
        assertEquals(L2, ash.similarityFunction());
        assertTrue("SIFT-scale vectors need a unit below 1", ash.unit < 1f);
        var encoded = ash.encodeAll(MockVectorValues.fromValues(data), ForkJoinPool.commonPool());
        double relative = 0;
        int n = 0;
        for (var q : queries) {
            var f = encoded.scoreFunctionFor(q, L2);
            assertEquals(L2, f.similarityFunction());
            for (int i = 0; i < data.length; i += 7) {
                float exact = exactSquaredDistance(q, data[i]);
                float estimate = squaredDistance(f.similarityTo(i));
                assertTrue(Float.isFinite(estimate));
                relative += Math.abs(estimate - exact) / exact;
                n++;
            }
        }
        assertTrue("mean relative squared-distance error " + relative / n, relative / n < 0.02);
    }

    @Test
    public void everyScorerGivesTheSameEuclideanScores() throws Exception {
        String single = System.getProperty("jvector.ash.singleKernel");
        String block = System.getProperty("jvector.ash.blockKernel");
        VectorFloat<?>[] data = siftLike(700, 11);
        VectorFloat<?>[] queries = siftLike(4, 12);
        try {
            for (int bits : new int[]{1, 2, 3, 4}) {
                for (int landmarks : new int[]{1, 3}) {
                    var ash = train(data, 32, bits, landmarks);
                    var encoded = ash.encodeAll(MockVectorValues.fromValues(data), ForkJoinPool.commonPool());
                    // Growable codes have one landmark and are encoded one at a time; compare them with codes
                    // encoded the same way (encodeAll's projected batches can differ in the last bit).
                    var mutable = landmarks == 1 ? new MutableASHVectors(ash) : null;
                    if (mutable != null) for (int i = 0; i < data.length; i++) mutable.encodeAndSet(i, data[i]);
                    var singly = new ASHVectors(ash, encodeEach(ash, data));
                    var preparer = encoded.queryPreparer();
                    for (var q : queries) {
                        System.setProperty("jvector.ash.singleKernel", "scalar");
                        System.setProperty("jvector.ash.blockKernel", "scalar");
                        var reference = encoded.scoreFunctionFor(q, L2);
                        float[] expected = new float[data.length];
                        for (int i = 0; i < expected.length; i++) expected[i] = reference.similarityTo(i);
                        for (String mode : new String[]{"scalar", "auto"}) {
                            System.setProperty("jvector.ash.singleKernel", mode);
                            System.setProperty("jvector.ash.blockKernel", mode);
                            String where = "bits=" + bits + " C=" + landmarks + " mode=" + mode;
                            check(where + " node", expected, encoded.scoreFunctionFor(q, L2));
                            check(where + " precomputed", expected, encoded.precomputedScoreFunctionFor(q, L2));
                            var prepared = preparer.newState();
                            preparer.prepare(q, prepared);
                            check(where + " prepared", expected, encoded.scoreFunctionFor(prepared));
                            if (mutable != null) {
                                var onePerVector = singly.scoreFunctionFor(q, L2);
                                float[] expectedSingly = new float[data.length];
                                for (int i = 0; i < data.length; i++) expectedSingly[i] = onePerVector.similarityTo(i);
                                check(where + " mutable", expectedSingly, mutable.scoreFunctionFor(q, L2));
                            }
                            check(where + " block", expected, encoded.blockScorerFor(q));
                            check(where + " block32", expected, encoded.blockScorerFor(q, 32));
                        }
                    }
                }
            }
        } finally {
            restore("jvector.ash.singleKernel", single);
            restore("jvector.ash.blockKernel", block);
        }
    }

    private static AsymmetricHashing.QuantizedVector[] encodeEach(AsymmetricHashing ash, VectorFloat<?>[] data) {
        var codes = new AsymmetricHashing.QuantizedVector[data.length];
        for (int i = 0; i < data.length; i++) codes[i] = ash.encode(data[i]);
        return codes;
    }

    private static void check(String where, float[] expected, ScoreFunction f) {
        assertEquals(where, L2, f.similarityFunction());
        int[] nodes = new int[expected.length];
        for (int i = 0; i < nodes.length; i++) nodes[i] = i;
        float[] batch = new float[expected.length];
        f.similarityTo(nodes, nodes.length, batch);
        for (int i = 0; i < expected.length; i++) {
            float tolerance = 1e-4f * expected[i];
            assertEquals(where + " node " + i, expected[i], f.similarityTo(i), tolerance);
            assertEquals(where + " batch " + i, expected[i], batch[i], tolerance);
            assertTrue(where, expected[i] > 0f && expected[i] <= 1f);
        }
    }

    private static void check(String where, float[] expected, ASHBlockScorer f) {
        float[] out = new float[expected.length];
        f.scoreRange(0, expected.length, out);
        for (int i = 0; i < expected.length; i++) assertEquals(where + " " + i, expected[i], out[i], 1e-4f * expected[i]);
    }

    @Test
    public void symmetricScoresApproachExactDistances() throws Exception {
        VectorFloat<?>[] data = siftLike(1200, 21);
        for (int landmarks : new int[]{1, 3}) {
            var ash = train(data, DIM, 8, landmarks);
            // One-at-a-time encoding, as growable codes use, so both hold the same codes.
            var encoded = new ASHVectors(ash, encodeEach(ash, data));
            var symmetric = new ASHSymmetricScorer(encoded);
            var mutable = landmarks == 1 ? new MutableASHVectors(ash) : null;
            if (mutable != null) for (int i = 0; i < data.length; i++) mutable.encodeAndSet(i, data[i]);
            double relative = 0;
            int n = 0;
            for (int a = 0; a < 40; a++) {
                var f = symmetric.scoreFunctionFor(a);
                assertEquals(L2, f.similarityFunction());
                var diversity = encoded.diversityFunctionFor(a, L2);
                var streaming = mutable == null ? null : mutable.diversityFunctionFor(a, L2);
                for (int b = 100; b < data.length; b += 11) {
                    float score = f.similarityTo(b);
                    assertEquals(score, diversity.similarityTo(b), 1e-6f);
                    // The two scorers can round differently (seen on the Panama provider); allow 0.1%.
                    if (streaming != null) assertEquals(score, streaming.similarityTo(b), 1e-3f * score);
                    float exact = exactSquaredDistance(data[a], data[b]);
                    relative += Math.abs(squaredDistance(score) - exact) / exact;
                    n++;
                }
            }
            assertTrue("C=" + landmarks + " mean relative error " + relative / n, relative / n < 0.03);
        }
    }

    @Test
    public void scalingByAPowerOfTwoChangesOnlyTheUnit() throws Exception {
        VectorFloat<?>[] data = siftLike(900, 31);
        VectorFloat<?>[] scaled = new VectorFloat<?>[data.length];
        for (int i = 0; i < data.length; i++) {
            scaled[i] = data[i].copy();
            for (int d = 0; d < DIM; d++) scaled[i].set(d, data[i].get(d) * 1024f);
        }
        var base = train(data, 32, 4, 1);
        var big = train(scaled, 32, 4, 1);
        assertEquals(base.unit / 1024f, big.unit, 0f);
        for (int i = 0; i < data.length; i += 13) {
            // Codes and half-precision headers are identical: the unit makes them independent of the data's scale.
            assertEquals(base.encode(data[i]), big.encode(scaled[i]));
        }
    }

    @Test
    public void theSimilarityIsFixedByTheModelAndSavedWithIt() throws Exception {
        VectorFloat<?>[] data = siftLike(400, 41);
        var ash = train(data, 32, 2, 1);
        var encoded = ash.encodeAll(MockVectorValues.fromValues(data), ForkJoinPool.commonPool());
        assertEquals(L2, encoded.similarityFunction());
        assertThrows(IllegalArgumentException.class, () -> encoded.scoreFunctionFor(data[0], DOT));
        assertThrows(IllegalArgumentException.class, () -> encoded.diversityFunctionFor(0, DOT));

        var legacy = AsymmetricHashing.initialize(MockVectorValues.fromValues(data), VectorSimilarityFunction.DOT_PRODUCT, AsymmetricHashing.ITQ,
                encodedBits(32, 2), 1, 2);
        assertEquals(DOT, legacy.similarityFunction());
        assertEquals(1f, legacy.unit, 0f);
        assertThrows(IllegalArgumentException.class,
                () -> legacy.encodeAll(MockVectorValues.fromValues(data)).scoreFunctionFor(data[0], L2));
        var mutable = new MutableASHVectors(ash);
        assertThrows(IllegalArgumentException.class, () -> mutable.addModel(legacy));

        int size = ash.compressorSize() + Integer.BYTES
                + data.length * AsymmetricHashing.QuantizedVector.serializedSizeBytes(ash.quantizedDim, ash.bitsPerDimension);
        ASHVectors loaded;
        try (var writer = ByteBufferIndexWriter.create(size, false)) {
            encoded.write(writer);
            var written = writer.getWrittenData();
            assertEquals(size, written.remaining());
            loaded = ASHVectors.load(new ByteBufferReader(written));
        }
        assertEquals(L2, loaded.similarityFunction());
        assertEquals(ash.unit, loaded.getCompressor().unit, 0f);
        var before = encoded.scoreFunctionFor(data[3], L2);
        var after = loaded.scoreFunctionFor(data[3], L2);
        for (int i = 0; i < data.length; i++) assertEquals(before.similarityTo(i), after.similarityTo(i), 0f);
    }

    /**
     * On unit-length vectors EUCLIDEAN and DOT_PRODUCT rank alike, so a EUCLIDEAN model must find neighbors as well
     * as a DOT_PRODUCT model of the same vectors.
     */
    @Test
    public void euclideanRanksUnitVectorsAsWellAsDotProduct() throws Exception {
        VectorFloat<?>[] data = siftLike(3000, 71);
        VectorFloat<?>[] queries = siftLike(40, 72);
        for (var v : data) io.github.jbellis.jvector.vector.VectorUtil.l2normalize(v);
        for (var q : queries) io.github.jbellis.jvector.vector.VectorUtil.l2normalize(q);
        var values = MockVectorValues.fromValues(data);
        for (int bits : new int[]{1, 2, 4}) {
            double[] recall = new double[2];
            VectorSimilarityFunction[] similarities = {L2, DOT};
            for (int s = 0; s < 2; s++) {
                var ash = AsymmetricHashing.initialize(values, similarities[s], AsymmetricHashing.ITQ,
                        encodedBits(32, bits), 1, bits);
                var encoded = ash.encodeAll(values, ForkJoinPool.commonPool());
                int hits = 0;
                for (var q : queries) {
                    var expected = exactTop(data, q, 10);
                    var f = encoded.scoreFunctionFor(q, similarities[s]);
                    Integer[] order = new Integer[data.length];
                    for (int i = 0; i < order.length; i++) order[i] = i;
                    float[] scores = new float[data.length];
                    for (int i = 0; i < scores.length; i++) scores[i] = f.similarityTo(i);
                    Arrays.sort(order, (a, b) -> Float.compare(scores[b], scores[a]));
                    for (int i = 0; i < 30; i++) if (expected.contains(order[i])) hits++;
                }
                recall[s] = hits / (10.0 * queries.length);
            }
            assertEquals("bits=" + bits + " EUCLIDEAN vs DOT_PRODUCT recall " + Arrays.toString(recall),
                    recall[1], recall[0], 0.02);
        }
    }

    @Test
    public void graphSearchFindsEuclideanNeighborsInMemoryAndOnDisk() throws Exception {
        VectorFloat<?>[] data = siftLike(2000, 51);
        VectorFloat<?>[] queries = siftLike(30, 52);
        var values = MockVectorValues.fromValues(data);
        var ash = train(data, 32, 4, 1);
        var encoded = ash.encodeAll(values, ForkJoinPool.commonPool());
        var builder = new GraphIndexBuilder(BuildScoreProvider.ashBuildScoreProvider(encoded), DIM, 16, 64,
                1.2f, 1.2f, false, true);
        var graph = builder.build(values);
        var path = Files.createTempFile("ash-euclidean-", ".index");
        try {
            var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
            try (var view = graph.getView(); var writer = new OnDiskGraphIndexWriter.Builder(graph, path)
                    .with(new FusedASH(graph.maxDegree(), ash, 32)).with(new InlineVectors(DIM)).build()) {
                suppliers.put(FeatureId.FUSED_ASH, n -> new FusedASH.State(view, encoded, n));
                suppliers.put(FeatureId.INLINE_VECTORS, n -> new InlineVectors.State(data[n]));
                writer.write(suppliers);
            }
            try (var reader = new SimpleMappedReader.Supplier(path); var disk = OnDiskGraphIndex.load(reader, 0)) {
                var preparer = disk.queryPreparer().orElseThrow();
                try (var view = disk.getView()) {
                    // The reranker must score by the index's ASH similarity.
                    assertThrows(IllegalArgumentException.class, () -> view.rerankerFor(queries[0], DOT));
                    view.rerankerFor(queries[0], L2);
                }
                for (boolean adaptive : new boolean[]{false, true}) {
                    int memoryHits = 0, diskHits = 0, preparedHits = 0;
                    for (var q : queries) {
                        var expected = exactTop(data, q, 10);
                        try (var searcher = new GraphSearcher(graph)) {
                            searcher.useAdaptiveTermination(adaptive);
                            var scores = new DefaultSearchScoreProvider(encoded.scoreFunctionFor(q, L2),
                                    n -> L2.compare(q, data[n]));
                            for (var ns : searcher.search(scores, 10, 50, 0f, 0f, Bits.ALL).getNodes())
                                if (expected.contains(ns.node)) memoryHits++;
                        }
                        try (var searcher = new GraphSearcher(disk)) {
                            searcher.useAdaptiveTermination(adaptive);
                            var view = (OnDiskGraphIndex.View) searcher.getView();
                            var fused = view.approximateScoreFunctionFor(q, L2);
                            assertEquals(L2, fused.similarityFunction());
                            var standalone = encoded.scoreFunctionFor(q, L2);
                            fused.enableSimilarityToNeighbors(0);
                            var neighbors = view.getNeighborsIterator(0, 0);
                            for (int lane = 0; neighbors.hasNext(); lane++) {
                                float s = standalone.similarityTo(neighbors.nextInt());
                                assertEquals(s, fused.similarityToNeighbor(0, lane), 2e-3f * s);
                            }
                            var scores = new DefaultSearchScoreProvider(fused, view.rerankerFor(q, L2));
                            for (var ns : searcher.search(scores, 10, 50, 0f, 0f, Bits.ALL).getNodes())
                                if (expected.contains(ns.node)) diskHits++;
                        }
                        try (var searcher = new GraphSearcher(disk)) {
                            searcher.useAdaptiveTermination(adaptive);
                            var view = (OnDiskGraphIndex.View) searcher.getView();
                            var prepared = preparer.newState();
                            preparer.prepare(q, prepared);
                            var scores = view.scoreProviderFor(prepared, true);
                            assertEquals(L2, scores.adaptiveSimilarity());
                            assertEquals(L2, scores.scoreFunction().similarityFunction());
                            for (var ns : searcher.search(scores, 10, 50, 0f, 0f, Bits.ALL).getNodes())
                                if (expected.contains(ns.node)) preparedHits++;
                        }
                    }
                    double memoryRecall = memoryHits / (10.0 * queries.length);
                    double diskRecall = diskHits / (10.0 * queries.length);
                    assertTrue("adaptive=" + adaptive + " in-memory recall " + memoryRecall, memoryRecall >= 0.85);
                    assertTrue("adaptive=" + adaptive + " on-disk recall " + diskRecall, diskRecall >= 0.85);
                    assertEquals("adaptive=" + adaptive + " prepared queries", diskHits, preparedHits);
                }
            }
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void streamingIndexTrainsAndSearchesEuclidean() throws Exception {
        VectorFloat<?>[] data = siftLike(3000, 61);
        VectorFloat<?>[] queries = siftLike(20, 62);
        var options = new StreamingASHIndex.Options();
        options.coldStartSize = 256;
        var index = StreamingASHIndex.coldStart(new ListRandomAccessVectorValues(Arrays.asList(data), DIM), L2,
                encodedBits(32, 4), 4, scores -> new GraphIndexBuilder(scores, DIM, 16, 64, 1.2f, 1.2f, false, false),
                options);
        for (int i = 0; i < data.length; i++) index.add(i);
        while (index.isTraining()) Thread.onSpinWait();
        assertTrue(index.hasModel());
        int hits = 0;
        for (var q : queries) {
            var expected = exactTop(data, q, 10);
            for (var ns : index.search(q, 10, 50, Bits.ALL).getNodes()) if (expected.contains(ns.node)) hits++;
        }
        double recall = hits / (10.0 * queries.length);
        // Without an executor the index keeps its first, random-projection model; compare with that model's codes
        // built into a graph in one pass.
        var model = index.model();
        assertEquals(L2, model.similarityFunction());
        var values = MockVectorValues.fromValues(data);
        var encoded = model.encodeAll(values, ForkJoinPool.commonPool());
        var graph = new GraphIndexBuilder(BuildScoreProvider.ashBuildScoreProvider(encoded), DIM, 16, 64,
                1.2f, 1.2f, false, false).build(values);
        int batchHits = 0;
        for (var q : queries) {
            var expected = exactTop(data, q, 10);
            try (var searcher = new GraphSearcher(graph)) {
                var scores = new DefaultSearchScoreProvider(encoded.scoreFunctionFor(q, L2), n -> L2.compare(q, data[n]));
                for (var ns : searcher.search(scores, 10, 50, 0f, 0f, Bits.ALL).getNodes())
                    if (expected.contains(ns.node)) batchHits++;
            }
        }
        double batch = batchHits / (10.0 * queries.length);
        assertTrue("streaming recall " + recall + " vs batch " + batch, recall > batch - 0.05);
    }

    private static HashSet<Integer> exactTop(VectorFloat<?>[] data, VectorFloat<?> q, int k) {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < data.length; i++) order.add(i);
        order.sort((a, b) -> Float.compare(exactSquaredDistance(q, data[a]), exactSquaredDistance(q, data[b])));
        return new HashSet<>(order.subList(0, k));
    }

    private static void restore(String key, String value) {
        if (value == null) System.clearProperty(key);
        else System.setProperty(key, value);
    }
}
