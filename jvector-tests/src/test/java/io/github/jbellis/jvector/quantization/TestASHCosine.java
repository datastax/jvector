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
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
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
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
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
 * COSINE ASH: normalizing vectors at training and encoding, and queries in FP32 before projection, then scoring by
 * the DOT_PRODUCT path, so COSINE on vectors of any length is DOT_PRODUCT on the same vectors normalized.
 */
public class TestASHCosine {
    private static final VectorSimilarityFunction COS = VectorSimilarityFunction.COSINE;
    private static final VectorSimilarityFunction DOT = VectorSimilarityFunction.DOT_PRODUCT;
    private static final int DIM = 64;

    /**
     * Embedding-like vectors whose spread decays across components, with lengths from 0.5 to 20 so that dot
     * products and cosines rank them differently.
     */
    static VectorFloat<?>[] varyingLengths(int count, long seed) {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        Random random = new Random(seed);
        float[] offset = new float[DIM];
        for (int d = 0; d < DIM; d++) offset[d] = 0.3f * (float) random.nextGaussian();
        VectorFloat<?>[] out = new VectorFloat<?>[count];
        for (int i = 0; i < count; i++) {
            out[i] = vts.createFloatVector(DIM);
            float length = 0.5f + 19.5f * random.nextFloat();
            for (int d = 0; d < DIM; d++) {
                out[i].set(d, length * (offset[d] + (float) random.nextGaussian() / (float) Math.sqrt(1 + d / 2.0)));
            }
        }
        return out;
    }

    private static VectorFloat<?>[] scaled(VectorFloat<?>[] vectors, float k) {
        VectorFloat<?>[] out = new VectorFloat<?>[vectors.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = vectors[i].copy();
            VectorUtil.scale(out[i], k);
        }
        return out;
    }

    private static VectorFloat<?>[] normalized(VectorFloat<?>[] vectors) {
        VectorFloat<?>[] out = new VectorFloat<?>[vectors.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = vectors[i].copy();
            VectorUtil.l2normalize(out[i]);
        }
        return out;
    }

    private static int encodedBits(int projected, int bits) {
        return AsymmetricHashing.HEADER_BITS + projected * bits;
    }

    @Test
    public void cosineOnRawVectorsIsDotProductOnNormalizedVectors() throws Exception {
        VectorFloat<?>[] raw = varyingLengths(1200, 3);
        VectorFloat<?>[] unit = normalized(raw);
        VectorFloat<?>[] queries = varyingLengths(6, 4);
        VectorFloat<?>[] unitQueries = normalized(queries);
        for (int bits : new int[]{1, 2, 3, 4}) {
            for (int landmarks : new int[]{1, 3}) {
                var cosine = AsymmetricHashing.initialize(MockVectorValues.fromValues(raw), COS, AsymmetricHashing.ITQ,
                        encodedBits(32, bits), landmarks, bits);
                var dot = AsymmetricHashing.initialize(MockVectorValues.fromValues(unit), DOT, AsymmetricHashing.ITQ,
                        encodedBits(32, bits), landmarks, bits);
                assertEquals(COS, cosine.similarityFunction());
                String where = "bits=" + bits + " C=" + landmarks;
                if (landmarks > 1) {
                    // k-means seeds itself randomly, so two models with several landmarks differ; check instead that
                    // one COSINE model ignores vector length. Power-of-two multiples normalize to exactly the same
                    // unit vector, so their codes and scores must match exactly. (Pre-normalized copies would not:
                    // normalizing again can change a last bit, which at a quantization boundary flips a code.)
                    var rawCodes = cosine.encodeAll(MockVectorValues.fromValues(raw), ForkJoinPool.commonPool());
                    for (float k : new float[]{2f, 0.5f}) {
                        var scaledCodes = cosine.encodeAll(MockVectorValues.fromValues(scaled(raw, k)),
                                ForkJoinPool.commonPool());
                        for (int q = 0; q < queries.length; q++) {
                            var a = rawCodes.scoreFunctionFor(queries[q], COS);
                            var b = scaledCodes.scoreFunctionFor(queries[q], COS);
                            for (int i = 0; i < raw.length; i++)
                                assertEquals(where + " x" + k, a.similarityTo(i), b.similarityTo(i), 0f);
                        }
                    }
                    continue;
                }
                for (int c = 0; c < landmarks; c++) {
                    for (int d = 0; d < DIM; d++) {
                        assertEquals(where, Float.floatToRawIntBits(dot.landmarks[c].get(d)),
                                Float.floatToRawIntBits(cosine.landmarks[c].get(d)));
                    }
                }
                for (int r = 0; r < dot.stiefelTransform.AData.length; r++)
                    assertArrayEquals(where, dot.stiefelTransform.AData[r], cosine.stiefelTransform.AData[r], 0.0);
                var cosineCodes = cosine.encodeAll(MockVectorValues.fromValues(raw), ForkJoinPool.commonPool());
                var dotCodes = dot.encodeAll(MockVectorValues.fromValues(unit), ForkJoinPool.commonPool());
                for (int i = 0; i < raw.length; i++) {
                    assertEquals(where + " code " + i, dotCodes.get(i), cosineCodes.get(i));
                    assertEquals(where + " single code " + i, dot.encode(unit[i]), cosine.encode(raw[i]));
                }
                var cosinePreparer = cosineCodes.queryPreparer();
                var dotPreparer = dotCodes.queryPreparer();
                for (int q = 0; q < queries.length; q++) {
                    var a = cosineCodes.scoreFunctionFor(queries[q], COS);
                    var b = dotCodes.scoreFunctionFor(unitQueries[q], DOT);
                    var blockA = new float[raw.length];
                    var blockB = new float[raw.length];
                    cosineCodes.blockScorerFor(queries[q], 32).scoreRange(0, raw.length, blockA);
                    dotCodes.blockScorerFor(unitQueries[q], 32).scoreRange(0, raw.length, blockB);
                    var preparedA = cosinePreparer.newState();
                    cosinePreparer.prepare(queries[q], preparedA);
                    var preparedB = dotPreparer.newState();
                    dotPreparer.prepare(unitQueries[q], preparedB);
                    // The prepared query keeps the caller's vector for reranking.
                    assertEquals(queries[q], preparedA.queryVector());
                    var fromPreparedA = cosineCodes.scoreFunctionFor(preparedA);
                    var fromPreparedB = dotCodes.scoreFunctionFor(preparedB);
                    for (int i = 0; i < raw.length; i++) {
                        assertEquals(where, b.similarityTo(i), a.similarityTo(i), 0f);
                        assertEquals(where, blockB[i], blockA[i], 0f);
                        assertEquals(where, fromPreparedB.similarityTo(i), fromPreparedA.similarityTo(i), 0f);
                    }
                }
            }
        }
    }

    @Test
    public void zeroVectorsAreRejected() throws Exception {
        VectorFloat<?>[] raw = varyingLengths(300, 5);
        var zero = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(DIM);
        var model = AsymmetricHashing.initialize(MockVectorValues.fromValues(raw), COS, AsymmetricHashing.ITQ,
                encodedBits(32, 2), 1, 2);
        var codes = (ASHVectors) model.encodeAll(MockVectorValues.fromValues(raw));
        assertThrows(IllegalArgumentException.class, () -> model.encode(zero));
        assertThrows(IllegalArgumentException.class, () -> codes.scoreFunctionFor(zero, COS));
        assertThrows(IllegalArgumentException.class, () -> codes.blockScorerFor(zero, 32));
        var preparer = codes.queryPreparer();
        assertThrows(IllegalArgumentException.class, () -> preparer.prepare(zero, preparer.newState()));
        var mutable = new MutableASHVectors(model);
        assertThrows(IllegalArgumentException.class, () -> mutable.encodeAndSet(0, zero));
        assertThrows(IllegalArgumentException.class, () -> mutable.scoreFunctionFor(zero, COS));

        VectorFloat<?>[] withZero = raw.clone();
        withZero[7] = zero;
        assertThrows(IllegalArgumentException.class, () -> model.encodeAll(MockVectorValues.fromValues(withZero)));
        assertThrows(IllegalArgumentException.class, () -> AsymmetricHashing.initialize(
                MockVectorValues.fromValues(withZero), COS, AsymmetricHashing.ITQ, encodedBits(32, 2), 1, 2));

        // DOT_PRODUCT and EUCLIDEAN keep encoding zero vectors.
        var dot = AsymmetricHashing.initialize(MockVectorValues.fromValues(withZero), DOT, AsymmetricHashing.ITQ,
                encodedBits(32, 2), 1, 2);
        dot.encode(zero);

        List<VectorFloat<?>> streamed = new ArrayList<>(Arrays.asList(raw));
        streamed.add(zero);
        var options = new StreamingASHIndex.Options();
        options.coldStartSize = 256;
        var index = StreamingASHIndex.coldStart(new ListRandomAccessVectorValues(streamed, DIM), COS,
                encodedBits(32, 2), 2, scores -> new GraphIndexBuilder(scores, DIM, 16, 64, 1.2f, 1.2f, false, false),
                options);
        index.add(0);
        assertThrows(IllegalArgumentException.class, () -> index.add(raw.length));
        assertThrows(IllegalArgumentException.class, () -> index.search(zero, 5, 10, Bits.ALL));
        for (int i = 1; i < raw.length; i++) index.add(i);
        while (index.isTraining()) Thread.onSpinWait();
        assertTrue(index.hasModel());
        assertThrows(IllegalArgumentException.class, () -> index.add(raw.length));
        assertThrows(IllegalArgumentException.class, () -> index.search(zero, 5, 10, Bits.ALL));
    }

    @Test
    public void theSimilarityIsSavedWithTheModel() throws Exception {
        VectorFloat<?>[] raw = varyingLengths(400, 6);
        var model = AsymmetricHashing.initialize(MockVectorValues.fromValues(raw), COS, AsymmetricHashing.ITQ,
                encodedBits(32, 4), 1, 4);
        var codes = (ASHVectors) model.encodeAll(MockVectorValues.fromValues(raw));
        assertThrows(IllegalArgumentException.class, () -> codes.scoreFunctionFor(raw[0], DOT));
        int size = model.compressorSize() + Integer.BYTES
                + raw.length * AsymmetricHashing.QuantizedVector.serializedSizeBytes(model.quantizedDim, model.bitsPerDimension);
        ASHVectors loaded;
        try (var writer = ByteBufferIndexWriter.create(size, false)) {
            codes.write(writer);
            loaded = ASHVectors.load(new ByteBufferReader(writer.getWrittenData()));
        }
        assertEquals(COS, loaded.similarityFunction());
        var before = codes.scoreFunctionFor(raw[1], COS);
        var after = loaded.scoreFunctionFor(raw[1], COS);
        for (int i = 0; i < raw.length; i++) assertEquals(before.similarityTo(i), after.similarityTo(i), 0f);
    }

    /** Recall of a DOT_PRODUCT index of the normalized vectors, which a COSINE index of the raw ones must match. */
    private static double dotRecallOnNormalized(VectorFloat<?>[] raw, VectorFloat<?>[] queries) throws Exception {
        VectorFloat<?>[] unit = normalized(raw);
        var values = MockVectorValues.fromValues(unit);
        var model = AsymmetricHashing.initialize(values, DOT, AsymmetricHashing.ITQ, encodedBits(32, 4), 1, 4);
        var codes = model.encodeAll(values, ForkJoinPool.commonPool());
        var graph = buildOnOneThread(codes, values);
        int hits = 0;
        for (var q : normalized(queries)) {
            var expected = exactTop(unit, q, 10);
            try (var searcher = new GraphSearcher(graph)) {
                var scores = new DefaultSearchScoreProvider(codes.scoreFunctionFor(q, DOT), n -> DOT.compare(q, unit[n]));
                for (var ns : searcher.search(scores, 10, 50, 0f, 0f, Bits.ALL).getNodes())
                    if (expected.contains(ns.node)) hits++;
            }
        }
        return hits / (10.0 * queries.length);
    }

    @Test
    public void graphSearchFindsCosineNeighborsOfVectorsOfAnyLength() throws Exception {
        VectorFloat<?>[] raw = varyingLengths(2000, 7);
        VectorFloat<?>[] queries = varyingLengths(30, 8);
        double reference = dotRecallOnNormalized(raw, queries);
        var values = MockVectorValues.fromValues(raw);
        var model = AsymmetricHashing.initialize(values, COS, AsymmetricHashing.ITQ, encodedBits(32, 4), 1, 4);
        var codes = model.encodeAll(values, ForkJoinPool.commonPool());
        var graph = buildOnOneThread(codes, values);
        var path = Files.createTempFile("ash-cosine-", ".index");
        try {
            var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
            try (var view = graph.getView(); var writer = new OnDiskGraphIndexWriter.Builder(graph, path)
                    .with(new FusedASH(graph.maxDegree(), model, 32)).with(new InlineVectors(DIM)).build()) {
                suppliers.put(FeatureId.FUSED_ASH, n -> new FusedASH.State(view, codes, n));
                suppliers.put(FeatureId.INLINE_VECTORS, n -> new InlineVectors.State(raw[n]));
                writer.write(suppliers);
            }
            try (var reader = new SimpleMappedReader.Supplier(path); var disk = OnDiskGraphIndex.load(reader, 0)) {
                int memoryHits = 0, diskHits = 0;
                for (var q : queries) {
                    var expected = exactTop(raw, q, 10);
                    try (var searcher = new GraphSearcher(graph)) {
                        var scores = new DefaultSearchScoreProvider(codes.scoreFunctionFor(q, COS),
                                n -> COS.compare(q, raw[n]));
                        var result = searcher.search(scores, 10, 50, 0f, 0f, Bits.ALL).getNodes();
                        for (var ns : result) {
                            // Reranking with the stored vectors gives exact cosine similarity.
                            assertEquals(COS.compare(q, raw[ns.node]), ns.score, 0f);
                            if (expected.contains(ns.node)) memoryHits++;
                        }
                    }
                    try (var searcher = new GraphSearcher(disk)) {
                        var view = (OnDiskGraphIndex.View) searcher.getView();
                        var fused = view.approximateScoreFunctionFor(q, COS);
                        var standalone = codes.scoreFunctionFor(q, COS);
                        fused.enableSimilarityToNeighbors(0);
                        var neighbors = view.getNeighborsIterator(0, 0);
                        for (int lane = 0; neighbors.hasNext(); lane++)
                            assertEquals(standalone.similarityTo(neighbors.nextInt()), fused.similarityToNeighbor(0, lane), 2e-3f);
                        var scores = new DefaultSearchScoreProvider(fused, view.rerankerFor(q, COS));
                        for (var ns : searcher.search(scores, 10, 50, 0f, 0f, Bits.ALL).getNodes())
                            if (expected.contains(ns.node)) diskHits++;
                    }
                }
                double memoryRecall = memoryHits / (10.0 * queries.length);
                double diskRecall = diskHits / (10.0 * queries.length);
                // Both graphs are built on one thread from identical codes, so they match; only the exact
                // rerank scores (cosine of raw vectors vs dot product of normalized ones) can differ in a last bit.
                assertEquals("in-memory recall vs DOT_PRODUCT on normalized vectors", reference, memoryRecall, 0.03);
                assertEquals("on-disk recall vs DOT_PRODUCT on normalized vectors", reference, diskRecall, 0.03);
            }
        } finally {
            Files.deleteIfExists(path);
        }
    }

    /** Builds on one thread so that graphs built from identical codes are identical. */
    private static ImmutableGraphIndex buildOnOneThread(ASHVectors codes, MockVectorValues values) throws Exception {
        var pool = new ForkJoinPool(1);
        try {
            return new GraphIndexBuilder(BuildScoreProvider.ashBuildScoreProvider(codes), DIM, 16, 64,
                    1.2f, 1.2f, false, true, pool, pool).build(values);
        } finally {
            pool.shutdown();
        }
    }

    private static HashSet<Integer> exactTop(VectorFloat<?>[] data, VectorFloat<?> q, int k) {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < data.length; i++) order.add(i);
        order.sort((a, b) -> Float.compare(COS.compare(q, data[b]), COS.compare(q, data[a])));
        return new HashSet<>(order.subList(0, k));
    }
}
