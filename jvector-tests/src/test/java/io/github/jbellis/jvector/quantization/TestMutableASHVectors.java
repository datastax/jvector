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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.similarity.PreparedQuery;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.*;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestMutableASHVectors extends RandomizedTest {
    private static final int DIM = 64;
    private static final int PROJECTED = 32;
    /** Small pages, so a few thousand vectors span many pages and page-table growth. */
    private static final int SMALL_PAGE_BYTES = 256;
    private static final VectorSimilarityFunction DOT = VectorSimilarityFunction.DOT_PRODUCT;

    private static List<VectorFloat<?>> vectors(int n, long seed) {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var random = new Random(seed);
        var out = new ArrayList<VectorFloat<?>>(n);
        for (int i = 0; i < n; i++) {
            float[] v = new float[DIM];
            for (int d = 0; d < DIM; d++) v[d] = (float) random.nextGaussian() + 0.3f;
            out.add(vts.createFloatVector(v));
        }
        return out;
    }

    private static AsymmetricHashing model(List<VectorFloat<?>> data, int optimizer, int bits) throws Exception {
        return AsymmetricHashing.initialize(new ListRandomAccessVectorValues(data, DIM), VectorSimilarityFunction.DOT_PRODUCT, optimizer,
                AsymmetricHashing.HEADER_BITS + PROJECTED * bits, 1, bits);
    }

    /** The model's codes for every vector, encoded one at a time, in a fixed collection. */
    /** A store of {@code data} encoded with {@code a}, every page then rebuilt with {@code b} (and back to {@code a} unless {@code toB}), fixed. */
    private static ASHVectors rebuiltTo(AsymmetricHashing a, AsymmetricHashing b, boolean toB,
                                        List<VectorFloat<?>> data, ListRandomAccessVectorValues ravv) {
        var store = new MutableASHVectors(a, SMALL_PAGE_BYTES);
        for (int i = 0; i < data.size(); i++) store.encodeAndSet(i, data.get(i));
        int idB = store.addModel(b);
        for (int p = 0; p < store.pageCount(); p++) assertTrue(store.reencodePage(p, idB, ravv));
        if (!toB) for (int p = 0; p < store.pageCount(); p++) assertTrue(store.reencodePage(p, 0, ravv));
        return store.snapshot();
    }

    private static ASHVectors fixed(AsymmetricHashing ash, List<VectorFloat<?>> data) {
        var codes = new AsymmetricHashing.QuantizedVector[data.size()];
        for (int i = 0; i < codes.length; i++) codes[i] = ash.encode(data.get(i));
        return new ASHVectors(ash, codes);
    }

    /** Component {@code j} of {@code code}, in quantization steps of one. */
    private static float component(AsymmetricHashing.QuantizedVector code, int j, int bits) {
        if (bits == 1) return (code.binaryVector[j >>> 6] >>> (j & 63) & 1L) == 0 ? -0.5f : 0.5f;
        if (AsymmetricHashing.usesFastScanProjectionCode(bits)) return AsymmetricHashing.projectionComponent(code.extraBits, 0, j, bits);
        return AsymmetricHashing.decodeOffsetBinary(AsymmetricHashing.offsetBinaryField(code.extraBits, 0, j, bits), bits);
    }

    /** Components a 1 ULP change may move, those at a rounding boundary. JDK 25 runs showed only header changes. */
    private static final int MAX_DIFFERING_COMPONENTS = 1;

    /** One unit in the last place of a half-precision header value. */
    private static float halfUlp(float value) {
        return Math.max(Math.ulp(value) * (1 << 13), 0x1p-24f);
    }

    /**
     * {@code actual} is {@code expected} up to a 1 ULP change in the encoder's FP32 sums: each component within one
     * quantization step, each half-precision header within 1 ULP. On JDK 25, C2 adds SIMD lanes in another order than
     * the interpreter and C1, so the same vector may round otherwise once the encoder is compiled.
     */
    private static void assertSameCodeWithinOneUlp(String where, AsymmetricHashing.QuantizedVector expected,
                                                   AsymmetricHashing.QuantizedVector actual, int bits) {
        assertEquals(where, expected.landmark, actual.landmark);
        float ulp = Math.max(halfUlp(expected.scale), halfUlp(actual.scale));
        assertEquals(where + " scale", expected.scale, actual.scale, ulp);
        ulp = Math.max(halfUlp(expected.offset), halfUlp(actual.offset));
        assertEquals(where + " offset", expected.offset, actual.offset, ulp);
        int differing = 0;
        for (int j = 0; j < PROJECTED; j++) {
            float e = component(expected, j, bits), a = component(actual, j, bits);
            assertEquals(where + " dim " + j, e, a, 1f);
            if (e != a) differing++;
        }
        assertTrue(where + ": " + differing + " components differ", differing <= MAX_DIFFERING_COMPONENTS);
    }

    /** Waits until {@code checks} grows, or a reader fails; fails if the readers make no progress for a minute. */
    private static void awaitMoreChecks(AtomicInteger checks, AtomicReference<Throwable> failure) throws InterruptedException {
        int start = checks.get();
        long deadline = System.nanoTime() + TimeUnit.MINUTES.toNanos(1);
        while (checks.get() == start && failure.get() == null) {
            if (System.nanoTime() > deadline) throw new AssertionError("readers made no progress");
            Thread.sleep(1);
        }
    }

    private static void addAll(MutableASHVectors growing, List<VectorFloat<?>> data, int threads) throws Exception {
        var failure = new AtomicReference<Throwable>();
        var list = new ArrayList<Thread>();
        for (int w = 0; w < threads; w++) {
            final int first = w;
            var t = new Thread(() -> {
                try {
                    for (int i = first; i < data.size(); i += threads) growing.encodeAndSet(i, data.get(i));
                } catch (Throwable e) { failure.compareAndSet(null, e); }
            });
            list.add(t);
            t.start();
        }
        for (var t : list) t.join();
        if (failure.get() != null) throw new AssertionError(failure.get());
    }

    @Test
    public void concurrentlyAddedVectorsScoreBitForBitAsAFixedCollection() throws Exception {
        var data = vectors(3000, 7);
        var query = vectors(1, 99).get(0);
        for (int bits : new int[]{1, 2, 4}) {
            var ash = model(data, AsymmetricHashing.ITQ, bits);
            var encoded = fixed(ash, data);
            var growing = new MutableASHVectors(ash, SMALL_PAGE_BYTES);
            long emptyBytes = growing.ramBytesUsed();
            addAll(growing, data, 8);
            assertEquals(data.size(), growing.count());
            assertTrue(growing.pageCount() > 10);
            assertTrue(growing.ramBytesUsed() > emptyBytes);
            assertArrayEquals(new int[]{0}, growing.modelsInUse());
            // Every vector's code landed at its own ordinal, as encoding it alone gives it.
            var stored = new AsymmetricHashing.QuantizedVector[data.size()];
            for (int i = 0; i < data.size(); i++) {
                stored[i] = growing.get(i);
                assertSameCodeWithinOneUlp("bits=" + bits + " node " + i, encoded.get(i), stored[i], bits);
            }
            // The same codes in a fixed collection score bit for bit as the growing store does.
            var reference = new ASHVectors(ash, stored);

            var refScore = reference.scoreFunctionFor(query, DOT);
            var growScore = growing.scoreFunctionFor(query, DOT);
            assertEquals(refScore.dotProductBound(), growScore.dotProductBound(), 0f);
            var refSym = new ASHSymmetricScorer(reference);
            for (int i = 0; i < data.size(); i++)
                assertEquals("bits=" + bits + " node " + i, refScore.similarityTo(i), growScore.similarityTo(i), 0f);
            var provider = growing.buildScoreProvider(new ListRandomAccessVectorValues(data, DIM));
            for (int a = 0; a < data.size(); a += 97) {
                var refFn = refSym.scoreFunctionFor(a);
                var growFn = provider.searchProviderFor(a).scoreFunction();
                for (int b = 0; b < data.size(); b += 13) assertEquals(refFn.similarityTo(b), growFn.similarityTo(b), 0f);
                assertEquals(refFn.supportsBatchScoring(), growFn.supportsBatchScoring());
                if (growFn.supportsBatchScoring()) {
                    int[] nodes = {0, 5, 1999, 2999, 17};
                    float[] expected = new float[nodes.length], actual = new float[nodes.length];
                    refFn.similarityTo(nodes, nodes.length, expected);
                    growFn.similarityTo(nodes, nodes.length, actual);
                    assertArrayEquals(expected, actual, 0f);
                }
            }
            // A snapshot is an ordinary fixed collection equal to the reference; a remapped one reorders.
            assertEquals(reference, growing.snapshot());
            var reordered = growing.snapshot(new int[]{2999, 0, 1500});
            assertEquals(reference.get(2999), reordered.get(0));
            assertEquals(reference.get(1500), reordered.get(2));
        }
    }

    @Test
    public void aNewModelTakesEffectAtTheNextPageAndPagesMoveToItWhole() throws Exception {
        var data = vectors(2000, 3);
        var oldModel = model(data, AsymmetricHashing.RANDOM, 2);
        var newModel = model(data, AsymmetricHashing.ITQ, 2);
        var oldRef = fixed(oldModel, data);
        var newRef = fixed(newModel, data);
        var growing = new MutableASHVectors(oldModel, SMALL_PAGE_BYTES);
        int perPage = growing.vectorsPerPage();
        int half = 1000 + perPage / 2; // mid-page
        for (int i = 0; i < half; i++) growing.encodeAndSet(i, data.get(i));
        int id = growing.addModel(newModel);
        growing.useModel(id);
        for (int i = half; i < data.size(); i++) growing.encodeAndSet(i, data.get(i));

        // The page that was open at the switch keeps the old model; later pages use the new one.
        int switchPage = (half - 1) / perPage;
        for (int i = 0; i < data.size(); i++) {
            int expected = i / perPage <= switchPage ? 0 : id;
            assertEquals("ordinal " + i, expected, growing.modelOf(i));
            assertEquals(expected == 0 ? oldRef.get(i) : newRef.get(i), growing.get(i));
        }
        assertArrayEquals(new int[]{0, id}, growing.modelsInUse());
        assertThrows(IllegalStateException.class, growing::snapshot);

        // During the change every node scores with its own model, exactly as that model's fixed collection does.
        var query = vectors(1, 11).get(0);
        var fn = growing.scoreFunctionFor(query, DOT);
        var oldScore = oldRef.scoreFunctionFor(query, DOT);
        var newScore = newRef.scoreFunctionFor(query, DOT);
        for (int i = 0; i < data.size(); i++) {
            var expected = growing.modelOf(i) == 0 ? oldScore : newScore;
            assertEquals(expected.similarityTo(i), fn.similarityTo(i), 0f);
        }
        assertEquals(Math.max(oldScore.dotProductBound(), newScore.dotProductBound()), fn.dotProductBound(), 0f);

        // Across models, a symmetric score is the source vector scored against the target's code.
        var provider = growing.buildScoreProvider(new ListRandomAccessVectorValues(data, DIM));
        int oldNode = 0, newNode = data.size() - 1;
        var fromOld = provider.searchProviderFor(oldNode).scoreFunction();
        assertEquals(newRef.scoreFunctionFor(data.get(oldNode), DOT).similarityTo(newNode), fromOld.similarityTo(newNode), 0f);
        var fromNew = provider.searchProviderFor(newNode).scoreFunction();
        assertEquals(oldRef.scoreFunctionFor(data.get(newNode), DOT).similarityTo(oldNode), fromNew.similarityTo(oldNode), 0f);

        // Move every old page, rebuilding one in steps; the result is the new model's collection.
        var ravv = new ListRandomAccessVectorValues(data, DIM);
        var stepwise = growing.rebuild(0, id);
        assertNotNull(stepwise);
        while (!stepwise.isFull()) assertTrue(stepwise.encode(7, ravv) > 0);
        assertTrue(stepwise.publish());
        assertNull("already on the new model", growing.rebuild(0, id));
        for (int p = 1; p <= switchPage; p++) assertTrue(growing.reencodePage(p, id, ravv));
        assertArrayEquals(new int[]{id}, growing.modelsInUse());
        assertEquals(newRef, growing.snapshot());
        assertEquals(0, growing.reencodedMissing());
    }

    @Test
    public void readersNeverSeeAPartlyReplacedPage() throws Exception {
        var data = vectors(4000, 5);
        var modelA = model(data, AsymmetricHashing.RANDOM, 4);
        var modelB = model(data, AsymmetricHashing.ITQ, 4);
        var ravv = new ListRandomAccessVectorValues(data, DIM);
        var query = vectors(1, 21).get(0);
        // A page holds the codes it was first given, or a rebuild's, which encodes in batches: references for both.
        var scoreA = fixed(modelA, data).scoreFunctionFor(query, DOT);
        var scoreRebuiltA = rebuiltTo(modelA, modelB, false, data, ravv).scoreFunctionFor(query, DOT);
        var scoreB = rebuiltTo(modelA, modelB, true, data, ravv).scoreFunctionFor(query, DOT);
        var growing = new MutableASHVectors(modelA, SMALL_PAGE_BYTES);
        for (int i = 0; i < data.size(); i++) growing.encodeAndSet(i, data.get(i));
        int idB = growing.addModel(modelB);
        var stop = new AtomicBoolean();
        var failure = new AtomicReference<Throwable>();
        var checks = new AtomicInteger();
        var readers = new ArrayList<Thread>();
        for (int r = 0; r < 4; r++) {
            var t = new Thread(() -> {
                try {
                    var random = new Random();
                    while (!stop.get()) {
                        var fn = growing.scoreFunctionFor(query, DOT);
                        for (int k = 0; k < 256; k++) {
                            int node = random.nextInt(data.size());
                            float s = fn.similarityTo(node);
                            // Either model's score, exactly; a mixed page would give neither.
                            if (s != scoreA.similarityTo(node) && s != scoreRebuiltA.similarityTo(node)
                                    && s != scoreB.similarityTo(node))
                                throw new AssertionError("node " + node + " scored " + s);
                            checks.incrementAndGet();
                        }
                    }
                } catch (Throwable e) { failure.compareAndSet(null, e); }
            });
            readers.add(t);
            t.start();
        }
        // Swap every page to B, back to A, and to B again while the readers run. Each round starts only once the
        // readers have checked scores since the last, so every round overlaps reads however the threads are scheduled.
        for (int round = 0; round < 3; round++) {
            awaitMoreChecks(checks, failure);
            int to = round % 2 == 0 ? idB : 0;
            for (int p = 0; p < growing.pageCount(); p++) assertTrue(growing.reencodePage(p, to, ravv));
        }
        stop.set(true);
        for (var t : readers) t.join();
        if (failure.get() != null) throw new AssertionError(failure.get());
        assertTrue(checks.get() > 0);
        assertArrayEquals(new int[]{idB}, growing.modelsInUse());
    }

    @Test
    public void preparedQueriesScoreAsRawQueriesInBothStores() throws Exception {
        var data = vectors(2000, 31);
        var queries = vectors(5, 32).toArray(new VectorFloat<?>[0]);
        for (int bits : new int[]{1, 2, 4}) {
            var ash = model(data, AsymmetricHashing.ITQ, bits);
            var fixed = fixed(ash, data);
            var growing = new MutableASHVectors(ash, SMALL_PAGE_BYTES);
            for (int i = 0; i < data.size(); i++) growing.encodeAndSet(i, data.get(i));
            var fixedPreparer = fixed.queryPreparer();
            var growingPreparer = growing.queryPreparer();
            for (var q : queries) {
                // One query is projected as a raw query is, so its scores match exactly.
                var fixedState = fixedPreparer.newState();
                var growingState = growingPreparer.newState();
                fixedPreparer.prepare(q, fixedState);
                growingPreparer.prepare(q, growingState);
                var raw = fixed.scoreFunctionFor(q, DOT);
                var fromFixed = fixed.scoreFunctionFor(fixedState);
                var fromGrowing = growing.scoreFunctionFor(growingState);
                assertEquals(raw.dotProductBound(), fromFixed.dotProductBound(), 0f);
                for (int node = 0; node < data.size(); node += 7) {
                    assertEquals("bits " + bits, raw.similarityTo(node), fromFixed.similarityTo(node), 0f);
                    assertEquals("bits " + bits, raw.similarityTo(node), fromGrowing.similarityTo(node), 0f);
                }
            }
            // A batch is projected in one matrix multiply, which may round differently.
            var states = new PreparedQuery[queries.length];
            for (int i = 0; i < states.length; i++) states[i] = growingPreparer.newState();
            growingPreparer.prepareBatch(queries, states, queries.length);
            for (int i = 0; i < queries.length; i++) {
                var raw = fixed.scoreFunctionFor(queries[i], DOT);
                var batched = growing.scoreFunctionFor(states[i]);
                for (int node = 0; node < data.size(); node += 7) {
                    float expected = raw.similarityTo(node);
                    assertEquals("bits " + bits, expected, batched.similarityTo(node), 1e-3f * Math.max(1f, Math.abs(expected)));
                }
            }
        }
    }

    @Test
    public void aQueryPreparedBeforeAModelChangeScoresEveryPageWithItsOwnModel() throws Exception {
        var data = vectors(3000, 41);
        var modelA = model(data, AsymmetricHashing.RANDOM, 2);
        var modelB = model(data, AsymmetricHashing.ITQ, 2);
        var growing = new MutableASHVectors(modelA, SMALL_PAGE_BYTES);
        for (int i = 0; i < data.size(); i++) growing.encodeAndSet(i, data.get(i));
        var preparer = growing.queryPreparer(); // for model A
        int idB = growing.addModel(modelB);
        var ravv = new ListRandomAccessVectorValues(data, DIM);
        for (int p = 0; p < growing.pageCount(); p += 2) assertTrue(growing.reencodePage(p, idB, ravv));
        var q = vectors(1, 42).get(0);
        var state = preparer.newState();
        preparer.prepare(q, state);
        var raw = growing.scoreFunctionFor(q, DOT);
        var prepared = growing.scoreFunctionFor(state);
        assertEquals(raw.dotProductBound(), prepared.dotProductBound(), 0f);
        for (int node = 0; node < data.size(); node++) assertEquals(raw.similarityTo(node), prepared.similarityTo(node), 0f);
    }

    @Test
    public void missingVectorsGetTheZeroCodeAndTheLastPageMovesOnlyWhenQuiesced() throws Exception {
        var data = vectors(300, 13);
        var oldModel = model(data, AsymmetricHashing.RANDOM, 2);
        var newModel = model(data, AsymmetricHashing.ITQ, 2);
        var growing = new MutableASHVectors(oldModel, SMALL_PAGE_BYTES);
        int perPage = growing.vectorsPerPage();
        int n = 3 * perPage + perPage / 2; // the last page is incomplete
        assertTrue(n <= data.size());
        for (int i = 0; i < n; i++) {
            if (i == 5) growing.setZero(i);
            else growing.encodeAndSet(i, data.get(i));
        }
        var zero = AsymmetricHashing.QuantizedVector.createEmpty(oldModel.quantizedDim, oldModel.bitsPerDimension);
        assertEquals(zero, growing.get(5));
        int id = growing.addModel(newModel);
        // Ordinal 7 was deleted and its vector is gone: the caller returns null for it.
        RandomAccessVectorValues withHole = new ListRandomAccessVectorValues(data, DIM) {
            @Override
            public VectorFloat<?> getVector(int node) {
                return node == 7 ? null : super.getVector(node);
            }
        };
        int last = growing.pageCount() - 1;
        assertFalse(growing.isPageComplete(last));
        assertNull("an incomplete page is not replaced while vectors may still be added", growing.rebuild(last, id));
        for (int p = 0; p < last; p++) assertTrue(growing.reencodePage(p, id, withHole));
        var rebuild = growing.rebuildQuiesced(last, id);
        rebuild.encode(perPage, withHole);
        assertTrue(rebuild.publish());
        assertArrayEquals(new int[]{id}, growing.modelsInUse());
        assertEquals(1, growing.reencodedMissing());
        assertEquals(zero, growing.get(7));
        // A missing vector scores at the landmark, a finite score, so searches order it like any other node.
        var query = data.get(9);
        float missing = growing.scoreFunctionFor(query, DOT).similarityTo(7);
        assertTrue("missing vector scored " + missing, Float.isFinite(missing));
        assertEquals(DOT.compare(query, newModel.landmarks[0]), missing, 1e-4f);
        assertEquals(newModel.encode(data.get(8)), growing.get(8));
        assertEquals(n, growing.count());
    }

    @Test
    public void aRemappedSnapshotGivesHolesTheZeroCode() throws Exception {
        var data = vectors(300, 31);
        var ash = model(data, AsymmetricHashing.ITQ, 2);
        var store = new MutableASHVectors(ash, SMALL_PAGE_BYTES);
        for (int i = 0; i < data.size(); i++) store.encodeAndSet(i, data.get(i));
        var zeros = new MutableASHVectors(ash, SMALL_PAGE_BYTES);
        zeros.setZero(0);

        // New ordinals 1 and 3 are holes, as an OrdinalMapper reports them.
        int hole = OrdinalMapper.OMITTED;
        var snapshot = store.snapshot(new int[]{5, hole, 299, hole, 0});
        assertEquals(5, snapshot.count());
        assertEquals(store.get(5), snapshot.get(0));
        assertEquals(store.get(299), snapshot.get(2));
        assertEquals(store.get(0), snapshot.get(4));
        assertEquals(zeros.get(0), snapshot.get(1));
        assertEquals(zeros.get(0), snapshot.get(3));

        // A hole scores as a missing vector does.
        var query = vectors(1, 32).get(0);
        var expected = zeros.scoreFunctionFor(query, DOT).similarityTo(0);
        var scores = snapshot.scoreFunctionFor(query, DOT);
        assertEquals(expected, scores.similarityTo(1), 0f);
        assertEquals(expected, scores.similarityTo(3), 0f);

        // Any other ordinal outside the store is still rejected.
        assertThrows(IndexOutOfBoundsException.class, () -> store.snapshot(new int[]{-1}));
        assertThrows(IndexOutOfBoundsException.class, () -> store.snapshot(new int[]{300}));
    }
}
