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

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.MockVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.*;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.PreparedQuery;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;
import java.nio.file.Files;
import java.util.EnumMap;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.function.IntFunction;
import static org.junit.Assert.*;

/** Numerical preparation ownership, batch tails, handoff, and resumed graph searches. */
public class TestPreparedASHQuery {
    private static final VectorSimilarityFunction DOT = VectorSimilarityFunction.DOT_PRODUCT;

    private static VectorFloat<?>[] inputs(int count) {
        var type = VectorizationProvider.getInstance().getVectorTypeSupport();
        var values = new VectorFloat<?>[count];
        Random random = new Random(341975);
        for (int n = 0; n < count; n++) {
            values[n] = type.createFloatVector(64);
            for (int d = 0; d < 64; d++) values[n].set(d, random.nextFloat() - .5f);
            VectorUtil.l2normalize(values[n]);
        }
        return values;
    }

    private static AsymmetricHashing model(VectorFloat<?>[] input, int bits) throws Exception {
        return AsymmetricHashing.initialize(MockVectorValues.fromValues(input), AsymmetricHashing.RANDOM,
                AsymmetricHashing.HEADER_BITS + 33 * bits, 1, bits);
    }

    @Test
    public void batchMatchesSinglePreparationForEveryFusedBitCountAndOwnsInputs() throws Exception {
        var values = inputs(97);
        for (int bits : new int[]{1, 2, 3, 4, 9}) {
            var ash = model(values, bits);
            var preparer = new ASHQueryPreparer(ash);
            var states = new PreparedQuery[17];
            for (int i = 0; i < states.length; i++) states[i] = preparer.newState();
            var reference = preparer.newState();
            for (int count : new int[]{0, 1, 2, 8, 17}) {
                preparer.prepareBatch(values, states, count);
                for (int i = 0; i < count; i++) {
                    preparer.prepare(values[i], reference);
                    var expected = ASHQueryPreparer.state(ash, reference, true);
                    var actual = ASHQueryPreparer.state(ash, states[i], true);
                    assertArrayEquals(expected.prepared.qProj, actual.prepared.qProj, 2e-6f);
                    assertArrayEquals(expected.prepared.dotQMuByLandmark, actual.prepared.dotQMuByLandmark, 0f);
                    // Offset-binary entries weight Aq by up to 2^(b-1), amplifying batch/single projection
                    // differences of up to 2e-6 by the same factor.
                    float lutTolerance = bits == 1 || bits == 2 || bits == 4 ? 5e-6f : 2e-6f * (1 << bits);
                    for (int e = 0; e < expected.lut.length; e++) {
                        assertEquals(expected.lut[e], actual.lut[e], lutTolerance * Math.max(1f, Math.abs(expected.lut[e])));
                    }
                }
            }
            var mutable = values[0].copy();
            preparer.prepare(mutable, reference);
            float first = reference.queryVector().get(0);
            mutable.set(0, first + 10);
            assertEquals(first, reference.queryVector().get(0), 0f);
            long bytes = reference.ramBytesUsed();
            for (int i = 0; i < 50; i++) preparer.prepare(values[i], reference);
            assertEquals(bytes, reference.ramBytesUsed());
            assertTrue(bytes > 4L * (64 + 33));
        }
    }

    @Test
    public void invalidBatchDoesNotOverwriteExistingStates() throws Exception {
        var values = inputs(97);
        var ash = model(values, 2);
        var preparer = new ASHQueryPreparer(ash);
        var first = preparer.newState();
        var second = preparer.newState();
        assertThrows(IllegalStateException.class, first::queryVector);
        preparer.prepare(values[0], first);
        float original = first.queryVector().get(0);
        assertThrows(IllegalArgumentException.class,
                () -> preparer.prepareBatch(values, new PreparedQuery[]{first, first}, 2));
        assertEquals(original, first.queryVector().get(0), 0f);
        var other = new ASHQueryPreparer(model(values, 2)).newState();
        assertThrows(IllegalArgumentException.class, () -> preparer.prepare(values[0], other));
        assertThrows(IndexOutOfBoundsException.class,
                () -> preparer.prepareBatch(values, new PreparedQuery[]{first}, 2));
        assertThrows(IndexOutOfBoundsException.class,
                () -> preparer.prepareBatch(values, new PreparedQuery[]{first}, -1));
        var bad = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(63);
        assertThrows(IllegalArgumentException.class,
                () -> preparer.prepareBatch(new VectorFloat<?>[]{values[1], bad}, new PreparedQuery[]{first, second}, 2));
        assertEquals(original, first.queryVector().get(0), 0f);
    }


    @Test
    public void oneRowUsesDirectProjectionEvenWhenBackendHasNoSingleRowShortcut() throws Exception {
        var values = inputs(17);
        var ash = model(values, 2);
        var preparer = new ASHQueryPreparer(ash);
        preparer.supportsBatchPreparation(); // Initialize the immutable query view.
        var field = AsymmetricHashing.StiefelTransform.class.getDeclaredField("queryProjection");
        field.setAccessible(true);
        var original = (io.github.jbellis.jvector.vector.VectorUtilSupport.AshProjection) field.get(ash.stiefelTransform);
        int[] calls = new int[2];
        var wrapper = new io.github.jbellis.jvector.vector.VectorUtilSupport.AshProjection() {
            public void project(float[] input, float[] output) {
                calls[0]++;
                original.project(input, output);
            }
            public boolean supportsBatchProjection() { return true; }
            public void projectBatch(float[][] input, float[][] output, int count) {
                assertTrue("One-row requests must not enter backend batching", count > 1);
                calls[1]++;
                original.projectBatch(input, output, count);
            }
            public long ramBytesUsed() { return original.ramBytesUsed(); }
        };
        var first = preparer.newState();
        var second = preparer.newState();
        field.set(ash.stiefelTransform, wrapper);
        try {
            preparer.prepareBatch(values, new PreparedQuery[]{first}, 0);
            assertArrayEquals(new int[]{0, 0}, calls);
            preparer.prepareBatch(values, new PreparedQuery[]{first}, 1);
            assertArrayEquals(new int[]{1, 0}, calls);
            var snapshot = ASHQueryPreparer.state(ash, first, true).prepared.qProj.clone();
            long generation = ASHQueryPreparer.state(ash, first, true).generation;
            var bad = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(63);
            assertThrows(IllegalArgumentException.class,
                    () -> preparer.prepareBatch(new VectorFloat<?>[]{bad}, new PreparedQuery[]{first}, 1));
            assertEquals(generation, ASHQueryPreparer.state(ash, first, true).generation);
            assertArrayEquals(snapshot, ASHQueryPreparer.state(ash, first, true).prepared.qProj, 0f);
            assertArrayEquals(new int[]{1, 0}, calls);
            preparer.prepareBatch(values, new PreparedQuery[]{first, second}, 2);
            assertArrayEquals(new int[]{1, 1}, calls);
        } finally {
            field.set(ash.stiefelTransform, original);
        }
    }

    @Test(timeout=30000)
    public void distinctStatesCanBePreparedAndHandedOffAcrossThreads() throws Exception {
        var values = inputs(97);
        var ash = model(values, 4);
        var preparer = new ASHQueryPreparer(ash);
        var pool = Executors.newFixedThreadPool(4);
        try {
            var tasks = new java.util.ArrayList<java.util.concurrent.Future<PreparedQuery>>();
            for (int i = 0; i < 16; i++) {
                final int ordinal = i;
                tasks.add(pool.submit(() -> {
                    var state = preparer.newState();
                    preparer.prepare(values[ordinal], state);
                    return state;
                }));
            }
            for (int i = 0; i < tasks.size(); i++) {
                var state = tasks.get(i).get(10, TimeUnit.SECONDS);
                for (int d = 0; d < 64; d++) assertEquals(values[i].get(d), state.queryVector().get(d), 0f);
            }
        } finally { pool.shutdownNow(); }
    }

    private static void assertSameResults(SearchResult a, SearchResult b) {
        assertArrayEquals(a.getNodes(), b.getNodes());
    }

    @Test
    public void realDiskSearchFilteringAndResumeUsePreparedStateWithoutReprojection() throws Exception {
        var values = inputs(17);
        var ravv = MockVectorValues.fromValues(values);
        var graph = new TestUtil.FullyConnectedGraphIndex(0, List.of(17, 1)) {
            @Override public List<Integer> maxDegrees() { return List.of(16, 0); }
            @Override public int getDimension() { return 64; }
        };
        for (int bits : new int[]{1, 2, 3, 4, 9}) {
            var ash = model(values, bits);
            var encoded = ash.encodeAll(ravv, ForkJoinPool.commonPool());
            var path = Files.createTempFile("ash-prepared-query-", ".index");
            try {
                var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
                try (var view = graph.getView(); var writer = new OnDiskGraphIndexWriter.Builder(graph, path)
                        .with(new FusedASH(16, ash, 32)).with(new InlineVectors(64)).build()) {
                    suppliers.put(FeatureId.FUSED_ASH, n -> new FusedASH.State(view, encoded, n));
                    suppliers.put(FeatureId.INLINE_VECTORS, n -> new InlineVectors.State(values[n]));
                    writer.write(suppliers);
                }
                try (var reader = new SimpleMappedReader.Supplier(path); var disk = OnDiskGraphIndex.load(reader, 0);
                     var direct = new GraphSearcher(disk); var queued = new GraphSearcher(disk)) {
                    var preparer = disk.queryPreparer(DOT).orElseThrow();
                    var state = preparer.newState();
                    var directView = (OnDiskGraphIndex.View) direct.getView();
                    var queuedView = (OnDiskGraphIndex.View) queued.getView();
                    for (int q = 0; q < 4; q++) {
                        preparer.prepareBatch(new VectorFloat<?>[]{values[q]}, new PreparedQuery[]{state}, 1);
                        var provider = queuedView.scoreProviderFor(state, true);
                        var ordinary = new DefaultSearchScoreProvider(
                                directView.approximateScoreFunctionFor(values[q], DOT), directView.rerankerFor(values[q], DOT), DOT);
                        Bits accepted = n -> (n & 1) == 0;
                        assertSameResults(direct.search(ordinary, 3, 17, 0f, 0f, accepted),
                                queued.search(provider, 3, 17, 0f, 0f, accepted));
                        assertSameResults(direct.resume(3, 17), queued.resume(3, 17));
                        preparer.prepare(values[q + 1], state);
                        assertThrows(IllegalStateException.class, () -> provider.scoreFunction().similarityTo(0));
                    }
                    assertThrows(UnsupportedOperationException.class, () -> disk.queryPreparer(VectorSimilarityFunction.COSINE));
                }
            } finally { Files.deleteIfExists(path); }
        }
    }
}
