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

package io.github.jbellis.jvector.graph.disk;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessByteVectorValues;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.PersistableGraphIndex;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineByteVectors;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.PreparedQuery;
import io.github.jbellis.jvector.graph.similarity.QueryPreparer;
import io.github.jbellis.jvector.index.Indexes;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.ByteVectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.function.IntFunction;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Tests that queries prepared for an on-disk graph with fused PQ, NVQ or inline vectors search exactly as the same
 * queries do through the explicit-function path.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestOnDiskQueryPreparer extends RandomizedTest {
    private static final int DIMENSION = 32;
    private static final int SIZE = 400;
    private Path directory;

    /** The ways a graph's vectors can be stored: what scores approximately, and what reranks. */
    private enum Layout { INLINE, NVQ, FUSED_PQ_INLINE, FUSED_PQ_NVQ }

    @Before
    public void setUp() throws IOException {
        directory = Files.createTempDirectory(getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(directory);
    }

    private OnDiskGraphIndex write(PersistableGraphIndex graph, ListRandomAccessVectorValues ravv, Layout layout,
                                   Integer version, String name) throws IOException {
        var path = directory.resolve(name);
        if (layout == Layout.FUSED_PQ_INLINE || layout == Layout.FUSED_PQ_NVQ) {
            var pqv = (PQVectors) ProductQuantization.compute(ravv, 8, 256, false).encodeAll(ravv);
            TestUtil.writeFusedGraph(graph, ravv, pqv,
                    layout == Layout.FUSED_PQ_INLINE ? FeatureId.INLINE_VECTORS : FeatureId.NVQ_VECTORS, path);
        } else {
            var builder = new OnDiskGraphIndexWriter.Builder(graph, path);
            if (version != null) {
                builder.withVersion(version);
            }
            var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
            if (layout == Layout.INLINE) {
                builder.with(new InlineVectors(DIMENSION));
                suppliers.put(FeatureId.INLINE_VECTORS, n -> new InlineVectors.State(ravv.getVector(n)));
            } else {
                var nvq = NVQuantization.compute(ravv, 2);
                builder.with(new NVQ(nvq));
                suppliers.put(FeatureId.NVQ_VECTORS, n -> new NVQ.State(nvq.encode(ravv.getVector(n))));
            }
            try (var writer = builder.build()) {
                writer.write(suppliers);
            }
        }
        return OnDiskGraphIndex.load(new SimpleMappedReader.Supplier(path));
    }

    private static void assertSameResults(String message, SearchResult expected, SearchResult actual) {
        assertEquals(message, expected.getNodes().length, actual.getNodes().length);
        for (int i = 0; i < expected.getNodes().length; i++) {
            assertEquals(message + " node " + i, expected.getNodes()[i].node, actual.getNodes()[i].node);
            assertEquals(message + " score " + i, expected.getNodes()[i].score, actual.getNodes()[i].score, 0f);
        }
    }

    @Test
    public void preparedQueriesSearchAsUnpreparedOnesForEveryLayoutAndFunction() throws Exception {
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(SIZE, DIMENSION), DIMENSION);
        var queries = createRandomVectors(5, DIMENSION);
        for (VectorSimilarityFunction vsf : VectorSimilarityFunction.values()) {
            var graph = Indexes.hnswBuilder(ravv, vsf).buildAndPopulate();
            for (Layout layout : Layout.values()) {
                String message = vsf + " " + layout;
                try (var onDisk = write(graph, ravv, layout, null, message);
                     var direct = new GraphSearcher(onDisk);
                     var prepared = new GraphSearcher(onDisk)) {
                    QueryPreparer preparer = onDisk.queryPreparer().orElseThrow();
                    PreparedQuery state = preparer.newState();
                    var directView = (OnDiskGraphIndex.View) direct.getView();
                    var preparedView = (OnDiskGraphIndex.View) prepared.getView();
                    for (VectorFloat<?> query : queries) {
                        preparer.prepare(query, state);
                        var reranker = directView.rerankerFor(query, vsf);
                        var unprepared = directView.hasApproximateScores()
                                ? new DefaultSearchScoreProvider(directView.approximateScoreFunctionFor(query, vsf), reranker)
                                : new DefaultSearchScoreProvider(reranker);
                        assertSameResults(message, direct.search(unprepared, 5, 20, 0f, 0f, Bits.ALL),
                                prepared.search(preparedView.scoreProviderFor(state, true), 5, 20, 0f, 0f, Bits.ALL));
                        assertSameResults(message + " resumed", direct.resume(5, 20), prepared.resume(5, 20));
                    }
                }
            }
        }
    }

    @Test
    public void batchPreparationMatchesSinglePreparation() throws Exception {
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(SIZE, DIMENSION), DIMENSION);
        var queries = createRandomVectors(3, DIMENSION).toArray(new VectorFloat<?>[0]);
        var graph = Indexes.hnswBuilder(ravv, VectorSimilarityFunction.COSINE).buildAndPopulate();
        try (var onDisk = write(graph, ravv, Layout.FUSED_PQ_NVQ, null, "batch");
             var single = new GraphSearcher(onDisk);
             var batched = new GraphSearcher(onDisk)) {
            var preparer = onDisk.queryPreparer().orElseThrow();
            assertFalse(preparer.supportsBatchPreparation());
            var states = new PreparedQuery[queries.length];
            for (int i = 0; i < states.length; i++) {
                states[i] = preparer.newState();
            }
            preparer.prepareBatch(queries, states, queries.length);
            var one = preparer.newState();
            var singleView = (OnDiskGraphIndex.View) single.getView();
            var batchedView = (OnDiskGraphIndex.View) batched.getView();
            for (int i = 0; i < queries.length; i++) {
                preparer.prepare(queries[i], one);
                assertSameResults("query " + i, single.search(singleView.scoreProviderFor(one, true), 5, Bits.ALL),
                        batched.search(batchedView.scoreProviderFor(states[i], true), 5, Bits.ALL));
            }
            // the same state twice in one batch is rejected before anything is prepared
            assertThrows(IllegalArgumentException.class,
                    () -> preparer.prepareBatch(queries, new PreparedQuery[]{states[0], states[0], states[1]}, 3));
        }
    }

    @Test
    public void withoutRerankingFusedPQScoresApproximatelyOnly() throws Exception {
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(SIZE, DIMENSION), DIMENSION);
        var graph = Indexes.hnswBuilder(ravv, VectorSimilarityFunction.EUCLIDEAN).buildAndPopulate();
        try (var onDisk = write(graph, ravv, Layout.FUSED_PQ_INLINE, null, "no-rerank");
             var view = (OnDiskGraphIndex.View) onDisk.getView()) {
            var preparer = onDisk.queryPreparer().orElseThrow();
            var state = preparer.newState();
            preparer.prepare(ravv.getVector(0), state);
            var provider = (DefaultSearchScoreProvider) view.scoreProviderFor(state, false);
            assertFalse(provider.scoreFunction().isExact());
            assertNull(provider.reranker());
            var exact = (DefaultSearchScoreProvider) view.scoreProviderFor(state, true);
            assertTrue(exact.reranker() != null);
        }
    }

    @Test
    public void graphsThatRecordNoFloatFunctionHaveNoPreparer() throws Exception {
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(100, DIMENSION), DIMENSION);
        var graph = Indexes.hnswBuilder(ravv, VectorSimilarityFunction.COSINE).buildAndPopulate();
        // version 7 records no function
        try (var v7 = write(graph, ravv, Layout.INLINE, 7, "v7")) {
            assertTrue(v7.queryPreparer().isEmpty());
        }
        // int8 records a byte function, and a preparer takes float queries
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var bytes = new ArrayList<ByteSequence<?>>();
        for (int i = 0; i < 100; i++) {
            var b = vts.createByteSequence(DIMENSION);
            for (int d = 0; d < DIMENSION; d++) {
                b.set(d, (byte) (randomIntBetween(-127, 127)));
            }
            bytes.add(b);
        }
        var rabvv = new ListRandomAccessByteVectorValues(bytes, DIMENSION);
        var byteGraph = GraphIndexBuilder.builder(rabvv, ByteVectorSimilarityFunction.EUCLIDEAN, 8)
                .withBeamWidth(20).withNeighborOverflow(1.2f).withAlpha(1.2f).build().build(rabvv);
        var path = directory.resolve("int8");
        try (var writer = new OnDiskGraphIndexWriter.Builder(byteGraph, path).with(new InlineByteVectors(DIMENSION)).build()) {
            writer.write(Feature.singleStateFactory(FeatureId.INLINE_BYTE_VECTORS,
                    n -> new InlineByteVectors.State(rabvv.getVector(n))));
        }
        try (var int8 = OnDiskGraphIndex.load(new SimpleMappedReader.Supplier(path))) {
            assertTrue(int8.queryPreparer().isEmpty());
        }
    }

    @Test
    public void statesAreCheckedBeforeUse() throws Exception {
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(100, DIMENSION), DIMENSION);
        var graph = Indexes.hnswBuilder(ravv, VectorSimilarityFunction.DOT_PRODUCT).buildAndPopulate();
        try (var first = write(graph, ravv, Layout.INLINE, null, "first");
             var second = write(graph, ravv, Layout.INLINE, null, "second");
             var view = (OnDiskGraphIndex.View) first.getView()) {
            var preparer = first.queryPreparer().orElseThrow();
            var unprepared = preparer.newState();
            assertThrows(IllegalStateException.class, () -> view.scoreProviderFor(unprepared, true));
            var other = second.queryPreparer().orElseThrow().newState();
            assertThrows(IllegalArgumentException.class, () -> preparer.prepare(ravv.getVector(0), other));
            assertThrows(IllegalArgumentException.class, () -> view.scoreProviderFor(other, true));
            assertThrows(IllegalArgumentException.class,
                    () -> preparer.prepare(createRandomVectors(1, DIMENSION + 1).get(0), unprepared));
        }
    }
}
