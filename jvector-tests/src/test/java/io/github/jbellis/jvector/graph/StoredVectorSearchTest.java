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
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.index.Indexes;
import io.github.jbellis.jvector.management.CompressionType;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.function.IntFunction;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests the conveniences for persisting a graph and searching it with the vectors it stores:
 * {@link PersistableGraphIndex#writeTo} and {@link GraphSearcher#search(VectorFloat, int, int,
 * VectorSimilarityFunction, Bits)}.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class StoredVectorSearchTest extends RandomizedTest {
    private static final int DIMENSION = 16;
    private static final VectorSimilarityFunction VSF = VectorSimilarityFunction.EUCLIDEAN;

    @Test
    public void writeToThenSearchWithInlineVectors() throws Exception {
        int n = 1_000;
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(n, DIMENSION), DIMENSION);
        PersistableGraphIndex graph = Indexes.hnswBuilder(ravv, VSF).buildAndPopulate();

        Path path = Files.createTempFile("stored-vector-search", ".graph");
        try {
            graph.writeTo(path, ravv);
            try (var rs = ReaderSupplierFactory.open(path);
                 var onDisk = OnDiskGraphIndex.load(rs);
                 var searcher = onDisk.searcher()) {
                assertEquals(n, onDisk.size(0));
                assertFalse(((GraphIndex.ScoringView) searcher.getView()).hasApproximateScores());

                // each vector finds itself, scored exactly with the inline vectors
                int found = 0;
                for (int i = 0; i < n; i += 10) {
                    SearchResult result = searcher.search(ravv.getVector(i), 1, 1, VSF, Bits.ALL);
                    if (result.getNodes()[0].node == i) {
                        found++;
                    }
                    assertEquals(1.0f, result.getNodes()[0].score, 1e-6f);
                }
                assertTrue(found > 95);

                // the filter is honored
                SearchResult filtered = searcher.search(ravv.getVector(0), 5, 10, VSF, node -> node % 2 == 1);
                for (var ns : filtered.getNodes()) {
                    assertTrue(ns.node % 2 == 1);
                }
            }
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void searchUsesFusedPqWhenTheGraphHasIt() throws Exception {
        int n = 2_000;
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(n, DIMENSION), DIMENSION);
        HnswIndexBuilder builder = Indexes.hnswBuilder(ravv, VSF).withCompressionType(CompressionType.PQ);
        PersistableGraphIndex graph = builder.buildAndPopulate();
        PQVectors pq = (PQVectors) builder.getCompressedVectors();

        Path path = Files.createTempFile("stored-vector-search-fused", ".graph");
        try {
            try (var view = graph.getView();
                 var writer = graph.getWriterBuilder(path)
                         .with(new InlineVectors(DIMENSION))
                         .with(new FusedPQ(graph.maxDegree(), pq.getCompressor()))
                         .build()) {
                var states = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
                states.put(FeatureId.INLINE_VECTORS, node -> new InlineVectors.State(ravv.getVector(node)));
                states.put(FeatureId.FUSED_PQ, node -> new FusedPQ.State(view, pq, node));
                writer.write(states);
            }
            try (var rs = ReaderSupplierFactory.open(path);
                 var onDisk = OnDiskGraphIndex.load(rs);
                 var convenient = onDisk.searcher();
                 var explicit = onDisk.searcher()) {
                var scoringView = (GraphIndex.ScoringView) explicit.getView();
                assertTrue(scoringView.hasApproximateScores());

                // the same results as traversing with fused PQ and reranking with the inline vectors by hand
                for (int i = 0; i < 20; i++) {
                    VectorFloat<?> q = ravv.getVector(i * 37);
                    var expected = explicit.search(new DefaultSearchScoreProvider(
                            scoringView.approximateScoreFunctionFor(q, VSF), scoringView.rerankerFor(q, VSF)),
                            10, 30, 0.0f, 0.0f, Bits.ALL);
                    var actual = convenient.search(q, 10, 30, VSF, Bits.ALL);
                    assertArrayEquals(nodes(expected), nodes(actual));
                }
            }
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void onDiskGraphRewritesThroughEachWriterAccessor() throws Exception {
        int n = 1_000;
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(n, DIMENSION), DIMENSION);
        PersistableGraphIndex graph = Indexes.hnswBuilder(ravv, VSF).buildAndPopulate();

        Path dir = Files.createTempDirectory("stored-vector-rewrite");
        Path original = dir.resolve("original.graph");
        Path randomAccess = dir.resolve("random-access.graph");
        Path parallel = dir.resolve("parallel.graph");
        Path sequential = dir.resolve("sequential.graph");
        try {
            graph.writeTo(original, ravv);
            try (var rs = ReaderSupplierFactory.open(original);
                 var source = OnDiskGraphIndex.load(rs)) {
                // The vectors come from the source graph itself. A view isn't thread-safe, so the parallel
                // writer, which calls the state function from its worker threads, gets one view per thread.
                var views = new java.util.concurrent.ConcurrentLinkedQueue<OnDiskGraphIndex.View>();
                ThreadLocal<OnDiskGraphIndex.View> viewPerThread = ThreadLocal.withInitial(() -> {
                    var v = source.getView();
                    views.add(v);
                    return v;
                });
                var fromSource = Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                        node -> new InlineVectors.State(viewPerThread.get().getVector(node)));
                try {
                    try (var writer = source.getWriterBuilder(randomAccess).with(new InlineVectors(DIMENSION)).build()) {
                        writer.write(fromSource);
                    }
                    try (var writer = source.getParallelWriterBuilder(parallel)
                            .with(new InlineVectors(DIMENSION))
                            .withParallelWorkerThreads(4)
                            .build()) {
                        writer.write(fromSource);
                    }
                    try (var out = new io.github.jbellis.jvector.disk.SimpleWriter(sequential);
                         var writer = source.getWriterBuilder(out).with(new InlineVectors(DIMENSION)).build()) {
                        writer.write(fromSource);
                    }
                } finally {
                    for (var v : views) {
                        v.close();
                    }
                }

                // Each copy has the same structure as the source, and searching it gives the same results.
                int[][] expected = new int[10][];
                try (var searcher = source.searcher()) {
                    for (int i = 0; i < expected.length; i++) {
                        expected[i] = nodes(searcher.search(ravv.getVector(i * 31), 10, 30, VSF, Bits.ALL));
                    }
                }
                for (Path rewritten : java.util.List.of(randomAccess, parallel, sequential)) {
                    try (var rs2 = ReaderSupplierFactory.open(rewritten);
                         var copy = OnDiskGraphIndex.load(rs2);
                         var searcher = copy.searcher()) {
                        io.github.jbellis.jvector.TestUtil.assertGraphEquals(source, copy);
                        for (int i = 0; i < expected.length; i++) {
                            assertArrayEquals(rewritten.toString(), expected[i],
                                    nodes(searcher.search(ravv.getVector(i * 31), 10, 30, VSF, Bits.ALL)));
                        }
                    }
                }
            }
        } finally {
            for (Path p : java.util.List.of(original, randomAccess, parallel, sequential)) {
                Files.deleteIfExists(p);
            }
            Files.deleteIfExists(dir);
        }
    }

    @Test
    public void searchNeedsAGraphThatStoresItsVectors() throws Exception {
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(100, DIMENSION), DIMENSION);
        PersistableGraphIndex inMemory = Indexes.hnswBuilder(ravv, VSF).buildAndPopulate();
        try (var searcher = inMemory.searcher()) {
            searcher.search(ravv.getVector(0), 5, 5, VSF, Bits.ALL);
            fail("expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("doesn't store its vectors"));
        }
    }

    private static int[] nodes(SearchResult result) {
        return Arrays.stream(result.getNodes()).mapToInt(ns -> ns.node).toArray();
    }
}
