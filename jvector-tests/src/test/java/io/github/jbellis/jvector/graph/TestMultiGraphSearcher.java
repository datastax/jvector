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

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.AbstractExecutorService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestMultiGraphSearcher extends LuceneTestCase {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.COSINE;

    private static VectorFloat<?> query() {
        // same point CircularFloatVectorValues places at ordinal 0
        return vectorTypeSupport.createFloatVector(new float[]{1f, 0f});
    }

    private static ImmutableGraphIndex buildShard(TestVectorGraph.CircularFloatVectorValues vectors) {
        var builder = new GraphIndexBuilder(vectors, SIMILARITY, 16, 100, 1.0f, 1.4f, false);
        return TestUtil.buildSequentially(builder, vectors);
    }

    /**
     * Ground truth computed directly from the vectors, independent of any graph traversal --
     * this is what MultiGraphSearcher's merged results should match exactly, given per-shard
     * exact scoring and generous rerankK on small, densely-connected graphs.
     */
    private static List<ShardedSearchResult.NodeScore> bruteForceMerge(VectorFloat<?> q, List<TestVectorGraph.CircularFloatVectorValues> shards, int topK) {
        var all = new ArrayList<ShardedSearchResult.NodeScore>();
        for (int s = 0; s < shards.size(); s++) {
            var vectors = shards.get(s);
            for (int i = 0; i < vectors.size(); i++) {
                all.add(new ShardedSearchResult.NodeScore(s, i, SIMILARITY.compare(q, vectors.getVector(i))));
            }
        }
        Collections.sort(all);
        return all.subList(0, Math.min(topK, all.size()));
    }

    @Test
    public void testMergesAcrossShards() throws Exception {
        // Different sizes -> different point sets, so the true nearest neighbors are
        // genuinely spread across both shards rather than trivially tied.
        var vectorsA = new TestVectorGraph.CircularFloatVectorValues(20);
        var vectorsB = new TestVectorGraph.CircularFloatVectorValues(15);
        var shardA = buildShard(vectorsA);
        var shardB = buildShard(vectorsB);

        var q = query();
        int topK = 5;
        var expected = bruteForceMerge(q, List.of(vectorsA, vectorsB), topK);

        try (var searcher = new MultiGraphSearcher(List.of(shardA, shardB))) {
            List<SearchScoreProvider> providers = List.of(
                    DefaultSearchScoreProvider.exact(q, SIMILARITY, vectorsA),
                    DefaultSearchScoreProvider.exact(q, SIMILARITY, vectorsB));

            var result = searcher.search(providers, topK, 20);

            assertEquals(topK, result.getNodes().length);
            assertEquals(1, result.getRoundsUsed());
            for (int i = 0; i < topK; i++) {
                var actual = result.getNodes()[i];
                var expectedNode = expected.get(i);
                assertEquals("shardIndex at rank " + i, expectedNode.shardIndex, actual.shardIndex);
                assertEquals("node at rank " + i, expectedNode.node, actual.node);
                assertEquals("score at rank " + i, expectedNode.score, actual.score, 1e-5);
            }

            // sanity: both shards actually contributed to the ground truth, so this test is
            // exercising cross-shard merging and not just returning shard 0 unchanged
            assertTrue("expected results from both shards",
                    expected.stream().anyMatch(n -> n.shardIndex == 0) && expected.stream().anyMatch(n -> n.shardIndex == 1));
        }
    }

    @Test
    public void testConvenienceOverloadMatchesExplicitAcceptAll() throws Exception {
        var vectorsA = new TestVectorGraph.CircularFloatVectorValues(20);
        var vectorsB = new TestVectorGraph.CircularFloatVectorValues(15);
        var shardA = buildShard(vectorsA);
        var shardB = buildShard(vectorsB);
        var q = query();

        try (var searcher = new MultiGraphSearcher(List.of(shardA, shardB))) {
            List<SearchScoreProvider> providers = List.of(
                    DefaultSearchScoreProvider.exact(q, SIMILARITY, vectorsA),
                    DefaultSearchScoreProvider.exact(q, SIMILARITY, vectorsB));

            var withExplicitBits = searcher.search(providers, List.of(Bits.ALL, Bits.ALL), 5, 20);
            var withConvenience = searcher.search(providers, 5, 20);

            assertEquals(withExplicitBits, withConvenience);
        }
    }

    @Test
    public void testRespectsPerShardAcceptOrds() throws Exception {
        var vectorsA = new TestVectorGraph.CircularFloatVectorValues(20);
        var vectorsB = new TestVectorGraph.CircularFloatVectorValues(15);
        var shardA = buildShard(vectorsA);
        var shardB = buildShard(vectorsB);
        var q = query();

        try (var searcher = new MultiGraphSearcher(List.of(shardA, shardB))) {
            List<SearchScoreProvider> providers = List.of(
                    DefaultSearchScoreProvider.exact(q, SIMILARITY, vectorsA),
                    DefaultSearchScoreProvider.exact(q, SIMILARITY, vectorsB));

            // reject everything in shard B; every result must come from shard A
            var result = searcher.search(providers, List.of(Bits.ALL, Bits.NONE), 5, 20);

            assertEquals(5, result.getNodes().length);
            for (var node : result.getNodes()) {
                assertEquals(0, node.shardIndex);
            }
        }
    }

    @Test
    public void testRequiresAtLeastOneShard() {
        assertThrows(IllegalArgumentException.class, () -> new MultiGraphSearcher(List.of()));
    }

    @Test
    public void testValidatesListSizesMatchShardCount() throws Exception {
        var vectorsA = new TestVectorGraph.CircularFloatVectorValues(20);
        var vectorsB = new TestVectorGraph.CircularFloatVectorValues(15);
        var shardA = buildShard(vectorsA);
        var shardB = buildShard(vectorsB);
        var q = query();

        try (var searcher = new MultiGraphSearcher(List.of(shardA, shardB))) {
            List<SearchScoreProvider> onlyOneProvider = List.of(DefaultSearchScoreProvider.exact(q, SIMILARITY, vectorsA));
            assertThrows(IllegalArgumentException.class, () -> searcher.search(onlyOneProvider, 5, 20));
        }
    }

    @Test
    public void testParallelExecutorMatchesSequentialResult() throws Exception {
        // Three shards so fan-out actually has something to parallelize.
        var vectorsA = new TestVectorGraph.CircularFloatVectorValues(20);
        var vectorsB = new TestVectorGraph.CircularFloatVectorValues(15);
        var vectorsC = new TestVectorGraph.CircularFloatVectorValues(12);
        var shardA = buildShard(vectorsA);
        var shardB = buildShard(vectorsB);
        var shardC = buildShard(vectorsC);
        var q = query();
        int topK = 5;

        List<SearchScoreProvider> providers = List.of(
                DefaultSearchScoreProvider.exact(q, SIMILARITY, vectorsA),
                DefaultSearchScoreProvider.exact(q, SIMILARITY, vectorsB),
                DefaultSearchScoreProvider.exact(q, SIMILARITY, vectorsC));

        ShardedSearchResult sequential;
        try (var searcher = new MultiGraphSearcher(List.of(shardA, shardB, shardC))) {
            sequential = searcher.search(providers, topK, 20);
        }

        var counting = new CountingExecutor(Executors.newFixedThreadPool(2));
        try {
            try (var searcher = MultiGraphSearcher.builder(List.of(shardA, shardB, shardC))
                    .withExecutor(counting)
                    .build())
            {
                var parallel = searcher.search(providers, topK, 20);
                assertEquals("parallel fan-out must merge to the same result as sequential search",
                        sequential, parallel);
            }
            assertEquals("expected one task submitted per shard", 3, counting.tasksExecuted.get());
        } finally {
            counting.shutdown();
            assertTrue(counting.awaitTermination(10, TimeUnit.SECONDS));
        }
    }

    @Test
    public void testBuilderWithoutExecutorBehavesLikeConstructor() throws Exception {
        var vectorsA = new TestVectorGraph.CircularFloatVectorValues(20);
        var vectorsB = new TestVectorGraph.CircularFloatVectorValues(15);
        var shardA = buildShard(vectorsA);
        var shardB = buildShard(vectorsB);
        var q = query();

        List<SearchScoreProvider> providers = List.of(
                DefaultSearchScoreProvider.exact(q, SIMILARITY, vectorsA),
                DefaultSearchScoreProvider.exact(q, SIMILARITY, vectorsB));

        try (var viaConstructor = new MultiGraphSearcher(List.of(shardA, shardB));
             var viaBuilder = MultiGraphSearcher.builder(List.of(shardA, shardB)).build())
        {
            assertEquals(viaConstructor.search(providers, 5, 20), viaBuilder.search(providers, 5, 20));
        }
    }

    /**
     * Delegating {@link ExecutorService} that counts how many tasks were actually submitted to it,
     * so tests can confirm the parallel fan-out path was exercised rather than silently falling
     * back to sequential execution.
     */
    private static final class CountingExecutor extends AbstractExecutorService {
        private final ExecutorService delegate;
        final AtomicInteger tasksExecuted = new AtomicInteger();

        CountingExecutor(ExecutorService delegate) {
            this.delegate = delegate;
        }

        @Override
        public void execute(Runnable command) {
            tasksExecuted.incrementAndGet();
            delegate.execute(command);
        }

        @Override
        public void shutdown() {
            delegate.shutdown();
        }

        @Override
        public List<Runnable> shutdownNow() {
            return delegate.shutdownNow();
        }

        @Override
        public boolean isShutdown() {
            return delegate.isShutdown();
        }

        @Override
        public boolean isTerminated() {
            return delegate.isTerminated();
        }

        @Override
        public boolean awaitTermination(long timeout, TimeUnit unit) throws InterruptedException {
            return delegate.awaitTermination(timeout, unit);
        }
    }
}
