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
            // Phase 2 sizes each shard's initial ask proportionally rather than asking every shard
            // for a full local top-K, so recovering the exact answer here relies on the resume loop
            // (unlike Phase 1, which was exact in a single round by construction). Bound it rather
            // than pin an exact count, since the round it converges on depends on the tuning
            // constants in growBudget()/OverqueryStrategy.DEFAULT.
            assertTrue("expected between 1 and 1 + maxResumeRounds rounds",
                    result.getRoundsUsed() >= 1 && result.getRoundsUsed() <= 3);
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
    public void testResumeRecoversAcrossSkewedShardSizes() throws Exception {
        // A large, finely-spaced shard and a tiny shard: proportional initial sizing gives the tiny
        // shard a local ask of 1 (round(5 * 4/64) rounds up to the floor of 1), even though its exact
        // query match (ordinal 0, tied for best possible score with the large shard's ordinal 0) must
        // appear in the true global top-5. Recovering it requires the resume loop to actually append
        // newly-discovered nodes across rounds rather than losing earlier rounds' results.
        var vectorsA = new TestVectorGraph.CircularFloatVectorValues(60);
        var vectorsB = new TestVectorGraph.CircularFloatVectorValues(4);
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
            for (int i = 0; i < topK; i++) {
                var actual = result.getNodes()[i];
                var expectedNode = expected.get(i);
                assertEquals("shardIndex at rank " + i, expectedNode.shardIndex, actual.shardIndex);
                assertEquals("node at rank " + i, expectedNode.node, actual.node);
                assertEquals("score at rank " + i, expectedNode.score, actual.score, 1e-5);
            }
        }
    }

    @Test
    public void testCallerResumeGrowsResultCount() throws Exception {
        // The caller-facing resume(additionalK) mirrors a lazy consumer (e.g. Cassandra) that ran its
        // own downstream filtering on the first batch, came up short of the results it actually
        // needed, and asks for more without restarting the multi-shard search from scratch.
        var vectorsA = new TestVectorGraph.CircularFloatVectorValues(20);
        var vectorsB = new TestVectorGraph.CircularFloatVectorValues(15);
        var shardA = buildShard(vectorsA);
        var shardB = buildShard(vectorsB);
        var q = query();

        var expected = bruteForceMerge(q, List.of(vectorsA, vectorsB), 8);

        try (var searcher = new MultiGraphSearcher(List.of(shardA, shardB))) {
            List<SearchScoreProvider> providers = List.of(
                    DefaultSearchScoreProvider.exact(q, SIMILARITY, vectorsA),
                    DefaultSearchScoreProvider.exact(q, SIMILARITY, vectorsB));

            var initial = searcher.search(providers, 3, 20);
            assertEquals(3, initial.getNodes().length);

            var grown = searcher.resume(5);
            assertEquals(8, grown.getNodes().length);
            // the first 3 results must still be there, in the same order, once grown to 8
            for (int i = 0; i < 3; i++) {
                assertEquals("rank " + i + " should be stable across resume", initial.getNodes()[i], grown.getNodes()[i]);
            }
            for (int i = 0; i < 8; i++) {
                var actual = grown.getNodes()[i];
                var expectedNode = expected.get(i);
                assertEquals("shardIndex at rank " + i, expectedNode.shardIndex, actual.shardIndex);
                assertEquals("node at rank " + i, expectedNode.node, actual.node);
                assertEquals("score at rank " + i, expectedNode.score, actual.score, 1e-5);
            }

            // metrics accumulate across the whole session, so resume's totals can't be less than search's
            assertTrue(grown.getVisitedCount() >= initial.getVisitedCount());
            assertTrue(grown.getRoundsUsed() >= initial.getRoundsUsed());
        }
    }

    @Test
    public void testResumeBeforeSearchThrows() throws Exception {
        var vectors = new TestVectorGraph.CircularFloatVectorValues(10);
        try (var searcher = new MultiGraphSearcher(List.of(buildShard(vectors)))) {
            assertThrows(IllegalStateException.class, () -> searcher.resume(5));
        }
    }

    @Test
    public void testResumeRejectsNonPositiveAdditionalK() throws Exception {
        var vectors = new TestVectorGraph.CircularFloatVectorValues(10);
        var shard = buildShard(vectors);
        var q = query();
        try (var searcher = new MultiGraphSearcher(List.of(shard))) {
            searcher.search(List.of(DefaultSearchScoreProvider.exact(q, SIMILARITY, vectors)), 3, 6);
            assertThrows(IllegalArgumentException.class, () -> searcher.resume(0));
            assertThrows(IllegalArgumentException.class, () -> searcher.resume(-1));
        }
    }

    @Test
    public void testSearchAfterResumeStartsAFreshSession() throws Exception {
        // search() must discard whatever session state a prior search()/resume() sequence left behind,
        // so a second, independent query on the same instance behaves exactly like a brand-new instance.
        var vectorsA = new TestVectorGraph.CircularFloatVectorValues(20);
        var vectorsB = new TestVectorGraph.CircularFloatVectorValues(15);
        var shardA = buildShard(vectorsA);
        var shardB = buildShard(vectorsB);
        var q = query();
        List<SearchScoreProvider> providers = List.of(
                DefaultSearchScoreProvider.exact(q, SIMILARITY, vectorsA),
                DefaultSearchScoreProvider.exact(q, SIMILARITY, vectorsB));

        ShardedSearchResult freshResult;
        try (var freshSearcher = new MultiGraphSearcher(List.of(shardA, shardB))) {
            freshResult = freshSearcher.search(providers, 5, 20);
        }

        try (var searcher = new MultiGraphSearcher(List.of(shardA, shardB))) {
            searcher.search(providers, 3, 20);
            searcher.resume(4);
            var second = searcher.search(providers, 5, 20);
            assertEquals(freshResult, second);
        }
    }

    @Test
    public void testMaxResumeRoundsZeroDisablesResume() throws Exception {
        var vectorsA = new TestVectorGraph.CircularFloatVectorValues(20);
        var vectorsB = new TestVectorGraph.CircularFloatVectorValues(15);
        var shardA = buildShard(vectorsA);
        var shardB = buildShard(vectorsB);
        var q = query();

        List<SearchScoreProvider> providers = List.of(
                DefaultSearchScoreProvider.exact(q, SIMILARITY, vectorsA),
                DefaultSearchScoreProvider.exact(q, SIMILARITY, vectorsB));

        try (var searcher = MultiGraphSearcher.builder(List.of(shardA, shardB))
                .withMaxResumeRounds(0)
                .build())
        {
            var result = searcher.search(providers, 5, 20);
            assertEquals("resume should never trigger with maxResumeRounds=0", 1, result.getRoundsUsed());
        }
    }

    @Test
    public void testRejectsNegativeMaxResumeRounds() {
        assertThrows(IllegalArgumentException.class,
                () -> MultiGraphSearcher.builder(List.of(buildShard(new TestVectorGraph.CircularFloatVectorValues(5))))
                        .withMaxResumeRounds(-1));
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
            // At minimum, the initial round dispatches one task per shard via the executor. Later
            // rounds only dispatch for shards that still qualify for resume (see search()'s javadoc),
            // so the exact total varies with the tuning constants in growBudget()/OverqueryStrategy;
            // this just confirms the parallel path was actually exercised, not silently bypassed.
            assertTrue("expected at least one task submitted per shard",
                    counting.tasksExecuted.get() >= 3);
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
