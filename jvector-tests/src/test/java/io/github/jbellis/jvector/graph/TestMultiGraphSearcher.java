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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.AbstractExecutorService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertFalse;

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
    public void testOverqueryKeepsMatteringAsShardCountGrows() throws Exception {
        // Regression test for a real bug found via MultiShardBench: proportionalShare() rounds each
        // shard's topK_i and rerankK_i independently. With 16 equal shards and topK=10, topK_i =
        // round(10/16) = round(0.625) = 1. At rerankK=10 (overquery=1x), rerankK_i = round(10/16) = 1
        // too -- exactly matching topK_i, as it should for "no extra overquery requested". But at
        // rerankK=20 (overquery=2x), rerankK_i = round(20/16) = round(1.25) = 1 -- rounds to the exact
        // same value as 1x, silently erasing the requested 2x margin for every shard.
        //
        // Before the fix, this was catastrophic: with 16 shards each asked for topK_i=1, the initial
        // round alone already returns 16 candidates >= topK=10, so the merge was already "done" and
        // resume never fired -- meaning raising rerankK from 10 to 20 had *zero* effect on the result,
        // no matter how high a caller cranked overquery, as long as it kept rounding to the same
        // per-shard integer. The fix flags shards whose rounded rerankK_i undershoots their fair share
        // of the caller's requested ratio and forces a correcting resume round even though the initial
        // count already satisfied topK.
        int numShards = 16;
        int perShardSize = 20;
        var vectorSets = new ArrayList<TestVectorGraph.CircularFloatVectorValues>();
        var shards = new ArrayList<ImmutableGraphIndex>();
        var providers = new ArrayList<SearchScoreProvider>();
        var q = query();
        for (int i = 0; i < numShards; i++) {
            var vectors = new TestVectorGraph.CircularFloatVectorValues(perShardSize);
            vectorSets.add(vectors);
            shards.add(buildShard(vectors));
            providers.add(DefaultSearchScoreProvider.exact(q, SIMILARITY, vectors));
        }
        int topK = 10;

        try (var searcher = new MultiGraphSearcher(shards)) {
            var noOverquery = searcher.search(providers, topK, topK);
            assertEquals("exact per-shard share already matches what was asked -- no correction needed",
                    1, noOverquery.getRoundsUsed());

            var withOverquery = searcher.search(providers, topK, topK * 2);
            assertTrue("2x overquery must actually trigger extra search once its per-shard share "
                            + "rounds away to nothing, instead of being silently ignored",
                    withOverquery.getRoundsUsed() > 1);
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

    @Test
    public void testCloseClosesAllShardSearchers() throws Exception {
        var vectorsA = new TestVectorGraph.CircularFloatVectorValues(20);
        var vectorsB = new TestVectorGraph.CircularFloatVectorValues(15);
        var trackedA = new CloseTrackingIndex(buildShard(vectorsA));
        var trackedB = new CloseTrackingIndex(buildShard(vectorsB));

        try (var searcher = new MultiGraphSearcher(List.of(trackedA, trackedB))) {
            assertFalse("shard A's view must not be closed before close()", trackedA.viewClosed.get());
            assertFalse("shard B's view must not be closed before close()", trackedB.viewClosed.get());
        }

        assertTrue("shard A's view must be closed by MultiGraphSearcher.close()", trackedA.viewClosed.get());
        assertTrue("shard B's view must be closed by MultiGraphSearcher.close()", trackedB.viewClosed.get());
    }

    @Test
    public void testExhaustedShardDoesNotTriggerSpuriousResume() throws Exception {
        // acceptOrds hard-caps this single shard to 3 live candidates, far fewer than the topK=50 asked
        // for -- so the shard is guaranteed to be exhausted (return fewer nodes than requested) after
        // the very first round, regardless of graph traversal specifics. A correct implementation
        // recognizes this and stops after round 1 rather than repeatedly re-asking a shard that has
        // nothing left to give; a buggy implementation that ignored exhaustion would keep resuming it
        // for maxResumeRounds extra rounds even though every one of those rounds can only ever return
        // the same 3 (or fewer) nodes.
        var vectors = new TestVectorGraph.CircularFloatVectorValues(20);
        var shard = buildShard(vectors);
        var q = query();
        Bits acceptFirstThree = index -> index < 3;

        var counting = new CountingExecutor(Executors.newFixedThreadPool(1));
        try {
            try (var searcher = MultiGraphSearcher.builder(List.of(shard))
                    .withExecutor(counting)
                    .withMaxResumeRounds(5)
                    .build())
            {
                var result = searcher.search(
                        List.of(DefaultSearchScoreProvider.exact(q, SIMILARITY, vectors)),
                        List.of(acceptFirstThree),
                        50, 50);

                assertEquals("exhausted shard should stop after the initial round", 1, result.getRoundsUsed());
                assertEquals("only the 3 accepted candidates can ever be found", 3, result.getNodes().length);
            }
            assertEquals("an exhausted shard must not be re-dispatched on later rounds",
                    1, counting.tasksExecuted.get());
        } finally {
            counting.shutdown();
            assertTrue(counting.awaitTermination(10, TimeUnit.SECONDS));
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

    /**
     * Delegates every {@link ImmutableGraphIndex} method to a real shard, except {@link #getView()},
     * which wraps the real view in a {@link CloseTrackingView} so tests can observe whether
     * {@link MultiGraphSearcher#close()} actually closed it.
     */
    private static final class CloseTrackingIndex implements ImmutableGraphIndex {
        private final ImmutableGraphIndex delegate;
        final java.util.concurrent.atomic.AtomicBoolean viewClosed = new java.util.concurrent.atomic.AtomicBoolean(false);

        CloseTrackingIndex(ImmutableGraphIndex delegate) {
            this.delegate = delegate;
        }

        @Override
        public NodesIterator getNodes(int level) {
            return delegate.getNodes(level);
        }

        @Override
        public View getView() {
            return new CloseTrackingView(delegate.getView(), viewClosed);
        }

        @Override
        public int maxDegree() {
            return delegate.maxDegree();
        }

        @Override
        public List<Integer> maxDegrees() {
            return delegate.maxDegrees();
        }

        @Override
        public int getDimension() {
            return delegate.getDimension();
        }

        @Override
        public void close() throws IOException {
            delegate.close();
        }

        @Override
        public boolean isHierarchical() {
            return delegate.isHierarchical();
        }

        @Override
        public int getMaxLevel() {
            return delegate.getMaxLevel();
        }

        @Override
        public int getDegree(int level) {
            return delegate.getDegree(level);
        }

        @Override
        public double getAverageDegree(int level) {
            return delegate.getAverageDegree(level);
        }

        @Override
        public int size(int level) {
            return delegate.size(level);
        }

        @Override
        public long ramBytesUsed() {
            return delegate.ramBytesUsed();
        }
    }

    /**
     * Delegates every {@link ImmutableGraphIndex.View} method to a real view, except {@link #close()},
     * which flags {@code closed} before delegating.
     */
    private static final class CloseTrackingView implements ImmutableGraphIndex.View {
        private final ImmutableGraphIndex.View delegate;
        private final java.util.concurrent.atomic.AtomicBoolean closed;

        CloseTrackingView(ImmutableGraphIndex.View delegate, java.util.concurrent.atomic.AtomicBoolean closed) {
            this.delegate = delegate;
            this.closed = closed;
        }

        @Override
        public NodesIterator getNeighborsIterator(int level, int node) {
            return delegate.getNeighborsIterator(level, node);
        }

        @Override
        public void processNeighbors(int level, int node, io.github.jbellis.jvector.graph.similarity.ScoreFunction scoreFunction,
                                      IntMarker visited, NeighborProcessor neighborProcessor) {
            delegate.processNeighbors(level, node, scoreFunction, visited, neighborProcessor);
        }

        @Override
        public int size() {
            return delegate.size();
        }

        @Override
        public NodeAtLevel entryNode() {
            return delegate.entryNode();
        }

        @Override
        public Bits liveNodes() {
            return delegate.liveNodes();
        }

        @Override
        public boolean contains(int level, int node) {
            return delegate.contains(level, node);
        }

        @Override
        public void close() throws IOException {
            closed.set(true);
            delegate.close();
        }
    }
}
