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
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/// Tests for the {@link ParallelExecutor} caller-runs build/finalize mode added to
/// {@link GraphIndexBuilder}. Verifies that {@link ParallelExecutor#callerRuns()} runs every
/// internal build/cleanup iteration on the calling thread (no worker threads), that it yields a
/// graph of equivalent quality to the {@link java.util.concurrent.ForkJoinPool} path (build,
/// cleanup, and the {@code removeDeletedNodes} path), and that the existing {@code ForkJoinPool}
/// constructors remain intact.
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class GraphIndexBuilderCallerRunsTest extends RandomizedTest {

    private static final int DIM = 32;
    private static final int SIZE = 2_000;
    private static final int M = 16;
    private static final int BEAM = 100;
    private static final int TOP_K = 10;
    private static final VectorSimilarityFunction VSF = VectorSimilarityFunction.EUCLIDEAN;

    private List<VectorFloat<?>> vectors;
    private ListRandomAccessVectorValues ravv;

    private void makeData() {
        vectors = TestUtil.createRandomVectors(SIZE, DIM);
        ravv = new ListRandomAccessVectorValues(vectors, DIM);
    }

    /// With {@code callerRuns()}, every build and cleanup iteration must execute on the calling
    /// thread — a probe recording the thread of each per-node scoring call must observe only the
    /// test thread, proving no worker threads (and not the common pool) were used.
    @Test
    public void callerRunsExecutesOnlyOnCallingThread() throws IOException {
        makeData();
        Thread testThread = Thread.currentThread();
        ThreadRecordingScoreProvider probe =
                new ThreadRecordingScoreProvider(BuildScoreProvider.randomAccessScoreProvider(ravv, VSF));

        try (var builder = new GraphIndexBuilder(probe, DIM, M, BEAM, 1.2f, 1.2f, true, true,
                ParallelExecutor.callerRuns(), ParallelExecutor.callerRuns())) {
            // Also exercise the removeDeletedNodes (sites 5/6) path under caller-runs.
            builder.build(ravv);
            for (int i = 0; i < SIZE; i += 25) {
                builder.markNodeDeleted(i);
            }
            builder.cleanup();
        }

        assertTrue("build must have invoked the score provider", probe.observedThreads.size() >= 1);
        assertEquals("caller-runs must execute all build/cleanup work on the calling thread only, but saw: "
                        + probe.observedThreads,
                Set.of(testThread), probe.observedThreads);
    }

    /// Caller-runs and ForkJoinPool builds of the same dataset must have equivalent recall. Graph
    /// structure is not bit-identical (concurrent insertion is order-sensitive), so the bar is
    /// recall-within-tolerance, both above a solid floor.
    @Test
    public void callerRunsMatchesForkJoinRecall() throws IOException {
        makeData();
        List<VectorFloat<?>> queries = TestUtil.createRandomVectors(100, DIM);
        List<Set<Integer>> groundTruth = bruteForceGroundTruth(queries);

        double callerRunsRecall;
        try (var builder = new GraphIndexBuilder(BuildScoreProvider.randomAccessScoreProvider(ravv, VSF),
                DIM, M, BEAM, 1.2f, 1.2f, true, true,
                ParallelExecutor.callerRuns(), ParallelExecutor.callerRuns())) {
            callerRunsRecall = recallAtK(builder.build(ravv), queries, groundTruth);
        }

        ForkJoinPool pool = new ForkJoinPool(4);
        double forkJoinRecall;
        try (var builder = new GraphIndexBuilder(BuildScoreProvider.randomAccessScoreProvider(ravv, VSF),
                DIM, M, BEAM, 1.2f, 1.2f, true, true, pool, pool)) {
            forkJoinRecall = recallAtK(builder.build(ravv), queries, groundTruth);
        } finally {
            pool.shutdown();
        }

        assertTrue("caller-runs recall too low: " + callerRunsRecall, callerRunsRecall > 0.85);
        assertTrue("fork-join recall too low: " + forkJoinRecall, forkJoinRecall > 0.85);
        assertTrue("caller-runs and fork-join recall must be equivalent, but were "
                        + callerRunsRecall + " vs " + forkJoinRecall,
                Math.abs(callerRunsRecall - forkJoinRecall) < 0.1);
    }

    /// The removeDeletedNodes path (cleanup sites 5/6) under caller-runs must produce a valid,
    /// good-recall graph over the surviving nodes, equivalent to the ForkJoinPool path.
    @Test
    public void callerRunsRemoveDeletedNodesMatchesForkJoin() throws IOException {
        makeData();
        List<VectorFloat<?>> queries = TestUtil.createRandomVectors(100, DIM);

        double callerRunsRecall = buildDeleteAndMeasure(ParallelExecutor.callerRuns(), ParallelExecutor.callerRuns(), queries);

        ForkJoinPool pool = new ForkJoinPool(4);
        double forkJoinRecall;
        try {
            forkJoinRecall = buildDeleteAndMeasure(ParallelExecutor.forkJoin(pool), ParallelExecutor.forkJoin(pool), queries);
        } finally {
            pool.shutdown();
        }

        assertTrue("caller-runs post-delete recall too low: " + callerRunsRecall, callerRunsRecall > 0.80);
        assertTrue("post-delete recall must be equivalent, but were " + callerRunsRecall + " vs " + forkJoinRecall,
                Math.abs(callerRunsRecall - forkJoinRecall) < 0.1);
    }

    /// The existing ForkJoinPool constructor must still build a good graph (back-compat).
    @Test
    public void forkJoinPoolConstructorStillBuildsGoodGraph() throws IOException {
        makeData();
        List<VectorFloat<?>> queries = TestUtil.createRandomVectors(100, DIM);
        List<Set<Integer>> groundTruth = bruteForceGroundTruth(queries);

        try (var builder = new GraphIndexBuilder(ravv, VSF, M, BEAM, 1.2f, 1.2f, true, true)) {
            // default-pool constructor path
            assertTrue(recallAtK(builder.build(ravv), queries, groundTruth) > 0.85);
        }
    }

    // ---- helpers ----

    private double buildDeleteAndMeasure(ParallelExecutor simd, ParallelExecutor parallel, List<VectorFloat<?>> queries) throws IOException {
        try (var builder = new GraphIndexBuilder(BuildScoreProvider.randomAccessScoreProvider(ravv, VSF),
                DIM, M, BEAM, 1.2f, 1.2f, true, true, simd, parallel)) {
            var graph = builder.build(ravv);
            Set<Integer> deleted = new HashSet<>();
            for (int i = 0; i < SIZE; i += 10) {
                builder.markNodeDeleted(i);
                deleted.add(i);
            }
            builder.cleanup();
            List<Set<Integer>> gt = bruteForceGroundTruthExcluding(queries, deleted);
            return recallAtK(graph, queries, gt);
        }
    }

    private List<Set<Integer>> bruteForceGroundTruth(List<VectorFloat<?>> queries) {
        return bruteForceGroundTruthExcluding(queries, Set.of());
    }

    private List<Set<Integer>> bruteForceGroundTruthExcluding(List<VectorFloat<?>> queries, Set<Integer> excluded) {
        List<Set<Integer>> out = new ArrayList<>(queries.size());
        for (VectorFloat<?> q : queries) {
            int[] best = new int[TOP_K];
            float[] bestScore = new float[TOP_K];
            int count = 0;
            for (int j = 0; j < vectors.size(); j++) {
                if (excluded.contains(j)) {
                    continue;
                }
                float s = VSF.compare(q, vectors.get(j));
                // insert into the small top-K buffer (higher score = closer)
                if (count < TOP_K) {
                    best[count] = j;
                    bestScore[count] = s;
                    count++;
                } else {
                    int minIdx = 0;
                    for (int t = 1; t < TOP_K; t++) {
                        if (bestScore[t] < bestScore[minIdx]) minIdx = t;
                    }
                    if (s > bestScore[minIdx]) {
                        best[minIdx] = j;
                        bestScore[minIdx] = s;
                    }
                }
            }
            Set<Integer> gt = new HashSet<>();
            for (int t = 0; t < count; t++) gt.add(best[t]);
            out.add(gt);
        }
        return out;
    }

    private double recallAtK(ImmutableGraphIndex graph, List<VectorFloat<?>> queries, List<Set<Integer>> groundTruth) throws IOException {
        int efSearch = 100;   // search width >> topK, matching TestVectorGraph's recall methodology
        double total = 0;
        for (int qi = 0; qi < queries.size(); qi++) {
            SearchResult.NodeScore[] nodes =
                    GraphSearcher.search(queries.get(qi), efSearch, ravv, VSF, graph, Bits.ALL).getNodes();
            Set<Integer> gt = groundTruth.get(qi);
            int limit = Math.min(TOP_K, nodes.length);
            int hits = 0;
            for (int t = 0; t < limit; t++) {
                if (gt.contains(nodes[t].node)) hits++;
            }
            total += (double) hits / Math.max(1, gt.size());
        }
        return total / queries.size();
    }

    /// A {@link BuildScoreProvider} that records the thread of every per-node scoring request,
    /// delegating all work. Used to prove caller-runs never leaves the calling thread.
    private static final class ThreadRecordingScoreProvider implements BuildScoreProvider {
        private final BuildScoreProvider delegate;
        final Set<Thread> observedThreads = ConcurrentHashMap.newKeySet();

        ThreadRecordingScoreProvider(BuildScoreProvider delegate) {
            this.delegate = delegate;
        }

        @Override
        public boolean isExact() {
            return delegate.isExact();
        }

        @Override
        public VectorFloat<?> approximateCentroid() {
            return delegate.approximateCentroid();
        }

        @Override
        public SearchScoreProvider searchProviderFor(VectorFloat<?> vector) {
            return delegate.searchProviderFor(vector);
        }

        @Override
        public SearchScoreProvider searchProviderFor(int node1) {
            observedThreads.add(Thread.currentThread());
            return delegate.searchProviderFor(node1);
        }

        @Override
        public SearchScoreProvider diversityProviderFor(int node1) {
            observedThreads.add(Thread.currentThread());
            return delegate.diversityProviderFor(node1);
        }

        @Override
        public ScoreFunction diversityScoreFunctionFor(int node1) {
            observedThreads.add(Thread.currentThread());
            return delegate.diversityScoreFunctionFor(node1);
        }
    }
}
