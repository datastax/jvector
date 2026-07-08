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
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinWorkerThread;
import java.util.function.IntFunction;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/// Acceptance tests for {@link EmbeddedExecutionContext}: the "no parallel work escapes to
/// {@code commonPool} / {@code PhysicalCoreExecutor.pool()}" guarantee. The context is built with a
/// single-thread {@link ForkJoinPool} whose worker is distinctively named; a thread-recording
/// {@link RandomAccessVectorValues} observes which threads execute each operation's parallel bodies.
/// Every operation that reads vectors through the context (build, cleanup, PQ train/refine/encode,
/// NVQ encode) must run only on that named worker (or the calling thread) — never on a foreign
/// pool. A second test drives the graph-build and fused-compaction (PQ retrain) paths end to end
/// through the context and asserts the output is correct.
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class EmbeddedExecutionContextTest extends RandomizedTest {

    private static final int DIM = 32;
    private static final int SIZE = 1_000;
    private static final VectorSimilarityFunction VSF = VectorSimilarityFunction.EUCLIDEAN;
    private static final String WORKER_PREFIX = "noescape-worker-";

    private static ForkJoinPool namedSingleThreadPool() {
        ForkJoinPool.ForkJoinWorkerThreadFactory factory = p -> {
            ForkJoinWorkerThread t = ForkJoinPool.defaultForkJoinWorkerThreadFactory.newThread(p);
            t.setName(WORKER_PREFIX + t.getPoolIndex());
            return t;
        };
        return new ForkJoinPool(1, factory, null, false);
    }

    /// Every parallel body that reads vectors through the context must run on the supplied pool's
    /// named worker (or the calling thread) — proving nothing escaped to the common pool or the
    /// physical-core pool.
    @Test
    public void noParallelWorkEscapesTheSuppliedPool() throws Exception {
        List<VectorFloat<?>> vectors = TestUtil.createRandomVectors(SIZE, DIM);
        Set<String> observed = ConcurrentHashMap.newKeySet();
        RecordingRAVV ravv = new RecordingRAVV(new ListRandomAccessVectorValues(vectors, DIM), observed);
        // A separate, non-recording view for constructing inputs whose reads must NOT be attributed
        // to the context (they run outside it).
        ListRandomAccessVectorValues plain = new ListRandomAccessVectorValues(vectors, DIM);

        ForkJoinPool pool = namedSingleThreadPool();
        try {
            var ctx = EmbeddedExecutionContext.of(pool);

            // build + cleanup (reads ravv via the build score provider and the build supplier)
            try (var builder = ctx.newBuilder(BuildScoreProvider.randomAccessScoreProvider(ravv, VSF),
                    DIM, 16, 100, 1.2f, 1.2f, true, true)) {
                builder.build(ravv);
            }

            // PQ train / refine / encode (all read ravv)
            ProductQuantization pq = ctx.trainPQ(ravv, 8, 256, true);
            ctx.refinePQ(pq, ravv);
            ctx.encodePQ(pq, ravv);

            // NVQ encode (the quantization is trained off the plain view; only the encode reads ravv)
            NVQuantization nvq = NVQuantization.compute(plain, 2);
            ctx.encodeNVQ(nvq, ravv);
        } finally {
            pool.shutdown();
        }

        assertFalse("expected some parallel work to have run on the supplied pool", observed.isEmpty());
        boolean sawPoolWorker = false;
        String mainName = Thread.currentThread().getName();
        for (String name : observed) {
            boolean onPool = name.startsWith(WORKER_PREFIX);
            sawPoolWorker |= onPool;
            assertTrue("work escaped the supplied pool to thread: " + name + " (observed=" + observed + ")",
                    onPool || name.equals(mainName));
        }
        assertTrue("no work actually ran on the supplied pool worker (observed=" + observed + ")", sawPoolWorker);
    }

    /// End-to-end through the context: build two FusedPQ source graphs, then merge them with a
    /// context-created compactor (which retrains PQ internally on the context's pool). Asserts the
    /// merged graph is complete and searchable — exercising newBuilder, trainPQ, encodePQ,
    /// newCompactor, and the (leak-closed) internal PQ retrain path.
    @Test
    public void graphBuildAndFusedCompactionThroughContext() throws Exception {
        Path dir = Files.createTempDirectory("jvector_ctx_test");
        ForkJoinPool pool = namedSingleThreadPool();
        try {
            var ctx = EmbeddedExecutionContext.of(pool);
            int perSource = 300;

            List<VectorFloat<?>> v0 = TestUtil.createRandomVectors(perSource, DIM);
            List<VectorFloat<?>> v1 = TestUtil.createRandomVectors(perSource, DIM);
            Path s0 = buildFusedSource(ctx, dir.resolve("s0.graph"), v0);
            Path s1 = buildFusedSource(ctx, dir.resolve("s1.graph"), v1);

            try (ReaderSupplier r0 = ReaderSupplierFactory.open(s0);
                 ReaderSupplier r1 = ReaderSupplierFactory.open(s1)) {
                OnDiskGraphIndex g0 = OnDiskGraphIndex.load(r0);
                OnDiskGraphIndex g1 = OnDiskGraphIndex.load(r1);

                FixedBitSet live0 = new FixedBitSet(perSource);
                live0.set(0, perSource);
                FixedBitSet live1 = new FixedBitSet(perSource);
                live1.set(0, perSource);
                Map<Integer, Integer> map0 = new HashMap<>();
                Map<Integer, Integer> map1 = new HashMap<>();
                for (int i = 0; i < perSource; i++) {
                    map0.put(i, i);
                    map1.put(i, perSource + i);
                }

                OnDiskGraphIndexCompactor compactor = ctx.newCompactor(
                        List.of(g0, g1), List.of(live0, live1),
                        List.of(new OrdinalMapper.MapMapper(map0), new OrdinalMapper.MapMapper(map1)),
                        VSF, -1);
                Path out = dir.resolve("merged.graph");
                compactor.compact(out);

                try (ReaderSupplier rOut = ReaderSupplierFactory.open(out)) {
                    OnDiskGraphIndex merged = OnDiskGraphIndex.load(rOut);
                    assertEquals("merged graph must contain all live nodes", 2 * perSource, merged.size(0));
                    // sanity: the merged graph is searchable and returns the exact match for a source vector
                    List<VectorFloat<?>> all = new ArrayList<>(v0);
                    all.addAll(v1);
                    var mergedRavv = new ListRandomAccessVectorValues(all, DIM);
                    int hits = 0;
                    for (int q = 0; q < 50; q++) {
                        var res = GraphSearcher.search(all.get(q), 50, mergedRavv, VSF, merged, Bits.ALL);
                        for (var ns : res.getNodes()) {
                            if (ns.node == q) { hits++; break; }
                        }
                    }
                    assertTrue("merged graph should find self for most source vectors, got " + hits + "/50", hits >= 45);
                }
            }
        } finally {
            pool.shutdown();
            TestUtil.deleteQuietly(dir);
        }
    }

    /// Builds a single-layer graph with INLINE_VECTORS + FUSED_PQ, using the context for PQ, and
    /// writes it. Mirrors the fused-source shape the compactor requires.
    private Path buildFusedSource(EmbeddedExecutionContext ctx, Path out, List<VectorFloat<?>> vecs) throws IOException {
        var ravv = new ListRandomAccessVectorValues(vecs, DIM);
        ProductQuantization pq = ctx.trainPQ(ravv, 8, 256, true);
        PQVectors pqv = ctx.encodePQ(pq, ravv);
        var bsp = BuildScoreProvider.pqBuildScoreProvider(VSF, pqv);
        try (var builder = ctx.newBuilder(bsp, DIM, 16, 100, 1.2f, 1.2f, true, true)) {
            var graph = builder.getGraph();
            var writerBuilder = new OnDiskGraphIndexWriter.Builder(graph, out)
                    .withMapper(new OrdinalMapper.IdentityMapper(vecs.size() - 1))
                    .with(new InlineVectors(DIM))
                    .with(new FusedPQ(graph.maxDegree(), pq));
            var writer = writerBuilder.build();
            Map<FeatureId, IntFunction<Feature.State>> suppliers = new EnumMap<>(FeatureId.class);
            suppliers.put(FeatureId.INLINE_VECTORS, ord -> new InlineVectors.State(ravv.getVector(ord)));
            for (int node = 0; node < ravv.size(); node++) {
                var stateMap = new EnumMap<FeatureId, Feature.State>(FeatureId.class);
                stateMap.put(FeatureId.INLINE_VECTORS, new InlineVectors.State(ravv.getVector(node)));
                writer.writeInline(node, stateMap);
                builder.addGraphNode(node, ravv.getVector(node));
            }
            builder.cleanup();
            suppliers.put(FeatureId.FUSED_PQ, ord -> new FusedPQ.State(graph.getView(), pqv, ord));
            writer.write(suppliers);
        }
        return out;
    }

    /// A {@link RandomAccessVectorValues} that records the thread of every {@link #getVector} call,
    /// funnelling all thread-local copies through the same shared set, then delegates.
    private static final class RecordingRAVV implements RandomAccessVectorValues {
        private final RandomAccessVectorValues delegate;
        private final Set<String> observed;

        RecordingRAVV(RandomAccessVectorValues delegate, Set<String> observed) {
            this.delegate = delegate;
            this.observed = observed;
        }

        @Override
        public int size() {
            return delegate.size();
        }

        @Override
        public int dimension() {
            return delegate.dimension();
        }

        @Override
        public VectorFloat<?> getVector(int nodeId) {
            observed.add(Thread.currentThread().getName());
            return delegate.getVector(nodeId);
        }

        @Override
        public boolean isValueShared() {
            return delegate.isValueShared();
        }

        @Override
        public RandomAccessVectorValues copy() {
            return new RecordingRAVV(delegate.copy(), observed);
        }
    }
}
