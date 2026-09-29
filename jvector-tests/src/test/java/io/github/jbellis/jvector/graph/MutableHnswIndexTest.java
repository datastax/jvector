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
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.index.Index;
import io.github.jbellis.jvector.index.Indexes;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests {@link MutableHnswIndex} and {@link HnswIndexBuilder#buildMutable()}: incremental and
 * concurrent insertion, searching during construction, deletes, and mid-build rescoring, i.e. the
 * {@link GraphIndexBuilder} usage patterns of Cassandra's memtable index and compaction and of
 * OpenSearch's segment merge.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class MutableHnswIndexTest extends RandomizedTest {
    private static final int DIMENSION = 16;
    private static final VectorSimilarityFunction VSF = VectorSimilarityFunction.EUCLIDEAN;

    private static HnswIndexBuilder configured() {
        return Indexes.hnswBuilder()
                .withMaxDegree(16)
                .withBeamWidth(50)
                .withNeighborOverflow(1.2f)
                .withAlpha(1.2f)
                .withAddHierarchy(true);
    }

    /** Fraction of nodes whose own vector finds them as the top search result. */
    private static double selfRecall(GraphIndex graph, RandomAccessVectorValues ravv, Iterable<Integer> ordinals) {
        int found = 0;
        int total = 0;
        try (GraphSearcher searcher = graph.searcher()) {
            for (int ord : ordinals) {
                var ssp = DefaultSearchScoreProvider.exact(ravv.getVector(ord), VSF, ravv);
                SearchResult result = searcher.search(ssp, 1, 10, 0.0f, 0.0f, Bits.ALL);
                if (result.getNodes().length > 0 && result.getNodes()[0].node == ord) {
                    found++;
                }
                total++;
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return found / (double) total;
    }

    private static Iterable<Integer> range(int from, int to) {
        return () -> IntStream.range(from, to).iterator();
    }

    @Test
    public void incrementalBuildWithScoreProviderAndNoVectorValues() throws Exception {
        int n = 2_000;
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(n, DIMENSION), DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);

        // CompactionGraph's shape: a score provider and a dimension, vectors supplied one at a time.
        try (MutableHnswIndex index = configured().withScoreProvider(bsp).withDimension(DIMENSION).buildMutable()) {
            assertEquals(0, index.graph().size(0));
            IntStream.range(0, n).parallel().forEach(i -> index.addNode(i, ravv.getVector(i)));
            index.cleanup();

            assertEquals(n, index.graph().size(0));
            assertTrue(index.graph().isHierarchical());
            assertEquals(DIMENSION, index.graph().getDimension());
            assertTrue(selfRecall(index.graph(), ravv, range(0, n)) > 0.95);

            // It is an Index too, and its graph is persistable without a cast.
            Index generic = index;
            assertTrue(generic.ramBytesUsed() > 0);
            PersistableGraphIndex persistable = index.graph();
            assertTrue(persistable instanceof OnHeapGraphIndex);
        }
    }

    @Test
    public void incrementalBuildWithSimilarityFunction() throws Exception {
        int n = 1_000;
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(n, DIMENSION), DIMENSION);

        // CassandraOnHeapGraph's shape: vector values + similarity function, but nothing is inserted
        // until the caller adds it.
        try (MutableHnswIndex index = configured().withVectorValues(ravv).withSimilarityFunction(VSF).buildMutable()) {
            assertEquals(0, index.graph().size(0));
            for (int i = 0; i < n; i++) {
                assertTrue(index.addNode(i, ravv.getVector(i)) > 0);
            }
            index.cleanup();
            assertEquals(n, index.graph().size(0));
            assertTrue(selfRecall(index.graph(), ravv, range(0, n)) > 0.95);
        }
    }

    @Test
    public void searchesAndCleanupRunConcurrentlyWithInserts() throws Exception {
        int n = 5_000;
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(n, DIMENSION), DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        ExecutorService pool = Executors.newFixedThreadPool(6);
        try (MutableHnswIndex index = configured().withScoreProvider(bsp).withDimension(DIMENSION).buildMutable()) {
            AtomicInteger next = new AtomicInteger();
            AtomicBoolean done = new AtomicBoolean();

            // Inserters, searchers and a periodic cleanup, with no external locking: the handle
            // must serialize cleanup against inserts itself.
            List<Future<?>> inserters = IntStream.range(0, 4).mapToObj(t -> pool.submit(() -> {
                int i;
                while ((i = next.getAndIncrement()) < n) {
                    index.addNode(i, ravv.getVector(i));
                }
            })).collect(java.util.stream.Collectors.toList());
            Future<?> searcher = pool.submit(() -> {
                while (!done.get()) {
                    try (GraphSearcher s = index.searcher()) {
                        var q = ravv.getVector(getRandom().nextInt(n));
                        s.search(DefaultSearchScoreProvider.exact(q, VSF, ravv), 10, Bits.ALL);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }
            });
            Future<?> cleaner = pool.submit(() -> {
                while (!done.get()) {
                    index.cleanup();
                    Thread.yield();
                }
            });

            for (Future<?> f : inserters) {
                f.get();
            }
            done.set(true);
            searcher.get();
            cleaner.get();

            index.cleanup();
            assertEquals(n, index.graph().size(0));
            assertEquals(0, index.insertsInProgress());
            assertTrue(selfRecall(index.graph(), ravv, range(0, n)) > 0.95);
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    public void deletedNodesAreHiddenAndThenRemoved() throws Exception {
        int n = 1_000;
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(n, DIMENSION), DIMENSION);
        try (MutableHnswIndex index = configured().withVectorValues(ravv).withSimilarityFunction(VSF).buildMutable()) {
            IntStream.range(0, n).parallel().forEach(i -> index.addNode(i, ravv.getVector(i)));
            IntStream.range(0, n).filter(i -> i % 10 == 0).parallel().forEach(index::markDeleted);

            // hidden from search immediately
            try (GraphSearcher s = index.searcher()) {
                for (int i = 0; i < n; i += 10) {
                    var result = s.search(DefaultSearchScoreProvider.exact(ravv.getVector(i), VSF, ravv), 5, Bits.ALL);
                    for (var ns : result.getNodes()) {
                        assertTrue("deleted node " + ns.node + " returned", ns.node % 10 != 0);
                    }
                }
            }

            // and physically removed by cleanup
            index.cleanup();
            assertEquals(n - n / 10, index.graph().size(0));
            for (int i = 0; i < n; i += 10) {
                assertFalse(index.graph().containsNode(i));
            }
            assertTrue(selfRecall(index.graph(), ravv, () -> IntStream.range(0, n).filter(i -> i % 10 != 0).iterator()) > 0.95);
        }
    }

    @Test
    public void rescoreRunsSupplierWithInsertsLockedOutAndKeepsEdges() throws Exception {
        int n = 4_000;
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(n, DIMENSION), DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        ExecutorService pool = Executors.newFixedThreadPool(4);
        try (MutableHnswIndex index = configured().withScoreProvider(bsp).withDimension(DIMENSION).buildMutable()) {
            AtomicInteger next = new AtomicInteger();
            List<Future<?>> inserters = IntStream.range(0, 4).mapToObj(t -> pool.submit(() -> {
                int i;
                while ((i = next.getAndIncrement()) < n) {
                    index.addNode(i, ravv.getVector(i));
                }
            })).collect(java.util.stream.Collectors.toList());

            // Rescore partway through, the way CompactionGraph swaps in a refined PQ codebook.
            while (next.get() < n / 2) {
                Thread.yield();
            }
            GraphIndex before = index.graph();
            AtomicInteger insertsDuringSupplier = new AtomicInteger(-1);
            index.rescore(() -> {
                insertsDuringSupplier.set(index.insertsInProgress());
                return BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
            });
            assertEquals("supplier must run with inserts locked out", 0, insertsDuringSupplier.get());
            assertNotSame(before, index.graph());

            for (Future<?> f : inserters) {
                f.get();
            }
            index.cleanup();
            assertEquals(n, index.graph().size(0));
            assertTrue(selfRecall(index.graph(), ravv, range(0, n)) > 0.95);
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    public void exclusiveOperationsDoNotDeadlockWhenInsertsRunOnTheBuildersOwnPool() throws Exception {
        // Inserts run on the same small ForkJoinPool the builder uses for cleanup/rescore work, and the
        // rescore supplier also runs parallel work on it. If inserting workers parked on the lock
        // without telling the pool, nothing would be left to run that work.
        int n = 3_000;
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(n, DIMENSION), DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        var pool = new java.util.concurrent.ForkJoinPool(2);
        ExecutorService driver = Executors.newSingleThreadExecutor();
        try (MutableHnswIndex index = configured()
                .withScoreProvider(bsp)
                .withDimension(DIMENSION)
                .withSimdExecutor(pool)
                .withParallelExecutor(pool)
                .buildMutable()) {
            Future<?> run = driver.submit(() -> {
                var inserts = new java.util.ArrayList<java.util.concurrent.ForkJoinTask<?>>();
                for (int i = 0; i < n; i++) {
                    int ord = i;
                    inserts.add(pool.submit(() -> index.addNode(ord, ravv.getVector(ord))));
                    if (i == n / 3) {
                        index.rescore(() -> {
                            pool.submit(() -> IntStream.range(0, 1_000).parallel().sum()).join();
                            return BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
                        });
                    }
                    if (i == 2 * n / 3) {
                        index.cleanup();
                    }
                }
                inserts.forEach(java.util.concurrent.ForkJoinTask::join);
                index.cleanup();
            });
            try {
                run.get(120, java.util.concurrent.TimeUnit.SECONDS);
            } catch (java.util.concurrent.TimeoutException e) {
                fail("deadlocked: inserts on the builder's pool starved rescore/cleanup");
            }
            assertEquals(n, index.graph().size(0));
            assertTrue(selfRecall(index.graph(), ravv, range(0, n)) > 0.95);
        } finally {
            driver.shutdownNow();
            pool.shutdownNow();
        }
    }

    @Test
    public void buildMutableOnAnExistingGraph() throws Exception {
        int n = 2_000;
        int base = 1_500;
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(n, DIMENSION), DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);

        MutableGraphIndex existing;
        try (MutableHnswIndex index = configured().withScoreProvider(bsp).withDimension(DIMENSION).buildMutable()) {
            IntStream.range(0, base).parallel().forEach(i -> index.addNode(i, ravv.getVector(i)));
            index.cleanup();
            existing = (MutableGraphIndex) index.graph();
        }

        // OpenSearch leading-segment merge: continue on the existing graph, add new nodes, delete some old ones.
        try (MutableHnswIndex index = Indexes.hnswBuilder()
                .withExistingGraph(existing)
                .withScoreProvider(bsp)
                .withDimension(DIMENSION)
                .withBeamWidth(50)
                .withNeighborOverflow(1.2f)
                .withAlpha(1.2f)
                .buildMutable()) {
            assertEquals(base, index.graph().size(0));
            IntStream.range(existing.getIdUpperBound(), n).parallel().forEach(i -> index.addNode(i, ravv.getVector(i)));
            index.markDeleted(0);
            index.cleanup();
            assertEquals(n - 1, index.graph().size(0));
            assertTrue(selfRecall(index.graph(), ravv, range(1, n)) > 0.95);
        }
    }

    @Test
    public void addingToACleanedUpGraphDoesNotCreateSelfEdges() throws Exception {
        // cleanup() freezes the graph (FrozenView, which does not hide incomplete nodes). Inserting
        // afterwards must unfreeze it, or an insert can find its own half-added node as a neighbor.
        int n = 3_000;
        int base = 1_000;
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(n, DIMENSION), DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        var baseRavv = new ListRandomAccessVectorValues(
                IntStream.range(0, base).mapToObj(ravv::getVector).collect(java.util.stream.Collectors.toList()), DIMENSION);
        // The graph keeps the diversity provider it was built with, so build it with a score provider
        // that already covers the ordinals that will be appended later.
        GraphIndex frozen = configured().withVectorValues(baseRavv).withScoreProvider(bsp).build();
        assertTrue(((OnHeapGraphIndex) frozen).allMutationsCompleted());

        try (MutableHnswIndex index = Indexes.hnswBuilder()
                .withExistingGraph((MutableGraphIndex) frozen)
                .withScoreProvider(bsp)
                .withDimension(DIMENSION)
                .withBeamWidth(50)
                .withNeighborOverflow(1.2f)
                .withAlpha(1.2f)
                .buildMutable()) {
            IntStream.range(base, n).parallel().forEach(i -> index.addNode(i, ravv.getVector(i)));
            assertFalse(((OnHeapGraphIndex) index.graph()).allMutationsCompleted());
            index.cleanup();

            var graph = index.graph();
            try (var view = graph.getView()) {
                for (int level = 0; level <= graph.getMaxLevel(); level++) {
                    for (var it = graph.getNodes(level); it.hasNext(); ) {
                        int node = it.nextInt();
                        for (var neighbors = view.getNeighborsIterator(level, node); neighbors.hasNext(); ) {
                            assertTrue("self edge at node " + node, neighbors.nextInt() != node);
                        }
                    }
                }
            }
            assertEquals(n, graph.size(0));
            assertTrue(selfRecall(graph, ravv, range(0, n)) > 0.95);
        }
    }

    @Test
    public void buildMutableValidation() {
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(4, DIMENSION), DIMENSION);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);

        // a score provider with neither vector values nor a dimension
        expectMissing(() -> configured().withScoreProvider(bsp).buildMutable(), "dimension");
        // a similarity function needs vector values to score against
        expectMissing(() -> configured().withSimilarityFunction(VSF).buildMutable(), "vectorValues");
        // everything else is still aggregated
        try {
            Indexes.hnswBuilder().buildMutable();
            fail("expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().startsWith("Cannot build MutableHnswIndex"));
            for (String name : List.of("similarityFunction", "maxDegree", "addHierarchy", "beamWidth", "neighborOverflow", "alpha")) {
                assertTrue(e.getMessage(), e.getMessage().contains(name));
            }
        }
        // a mismatched dimension is still rejected when vector values are present
        expectMissing(() -> configured().withVectorValues(ravv).withScoreProvider(bsp).withDimension(DIMENSION + 1).buildMutable(),
                "does not match");
        // build() still requires vector values even with a score provider and dimension
        expectMissing(() -> configured().withScoreProvider(bsp).withDimension(DIMENSION).build(), "vectorValues");
    }

    private static void expectMissing(Runnable r, String expected) {
        try {
            r.run();
            fail("expected IllegalStateException mentioning " + expected);
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains(expected));
        }
    }

    @Test
    public void buildStillProducesAFinishedGraph() throws Exception {
        int n = 1_000;
        List<VectorFloat<?>> vectors = createRandomVectors(n, DIMENSION);
        var ravv = new ListRandomAccessVectorValues(vectors, DIMENSION);
        GraphIndex graph = configured().withVectorValues(ravv).withSimilarityFunction(VSF).build();
        assertEquals(n, graph.size(0));
        assertTrue(((OnHeapGraphIndex) graph).allMutationsCompleted());
        assertTrue(selfRecall(graph, ravv, range(0, n)) > 0.95);
    }
}
