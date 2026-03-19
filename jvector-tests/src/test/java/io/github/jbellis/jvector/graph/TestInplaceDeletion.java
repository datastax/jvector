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
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.graph.TestVectorGraph.createRandomFloatVectors;
import static org.junit.Assert.*;

/*
* These 3 tests are the acceptance criteria for POC:
* 1. Single-threaded sequential deletes: recall@10 must not degrade > 3% after 1k deletes on a 10K-vector index
* 2. Two-thread concurrent delete + insert : no exceptions, no deadlock. Validates that CAS-based ConcurrentNeighborMap is sufficient for concurrency - no external locking protocol needed.
* 3. Entry point deletion: after deleting the current entry point, the graph must update to a live node and search must still work.
*
* All tests run with addHierarchy = false (flat Vamana) and addHierarchy = true (hierarchical) to ensure correctness across both graph modes.
* Graph parameters match the paper's high-recall regime: dimension = 128, cosine, m = 16, efConstruction = 200*/
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestInplaceDeletion extends LuceneTestCase {
    private static final int DIMENSION = 128;
    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.COSINE;

    // SIFT dataset paths
    private static final String SIFTSMALL_BASE       = "/Users/kartikeysrivastava/Desktop/projects/dataset/siftsmall/siftsmall_base.fvecs";
    private static final String SIFTSMALL_QUERY      = "/Users/kartikeysrivastava/Desktop/projects/dataset/siftsmall/siftsmall_query.fvecs";
    private static final String SIFTSMALL_GROUNDTRUTH = "/Users/kartikeysrivastava/Desktop/projects/dataset/siftsmall/siftsmall_groundtruth.ivecs";
    // SIFT ground truth is L2 — must use EUCLIDEAN when testing against it
    private static final VectorSimilarityFunction SIFT_SIMILARITY = VectorSimilarityFunction.EUCLIDEAN;

    // high-recall regime
    // alpha = 1.2f (vamana diversity rule), neighborOverflow = 1.5f
    private static final int M = 16;
    private static final int EF_CONSTRUCTION = 200;
    private static final int EF_SEARCH = 100;  // beam width at query time; topK=10 alone gives ~56% on 1M
    private static final float ALPHA = 1.2f;
    private static final float NEIGHBOR_OVERFLOW = 1.5f;
    // ============================================================
    // SIFT-based tests (real vectors, precomputed ground truth)
    // ============================================================

    @Test
    public void testRecallDegradationSift() {
        testRecallDegradationSift(false);
        testRecallDegradationSift(true);
    }

    private void testRecallDegradationSift(boolean addHierarchy) {
        int deleteCount = 1_000;   // 10% of 10K
        int batchSize   = 100;     // rolling recall checkpoint every 100 deletes
        int queryCount  = 100;
        int topK        = 10;

        System.out.println("\n=== SIFT TEST 1: testRecallDegradationSift addHierarchy=" + addHierarchy + " ===");

        var baseVectorsList = SiftLoader.readFvecs(SIFTSMALL_BASE);
        var queryVectors    = SiftLoader.readFvecs(SIFTSMALL_QUERY).subList(0, queryCount);
        var groundTruth     = SiftLoader.readIvecs(SIFTSMALL_GROUNDTRUTH);

        int indexSize = baseVectorsList.size();
        System.out.println("[setup] indexSize=" + indexSize + " deleteCount=" + deleteCount
                + " batchSize=" + batchSize + " queryCount=" + queryCount + " topK=" + topK
                + " M=" + M + " efConstruction=" + EF_CONSTRUCTION + " alpha=" + ALPHA);

        var ravv    = MockVectorValues.fromValues(baseVectorsList.toArray(new VectorFloat[0]));
        var builder = new GraphIndexBuilder(ravv, SIFT_SIMILARITY, M, EF_CONSTRUCTION, ALPHA, NEIGHBOR_OVERFLOW, addHierarchy);

        long buildStart = System.currentTimeMillis();
        var graph = builder.build(ravv);
        long buildMs = System.currentTimeMillis() - buildStart;
        System.out.println("[build] parallel build done in " + buildMs + "ms — graph.size(0)=" + graph.size(0));

        double baselineRecall = measureRecallSift(queryVectors, graph, ravv, groundTruth,
                Collections.emptySet(), topK);
        System.out.println("[baseline] recall@" + topK + "=" + String.format("%.4f", baselineRecall));

        var allOrdinals = IntStream.range(0, indexSize)
                .boxed()
                .collect(Collectors.toCollection(ArrayList::new));
        Collections.shuffle(allOrdinals, getRandom());

        var deletedNodes = new HashSet<Integer>();
        long totalDeletionMs = 0;
        int numBatches = deleteCount / batchSize;

        for (int batch = 0; batch < numBatches; batch++) {
            int from = batch * batchSize;
            long batchT0 = System.currentTimeMillis();
            for (int i = from; i < from + batchSize; i++) {
                builder.markNodeDeleted(allOrdinals.get(i));
                deletedNodes.add(allOrdinals.get(i));
            }
            long batchMs = System.currentTimeMillis() - batchT0;
            totalDeletionMs += batchMs;

            double rollingRecall = measureRecallSift(queryVectors, graph, ravv, groundTruth,
                    deletedNodes, topK);
            System.out.printf("[batch %2d/%d] deleted=%6d  batchTime=%5dms  avgPerDelete=%.2fms  recall@%d=%.4f  degradation=%.2f%%%n",
                    batch + 1, numBatches, deletedNodes.size(),
                    batchMs, (double) batchMs / batchSize,
                    topK, rollingRecall, (baselineRecall - rollingRecall) * 100);
        }

        System.out.printf("[deletion summary] totalDeleted=%d  totalTime=%dms  avgPerDelete=%.2fms%n",
                deletedNodes.size(), totalDeletionMs, (double) totalDeletionMs / deleteCount);

        // final recall with verbose query-0 hit/miss detail
        double postRecall = measureRecallSiftVerbose(queryVectors, graph, ravv, groundTruth,
                deletedNodes, topK);
        double degradation = baselineRecall - postRecall;
        System.out.println("[result] baseline=" + String.format("%.4f", baselineRecall)
                + " post=" + String.format("%.4f", postRecall)
                + " degradation=" + String.format("%.2f%%", degradation * 100)
                + " threshold=3.00%  PASS=" + (degradation <= 0.03));

        assertTrue(
                String.format(
                        "Recall degraded by %.1f%% (baseline=%.3f, post=%.3f) — exceeds 3%% threshold. "
                                + "addHierarchy=%b.",
                        degradation * 100, baselineRecall, postRecall, addHierarchy),
                degradation <= 0.03);
    }

    @Test
    public void testConcurrentDeleteAndInsertSift() throws InterruptedException {
        testConcurrentDeleteAndInsertSift(false);
        testConcurrentDeleteAndInsertSift(true);
    }

    private void testConcurrentDeleteAndInsertSift(boolean addHierarchy) throws InterruptedException {
        int initialSize  = 10_000;
        int opsPerThread = 500;

        System.out.println("\n=== SIFT TEST 2: testConcurrentDeleteAndInsertSift addHierarchy=" + addHierarchy + " ===");
        System.out.println("[setup] initialSize=" + initialSize + " opsPerThread=" + opsPerThread
                + " deleteRange=[0," + (opsPerThread - 1) + "] insertRange=random");

        // Build a combined ravv: first 1K SIFT vectors + 500 random insert vectors.
        // MockVectorValues must cover ALL ordinals that JVector will look up during
        // addGraphNode/repairDeletion — including the new ordinals 1000..1499.
        var siftBase      = SiftLoader.readFvecs(SIFTSMALL_BASE).subList(0, initialSize);
        var insertVectors = createRandomFloatVectors(opsPerThread, DIMENSION, getRandom());

        var combined = new VectorFloat[initialSize + opsPerThread];
        for (int i = 0; i < initialSize; i++)  combined[i]              = siftBase.get(i);
        for (int i = 0; i < opsPerThread; i++) combined[initialSize + i] = insertVectors[i];

        var ravv = MockVectorValues.fromValues(combined);

        var builder = new GraphIndexBuilder(ravv, SIFT_SIMILARITY, M, EF_CONSTRUCTION, ALPHA, NEIGHBOR_OVERFLOW, addHierarchy);
        for (int i = 0; i < initialSize; i++) {
            builder.addGraphNode(i, ravv.getVector(i));
        }
        builder.cleanup();
        var graph = builder.getGraph();
        System.out.println("[setup] initial index built — graph.size(0)=" + graph.size(0));

        var errors      = new AtomicReference<Throwable>(null);
        var startLatch  = new CountDownLatch(1);

        var deleteThread = new Thread(() -> {
            try {
                startLatch.await();
                System.out.println("[delete-thread] STARTED");
                for (int i = 0; i < opsPerThread; i++) {
                    builder.markNodeDeleted(i);
                }
                System.out.println("[delete-thread] DONE — " + opsPerThread + " deletions completed without exception");
            } catch (Throwable t) {
                System.out.println("[delete-thread] EXCEPTION: " + t);
                t.printStackTrace();
                errors.compareAndSet(null, t);
            }
        }, "delete-thread");

        var insertThread = new Thread(() -> {
            try {
                startLatch.await();
                System.out.println("[insert-thread] STARTED");
                for (int i = 0; i < opsPerThread; i++) {
                    // ordinals initialSize..initialSize+499 are fresh — no collision with deleted range
                    builder.addGraphNode(initialSize + i, insertVectors[i]);
                }
                System.out.println("[insert-thread] DONE — " + opsPerThread + " inserts completed without exception");
            } catch (Throwable t) {
                System.out.println("[insert-thread] EXCEPTION: " + t);
                t.printStackTrace();
                errors.compareAndSet(null, t);
            }
        }, "insert-thread");

        deleteThread.start();
        insertThread.start();
        startLatch.countDown();

        deleteThread.join(10_000);
        insertThread.join(10_000);

        System.out.println("[result] delete-thread alive=" + deleteThread.isAlive()
                + " insert-thread alive=" + insertThread.isAlive()
                + " exception=" + errors.get());

        assertFalse("delete-thread still running after 10s — likely deadlocked.", deleteThread.isAlive());
        assertFalse("insert-thread still running after 10s — likely deadlocked.", insertThread.isAlive());
        assertNull("Exception thrown during concurrent operation: " + errors.get(), errors.get());

        var queryVec = SiftLoader.readFvecs(SIFTSMALL_QUERY).get(0);
        var results  = GraphSearcher.search(queryVec, 5, ravv, SIFT_SIMILARITY, graph, Bits.ALL);
        var sb = new StringBuilder();
        if (results != null) {
            for (var ns : results.getNodes())
                sb.append(ns.node).append("(").append(String.format("%.4f", ns.score)).append(") ");
        }
        System.out.println("[smoke-check] results=" + (results == null ? "null" : "[" + sb.toString().trim() + "]"));
        assertNotNull("Search returned null after concurrent delete+insert", results);
        assertTrue("Search returned no results — graph may be corrupted", results.getNodes().length > 0);
    }

    @Test
    public void testEntryPointDeletionSift() {
        testEntryPointDeletionSift(false);
        testEntryPointDeletionSift(true);
    }

    private void testEntryPointDeletionSift(boolean addHierarchy) {
        int indexSize = 100;

        System.out.println("\n=== SIFT TEST 3: testEntryPointDeletionSift addHierarchy=" + addHierarchy + " ===");

        var baseVectorsList = SiftLoader.readFvecs(SIFTSMALL_BASE).subList(0, indexSize);
        var ravv            = MockVectorValues.fromValues(baseVectorsList.toArray(new VectorFloat[0]));
        var builder         = new GraphIndexBuilder(ravv, SIFT_SIMILARITY, M, EF_CONSTRUCTION, ALPHA, NEIGHBOR_OVERFLOW, addHierarchy);

        long buildStart = System.currentTimeMillis();
        var graph = builder.build(ravv);
        long buildMs = System.currentTimeMillis() - buildStart;
        System.out.println("[build] parallel build done in " + buildMs + "ms — graph.size(0)=" + graph.size(0));

        var originalEntry = graph.getView().entryNode();
        assertNotNull("Graph must have a valid entry point before any deletion", originalEntry);
        int originalEntryNode = originalEntry.node;
        System.out.println("[entry-point] before deletion: node=" + originalEntryNode + " level=" + originalEntry.level);

        long deleteStart = System.currentTimeMillis();
        builder.markNodeDeleted(originalEntryNode);
        long deleteMs = System.currentTimeMillis() - deleteStart;
        System.out.println("[delete] entry point deletion took " + deleteMs + "ms");

        var newEntry = graph.getView().entryNode();
        System.out.println("[entry-point] after deletion: "
                + (newEntry == null ? "null" : "node=" + newEntry.node + " level=" + newEntry.level)
                + " — changed=" + (newEntry == null || newEntry.node != originalEntryNode));

        assertNotNull("Entry point must not be null after deleting old entry point", newEntry);
        assertNotEquals(
                "Entry point must change after deleting node " + originalEntryNode,
                originalEntryNode, newEntry.node);

        var queryVectors = SiftLoader.readFvecs(SIFTSMALL_QUERY).subList(0, 20);
        for (int i = 0; i < queryVectors.size(); i++) {
            var queryVec = queryVectors.get(i);
            var results  = GraphSearcher.search(queryVec, 5, ravv, SIFT_SIMILARITY, graph, Bits.ALL);

            assertNotNull("Search returned null after entry point deletion (query " + i + ")", results);

            var resultNodes = new StringBuilder();
            for (var ns : results.getNodes()) {
                resultNodes.append(ns.node).append("(").append(String.format("%.4f", ns.score)).append(") ");
                assertNotEquals(
                        "Deleted entry point node " + originalEntryNode + " must not appear in results",
                        originalEntryNode, ns.node);
            }
            if (i == 0) {
                System.out.println("[search query 0] results=[" + resultNodes.toString().trim()
                        + "] — deleted node=" + originalEntryNode + " absent=PASS");
            }
        }
        System.out.println("[search] all 20 queries passed — deleted entry point never returned");
    }

    /**
     * Algorithm 6 correctness: after calling consolidateDanglingEdges(), no live node
     * at any level may hold an out-edge pointing to a node that is structurally absent
     * from that level.
     * <p>
     * We disable auto-trigger (threshold=1.0) so that the sweep only runs when we
     * explicitly invoke it, giving us full control over the before/after observation.
     */
    @Test
    public void testConsolidateDanglingEdges() {
        testConsolidateDanglingEdges(false);
        testConsolidateDanglingEdges(true);
    }

    private void testConsolidateDanglingEdges(boolean addHierarchy) {
        int indexSize   = 500;
        int deleteCount = 100;   // 20% deletions — well above default 20% threshold

        System.out.println("\n=== testConsolidateDanglingEdges addHierarchy=" + addHierarchy + " ===");

        var vectors = createRandomFloatVectors(indexSize, DIMENSION, getRandom());
        var ravv    = MockVectorValues.fromValues(vectors);
        var builder = new GraphIndexBuilder(ravv, SIMILARITY, M, EF_CONSTRUCTION, ALPHA, NEIGHBOR_OVERFLOW, addHierarchy);
        var graph   = builder.build(ravv);

        // Disable auto-trigger so we control exactly when Algorithm 6 fires.
        builder.setConsolidationThreshold(1.0);

        // Delete nodes sequentially. Algorithm 5 runs for each one but Algorithm 6 does not.
        var allOrdinals = IntStream.range(0, indexSize)
                .boxed()
                .collect(Collectors.toCollection(ArrayList::new));
        Collections.shuffle(allOrdinals, getRandom());

        for (int i = 0; i < deleteCount; i++) {
            builder.markNodeDeleted(allOrdinals.get(i));
        }

        // Scan for dangling edges BEFORE consolidation — informational only.
        long danglingBefore = countDanglingEdges((OnHeapGraphIndex) graph);
        System.out.println("[before consolidation] danglingEdges=" + danglingBefore);

        // Run Algorithm 6.
        builder.consolidateDanglingEdges();

        // Every out-edge of every live node must now point to a structurally present node.
        long danglingAfter = countDanglingEdges((OnHeapGraphIndex) graph);
        System.out.println("[after  consolidation] danglingEdges=" + danglingAfter);

        assertEquals(
                "consolidateDanglingEdges() must leave zero dangling edges at all levels. addHierarchy=" + addHierarchy,
                0L, danglingAfter);
    }

    /**
     * Counts out-edges across all levels that point to a structurally absent neighbor node.
     */
    private long countDanglingEdges(OnHeapGraphIndex graph) {
        long dangling = 0;
        var view = graph.getView();
        int maxLevel = graph.getMaxLevel();
        for (int level = 0; level <= maxLevel; level++) {
            var nodeIt = graph.nodeStream(level).iterator();
            while (nodeIt.hasNext()) {
                int node = nodeIt.nextInt();
                var it = view.getNeighborsIterator(level, node);
                while (it.hasNext()) {
                    int neighbor = it.nextInt();
                    if (!view.contains(level, neighbor)) dangling++;
                }
            }
        }
        return dangling;
    }

    /**
     * Measures recall using precomputed SIFT ground truth.
     * Deleted ordinals are stripped from the GT list before scoring so they
     * don't count as false misses.
     */
    private double measureRecallSift(List<? extends VectorFloat<?>> queries,
                                     ImmutableGraphIndex graph,
                                     RandomAccessVectorValues ravv,
                                     List<List<Integer>> groundTruth,
                                     Set<Integer> deletedNodes,
                                     int topK) {
        double totalRecall = 0.0;
        for (int q = 0; q < queries.size(); q++) {
            var query = queries.get(q);
            // filter deleted ordinals from GT, then take topK
            List<Integer> filteredGT = groundTruth.get(q).stream()
                    .filter(id -> !deletedNodes.contains(id))
                    .limit(topK)
                    .collect(Collectors.toList());
            Set<Integer> gtSet = new HashSet<>(filteredGT);

            var results = GraphSearcher.search(query, topK, EF_SEARCH, ravv, SIFT_SIMILARITY, graph, Bits.ALL);
            int hits = 0;
            for (var ns : results.getNodes()) {
                if (gtSet.contains(ns.node)) hits++;
            }
            totalRecall += (double) hits / Math.max(1, gtSet.size());
        }
        return totalRecall / queries.size();
    }

    /** Same as measureRecallSift but prints query 0 hit/miss detail. */
    private double measureRecallSiftVerbose(List<? extends VectorFloat<?>> queries,
                                            ImmutableGraphIndex graph,
                                            RandomAccessVectorValues ravv,
                                            List<List<Integer>> groundTruth,
                                            Set<Integer> deletedNodes,
                                            int topK) {
        double totalRecall = 0.0;
        for (int q = 0; q < queries.size(); q++) {
            var query = queries.get(q);
            List<Integer> filteredGT = groundTruth.get(q).stream()
                    .filter(id -> !deletedNodes.contains(id))
                    .limit(topK)
                    .collect(Collectors.toList());
            Set<Integer> gtSet = new HashSet<>(filteredGT);

            var results = GraphSearcher.search(query, topK, EF_SEARCH, ravv, SIFT_SIMILARITY, graph, Bits.ALL);
            int hits = 0;
            var graphNodes = new StringBuilder();
            for (var ns : results.getNodes()) {
                boolean isHit = gtSet.contains(ns.node);
                if (isHit) hits++;
                graphNodes.append(ns.node).append(isHit ? "[HIT]" : "[miss]")
                        .append("(").append(String.format("%.4f", ns.score)).append(") ");
            }
            double queryRecall = (double) hits / Math.max(1, gtSet.size());
            totalRecall += queryRecall;
            if (q == 0) {
                System.out.println("  [query 0] hits=" + hits + "/" + gtSet.size()
                        + " filteredGT=" + filteredGT
                        + " graphResults=[" + graphNodes.toString().trim() + "]");
            }
        }
        return totalRecall / queries.size();
    }


}
