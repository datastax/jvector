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
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.graph.TestVectorGraph.createRandomFloatVectors;
import static org.junit.Assert.*;

/*
 * These 3 tests are the acceptance criteria for POC:
 * 1. Single-threaded sequential deletes: recall@10 must not degrade > 3% after 1k deletes on a 10K-vector index
 * 2. Entry point deletion: after deleting the current entry point, the graph must update to a live node and search must still work.
 * 3. Algorithm 6 correctness: after consolidateDanglingEdges(), no live node holds an out-edge to a structurally absent node.
 *
 * All tests run with addHierarchy = false (flat Vamana) and addHierarchy = true (hierarchical) to ensure correctness across both graph modes.
 * Graph parameters match the paper's high-recall regime: dimension = 128, cosine, m = 16, efConstruction = 200*/
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestInplaceDeletion extends LuceneTestCase {
    private static final int DIMENSION = 128;
    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.COSINE;

    // high-recall regime
    // alpha = 1.2f (vamana diversity rule), neighborOverflow = 1.5f
    private static final int M = 16;
    private static final int EF_CONSTRUCTION = 200;
    private static final int EF_SEARCH = 100;  // beam width at query time; topK=10 alone gives ~56% on 1M
    private static final float ALPHA = 1.2f;
    private static final float NEIGHBOR_OVERFLOW = 1.5f;

    @Test
    public void testRecallDegradation() {
        testRecallDegradation(false);
        testRecallDegradation(true);
    }

    private void testRecallDegradation(boolean addHierarchy) {
        int indexSize   = 10_000;
        int deleteCount = 1_000;   // 10% of 10K
        int batchSize   = 100;     // rolling recall checkpoint every 100 deletes
        int queryCount  = 100;
        int topK        = 10;

        System.out.println("\n=== TEST 1: testRecallDegradation addHierarchy=" + addHierarchy + " ===");

        var baseVectors  = createRandomFloatVectors(indexSize, DIMENSION, getRandom());
        var queryVectors = Arrays.asList(createRandomFloatVectors(queryCount, DIMENSION, getRandom()));

        System.out.println("[setup] indexSize=" + indexSize + " deleteCount=" + deleteCount
                + " batchSize=" + batchSize + " queryCount=" + queryCount + " topK=" + topK
                + " M=" + M + " efConstruction=" + EF_CONSTRUCTION + " alpha=" + ALPHA);

        var ravv    = MockVectorValues.fromValues(baseVectors);
        var builder = new GraphIndexBuilder(ravv, SIMILARITY, M, EF_CONSTRUCTION, ALPHA, NEIGHBOR_OVERFLOW, addHierarchy);

        long buildStart = System.currentTimeMillis();
        var graph = builder.build(ravv);
        long buildMs = System.currentTimeMillis() - buildStart;
        System.out.println("[build] parallel build done in " + buildMs + "ms — graph.size(0)=" + graph.size(0));

        double baselineRecall = measureRecallBrute(queryVectors, graph, ravv, Collections.emptySet(), topK);
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

            double rollingRecall = measureRecallBrute(queryVectors, graph, ravv, deletedNodes, topK);
            System.out.printf("[batch %2d/%d] deleted=%6d  batchTime=%5dms  avgPerDelete=%.2fms  recall@%d=%.4f  degradation=%.2f%%%n",
                    batch + 1, numBatches, deletedNodes.size(),
                    batchMs, (double) batchMs / batchSize,
                    topK, rollingRecall, (baselineRecall - rollingRecall) * 100);
        }

        System.out.printf("[deletion summary] totalDeleted=%d  totalTime=%dms  avgPerDelete=%.2fms%n",
                deletedNodes.size(), totalDeletionMs, (double) totalDeletionMs / deleteCount);

        double postRecall = measureRecallBruteVerbose(queryVectors, graph, ravv, deletedNodes, topK);
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
    public void testEntryPointDeletion() {
        testEntryPointDeletion(false);
        testEntryPointDeletion(true);
    }

    private void testEntryPointDeletion(boolean addHierarchy) {
        int indexSize = 100;

        System.out.println("\n=== TEST 2: testEntryPointDeletion addHierarchy=" + addHierarchy + " ===");

        var baseVectors = createRandomFloatVectors(indexSize, DIMENSION, getRandom());
        var ravv        = MockVectorValues.fromValues(baseVectors);
        var builder     = new GraphIndexBuilder(ravv, SIMILARITY, M, EF_CONSTRUCTION, ALPHA, NEIGHBOR_OVERFLOW, addHierarchy);

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

        var queryVectors = Arrays.asList(createRandomFloatVectors(20, DIMENSION, getRandom()));
        for (int i = 0; i < queryVectors.size(); i++) {
            var queryVec = queryVectors.get(i);
            var results  = GraphSearcher.search(queryVec, 5, ravv, SIMILARITY, graph, Bits.ALL);

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
     * Measures recall using brute-force exact search as ground truth.
     * Deleted ordinals are excluded from both the ground truth and search scoring.
     */
    private double measureRecallBrute(List<? extends VectorFloat<?>> queries,
                                      ImmutableGraphIndex graph,
                                      RandomAccessVectorValues ravv,
                                      Set<Integer> deletedNodes,
                                      int topK) {
        double totalRecall = 0.0;
        for (VectorFloat<?> query : queries) {
            Set<Integer> gtSet = bruteForceTopK(query, ravv, deletedNodes, topK);
            var results = GraphSearcher.search(query, topK, EF_SEARCH, ravv, SIMILARITY, graph, Bits.ALL);
            int hits = 0;
            for (var ns : results.getNodes()) {
                if (gtSet.contains(ns.node)) hits++;
            }
            totalRecall += (double) hits / Math.max(1, gtSet.size());
        }
        return totalRecall / queries.size();
    }

    /** Same as measureRecallBrute but prints query 0 hit/miss detail. */
    private double measureRecallBruteVerbose(List<? extends VectorFloat<?>> queries,
                                             ImmutableGraphIndex graph,
                                             RandomAccessVectorValues ravv,
                                             Set<Integer> deletedNodes,
                                             int topK) {
        double totalRecall = 0.0;
        for (int q = 0; q < queries.size(); q++) {
            VectorFloat<?> query = queries.get(q);
            Set<Integer> gtSet = bruteForceTopK(query, ravv, deletedNodes, topK);
            var results = GraphSearcher.search(query, topK, EF_SEARCH, ravv, SIMILARITY, graph, Bits.ALL);
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
                        + " gt=" + gtSet
                        + " graphResults=[" + graphNodes.toString().trim() + "]");
            }
        }
        return totalRecall / queries.size();
    }

    /**
     * Returns the topK nearest neighbour ordinals for the given query via linear scan,
     * excluding all ordinals in {@code excluded}.
     */
    private Set<Integer> bruteForceTopK(VectorFloat<?> query,
                                        RandomAccessVectorValues ravv,
                                        Set<Integer> excluded,
                                        int topK) {
        return IntStream.range(0, ravv.size())
                .filter(i -> !excluded.contains(i))
                .boxed()
                .sorted((a, b) -> Float.compare(
                        SIMILARITY.compare(query, ravv.getVector(b)),
                        SIMILARITY.compare(query, ravv.getVector(a))))
                .limit(topK)
                .collect(Collectors.toCollection(LinkedHashSet::new));
    }
}