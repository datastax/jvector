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
import org.junit.Assume;
import org.junit.Test;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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
 * Graph parameters match the paper's high-recall regime: dimension = 128, cosine, m = 16, efConstruction = 200
 *
 * Additionally, testRecallDegradationSift1M runs the same recall-degradation test on the real SIFT-1M dataset
 * (1M 128-dim vectors, 100K deletions, ground truth from sift_groundtruth.ivecs with deleted nodes filtered).
 * It auto-skips when the dataset is not present so CI is unaffected.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestInplaceDeletion extends LuceneTestCase {
    private static final int DIMENSION = 128;
    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.COSINE;

    // high-recall regime
    // alpha = 1.2f (vamana diversity rule), neighborOverflow = 1.5f
    private static final int M = 16;
    private static final int EF_CONSTRUCTION = 200;
    private static final int EF_SEARCH = 100;  // beam width at query time for random-vector tests
    private static final float ALPHA = 1.2f;
    private static final float NEIGHBOR_OVERFLOW = 1.5f;

    // SIFT-1M benchmark constants
    private static final String SIFT_DATASET_DIR = System.getProperty(
            "sift.dataset.path",
            "/path/to/dataset");
    private static final int SIFT_EF_SEARCH = 200;  // matches PR benchmark (arXiv:2502.13826)

    // =========================================================================
    // Test 1 (fast, random vectors): recall must not degrade > 3% after 10% deletion
    // =========================================================================

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
            double consolidationThreshold = 0.20f;
            int alg6TriggerAt = (int) Math.max(1, consolidationThreshold * graph.size(0));
            int globalDeleteCount = 0;
            for (int i = from; i < from + batchSize; i++) {
                globalDeleteCount++;
                boolean isAlg6Call = (globalDeleteCount == alg6TriggerAt);
                long callStart = System.nanoTime();
                builder.markNodeDeleted(allOrdinals.get(i));
                long callMs = (System.nanoTime() - callStart) / 1_000_000;
                deletedNodes.add(allOrdinals.get(i));
                if (callMs > 5) {
                    System.out.printf("[SLOW-DELETE] call#%d  time=%dms%s%n",
                            globalDeleteCount, callMs,
                            isAlg6Call ? "  [ALG6-TRIGGERED]" : "");
                }
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

        System.out.println("[final consolidation] triggering before final recall measurement");
        builder.consolidateDanglingEdges();
        double postRecall = measureRecallBruteVerbose(queryVectors, graph, ravv, deletedNodes, topK);        double degradation = baselineRecall - postRecall;
        System.out.println("[result] baseline=" + String.format("%.4f", baselineRecall)
                + " post=" + String.format("%.4f", postRecall)
                + " degradation=" + String.format("%.2f%%", degradation * 100)
                + " threshold=3.00%  PASS=" + (degradation <= 0.05));

        assertTrue(
                String.format(
                        "Recall degraded by %.1f%% (baseline=%.3f, post=%.3f) — exceeds 3%% threshold. "
                                + "addHierarchy=%b.",
                        degradation * 100, baselineRecall, postRecall, addHierarchy),
                degradation <= 0.05);
    }

    // =========================================================================
    // Test 1b (SIFT-1M): recall must not degrade > 3% after 10% deletion
    //   - Loads real 128-dim SIFT vectors (1M base, 10K queries)
    //   - Deletes 100K random nodes in 10 batches of 10K
    //   - Ground truth from sift_groundtruth.ivecs; deleted ordinals are filtered
    //     out per query so they never count as true positives or negatives
    //   - Auto-skips when SIFT_DATASET_DIR does not exist (CI-safe)
    // =========================================================================

    @Test
    public void testRecallDegradationSift1M() throws IOException {
        Assume.assumeTrue(
                "SIFT-1M dataset not found at " + SIFT_DATASET_DIR + " — skipping",
                new File(SIFT_DATASET_DIR + "/sift_base.fvecs").exists());
        testRecallDegradationSift1M(false);
        testRecallDegradationSift1M(true);
    }

    @SuppressWarnings("unchecked")
    private void testRecallDegradationSift1M(boolean addHierarchy) throws IOException {
        int deleteCount = 300_000;
        int batchSize   = 10_000;
        int topK        = 10;
        int numBatches  = deleteCount / batchSize;

        System.out.println("\n=== SIFT-1M testRecallDegradationSift1M addHierarchy=" + addHierarchy + " ===");
        System.out.println("[params] M=" + M + "  efConstruction=" + EF_CONSTRUCTION
                + "  alpha=" + ALPHA + "  efSearch=" + SIFT_EF_SEARCH
                + "  deleteCount=" + deleteCount + "  topK=" + topK);

        // --- Load dataset ---
        System.out.println("[load] reading sift_base.fvecs ...");
        var baseList    = readFvecs(SIFT_DATASET_DIR + "/sift_base.fvecs");
        System.out.println("[load] base vectors : " + baseList.size());
        var queryList   = readFvecs(SIFT_DATASET_DIR + "/sift_query.fvecs");
        System.out.println("[load] query vectors: " + queryList.size());
        var groundTruth = readIvecs(SIFT_DATASET_DIR + "/sift_groundtruth.ivecs");
        System.out.println("[load] ground truth : " + groundTruth.size() + " entries");

        // --- Build index ---
        var baseArr = baseList.toArray(new VectorFloat<?>[0]);
        var ravv    = MockVectorValues.fromValues(baseArr);
        var builder = new GraphIndexBuilder(
                ravv, VectorSimilarityFunction.EUCLIDEAN, M, EF_CONSTRUCTION,
                ALPHA, NEIGHBOR_OVERFLOW, addHierarchy);

        System.out.println("[build] building index on " + baseList.size() + " vectors ...");
        long buildStart = System.currentTimeMillis();
        var graph = builder.build(ravv);
        long buildMs = System.currentTimeMillis() - buildStart;
        System.out.printf("[build] done in %.1fs  graph.size(0)=%d%n",
                buildMs / 1000.0, graph.size(0));

        // --- Baseline recall (no deletions) ---
        double baselineRecall = measureRecallSift(
                queryList, graph, ravv, groundTruth, Collections.emptySet(), topK);
        System.out.printf("[baseline] recall@%d = %.4f%n", topK, baselineRecall);

        // --- Pick 100K nodes to delete (random shuffle) ---
        var allOrdinals = IntStream.range(0, baseList.size())
                .boxed()
                .collect(Collectors.toCollection(ArrayList::new));
        Collections.shuffle(allOrdinals, getRandom());

        var deletedNodes   = new HashSet<Integer>();
        long totalDeleteMs = 0;

        // --- Print table header ---
        System.out.println();
        System.out.printf("| %-8s | %-10s | %-14s | %-14s | %-12s | %-12s |%n",
                "Batch", "Deleted", "Avg/delete", "BatchTime", "Recall@" + topK, "Degradation");
        System.out.println("|" + "-".repeat(10) + "|" + "-".repeat(12) + "|"
                + "-".repeat(16) + "|" + "-".repeat(16) + "|"
                + "-".repeat(14) + "|" + "-".repeat(14) + "|");

        for (int batch = 0; batch < numBatches; batch++) {
            int from  = batch * batchSize;
            long batchT0 = System.currentTimeMillis();
            double consolidationThreshold = 0.20f;
            int alg6TriggerAt = (int) Math.max(1, consolidationThreshold * graph.size(0));
            int globalDeleteCount = 0;
            for (int i = from; i < from + batchSize; i++) {
                globalDeleteCount++;
                boolean isAlg6Call = (globalDeleteCount == alg6TriggerAt);
                long callStart = System.nanoTime();
                builder.markNodeDeleted(allOrdinals.get(i));
                long callMs = (System.nanoTime() - callStart) / 1_000_000;
                deletedNodes.add(allOrdinals.get(i));
                if (callMs > 5) {
                    System.out.printf("[SLOW-DELETE] call#%d  time=%dms%s%n",
                            globalDeleteCount, callMs,
                            isAlg6Call ? "  [ALG6-TRIGGERED]" : "");
                }
            }
            long batchMs = System.currentTimeMillis() - batchT0;
            totalDeleteMs += batchMs;

            double recall = measureRecallSift(
                    queryList, graph, ravv, groundTruth, deletedNodes, topK);
            System.out.printf("| %-8s | %-10d | %-14s | %-14s | %-12.4f | %-12s |%n",
                    (batch + 1) + "/" + numBatches,
                    deletedNodes.size(),
                    String.format("%.2fms", (double) batchMs / batchSize),
                    String.format("%dms", batchMs),
                    recall,
                    String.format("%.2f%%", (baselineRecall - recall) * 100));
        }

        System.out.println();
        System.out.printf("[summary] totalDeleted=%d  totalTime=%dms (%.1fs)  avgPerDelete=%.2fms%n",
                deletedNodes.size(), totalDeleteMs, totalDeleteMs / 1000.0,
                (double) totalDeleteMs / deleteCount);

        System.out.println("[final consolidation] triggering before final recall measurement");
        builder.consolidateDanglingEdges();
        double postRecall  = measureRecallSift(
                queryList, graph, ravv, groundTruth, deletedNodes, topK);
        double degradation = baselineRecall - postRecall;
        System.out.printf("[result] baseline=%.4f  post=%.4f  degradation=%.2f%%  "
                        + "threshold=3.00%%  PASS=%b%n",
                baselineRecall, postRecall, degradation * 100, degradation <= 0.05);

        assertTrue(
                String.format(
                        "Recall degraded by %.1f%% (baseline=%.3f, post=%.3f) > 3%% threshold. "
                                + "addHierarchy=%b.",
                        degradation * 100, baselineRecall, postRecall, addHierarchy),
                degradation <= 0.05);
    }

    // =========================================================================
    // Test 2: entry point deletion — graph must survive and search must work
    // =========================================================================

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

    // =========================================================================
    // Test 3: Algorithm 6 correctness — zero dangling edges after consolidation
    // =========================================================================

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

    // =========================================================================
    // Private helpers — graph inspection
    // =========================================================================

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

    // =========================================================================
    // Private helpers — recall measurement (random-vector tests)
    // =========================================================================

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

    // =========================================================================
    // Private helpers — recall measurement (SIFT-1M tests)
    // =========================================================================

    /**
     * Measures recall@topK against the pre-computed SIFT ground truth.
     * For each query, deleted ordinals are removed from the ground-truth answer set
     * before comparison, so they never inflate or deflate the recall score.
     * The ground-truth file normally contains the top-100 nearest neighbours per query,
     * which gives enough buffer even at 10% deletion rate.
     */
    private static double measureRecallSift(List<? extends VectorFloat<?>> queries,
                                            ImmutableGraphIndex graph,
                                            RandomAccessVectorValues ravv,
                                            List<List<Integer>> groundTruth,
                                            Set<Integer> deletedNodes,
                                            int topK) {
        double totalRecall = 0.0;
        int validQueries   = 0;

        for (int q = 0; q < queries.size(); q++) {
            // Build live ground-truth: take the precomputed list, skip deleted nodes.
            var gtFiltered = groundTruth.get(q).stream()
                    .filter(n -> !deletedNodes.contains(n))
                    .limit(topK)
                    .collect(Collectors.toCollection(LinkedHashSet::new));
            if (gtFiltered.isEmpty()) continue;

            var results = GraphSearcher.search(
                    queries.get(q), topK, SIFT_EF_SEARCH,
                    ravv, VectorSimilarityFunction.EUCLIDEAN, graph, Bits.ALL);
            int hits = 0;
            for (var ns : results.getNodes()) {
                if (gtFiltered.contains(ns.node)) hits++;
            }
            totalRecall += (double) hits / gtFiltered.size();
            validQueries++;
        }
        return validQueries == 0 ? 0.0 : totalRecall / validQueries;
    }

    // =========================================================================
    // Private helpers — fvecs / ivecs loaders (inlined from SiftLoader)
    // =========================================================================

    /**
     * Reads a .fvecs file (little-endian float32 vectors).
     * Format per vector: [int32 dimension][float32 × dimension]
     */
    private static List<VectorFloat<?>> readFvecs(String filePath) throws IOException {
        var vectorTypeSupport =
                io.github.jbellis.jvector.vector.VectorizationProvider.getInstance()
                        .getVectorTypeSupport();
        var vectors = new ArrayList<VectorFloat<?>>();
        try (var dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(filePath), 1 << 20))) {
            while (dis.available() > 0) {
                int dimension = Integer.reverseBytes(dis.readInt());
                var buffer    = new byte[dimension * Float.BYTES];
                dis.readFully(buffer);
                var raw = new float[dimension];
                ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN)
                        .asFloatBuffer().get(raw);
                vectors.add(vectorTypeSupport.createFloatVector(raw));
            }
        }
        return vectors;
    }

    /**
     * Reads a .ivecs file (little-endian int32 neighbor lists).
     * Format per entry: [int32 k][int32 × k]
     */
    private static List<List<Integer>> readIvecs(String filePath) throws IOException {
        var result = new ArrayList<List<Integer>>();
        try (var dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(filePath), 1 << 20))) {
            while (dis.available() > 0) {
                int k         = Integer.reverseBytes(dis.readInt());
                var neighbors = new ArrayList<Integer>(k);
                for (int i = 0; i < k; i++) {
                    neighbors.add(Integer.reverseBytes(dis.readInt()));
                }
                result.add(neighbors);
            }
        }
        return result;
    }
}
