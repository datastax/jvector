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

package io.github.jbellis.jvector.graph.disk;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.BoundedLongHeap;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.util.work.ProgressLimiter;
import io.github.jbellis.jvector.util.work.WorkLimiter;
import io.github.jbellis.jvector.util.work.WorkStage;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.IntFunction;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.fail;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestOnDiskGraphIndexCompactor extends RandomizedTest {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    private ImmutableGraphIndex golden;
    private Path testDirectory;
    List<VectorFloat<?>> allVecs = new ArrayList<>();
    int dimension = 32;
    int numVectorsPerGraph = 256;
    int numSources = 3;
    int numQueries = 20;
    VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.COSINE;
    RandomAccessVectorValues allravv;
    private final ForkJoinPool simdExecutor = ForkJoinPool.commonPool();
    private final ForkJoinPool parallelExecutor = ForkJoinPool.commonPool();

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory("jvector_test");
        buildFusedPQ();
        buildGoldenPQ();
    }

    /**
     * Builds source graphs with FusedPQ feature enabled.
     * Uses random vectors with COSINE similarity.
     */
    void buildFusedPQ() throws IOException {
        for(int i = 0; i < numSources; ++i) {
            List<VectorFloat<?>> vecs = createRandomVectors(numVectorsPerGraph, dimension);

            RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vecs, dimension);
            ProductQuantization pq = ProductQuantization.compute(ravv, 8, 256, true, UNWEIGHTED, simdExecutor, parallelExecutor);
            PQVectors pqv = (PQVectors) pq.encodeAll(ravv, simdExecutor);
            var bsp = BuildScoreProvider.pqBuildScoreProvider(similarityFunction, pqv);
            var builder = new GraphIndexBuilder(bsp, dimension, 16, 100, 1.2f, 1.2f, false, true, simdExecutor, parallelExecutor);
            var graph = builder.getGraph();

            var outputPath = testDirectory.resolve("test_graph_" + i);
            Map<FeatureId, IntFunction<Feature.State>> writeSuppliers = new EnumMap<>(FeatureId.class);
            writeSuppliers.put(FeatureId.INLINE_VECTORS, ordinal -> new InlineVectors.State(ravv.getVector(ordinal)));

            var identityMapper = new OrdinalMapper.IdentityMapper(ravv.size() - 1);
            var writerBuilder = new OnDiskGraphIndexWriter.Builder(graph, outputPath);
            writerBuilder.withMapper(identityMapper);
            writerBuilder.with(new InlineVectors(dimension));
            writerBuilder.with(new FusedPQ(graph.maxDegree(), pq));
            var writer = writerBuilder.build();

            for (var node = 0; node < ravv.size(); node++) {
                var stateMap = new EnumMap<FeatureId, Feature.State>(FeatureId.class);
                stateMap.put(FeatureId.INLINE_VECTORS, writeSuppliers.get(FeatureId.INLINE_VECTORS).apply(node));
                writer.writeInline(node, stateMap);
                builder.addGraphNode(node, ravv.getVector(node));
            }
            builder.cleanup();

            writeSuppliers.put(FeatureId.FUSED_PQ, ordinal -> new FusedPQ.State(graph.getView(), pqv, ordinal));
            writer.write(writeSuppliers);
            allVecs.addAll(vecs);
        }
    }

    /**
     * Builds the golden graph from all vectors combined.
     * This represents the ideal case of building from scratch.
     */
    void buildGoldenPQ() throws IOException {
        allravv = new ListRandomAccessVectorValues(allVecs, dimension);

        ProductQuantization pq = ProductQuantization.compute(allravv, 8, 256, true, UNWEIGHTED, simdExecutor, parallelExecutor);
        PQVectors pqv = (PQVectors) pq.encodeAll(allravv, simdExecutor);
        var bsp = BuildScoreProvider.pqBuildScoreProvider(similarityFunction, pqv);
        var builder = new GraphIndexBuilder(bsp, dimension, 16, 100, 1.2f, 1.2f, false, true, simdExecutor, parallelExecutor);
        for (var i = 0; i < allravv.size(); i++) {
            builder.addGraphNode(i, allravv.getVector(i));
        }
        builder.cleanup();
        golden = builder.getGraph();
    }
    List<SearchResult> searchFromAll(List<VectorFloat<?>> queries, int topK) {
        List<SearchResult> srs = new ArrayList<>();
        try (GraphSearcher searcher = new GraphSearcher(golden)) {
            for(VectorFloat<?> q: queries) {
                var row = new ArrayList<Integer>();
                SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(q, similarityFunction, allravv);
                SearchResult sr = searcher.search(ssp, topK, Bits.ALL);
                srs.add(sr);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return srs;
    }
    List<List<Integer>> buildGT(List<VectorFloat<?>> queries, int topK) {
        List<List<Integer>> rows = new ArrayList<>();

        for(int i = 0; i < queries.size(); ++i) {
            NodeQueue expected = new NodeQueue(new BoundedLongHeap(topK), NodeQueue.Order.MIN_HEAP);
            for (int j = 0; j < allVecs.size(); j++) {
                expected.push(j, similarityFunction.compare(queries.get(i), allVecs.get(j)));
            }

            var row = new ArrayList<Integer>();
            for(int k = 0; k < topK; ++k) {
                row.add(expected.pop());
            }
            rows.add(row);
        }
        return rows;
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    /**
     * Builds a small source graph with InlineVectors only (no FusedPQ), using exact scoring.
     * Returns the path to the written graph file.
     */
    private Path buildSimpleSourceGraph(List<VectorFloat<?>> vecs, int dim, VectorSimilarityFunction vsf, String name) throws IOException {
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vecs, dim);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, vsf);
        var builder = new GraphIndexBuilder(bsp, dim, 4, 20, 1.2f, 1.2f, false, true, simdExecutor, parallelExecutor);
        for (int i = 0; i < vecs.size(); i++) {
            builder.addGraphNode(i, vecs.get(i));
        }
        builder.cleanup();
        var graph = builder.getGraph();

        var outputPath = testDirectory.resolve(name);
        var identityMapper = new OrdinalMapper.IdentityMapper(vecs.size() - 1);
        var writerBuilder = new OnDiskGraphIndexWriter.Builder(graph, outputPath);
        writerBuilder.withMapper(identityMapper);
        writerBuilder.with(new InlineVectors(dim));
        var writer = writerBuilder.build();

        Map<FeatureId, IntFunction<Feature.State>> writeSuppliers = new EnumMap<>(FeatureId.class);
        writeSuppliers.put(FeatureId.INLINE_VECTORS, ordinal -> new InlineVectors.State(ravv.getVector(ordinal)));

        for (int node = 0; node < vecs.size(); node++) {
            var stateMap = new EnumMap<FeatureId, Feature.State>(FeatureId.class);
            stateMap.put(FeatureId.INLINE_VECTORS, writeSuppliers.get(FeatureId.INLINE_VECTORS).apply(node));
            writer.writeInline(node, stateMap);
        }
        writer.write(writeSuppliers);
        return outputPath;
    }

    /** Creates a vector of the given dimension with value at index {@code hot} set to {@code val}, rest 0. */
    private VectorFloat<?> makeVec(int dim, int hot, float val) {
        VectorFloat<?> v = vectorTypeSupport.createFloatVector(dim);
        for (int d = 0; d < dim; d++) {
            v.set(d, d == hot ? val : 0.0f);
        }
        return v;
    }

    private void assertVecEquals(VectorFloat<?> expected, VectorFloat<?> actual, int ordinal) {
        int dim = expected.length();
        assertEquals("dimension mismatch at ordinal " + ordinal, dim, actual.length());
        for (int d = 0; d < dim; d++) {
            assertEquals(String.format("vector[%d] dim %d mismatch", ordinal, d), expected.get(d), actual.get(d), 0.0f);
        }
    }

    /**
     * Tests that vectors are stored exactly at the expected global ordinals after compaction.
     * Uses two small sources with simple, known float values and identity mapping.
     */
    @Test
    public void testExactVectorValuesAfterCompaction() throws Exception {
        int dim = 4;
        int n = 6; // nodes per source
        VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;

        // Source 0: vectors with first dim varying by index
        List<VectorFloat<?>> vecs0 = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            vecs0.add(makeVec(dim, 0, (float)(i + 1)));
        }
        // Source 1: vectors with second dim varying by index
        List<VectorFloat<?>> vecs1 = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            vecs1.add(makeVec(dim, 1, (float)(i + 10)));
        }

        Path path0 = buildSimpleSourceGraph(vecs0, dim, vsf, "simple_src_0");
        Path path1 = buildSimpleSourceGraph(vecs1, dim, vsf, "simple_src_1");

        ReaderSupplier rs0 = ReaderSupplierFactory.open(path0);
        ReaderSupplier rs1 = ReaderSupplierFactory.open(path1);
        OnDiskGraphIndex g0 = OnDiskGraphIndex.load(rs0);
        OnDiskGraphIndex g1 = OnDiskGraphIndex.load(rs1);

        // Identity remapping: source i -> global ordinals [i*n, (i+1)*n)
        Map<Integer, Integer> map0 = new HashMap<>();
        Map<Integer, Integer> map1 = new HashMap<>();
        for (int i = 0; i < n; i++) {
            map0.put(i, i);
            map1.put(i, n + i);
        }

        FixedBitSet live0 = new FixedBitSet(n);
        live0.set(0, n);
        FixedBitSet live1 = new FixedBitSet(n);
        live1.set(0, n);

        var compactor = new OnDiskGraphIndexCompactor(
                List.of(g0, g1),
                List.of(live0, live1),
                List.of(new OrdinalMapper.MapMapper(map0), new OrdinalMapper.MapMapper(map1)),
                vsf, null, -1);

        Path outPath = testDirectory.resolve("simple_compact_out");
        compactor.compact(outPath);

        ReaderSupplier rsOut = ReaderSupplierFactory.open(outPath);
        OnDiskGraphIndex compacted = OnDiskGraphIndex.load(rsOut);
        assertEquals(2 * n, compacted.size(0));

        var view = compacted.getView();
        VectorFloat<?> buf = vectorTypeSupport.createFloatVector(dim);

        // Source 0 vectors must be at ordinals 0..n-1
        for (int i = 0; i < n; i++) {
            view.getVectorInto(i, buf, 0);
            assertVecEquals(vecs0.get(i), buf, i);
        }
        // Source 1 vectors must be at ordinals n..2n-1
        for (int i = 0; i < n; i++) {
            view.getVectorInto(n + i, buf, 0);
            assertVecEquals(vecs1.get(i), buf, n + i);
        }
    }

    /**
     * Tests that only live vectors appear after compaction, placed at the correct remapped ordinals.
     * Deletes every other node from each source and verifies the compacted output exactly.
     */
    @Test
    public void testExactVectorValuesWithDeletions() throws Exception {
        int dim = 4;
        int n = 8; // nodes per source
        VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;

        // Source 0: vectors [1,0,0,0] through [8,0,0,0]
        List<VectorFloat<?>> vecs0 = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            vecs0.add(makeVec(dim, 0, (float)(i + 1)));
        }
        // Source 1: vectors [0,10,0,0] through [0,170,0,0]
        List<VectorFloat<?>> vecs1 = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            vecs1.add(makeVec(dim, 1, (float)((i + 1) * 10)));
        }

        Path path0 = buildSimpleSourceGraph(vecs0, dim, vsf, "del_src_0");
        Path path1 = buildSimpleSourceGraph(vecs1, dim, vsf, "del_src_1");

        ReaderSupplier rs0 = ReaderSupplierFactory.open(path0);
        ReaderSupplier rs1 = ReaderSupplierFactory.open(path1);
        OnDiskGraphIndex g0 = OnDiskGraphIndex.load(rs0);
        OnDiskGraphIndex g1 = OnDiskGraphIndex.load(rs1);

        // Keep only even-indexed nodes (0, 2, 4, 6) in both sources
        FixedBitSet live0 = new FixedBitSet(n);
        FixedBitSet live1 = new FixedBitSet(n);
        Map<Integer, Integer> map0 = new HashMap<>();
        Map<Integer, Integer> map1 = new HashMap<>();
        int globalOrdinal = 0;
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) {
                live0.set(i);
                map0.put(i, globalOrdinal++);
            }
        }
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) {
                live1.set(i);
                map1.put(i, globalOrdinal++);
            }
        }
        int expectedTotal = globalOrdinal;

        var compactor = new OnDiskGraphIndexCompactor(
                List.of(g0, g1),
                List.of(live0, live1),
                List.of(new OrdinalMapper.MapMapper(map0), new OrdinalMapper.MapMapper(map1)),
                vsf, null, -1);

        Path outPath = testDirectory.resolve("del_compact_out");
        compactor.compact(outPath);

        ReaderSupplier rsOut = ReaderSupplierFactory.open(outPath);
        OnDiskGraphIndex compacted = OnDiskGraphIndex.load(rsOut);
        assertEquals(expectedTotal, compacted.size(0));

        var view = compacted.getView();
        VectorFloat<?> buf = vectorTypeSupport.createFloatVector(dim);

        // Verify source 0 live nodes at their mapped ordinals
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) {
                int ord = map0.get(i);
                view.getVectorInto(ord, buf, 0);
                assertVecEquals(vecs0.get(i), buf, ord);
            }
        }
        // Verify source 1 live nodes at their mapped ordinals
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) {
                int ord = map1.get(i);
                view.getVectorInto(ord, buf, 0);
                assertVecEquals(vecs1.get(i), buf, ord);
            }
        }
    }

    /**
     * Tests that vectors end up at the correct ordinals when a non-sequential remapping is used.
     * Source 0 is mapped in reverse order; source 1 is mapped in forward order.
     * Verifies exact vector values at every remapped position.
     */
    @Test
    public void testExactVectorValuesWithCustomRemapping() throws Exception {
        int dim = 4;
        int n = 6;
        VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;

        List<VectorFloat<?>> vecs0 = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            vecs0.add(makeVec(dim, 2, (float)(i + 1)));
        }
        List<VectorFloat<?>> vecs1 = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            vecs1.add(makeVec(dim, 3, (float)(i + 100)));
        }

        Path path0 = buildSimpleSourceGraph(vecs0, dim, vsf, "remap_src_0");
        Path path1 = buildSimpleSourceGraph(vecs1, dim, vsf, "remap_src_1");

        ReaderSupplier rs0 = ReaderSupplierFactory.open(path0);
        ReaderSupplier rs1 = ReaderSupplierFactory.open(path1);
        OnDiskGraphIndex g0 = OnDiskGraphIndex.load(rs0);
        OnDiskGraphIndex g1 = OnDiskGraphIndex.load(rs1);

        // Source 0: reverse mapping (local 0 -> global n-1, local 1 -> global n-2, ...)
        Map<Integer, Integer> map0 = new HashMap<>();
        for (int i = 0; i < n; i++) {
            map0.put(i, n - 1 - i);
        }
        // Source 1: forward mapping (local 0 -> global n, local 1 -> global n+1, ...)
        Map<Integer, Integer> map1 = new HashMap<>();
        for (int i = 0; i < n; i++) {
            map1.put(i, n + i);
        }

        FixedBitSet live0 = new FixedBitSet(n);
        live0.set(0, n);
        FixedBitSet live1 = new FixedBitSet(n);
        live1.set(0, n);

        var compactor = new OnDiskGraphIndexCompactor(
                List.of(g0, g1),
                List.of(live0, live1),
                List.of(new OrdinalMapper.MapMapper(map0), new OrdinalMapper.MapMapper(map1)),
                vsf, null, -1);

        Path outPath = testDirectory.resolve("remap_compact_out");
        compactor.compact(outPath);

        ReaderSupplier rsOut = ReaderSupplierFactory.open(outPath);
        OnDiskGraphIndex compacted = OnDiskGraphIndex.load(rsOut);
        assertEquals(2 * n, compacted.size(0));

        var view = compacted.getView();
        VectorFloat<?> buf = vectorTypeSupport.createFloatVector(dim);

        for (int i = 0; i < n; i++) {
            int ord = map0.get(i);
            view.getVectorInto(ord, buf, 0);
            assertVecEquals(vecs0.get(i), buf, ord);
        }
        for (int i = 0; i < n; i++) {
            int ord = map1.get(i);
            view.getVectorInto(ord, buf, 0);
            assertVecEquals(vecs1.get(i), buf, ord);
        }
    }

    /**
     * Tests basic compaction: merging multiple graphs without deletions.
     * Verifies that compacted graph recall is comparable to golden graph.
     */
    @Test
    public void testCompact() throws Exception {
        List<OnDiskGraphIndex> graphs = new ArrayList<>();
        List<ReaderSupplier> rss = new ArrayList<>();
        List<FixedBitSet> liveNodes = new ArrayList<>();
        List<OrdinalMapper> remappers = new ArrayList<>();

        // Load all source graphs
        for(int i = 0; i < numSources; ++i) {
            var outputPath = testDirectory.resolve("test_graph_" + i);
            rss.add(ReaderSupplierFactory.open(outputPath.toAbsolutePath()));
            var onDiskGraph = OnDiskGraphIndex.load(rss.get(i));
            graphs.add(onDiskGraph);
        }

        // Create identity mapping and all nodes live
        int globalOrdinal = 0;
        for (int n = 0; n < numSources; n++) {
            Map<Integer, Integer> map = new HashMap<>(numVectorsPerGraph);
            for (int i = 0; i < numVectorsPerGraph; i++) {
                map.put(i, globalOrdinal++);
            }
            remappers.add(new OrdinalMapper.MapMapper(map));

            var lives = new FixedBitSet(numVectorsPerGraph);
            lives.set(0, numVectorsPerGraph);
            liveNodes.add(lives);
        }

        var compactor = new OnDiskGraphIndexCompactor(graphs, liveNodes, remappers, similarityFunction, null, -1);
        int topK = 10;

        // Select query vectors from the dataset
        var outputPath = testDirectory.resolve("test_compact_graph_");
        List<VectorFloat<?>> queries = new ArrayList<>();
        for(int i = 0; i < numQueries; ++i) {
            queries.add(allVecs.get(randomIntBetween(0, allVecs.size() - 1)));
        }

        // Get golden results and ground truth
        List<SearchResult> goldenResults = searchFromAll(queries, topK);
        List<List<Integer>> groundTruth = buildGT(queries, topK);

        // Compact and test
        compactor.compact(outputPath);

        ReaderSupplier rs = ReaderSupplierFactory.open(outputPath);
        var compactGraph = OnDiskGraphIndex.load(rs);

        // Verify basic properties
        assertEquals("Compacted graph should have all nodes", numSources * numVectorsPerGraph, compactGraph.size(0));

        GraphSearcher searcher = new GraphSearcher(compactGraph);
        List<SearchResult> compactResults = new ArrayList<>();
        for(VectorFloat<?> q: queries) {
            SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(q, similarityFunction, allravv);
            compactResults.add(searcher.search(ssp, topK, Bits.ALL));
        }

        // Calculate recalls
        double goldenRecall = AccuracyMetrics.recallFromSearchResults(groundTruth, goldenResults, topK, topK);
        double compactRecall = AccuracyMetrics.recallFromSearchResults(groundTruth, compactResults, topK, topK);

        System.out.printf("Golden (built from scratch) Recall: %.4f%n", goldenRecall);
        System.out.printf("Compacted Recall: %.4f%n", compactRecall);
        System.out.printf("Recall difference: %.4f%n", Math.abs(goldenRecall - compactRecall));

        // For random vectors with COSINE, both golden and compact should have similar recall
        // The key is that they're comparable to each other, showing compaction preserves graph quality
        double recallDifference = Math.abs(goldenRecall - compactRecall);
        assertTrue(String.format("Compacted recall (%.4f) should be comparable to golden recall (%.4f), difference: %.4f",
                                compactRecall, goldenRecall, recallDifference),
                  recallDifference < 0.2); // Allow up to 20% difference for random vectors

        // Verify both are reasonable (not completely broken)
        assertTrue(String.format("Golden recall should be at least 0.2, got %.4f", goldenRecall),
                  goldenRecall >= 0.2);
        assertTrue(String.format("Compacted recall should be at least 0.2, got %.4f", compactRecall),
                  compactRecall >= 0.2);

        searcher.close();
    }

    // ---- ProgressLimiter SPI (io.github.jbellis.jvector.util.work) ----

    /** A ProgressLimiter that records observations and hands out no-op grants (rate-limiter style). */
    private static final class RecordingLimiter implements ProgressLimiter {
        final Set<String> stages = ConcurrentHashMap.newKeySet();
        final Map<String, Long> lastCompleted = new ConcurrentHashMap<>();
        final Map<String, Long> lastTotal = new ConcurrentHashMap<>();
        final Map<String, Boolean> monotonic = new ConcurrentHashMap<>();
        final AtomicInteger acquires = new AtomicInteger();
        final AtomicInteger closes = new AtomicInteger();
        final AtomicLong bytesAcquired = new AtomicLong();

        @Override
        public void onProgress(WorkStage stage, long completed, long total) {
            String s = stage.name();
            stages.add(s);
            lastTotal.put(s, total);
            Long prev = lastCompleted.put(s, completed);
            if (prev != null && completed < prev) monotonic.put(s, false);
            else monotonic.putIfAbsent(s, true);
        }

        @Override
        public WorkLimiter.Grant acquire(long amount) {
            acquires.incrementAndGet();
            bytesAcquired.addAndGet(amount);
            return () -> { closes.incrementAndGet(); };
        }
    }

    /** A ProgressLimiter whose grants hold real semaphore permits, released on close. */
    private static final class SemaphoreBytesLimiter implements ProgressLimiter {
        final Semaphore permits;
        final int cap;
        final AtomicInteger open = new AtomicInteger();

        SemaphoreBytesLimiter(int totalPermits) {
            this.permits = new Semaphore(totalPermits);
            this.cap = totalPermits;
        }

        @Override
        public WorkLimiter.Grant acquire(long amount) throws InterruptedException {
            final int n = (int) Math.max(1, Math.min(amount, cap)); // never exceed total -> no single-acquire deadlock
            permits.acquire(n);
            open.incrementAndGet();
            return () -> { permits.release(n); open.decrementAndGet(); };
        }
    }

    /** Loads the fixture's source graphs with every node live and identity remapping. */
    private OnDiskGraphIndexCompactor newAllLiveCompactor(List<ReaderSupplier> rss) throws IOException {
        return newAllLiveCompactor(rss, (Executor) null, -1);
    }

    private OnDiskGraphIndexCompactor newAllLiveCompactor(List<ReaderSupplier> rss, Executor executor, int taskWindowSize) throws IOException {
        List<OnDiskGraphIndex> graphs = new ArrayList<>();
        List<FixedBitSet> liveNodes = new ArrayList<>();
        List<OrdinalMapper> remappers = new ArrayList<>();
        for (int i = 0; i < numSources; ++i) {
            var rs = ReaderSupplierFactory.open(testDirectory.resolve("test_graph_" + i).toAbsolutePath());
            rss.add(rs);
            graphs.add(OnDiskGraphIndex.load(rs));
        }
        int globalOrdinal = 0;
        for (int n = 0; n < numSources; n++) {
            Map<Integer, Integer> map = new HashMap<>(numVectorsPerGraph);
            for (int i = 0; i < numVectorsPerGraph; i++) map.put(i, globalOrdinal++);
            remappers.add(new OrdinalMapper.MapMapper(map));
            var lives = new FixedBitSet(numVectorsPerGraph);
            lives.set(0, numVectorsPerGraph);
            liveNodes.add(lives);
        }
        return new OnDiskGraphIndexCompactor(graphs, liveNodes, remappers, similarityFunction, executor, taskWindowSize);
    }

    @Test
    public void testTaskWindowSizeResolution() throws Exception {
        List<ReaderSupplier> rss1 = new ArrayList<>();
        try {
            assertEquals("explicit window must be honored",
                    5, newAllLiveCompactor(rss1, (Executor) Runnable::run, 5).getTaskWindowSize());
        } finally { for (var rs : rss1) rs.close(); }

        List<ReaderSupplier> rss2 = new ArrayList<>();
        try {
            assertEquals("non-FJP executor with window<=0 must default to 1 (serial)",
                    1, newAllLiveCompactor(rss2, (Executor) Runnable::run, 0).getTaskWindowSize());
        } finally { for (var rs : rss2) rs.close(); }

        List<ReaderSupplier> rss3 = new ArrayList<>();
        ForkJoinPool pool = new ForkJoinPool(3);
        try {
            assertEquals("window<=0 with a ForkJoinPool must derive getParallelism()",
                    3, newAllLiveCompactor(rss3, pool, -1).getTaskWindowSize());
        } finally { pool.shutdown(); for (var rs : rss3) rs.close(); }
    }

    @Test
    public void testCallerRunsProducesValidGraph() throws Exception {
        // Caller-runs + serial must produce a correct, searchable graph of equivalent quality to
        // the ForkJoinPool path. (Not asserted byte-identical: fused-PQ retraining uses parallel
        // floating-point reduction, which is order-sensitive, so serialization can shift low bits.)
        List<ReaderSupplier> rss = new ArrayList<>();
        try {
            var path = testDirectory.resolve("cr_caller");
            newAllLiveCompactor(rss, (Executor) Runnable::run, 1).compact(path);

            try (ReaderSupplier out = ReaderSupplierFactory.open(path)) {
                var g = OnDiskGraphIndex.load(out);
                assertEquals(numSources * numVectorsPerGraph, g.size(0));

                List<VectorFloat<?>> queries = new ArrayList<>();
                for (int i = 0; i < numQueries; ++i) queries.add(allVecs.get(randomIntBetween(0, allVecs.size() - 1)));
                List<List<Integer>> gt = buildGT(queries, 10);
                GraphSearcher searcher = new GraphSearcher(g);
                List<SearchResult> results = new ArrayList<>();
                for (VectorFloat<?> q : queries) {
                    SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(q, similarityFunction, allravv);
                    results.add(searcher.search(ssp, 10, Bits.ALL));
                }
                double recall = AccuracyMetrics.recallFromSearchResults(gt, results, 10, 10);
                assertTrue("caller-runs graph recall should be reasonable, got " + recall, recall >= 0.2);
                searcher.close();
            }
        } finally {
            for (var rs : rss) rs.close();
        }
    }

    @Test
    public void testCallerRunsRunsOnCallingThreadOnly() throws Exception {
        List<ReaderSupplier> rss = new ArrayList<>();
        Set<Thread> taskThreads = Collections.synchronizedSet(new HashSet<>());
        Executor recordingCallerRuns = command -> {
            taskThreads.add(Thread.currentThread());
            command.run();
        };
        try {
            newAllLiveCompactor(rss, recordingCallerRuns, 1).compact(testDirectory.resolve("cr_thread"));
            assertEquals("all batch + pre-encode work must run on the calling thread only",
                    Collections.singleton(Thread.currentThread()), taskThreads);
        } finally {
            for (var rs : rss) rs.close();
        }
    }

    @Test
    public void testProgressLimiterObservesAndPairsGrants() throws Exception {
        List<ReaderSupplier> rss = new ArrayList<>();
        var compactor = newAllLiveCompactor(rss);
        var rec = new RecordingLimiter();
        compactor.setProgressLimiter(rec);
        // Refinement is opt-in (default off); this test asserts progress across both the
        // MERGE_LEVELS and REFINE phases, so it must enable the refinement pass explicitly.
        compactor.setRefineAfterCompaction(true);

        var outputPath = testDirectory.resolve("test_compact_progress");
        compactor.compact(outputPath);

        // Both stages observed, progress monotonic and finishing at 100% of a known total.
        assertTrue("MERGE_LEVELS progress not reported", rec.stages.contains("MERGE_LEVELS"));
        assertTrue("REFINE progress not reported", rec.stages.contains("REFINE"));
        assertTrue("MERGE_LEVELS progress not monotonic", rec.monotonic.getOrDefault("MERGE_LEVELS", true));
        assertTrue("REFINE progress not monotonic", rec.monotonic.getOrDefault("REFINE", true));
        assertEquals("MERGE_LEVELS did not reach total",
                rec.lastTotal.get("MERGE_LEVELS"), rec.lastCompleted.get("MERGE_LEVELS"));
        assertEquals("REFINE did not reach total",
                rec.lastTotal.get("REFINE"), rec.lastCompleted.get("REFINE"));
        assertEquals("MERGE_LEVELS total should be the live-node count",
                (long) numSources * numVectorsPerGraph, (long) rec.lastTotal.get("MERGE_LEVELS"));

        // Throttle invoked with real byte amounts, and every acquire paired with exactly one close.
        assertTrue("acquire never called", rec.acquires.get() > 0);
        assertTrue("acquire amounts were all zero", rec.bytesAcquired.get() > 0);
        assertEquals("every grant must be closed exactly once", rec.acquires.get(), rec.closes.get());

        // Output is a valid, complete graph.
        try (ReaderSupplier rs = ReaderSupplierFactory.open(outputPath)) {
            var compactGraph = OnDiskGraphIndex.load(rs);
            assertEquals(numSources * numVectorsPerGraph, compactGraph.size(0));
        }
        for (var rs : rss) rs.close();
    }

    @Test
    public void testSemaphoreLimiterReleasesAllGrantsWithoutDeadlock() throws Exception {
        List<ReaderSupplier> rss = new ArrayList<>();
        var compactor = newAllLiveCompactor(rss);
        var sem = new SemaphoreBytesLimiter(64 * 1024 * 1024); // ample: exercises release-on-close, not undersized
        compactor.setProgressLimiter(sem);

        var outputPath = testDirectory.resolve("test_compact_semaphore");
        final Throwable[] thrown = new Throwable[1];
        Thread t = new Thread(() -> {
            try { compactor.compact(outputPath); }
            catch (Throwable e) { thrown[0] = e; }
        }, "compact-semaphore");
        t.start();
        t.join(90_000);

        assertFalse("compaction did not finish within 90s — possible throttle deadlock", t.isAlive());
        if (thrown[0] != null) throw new AssertionError("compaction failed under semaphore limiter", thrown[0]);
        assertEquals("all semaphore grants must be released (acquire/close paired)", 0, sem.open.get());

        try (ReaderSupplier rs = ReaderSupplierFactory.open(outputPath)) {
            var compactGraph = OnDiskGraphIndex.load(rs);
            assertEquals(numSources * numVectorsPerGraph, compactGraph.size(0));
        }
        for (var rs : rss) rs.close();
    }

    @Test
    public void testCompactAtNonZeroStartOffset() throws Exception {
        List<ReaderSupplier> rss = new ArrayList<>();
        var compactor = newAllLiveCompactor(rss);

        // Reserve a prefix (as an embedder would for its container header) and record its bytes.
        var outputPath = testDirectory.resolve("test_compact_offset");
        byte[] prefix = new byte[64];
        for (int i = 0; i < prefix.length; i++) prefix[i] = (byte) (0xA0 + (i % 16));
        Files.write(outputPath, prefix);
        long startOffset = prefix.length;

        compactor.compact(outputPath, startOffset);

        // The reserved prefix must be untouched by the no-copy write.
        byte[] afterPrefix = Arrays.copyOf(Files.readAllBytes(outputPath), prefix.length);
        assertArrayEquals("compaction clobbered the reserved prefix", prefix, afterPrefix);

        // The graph loads from startOffset and is complete + searchable (proves refine wrote valid
        // records at the base-shifted offsets).
        try (ReaderSupplier rs = ReaderSupplierFactory.open(outputPath)) {
            var compactGraph = OnDiskGraphIndex.load(rs, startOffset);
            assertEquals(numSources * numVectorsPerGraph, compactGraph.size(0));

            GraphSearcher searcher = new GraphSearcher(compactGraph);
            for (int i = 0; i < 5; i++) {
                VectorFloat<?> q = allVecs.get(randomIntBetween(0, allVecs.size() - 1));
                SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(q, similarityFunction, allravv);
                SearchResult r = searcher.search(ssp, 10, Bits.ALL);
                assertTrue("search from offset-loaded graph returned nothing", r.getNodes().length > 0);
            }
            searcher.close();
        }
        for (var rs : rss) rs.close();
    }

    @Test
    public void testCompactToFileDestination() throws Exception {
        List<ReaderSupplier> rss = new ArrayList<>();
        var compactor = newAllLiveCompactor(rss);
        var path = testDirectory.resolve("dest_tofile");

        long bodyLength = compactor.compact(CompactionDestination.toFile(path));
        assertEquals("returned body length must equal the standalone file size",
                Files.size(path), bodyLength);
        try (ReaderSupplier out = ReaderSupplierFactory.open(path)) {
            assertEquals(numSources * numVectorsPerGraph, OnDiskGraphIndex.load(out).size(0));
        }
        for (var rs : rss) rs.close();
    }

    @Test
    public void testCompactToContainerDestinationCommitsAndPreservesPrefix() throws Exception {
        List<ReaderSupplier> rss = new ArrayList<>();
        var compactor = newAllLiveCompactor(rss);
        var container = testDirectory.resolve("dest_container");
        byte[] prefix = new byte[32];
        for (int i = 0; i < prefix.length; i++) prefix[i] = (byte) (0xC0 + (i % 16));
        Files.write(container, prefix);
        final long base = prefix.length;

        final long[] committed = { -1 };
        final boolean[] closed = { false };
        CompactionDestination dest = () -> new CompactionDestination.Target() {
            public Path file() { return container; }
            public long startOffset() { return base; }
            public void commit(long bodyLength) { committed[0] = bodyLength; }
            public void close() { closed[0] = true; }
        };

        long returned = compactor.compact(dest);
        assertTrue("commit must have been called", committed[0] >= 0);
        assertEquals("commit bodyLength must equal the returned value", committed[0], returned);
        assertTrue("target must be closed", closed[0]);
        assertEquals("body length must be file size minus base", Files.size(container) - base, returned);

        byte[] afterPrefix = Arrays.copyOf(Files.readAllBytes(container), prefix.length);
        assertArrayEquals("the reserved prefix must be preserved", prefix, afterPrefix);
        try (ReaderSupplier out = ReaderSupplierFactory.open(container)) {
            assertEquals(numSources * numVectorsPerGraph, OnDiskGraphIndex.load(out, base).size(0));
        }
        for (var rs : rss) rs.close();
    }

    @Test
    public void testCompactToDestinationAbortsWithoutCommitOnFailure() throws Exception {
        List<ReaderSupplier> rss = new ArrayList<>();
        var compactor = newAllLiveCompactor(rss);
        // Parent directory does not exist, so the graph writer fails partway.
        final var badFile = testDirectory.resolve("no_such_dir").resolve("graph");

        final boolean[] committed = { false };
        final boolean[] closed = { false };
        CompactionDestination dest = () -> new CompactionDestination.Target() {
            public Path file() { return badFile; }
            public long startOffset() { return 0; }
            public void commit(long bodyLength) { committed[0] = true; }
            public void close() { closed[0] = true; }
        };

        try {
            compactor.compact(dest);
            fail("expected compaction to fail for an unwritable destination");
        } catch (Exception expected) {
            // ok
        }
        assertFalse("commit must NOT be called on failure", committed[0]);
        assertTrue("close (abort) must run on failure", closed[0]);
        for (var rs : rss) rs.close();
    }

    /**
     * Tests compaction with deleted nodes.
     * Verifies that deleted nodes are properly excluded from the compacted graph.
     */
    @Test
    public void testCompactWithDeletions() throws Exception {
        List<OnDiskGraphIndex> graphs = new ArrayList<>();
        List<ReaderSupplier> rss = new ArrayList<>();
        List<FixedBitSet> liveNodes = new ArrayList<>();
        List<OrdinalMapper> remappers = new ArrayList<>();

        for(int i = 0; i < numSources; ++i) {
            var outputPath = testDirectory.resolve("test_graph_" + i);
            rss.add(ReaderSupplierFactory.open(outputPath.toAbsolutePath()));
            var onDiskGraph = OnDiskGraphIndex.load(rss.get(i));
            graphs.add(onDiskGraph);
        }

        // Mark some nodes as deleted (not live)
        int globalOrdinal = 0;
        int totalLiveNodes = 0;
        Set<Integer> deletedGlobalOrdinals = new HashSet<>();

        for (int n = 0; n < numSources; n++) {
            Map<Integer, Integer> map = new HashMap<>();
            var lives = new FixedBitSet(numVectorsPerGraph);

            // Delete every 5th node
            for (int i = 0; i < numVectorsPerGraph; i++) {
                int originalGlobalOrdinal = n * numVectorsPerGraph + i;
                if (i % 5 != 0) {
                    lives.set(i);
                    map.put(i, globalOrdinal++);
                    totalLiveNodes++;
                } else {
                    deletedGlobalOrdinals.add(originalGlobalOrdinal);
                }
            }

            remappers.add(new OrdinalMapper.MapMapper(map));
            liveNodes.add(lives);
        }

        var compactor = new OnDiskGraphIndexCompactor(graphs, liveNodes, remappers, similarityFunction, null, -1);
        var outputPath = testDirectory.resolve("test_compact_with_deletions");

        compactor.compact(outputPath);

        ReaderSupplier rs = ReaderSupplierFactory.open(outputPath);
        var compactGraph = OnDiskGraphIndex.load(rs);

        // Verify the compacted graph has the correct size (excluding deleted nodes)
        assertEquals("Compacted graph size should equal live nodes", totalLiveNodes, compactGraph.size(0));

        // Verify search functionality still works
        GraphSearcher searcher = new GraphSearcher(compactGraph);
        var query = allVecs.get(randomIntBetween(0, allVecs.size() - 1));
        SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(query, similarityFunction, allravv);
        SearchResult result = searcher.search(ssp, 10, Bits.ALL);

        // Verify we get results and they're all valid
        assertTrue("Should return some results", result.getNodes().length > 0);

        searcher.close();
    }

    /**
     * Tests compaction with custom ordinal mappings.
     * Verifies that vectors are correctly placed at their mapped ordinals.
     */
    @Test
    public void testOrdinalMapping() throws Exception {
        List<OnDiskGraphIndex> graphs = new ArrayList<>();
        List<ReaderSupplier> rss = new ArrayList<>();
        List<FixedBitSet> liveNodes = new ArrayList<>();
        List<OrdinalMapper> remappers = new ArrayList<>();

        for(int i = 0; i < numSources; ++i) {
            var outputPath = testDirectory.resolve("test_graph_" + i);
            rss.add(ReaderSupplierFactory.open(outputPath.toAbsolutePath()));
            var onDiskGraph = OnDiskGraphIndex.load(rss.get(i));
            graphs.add(onDiskGraph);
        }

        // Create custom ordinal mappings (non-sequential)
        int globalOrdinal = 0;
        List<Map<Integer, Integer>> mappingList = new ArrayList<>();

        for (int n = 0; n < numSources; n++) {
            Map<Integer, Integer> map = new HashMap<>();
            // Use a custom mapping: reverse order for even sources, normal order for odd
            if (n % 2 == 0) {
                for (int i = 0; i < numVectorsPerGraph; i++) {
                    int newOrdinal = globalOrdinal + (numVectorsPerGraph - 1 - i);
                    map.put(i, newOrdinal);
                }
                globalOrdinal += numVectorsPerGraph;
            } else {
                for (int i = 0; i < numVectorsPerGraph; i++) {
                    map.put(i, globalOrdinal++);
                }
            }
            mappingList.add(map);
            remappers.add(new OrdinalMapper.MapMapper(map));

            var lives = new FixedBitSet(numVectorsPerGraph);
            lives.set(0, numVectorsPerGraph);
            liveNodes.add(lives);
        }

        var compactor = new OnDiskGraphIndexCompactor(graphs, liveNodes, remappers, similarityFunction, null, -1);
        var outputPath = testDirectory.resolve("test_compact_with_ordinal_mapping");

        compactor.compact(outputPath);

        ReaderSupplier rs = ReaderSupplierFactory.open(outputPath);
        var compactGraph = OnDiskGraphIndex.load(rs);

        // Verify the graph was created with correct ordinal mapping
        assertEquals("Compacted graph should have all nodes", numSources * numVectorsPerGraph, compactGraph.size(0));

        // Verify that the vectors are correctly mapped in the compacted graph
        var compactView = compactGraph.getView();

        // Check a few vectors to ensure they're at the correct ordinals
        for (int sourceIdx = 0; sourceIdx < numSources; sourceIdx++) {
            Map<Integer, Integer> mapping = mappingList.get(sourceIdx);
            // Check first, middle, and last nodes
            int[] testIndices = {0, numVectorsPerGraph / 2, numVectorsPerGraph - 1};

            for (int localIdx : testIndices) {
                int expectedGlobalOrdinal = mapping.get(localIdx);
                int originalVectorIdx = sourceIdx * numVectorsPerGraph + localIdx;

                VectorFloat<?> originalVec = allVecs.get(originalVectorIdx);
                VectorFloat<?> compactVec = vectorTypeSupport.createFloatVector(dimension);
                compactView.getVectorInto(expectedGlobalOrdinal, compactVec, 0);

                // Verify the vectors match (use similarity for normalized vectors)
                float similarity = similarityFunction.compare(originalVec, compactVec);
                assertTrue(String.format("Vector at ordinal %d should match (similarity=%.4f)",
                                       expectedGlobalOrdinal, similarity),
                         similarity > 0.9999f);
            }
        }
    }

    /**
     * Tests compaction with both deletions and custom ordinal mappings combined.
     * Verifies that both features work correctly together.
     */
    @Test
    public void testDeletionsAndOrdinalMapping() throws Exception {
        List<OnDiskGraphIndex> graphs = new ArrayList<>();
        List<ReaderSupplier> rss = new ArrayList<>();
        List<FixedBitSet> liveNodes = new ArrayList<>();
        List<OrdinalMapper> remappers = new ArrayList<>();

        for(int i = 0; i < numSources; ++i) {
            var outputPath = testDirectory.resolve("test_graph_" + i);
            rss.add(ReaderSupplierFactory.open(outputPath.toAbsolutePath()));
            var onDiskGraph = OnDiskGraphIndex.load(rss.get(i));
            graphs.add(onDiskGraph);
        }

        // Combine deletions with custom ordinal mapping
        int globalOrdinal = 0;
        int totalLiveNodes = 0;
        List<Map<Integer, Integer>> mappingList = new ArrayList<>();

        for (int n = 0; n < numSources; n++) {
            Map<Integer, Integer> map = new HashMap<>();
            var lives = new FixedBitSet(numVectorsPerGraph);

            // Delete every 4th node
            for (int i = 0; i < numVectorsPerGraph; i++) {
                if (i % 4 != 0) {
                    lives.set(i);
                    map.put(i, globalOrdinal++);
                    totalLiveNodes++;
                }
            }

            mappingList.add(map);
            remappers.add(new OrdinalMapper.MapMapper(map));
            liveNodes.add(lives);
        }

        var compactor = new OnDiskGraphIndexCompactor(graphs, liveNodes, remappers, similarityFunction, null, -1);
        var outputPath = testDirectory.resolve("test_compact_deletions_and_mapping");

        compactor.compact(outputPath);

        ReaderSupplier rs = ReaderSupplierFactory.open(outputPath);
        var compactGraph = OnDiskGraphIndex.load(rs);

        // Verify correct size
        assertEquals("Compacted graph should only contain live nodes", totalLiveNodes, compactGraph.size(0));

        // Verify a sample of vectors are at correct ordinals
        var compactView = compactGraph.getView();
        int samplesVerified = 0;
        for (int sourceIdx = 0; sourceIdx < numSources; sourceIdx++) {
            Map<Integer, Integer> mapping = mappingList.get(sourceIdx);

            // Check a few live nodes per source
            for (int localIdx = 1; localIdx < numVectorsPerGraph && samplesVerified < 20; localIdx++) {
                if (localIdx % 4 == 0) continue; // Skip deleted nodes

                int expectedGlobalOrdinal = mapping.get(localIdx);
                int originalVectorIdx = sourceIdx * numVectorsPerGraph + localIdx;

                VectorFloat<?> originalVec = allVecs.get(originalVectorIdx);
                VectorFloat<?> compactVec = vectorTypeSupport.createFloatVector(dimension);
                compactView.getVectorInto(expectedGlobalOrdinal, compactVec, 0);

                // Verify the vectors match using similarity
                float similarity = similarityFunction.compare(originalVec, compactVec);
                assertTrue(String.format("Vector at ordinal %d should match (similarity=%.4f)",
                                       expectedGlobalOrdinal, similarity),
                         similarity > 0.9999f);
                samplesVerified++;
            }
        }

        // Verify search functionality
        GraphSearcher searcher = new GraphSearcher(compactGraph);
        var query = allVecs.get(randomIntBetween(0, allVecs.size() - 1));
        SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(query, similarityFunction, allravv);
        SearchResult result = searcher.search(ssp, 10, Bits.ALL);

        assertTrue("Search should return results", result.getNodes().length > 0);

        searcher.close();
    }

    // -----------------------------------------------------------------------------------------
    // Tests for non-fused compressed-sidecar compaction (compact(graphPath, compressedPath))
    // -----------------------------------------------------------------------------------------

    /**
     * Happy path: merge two sources whose PQ codes are shipped as a non-fused {@link PQVectors}
     * sidecar, and verify both outputs — graph and compressed sidecar — are produced correctly.
     * Asserts:
     * <ul>
     *     <li>merged graph has the expected node count and per-ordinal vector values,</li>
     *     <li>merged sidecar loads as PQVectors with the same {@code count}, subspace count, and
     *         cluster count as the inputs,</li>
     *     <li>each merged code decodes to a vector close to the original raw vector (within PQ
     *         reconstruction error).</li>
     * </ul>
     */
    @Test
    public void testCompactWithCompressedSidecar() throws Exception {
        int dim = 16;
        int n = 32;     // nodes per source
        int M = 8;      // PQ subspaces
        int clusters = 16;  // small for fast test
        VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;

        List<VectorFloat<?>> vecs0 = createRandomVectors(n, dim);
        List<VectorFloat<?>> vecs1 = createRandomVectors(n, dim);

        Path graph0 = buildSimpleSourceGraph(vecs0, dim, vsf, "sidecar_src_0");
        Path graph1 = buildSimpleSourceGraph(vecs1, dim, vsf, "sidecar_src_1");

        ReaderSupplier rs0 = ReaderSupplierFactory.open(graph0);
        ReaderSupplier rs1 = ReaderSupplierFactory.open(graph1);
        OnDiskGraphIndex g0 = OnDiskGraphIndex.load(rs0);
        OnDiskGraphIndex g1 = OnDiskGraphIndex.load(rs1);

        // Per-source PQVectors — the non-fused sidecar input.
        RandomAccessVectorValues ravv0 = new ListRandomAccessVectorValues(vecs0, dim);
        RandomAccessVectorValues ravv1 = new ListRandomAccessVectorValues(vecs1, dim);
        ProductQuantization pq0 = ProductQuantization.compute(ravv0, M, clusters, true, UNWEIGHTED, simdExecutor, parallelExecutor);
        ProductQuantization pq1 = ProductQuantization.compute(ravv1, M, clusters, true, UNWEIGHTED, simdExecutor, parallelExecutor);
        PQVectors pqv0 = (PQVectors) pq0.encodeAll(ravv0, simdExecutor);
        PQVectors pqv1 = (PQVectors) pq1.encodeAll(ravv1, simdExecutor);

        // Identity remapping: source 0 -> [0, n), source 1 -> [n, 2n)
        Map<Integer, Integer> map0 = new HashMap<>();
        Map<Integer, Integer> map1 = new HashMap<>();
        for (int i = 0; i < n; i++) {
            map0.put(i, i);
            map1.put(i, n + i);
        }
        FixedBitSet live0 = new FixedBitSet(n); live0.set(0, n);
        FixedBitSet live1 = new FixedBitSet(n); live1.set(0, n);

        var compactor = new OnDiskGraphIndexCompactor(
                List.of(g0, g1),
                List.<io.github.jbellis.jvector.quantization.CompressedVectors>of(pqv0, pqv1),
                List.of(live0, live1),
                List.of(new OrdinalMapper.MapMapper(map0), new OrdinalMapper.MapMapper(map1)),
                vsf, null, -1);

        Path graphOut = testDirectory.resolve("sidecar_graph_out");
        Path compressedOut = testDirectory.resolve("sidecar_pq_out");
        compactor.compact(graphOut, compressedOut);

        // ---- Verify merged graph ----
        ReaderSupplier rsOut = ReaderSupplierFactory.open(graphOut);
        OnDiskGraphIndex compacted = OnDiskGraphIndex.load(rsOut);
        assertEquals("compacted graph node count", 2 * n, compacted.size(0));

        var view = compacted.getView();
        VectorFloat<?> buf = vectorTypeSupport.createFloatVector(dim);
        for (int i = 0; i < n; i++) {
            view.getVectorInto(i, buf, 0);
            assertVecEquals(vecs0.get(i), buf, i);
            view.getVectorInto(n + i, buf, 0);
            assertVecEquals(vecs1.get(i), buf, n + i);
        }

        // ---- Verify merged compressed sidecar ----
        try (var rsCompressed = ReaderSupplierFactory.open(compressedOut); var reader = rsCompressed.get()) {
            PQVectors mergedPqv = PQVectors.load(reader);
            assertEquals("merged PQVectors count", 2 * n, mergedPqv.count());
            ProductQuantization mergedPQ = mergedPqv.getCompressor();
            assertEquals("merged PQ subspaceCount", M, mergedPQ.getSubspaceCount());
            assertEquals("merged PQ clusterCount", clusters, mergedPQ.getClusterCount());
            assertEquals("merged PQ compressedVectorSize", M, mergedPQ.compressedVectorSize());

            // Each merged code should decode to a vector close to the original (PQ is lossy
            // but with these params reconstruction error stays bounded). We check that the
            // re-encoded code matches the stored code — i.e. encoding is consistent under the
            // retrained codebook.
            VectorFloat<?> reEncoded = vectorTypeSupport.createFloatVector(dim);
            io.github.jbellis.jvector.vector.types.ByteSequence<?> tmpCode = vectorTypeSupport.createByteSequence(M);
            for (int i = 0; i < n; i++) {
                mergedPQ.encodeTo(vecs0.get(i), tmpCode);
                io.github.jbellis.jvector.vector.types.ByteSequence<?> stored = mergedPqv.get(i);
                for (int b = 0; b < M; b++) {
                    assertEquals("ord " + i + " code byte " + b, tmpCode.get(b), stored.get(b));
                }
                mergedPQ.encodeTo(vecs1.get(i), tmpCode);
                stored = mergedPqv.get(n + i);
                for (int b = 0; b < M; b++) {
                    assertEquals("ord " + (n + i) + " code byte " + b, tmpCode.get(b), stored.get(b));
                }
            }
        }
    }

    /**
     * Validation: combining {@code sourceCompressed} with sources that already carry FUSED_PQ
     * inline must throw, since the two are mutually exclusive ways to ship PQ codes.
     */
    @Test
    public void testCompactCompressedSidecarRejectsFusedPQ() throws Exception {
        // Reuse the FusedPQ sources built by setup().
        ReaderSupplier rs0 = ReaderSupplierFactory.open(testDirectory.resolve("test_graph_0"));
        ReaderSupplier rs1 = ReaderSupplierFactory.open(testDirectory.resolve("test_graph_1"));
        OnDiskGraphIndex g0 = OnDiskGraphIndex.load(rs0);
        OnDiskGraphIndex g1 = OnDiskGraphIndex.load(rs1);

        // Throwaway PQVectors just to exercise the validation; values don't matter.
        var ravv = new ListRandomAccessVectorValues(allVecs.subList(0, numVectorsPerGraph), dimension);
        ProductQuantization pq = ProductQuantization.compute(ravv, 8, 16, true, UNWEIGHTED, simdExecutor, parallelExecutor);
        PQVectors pqv0 = (PQVectors) pq.encodeAll(ravv, simdExecutor);
        PQVectors pqv1 = (PQVectors) pq.encodeAll(ravv, simdExecutor);

        Map<Integer, Integer> map0 = new HashMap<>();
        Map<Integer, Integer> map1 = new HashMap<>();
        for (int i = 0; i < numVectorsPerGraph; i++) {
            map0.put(i, i);
            map1.put(i, numVectorsPerGraph + i);
        }
        FixedBitSet live0 = new FixedBitSet(numVectorsPerGraph); live0.set(0, numVectorsPerGraph);
        FixedBitSet live1 = new FixedBitSet(numVectorsPerGraph); live1.set(0, numVectorsPerGraph);

        try {
            new OnDiskGraphIndexCompactor(
                    List.of(g0, g1),
                    List.<io.github.jbellis.jvector.quantization.CompressedVectors>of(pqv0, pqv1),
                    List.of(live0, live1),
                    List.of(new OrdinalMapper.MapMapper(map0), new OrdinalMapper.MapMapper(map1)),
                    similarityFunction, null, -1);
            org.junit.Assert.fail("expected IllegalArgumentException for FUSED_PQ + sourceCompressed");
        } catch (IllegalArgumentException expected) {
            assertTrue("error message mentions FUSED_PQ",
                    expected.getMessage().toLowerCase().contains("fused_pq")
                    || expected.getMessage().toLowerCase().contains("fused pq"));
        }
    }

    /**
     * Validation: {@code sourceCompressed.size()} must equal {@code sources.size()}.
     */
    @Test
    public void testCompactCompressedSidecarRejectsSizeMismatch() throws Exception {
        int dim = 8;
        int n = 32;     // need >= clusters for k-means training
        int clusters = 16;
        VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;

        List<VectorFloat<?>> vecs0 = createRandomVectors(n, dim);
        List<VectorFloat<?>> vecs1 = createRandomVectors(n, dim);

        Path graph0 = buildSimpleSourceGraph(vecs0, dim, vsf, "size_src_0");
        Path graph1 = buildSimpleSourceGraph(vecs1, dim, vsf, "size_src_1");

        OnDiskGraphIndex g0 = OnDiskGraphIndex.load(ReaderSupplierFactory.open(graph0));
        OnDiskGraphIndex g1 = OnDiskGraphIndex.load(ReaderSupplierFactory.open(graph1));

        RandomAccessVectorValues ravv0 = new ListRandomAccessVectorValues(vecs0, dim);
        ProductQuantization pq = ProductQuantization.compute(ravv0, 4, clusters, true, UNWEIGHTED, simdExecutor, parallelExecutor);
        PQVectors pqv0 = (PQVectors) pq.encodeAll(ravv0, simdExecutor);

        Map<Integer, Integer> map0 = new HashMap<>();
        Map<Integer, Integer> map1 = new HashMap<>();
        for (int i = 0; i < n; i++) { map0.put(i, i); map1.put(i, n + i); }
        FixedBitSet live = new FixedBitSet(n); live.set(0, n);

        try {
            new OnDiskGraphIndexCompactor(
                    List.of(g0, g1),
                    List.<io.github.jbellis.jvector.quantization.CompressedVectors>of(pqv0),  // size 1 vs sources size 2
                    List.of(live, live),
                    List.of(new OrdinalMapper.MapMapper(map0), new OrdinalMapper.MapMapper(map1)),
                    vsf, null, -1);
            org.junit.Assert.fail("expected IllegalArgumentException for size mismatch");
        } catch (IllegalArgumentException expected) {
            assertTrue("error message mentions size",
                    expected.getMessage().toLowerCase().contains("size"));
        }
    }

    /**
     * Calling the two-arg compact() without supplying {@code sourceCompressed} must fail —
     * there is no source for the merged sidecar.
     */
    @Test
    public void testCompactTwoArgRequiresSourceCompressed() throws Exception {
        int dim = 8;
        int n = 8;
        VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;

        List<VectorFloat<?>> vecs0 = createRandomVectors(n, dim);
        List<VectorFloat<?>> vecs1 = createRandomVectors(n, dim);
        Path graph0 = buildSimpleSourceGraph(vecs0, dim, vsf, "noarg_src_0");
        Path graph1 = buildSimpleSourceGraph(vecs1, dim, vsf, "noarg_src_1");

        OnDiskGraphIndex g0 = OnDiskGraphIndex.load(ReaderSupplierFactory.open(graph0));
        OnDiskGraphIndex g1 = OnDiskGraphIndex.load(ReaderSupplierFactory.open(graph1));

        Map<Integer, Integer> map0 = new HashMap<>();
        Map<Integer, Integer> map1 = new HashMap<>();
        for (int i = 0; i < n; i++) { map0.put(i, i); map1.put(i, n + i); }
        FixedBitSet live = new FixedBitSet(n); live.set(0, n);

        // Use the legacy 5-arg constructor — sourceCompressed defaults to null.
        var compactor = new OnDiskGraphIndexCompactor(
                List.of(g0, g1),
                List.of(live, live),
                List.of(new OrdinalMapper.MapMapper(map0), new OrdinalMapper.MapMapper(map1)),
                vsf, null, -1);

        Path graphOut = testDirectory.resolve("noarg_graph_out");
        Path compressedOut = testDirectory.resolve("noarg_pq_out");
        try {
            compactor.compact(graphOut, compressedOut);
            org.junit.Assert.fail("expected IllegalStateException without sourceCompressed");
        } catch (IllegalStateException expected) {
            assertTrue("error message mentions sourceCompressed",
                    expected.getMessage().toLowerCase().contains("sourcecompressed"));
        }
    }

    /**
     * Compaction with deletions: only live nodes appear in the merged sidecar at their remapped
     * ordinals, and the count matches the number of live nodes (dense merged ordinal range).
     */
    @Test
    public void testCompactCompressedSidecarWithDeletions() throws Exception {
        int dim = 16;
        int n = 16;
        int M = 8;
        int clusters = 16;
        VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;

        List<VectorFloat<?>> vecs0 = createRandomVectors(n, dim);
        List<VectorFloat<?>> vecs1 = createRandomVectors(n, dim);
        Path graph0 = buildSimpleSourceGraph(vecs0, dim, vsf, "delsidecar_src_0");
        Path graph1 = buildSimpleSourceGraph(vecs1, dim, vsf, "delsidecar_src_1");

        OnDiskGraphIndex g0 = OnDiskGraphIndex.load(ReaderSupplierFactory.open(graph0));
        OnDiskGraphIndex g1 = OnDiskGraphIndex.load(ReaderSupplierFactory.open(graph1));

        RandomAccessVectorValues ravv0 = new ListRandomAccessVectorValues(vecs0, dim);
        RandomAccessVectorValues ravv1 = new ListRandomAccessVectorValues(vecs1, dim);
        ProductQuantization pq0 = ProductQuantization.compute(ravv0, M, clusters, true, UNWEIGHTED, simdExecutor, parallelExecutor);
        ProductQuantization pq1 = ProductQuantization.compute(ravv1, M, clusters, true, UNWEIGHTED, simdExecutor, parallelExecutor);
        PQVectors pqv0 = (PQVectors) pq0.encodeAll(ravv0, simdExecutor);
        PQVectors pqv1 = (PQVectors) pq1.encodeAll(ravv1, simdExecutor);

        // Keep even nodes live; map them densely.
        FixedBitSet live0 = new FixedBitSet(n);
        FixedBitSet live1 = new FixedBitSet(n);
        Map<Integer, Integer> map0 = new HashMap<>();
        Map<Integer, Integer> map1 = new HashMap<>();
        int next = 0;
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) { live0.set(i); map0.put(i, next++); }
        }
        int firstSourceCount = next;
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) { live1.set(i); map1.put(i, next++); }
        }
        int totalLive = next;

        var compactor = new OnDiskGraphIndexCompactor(
                List.of(g0, g1),
                List.<io.github.jbellis.jvector.quantization.CompressedVectors>of(pqv0, pqv1),
                List.of(live0, live1),
                List.of(new OrdinalMapper.MapMapper(map0), new OrdinalMapper.MapMapper(map1)),
                vsf, null, -1);

        Path graphOut = testDirectory.resolve("delsidecar_graph_out");
        Path compressedOut = testDirectory.resolve("delsidecar_pq_out");
        compactor.compact(graphOut, compressedOut);

        // Verify graph
        OnDiskGraphIndex compacted = OnDiskGraphIndex.load(ReaderSupplierFactory.open(graphOut));
        assertEquals("compacted graph live count", totalLive, compacted.size(0));

        // Verify sidecar: count matches dense live total; each live ordinal's code matches a
        // fresh re-encoding of the corresponding raw vector under the retrained codebook.
        try (var rsCompressed = ReaderSupplierFactory.open(compressedOut); var reader = rsCompressed.get()) {
            PQVectors mergedPqv = PQVectors.load(reader);
            assertEquals("merged sidecar count", totalLive, mergedPqv.count());

            ProductQuantization mergedPQ = mergedPqv.getCompressor();
            io.github.jbellis.jvector.vector.types.ByteSequence<?> tmp = vectorTypeSupport.createByteSequence(M);
            for (int i = 0; i < n; i++) {
                if (i % 2 != 0) continue;
                mergedPQ.encodeTo(vecs0.get(i), tmp);
                io.github.jbellis.jvector.vector.types.ByteSequence<?> stored = mergedPqv.get(map0.get(i));
                for (int b = 0; b < M; b++) {
                    assertEquals("source 0 ord " + i + " byte " + b, tmp.get(b), stored.get(b));
                }
                mergedPQ.encodeTo(vecs1.get(i), tmp);
                stored = mergedPqv.get(map1.get(i));
                for (int b = 0; b < M; b++) {
                    assertEquals("source 1 ord " + i + " byte " + b, tmp.get(b), stored.get(b));
                }
            }
            // sanity check on dense layout
            assertEquals("first-source live count", firstSourceCount, n / 2);
        }
    }
}
