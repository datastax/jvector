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
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntFunction;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertFalse;

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
                vsf, null);

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
                vsf, null);

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
                vsf, null);

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

        var compactor = new OnDiskGraphIndexCompactor(graphs, liveNodes, remappers, similarityFunction, null);
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

    /**
     * Compaction with compactor-assigned similarity ordinals: verifies the effective mapping is
     * a bijection with a working newToOld round-trip, and that search recall over the reordered
     * graph (results translated back through effectiveRemappers) matches the golden build.
     */
    @Test
    public void testCompactWithSimilarityOrdinals() throws Exception {
        List<OnDiskGraphIndex> graphs = new ArrayList<>();
        List<ReaderSupplier> rss = new ArrayList<>();
        List<FixedBitSet> liveNodes = new ArrayList<>();
        List<OrdinalMapper> remappers = new ArrayList<>();

        for (int i = 0; i < numSources; ++i) {
            var sourcePath = testDirectory.resolve("test_graph_" + i);
            rss.add(ReaderSupplierFactory.open(sourcePath.toAbsolutePath()));
            graphs.add(OnDiskGraphIndex.load(rss.get(i)));
        }
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

        var compactor = new OnDiskGraphIndexCompactor(graphs, liveNodes, remappers, similarityFunction, null);
        compactor.setSimilarityOrdinals(true);
        int topK = 10;

        var outputPath = testDirectory.resolve("test_compact_simord_graph");
        List<VectorFloat<?>> queries = new ArrayList<>();
        for (int i = 0; i < numQueries; ++i) {
            queries.add(allVecs.get(randomIntBetween(0, allVecs.size() - 1)));
        }
        List<SearchResult> goldenResults = searchFromAll(queries, topK);
        List<List<Integer>> groundTruth = buildGT(queries, topK);

        compactor.compact(outputPath);

        // The mapping in effect must be a total bijection with a working reverse.
        var effective = compactor.effectiveRemappers();
        int total = numSources * numVectorsPerGraph;
        int[] newToDataset = new int[total];
        boolean[] seen = new boolean[total];
        for (int src = 0; src < numSources; src++) {
            for (int old = 0; old < numVectorsPerGraph; old++) {
                int n = effective.get(src).oldToNew(old);
                assertTrue("new ordinal in range: " + n, n >= 0 && n < total);
                assertFalse("no ordinal collisions", seen[n]);
                seen[n] = true;
                assertEquals("newToOld round-trip", old, effective.get(src).newToOld(n));
                newToDataset[n] = src * numVectorsPerGraph + old;
            }
        }

        ReaderSupplier rs = ReaderSupplierFactory.open(outputPath);
        var compactGraph = OnDiskGraphIndex.load(rs);
        assertEquals(total, compactGraph.size(0));

        // Score by graph ordinal: reorder the exact vectors into new-ordinal order.
        List<VectorFloat<?>> reordered = new ArrayList<>(total);
        for (int n = 0; n < total; n++) {
            reordered.add(allVecs.get(newToDataset[n]));
        }
        var reorderedRavv = new ListRandomAccessVectorValues(reordered, allVecs.get(0).length());

        GraphSearcher searcher = new GraphSearcher(compactGraph);
        int hits = 0;
        int possible = 0;
        for (int qi = 0; qi < queries.size(); qi++) {
            SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(queries.get(qi), similarityFunction, reorderedRavv);
            SearchResult sr = searcher.search(ssp, topK, Bits.ALL);
            var gt = groundTruth.get(qi);
            for (var ns : sr.getNodes()) {
                if (gt.contains(newToDataset[ns.node])) {
                    hits++;
                }
            }
            possible += topK;
        }
        double compactRecall = (double) hits / possible;
        double goldenRecall = AccuracyMetrics.recallFromSearchResults(groundTruth, goldenResults, topK, topK);
        System.out.printf("SimilarityOrdinals compact recall: %.4f (golden %.4f)%n", compactRecall, goldenRecall);
        assertTrue(String.format("similarity-ordinal recall (%.4f) should be comparable to golden (%.4f)",
                        compactRecall, goldenRecall),
                Math.abs(goldenRecall - compactRecall) < 0.2);
        assertTrue("similarity-ordinal recall should be at least 0.2, got " + compactRecall,
                compactRecall >= 0.2);
        searcher.close();
    }

    /**
     * Tests the retained-only fast path with a heavily skewed merge: the small source's few
     * searches can only offer reverse candidates to a bounded set of large-source nodes, so
     * most large-source nodes must take the fast path (their merged edges are exactly their
     * retained edges, remapped). Verifies the path fired, that fast-path records preserve the
     * source adjacency, and that the merged graph searches sanely.
     */
    @Test
    public void testCompactRetainedOnlyFastPath() throws Exception {
        int dim = 16;
        VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;
        List<VectorFloat<?>> smallVecs = createRandomVectors(8, dim);
        List<VectorFloat<?>> bigVecs = createRandomVectors(300, dim);

        Path smallPath = buildSimpleSourceGraph(smallVecs, dim, vsf, "fastpath_small");
        Path bigPath = buildSimpleSourceGraph(bigVecs, dim, vsf, "fastpath_big");

        try (ReaderSupplier smallRs = ReaderSupplierFactory.open(smallPath);
             ReaderSupplier bigRs = ReaderSupplierFactory.open(bigPath)) {
            var smallGraph = OnDiskGraphIndex.load(smallRs);
            var bigGraph = OnDiskGraphIndex.load(bigRs);

            List<OnDiskGraphIndex> graphs = new ArrayList<>(List.of(smallGraph, bigGraph));
            List<FixedBitSet> live = new ArrayList<>();
            var liveSmall = new FixedBitSet(smallVecs.size());
            liveSmall.set(0, smallVecs.size());
            var liveBig = new FixedBitSet(bigVecs.size());
            liveBig.set(0, bigVecs.size());
            live.add(liveSmall);
            live.add(liveBig);
            List<OrdinalMapper> remappers = new ArrayList<>(List.of(
                    new OrdinalMapper.OffsetMapper(0, smallVecs.size()),
                    new OrdinalMapper.OffsetMapper(smallVecs.size(), bigVecs.size())));

            var compactor = new OnDiskGraphIndexCompactor(graphs, live, remappers, vsf, null);
            var outputPath = testDirectory.resolve("fastpath_compacted");
            compactor.compact(outputPath);

            assertTrue("fast path should fire for offer-free big-source nodes, got "
                            + compactor.retainedOnlyNodes.get(),
                    compactor.retainedOnlyNodes.get() > 0);

            try (ReaderSupplier rs = ReaderSupplierFactory.open(outputPath)) {
                var merged = OnDiskGraphIndex.load(rs);
                assertEquals(smallVecs.size() + bigVecs.size(), merged.size(0));
                try (var mergedView = merged.getView(); var bigView = bigGraph.getView()) {
                    VectorFloat<?> tmp = vectorTypeSupport.createFloatVector(dim);
                    int offset = smallVecs.size();
                    int verifiedRetained = 0;
                    for (int n = 0; n < bigVecs.size(); n++) {
                        // Vector placement always holds.
                        mergedView.getVectorInto(offset + n, tmp, 0);
                        assertVecEquals(bigVecs.get(n), tmp, offset + n);
                        // Collect merged neighbors; for nodes whose merged edges are entirely
                        // big-source, they must equal the retained adjacency (fast path keeps
                        // order and membership).
                        List<Integer> mergedNbrs = new ArrayList<>();
                        var mit = mergedView.getNeighborsIterator(0, offset + n);
                        boolean anyCross = false;
                        while (mit.hasNext()) {
                            int nb = mit.nextInt();
                            if (nb < offset) anyCross = true;
                            mergedNbrs.add(nb);
                        }
                        if (anyCross) continue;
                        // Fast-path nodes keep source order; slow-path nodes whose offers all
                        // lost to diversity keep the same membership re-ordered by score — so
                        // membership equality is the invariant common to both.
                        List<Integer> retained = new ArrayList<>();
                        var bit = bigView.getNeighborsIterator(0, n);
                        while (bit.hasNext()) {
                            retained.add(offset + bit.nextInt());
                        }
                        assertEquals("all-retained node " + n + " must keep source adjacency membership",
                                new HashSet<>(retained), new HashSet<>(mergedNbrs));
                        verifiedRetained++;
                    }
                    assertTrue("expected some purely-retained records", verifiedRetained > 0);
                }

                // Search sanity on the merged graph.
                var allFast = new ArrayList<VectorFloat<?>>();
                allFast.addAll(smallVecs);
                allFast.addAll(bigVecs);
                var fastRavv = new ListRandomAccessVectorValues(allFast, dim);
                try (GraphSearcher searcher = new GraphSearcher(merged)) {
                    int found = 0;
                    for (int q = 0; q < 10; q++) {
                        VectorFloat<?> query = allFast.get(randomIntBetween(0, allFast.size() - 1));
                        SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(query, vsf, fastRavv);
                        SearchResult sr = searcher.search(ssp, 5, Bits.ALL);
                        if (sr.getNodes().length > 0) found++;
                    }
                    assertEquals(10, found);
                }
                merged.close();
            }
        }
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

        var compactor = new OnDiskGraphIndexCompactor(graphs, liveNodes, remappers, similarityFunction, null);
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

        var compactor = new OnDiskGraphIndexCompactor(graphs, liveNodes, remappers, similarityFunction, null);
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

        var compactor = new OnDiskGraphIndexCompactor(graphs, liveNodes, remappers, similarityFunction, null);
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
                vsf, null);

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
                    similarityFunction, null);
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
                    vsf, null);
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
                vsf, null);

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
                vsf, null);

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
    /** Builds a FusedPQ source graph from the given vectors (mirrors {@link #buildFusedPQ}). */
    private Path buildFusedSourceGraph(List<VectorFloat<?>> vecs, String name) throws IOException {
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vecs, dimension);
        ProductQuantization pq = ProductQuantization.compute(ravv, 8, 256, true, UNWEIGHTED, simdExecutor, parallelExecutor);
        PQVectors pqv = (PQVectors) pq.encodeAll(ravv, simdExecutor);
        var bsp = BuildScoreProvider.pqBuildScoreProvider(similarityFunction, pqv);
        var builder = new GraphIndexBuilder(bsp, dimension, 16, 100, 1.2f, 1.2f, false, true, simdExecutor, parallelExecutor);
        var graph = builder.getGraph();
        var outputPath = testDirectory.resolve(name);
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
        return outputPath;
    }

    /**
     * Cluster certification needs consecutive similarity-ordered queries to be near-twins; the
     * random-vector fixtures never certify, leaving the certificate paths untested. This merge
     * of jittered copies of a small vector pool certifies heavily, and validates recall of the
     * certified output against brute-force ground truth. When run with
     * {@code -Djvector.compaction.intervalCertify=true}, additionally asserts the RAM-interval
     * certificate path engaged.
     */
    @Test
    public void testClusterCertificationWithNearDuplicates() throws Exception {
        int poolSize = 64;
        int perSource = 320;
        int nSrc = 3;
        List<VectorFloat<?>> pool = createRandomVectors(poolSize, dimension);
        var rnd = new java.util.Random(12345);

        List<VectorFloat<?>> all = new ArrayList<>();
        List<OnDiskGraphIndex> graphs = new ArrayList<>();
        List<ReaderSupplier> rss = new ArrayList<>();
        List<FixedBitSet> live = new ArrayList<>();
        List<OrdinalMapper> remappers = new ArrayList<>();
        int base = 0;
        for (int sIdx = 0; sIdx < nSrc; sIdx++) {
            List<VectorFloat<?>> vecs = new ArrayList<>(perSource);
            for (int i = 0; i < perSource; i++) {
                VectorFloat<?> b = pool.get(i % poolSize);
                VectorFloat<?> v = vectorTypeSupport.createFloatVector(dimension);
                for (int d = 0; d < dimension; d++) {
                    v.set(d, b.get(d) + (float) (rnd.nextGaussian() * 1e-3));
                }
                vecs.add(v);
            }
            Path path = buildFusedSourceGraph(vecs, "certify_src_" + sIdx);
            rss.add(ReaderSupplierFactory.open(path));
            graphs.add(OnDiskGraphIndex.load(rss.get(sIdx)));
            var lv = new FixedBitSet(perSource);
            lv.set(0, perSource);
            live.add(lv);
            remappers.add(new OrdinalMapper.OffsetMapper(base, perSource));
            base += perSource;
            all.addAll(vecs);
        }

        var compactor = new OnDiskGraphIndexCompactor(graphs, live, remappers, similarityFunction, null);
        compactor.setSimilarityOrdinals(true);
        var outputPath = testDirectory.resolve("certify_compacted");
        compactor.compact(outputPath);

        assertTrue("near-duplicate merge should certify members, got "
                        + compactor.clusterCertified.get(),
                compactor.clusterCertified.get() > 0);
        assertTrue("anchor-relative certificates should engage on near-duplicate data, got "
                        + compactor.anchorRelCertified.get(),
                compactor.anchorRelCertified.get() > 0);

        // recall of the certified output vs brute force over the union
        int total = nSrc * perSource;
        var effective = compactor.effectiveRemappers();
        int[] newToDataset = new int[total];
        for (int sIdx = 0; sIdx < nSrc; sIdx++) {
            for (int i = 0; i < perSource; i++) {
                int newOrd = effective.get(sIdx).oldToNew(i);
                newToDataset[newOrd] = sIdx * perSource + i;
            }
        }
        int topK = 10;
        List<VectorFloat<?>> queries = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            queries.add(all.get(rnd.nextInt(all.size())));
        }
        List<List<Integer>> gt = new ArrayList<>();
        for (var q : queries) {
            List<Integer> idx = new ArrayList<>();
            for (int i = 0; i < total; i++) idx.add(i);
            idx.sort((a, b) -> Float.compare(
                    similarityFunction.compare(q, all.get(b)),
                    similarityFunction.compare(q, all.get(a))));
            gt.add(new ArrayList<>(idx.subList(0, topK)));
        }
        try (ReaderSupplier rs = ReaderSupplierFactory.open(outputPath)) {
            var compactGraph = OnDiskGraphIndex.load(rs);
            List<VectorFloat<?>> reordered = new ArrayList<>(total);
            for (int n = 0; n < total; n++) {
                reordered.add(all.get(newToDataset[n]));
            }
            var reorderedRavv = new ListRandomAccessVectorValues(reordered, dimension);
            try (GraphSearcher searcher = new GraphSearcher(compactGraph)) {
                int hits = 0;
                for (int qi = 0; qi < queries.size(); qi++) {
                    SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(queries.get(qi), similarityFunction, reorderedRavv);
                    SearchResult sr = searcher.search(ssp, topK, Bits.ALL);
                    for (var ns : sr.getNodes()) {
                        if (gt.get(qi).contains(newToDataset[ns.node])) hits++;
                    }
                }
                double recall = (double) hits / (queries.size() * topK);
                System.out.printf("Certification-merge recall: %.4f (certified %d, anchor-relative %d)%n",
                        recall, compactor.clusterCertified.get(), compactor.anchorRelCertified.get());
                // near-duplicate pools make GT ties common; the bar is deliberately moderate
                assertTrue("certified-merge recall should be >= 0.5, got " + recall, recall >= 0.5);
            }
        }
        for (var r : rss) r.close();
    }

    /** Inline-vectors-only source graph with the same build parameters as
     *  {@link #buildFusedSourceGraph}, so sidecar-vs-fused comparisons share source quality. */
    private Path buildPlainSourceGraph(List<VectorFloat<?>> vecs, VectorSimilarityFunction vsf, String name) throws IOException {
        var ravv = new ListRandomAccessVectorValues(vecs, dimension);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, vsf);
        var builder = new GraphIndexBuilder(bsp, dimension, 16, 100, 1.2f, 1.2f, false, true, simdExecutor, parallelExecutor);
        var graph = builder.build(ravv);
        Path path = testDirectory.resolve(name);
        var writerBuilder = new OnDiskGraphIndexWriter.Builder(graph, path);
        writerBuilder.with(new InlineVectors(dimension));
        try (var writer = writerBuilder.build()) {
            var writeSuppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
            writeSuppliers.put(FeatureId.INLINE_VECTORS, ordinal -> new InlineVectors.State(ravv.getVector(ordinal)));
            writer.write(writeSuppliers);
        }
        return path;
    }

    /**
     * Sidecar-parity counterpart of {@link #testClusterCertificationWithNearDuplicates}: sources
     * are plain graphs with separate PQVectors sidecars (the SAI layout). With parity, similarity
     * ordinals derive from the retrained sidecar PQ, the strategy pre-encode cache backs
     * approximate traversal scoring, and cluster certification must engage just as in fused mode.
     */
    @Test
    public void testSidecarParityCertificationWithNearDuplicates() throws Exception {
        // COSINE: the LUT declines unsupported metrics, so this exercises the exact-scoring
        // fallback with the full parity stack (ordinals, cache, cluster search).
        sidecarParityCertification(VectorSimilarityFunction.COSINE);
    }

    @Test
    public void testSidecarParityCertificationLutScoring() throws Exception {
        // EUCLIDEAN: traversal scores through the cache-LUT (center-adjusted PQ), the path
        // production DOT/EUCLIDEAN merges take.
        sidecarParityCertification(VectorSimilarityFunction.EUCLIDEAN);
    }

    private void sidecarParityCertification(VectorSimilarityFunction vsf) throws Exception {
        int poolSize = 64;
        int perSource = 320;
        int nSrc = 3;
        List<VectorFloat<?>> pool = createRandomVectors(poolSize, dimension);
        var rnd = new java.util.Random(54321);

        List<VectorFloat<?>> all = new ArrayList<>();
        List<OnDiskGraphIndex> graphs = new ArrayList<>();
        List<ReaderSupplier> rss = new ArrayList<>();
        List<io.github.jbellis.jvector.quantization.CompressedVectors> compressed = new ArrayList<>();
        List<FixedBitSet> live = new ArrayList<>();
        List<OrdinalMapper> remappers = new ArrayList<>();
        int base = 0;
        for (int sIdx = 0; sIdx < nSrc; sIdx++) {
            List<VectorFloat<?>> vecs = new ArrayList<>(perSource);
            for (int i = 0; i < perSource; i++) {
                VectorFloat<?> b = pool.get(i % poolSize);
                VectorFloat<?> v = vectorTypeSupport.createFloatVector(dimension);
                for (int d = 0; d < dimension; d++) {
                    v.set(d, b.get(d) + (float) (rnd.nextGaussian() * 1e-3));
                }
                vecs.add(v);
            }
            Path path = buildPlainSourceGraph(vecs, vsf, "sc_certify_src_" + sIdx);
            rss.add(ReaderSupplierFactory.open(path));
            graphs.add(OnDiskGraphIndex.load(rss.get(sIdx)));
            var ravv = new ListRandomAccessVectorValues(vecs, dimension);
            ProductQuantization pq = ProductQuantization.compute(ravv, dimension / 2, 256, true, UNWEIGHTED, simdExecutor, parallelExecutor);
            compressed.add(pq.encodeAll(ravv, simdExecutor));
            var lv = new FixedBitSet(perSource);
            lv.set(0, perSource);
            live.add(lv);
            remappers.add(new OrdinalMapper.OffsetMapper(base, perSource));
            base += perSource;
            all.addAll(vecs);
        }

        var compactor = new OnDiskGraphIndexCompactor(graphs, compressed, live, remappers, vsf, null);
        compactor.setSimilarityOrdinals(true);
        var outputPath = testDirectory.resolve("sc_certify_compacted_" + vsf);
        var pqOutPath = testDirectory.resolve("sc_certify_pq_" + vsf);
        compactor.compact(outputPath, pqOutPath);

        assertTrue("sidecar-parity near-duplicate merge should certify members, got "
                        + compactor.clusterCertified.get(),
                compactor.clusterCertified.get() > 0);

        // recall of the merged output vs brute force over the union
        int total = nSrc * perSource;
        var effective = compactor.effectiveRemappers();
        int[] newToDataset = new int[total];
        for (int sIdx = 0; sIdx < nSrc; sIdx++) {
            for (int i = 0; i < perSource; i++) {
                int newOrd = effective.get(sIdx).oldToNew(i);
                newToDataset[newOrd] = sIdx * perSource + i;
            }
        }
        int topK = 10;
        List<VectorFloat<?>> queries = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            queries.add(all.get(rnd.nextInt(all.size())));
        }
        // Tie-aware ground truth: near-duplicate pools make top-K membership arbitrary among
        // equal-distance twins, so a result counts if its exact similarity clears the query's
        // kth-best similarity (minus float slack) rather than matching set identity.
        float[] kthSim = new float[queries.size()];
        for (int qi = 0; qi < queries.size(); qi++) {
            var q = queries.get(qi);
            float[] sims = new float[total];
            for (int i = 0; i < total; i++) {
                sims[i] = vsf.compare(q, all.get(i));
            }
            java.util.Arrays.sort(sims);
            kthSim[qi] = sims[total - topK];
        }
        try (ReaderSupplier rs = ReaderSupplierFactory.open(outputPath)) {
            var compactGraph = OnDiskGraphIndex.load(rs);
            List<VectorFloat<?>> reordered = new ArrayList<>(total);
            for (int n = 0; n < total; n++) {
                reordered.add(all.get(newToDataset[n]));
            }
            var reorderedRavv = new ListRandomAccessVectorValues(reordered, dimension);
            try (GraphSearcher searcher = new GraphSearcher(compactGraph)) {
                int hits = 0;
                for (int qi = 0; qi < queries.size(); qi++) {
                    SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(queries.get(qi), vsf, reorderedRavv);
                    SearchResult sr = searcher.search(ssp, topK, Bits.ALL);
                    for (var ns : sr.getNodes()) {
                        float sim = vsf.compare(queries.get(qi), all.get(newToDataset[ns.node]));
                        if (sim >= kthSim[qi] - 1e-6f) hits++;
                    }
                }
                double recall = (double) hits / (queries.size() * topK);
                System.out.printf("Sidecar-parity merge recall (tie-aware): %.4f (certified %d)%n",
                        recall, compactor.clusterCertified.get());
                assertTrue("sidecar-parity merge tie-aware recall should be >= 0.8, got " + recall, recall >= 0.8);
            }
        }
        // The merged sidecar must decode consistently under the retrained codebook.
        try (var rsCompressed = ReaderSupplierFactory.open(pqOutPath); var reader = rsCompressed.get()) {
            PQVectors mergedPqv = PQVectors.load(reader);
            assertEquals("merged PQVectors count", total, mergedPqv.count());
        }
        for (var r : rss) r.close();
    }
}
