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
}
