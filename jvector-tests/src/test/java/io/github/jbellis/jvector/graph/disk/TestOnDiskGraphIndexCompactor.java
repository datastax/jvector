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
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import software.amazon.awssdk.services.s3.endpoints.internal.Value;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntFunction;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestOnDiskGraphIndexCompactor extends RandomizedTest {
    private List<TestUtil.RandomlyConnectedGraphIndex> randomlyConnectedGraphs;
    private List<TestVectorGraph.CircularFloatVectorValues> ravvs;
    private ImmutableGraphIndex golden;
    private Path testDirectory;
    List<VectorFloat<?>> allVecs = new ArrayList<>();
    List<OnDiskGraphIndex> graphs = new ArrayList<>();
    int dimension = 32;
    int numVectorsPerGraph = 1000;
    int numGraphs = 5;
    int numQueries = 10;
    VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.COSINE;
    RandomAccessVectorValues allravv;
    private final ForkJoinPool simdExecutor = ForkJoinPool.commonPool();
    private final ForkJoinPool parallelExecutor = ForkJoinPool.commonPool();

    @Before
    public void setup() throws IOException {
        //testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
        testDirectory = Files.createTempDirectory("jvector_test");
        // buildFP();
        buildFusedPQ();
        buildGoldenPQ();
    }

    void buildFP() throws IOException {
        for(int i = 0; i < numGraphs; ++i) {
            List<VectorFloat<?>> vecs = createRandomVectors(numVectorsPerGraph, 32);
            RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vecs, dimension);
            var builder = new GraphIndexBuilder(
                    BuildScoreProvider.randomAccessScoreProvider(ravv, similarityFunction),
                    dimension,
                    10,
                    100,
                    1.0f,
                    1.0f,
                    true,
                    true,
                    simdExecutor,
                    parallelExecutor);
            var graph = TestUtil.buildSequentially(builder, ravv);

            var outputPath = testDirectory.resolve("test_graph_" + i);
            TestUtil.writeGraph(graph, ravv, outputPath);
            allVecs.addAll(vecs);
        }
    }
    void buildFusedPQ() throws IOException {
        for(int i = 0; i < numGraphs; ++i) {
            List<VectorFloat<?>> vecs = createRandomVectors(numVectorsPerGraph, 32);
            RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vecs, dimension);
            ProductQuantization pq = ProductQuantization.compute(ravv, 8, 256, true, UNWEIGHTED, simdExecutor, parallelExecutor);
            PQVectors pqv = (PQVectors) pq.encodeAll(ravv, simdExecutor);
            var bsp = BuildScoreProvider.pqBuildScoreProvider(similarityFunction, pqv);
            var builder = new GraphIndexBuilder(bsp, dimension, 32, 1000, 1.0f, 1.0f, true, true, simdExecutor, parallelExecutor);
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
                writer.writeInline(node, stateMap);;
                builder.addGraphNode(node, ravv.getVector(i));
            }
            builder.cleanup();

            writeSuppliers.put(FeatureId.FUSED_PQ, ordinal -> new FusedPQ.State(graph.getView(), pqv, ordinal));
            writer.write(writeSuppliers);
            allVecs.addAll(vecs);
        }
    }
    void buildGoldenFP() throws IOException {
        // golden
        allravv = new ListRandomAccessVectorValues(allVecs, dimension);
        var builder = new GraphIndexBuilder(allravv, similarityFunction, 10, 100, 1.0f, 1.0f, true);
        golden = TestUtil.buildSequentially(builder, allravv);
    }
    void buildGoldenPQ() throws IOException {
        allravv = new ListRandomAccessVectorValues(allVecs, dimension);

        ProductQuantization pq = ProductQuantization.compute(allravv, 8, 256, true, UNWEIGHTED, simdExecutor, parallelExecutor);
        PQVectors pqv = (PQVectors) pq.encodeAll(allravv, simdExecutor);
        var bsp = BuildScoreProvider.pqBuildScoreProvider(similarityFunction, pqv);
        var builder = new GraphIndexBuilder(bsp, dimension, 10, 100, 1.0f, 1.0f, true, true, simdExecutor, parallelExecutor);
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

    @Test
    public void testCompact() throws Exception {
        List<ReaderSupplier> rss = new ArrayList<>();
        for(int i = 0; i < numGraphs; ++i) {
            var outputPath = testDirectory.resolve("test_graph_" + i);
            rss.add(ReaderSupplierFactory.open(outputPath.toAbsolutePath()));
            var onDiskGraph = OnDiskGraphIndex.load(rss.get(i));
            graphs.add(onDiskGraph);
        }

        var compactor = new OnDiskGraphIndexCompactor(graphs);
        int topK = 10;
        int globalOrdinal = 0;
        for(int n = 0; n < numGraphs; ++n) {
            Map<Integer, Integer> map = new HashMap<>();
            for(int i = 0; i < numVectorsPerGraph; ++i) {
                map.put(i, globalOrdinal++);
            }
            var remapper = new OrdinalMapper.MapMapper(map);
            compactor.setRemapper(graphs.get(n), remapper);
        }
        var outputPath = testDirectory.resolve("test_compact_graph_");
        List<VectorFloat<?>> queries = new ArrayList<>();
        for(int i = 0; i < numQueries; ++i) {
            queries.add(allVecs.get(randomIntBetween(0, allVecs.size() - 1)));
        }
        List<SearchResult> srs = searchFromAll(queries, topK);
        List<List<Integer>> groundTruth = buildGT(queries, topK);

        List<SearchResult> csrs = new ArrayList<>();
        System.out.printf("start compaction%n");
        compactor.compact(outputPath, similarityFunction);
        System.out.printf("done%n");

        ReaderSupplier rs;
        try {
            rs = ReaderSupplierFactory.open(outputPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        var compactGraph = OnDiskGraphIndex.load(rs);
        GraphSearcher searcher = new GraphSearcher(compactGraph);

        for(VectorFloat<?> q: queries) {
            SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(q, similarityFunction, allravv);
            csrs.add(searcher.search(ssp, topK, Bits.ALL));
        }
        var recall = AccuracyMetrics.recallFromSearchResults(groundTruth, srs, topK, topK);
        var crecall = AccuracyMetrics.recallFromSearchResults(groundTruth, csrs, topK, topK);
        System.out.printf("Recall: %.4f%n", recall);
        System.out.printf("Compacted Recall: %.4f%n", crecall);
    }
}
