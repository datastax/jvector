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

package io.github.jbellis.jvector.quantization;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.MockVectorValues;
import io.github.jbellis.jvector.graph.NodeQueue;
import io.github.jbellis.jvector.graph.NodeScoreArray;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.BoundedLongHeap;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static io.github.jbellis.jvector.graph.TestVectorGraph.createRandomFloatVectors;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestADCGraphIndex extends RandomizedTest {

    private Path testDirectory;
    private Random random;

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
        random = getRandom();
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    @Test
    public void testFusedGraph() throws Exception {
        // generate random graph, M=32, 256-dimension vectors
        var graph = new TestUtil.RandomlyConnectedGraphIndex(1000, 32, getRandom());
        var outputPath = testDirectory.resolve("large_graph");
        var vectors = createRandomVectors(1000,  512);
        var ravv = new ListRandomAccessVectorValues(vectors, 512);
        var pq = ProductQuantization.compute(ravv, 8, 256, false);
        var pqv = (PQVectors) pq.encodeAll(ravv);

        TestUtil.writeFusedGraph(graph, ravv, pqv, FeatureId.INLINE_VECTORS, outputPath);

        try (var readerSupplier = new SimpleMappedReader.Supplier(outputPath);
             var onDiskGraph = OnDiskGraphIndex.load(readerSupplier, 0))
        {
            TestUtil.assertGraphEquals(graph, onDiskGraph);
            try (var cachedOnDiskView = onDiskGraph.getView())
            {
                for (var similarityFunction : VectorSimilarityFunction.values()) {
                    var queryVector = TestUtil.randomVector(getRandom(), 512);
                    var pqScoreFunction = pqv.precomputedScoreFunctionFor(queryVector, similarityFunction);
                    for (int i = 0; i < 50; i++) {
                        var fusedScoreFunction = cachedOnDiskView.approximateScoreFunctionFor(queryVector, similarityFunction);
                        var ordinal = getRandom().nextInt(graph.size());

                        // first pass compares fused ADC's edge similarity prior to having enough information for quantization to PQ
                        var neighbors = cachedOnDiskView.getNeighborsIterator(0, ordinal);
                        var edgeSimilarities = fusedScoreFunction.edgeLoadingSimilarityTo(ordinal);
                        for (int j = 0; neighbors.hasNext(); j++) {
                            var neighbor = neighbors.next();
                            assertEquals(pqScoreFunction.similarityTo(neighbor), edgeSimilarities.getScore(j), 0.01);
                        }
                        // second pass compares fused ADC's edge similarity after quantization to edge similarity before quantization
                        var edgeSimilaritiesCopy = copy(edgeSimilarities); // results of second pass
                        var fusedEdgeSimilarities = fusedScoreFunction.edgeLoadingSimilarityTo(ordinal); // results of third pass
                        for (int j = 0; j < fusedEdgeSimilarities.size(); j++) {
                            assertEquals(fusedEdgeSimilarities.getScore(j), edgeSimilaritiesCopy.getScore(j), 0.01);
                        }
                    }
                }
            }
        }
    }

    NodeScoreArray copy(NodeScoreArray nodeScoreArray) {
        var copy = new NodeScoreArray(nodeScoreArray.size());
        for (int i = 0; i < nodeScoreArray.size(); i++) {
            copy.setNode(i, nodeScoreArray.getNode(i));
            copy.setScore(i, nodeScoreArray.getScore(i));
        }
        return copy;
    }

    @Test
    // build a random graph, then check that it has at least 90% recall
    public void testRecallOnGraphWithRandomVectors() throws IOException {
        for (var similarityFunction : List.of(VectorSimilarityFunction.COSINE, VectorSimilarityFunction.DOT_PRODUCT, VectorSimilarityFunction.EUCLIDEAN)) {
            for (var addHierarchy : List.of(false, true)) {
                for (var featureId: List.of(FeatureId.INLINE_VECTORS, FeatureId.NVQ_VECTORS)) {
                    testRecallOnGraphWithRandomVectors(addHierarchy, similarityFunction, featureId);
                }
            }
        }
    }

    // build a random graph, then check that it has at least 90% recall
    public void testRecallOnGraphWithRandomVectors(boolean addHierarchy, VectorSimilarityFunction similarityFunction, FeatureId featureId) throws IOException {
        var outputPath = testDirectory.resolve("random_fused_graph");

        int size = 1_000;
        int dim = 32;
        MockVectorValues vectors = vectorValues(size, dim);

        int topK = 5;
        int efSearch = 20;

        GraphIndexBuilder builder = new GraphIndexBuilder(vectors, similarityFunction, 32, 32, 1.2f, 1.2f, addHierarchy);
        var tempGraph = builder.build(vectors);

        var pq = ProductQuantization.compute(vectors, 8, 256, false);
        var pqv = (PQVectors) pq.encodeAll(vectors);

        TestUtil.writeFusedGraph(tempGraph, vectors, pqv, featureId, outputPath);

        try (var readerSupplier = new SimpleMappedReader.Supplier(outputPath);
            var graph = OnDiskGraphIndex.load(readerSupplier, 0)) {
            var searcher = new GraphSearcher(graph);

            Map<Boolean, Integer> totalMatches = new HashMap<>();
            totalMatches.put(true, 0); // true will be used for fused computations
            totalMatches.put(false, 0); // false will be used for unfused computations

            for (int i = 0; i < 100; i++) {
                SearchResult.NodeScore[] actual;
                VectorFloat<?> query = randomVector(dim);

                for (var fused : List.of(true, false)) {
                    SearchScoreProvider ssp = scoreProviderFor(fused, query, similarityFunction, graph.getView(), pqv);
                    actual = searcher.search(ssp, topK, efSearch, 0.0f, 0.0f, Bits.ALL).getNodes();

                    NodeQueue expected = new NodeQueue(new BoundedLongHeap(topK), NodeQueue.Order.MIN_HEAP);
                    for (int j = 0; j < size; j++) {
                        expected.push(j, similarityFunction.compare(query, vectors.getVector(j)));
                    }
                    var actualNodeIds = Arrays.stream(actual, 0, topK).mapToInt(nodeScore -> nodeScore.node).toArray();

                    assertEquals(topK, actualNodeIds.length);
                    totalMatches.put(fused, totalMatches.get(fused) + computeOverlap(actualNodeIds, expected.nodesCopy()));
                }
            }
            assertTrue(Math.abs(totalMatches.get(true) - totalMatches.get(false)) < 2);
            for (var fused : List.of(true, false)) {
                double overlap = totalMatches.get(fused) / (double) (100 * topK);
                assertTrue("overlap=" + overlap, overlap > 0.90);
            }
        }
        Files.deleteIfExists(outputPath);
    }

    @Test
    // build a random graph, then check that it has at least 90% recall
    public void testScoresWithRandomVectors() throws IOException {
        for (var similarityFunction : List.of(VectorSimilarityFunction.COSINE, VectorSimilarityFunction.DOT_PRODUCT, VectorSimilarityFunction.EUCLIDEAN)) {
            for (var addHierarchy : List.of(false, true)) {
                for (var featureId: List.of(FeatureId.INLINE_VECTORS, FeatureId.NVQ_VECTORS)) {
                    testScoresWithRandomVectors(addHierarchy, similarityFunction, featureId);
                }
            }
        }
    }

    public void testScoresWithRandomVectors(boolean addHierarchy, VectorSimilarityFunction similarityFunction, FeatureId featureId) throws IOException {
        var outputPath = testDirectory.resolve("random_fused_graph");

        int size = 1_000;
        int dim = 32;
        MockVectorValues vectors = vectorValues(size, dim);

        GraphIndexBuilder builder = new GraphIndexBuilder(vectors, similarityFunction, 32, 32, 1.2f, 1.2f, addHierarchy);
        var tempGraph = builder.build(vectors);

        var pq = ProductQuantization.compute(vectors, 8, 256, false);
        var pqv = (PQVectors) pq.encodeAll(vectors);

        TestUtil.writeFusedGraph(tempGraph, vectors, pqv, featureId, outputPath);

        try (var readerSupplier = new SimpleMappedReader.Supplier(outputPath);
            var graph = OnDiskGraphIndex.load(readerSupplier, 0)) {

            for (int iQuery = 0; iQuery < 10; iQuery++) {
                VectorFloat<?> query = randomVector(dim);

                for (int node = 0; node < size; node++) {
                    // Fused computations
                    SearchScoreProvider sspFused = scoreProviderFor(true, query, similarityFunction, graph.getView(), pqv);
                    var similarities = sspFused.scoreFunction().edgeLoadingSimilarityTo(node);

                    // Regular (un-fused) computations
                    SearchScoreProvider ssp = scoreProviderFor(false, query, similarityFunction, graph.getView(), pqv);
                    var it = graph.getView().getNeighborsIterator(0, node);
                    int position = 0;
                    assertEquals(similarities.size(), it.size());
                    while (it.hasNext()) {
                        int neighbor = it.next();
                        float score = ssp.scoreFunction().similarityTo(neighbor);
                        assertEquals(similarities.getNode(position), neighbor);
                        assertEquals(similarities.getScore(position), score, 1e-6);
                        position++;
                    }
                }
            }
        }
        Files.deleteIfExists(outputPath);
    }

    public SearchScoreProvider scoreProviderFor(boolean fused, VectorFloat<?> queryVector, VectorSimilarityFunction similarityFunction, GraphIndex.View view, CompressedVectors cv) {
        var scoringView = (GraphIndex.ScoringView) view;
        ScoreFunction.ApproximateScoreFunction asf;
        if (fused) {
            asf = scoringView.approximateScoreFunctionFor(queryVector, similarityFunction);
        } else {
            asf = cv.precomputedScoreFunctionFor(queryVector, similarityFunction);
        }
        var rr = scoringView.rerankerFor(queryVector, similarityFunction);
        return new DefaultSearchScoreProvider(asf, rr);
    }

    MockVectorValues vectorValues(int size, int dimension) {
        return MockVectorValues.fromValues(createRandomFloatVectors(size, dimension, random));
    }

    VectorFloat<?> randomVector(int dim) {
        return TestUtil.randomVector(random, dim);
    }

    private int computeOverlap(int[] a, int[] b) {
        Arrays.sort(a);
        Arrays.sort(b);
        int overlap = 0;
        for (int i = 0, j = 0; i < a.length && j < b.length; ) {
            if (a[i] == b[j]) {
                ++overlap;
                ++i;
                ++j;
            } else if (a[i] > b[j]) {
                ++j;
            } else {
                ++i;
            }
        }
        return overlap;
    }
}
