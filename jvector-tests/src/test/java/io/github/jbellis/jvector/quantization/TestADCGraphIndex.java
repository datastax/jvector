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
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static io.github.jbellis.jvector.graph.TestVectorGraph.createRandomFloatVectors;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestADCGraphIndex extends RandomizedTest {

    private Path testDirectory;

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
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

        TestUtil.writeFusedGraph(graph, ravv, pqv, outputPath);

        try (var readerSupplier = new SimpleMappedReader.Supplier(outputPath);
             var onDiskGraph = OnDiskGraphIndex.load(readerSupplier, 0))
        {
            TestUtil.assertGraphEquals(graph, onDiskGraph);
            try (var cachedOnDiskView = onDiskGraph.getView())
            {
                for (var similarityFunction : VectorSimilarityFunction.values()) {
                    var queryVector = TestUtil.randomVector(getRandom(), 512);
                    var pqScoreFunction = pqv.precomputedScoreFunctionFor(queryVector, similarityFunction);
                    var reranker = cachedOnDiskView.rerankerFor(queryVector, similarityFunction);
                    for (int i = 0; i < 50; i++) {
                        var fusedScoreFunction = cachedOnDiskView.approximateScoreFunctionFor(queryVector, similarityFunction);
                        var ordinal = getRandom().nextInt(graph.size());
                        // first pass compares fused ADC's direct similarity to reranker's similarity, used for comparisons to a specific node
                        var neighbors = cachedOnDiskView.getNeighborsIterator(0, ordinal);
                        for (; neighbors.hasNext(); ) {
                            var neighbor = neighbors.next();
                            var similarity = fusedScoreFunction.similarityTo(neighbor);
                            assertEquals(reranker.similarityTo(neighbor), similarity, 0.01);
                        }
                        // second pass compares fused ADC's edge similarity prior to having enough information for quantization to PQ
                        neighbors = cachedOnDiskView.getNeighborsIterator(0, ordinal);
                        var edgeSimilarities = fusedScoreFunction.edgeLoadingSimilarityTo(ordinal);
                        for (int j = 0; neighbors.hasNext(); j++) {
                            var neighbor = neighbors.next();
                            assertEquals(pqScoreFunction.similarityTo(neighbor), edgeSimilarities.getScore(j), 0.01);
                        }
                        // third pass compares fused ADC's edge similarity after quantization to edge similarity before quantization
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
    public void testRandom() throws IOException {
        testRandom(false);
        testRandom(true);
    }

    // build a random graph, then check that it has at least 90% recall
    public void testRandom(boolean addHierarchy) throws IOException {
        var outputPath = testDirectory.resolve("random_fused_graph");

        int size = 10_000;
        int dim = 32;
        MockVectorValues vectors = vectorValues(size, dim);

        int topK = 5;
        int efSearch = 50;

        var similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
        GraphIndexBuilder builder = new GraphIndexBuilder(vectors, similarityFunction, 32, 30, 1.0f, 1.4f, addHierarchy);
        var tempGraph = builder.build(vectors);

        var pq = ProductQuantization.compute(vectors, 8, 256, false);
        var pqv = (PQVectors) pq.encodeAll(vectors);

        TestUtil.writeFusedGraph(tempGraph, vectors, pqv, outputPath);

        try (var readerSupplier = new SimpleMappedReader.Supplier(outputPath);
            var graph = OnDiskGraphIndex.load(readerSupplier, 0)) {
            var searcher = new GraphSearcher(graph);

            for (var fused : List.of(true, false)) {
                int totalMatches = 0;
                for (int i = 0; i < 100; i++) {
                    SearchResult.NodeScore[] actual;
                    VectorFloat<?> query = randomVector(dim);

                    SearchScoreProvider ssp = scoreProviderFor(fused, query, similarityFunction, graph.getView(), pqv);
                    actual = searcher.search(ssp, topK, efSearch, 0.0f, 0.0f, Bits.ALL).getNodes();

                    NodeQueue expected = new NodeQueue(new BoundedLongHeap(topK), NodeQueue.Order.MIN_HEAP);
                    for (int j = 0; j < size; j++) {
                        expected.push(j, similarityFunction.compare(query, vectors.getVector(j)));
                    }
                    var actualNodeIds = Arrays.stream(actual, 0, topK).mapToInt(nodeScore -> nodeScore.node).toArray();

                    assertEquals(topK, actualNodeIds.length);
                    totalMatches += computeOverlap(actualNodeIds, expected.nodesCopy());
                }
                double overlap = totalMatches / (double) (100 * topK);
                assertTrue("overlap=" + overlap, overlap > 0.9);
            }
        }
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
        return MockVectorValues.fromValues(createRandomFloatVectors(size, dimension, getRandom()));
//        assertEquals(size, 900);
//        assertEquals(dimension, 2);
//
//        var vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
//
//        VectorFloat<?>[] vectors = new VectorFloat<?>[size];
//        for (int i = 0; i < 30; i++) {
//            for (int j = 0; j < 30; j++) {
//                var vec = vectorTypeSupport.createFloatVector(dimension);
//                vec.set(0, i);
//                vec.set(1, j);
//                vectors[i * 30 + j] = vec;
//            }
//        }
//        return MockVectorValues.fromValues(vectors);
    }

    VectorFloat<?> randomVector(int dim) {
        return TestUtil.randomVector(getRandom(), dim);
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
