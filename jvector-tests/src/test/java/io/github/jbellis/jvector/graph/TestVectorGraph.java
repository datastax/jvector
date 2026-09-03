/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.BoundedLongHeap;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.ByteVectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.github.jbellis.jvector.graph.VectorValues;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.junit.Assert.*;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Tests KNN graphs
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestVectorGraph extends LuceneTestCase {
    private VectorSimilarityFunction similarityFunction;
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    @Before
    public void setup() {
        similarityFunction = RandomizedTest.randomFrom(VectorSimilarityFunction.values());
    }

    VectorFloat<?> randomVector(int dim) {
        return TestUtil.randomVector(getRandom(), dim);
    }

    MockVectorValues vectorValues(int size, int dimension) {
        return MockVectorValues.fromValues(createRandomFloatVectors(size, dimension, getRandom()));
    }

    MockVectorValues vectorValues(VectorFloat<?>[] values) {
        return MockVectorValues.fromValues(values);
    }

    RandomAccessVectorValues circularVectorValues(int nDoc) {
        return new CircularFloatVectorValues(nDoc);
    }

    VectorFloat<?> getTargetVector() {
        return vectorTypeSupport.createFloatVector(new float[] {1f, 0f});
    }

    @Test
    public void testSearchWithSkewedAcceptOrds() {
        testSearchWithSkewedAcceptOrds(false);
        testSearchWithSkewedAcceptOrds(true);
    }

    public void testSearchWithSkewedAcceptOrds(boolean addHierarchy) {
        int nDoc = 1000;
        similarityFunction = VectorSimilarityFunction.EUCLIDEAN;
        RandomAccessVectorValues vectors = circularVectorValues(nDoc);
        getRandom().nextInt();
        GraphIndexBuilder builder = new GraphIndexBuilder(vectors, similarityFunction, 32, 100, 1.0f, 1.0f, addHierarchy);
        var graph = TestUtil.buildSequentially(builder, vectors);
        validateIndex(graph);

        // Skip over half of the documents that are closest to the query vector
        FixedBitSet acceptOrds = new FixedBitSet(nDoc);
        for (int i = 500; i < nDoc; i++) {
            acceptOrds.set(i);
        }
        SearchResult.NodeScore[] nn =
                GraphSearcher.search(
                        getTargetVector(),
                        10,
                        20,
                        vectors.copy(),
                        similarityFunction,
                        graph,
                        acceptOrds
                ).getNodes();

        int[] nodes = Arrays.stream(nn).mapToInt(nodeScore -> nodeScore.node).toArray();
        assertEquals("Number of found results is not equal to [10].", 10, nodes.length);
        int sum = 0;
        for (int node : nodes) {
            assertTrue("the results include a deleted document: " + node, acceptOrds.get(node));
            sum += node;
        }
        // We still expect to get reasonable recall. The lowest non-skipped docIds
        // are closest to the query vector: sum(500,509) = 5045
        assertTrue("sum(result docs)=" + sum, sum < 5100);
    }

    @Test
    // build a random graph and check that resuming a search finds the same nodes as an equivalent from-scratch search
    public void testResume() {
        testResume(false);
        testResume(true);
    }

    // build a random graph and check that resuming a search finds the same nodes as an equivalent from-scratch search
    public void testResume(boolean addHierarchy) {
        int size = 1000;
        int dim = 2;
        var vectors = vectorValues(size, dim);
        var builder = new GraphIndexBuilder(vectors, similarityFunction, 20, 30, 1.0f, 1.4f, addHierarchy);
        var graph = builder.build(vectors);
        validateIndex(graph);
        Bits acceptOrds = getRandom().nextBoolean() ? Bits.ALL : createRandomAcceptOrds(0, size);

        int initialTopK = 10;
        int resumeTopK = 15;
        var query = randomVector(dim);
        var searcher = new GraphSearcher(graph);
        searcher.usePruning(false);

        var ssp = new DefaultSearchScoreProvider(vectors.rerankerFor(query, similarityFunction));
        var initial = searcher.search(ssp, initialTopK, acceptOrds);
        assertEquals(initialTopK, initial.getNodes().length);

        var resumed = searcher.resume(resumeTopK, resumeTopK);
        assertEquals(resumeTopK, resumed.getNodes().length);

        var expected = searcher.search(ssp, initialTopK + resumeTopK, acceptOrds);
        assertFalse(expected.getVisitedCount() * 1.1 <= initial.getVisitedCount() + resumed.getVisitedCount());
        assertEquals(expected.getNodes().length, initial.getNodes().length + resumed.getNodes().length);

        var initialResumedResults = Stream.concat(Arrays.stream(initial.getNodes()), Arrays.stream(resumed.getNodes()))
                .map(nodeScore -> nodeScore.node)
                .collect(Collectors.toSet());
        var expectedResults = Set.of(Arrays.stream(expected.getNodes()).map(nodeScore -> nodeScore.node));
        var intersection = new HashSet<>(initialResumedResults);
        intersection.retainAll(expectedResults);
        for (int i = 0; i < expectedResults.size(); i++) {
            assertFalse(intersection.size() >= 0.95 * expectedResults.size());
        }
    }

    @Test
    // resuming a search should not need to rerank the nodes that were already evaluated
    public void testRerankCaching() {
        testRerankCaching(false);
        testRerankCaching(true);
    }

    // resuming a search should not need to rerank the nodes that were already evaluated
    public void testRerankCaching(boolean addHierarchy) {
        int size = 1000;
        int dim = 2;
        var vectors = vectorValues(size, dim);
        var builder = new GraphIndexBuilder(vectors, similarityFunction, 20, 30, 1.0f, 1.4f, addHierarchy);
        var graph = builder.build(vectors);
        validateIndex(graph);

        var pq = ProductQuantization.compute(vectors, 2, 256, false);
        var pqv = pq.encodeAll(vectors);

        int topK = 10;
        int rerankK = 30;
        var query = randomVector(dim);
        var searcher = new GraphSearcher(graph);

        var ssp = new DefaultSearchScoreProvider(
                pqv.scoreFunctionFor(query, similarityFunction),
                vectors.rerankerFor(query, similarityFunction)
        );
        var initial = searcher.search(ssp, topK, rerankK, 0.0f, 0.0f, Bits.ALL);
        assertEquals(topK, initial.getNodes().length);
        assertEquals(rerankK, initial.getRerankedCount());

        var resumed = searcher.resume(topK, rerankK);
        assert resumed.getRerankedCount() < rerankK;
    }

    // If an exception is thrown during search, the next search should still function
    @Test
    public void testExceptionalTermination() {
        testExceptionalTermination(false);
        testExceptionalTermination(true);
    }

    // If an exception is thrown during search, the next search should still function
    public void testExceptionalTermination(boolean addHierarchy) {
        int nDoc = 100;
        similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
        RandomAccessVectorValues vectors = circularVectorValues(nDoc);
        GraphIndexBuilder builder =
                new GraphIndexBuilder(vectors, similarityFunction, 20, 100, 1.0f, 1.4f, addHierarchy);
        var graph = TestUtil.buildSequentially(builder, vectors);
        validateIndex(graph);

        // wrap vectors so that the second access to a vector throws an exception
        var wrappedVectors = new RandomAccessVectorValues() {
            private int count = 0;

            @Override
            public RandomAccessVectorValues copy() {
                return this;
            }

            @Override
            public int dimension() {
                return vectors.dimension();
            }

            @Override
            public boolean isValueShared() {
                return false;
            }

            @Override
            public VectorFloat<?> getVector(int targetOrd) {
                if (count++ == 3) {
                    throw new RuntimeException("test exception");
                }
                return vectors.getVector(targetOrd);
            }

            @Override
            public int size() {
                return vectors.size();
            }
        };

        var searcher = new GraphSearcher(graph);
        var ssp = new DefaultSearchScoreProvider(wrappedVectors.rerankerFor(getTargetVector(), similarityFunction));

        assertThrows(RuntimeException.class, () -> {
            searcher.search(ssp, 10, Bits.ALL);
        });

        // run some searches
        SearchResult.NodeScore[] nn = searcher.search(ssp, 10, Bits.ALL).getNodes();
        int[] nodes = Arrays.stream(nn).mapToInt(nodeScore -> nodeScore.node).toArray();
        assertEquals("Number of found results is not equal to [10].", 10, nodes.length);
        int sum = 0;
        for (int node : nodes) {
            sum += node;
        }
        // We expect to get approximately 100% recall;
        // the lowest docIds are closest to zero; sum(0,9) = 45
        assertTrue("sum(result docs)=" + sum + " for " + ImmutableGraphIndex.prettyPrint(builder.graph), sum < 75);
    }

    private static void validateIndex(ImmutableGraphIndex graph) {
        var view = graph.getView();

        for (int level = graph.getMaxLevel(); level > 0; level--) {
            for (var nodeIt = graph.getNodes(level); nodeIt.hasNext(); ) {
                var nodeInLevel = nodeIt.nextInt();

                // node's neighbors should also exist in the same level
                var it = view.getNeighborsIterator(level, nodeInLevel);
                while (it.hasNext()) {
                    int neighbor = it.nextInt();
                    assertTrue(view.contains(level, neighbor));
                }

                // node should exist at every layer below it
                for (int lowerLevel = level - 1; lowerLevel >= 0; lowerLevel--) {
                    assertTrue(view.contains(lowerLevel, nodeInLevel));
                }
            }
        }

        // no holes in lowest level (not true for all graphs but true for the ones constructed here)
        for (int i = 0; i < graph.getIdUpperBound(); i++) {
            assertTrue(view.getNeighborsIterator(0, i).size() >= 0);
        }
    }

    // Make sure we actually approximately find the closest k elements. Mostly this is about
    // ensuring that we have all the distance functions, comparators, priority queues and so on
    // oriented in the right directions
    @Test
    public void testAknnDiverse() {
        testAknnDiverse(false);
        testAknnDiverse(true);
    }

    // Make sure we actually approximately find the closest k elements. Mostly this is about
    // ensuring that we have all the distance functions, comparators, priority queues and so on
    // oriented in the right directions
    public void testAknnDiverse(boolean addHierarchy) {
        int nDoc = 100;
        similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
        RandomAccessVectorValues vectors = circularVectorValues(nDoc);
        GraphIndexBuilder builder =
                new GraphIndexBuilder(vectors, similarityFunction, 20, 100, 1.0f, 1.4f, addHierarchy);
        var graph = TestUtil.buildSequentially(builder, vectors);
        validateIndex(graph);
        // run some searches
        SearchResult.NodeScore[] nn = GraphSearcher.search(getTargetVector(),
                10,
                vectors.copy(),
                similarityFunction,
                graph,
                Bits.ALL
        ).getNodes();
        int[] nodes = Arrays.stream(nn).mapToInt(nodeScore -> nodeScore.node).toArray();
        assertEquals("Number of found results is not equal to [10].", 10, nodes.length);
        int sum = 0;
        for (int node : nodes) {
            sum += node;
        }
        // We expect to get approximately 100% recall;
        // the lowest docIds are closest to zero; sum(0,9) = 45
        assertTrue("sum(result docs)=" + sum + " for " + ImmutableGraphIndex.prettyPrint(builder.graph), sum < 75);
    }

    @Test
    public void testSearchWithAcceptOrds() {
        testSearchWithAcceptOrds(false);
        testSearchWithAcceptOrds(true);
    }

    public void testSearchWithAcceptOrds(boolean addHierarchy) {
        int nDoc = 100;
        RandomAccessVectorValues vectors = circularVectorValues(nDoc);
        similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
        GraphIndexBuilder builder =
                new GraphIndexBuilder(vectors, similarityFunction, 32, 100, 1.0f, 1.4f, addHierarchy);
        var graph = TestUtil.buildSequentially(builder, vectors);
        validateIndex(graph);
        // the first 10 docs must not be deleted to ensure the expected recall
        Bits acceptOrds = createRandomAcceptOrds(10, nDoc);
        SearchResult.NodeScore[] nn = GraphSearcher.search(getTargetVector(),
                10,
                vectors.copy(),
                similarityFunction,
                graph,
                acceptOrds
        ).getNodes();
        int[] nodes = Arrays.stream(nn).mapToInt(nodeScore -> nodeScore.node).toArray();
        assertEquals("Number of found results is not equal to [10].", 10, nodes.length);
        int sum = 0;
        for (int node : nodes) {
            assertTrue("the results include a deleted document: " + node, acceptOrds.get(node));
            sum += node;
        }
        // We expect to get approximately 100% recall;
        // the lowest docIds are closest to zero; sum(0,9) = 45
        assertTrue("sum(result docs)=" + sum + " for " + ImmutableGraphIndex.prettyPrint(builder.graph), sum < 75);
    }

    @Test
    public void testSearchWithSelectiveAcceptOrds() {
        testSearchWithSelectiveAcceptOrds(false);
        testSearchWithSelectiveAcceptOrds(true);
    }

    public void testSearchWithSelectiveAcceptOrds(boolean addHierarchy) {
        int nDoc = 100;
        RandomAccessVectorValues vectors = circularVectorValues(nDoc);
        similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
        GraphIndexBuilder builder =
                new GraphIndexBuilder(vectors, similarityFunction, 32, 100, 1.0f, 1.4f, addHierarchy);
        var graph = TestUtil.buildSequentially(builder, vectors);
        validateIndex(graph);
        // Only mark a few vectors as accepted
        var acceptOrds = new FixedBitSet(nDoc);
        for (int i = 0; i < nDoc; i += nextInt(15, 20)) {
            acceptOrds.set(i);
        }

        // Check the search finds all accepted vectors
        int numAccepted = acceptOrds.cardinality();
        SearchResult.NodeScore[] nn = GraphSearcher.search(getTargetVector(),
                numAccepted,
                vectors.copy(),
                similarityFunction,
                graph,
                acceptOrds
        ).getNodes();

        int[] nodes = Arrays.stream(nn).mapToInt(nodeScore -> nodeScore.node).toArray();
        for (int node : nodes) {
            assertTrue(String.format("the results include a deleted document: %d for %s",
                    node, ImmutableGraphIndex.prettyPrint(builder.graph)), acceptOrds.get(node));
        }
        for (int i = 0; i < acceptOrds.length(); i++) {
            if (acceptOrds.get(i)) {
                int finalI = i;
                assertTrue(String.format("the results do not include an accepted document: %d for %s",
                        i, ImmutableGraphIndex.prettyPrint(builder.graph)), Arrays.stream(nodes).anyMatch(j -> j == finalI));
            }
        }
    }

    @Test
    public void testGraphIndexBuilderInvalid() {
        testGraphIndexBuilderInvalid(false);
        testGraphIndexBuilderInvalid(true);
    }

    public void testGraphIndexBuilderInvalid(boolean addHierarchy) {
        assertThrows(NullPointerException.class,
                () -> new GraphIndexBuilder((RandomAccessVectorValues) null, (VectorSimilarityFunction) null, 0, 0, 1.0f, 1.0f, addHierarchy));
        // M must be > 0
        assertThrows(IllegalArgumentException.class,
                () -> {
                    RandomAccessVectorValues vectors = vectorValues(1, 1);
                    new GraphIndexBuilder(vectors, similarityFunction, 0, 10, 1.0f, 1.0f, addHierarchy);
                });
        // beamWidth must be > 0
        assertThrows(IllegalArgumentException.class,
                () -> {
                    RandomAccessVectorValues vectors = vectorValues(1, 1);
                    new GraphIndexBuilder(vectors, similarityFunction, 10, 0, 1.0f, 1.0f, addHierarchy);
                });
    }

    // FIXME
    @Test
    public void testRamUsageEstimate() {
    }

    @Test
    public void testDiversity() {
        testDiversity(false);
        testDiversity(true);
    }

    public void testDiversity(boolean addHierarchy) {
        similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
        // Some carefully checked test cases with simple 2d vectors on the unit circle:
        VectorFloat<?>[] values = {
                unitVector2d(0.5),
                unitVector2d(0.75),
                unitVector2d(0.2),
                unitVector2d(0.9),
                unitVector2d(0.8),
                unitVector2d(0.77),
                unitVector2d(0.6)
        };
        MockVectorValues vectors = vectorValues(values);
        // First add nodes until everybody gets a full neighbor list
        GraphIndexBuilder builder =
                new GraphIndexBuilder(vectors, similarityFunction, 4, 10, 1.0f, 1.0f, addHierarchy);
        // node 0 is added by the builder constructor
        builder.addGraphNode(0, vectors.getVector(0));
        builder.addGraphNode(1, vectors.getVector(1));
        builder.addGraphNode(2, vectors.getVector(2));

        var view = builder.graph.getView();

        // now every node has tried to attach every other node as a neighbor, but
        // some were excluded based on diversity check.
        assertNeighbors(view, 0, 1, 2);
        assertNeighbors(view, 1, 0);
        assertNeighbors(view, 2, 0);

        builder.addGraphNode(3, vectors.getVector(3));

        view = builder.graph.getView();

        assertNeighbors(view, 0, 1, 2);
        // we added 3 here
        assertNeighbors(view, 1, 0, 3);
        assertNeighbors(view, 2, 0);
        assertNeighbors(view, 3, 1);

        // supplant an existing neighbor
        builder.addGraphNode(4, vectors.getVector(4));

        view = builder.graph.getView();

        // 4 is the same distance from 0 that 2 is; we leave the existing node in place
        assertNeighbors(view, 0, 1, 2);
        assertNeighbors(view, 1, 0, 3, 4);
        assertNeighbors(view, 2, 0);
        // 1 survives the diversity check
        assertNeighbors(view, 3, 1, 4);
        assertNeighbors(view, 4, 1, 3);

        builder.addGraphNode(5, vectors.getVector(5));

        view = builder.graph.getView();

        assertNeighbors(view, 0, 1, 2);
        assertNeighbors(view, 1, 0, 3, 4, 5);
        assertNeighbors(view, 2, 0);
        // even though 5 is closer, 3 is not a neighbor of 5, so no update to *its* neighbors occurs
        assertNeighbors(view, 3, 1, 4);
        assertNeighbors(view, 4, 1, 3, 5);
        assertNeighbors(view, 5, 1, 4);
    }

    @Test
    public void testDiversityFallback() {
        testDiversityFallback(false);
        testDiversityFallback(true);
    }

    public void testDiversityFallback(boolean addHierarchy) {
        similarityFunction = VectorSimilarityFunction.EUCLIDEAN;
        // Some test cases can't be exercised in two dimensions;
        // in particular if a new neighbor displaces an existing neighbor
        // by being closer to the target, yet none of the existing neighbors is closer to the new vector
        // than to the target -- ie they all remain diverse, so we simply drop the farthest one.
        VectorFloat<?>[] values = {
                vectorTypeSupport.createFloatVector(new float[]{0, 0, 0}),
                vectorTypeSupport.createFloatVector(new float[]{0, 10, 0}),
                vectorTypeSupport.createFloatVector(new float[]{0, 0, 20}),
                vectorTypeSupport.createFloatVector(new float[]{10, 0, 0}),
                vectorTypeSupport.createFloatVector(new float[]{0, 4, 0})
        };
        MockVectorValues vectors = vectorValues(values);
        // First add nodes until everybody gets a full neighbor list
        GraphIndexBuilder builder =
                new GraphIndexBuilder(vectors, similarityFunction, 2, 10, 1.0f, 1.0f, addHierarchy);
        builder.addGraphNode(0, vectors.getVector(0));
        builder.addGraphNode(1, vectors.getVector(1));
        builder.addGraphNode(2, vectors.getVector(2));

        var view = builder.graph.getView();

        assertNeighbors(view, 0, 1, 2);
        // 2 is closer to 0 than 1, so it is excluded as non-diverse
        assertNeighbors(view, 1, 0);
        // 1 is closer to 0 than 2, so it is excluded as non-diverse
        assertNeighbors(view, 2, 0);

        builder.addGraphNode(3, vectors.getVector(3));

        view = builder.graph.getView();

        // this is one case we are testing; 2 has been displaced by 3
        assertNeighbors(view, 0, 1, 3);
        assertNeighbors(view, 1, 0);
        assertNeighbors(view, 2, 0);
        assertNeighbors(view, 3, 0);
    }

    @Test
    public void testDiversity3d() {
        testDiversity3d(false);
        testDiversity3d(true);
    }

    public void testDiversity3d(boolean addHierarchy) {
        similarityFunction = VectorSimilarityFunction.EUCLIDEAN;
        // test the case when a neighbor *becomes* non-diverse when a newer better neighbor arrives
        VectorFloat<?>[] values = {
                vectorTypeSupport.createFloatVector(new float[]{0, 0, 0}),
                vectorTypeSupport.createFloatVector(new float[]{0, 10, 0}),
                vectorTypeSupport.createFloatVector(new float[]{0, 0, 20}),
                vectorTypeSupport.createFloatVector(new float[]{0, 9, 0})
        };
        MockVectorValues vectors = vectorValues(values);
        // First add nodes until everybody gets a full neighbor list
        GraphIndexBuilder builder =
                new GraphIndexBuilder(vectors, similarityFunction, 2, 10, 1.0f, 1.0f, addHierarchy);
        builder.addGraphNode(0, vectors.getVector(0));
        builder.addGraphNode(1, vectors.getVector(1));
        builder.addGraphNode(2, vectors.getVector(2));

        var view = builder.graph.getView();

        assertNeighbors(view, 0, 1, 2);
        // 2 is closer to 0 than 1, so it is excluded as non-diverse
        assertNeighbors(view, 1, 0);
        // 1 is closer to 0 than 2, so it is excluded as non-diverse
        assertNeighbors(view, 2, 0);

        builder.addGraphNode(3, vectors.getVector(3));

        view = builder.graph.getView();

        // this is one case we are testing; 1 has been displaced by 3
        assertNeighbors(view, 0, 2, 3);
        assertNeighbors(view, 1, 0, 3);
        assertNeighbors(view, 2, 0);
        assertNeighbors(view, 3, 0, 1);
    }

    private void assertNeighbors(ImmutableGraphIndex.View graph, int node, int... expected) {
        Arrays.sort(expected);
        NodesIterator it = graph.getNeighborsIterator(0, node);
        int[] actual = new int[it.size()];
        for (int i = 0; i < actual.length; i++) {
            actual[i] = it.next();
        }
        Arrays.sort(actual);
        assertArrayEquals(expected, actual);
    }

    @Test
    // build a random graph, then check that it has at least 90% recall
    public void testRandom() {
        testRandom(false);
        testRandom(true);
    }

    // build a random graph, then check that it has at least 90% recall
    public void testRandom(boolean addHierarchy) {
        int size = between(100, 150);
        int dim = between(2, 15);
        MockVectorValues vectors = vectorValues(size, dim);
        int topK = 5;
        GraphIndexBuilder builder = new GraphIndexBuilder(vectors, similarityFunction, 20, 30, 1.0f, 1.4f, addHierarchy);
        var graph = builder.build(vectors);
        validateIndex(graph);
        Bits acceptOrds = getRandom().nextBoolean() ? Bits.ALL : createRandomAcceptOrds(0, size);

        int efSearch = 100;
        int totalMatches = 0;
        for (int i = 0; i < 100; i++) {
            SearchResult.NodeScore[] actual;
            VectorFloat<?> query = randomVector(dim);
            actual = GraphSearcher.search(query,
                    efSearch,
                    vectors,
                    similarityFunction,
                    graph,
                    acceptOrds
            ).getNodes();

            NodeQueue expected = new NodeQueue(new BoundedLongHeap(topK), NodeQueue.Order.MIN_HEAP);
            for (int j = 0; j < size; j++) {
                if (vectors.getVector(j) != null && acceptOrds.get(j)) {
                    expected.push(j, similarityFunction.compare(query, vectors.getVector(j)));
                }
            }
            var actualNodeIds = Arrays.stream(actual, 0, topK).mapToInt(nodeScore -> nodeScore.node).toArray();

            assertEquals(topK, actualNodeIds.length);
            totalMatches += computeOverlap(actualNodeIds, expected.nodesCopy());
        }
        // with the current settings, we can visit every node in the graph, so this should actually be 100%
        // except in cases where the graph ends up partitioned.  If that happens, it probably means
        // a bug has been introduced in graph construction.
        double overlap = totalMatches / (double) (100 * topK);
        assertTrue("overlap=" + overlap, overlap > 0.9);
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

    @Test
    public void testConcurrentNeighbors() {
        testConcurrentNeighbors(false);
        testConcurrentNeighbors(true);
    }

    public void testConcurrentNeighbors(boolean addHierarchy) {
        RandomAccessVectorValues vectors = circularVectorValues(100);
        GraphIndexBuilder builder = new GraphIndexBuilder(vectors, similarityFunction, 2, 30, 1.0f, 1.4f, addHierarchy);
        var graph = builder.build(vectors);
        validateIndex(graph);
        var view = graph.getView();
        for (int i = 0; i < vectors.size(); i++) {
            assertTrue(view.getNeighborsIterator(0, i).size() <= 2); // TODO
        }
    }

    @Test
    public void testZeroCentroid() {
        testZeroCentroid(false);
        testZeroCentroid(true);
    }

    public void testZeroCentroid(boolean addHierarchy) {
        var rawVectors = List.of(vectorTypeSupport.createFloatVector(new float[] {-1, -1}),
                                 vectorTypeSupport.createFloatVector(new float[] {1, 1}));
        var vectors = new ListRandomAccessVectorValues(rawVectors, 2);
        var builder = new GraphIndexBuilder(vectors, VectorSimilarityFunction.COSINE, 2, 2, 1.0f, 1.0f, addHierarchy);
        try (var graph = builder.build(vectors)) {
            validateIndex(graph);
            var qv = vectorTypeSupport.createFloatVector(new float[] {0.5f, 0.5f});
            var results = GraphSearcher.search(qv, 1, vectors, VectorSimilarityFunction.COSINE, graph, Bits.ALL);
            assertEquals(1, results.getNodes().length);
            assertEquals(1, results.getNodes()[0].node);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Returns vectors evenly distributed around the upper unit semicircle.
     */
    public static class CircularFloatVectorValues implements RandomAccessVectorValues {
        private final int size;

        public CircularFloatVectorValues(int size) {
            this.size = size;
        }

        @Override
        public CircularFloatVectorValues copy() {
            return new CircularFloatVectorValues(size);
        }

        @Override
        public int dimension() {
            return 2;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public VectorFloat<?> getVector(int ord) {
            return unitVector2d(ord / (double) size);
        }

        @Override
        public boolean isValueShared() {
            return false;
        }
    }

    private static VectorFloat<?> unitVector2d(double piRadians) {
        return vectorTypeSupport.createFloatVector(new float[]{
                (float) Math.cos(Math.PI * piRadians), (float) Math.sin(Math.PI * piRadians)
        });
    }

    public static VectorFloat<?>[] createRandomFloatVectors(int size, int dimension, Random R) {
        return IntStream.range(0, size).mapToObj(i -> TestUtil.randomVector(R, dimension))
                .toArray(sz -> new VectorFloat<?>[sz]);
    }

    public static VectorFloat<?>[] createRandomFloatVectorsParallel(int size, int dimension) {
        return IntStream.range(0, size).parallel()
                .mapToObj(i -> TestUtil.randomVector(ThreadLocalRandom.current(), dimension))
                .toArray(sz -> new VectorFloat<?>[sz]);
    }

    /**
     * Generate a random bitset where before startIndex all bits are set, and after startIndex each
     * entry has a 2/3 probability of being set.
     */
    protected static Bits createRandomAcceptOrds(int startIndex, int length) {
        FixedBitSet bits = new FixedBitSet(length);
        // all bits are set before startIndex
        for (int i = 0; i < startIndex; i++) {
            bits.set(i);
        }
        // after startIndex, bits are set with 2/3 probability
        for (int i = startIndex; i < bits.length(); i++) {
            if (getRandom().nextFloat() < 0.667f) {
                bits.set(i);
            }
        }
        return bits;
    }

    // -----------------------------------------------------------------------
    // Byte-vector (int8) graph construction tests
    // -----------------------------------------------------------------------

    /** Build a small int8 graph using every similarity function and verify it is navigable. */
    @Test
    public void testByteVectorBuildAllSimilarityFunctions() {
        for (var bvsf : ByteVectorSimilarityFunction.values()) {
            testByteVectorBuild(bvsf, false);
            testByteVectorBuild(bvsf, true);
        }
    }

    private void testByteVectorBuild(ByteVectorSimilarityFunction bvsf, boolean addHierarchy) {
        int n = 50, dim = 8;
        var rabvv = randomByteVectorValues(n, dim);
        var builder = new GraphIndexBuilder(rabvv, bvsf, 8, 20, 1.2f, 1.2f, addHierarchy);
        var graph = builder.build(rabvv);
        validateIndex(graph);
        assertNotNull("entry node must be set", graph.getView().entryNode());
        assertEquals(n, graph.size(0));
    }

    /** addGraphNode(int, ByteSequence) inserts nodes one by one and yields the right count. */
    @Test
    public void testByteVectorAddGraphNodeOneByOne() {
        int n = 30, dim = 4;
        var rabvv = randomByteVectorValues(n, dim);
        var builder = new GraphIndexBuilder(rabvv, ByteVectorSimilarityFunction.EUCLIDEAN, 4, 10, 1.2f, 1.2f, false);
        for (int i = 0; i < n; i++) {
            builder.addGraphNode(i, rabvv.getVector(i));
        }
        builder.cleanup();
        var graph = builder.getGraph();
        validateIndex(graph);
        assertEquals(n, graph.size(0));
        // every node's neighbor count is within the declared degree
        var view = graph.getView();
        for (int i = 0; i < n; i++) {
            assertTrue(view.getNeighborsIterator(0, i).size() <= graph.getDegree(0));
        }
    }

    /**
     * addGraphNode(int, ByteSequence) must throw when the builder was constructed with a
     * float-vector score provider.
     */
    @Test
    public void testByteAddGraphNodeThrowsOnFloatBuilder() {
        var floatRavv = circularVectorValues(10);
        var builder = new GraphIndexBuilder(floatRavv, VectorSimilarityFunction.EUCLIDEAN, 4, 10, 1.2f, 1.2f, false);
        var bs = vectorTypeSupport.createByteSequence(4);
        assertThrows(UnsupportedOperationException.class, () -> builder.addGraphNode(0, bs));
    }

    /**
     * build(VectorValues<?>) dispatches correctly when given a RandomAccessByteVectorValues.
     */
    @Test
    public void testByteVectorBuildViaGenericBuildMethod() {
        int n = 40, dim = 6;
        var rabvv = randomByteVectorValues(n, dim);
        var builder = new GraphIndexBuilder(rabvv, ByteVectorSimilarityFunction.DOT_PRODUCT, 6, 20, 1.2f, 1.2f, false);
        // call the generic VectorValues<?> overload explicitly
        var graph = builder.build((VectorValues<?>) rabvv);
        validateIndex(graph);
        assertEquals(n, graph.size(0));
    }

    /** refineFinalGraph=false and refineFinalGraph=true both produce valid graphs. */
    @Test
    public void testByteVectorRefineFinalGraphVariants() {
        int n = 40, dim = 4;
        for (boolean refine : new boolean[]{false, true}) {
            var rabvv = randomByteVectorValues(n, dim);
            // Use the 7-arg constructor (refineFinalGraph defaults to true); test both via the
            // BSP constructor which exposes the refineFinalGraph knob
            var bsp = io.github.jbellis.jvector.graph.similarity.BuildScoreProvider.byteVectorScoreProvider(
                    rabvv, ByteVectorSimilarityFunction.EUCLIDEAN);
            var builder = new GraphIndexBuilder(bsp, dim, 4, 20, 1.2f, 1.2f, false, refine);
            var graph = builder.build(rabvv);
            validateIndex(graph);
            assertEquals(n, graph.size(0));
            var view = graph.getView();
            int ub = graph.getIdUpperBound();
            for (int i = 0; i < n; i++) {
                for (var it = view.getNeighborsIterator(0, i); it.hasNext(); ) {
                    int nb = it.nextInt();
                    assertTrue("neighbor " + nb + " out of bounds", nb >= 0 && nb < ub);
                }
            }
        }
    }

    /** Smoke-test: a built byte-vector graph achieves reasonable top-5 recall. */
    @Test
    public void testByteVectorSearchRecall() {
        int n = 200, dim = 16;
        int topK = 5;
        var rabvv  = randomByteVectorValues(n, dim);
        var bvsf   = ByteVectorSimilarityFunction.EUCLIDEAN;
        var builder = new GraphIndexBuilder(rabvv, bvsf, 16, 50, 1.2f, 1.2f, false);
        var graph  = builder.build(rabvv);

        int totalMatches = 0;
        int trials = 100;
        for (int t = 0; t < trials; t++) {
            byte[] rawQ = new byte[dim];
            getRandom().nextBytes(rawQ);
            ByteSequence<?> query = vectorTypeSupport.createByteSequence(rawQ);

            // brute-force top-K
            NodeQueue expected = new NodeQueue(new BoundedLongHeap(topK), NodeQueue.Order.MIN_HEAP);
            for (int i = 0; i < n; i++) {
                expected.push(i, bvsf.compare(query, rabvv.getVector(i)));
            }

            // graph search with efSearch = 100 (larger than topK to explore more candidates)
            var ssp = new DefaultSearchScoreProvider(
                    (io.github.jbellis.jvector.graph.similarity.ScoreFunction.ExactScoreFunction)
                    node -> bvsf.compare(query, rabvv.getVector(node)));
            var result = new GraphSearcher(graph).search(ssp, topK, 100, 0f, 0f, Bits.ALL);
            var actualNodeIds = Arrays.stream(result.getNodes(), 0, topK)
                    .mapToInt(ns -> ns.node).toArray();

            totalMatches += computeOverlap(actualNodeIds, expected.nodesCopy());
        }
        // with efSearch=100 over a 200-node graph, we can visit most nodes; expect >=90% recall
        double overlap = totalMatches / (double) (trials * topK);
        assertTrue("byte-vector recall " + overlap + " is too low", overlap > 0.9);
    }

    /** Creates a list-backed RandomAccessByteVectorValues with random int8 vectors. */
    private ListRandomAccessByteVectorValues randomByteVectorValues(int n, int dim) {
        var list = new java.util.ArrayList<ByteSequence<?>>(n);
        for (int i = 0; i < n; i++) {
            byte[] raw = new byte[dim];
            getRandom().nextBytes(raw);
            list.add(vectorTypeSupport.createByteSequence(raw));
        }
        return new ListRandomAccessByteVectorValues(list, dim);
    }
}
