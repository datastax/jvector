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
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
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
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Supplier;

import static io.github.jbellis.jvector.TestUtil.assertGraphEquals;
import static io.github.jbellis.jvector.graph.TestVectorGraph.createRandomFloatVectors;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class GraphIndexBuilderTest extends LuceneTestCase {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    private Path testDirectory;

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory(getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    @Test
    public void testEstimatedBytes() throws IOException {
        // Create test vectors where each vector is [node_id, 0]
        var vectors = new ArrayList<VectorFloat<?>>();
        vectors.add(vts.createFloatVector(new float[] {0, 0}));
        vectors.add(vts.createFloatVector(new float[] {0, 1}));
        vectors.add(vts.createFloatVector(new float[] {2, 0}));
        var ravv = new ListRandomAccessVectorValues(vectors, 2);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
        try (var builder = new GraphIndexBuilder(bsp, 2, 2, 10, 1.0f, 1.0f, false)) {
            var bytesUsed = builder.addGraphNode(0, ravv.getVector(0));
            // The actual value is not critical, but this confirms we do not get unexpected changes (for this config)
            assertEquals(92, bytesUsed);
        }
    }

    @Test
    public void testRescore() {
        testRescore(false);
        testRescore(true);
    }

    public void testRescore(boolean addHierarchy) {
        // Create test vectors where each vector is [node_id, 0]
        var vectors = new ArrayList<VectorFloat<?>>();
        vectors.add(vts.createFloatVector(new float[] {0, 0}));
        vectors.add(vts.createFloatVector(new float[] {0, 1}));
        vectors.add(vts.createFloatVector(new float[] {2, 0}));
        var ravv = new ListRandomAccessVectorValues(vectors, 2);
        
        // Initial score provider uses dot product, so scores will equal node IDs
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
        var builder = new GraphIndexBuilder(bsp, 2, 2, 10, 1.0f, 1.0f, addHierarchy);

        // Add 3 nodes
        builder.addGraphNode(0, ravv.getVector(0));
        builder.addGraphNode(1, ravv.getVector(1));
        builder.addGraphNode(2, ravv.getVector(2));

        var ohgi = (OnHeapGraphIndex) builder.graph;
        var neighbors = ohgi.getNeighbors(0, 0);
        assertEquals(1, neighbors.getNode(0));
        assertEquals(2, neighbors.getNode(1));
        assertEquals(0.5f, neighbors.getScore(0), 1E-6);
        assertEquals(0.2f, neighbors.getScore(1), 1E-6);

        // Create new vectors where each is [-node_id, 0] so dot products will be negative node IDs
        vectors.clear();
        vectors.add(vts.createFloatVector(new float[] {0, 0}));
        vectors.add(vts.createFloatVector(new float[] {0, 4}));
        vectors.add(vts.createFloatVector(new float[] {2, 0}));

        // Rescore the graph
        // (The score provider didn't change, but the vectors did, which provides the same effect)
        var rescored = GraphIndexBuilder.rescore(builder, bsp);

        // Verify edges still exist
        var newGraph = (OnHeapGraphIndex) rescored.getGraph();
        assertTrue(newGraph.containsNode(0));
        assertTrue(newGraph.containsNode(1));
        assertTrue(newGraph.containsNode(2));

        // Check node 0's neighbors, score and order should be different
        var newNeighbors = newGraph.getNeighbors(0, 0);
        assertEquals(2, newNeighbors.getNode(0));
        assertEquals(1, newNeighbors.getNode(1));
        assertEquals(0.2f, newNeighbors.getScore(0), 1E-6);
        assertEquals(0.05882353f, newNeighbors.getScore(1), 1E-6);

    }

    @Test
    public void testRescoreMarksCopiedNodesComplete() {
        testRescoreMarksCopiedNodesComplete(false);
        testRescoreMarksCopiedNodesComplete(true);
    }

    /**
     * Regression test: {@link GraphIndexBuilder#rescore} is used to refine the PQ
     * codebook mid-build, without pausing insertion of the remaining rows. rescore() copies each
     * node's edges into the new builder via connectNode(), which does not mark the node complete
     * in the new builder's CompletionTracker. A node stuck at the tracker's default completion
     * time (Integer.MAX_VALUE) is filtered out of every ConcurrentGraphIndexView taken on the new
     * graph from then on, including the entry node's own neighbor list, which is exactly the
     * view addGraphNode() and cleanup() use while the graph is still mutable (allMutationsCompleted
     * == false). So a fresh concurrent view taken right after rescore() must already see every
     * copied edge, not an empty/filtered list.
     */
    public void testRescoreMarksCopiedNodesComplete(boolean addHierarchy) {
        int dimension = 8;
        int count = 50;

        var vectors = createRandomFloatVectors(count, dimension, getRandom());
        var ravv = MockVectorValues.fromValues(vectors);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.COSINE);

        var builder = new GraphIndexBuilder(bsp, dimension, 8, 30, 1.2f, 1.2f, addHierarchy);
        for (int i = 0; i < count; i++) {
            builder.addGraphNode(i, ravv.getVector(i));
        }

        // Simulates calling application refining its PQ codebook mid-build
        var rescored = GraphIndexBuilder.rescore(builder, bsp);
        var rescoredGraph = (OnHeapGraphIndex) rescored.getGraph();

        // The graph is still mutable at this point (cleanup() hasn't run), so getView() returns a
        // ConcurrentGraphIndexView that filters on completion time.
        assertTrue(!rescoredGraph.allMutationsCompleted());
        var view = rescoredGraph.getView();

        int entry = view.entryNode().node;
        int rawEntryDegree = rescoredGraph.getNeighbors(0, entry).size();
        assertTrue("test setup: entry node should have neighbors after rescore", rawEntryDegree > 0);

        int visibleEntryDegree = view.getNeighborsIterator(0, entry).size();
        assertEquals("entry node's neighbors are hidden from a fresh concurrent view after rescore -- " +
                     "rescore() must mark copied nodes complete",
                     rawEntryDegree, visibleEntryDegree);
    }

    @Test
    public void testSaveAndLoad() throws IOException {
        int dimension = randomIntBetween(2, 32);
        int size = randomIntBetween(10, 100);
        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(size, dimension, getRandom()));

        Supplier<GraphIndexBuilder> newBuilder = () ->
            new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f, true);

        var indexDataPath = testDirectory.resolve("index_builder.data");
        var builder = newBuilder.get();

        var graph = TestUtil.buildSequentially(builder, ravv);

        try (var out = TestUtil.openDataOutputStream(indexDataPath)) {
            ((OnHeapGraphIndex) graph).save(out);
        }

        builder = newBuilder.get();
        try(var readerSupplier = new SimpleMappedReader.Supplier(indexDataPath)) {
            builder.load(readerSupplier.get());
        }

        assertEquals(ravv.size(), builder.graph.size(0));
        for (int i = 0; i < ravv.size(); i++) {
            assertTrue(builder.graph.containsNode(i));
        }
        assertGraphEquals(graph, builder.graph);
    }

    @Test
    public void testSaveAndLoadEmptyGraph() throws IOException {
        int dimension = randomIntBetween(2, 32);
        var ravv = MockVectorValues.empty(dimension);

        Supplier<GraphIndexBuilder> newBuilder = () ->
            new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f, true);

        var indexDataPath = testDirectory.resolve("index_builder_empty.data");
        var builder = newBuilder.get();

        var graph = TestUtil.buildSequentially(builder, ravv);

        try (var out = TestUtil.openDataOutputStream(indexDataPath)) {
            ((OnHeapGraphIndex) graph).setAllMutationsCompleted();
            ((OnHeapGraphIndex) graph).save(out);
        }

        builder = newBuilder.get();
        try(var readerSupplier = new SimpleMappedReader.Supplier(indexDataPath)) {
            builder.load(readerSupplier.get());
        }

        assertEquals(ravv.size(), builder.graph.size(0));
        assertNull(builder.graph.entryNode());
        assertGraphEquals(graph, builder.graph);
    }

    // Because RandomAccessVectorValues is exposed in such a way that it allows for subsequent additions to the
    // vector source, we need to ensure that GraphIndexBuilder can handle this.
    @Test
    public void testAddNodesToVectorValuesIteratively() throws IOException {
        int dimension = randomIntBetween(2, 32);
        var mutableVectors = new ArrayList<VectorFloat<?>>();
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(mutableVectors, dimension);
        try (var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f, true)) {
            for (int i = 0; i < 10; i++) {
                mutableVectors.add(TestUtil.randomVector(random(), dimension));
                builder.addGraphNode(i, ravv.getVector(i));
            }
        }
    }

    @Test
    public void testFullPrecisionNeighborUniquenessAcrossLayers() throws IOException {
        var random = new Random(522L);
        var vectors = new ArrayList<VectorFloat<?>>();
        for (int node = 0; node < 192; node++) {
            float[] values = new float[16];
            double norm = 0;
            for (int d = 0; d < values.length; d++) {
                values[d] = random.nextFloat() - 0.5f;
                norm += values[d] * values[d];
            }
            for (int d = 0; d < values.length; d++) {
                values[d] /= (float) Math.sqrt(norm);
            }
            vectors.add(vts.createFloatVector(values));
        }
        var ravv = new ListRandomAccessVectorValues(vectors, 16);
        for (boolean hierarchy : new boolean[] {false, true}) {
            for (int degree : new int[] {2, 3, 16, 32, 64}) {
                try (var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.DOT_PRODUCT,
                        degree, 100, 1.2f, 1.2f, hierarchy)) {
                    for (int node = 0; node < ravv.size(); node++) {
                        builder.addGraphNode(node, ravv.getVector(node));
                    }
                    assertUniqueNeighbors(builder.getGraph());
                    builder.cleanup();
                    assertUniqueNeighbors(builder.getGraph());
                }
            }
        }
    }

    /**
     * PQ build scores are asymmetric: a node's own search and a backlink from its neighbor can score
     * the same edge differently, which defeated the old score-local duplicate check.
     */
    @Test
    public void testPQNeighborUniquenessAcrossLayers() throws IOException {
        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(2000, 32, getRandom()));
        var pq = ProductQuantization.compute(ravv, 8, 256, true);
        var bsp = BuildScoreProvider.pqBuildScoreProvider(VectorSimilarityFunction.EUCLIDEAN, (PQVectors) pq.encodeAll(ravv));
        for (int degree : new int[] {8, 16, 32}) {
            try (var builder = new GraphIndexBuilder(bsp, ravv.dimension(), degree, 100, 1.2f, 1.2f, true)) {
                for (int node = 0; node < ravv.size(); node++) {
                    builder.addGraphNode(node, ravv.getVector(node));
                }
                assertUniqueNeighbors(builder.getGraph());
                builder.cleanup();
                assertUniqueNeighbors(builder.getGraph());
            }
        }
    }

    private static void assertUniqueNeighbors(ImmutableGraphIndex graph) throws IOException {
        try (var view = graph.getView()) {
            for (int level = 0; level <= graph.getMaxLevel(); level++) {
                var nodes = graph.getNodes(level);
                while (nodes.hasNext()) {
                    int source = nodes.nextInt();
                    var seen = new HashSet<Integer>();
                    var neighbors = view.getNeighborsIterator(level, source);
                    while (neighbors.hasNext()) {
                        int target = neighbors.nextInt();
                        assertTrue("Repeated edge at level " + level + ": " + source + " -> " + target,
                                seen.add(target));
                    }
                }
            }
        }
    }

    /**
     * DOT_PRODUCT scores are (1 + dot) / 2, which is negative for non-normalized vectors. A vector that is dissimilar
     * to all the others must still get neighbors when it is inserted; searching with a threshold of 0 discards every
     * candidate it finds and leaves it with none.
     */
    @Test
    public void testNodeWithOnlyNegativeSimilaritiesGetsNeighbors() throws IOException {
        for (int position : new int[] {0, 25, 50, 100}) {
            var random = new Random(1);
            var vectors = new ArrayList<VectorFloat<?>>();
            for (int i = 0; i < 200; i++) {
                vectors.add(vts.createFloatVector(new float[] {10f + (float) random.nextGaussian(), (float) random.nextGaussian()}));
            }
            vectors.add(position, vts.createFloatVector(new float[] {-10f, 0f}));
            var ravv = new ListRandomAccessVectorValues(vectors, 2);
            var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.DOT_PRODUCT);
            try (var builder = new GraphIndexBuilder(bsp, 2, 8, 40, 1.2f, 1.2f, false)) {
                for (int node = 0; node < ravv.size(); node++) {
                    builder.addGraphNode(node, ravv.getVector(node));
                }
                try (var view = builder.getGraph().getView()) {
                    assertTrue("node " + position + " has no neighbors", view.getNeighborsIterator(0, position).size() > 0);
                }
            }
        }
    }

    /**
     * The layer-0 search in addGraphNode used a rerank floor of 0. With a score provider that has a reranker, the
     * candidates whose approximate score is negative (DOT_PRODUCT with non-normalized vectors) were then not reranked,
     * so a node dissimilar to everything got at most the single best candidate instead of its neighbors.
     */
    @Test
    public void testNegativeScoresAreRerankedWhenInsertingWithAReranker() throws IOException {
        var random = new Random(1);
        var vectors = new ArrayList<VectorFloat<?>>();
        for (int i = 0; i < 200; i++) {
            vectors.add(vts.createFloatVector(new float[] {10f + (float) random.nextGaussian(), (float) random.nextGaussian()}));
        }
        int position = 50;
        vectors.add(position, vts.createFloatVector(new float[] {-10f, 0f}));
        var ravv = new ListRandomAccessVectorValues(vectors, 2);
        var exact = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.DOT_PRODUCT);
        var withReranker = new BuildScoreProvider() {
            @Override
            public boolean isExact() {
                return exact.isExact();
            }

            @Override
            public VectorFloat<?> approximateCentroid() {
                return exact.approximateCentroid();
            }

            @Override
            public SearchScoreProvider searchProviderFor(VectorFloat<?> vector) {
                return withReranker(exact.searchProviderFor(vector));
            }

            @Override
            public SearchScoreProvider searchProviderFor(int node1) {
                return withReranker(exact.searchProviderFor(node1));
            }

            @Override
            public SearchScoreProvider diversityProviderFor(int node1) {
                return exact.diversityProviderFor(node1);
            }

            private SearchScoreProvider withReranker(SearchScoreProvider provider) {
                var scoreFunction = (ScoreFunction.ExactScoreFunction) provider.scoreFunction();
                return new DefaultSearchScoreProvider(scoreFunction, scoreFunction);
            }
        };
        try (var builder = new GraphIndexBuilder(withReranker, 2, 8, 40, 1.2f, 1.2f, false)) {
            for (int node = 0; node < ravv.size(); node++) {
                builder.addGraphNode(node, ravv.getVector(node));
            }
            try (var view = builder.getGraph().getView()) {
                int degree = view.getNeighborsIterator(0, position).size();
                assertTrue("node " + position + " has " + degree + " neighbors", degree > 1);
            }
        }
    }

    /**
     * improveConnections runs from cleanup() on the nodes of the upper layers of a hierarchical graph. It must refine a
     * node whose similarity to everything is negative (DOT_PRODUCT with non-normalized vectors) like any other: with a
     * search threshold of 0 it found no candidates for such a node and left its neighbor lists as insertion made them.
     * The vector goes in at positions where it lands in layer 1 or above (layers depend only on the builder's seeded
     * random generator and the insertion order), and refinement is judged by the smallest neighbor count the node
     * ends up with in any of its layers, in most of the builds so that the check does not hinge on a single graph.
     */
    @Test
    public void testImproveConnectionsRefinesUpperLayerNodeWithOnlyNegativeSimilarities() throws IOException {
        int[] positions = {71, 72, 85, 87, 138};
        int refined = 0;
        for (int position : positions) {
            var random = new Random(1);
            var vectors = new ArrayList<VectorFloat<?>>();
            for (int i = 0; i < 200; i++) {
                vectors.add(vts.createFloatVector(new float[] {10f + (float) random.nextGaussian(), (float) random.nextGaussian()}));
            }
            vectors.add(position, vts.createFloatVector(new float[] {-10f, 0f}));
            var ravv = new ListRandomAccessVectorValues(vectors, 2);
            var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.DOT_PRODUCT);
            var pool = new ForkJoinPool(1);
            try (var builder = new GraphIndexBuilder(bsp, 2, List.of(8), 40, 1.2f, 1.2f, true, true, pool, pool)) {
                for (int node = 0; node < ravv.size(); node++) {
                    builder.addGraphNode(node, ravv.getVector(node));
                }
                var graph = (OnHeapGraphIndex) builder.getGraph();
                int topLevel = graph.getMaxLevelForNode(position);
                assertTrue("the vector at " + position + " is expected to be in layer 1 or above", topLevel >= 1);

                builder.cleanup();

                int smallest = Integer.MAX_VALUE;
                try (var view = graph.getView()) {
                    for (int level = 0; level <= topLevel; level++) {
                        smallest = Math.min(smallest, view.getNeighborsIterator(level, position).size());
                    }
                }
                if (smallest >= 4) {
                    refined++;
                }
            } finally {
                pool.shutdown();
            }
        }
        assertTrue("only " + refined + " of " + positions.length + " builds refined the node", refined >= 4);
    }
}
