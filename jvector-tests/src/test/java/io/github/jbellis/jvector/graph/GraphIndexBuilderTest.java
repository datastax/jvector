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
     * Regression test: {@link GraphIndexBuilder#rescore} is used by Cassandra to refine the PQ
     * codebook mid-build, without pausing insertion of the remaining rows. rescore() copies each
     * node's edges into the new builder via connectNode(), which does not mark the node complete
     * in the new builder's CompletionTracker. A node stuck at the tracker's default completion
     * time (Integer.MAX_VALUE) is filtered out of every ConcurrentGraphIndexView taken on the new
     * graph from then on -- including the entry node's own neighbor list -- which is exactly the
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

        // Simulates CassandraOnHeapGraph/CompactionGraph refining its PQ codebook mid-build
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
}
