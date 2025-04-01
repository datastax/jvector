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
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.TestUtil.assertGraphEquals;
import static io.github.jbellis.jvector.graph.TestVectorGraph.createRandomFloatVectors;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestDeletions extends LuceneTestCase {
    @Test
    public void testMarkDeleted() {
        testMarkDeleted(false);
        testMarkDeleted(true);
    }

    public void testMarkDeleted(boolean addHierarchy) {
        // graph of 10 vectors
        int dimension = 2;
        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(10, dimension, getRandom()));
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 4, 10, 1.0f, 1.0f, addHierarchy);
        var graph = TestUtil.buildSequentially(builder, ravv);

        // delete a random entry
        int n = getRandom().nextInt(ravv.size());
        builder.markNodeDeleted(n);
        // check that searching for random vectors never results in the deleted one
        for (int i = 0; i < 100; i++) {
            var v = TestUtil.randomVector(getRandom(), dimension);
            var results = GraphSearcher.search(v, 3, ravv, VectorSimilarityFunction.COSINE, graph, Bits.ALL);
            for (var ns : results.getNodes()) {
                assertNotEquals(n, ns.node);
            }
        }
        // check that asking for the entire graph back still doesn't surface the deleted one
        var v = ravv.getVector(n).copy();
        var results = GraphSearcher.search(v, ravv.size(), ravv, VectorSimilarityFunction.COSINE, graph, Bits.ALL);
        assertEquals(GraphIndex.prettyPrint(graph), ravv.size() - 1, results.getNodes().length);
        for (var ns : results.getNodes()) {
            assertNotEquals(n, ns.node);
        }
    }

    @Test
    public void testCleanup() throws IOException {
        testCleanup(false);
        testCleanup(true);
    }

    public void testCleanup(boolean addHierarchy) throws IOException {
        // graph of 10 vectors
        int dimension = 2;
        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(10, dimension, getRandom()));
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 4, 10, 1.0f, 1.0f, addHierarchy);
        var graph = TestUtil.buildSequentially(builder, ravv);

        // delete all nodes that connect to a random node
        int nodeToIsolate = getRandom().nextInt(ravv.size());
        int nDeleted = 0;
        try (var view = graph.getView()) {
            for (var i = 0; i < view.size(); i++) {
                for (var it = view.getNeighborsIterator(0, i); it.hasNext(); ) { // TODO hardcoded level
                    if (nodeToIsolate == it.nextInt()) {
                        builder.markNodeDeleted(i);
                        nDeleted++;
                        break;
                    }
                }
            }
        }
        assertNotEquals(0, nDeleted);

        // cleanup removes the deleted nodes
        builder.cleanup();
        assertEquals(ravv.size() - nDeleted, graph.size());

        // cleanup should have added new connections to the node that would otherwise have been disconnected
        var v = ravv.getVector(nodeToIsolate).copy();
        var results = GraphSearcher.search(v, 10, ravv, VectorSimilarityFunction.COSINE, graph, Bits.ALL);
        assertEquals(nodeToIsolate, results.getNodes()[0].node);

        // check that we can save and load the graph with "holes" from the deletion
        var testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
        var outputPath = testDirectory.resolve("on_heap_graph");
        try (var out = TestUtil.openDataOutputStream(outputPath)) {
            graph.save(out);
        }
        // TODO when we fix load()
//        var b2 = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f);
//        try (var readerSupplier = new SimpleMappedReader.Supplier(outputPath)) {
//            b2.load(readerSupplier.get());
//        }
//        var reloadedGraph = b2.getGraph();
//        assertGraphEquals(graph, reloadedGraph);
    }

    @Test
    public void testMarkingAllNodesAsDeleted() {
        testMarkingAllNodesAsDeleted(false);
        testMarkingAllNodesAsDeleted(true);
    }

    public void testMarkingAllNodesAsDeleted(boolean addHierarchy) {
        // build graph
        int dimension = 2;
        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(10, dimension, getRandom()));
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f, addHierarchy);
        var graph = TestUtil.buildSequentially(builder, ravv);

        // mark all deleted
        for (var i = 0; i < graph.size(); i++) {
            graph.markDeleted(i);
        }

        // removeDeletedNodes should leave the graph empty
        builder.removeDeletedNodes();
        assertEquals(0, graph.size());
        assertNull(graph.entry());
    }

    @Test
    public void testNoPathToLiveNodesWhenRemovingDeletedNodes2() throws IOException {
        testNoPathToLiveNodesWhenRemovingDeletedNodes2(false);
        testNoPathToLiveNodesWhenRemovingDeletedNodes2(true);
    }

    public void testNoPathToLiveNodesWhenRemovingDeletedNodes2(boolean addHierarchy) throws IOException {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var random = getRandom();
        // generate two clusters of vectors
        var ravv = MockVectorValues.fromValues(
                IntStream.range(0, 1100).mapToObj(i -> {
                    if (i < 1000) {
                        return vts.createFloatVector(new float[]{0.01f + 100 * random.nextFloat(), 0.01f + 100 * random.nextFloat()});
                    } else {
                        return vts.createFloatVector(new float[]{10_000.0f + 100 * random.nextFloat(), 10_000.0f + 100 * random.nextFloat()});
                    }
                }).toArray(VectorFloat<?>[]::new)
        );

        // add the vectors, then delete all the ones from the first (larger) cluster
        try (var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 4, 10, 1.0f, 1.0f, addHierarchy)) {
            for (int i = 0; i < 1100; i++) {
                builder.addGraphNode(i, ravv.getVector(i));
            }

            for (int i = 0; i < 1000; i++) {
                builder.markNodeDeleted(i);
            }

            builder.cleanup();
            assert builder.graph.getView().entryNode() != null;
        }
    }
}
