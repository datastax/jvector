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

import static io.github.jbellis.jvector.graph.TestVectorGraph.createRandomFloatVectors;
import static org.junit.Assert.assertEquals;
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
        var neighbors = builder.graph.getNeighbors(0, 0); // TODO
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
        var newGraph = rescored.getGraph();
        assertTrue(newGraph.containsNode(0));
        assertTrue(newGraph.containsNode(1));
        assertTrue(newGraph.containsNode(2));

        // Check node 0's neighbors, score and order should be different
        var newNeighbors = newGraph.getNeighbors(0, 0); // TODO
        assertEquals(2, newNeighbors.getNode(0));
        assertEquals(1, newNeighbors.getNode(1));
        assertEquals(0.2f, newNeighbors.getScore(0), 1E-6);
        assertEquals(0.05882353f, newNeighbors.getScore(1), 1E-6);

    }

    // TODO
//    @Test
//    public void testSaveAndLoad() throws IOException {
//        int dimension = randomIntBetween(2, 32);
//        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(randomIntBetween(10, 100), dimension, getRandom()));
//        Supplier<GraphIndexBuilder> newBuilder = () ->
//            new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f);
//
//        var indexDataPath = testDirectory.resolve("index_builder.data");
//        var builder = newBuilder.get();
//
//        try (var graph = TestUtil.buildSequentially(builder, ravv);
//             var out = TestUtil.openDataOutputStream(indexDataPath))
//        {
//            graph.save(out);
//        }
//
//        builder = newBuilder.get();
//        try(var readerSupplier = new SimpleMappedReader.Supplier(indexDataPath)) {
//            builder.load(readerSupplier.get());
//        }
//
//        assertEquals(ravv.size(), builder.graph.size());
//        for (int i = 0; i < ravv.size(); i++) {
//            assertTrue(builder.graph.containsNode(i));
//        }
//    }
}
