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
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessByteVectorValues;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.TestVectorGraph;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineByteVectors;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.disk.feature.SeparatedNVQ;
import io.github.jbellis.jvector.graph.disk.feature.SeparatedVectors;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.ByteVectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

import static io.github.jbellis.jvector.TestUtil.getNeighborNodes;
import static io.github.jbellis.jvector.TestUtil.randomVector;
import static org.junit.Assert.*;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestOnDiskGraphIndex extends RandomizedTest {

    private Path testDirectory;

    private TestUtil.FullyConnectedGraphIndex fullyConnectedGraph;
    private TestUtil.RandomlyConnectedGraphIndex randomlyConnectedGraph;
    private ImmutableGraphIndex emptyGraph;

    @Before
    public void setup() throws IOException {
        fullyConnectedGraph = new TestUtil.FullyConnectedGraphIndex(0, 6);
        randomlyConnectedGraph = new TestUtil.RandomlyConnectedGraphIndex(10, 4, getRandom());
        emptyGraph = new TestUtil.EmptyGraphIndex(10, getRandom());
        testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    @Test
    public void testSimpleGraphs() throws Exception {
        for (var graph : List.of(fullyConnectedGraph, randomlyConnectedGraph, emptyGraph))
        {
            var outputPath = testDirectory.resolve("test_graph_" + graph.getClass().getSimpleName());
            var ravv = new TestVectorGraph.CircularFloatVectorValues(graph.size(0));
            TestUtil.writeGraph(graph, ravv, outputPath);
            try (var readerSupplier = new SimpleMappedReader.Supplier(outputPath.toAbsolutePath());
                 var onDiskGraph = OnDiskGraphIndex.load(readerSupplier))
            {
                TestUtil.assertGraphEquals(graph, onDiskGraph);
                try (var onDiskView = onDiskGraph.getView()) {
                    validateVectors(onDiskView, ravv);
                }
            }
        }
    }

    @Test
    public void testRenumberingOnDelete() throws IOException {
        testRenumberingOnDelete(false);
        testRenumberingOnDelete(true);
    }

    public void testRenumberingOnDelete(boolean addHierarchy) throws IOException {
        // graph of 3 vectors
        var ravv = new TestVectorGraph.CircularFloatVectorValues(3);
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f, addHierarchy);
        var original = TestUtil.buildSequentially(builder, ravv);

        // delete the first node
        builder.markNodeDeleted(0);
        builder.cleanup();
        builder.setEntryPoint(0, builder.getGraph().getIdUpperBound() - 1); // TODO

        // check
        assertEquals(2, original.size(0));
        var originalView = original.getView();
        // 1 -> 2
        assertEquals(1, getNeighborNodes(originalView, 0, 1).size());
        assertTrue(getNeighborNodes(originalView, 0, 1).contains(2));
        // 2 -> 1
        assertEquals(1, getNeighborNodes(originalView, 0, 2).size());
        assertTrue(getNeighborNodes(originalView, 0, 2).contains(1));

        // create renumbering map
        Map<Integer, Integer> oldToNewMap = OnDiskGraphIndexWriter.sequentialRenumbering(original);
        assertEquals(2, oldToNewMap.size());
        assertEquals(0, (int) oldToNewMap.get(1));
        assertEquals(1, (int) oldToNewMap.get(2));

        // write the graph
        var outputPath = testDirectory.resolve("renumbered_graph");
        OnDiskGraphIndex.write(original, ravv, oldToNewMap, outputPath);
        // check that written graph ordinals match the new ones
        try (var readerSupplier = new SimpleMappedReader.Supplier(outputPath.toAbsolutePath());
             var onDiskGraph = OnDiskGraphIndex.load(readerSupplier);
             var onDiskView = onDiskGraph.getView())
        {
            // entry point renumbering
            assertNotNull(onDiskView.getVector(onDiskGraph.entryNode.node));
            // 0 -> 1
            assertTrue(getNeighborNodes(onDiskView, 0, 0).contains(1));
            // 1 -> 0
            assertTrue(getNeighborNodes(onDiskView, 0, 1).contains(0));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
    @Test
    public void testReorderingRenumbering() throws IOException {
        testReorderingRenumbering(false);
        testReorderingRenumbering(true);
    }

    public void testReorderingRenumbering(boolean addHierarchy) throws IOException {
        // graph of 3 vectors
        var ravv = new TestVectorGraph.CircularFloatVectorValues(3);
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f, addHierarchy);
        var original = TestUtil.buildSequentially(builder, ravv);

        // create renumbering map
        Map<Integer, Integer> oldToNewMap = new HashMap<>();
        oldToNewMap.put(0, 2);
        oldToNewMap.put(1, 1);
        oldToNewMap.put(2, 0);

        // write the graph
        var outputPath = testDirectory.resolve("renumbered_graph");
        OnDiskGraphIndex.write(original, ravv, oldToNewMap, outputPath);
        // check that written graph ordinals match the new ones
        try (var readerSupplier = new SimpleMappedReader.Supplier(outputPath);
             var onDiskGraph = OnDiskGraphIndex.load(readerSupplier);
             var onDiskView = onDiskGraph.getView())
        {
            assertEquals(onDiskView.getVector(0), ravv.getVector(2));
            assertEquals(onDiskView.getVector(1), ravv.getVector(1));
            assertEquals(onDiskView.getVector(2), ravv.getVector(0));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void testReorderingWithHoles() throws IOException {
        testReorderingWithHoles(false);
        testReorderingWithHoles(true);
    }

    public void testReorderingWithHoles(boolean addHierarchy) throws IOException {
        // graph of 3 vectors
        var ravv = new TestVectorGraph.CircularFloatVectorValues(3);
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f, addHierarchy);
        var original = TestUtil.buildSequentially(builder, ravv);

        // create renumbering map
        Map<Integer, Integer> oldToNewMap = new HashMap<>();
        oldToNewMap.put(0, 2);
        oldToNewMap.put(1, 10);
        oldToNewMap.put(2, 0);

        // write the graph
        var outputPath = testDirectory.resolve("renumbered_graph");
        OnDiskGraphIndex.write(original, ravv, oldToNewMap, outputPath);
        // check that written graph ordinals match the new ones
        try (var readerSupplier = new SimpleMappedReader.Supplier(outputPath.toAbsolutePath());
             var onDiskGraph = OnDiskGraphIndex.load(readerSupplier);
             var onDiskView = onDiskGraph.getView())
        {
            assertEquals(11, onDiskGraph.getIdUpperBound());

            Set<Integer> nodesInGraph = new HashSet<>();
            for (NodesIterator it = onDiskGraph.getNodes(0); it.hasNext(); ) {
                nodesInGraph.add(it.next());
            }
            assertEquals(nodesInGraph, Set.of(0, 2, 10));

            assertEquals(onDiskView.getVector(0), ravv.getVector(2));
            assertEquals(onDiskView.getVector(10), ravv.getVector(1));
            assertEquals(onDiskView.getVector(2), ravv.getVector(0));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static void validateVectors(OnDiskGraphIndex.View view, RandomAccessVectorValues ravv) {
        for (int i = 0; i < view.size(); i++) {
            assertEquals("Incorrect vector at " + i, ravv.getVector(i), view.getVector(i));
        }
    }

    private static void validateSeparatedNVQ(OnDiskGraphIndex.View view,
                                             RandomAccessVectorValues ravv,
                                             NVQuantization nvq) throws IOException
    {
        assertEquals("Sizes differ", ravv.size(), view.size());
        // Reusable buffer for decoding
        var quantized = NVQuantization.QuantizedVector.createEmpty(nvq.subvectorSizesAndOffsets,
                                                                   nvq.bitsPerDimension);
        for (int i = 0; i < view.size(); i++) {
            try (var reader = view.featureReaderForNode(i, FeatureId.SEPARATED_NVQ)) {
                NVQuantization.QuantizedVector.loadInto(reader, quantized);
            }
            // sanity check?
        }
    }

    @Test
    public void testSimpleGraphSeparated() throws Exception {
        for (var graph : List.of(fullyConnectedGraph, randomlyConnectedGraph)) {
            var outputPath = testDirectory.resolve("test_graph_separated_" + graph.getClass().getSimpleName());
            var ravv = new TestVectorGraph.CircularFloatVectorValues(graph.size(0));

            // Write graph with SEPARATED_VECTORS
            try (var writer = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
                    .with(new SeparatedVectors(ravv.dimension(), 0L))
                    .build())
            {
                writer.write(Feature.singleStateFactory(
                    FeatureId.SEPARATED_VECTORS,
                    nodeId -> new InlineVectors.State(ravv.getVector(nodeId))
                ));
            }

            // Read and validate
            try (var readerSupplier = new SimpleMappedReader.Supplier(outputPath.toAbsolutePath());
                 var onDiskGraph = OnDiskGraphIndex.load(readerSupplier);
                 var onDiskView = onDiskGraph.getView())
            {
                TestUtil.assertGraphEquals(graph, onDiskGraph);
                validateVectors(onDiskView, ravv);
            }
        }
    }

    @Test
    public void testLargeGraphSeparatedNVQ() throws Exception {
        // Build a large-ish graph
        var nodeCount = 100_000;
        var maxDegree = 32;
        var graph = new TestUtil.RandomlyConnectedGraphIndex(nodeCount, maxDegree, getRandom());
        var outputPath = testDirectory.resolve("large_graph_nvq");

        // Create random vectors
        var dimension = 64;
        var vectors = TestUtil.createRandomVectors(nodeCount, dimension);
        var ravv = new ListRandomAccessVectorValues(vectors, dimension);

        // Compute NVQ and build a SeparatedNVQ feature
        var nvq = NVQuantization.compute(ravv, /* e.g. subquantizers=2 */ 2);
        var separatedNVQ = new SeparatedNVQ(nvq, 0L);

        // Write the graph with SEPARATED_NVQ
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
                .with(separatedNVQ)
                .build())
        {
            // Provide the states for each node
            writer.write(Feature.singleStateFactory(
                FeatureId.SEPARATED_NVQ,
                nodeId -> new NVQ.State(nvq.encode(ravv.getVector(nodeId)))
            ));
        }

        // Read back the graph & check structure, then decode vectors
        try (var readerSupplier = new SimpleMappedReader.Supplier(outputPath.toAbsolutePath());
             var onDiskGraph = OnDiskGraphIndex.load(readerSupplier);
             var onDiskView = onDiskGraph.getView())
        {
            // structure check
            TestUtil.assertGraphEquals(graph, onDiskGraph);

            // decode and compare vectors
            validateSeparatedNVQ(onDiskView, ravv, nvq);
        }
    }

    @Test
    public void testLargeGraph() throws Exception
    {
        var graph = new TestUtil.RandomlyConnectedGraphIndex(1_000_000, 32, getRandom());
        var outputPath = testDirectory.resolve("large_graph");
        var ravv = new TestVectorGraph.CircularFloatVectorValues(graph.size(0));
        TestUtil.writeGraph(graph, ravv, outputPath);

        try (var readerSupplier = new SimpleMappedReader.Supplier(outputPath.toAbsolutePath());
             var onDiskGraph = OnDiskGraphIndex.load(readerSupplier))
        {
            TestUtil.assertGraphEquals(graph, onDiskGraph);
            try (var onDiskView = onDiskGraph.getView())
            {
                validateVectors(onDiskView, ravv);
            }
        }
    }

    @Test
    public void testV0Read() throws IOException {
        // using a random graph from testLargeGraph generated on old version
        var file = new File("resources/version0.odgi");
        try (var readerSupplier = new SimpleMappedReader.Supplier(file.toPath());
             var onDiskGraph = OnDiskGraphIndex.load(readerSupplier);
             var onDiskView = onDiskGraph.getView())
        {
            assertEquals(96, onDiskGraph.ramBytesUsed()); // Current size of graph that hasn't been read from
            assertEquals(32, onDiskGraph.getDegree(0));
            assertEquals(2, onDiskGraph.version);
            assertEquals(100_000, onDiskGraph.size(0));
            assertEquals(2, onDiskGraph.dimension);
            assertEquals(99779, onDiskGraph.entryNode.node);
            assertEquals(EnumSet.of(FeatureId.INLINE_VECTORS), onDiskGraph.features.keySet());
            var actualNeighbors = getNeighborNodes(onDiskView, 0, 12345);
            var expectedNeighbors = Set.of(67461, 9540, 85444, 13638, 89415, 21255, 73737, 46985, 71373, 47436, 94863, 91343, 27215, 59730, 69911, 91867, 89373, 6621, 59106, 98922, 69679, 47728, 60722, 56052, 28854, 38902, 21561, 20665, 41722, 57917, 34495, 5183);
            assertEquals(expectedNeighbors, actualNeighbors);
        }
    }

    @Test
    public void testV0Write() throws IOException {
        var fileIn = new File("resources/version0.odgi");
        var fileOut = File.createTempFile("version0", ".odgi");

        try (var readerSupplier = new SimpleMappedReader.Supplier(fileIn.toPath());
             var graph = OnDiskGraphIndex.load(readerSupplier);
             var view = graph.getView())
        {
             try (var writer = new OnDiskGraphIndexWriter.Builder(graph, fileOut.toPath())
                     .withVersion(2)
                     .with(new InlineVectors(graph.dimension))
                     .build())
             {
                 writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS, nodeId -> new InlineVectors.State(view.getVector(nodeId))));
             }
        }

        // check that the contents match
        var contents1 = Files.readAllBytes(fileIn.toPath());
        var contents2 = Files.readAllBytes(fileOut.toPath());
        assertArrayEquals(contents1, contents2);
    }

    @Test
    public void testMultiLayerFullyConnected() throws Exception {
        // Suppose we have 3 layers of sizes 5, 4, 3
        var graph = new TestUtil.FullyConnectedGraphIndex(1, List.of(5, 4, 3));
        var ravv = new TestVectorGraph.CircularFloatVectorValues(graph.size(0));
        var outputPath = testDirectory.resolve("fully_connected_multilayer");
        TestUtil.writeGraph(graph, ravv, outputPath);

        // read back
        try (var readerSupplier = new SimpleMappedReader.Supplier(outputPath);
             var onDiskGraph = OnDiskGraphIndex.load(readerSupplier))
        {
            // verify the multi-layer structure
            assertEquals(2, onDiskGraph.getMaxLevel());
            assertEquals(5, onDiskGraph.size(0));
            assertEquals(4, onDiskGraph.size(1));
            assertEquals(3, onDiskGraph.size(2));
            TestUtil.assertGraphEquals(graph, onDiskGraph);

            var q = randomVector(ThreadLocalRandom.current(), ravv.dimension());
            var results1 = GraphSearcher.search(q, 10, ravv, VectorSimilarityFunction.EUCLIDEAN, graph, Bits.ALL);
            var results2 = GraphSearcher.search(q, 10, ravv, VectorSimilarityFunction.EUCLIDEAN, onDiskGraph, Bits.ALL);
            assertEquals(results1, results2);
        }
    }

    @Test
    public void testMultiLayerRandomlyConnected() throws Exception {
        // 3 layers
        var layerInfo = List.of(
            new CommonHeader.LayerInfo(100, 8),
            new CommonHeader.LayerInfo(10, 3),
            new CommonHeader.LayerInfo(5, 2)
        );
        var graph = new TestUtil.RandomlyConnectedGraphIndex(layerInfo, getRandom());
        var ravv = new TestVectorGraph.CircularFloatVectorValues(graph.size(0));
        var outputPath = testDirectory.resolve("random_multilayer");

        TestUtil.writeGraph(graph, ravv, outputPath);

        // read back
        try (var readerSupplier = new SimpleMappedReader.Supplier(outputPath);
             var onDiskGraph = OnDiskGraphIndex.load(readerSupplier))
        {
            // confirm multi-layer
            assertEquals(2, onDiskGraph.getMaxLevel());
            assertEquals(100, onDiskGraph.size(0));
            assertEquals(10, onDiskGraph.size(1));
            assertEquals(5, onDiskGraph.size(2));
            TestUtil.assertGraphEquals(graph, onDiskGraph);

            var q = randomVector(ThreadLocalRandom.current(), ravv.dimension());
            var results1 = GraphSearcher.search(q, 10, ravv, VectorSimilarityFunction.EUCLIDEAN, graph, Bits.ALL);
            var results2 = GraphSearcher.search(q, 10, ravv, VectorSimilarityFunction.EUCLIDEAN, onDiskGraph, Bits.ALL);
            assertEquals(results1, results2);
        }
    }

    @Test
    public void testV0WriteIncremental() throws IOException {
        var fileIn = new File("resources/version0.odgi");
        var fileOut = File.createTempFile("version0", ".odgi");

        try (var readerSupplier = new SimpleMappedReader.Supplier(fileIn.toPath());
             var graph = OnDiskGraphIndex.load(readerSupplier);
             var view = graph.getView())
        {
            try (var writer = new OnDiskGraphIndexWriter.Builder(graph, fileOut.toPath())
                    .withVersion(2)
                    .with(new InlineVectors(graph.dimension))
                    .build())
            {
                for (int i = 0; i < view.size(); i++) {
                    var state = Feature.singleState(FeatureId.INLINE_VECTORS, new InlineVectors.State(view.getVector(i)));
                    writer.writeInline(i, state);
                }
                writer.write(Map.of());
            }
        }

        // check that the contents match
        var contents1 = Files.readAllBytes(fileIn.toPath());
        var contents2 = Files.readAllBytes(fileOut.toPath());
        assertArrayEquals(contents1, contents2);
    }

    @Test
    public void testIncrementalWrites() throws IOException {
        // generate 1000 node random graph
        var graph = new TestUtil.RandomlyConnectedGraphIndex(1000, 32, getRandom());
        var vectors = TestUtil.createRandomVectors(1000, 256);
        var ravv = new ListRandomAccessVectorValues(vectors, 256);

        // write out graph all at once
        var bulkPath = testDirectory.resolve("bulk_graph");
        OnDiskGraphIndex.write(graph, ravv, bulkPath);

        // write incrementally
        var incrementalPath = testDirectory.resolve("bulk_graph");
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, incrementalPath)
                .with(new InlineVectors(ravv.dimension()))
                .build())
        {
            // write inline vectors incrementally
            for (int i = 0; i < vectors.size(); i++) {
                var state = Feature.singleState(FeatureId.INLINE_VECTORS, new InlineVectors.State(ravv.getVector(i)));
                writer.writeInline(i, state);
            }

            // write graph structure
            writer.write(Map.of());
        }

        // all-at-once and incremental builds should be identical on disk
        var bulkContents = Files.readAllBytes(bulkPath);
        var incrementalContents = Files.readAllBytes(incrementalPath);
        assertArrayEquals(bulkContents, incrementalContents);

        // write incrementally and add Fused ADC Feature
        var incrementalFadcPath = testDirectory.resolve("incremental_graph");
        var pq = ProductQuantization.compute(ravv, 64, 256, false);
        var pqv = (PQVectors) pq.encodeAll(ravv);
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, incrementalFadcPath)
                .with(new InlineVectors(ravv.dimension()))
                .with(new FusedPQ(graph.getDegree(0), pq))
                .build())
        {
            // write inline vectors incrementally
            for (int i = 0; i < vectors.size(); i++) {
                var state = Feature.singleState(FeatureId.INLINE_VECTORS, new InlineVectors.State(ravv.getVector(i)));
                writer.writeInline(i, state);
            }
            // write graph structure, fused ADC
            writer.write(Feature.singleStateFactory(FeatureId.FUSED_PQ, i -> new FusedPQ.State(graph.getView(), pqv, i)));
            writer.write(Map.of(FeatureId.FUSED_PQ, ordinal -> new FusedPQ.State(graph.getView(), pqv, ordinal)));
        }

        // graph and vectors should be identical
        try (var bulkReaderSupplier = new SimpleMappedReader.Supplier(bulkPath.toAbsolutePath());
             var bulkGraph = OnDiskGraphIndex.load(bulkReaderSupplier);
             var incrementalReaderSupplier = new SimpleMappedReader.Supplier(incrementalFadcPath.toAbsolutePath());
             var incrementalGraph = OnDiskGraphIndex.load(incrementalReaderSupplier);
             var incrementalView = incrementalGraph.getView())
        {
            assertTrue(OnDiskGraphIndex.areHeadersEqual(incrementalGraph, bulkGraph));
            TestUtil.assertGraphEquals(incrementalGraph, bulkGraph); // incremental and bulk graph should have same structure
            validateVectors(incrementalView, ravv); // inline vectors should be the same
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    // -----------------------------------------------------------------------
    // InlineByteVectors (int8) tests
    // -----------------------------------------------------------------------

    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    /** Builds a list of n random int8 vectors of the given dimension. */
    private List<ByteSequence<?>> randomByteVectors(int n, int dim) {
        var list = new ArrayList<ByteSequence<?>>(n);
        for (int i = 0; i < n; i++) {
            byte[] raw = new byte[dim];
            getRandom().nextBytes(raw);
            list.add(vts.createByteSequence(raw));
        }
        return list;
    }

    /**
     * Full round-trip: build with InlineByteVectors, write, reload, verify
     * getByteVector() returns the original bytes for every node.
     */
    @Test
    public void testInlineByteVectorsRoundTrip() throws IOException {
        int n = 50, dim = 8;
        var vectors = randomByteVectors(n, dim);
        var rabvv   = new ListRandomAccessByteVectorValues(vectors, dim);

        // Build a graph
        var builder = new GraphIndexBuilder(rabvv, ByteVectorSimilarityFunction.EUCLIDEAN, 8, 20, 1.2f, 1.2f, false);
        var graph   = builder.build(rabvv);

        // Write with InlineByteVectors feature
        var outputPath = testDirectory.resolve("byte_vectors_roundtrip");
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
                .with(new InlineByteVectors(dim))
                .build())
        {
            writer.write(Feature.singleStateFactory(FeatureId.INLINE_BYTE_VECTORS,
                    nodeId -> new InlineByteVectors.State(rabvv.getVector(nodeId))));
        }

        // Reload and verify every vector
        try (var rs = new SimpleMappedReader.Supplier(outputPath.toAbsolutePath());
             var onDisk = OnDiskGraphIndex.load(rs);
             var view   = onDisk.getView())
        {
            assertTrue("INLINE_BYTE_VECTORS feature missing",
                    onDisk.getFeatureSet().contains(FeatureId.INLINE_BYTE_VECTORS));
            assertEquals(dim, onDisk.dimension);

            for (int i = 0; i < n; i++) {
                var expected = rabvv.getVector(i);
                var actual   = view.getByteVector(i);
                assertEquals("byte vector length mismatch at node " + i, expected.length(), actual.length());
                for (int d = 0; d < dim; d++) {
                    assertEquals("byte mismatch at node " + i + " dim " + d,
                            expected.get(d), actual.get(d));
                }
            }
        }
    }

    /**
     * featureSize() of InlineByteVectors must equal the dimension (1 byte per component),
     * confirming 4× compression vs float32 InlineVectors.
     */
    @Test
    public void testInlineByteVectorsFeatureSize() {
        for (int dim : new int[]{1, 8, 128, 256}) {
            var ibv = new InlineByteVectors(dim);
            assertEquals("featureSize should equal dimension", dim, ibv.featureSize());
            // float32 InlineVectors uses 4 * dim bytes
            assertEquals("byte feature 4x smaller than float32",
                    new InlineVectors(dim).featureSize(), 4 * ibv.featureSize());
        }
    }

    /**
     * getByteVector throws UnsupportedOperationException when the graph has no INLINE_BYTE_VECTORS feature.
     */
    @Test
    public void testGetByteVectorThrowsWithoutFeature() throws IOException {
        // Build and write a float-vector graph (no INLINE_BYTE_VECTORS)
        var graph = new TestUtil.RandomlyConnectedGraphIndex(10, 4, getRandom());
        var ravv  = new TestVectorGraph.CircularFloatVectorValues(10);
        var outputPath = testDirectory.resolve("float_graph_no_bytes");
        TestUtil.writeGraph(graph, ravv, outputPath);

        try (var rs = new SimpleMappedReader.Supplier(outputPath.toAbsolutePath());
             var onDisk = OnDiskGraphIndex.load(rs);
             var view   = onDisk.getView())
        {
            assertFalse(onDisk.getFeatureSet().contains(FeatureId.INLINE_BYTE_VECTORS));
            assertThrows(UnsupportedOperationException.class, () -> view.getByteVector(0));
        }
    }

    /**
     * byteVectorRerankerFor throws UnsupportedOperationException when the graph has no INLINE_BYTE_VECTORS feature.
     */
    @Test
    public void testByteVectorRerankerThrowsWithoutFeature() throws IOException {
        var graph = new TestUtil.RandomlyConnectedGraphIndex(10, 4, getRandom());
        var ravv  = new TestVectorGraph.CircularFloatVectorValues(10);
        var outputPath = testDirectory.resolve("float_graph_reranker");
        TestUtil.writeGraph(graph, ravv, outputPath);

        try (var rs = new SimpleMappedReader.Supplier(outputPath.toAbsolutePath());
             var onDisk = OnDiskGraphIndex.load(rs);
             var view   = onDisk.getView())
        {
            var qb = vts.createByteSequence(2);
            assertThrows(UnsupportedOperationException.class,
                    () -> view.byteVectorRerankerFor(qb, ByteVectorSimilarityFunction.EUCLIDEAN));
        }
    }

    /**
     * byteVectorRerankerFor returns scores consistent with a direct bvsf.compare() call.
     */
    @Test
    public void testByteVectorRerankerScoresMatchDirectCompare() throws IOException {
        int n = 30, dim = 4;
        var vectors = randomByteVectors(n, dim);
        var rabvv   = new ListRandomAccessByteVectorValues(vectors, dim);
        var bvsf    = ByteVectorSimilarityFunction.EUCLIDEAN;

        var builder = new GraphIndexBuilder(rabvv, bvsf, 8, 20, 1.2f, 1.2f, false);
        var graph   = builder.build(rabvv);

        var outputPath = testDirectory.resolve("byte_reranker_test");
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
                .with(new InlineByteVectors(dim))
                .build())
        {
            writer.write(Feature.singleStateFactory(FeatureId.INLINE_BYTE_VECTORS,
                    nodeId -> new InlineByteVectors.State(rabvv.getVector(nodeId))));
        }

        try (var rs = new SimpleMappedReader.Supplier(outputPath.toAbsolutePath());
             var onDisk = OnDiskGraphIndex.load(rs);
             var view   = onDisk.getView())
        {
            byte[] rawQ = new byte[dim];
            getRandom().nextBytes(rawQ);
            var query    = vts.createByteSequence(rawQ);
            var reranker = view.byteVectorRerankerFor(query, bvsf);

            for (int i = 0; i < n; i++) {
                float expected = bvsf.compare(query, rabvv.getVector(i));
                float actual   = reranker.similarityTo(i);
                assertEquals("reranker score mismatch at node " + i, expected, actual, 1e-5f);
            }
        }
    }

    /**
     * All three byte-vector similarity functions produce results in [0, 1] for all nodes
     * after a full int8 end-to-end write/search cycle.
     */
    @Test
    public void testByteVectorEndToEndAllSimilarityFunctions() throws IOException {
        int n = 40, dim = 8;

        for (var bvsf : ByteVectorSimilarityFunction.values()) {
            var vectors = randomByteVectors(n, dim);
            var rabvv   = new ListRandomAccessByteVectorValues(vectors, dim);

            var builder = new GraphIndexBuilder(rabvv, bvsf, 8, 20, 1.2f, 1.2f, false);
            var graph   = builder.build(rabvv);

            var outputPath = testDirectory.resolve("byte_e2e_" + bvsf.name());
            try (var writer = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
                    .with(new InlineByteVectors(dim))
                    .build())
            {
                writer.write(Feature.singleStateFactory(FeatureId.INLINE_BYTE_VECTORS,
                        nodeId -> new InlineByteVectors.State(rabvv.getVector(nodeId))));
            }

            try (var rs = new SimpleMappedReader.Supplier(outputPath.toAbsolutePath());
                 var onDisk = OnDiskGraphIndex.load(rs);
                 var view   = onDisk.getView())
            {
                byte[] rawQ = new byte[dim];
                getRandom().nextBytes(rawQ);
                var query    = vts.createByteSequence(rawQ);
                var reranker = view.byteVectorRerankerFor(query, bvsf);
                var ssp      = new DefaultSearchScoreProvider(reranker);
                var result   = new GraphSearcher(onDisk).search(ssp, 5, Bits.ALL);

                for (var ns : result.getNodes()) {
                    float score = ns.score;
                    assertTrue(bvsf + " score out of [0,1] for node " + ns.node + ": " + score,
                            score >= 0f && score <= 1.0f);
                }
            }
        }
    }
}
