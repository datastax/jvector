package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.disk.SimpleWriter;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.MockVectorValues;
import io.github.jbellis.jvector.graph.TestUtil;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class OnDiskSequentialGraphIndexWriterTest {

    @TempDir
    Path tempDir;

    private float[][] createRandomFloatVectors(int size, int dimension, Random random) {
        float[][] vectors = new float[size][dimension];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = random.nextFloat();
            }
        }
        return vectors;
    }

    @Test
    public void testWriteAndLoadFromFooter() throws IOException {
        // Setup test parameters
        int dimension = 16;
        int size = 50;
        int maxConnections = 8;
        Random random = new Random(42);
        
        // Create random vectors and build a graph
        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(size, dimension, random));
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, maxConnections, 1.0f, 1.0f, true);
        GraphIndex graph = TestUtil.buildSequentially(builder, ravv);
        
        // Create a sequential writer and write the graph
        Path indexPath = tempDir.resolve("sequential_graph.data");
        try (var out = new SimpleWriter(indexPath)) {
            var ordinalMapper = new OrdinalMapper.MapMapper(OnDiskSequentialGraphIndexWriter.sequentialRenumbering(graph));
            var writer = new OnDiskSequentialGraphIndexWriter.Builder(graph, out)
                    .with(new InlineVectors(dimension))
                    .withMapper(ordinalMapper)
                    .build();
            
            // Create feature state suppliers
            var suppliers = Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                    nodeId -> new InlineVectors.State(ravv.getVector(nodeId)));
            
            // Write the graph
            writer.write(suppliers);
            
            // Write the footer
            long headerOffset = out.bytesWrittenSinceStart();
            writer.writeFooter(headerOffset);
        }
        
        // Load the graph using loadFromFooter
        try (var readerSupplier = new SimpleMappedReader.Supplier(indexPath)) {
            var loadedGraph = OnDiskGraphIndex.loadFromFooter(readerSupplier);
            
            // Validate the loaded graph
            assertNotNull(loadedGraph);
            assertEquals(size, loadedGraph.size());
            assertEquals(dimension, loadedGraph.getDimension());
            
            // Verify graph structure by checking a few nodes
            try (var view = loadedGraph.getView()) {
                // Check entry node exists
                assertNotNull(view.entryNode());
                
                // Check a few random nodes have neighbors
                for (int i = 0; i < 5; i++) {
                    int nodeId = random.nextInt(size);
                    var neighbors = view.getNeighborsIterator(0, nodeId);
                    // At least the entry node should have neighbors
                    if (nodeId == view.entryNode().node) {
                        assert(neighbors.size() > 0);
                    }
                }
            }
        }
    }
    
    @Test
    public void testMultiLayerGraphWriteAndLoad() throws IOException {
        // Setup test parameters for multi-layer graph
        int dimension = 16;
        int size = 100;
        int maxConnections = 8;
        Random random = new Random(42);
        
        // Create random vectors and build a multi-layer graph
        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(size, dimension, random));
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, maxConnections, 1.0f, 1.0f, true);
        builder.setNumLayers(2); // Set multiple layers
        GraphIndex graph = TestUtil.buildSequentially(builder, ravv);
        
        // Create a sequential writer and write the graph
        Path indexPath = tempDir.resolve("multilayer_graph.data");
        try (var out = new SimpleWriter(indexPath)) {
            var ordinalMapper = new OrdinalMapper.MapMapper(OnDiskSequentialGraphIndexWriter.sequentialRenumbering(graph));
            var writer = new OnDiskSequentialGraphIndexWriter.Builder(graph, out)
                    .with(new InlineVectors(dimension))
                    .withMapper(ordinalMapper)
                    .build();
            
            // Create feature state suppliers
            var suppliers = Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                    nodeId -> new InlineVectors.State(ravv.getVector(nodeId)));
            
            // Write the graph
            writer.write(suppliers);
            
            // Write the footer
            long headerOffset = out.bytesWrittenSinceStart();
            writer.writeFooter(headerOffset);
        }
        
        // Load the graph using loadFromFooter
        try (var readerSupplier = new SimpleMappedReader.Supplier(indexPath)) {
            var loadedGraph = OnDiskGraphIndex.loadFromFooter(readerSupplier);
            
            // Validate the loaded graph
            assertNotNull(loadedGraph);
            assertEquals(size, loadedGraph.size());
            assertEquals(dimension, loadedGraph.getDimension());
            assertEquals(2, loadedGraph.getMaxLevel()); // Verify multiple layers
            
            // Verify graph structure by checking nodes in different layers
            try (var view = loadedGraph.getView()) {
                // Check entry node exists in the highest layer
                assertEquals(2, view.entryNode().level);
                
                // Verify we have nodes in layer 1
                assert(graph.size(1) > 0);
                
                // Check that entry node has neighbors in its layer
                var entryNeighbors = view.getNeighborsIterator(view.entryNode().level, view.entryNode().node);
                assert(entryNeighbors.size() > 0);
            }
        }
    }
}