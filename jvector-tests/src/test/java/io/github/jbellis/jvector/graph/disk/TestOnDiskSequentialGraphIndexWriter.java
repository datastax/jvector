package io.github.jbellis.jvector.graph.disk;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.disk.SimpleWriter;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestOnDiskSequentialGraphIndexWriter extends LuceneTestCase {
    private Path testDirectory;

    @Before
    public void setup() throws IOException {
        // Create a temporary directory for testing
        testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    @Test
    public void testWriteAndLoadFromFooter() throws IOException {
        // Setup test parameters
        int dimension = 16;
        int size = 50;
        int maxConnections = 8;
        int beamWidth = 100;
        float alpha = 1.2f;
        float neighborOverflow = 1.2f;
        boolean addHierarchy = false;

        buildAndCompareGraphs(size, dimension, maxConnections, beamWidth, alpha, neighborOverflow, addHierarchy);
    }
    
    @Test
    public void testMultiLayerGraphWriteAndLoad() throws IOException {
        // Setup test parameters
        int dimension = 16;
        int size = 50;
        int maxConnections = 8;
        int beamWidth = 100;
        float alpha = 1.2f;
        float neighborOverflow = 1.2f;
        boolean addHierarchy = true;

        buildAndCompareGraphs(size, dimension, maxConnections, beamWidth, alpha, neighborOverflow, addHierarchy);
    }

    void buildAndCompareGraphs(int size, int dimension, int maxConnections, int beamWidth, float alpha, float neighborOverflow, boolean addHierarchy) throws IOException {
        // Create random vectors and build a graph
        var ravv = new ListRandomAccessVectorValues(new ArrayList<>(TestUtil.createRandomVectors(size, dimension)), dimension);
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, maxConnections, beamWidth, neighborOverflow, alpha, addHierarchy);
        GraphIndex graph = TestUtil.buildSequentially(builder, ravv);

        // Create a sequential writer and write the graph
        Path indexPath = testDirectory.resolve("sequential_graph.data");
        try (var out = new SimpleWriter(indexPath)) {
            // Create an ordinal mapper
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
        }

        // Load the graph using loadFromFooter
        try (var readerSupplier = new SimpleMappedReader.Supplier(indexPath)) {
            var onDiskGraph = OnDiskGraphIndex.load(readerSupplier);

            // Validate the loaded graph
            TestUtil.assertGraphEquals(graph, onDiskGraph);
            try (var onDiskView = onDiskGraph.getView()) {
                TestOnDiskGraphIndex.validateVectors(onDiskView, ravv);
            }
        }
    }
}