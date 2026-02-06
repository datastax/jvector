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

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
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
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestOnDiskParallelGraphIndexWriter extends LuceneTestCase {
    private Path testDirectory;

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    /**
     * Tests that OnDiskParallelGraphIndexWriter produces identical output whether:
     * 1. Using two-phase approach: writeFeaturesInline() then write()
     * 2. Using single-phase approach: just write()
     */
    @Test
    public void testTwoPhaseVsSinglePhaseWriting() throws IOException {
        // Test with single layer graph
        testTwoPhaseVsSinglePhaseWriting(false);
        // Test with multi-layer graph
        testTwoPhaseVsSinglePhaseWriting(true);
    }

    private void testTwoPhaseVsSinglePhaseWriting(boolean addHierarchy) throws IOException {
        // Setup test parameters
        int dimension = 16;
        int size = 100;
        int maxConnections = 8;
        int beamWidth = 100;
        float alpha = 1.2f;
        float neighborOverflow = 1.2f;

        // Create random vectors and build a graph
        var ravv = new ListRandomAccessVectorValues(
                new ArrayList<>(TestUtil.createRandomVectors(size, dimension)), 
                dimension
        );
        var builder = new GraphIndexBuilder(
                ravv, 
                VectorSimilarityFunction.COSINE, 
                maxConnections, 
                beamWidth, 
                neighborOverflow, 
                alpha, 
                addHierarchy
        );
        ImmutableGraphIndex graph = TestUtil.buildSequentially(builder, ravv);

        // Path for two-phase write
        Path twoPhaseIndexPath = testDirectory.resolve("graph_two_phase_hierarchy_" + addHierarchy);
        // Path for single-phase write
        Path singlePhaseIndexPath = testDirectory.resolve("graph_single_phase_hierarchy_" + addHierarchy);

        // Create feature state suppliers
        var suppliers = Feature.singleStateFactory(
                FeatureId.INLINE_VECTORS,
                nodeId -> new InlineVectors.State(ravv.getVector(nodeId))
        );

        // TWO-PHASE APPROACH: writeFeaturesInline then write
        try (var writer = new OnDiskParallelGraphIndexWriter.Builder(graph, twoPhaseIndexPath)
                .with(new InlineVectors(ravv.dimension()))
                .build()) {
            
            // Phase 1: Write inline features for all nodes
            for (int ordinal = 0; ordinal < graph.size(0); ordinal++) {
                Map<FeatureId, Feature.State> stateMap = Map.of(
                        FeatureId.INLINE_VECTORS, 
                        new InlineVectors.State(ravv.getVector(ordinal))
                );
                writer.writeFeaturesInline(ordinal, stateMap);
            }
            
            // Phase 2: Write the complete graph (including edges and metadata)
            writer.write(suppliers);
        }

        // SINGLE-PHASE APPROACH: just write
        try (var writer = new OnDiskParallelGraphIndexWriter.Builder(graph, singlePhaseIndexPath)
                .with(new InlineVectors(ravv.dimension()))
                .build()) {
            
            // Write everything in one call
            writer.write(suppliers);
        }

        // Verify both files are identical
        assertFilesIdentical(twoPhaseIndexPath, singlePhaseIndexPath);

        // Verify both graphs load correctly and are equivalent
        try (var readerSupplier1 = new SimpleMappedReader.Supplier(twoPhaseIndexPath);
             var readerSupplier2 = new SimpleMappedReader.Supplier(singlePhaseIndexPath);
             var onDiskGraph1 = OnDiskGraphIndex.load(readerSupplier1);
             var onDiskGraph2 = OnDiskGraphIndex.load(readerSupplier2)) {
            
            // Both should match the original graph
            TestUtil.assertGraphEquals(graph, onDiskGraph1);
            TestUtil.assertGraphEquals(graph, onDiskGraph2);
            
            // Both should be identical to each other
            TestUtil.assertGraphEquals(onDiskGraph1, onDiskGraph2);
            
            // Verify vectors are correct in both
            try (var view1 = onDiskGraph1.getView();
                 var view2 = onDiskGraph2.getView()) {
                TestOnDiskGraphIndex.validateVectors(view1, ravv);
                TestOnDiskGraphIndex.validateVectors(view2, ravv);
            }
        }
    }

    /**
     * Tests that parallel writing with different worker thread counts produces identical output.
     */
    @Test
    public void testParallelWritingWithDifferentThreadCounts() throws IOException {
        // Setup test parameters
        int dimension = 16;
        int size = 100;
        int maxConnections = 8;
        int beamWidth = 100;
        float alpha = 1.2f;
        float neighborOverflow = 1.2f;
        boolean addHierarchy = false;

        // Create random vectors and build a graph
        var ravv = new ListRandomAccessVectorValues(
                new ArrayList<>(TestUtil.createRandomVectors(size, dimension)), 
                dimension
        );
        var builder = new GraphIndexBuilder(
                ravv, 
                VectorSimilarityFunction.COSINE, 
                maxConnections, 
                beamWidth, 
                neighborOverflow, 
                alpha, 
                addHierarchy
        );
        ImmutableGraphIndex graph = TestUtil.buildSequentially(builder, ravv);

        // Create feature state suppliers
        var suppliers = Feature.singleStateFactory(
                FeatureId.INLINE_VECTORS,
                nodeId -> new InlineVectors.State(ravv.getVector(nodeId))
        );

        // Write with different thread counts
        int[] threadCounts = {0, 1, 2, 4}; // 0 means use available processors
        Path[] paths = new Path[threadCounts.length];

        for (int i = 0; i < threadCounts.length; i++) {
            paths[i] = testDirectory.resolve("graph_threads_" + threadCounts[i]);
            
            OnDiskParallelGraphIndexWriter.Builder writerBuilder = new OnDiskParallelGraphIndexWriter.Builder(graph, paths[i]);
            writerBuilder.withParallelWorkerThreads(threadCounts[i]);
            writerBuilder.with(new InlineVectors(ravv.dimension()));
            
            try (OnDiskParallelGraphIndexWriter writer = writerBuilder.build()) {
                writer.write(suppliers);
            }
        }

        // Verify all files are identical
        for (int i = 1; i < paths.length; i++) {
            assertFilesIdentical(paths[0], paths[i]);
        }

        // Verify all graphs load correctly
        for (Path path : paths) {
            try (var readerSupplier = new SimpleMappedReader.Supplier(path);
                 var onDiskGraph = OnDiskGraphIndex.load(readerSupplier)) {
                TestUtil.assertGraphEquals(graph, onDiskGraph);
                try (var view = onDiskGraph.getView()) {
                    TestOnDiskGraphIndex.validateVectors(view, ravv);
                }
            }
        }
    }

    /**
     * Tests two-phase writing with parallel configuration.
     */
    @Test
    public void testTwoPhaseParallelWriting() throws IOException {
        // Setup test parameters
        int dimension = 16;
        int size = 100;
        int maxConnections = 8;
        int beamWidth = 100;
        float alpha = 1.2f;
        float neighborOverflow = 1.2f;
        boolean addHierarchy = true;

        // Create random vectors and build a graph
        var ravv = new ListRandomAccessVectorValues(
                new ArrayList<>(TestUtil.createRandomVectors(size, dimension)), 
                dimension
        );
        var builder = new GraphIndexBuilder(
                ravv, 
                VectorSimilarityFunction.COSINE, 
                maxConnections, 
                beamWidth, 
                neighborOverflow, 
                alpha, 
                addHierarchy
        );
        ImmutableGraphIndex graph = TestUtil.buildSequentially(builder, ravv);

        Path indexPath = testDirectory.resolve("graph_two_phase_parallel");

        // Create feature state suppliers
        var suppliers = Feature.singleStateFactory(
                FeatureId.INLINE_VECTORS,
                nodeId -> new InlineVectors.State(ravv.getVector(nodeId))
        );

        // Two-phase write with parallel configuration
        OnDiskParallelGraphIndexWriter.Builder writerBuilder = new OnDiskParallelGraphIndexWriter.Builder(graph, indexPath);
        writerBuilder.withParallelWorkerThreads(4);
        writerBuilder.withParallelDirectBuffers(true);
        writerBuilder.with(new InlineVectors(ravv.dimension()));
        
        try (OnDiskParallelGraphIndexWriter writer = writerBuilder.build()) {
            // Phase 1: Write inline features
            for (int ordinal = 0; ordinal < graph.size(0); ordinal++) {
                Map<FeatureId, Feature.State> stateMap = Map.of(
                        FeatureId.INLINE_VECTORS,
                        new InlineVectors.State(ravv.getVector(ordinal))
                );
                writer.writeFeaturesInline(ordinal, stateMap);
            }
            
            // Phase 2: Write complete graph with parallel workers
            writer.write(Map.of());
        }

        // Verify the graph loads correctly
        try (var readerSupplier = new SimpleMappedReader.Supplier(indexPath);
             var onDiskGraph = OnDiskGraphIndex.load(readerSupplier)) {
            TestUtil.assertGraphEquals(graph, onDiskGraph);
            try (var view = onDiskGraph.getView()) {
                TestOnDiskGraphIndex.validateVectors(view, ravv);
            }
        }
    }

    /**
     * Helper method to assert that two files are byte-for-byte identical.
     */
    private void assertFilesIdentical(Path path1, Path path2) throws IOException {
        byte[] bytes1 = Files.readAllBytes(path1);
        byte[] bytes2 = Files.readAllBytes(path2);
        
        assertEquals("File sizes differ", bytes1.length, bytes2.length);
        assertArrayEquals(
                "Files are not identical: " + path1 + " vs " + path2,
                bytes1,
                bytes2
        );
    }
}

