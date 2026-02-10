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

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestRandomAccessOnDiskGraphIndexWriter extends LuceneTestCase {
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
    public void testSingleLayerTwoPhase() throws IOException {
        testGraphWrite(false, true);
    }
    @Test
    public void testSingleLayerOnePhase() throws IOException {
        testGraphWrite(false, false);
    }
    @Test
    public void testMultiLayerTwoPhase() throws IOException {
        testGraphWrite(true, true);
    }
    @Test
    public void testMultiLayerSinglePhase() throws IOException {
        testGraphWrite(true, false);
    }

    private void testGraphWrite(boolean addHierarchy, boolean twoPhase) throws IOException {
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
        Path indexPath = testDirectory.resolve("graph_index");

        // Create feature state suppliers
        var suppliers = Feature.singleStateFactory(
                FeatureId.INLINE_VECTORS,
                nodeId -> new InlineVectors.State(ravv.getVector(nodeId))
        );

        // TWO-PHASE APPROACH: writeFeaturesInline then write
        try (var writer = new OnDiskParallelGraphIndexWriter.Builder(graph, indexPath)
                .withParallelDirectBuffers(true)
                .with(new InlineVectors(ravv.dimension()))
                .build()) {
            if (twoPhase) {
                // Phase 1: Write inline features for all nodes
                for (int ordinal = 0; ordinal < graph.size(0); ordinal++) {
                    Map<FeatureId, Feature.State> stateMap = Map.of(
                            FeatureId.INLINE_VECTORS,
                            new InlineVectors.State(ravv.getVector(ordinal))
                    );
                    writer.writeFeaturesInline(ordinal, stateMap);
                }
                // Phase 2: Write the complete graph (including edges and metadata)
                writer.write(Map.of());
            } else {
                // Write everything in one call
                writer.write(suppliers);
            }
        }
        // Verify both graphs load correctly and are equivalent
        try (var readerSupplier = new SimpleMappedReader.Supplier(indexPath)) {
             var onDiskGraph = OnDiskGraphIndex.load(readerSupplier);
             TestUtil.assertGraphEquals(graph, onDiskGraph);
             TestOnDiskGraphIndex.validateVectors(onDiskGraph.getView(), ravv);
        }
    }

}
