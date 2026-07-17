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
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.RandomAccessOnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.management.GraphIndexBuilderConfig;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

/**
 * Verifies that GraphIndexBuilderConfig (JMX) values are correctly routed through
 * the non-deprecated constructor path, and that deprecated constructors continue to
 * honour their caller-supplied values without reading JMX.
 *
 * Covers:
 *   - addHierarchy: deprecated (old path) and JMX (new path), both true and false
 *   - refineFinalGraph: deprecated (old path) and JMX (new path), both true and false
 *   - parallelBuild: unified RandomAccessOnDiskGraphIndexWriter.Builder, both serial and parallel
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestGraphIndexBuilderConfig extends LuceneTestCase {

    private static final int DIMENSION = 16;
    private static final int SIZE = 200;
    private static final int M = 16;
    private static final int BEAM_WIDTH = 100;
    private static final float NEIGHBOR_OVERFLOW = 1.2f;
    private static final float ALPHA = 1.2f;

    private Path testDirectory;
    private boolean savedAddHierarchy;
    private boolean savedRefineFinalGraph;
    private boolean savedParallelBuild;

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory(getClass().getSimpleName());
        var config = GraphIndexBuilderConfig.getInstance();
        savedAddHierarchy = config.isAddHierarchy();
        savedRefineFinalGraph = config.isRefineFinalGraph();
        savedParallelBuild = config.isParallelBuild();
    }

    @After
    public void tearDown() throws Exception {
        TestUtil.deleteQuietly(testDirectory);
        var config = GraphIndexBuilderConfig.getInstance();
        config.setAddHierarchy(savedAddHierarchy);
        config.setRefineFinalGraph(savedRefineFinalGraph);
        config.setParallelBuild(savedParallelBuild);
    }

    // ── addHierarchy ──────────────────────────────────────────────────────────

    @Test
    @SuppressWarnings("deprecation")
    public void testAddHierarchy_deprecated_false() {
        var ravv = buildVectors();
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE,
                M, BEAM_WIDTH, NEIGHBOR_OVERFLOW, ALPHA, false);
        TestUtil.buildSequentially(builder, ravv);
        assertEquals(0, ((OnHeapGraphIndex) builder.graph).getMaxLevel());
    }

    @Test
    @SuppressWarnings("deprecation")
    public void testAddHierarchy_deprecated_true() {
        var ravv = buildVectors();
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE,
                M, BEAM_WIDTH, NEIGHBOR_OVERFLOW, ALPHA, true);
        TestUtil.buildSequentially(builder, ravv);
        assertTrue(((OnHeapGraphIndex) builder.graph).getMaxLevel() > 0);
    }

    @Test
    public void testAddHierarchy_jmx_false() {
        GraphIndexBuilderConfig.getInstance().setAddHierarchy(false);
        var ravv = buildVectors();
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE,
                M, BEAM_WIDTH, NEIGHBOR_OVERFLOW, ALPHA);
        TestUtil.buildSequentially(builder, ravv);
        assertEquals(0, ((OnHeapGraphIndex) builder.graph).getMaxLevel());
    }

    @Test
    public void testAddHierarchy_jmx_true() {
        GraphIndexBuilderConfig.getInstance().setAddHierarchy(true);
        var ravv = buildVectors();
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE,
                M, BEAM_WIDTH, NEIGHBOR_OVERFLOW, ALPHA);
        TestUtil.buildSequentially(builder, ravv);
        assertTrue(((OnHeapGraphIndex) builder.graph).getMaxLevel() > 0);
    }

    // ── refineFinalGraph ──────────────────────────────────────────────────────

    @Test
    @SuppressWarnings("deprecation")
    public void testRefineFinalGraph_deprecated_false() {
        var ravv = buildVectors();
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE,
                M, BEAM_WIDTH, NEIGHBOR_OVERFLOW, ALPHA, true, false);
        assertFalse(builder.isRefineFinalGraph());
    }

    @Test
    @SuppressWarnings("deprecation")
    public void testRefineFinalGraph_deprecated_true() {
        var ravv = buildVectors();
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE,
                M, BEAM_WIDTH, NEIGHBOR_OVERFLOW, ALPHA, true, true);
        assertTrue(builder.isRefineFinalGraph());
    }

    @Test
    public void testRefineFinalGraph_jmx_false() {
        GraphIndexBuilderConfig.getInstance().setRefineFinalGraph(false);
        var ravv = buildVectors();
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE,
                M, BEAM_WIDTH, NEIGHBOR_OVERFLOW, ALPHA);
        assertFalse(builder.isRefineFinalGraph());
    }

    @Test
    public void testRefineFinalGraph_jmx_true() {
        GraphIndexBuilderConfig.getInstance().setRefineFinalGraph(true);
        var ravv = buildVectors();
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE,
                M, BEAM_WIDTH, NEIGHBOR_OVERFLOW, ALPHA);
        assertTrue(builder.isRefineFinalGraph());
    }

    // ── parallel vs. sequential build ─────────────────────────────────────────

    @Test
    public void testUnifiedBuilder_sequential() throws IOException {
        GraphIndexBuilderConfig.getInstance().setParallelBuild(false);
        var ravv = buildVectors();
        var graph = TestUtil.buildSequentially(
                new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, M, BEAM_WIDTH, NEIGHBOR_OVERFLOW, ALPHA),
                ravv);
        writeAndVerify(graph, ravv, testDirectory.resolve("sequential.index"));
    }

    @Test
    public void testUnifiedBuilder_parallel() throws IOException {
        GraphIndexBuilderConfig.getInstance().setParallelBuild(true);
        var ravv = buildVectors();
        var graph = TestUtil.buildSequentially(
                new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, M, BEAM_WIDTH, NEIGHBOR_OVERFLOW, ALPHA),
                ravv);
        writeAndVerify(graph, ravv, testDirectory.resolve("parallel.index"));
    }

    @Test
    public void testUnifiedBuilder_parallelAndSequentialProduceIdenticalGraph() throws IOException {
        var ravv = buildVectors();
        var graph = TestUtil.buildSequentially(
                new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, M, BEAM_WIDTH, NEIGHBOR_OVERFLOW, ALPHA),
                ravv);

        var seqPath = testDirectory.resolve("seq.index");
        var parPath = testDirectory.resolve("par.index");

        GraphIndexBuilderConfig.getInstance().setParallelBuild(false);
        writeGraph(graph, ravv, seqPath);

        GraphIndexBuilderConfig.getInstance().setParallelBuild(true);
        writeGraph(graph, ravv, parPath);

        try (var seqSupplier = new SimpleMappedReader.Supplier(seqPath);
             var parSupplier = new SimpleMappedReader.Supplier(parPath)) {
            var seqLoaded = OnDiskGraphIndex.load(seqSupplier);
            var parLoaded = OnDiskGraphIndex.load(parSupplier);
            TestUtil.assertGraphEquals(seqLoaded, parLoaded);
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private ListRandomAccessVectorValues buildVectors() {
        return new ListRandomAccessVectorValues(
                new ArrayList<>(TestUtil.createRandomVectors(SIZE, DIMENSION)),
                DIMENSION
        );
    }

    private void writeGraph(ImmutableGraphIndex graph, ListRandomAccessVectorValues ravv, Path path) throws IOException {
        var suppliers = Feature.singleStateFactory(
                FeatureId.INLINE_VECTORS,
                nodeId -> new InlineVectors.State(ravv.getVector(nodeId))
        );
        try (var writer = new RandomAccessOnDiskGraphIndexWriter.Builder(graph, path)
                .with(new InlineVectors(ravv.dimension()))
                .build()) {
            writer.write(suppliers);
        }
    }

    private void writeAndVerify(ImmutableGraphIndex graph, ListRandomAccessVectorValues ravv, Path path) throws IOException {
        writeGraph(graph, ravv, path);
        try (var readerSupplier = new SimpleMappedReader.Supplier(path)) {
            var onDiskGraph = OnDiskGraphIndex.load(readerSupplier);
            TestUtil.assertGraphEquals(graph, onDiskGraph);
        }
    }
}
