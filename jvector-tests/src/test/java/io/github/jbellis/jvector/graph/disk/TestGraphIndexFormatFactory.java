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
import io.github.jbellis.jvector.graph.TestVectorGraph;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Tests for {@link GraphIndexFormatFactory}: version dispatch, rejection of unsupported
 * versions, magic-number-based version detection, and the per-version characteristics
 * (feature set, multi-layer support, footer usage) that the writer/reader dispatch relies on.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestGraphIndexFormatFactory extends RandomizedTest {

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
    public void testForVersionReturnsMatchingFormat() {
        for (int version = 2; version <= OnDiskGraphIndex.CURRENT_VERSION; version++) {
            assertEquals(version, GraphIndexFormatFactory.forVersion(version).getVersion());
        }
    }

    @Test
    public void testForVersionRejectsUnsupportedVersions() {
        for (int version : new int[]{Integer.MIN_VALUE, -1, 0, 1, OnDiskGraphIndex.CURRENT_VERSION + 1, 100}) {
            assertThrows(GraphIndexFormatFactory.UnsupportedVersionException.class,
                    () -> GraphIndexFormatFactory.forVersion(version));
        }
    }

    @Test
    public void testGetCurrentVersion() {
        assertEquals(OnDiskGraphIndex.CURRENT_VERSION, GraphIndexFormatFactory.getCurrentVersion());
    }

    /**
     * Pins down the per-version characteristics that the rest of the read/write path dispatches
     * on, so that a change to one version's format can't silently change another's behavior.
     */
    @Test
    public void testFormatCharacteristicsPerVersion() {
        var v2 = GraphIndexFormatFactory.forVersion(2);
        assertFalse(v2.supportsMultiLayer());
        assertFalse(v2.usesFooter());
        assertEquals(EnumSet.of(FeatureId.INLINE_VECTORS), v2.getSupportedFeatures());

        // versions 3-5 support every feature except FUSED_PQ; multi-layer arrives at v4,
        // footer-based metadata arrives at v5.
        var nonFused = EnumSet.complementOf(EnumSet.of(FeatureId.FUSED_PQ));
        for (int version : new int[]{3, 4, 5}) {
            var format = GraphIndexFormatFactory.forVersion(version);
            assertEquals("version " + version + " supported features", nonFused, format.getSupportedFeatures());
            assertFalse("version " + version + " should not support FUSED_PQ", format.supportsFeature(FeatureId.FUSED_PQ));
            assertEquals("version " + version + " multi-layer support", version >= 4, format.supportsMultiLayer());
            assertEquals("version " + version + " footer usage", version >= 5, format.usesFooter());
        }

        var v6 = GraphIndexFormatFactory.forVersion(6);
        assertTrue(v6.supportsMultiLayer());
        assertTrue(v6.usesFooter());
        assertTrue(v6.supportsFeature(FeatureId.FUSED_PQ));
        assertEquals(EnumSet.allOf(FeatureId.class), v6.getSupportedFeatures());
    }

    @Test
    public void testDetectVersionNoMagic() throws Exception {
        // version 2 predates the magic number, so detection falls back to reading the raw
        // base-layer size as the first int; detectVersion must still identify it as v2 and
        // must leave the reader positioned exactly where it started.
        var path = writeSingleNodeGraph(2);
        try (var readerSupplier = new SimpleMappedReader.Supplier(path);
             var reader = readerSupplier.get())
        {
            long startPosition = reader.getPosition();
            var format = GraphIndexFormatFactory.detectVersion(reader);
            assertEquals(2, format.getVersion());
            assertEquals(startPosition, reader.getPosition());
        }
    }

    @Test
    public void testDetectVersionWithMagic() throws Exception {
        var path = writeSingleNodeGraph(OnDiskGraphIndex.CURRENT_VERSION);
        try (var readerSupplier = new SimpleMappedReader.Supplier(path);
             var reader = readerSupplier.get())
        {
            long startPosition = reader.getPosition();
            var format = GraphIndexFormatFactory.detectVersion(reader);
            assertEquals(OnDiskGraphIndex.CURRENT_VERSION, format.getVersion());
            assertEquals(startPosition, reader.getPosition());
        }
    }

    private Path writeSingleNodeGraph(int version) throws IOException {
        var graph = new TestUtil.RandomlyConnectedGraphIndex(5, 2, getRandom());
        var ravv = new TestVectorGraph.CircularFloatVectorValues(graph.size(0));
        var outputPath = testDirectory.resolve("format_" + version);
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
                .withVersion(version)
                .with(new InlineVectors(ravv.dimension()))
                .build())
        {
            writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                    nodeId -> new InlineVectors.State(ravv.getVector(nodeId))));
        }
        return outputPath;
    }
}
