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
import io.github.jbellis.jvector.disk.SimpleWriter;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.PageAlignment;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestPageAlignment extends LuceneTestCase {
    private static final int PAGE = 512;
    private static final int DEGREE = 8;
    private Path directory;

    private enum Writer { PARALLEL, PARALLEL_TWO_PHASE, SEQUENTIAL_RANDOM_ACCESS, SEQUENTIAL_STREAM }

    @Before
    public void setup() throws IOException {
        directory = Files.createTempDirectory(getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(directory);
    }

    @Test
    public void layoutPacksPadsAndSkips() {
        var packed = L0Layout.aligned(104, PAGE);
        assertEquals(4, packed.recordsPerGroup);
        assertEquals(PAGE, packed.groupBytes);
        assertEquals(PAGE + 104, packed.offset(5));
        assertEquals(2 * PAGE, packed.totalBytes(5));
        assertEquals(0, packed.paddingAfter(2, 9));
        assertEquals(PAGE - 4 * 104, packed.paddingAfter(3, 9));
        assertEquals(PAGE - 2 * 104, packed.paddingAfter(9, 9));

        var onePerPage = L0Layout.aligned(440, PAGE);
        assertEquals(1, onePerPage.recordsPerGroup);
        assertEquals(3L * PAGE, onePerPage.offset(3));

        var twoPages = L0Layout.aligned(920, PAGE);
        assertEquals(2 * PAGE, twoPages.groupBytes);
        assertEquals(2 * PAGE - 920, twoPages.paddingAfter(0, 5));

        assertTrue(L0Layout.alignmentPays(PAGE, PAGE));
        assertTrue(L0Layout.alignmentPays(PAGE + PAGE / 2, PAGE));
        assertFalse(L0Layout.alignmentPays(PAGE + PAGE / 2 - 1, PAGE));

        var unaligned = L0Layout.unaligned(104);
        assertEquals(5 * 104, unaligned.totalBytes(5));
        assertEquals(0, unaligned.paddingAfter(9, 9));
    }

    @Test
    public void packedRecordsRoundTrip() throws IOException {
        // 4 + 16 * 4 + 4 + 8 * 4 = 104 bytes: four records per 512-byte page.
        for (var writer : Writer.values()) roundTrip(writer, 16, true, true);
    }

    @Test
    public void onePerPageRecordsRoundTrip() throws IOException {
        // 4 + 100 * 4 + 36 = 440 bytes: one record per page.
        for (var writer : Writer.values()) roundTrip(writer, 100, true, true);
    }

    @Test
    public void multiPageRecordsRoundTrip() throws IOException {
        // 4 + 220 * 4 + 36 = 920 bytes, last page 408 of 512: padded to two pages.
        for (var writer : Writer.values()) roundTrip(writer, 220, false, true);
    }

    @Test
    public void barelyMultiPageRecordsStayUnaligned() throws IOException {
        // 4 + 160 * 4 + 36 = 680 bytes, last page 168 of 512: padding would cost more than it saves.
        roundTrip(Writer.PARALLEL, 160, true, false);
    }

    @Test
    public void rejectsOldFormatsAndInvalidPages() throws IOException {
        var vectors = vectors(16);
        var graph = graph(vectors, false);
        assertThrows(IllegalArgumentException.class, () -> new OnDiskGraphIndexWriter.Builder(graph, directory.resolve("v6"))
                .withVersion(6).with(new InlineVectors(16)).withPageAlignment(PAGE).build());
        assertThrows(IllegalArgumentException.class, () -> new OnDiskGraphIndexWriter.Builder(graph, directory.resolve("odd"))
                .with(new InlineVectors(16)).withPageAlignment(1000));
        assertThrows(IllegalArgumentException.class, () -> new OnDiskGraphIndexWriter.Builder(graph, directory.resolve("direct"))
                .with(new InlineVectors(16)).with(new PageAlignment(PAGE, 0)).build());
    }

    private void roundTrip(Writer kind, int dimension, boolean hierarchy, boolean expectAligned) throws IOException {
        var vectors = vectors(dimension);
        var graph = graph(vectors, hierarchy);
        Path plain = directory.resolve(kind + "-" + dimension + "-plain");
        Path aligned = directory.resolve(kind + "-" + dimension + "-aligned");
        String layout = write(kind, graph, vectors, plain, false);
        String alignedLayout = write(kind, graph, vectors, aligned, true);
        assertEquals(expectAligned, !alignedLayout.contains("unaligned"));
        if (!expectAligned) assertTrue(alignedLayout, alignedLayout.contains("not applied"));
        assertTrue(layout, layout.endsWith("unaligned"));

        try (var plainReader = new SimpleMappedReader.Supplier(plain);
             var alignedReader = new SimpleMappedReader.Supplier(aligned)) {
            var plainIndex = OnDiskGraphIndex.load(plainReader);
            var alignedIndex = OnDiskGraphIndex.load(alignedReader);
            TestUtil.assertGraphEquals(graph, alignedIndex);
            TestOnDiskGraphIndex.validateVectors(alignedIndex.getView(), vectors);
            assertEquals(expectAligned, alignedIndex.getFeatureSet().contains(FeatureId.PAGE_ALIGNMENT));
            if (expectAligned) assertRecordsAligned(alignedIndex, Files.size(aligned));
            for (int q = 0; q < 10; q++) {
                var query = vectors.getVector(q);
                var expected = GraphSearcher.search(query, 5, vectors, VectorSimilarityFunction.COSINE, plainIndex, Bits.ALL);
                var actual = GraphSearcher.search(query, 5, vectors, VectorSimilarityFunction.COSINE, alignedIndex, Bits.ALL);
                assertArrayEquals(nodes(expected), nodes(actual));
            }
        }
    }

    private String write(Writer kind, ImmutableGraphIndex graph, RandomAccessVectorValues vectors, Path path,
                         boolean align) throws IOException {
        var suppliers = Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                node -> new InlineVectors.State(vectors.getVector(node)));
        if (kind == Writer.SEQUENTIAL_STREAM) {
            try (var out = new SimpleWriter(path)) {
                var builder = new OnDiskSequentialGraphIndexWriter.Builder(graph, out).with(new InlineVectors(vectors.dimension()));
                if (align) builder.withPageAlignment(PAGE);
                try (var writer = builder.build()) {
                    writer.write(suppliers);
                    return writer.describeRecordLayout();
                }
            }
        }
        AbstractGraphIndexWriter.Builder<? extends RandomAccessOnDiskGraphIndexWriter, ?> builder =
                kind == Writer.SEQUENTIAL_RANDOM_ACCESS
                        ? new OnDiskGraphIndexWriter.Builder(graph, path)
                        : new OnDiskParallelGraphIndexWriter.Builder(graph, path).withParallelWorkerThreads(3);
        builder.with(new InlineVectors(vectors.dimension()));
        if (align) builder.withPageAlignment(PAGE);
        try (var writer = builder.build()) {
            if (kind == Writer.PARALLEL_TWO_PHASE) {
                for (int node = 0; node < graph.size(0); node++) {
                    writer.writeFeaturesInline(node, Map.of(FeatureId.INLINE_VECTORS, new InlineVectors.State(vectors.getVector(node))));
                }
                writer.write(Map.of());
            } else {
                writer.write(suppliers);
            }
            return writer.describeRecordLayout();
        }
    }

    /** Every record lies within whole pages: none crosses a page boundary, and multi-page records start one. */
    private static void assertRecordsAligned(OnDiskGraphIndex index, long fileBytes) {
        long first;
        try {
            var field = OnDiskGraphIndex.class.getDeclaredField("neighborsOffset");
            field.setAccessible(true);
            first = field.getLong(index);
        } catch (ReflectiveOperationException e) {
            throw new AssertionError(e);
        }
        var layout = index.l0Layout;
        assertEquals(0, first % PAGE);
        for (int node = 0; node < index.getIdUpperBound(); node++) {
            long start = first + layout.offset(node);
            long end = start + layout.recordBytes - 1;
            if (layout.recordBytes <= PAGE) assertEquals("node " + node, start / PAGE, end / PAGE);
            else assertEquals("node " + node, 0, start % PAGE);
        }
        assertTrue(fileBytes >= first + layout.totalBytes(index.getIdUpperBound()));
    }

    private static int[] nodes(io.github.jbellis.jvector.graph.SearchResult result) {
        return Arrays.stream(result.getNodes()).mapToInt(n -> n.node).toArray();
    }

    private static RandomAccessVectorValues vectors(int dimension) {
        return new ListRandomAccessVectorValues(new ArrayList<>(TestUtil.createRandomVectors(200, dimension)), dimension);
    }

    private static ImmutableGraphIndex graph(RandomAccessVectorValues vectors, boolean hierarchy) {
        var builder = new GraphIndexBuilder(vectors, VectorSimilarityFunction.COSINE, DEGREE, 64, 1.2f, 1.2f, hierarchy);
        return TestUtil.buildSequentially(builder, vectors);
    }
}
