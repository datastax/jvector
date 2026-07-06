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
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntFunction;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/// Guard tests for the F3 hardening of `OnDiskGraphIndex.View` (prescription:
/// `local/memory_safety_fix_plan.md` F3): out-of-range node ordinals must fail with
/// `IllegalArgumentException` at the offset-computation entry points instead of becoming silent
/// wild offsets into the mapped file, and a corrupt (or stale-metadata) on-disk neighbor degree
/// must fail with a stack-bearing `IllegalStateException` at first touch instead of passing an
/// assert-only guard in production (`-da`) runs and turning into garbage node ids downstream.
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestOnDiskGraphIndexViewGuards extends RandomizedTest {
    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final int DIM = 4;
    private static final int N = 8;

    private Path testDirectory;
    private Path graphPath;

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory("jvector_test");
        graphPath = buildInlineGraph(testDirectory.resolve("guards.graph"), TestUtil.createRandomVectors(N, DIM));
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    /// Single-layer graph with inline vectors, mirroring
    /// `TestOnDiskGraphIndexCompactor.buildSimpleSourceGraph`.
    private Path buildInlineGraph(Path outputPath, List<VectorFloat<?>> vecs) throws IOException {
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vecs, DIM);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
        var builder = new GraphIndexBuilder(bsp, DIM, 4, 20, 1.2f, 1.2f, false, true,
                ForkJoinPool.commonPool(), ForkJoinPool.commonPool());
        for (int i = 0; i < vecs.size(); i++) {
            builder.addGraphNode(i, vecs.get(i));
        }
        builder.cleanup();
        var graph = builder.getGraph();

        var writerBuilder = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
                .withMapper(new OrdinalMapper.IdentityMapper(vecs.size() - 1))
                .with(new InlineVectors(DIM));
        var writer = writerBuilder.build();
        Map<FeatureId, IntFunction<Feature.State>> writeSuppliers = new EnumMap<>(FeatureId.class);
        writeSuppliers.put(FeatureId.INLINE_VECTORS, ordinal -> new InlineVectors.State(ravv.getVector(ordinal)));
        for (int node = 0; node < vecs.size(); node++) {
            var stateMap = new EnumMap<FeatureId, Feature.State>(FeatureId.class);
            stateMap.put(FeatureId.INLINE_VECTORS, writeSuppliers.get(FeatureId.INLINE_VECTORS).apply(node));
            writer.writeInline(node, stateMap);
        }
        writer.write(writeSuppliers);
        return outputPath;
    }

    private static void expectOutOfRange(Runnable access) {
        try {
            access.run();
            fail("expected IllegalArgumentException for an out-of-range node ordinal");
        } catch (IllegalArgumentException e) {
            assertTrue("message must identify the bad ordinal and the valid range, was: " + e.getMessage(),
                    e.getMessage().contains("out of range"));
        }
    }

    private static void expectCorruptDegree(Runnable access) {
        try {
            access.run();
            fail("expected IllegalStateException for a corrupt on-disk degree");
        } catch (IllegalStateException e) {
            assertTrue("message must identify the corrupt degree, was: " + e.getMessage(),
                    e.getMessage().contains("degree"));
        }
    }

    /// Out-of-range ordinals — negative and one-past-the-end — must throw at both record-access
    /// entry points (vector reads and neighbor reads), while in-range accesses keep working.
    @Test
    public void outOfRangeNodeOrdinalsThrow() throws Exception {
        try (ReaderSupplier rs = ReaderSupplierFactory.open(graphPath)) {
            var graph = OnDiskGraphIndex.load(rs);
            try (var view = graph.getView()) {
                VectorFloat<?> buf = VTS.createFloatVector(DIM);
                expectOutOfRange(() -> view.getVectorInto(-1, buf, 0));
                expectOutOfRange(() -> view.getVectorInto(N, buf, 0));
                expectOutOfRange(() -> view.getNeighborsIterator(0, -1));
                expectOutOfRange(() -> view.getNeighborsIterator(0, N));

                // in-range accesses are unaffected by the guard
                view.getVectorInto(0, buf, 0);
                assertNotNull(view.getNeighborsIterator(0, N - 1));
            }
        }
    }

    /// A degree field patched to garbage — huge or negative — must throw at first touch. The
    /// degree offsets are located through the package-private `neighborsOffsetFor` before the
    /// file is patched on disk (big-endian, matching the format) and reopened.
    @Test
    public void corruptOnDiskDegreeThrows() throws Exception {
        long hugeDegreeOffset;
        long negativeDegreeOffset;
        try (ReaderSupplier rs = ReaderSupplierFactory.open(graphPath)) {
            var graph = OnDiskGraphIndex.load(rs);
            try (var view = graph.getView()) {
                hugeDegreeOffset = view.neighborsOffsetFor(0, 2);
                negativeDegreeOffset = view.neighborsOffsetFor(0, 5);
            }
        }

        try (FileChannel ch = FileChannel.open(graphPath, StandardOpenOption.WRITE)) {
            patchInt(ch, hugeDegreeOffset, Integer.MAX_VALUE / 2);
            patchInt(ch, negativeDegreeOffset, -17);
        }

        try (ReaderSupplier rs = ReaderSupplierFactory.open(graphPath)) {
            var graph = OnDiskGraphIndex.load(rs);
            try (var view = graph.getView()) {
                expectCorruptDegree(() -> view.getNeighborsIterator(0, 2));
                expectCorruptDegree(() -> view.getNeighborsIterator(0, 5));

                // an unpatched node still reads fine
                assertNotNull(view.getNeighborsIterator(0, 0));
            }
        }
    }

    private static void patchInt(FileChannel ch, long offset, int value) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(Integer.BYTES);   // big-endian by default, matching the on-disk format
        bb.putInt(value);
        bb.flip();
        while (bb.hasRemaining()) {
            offset += ch.write(bb, offset);
        }
    }
}
