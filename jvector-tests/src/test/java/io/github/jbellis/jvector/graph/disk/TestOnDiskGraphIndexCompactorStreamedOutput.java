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
import io.github.jbellis.jvector.disk.ByteBufferReader;
import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.NodeQueue;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.BoundedLongHeap;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Compaction into a caller-provided {@link IndexWriter}: the graph is written sequentially at the
 * writer's position, byte for byte what {@code compact(Path)} writes to a file, and the scratch and
 * spill files the merge needs live in the work directory only while it runs.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestOnDiskGraphIndexCompactorStreamedOutput extends RandomizedTest {
    private static final int DIMENSION = 32;
    private static final int NUM_SOURCES = 3;
    private static final int VECTORS_PER_SOURCE = 256;
    // Odd on purpose: nothing in the output may assume it starts at 0 or on an aligned offset.
    private static final int PREFIX = 13;

    private Path testDirectory;
    private Path workDirectory;
    private final List<List<VectorFloat<?>>> sourceVecs = new ArrayList<>();
    private final List<ReaderSupplier> readers = new ArrayList<>();

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory("jvector_streamed_output_test");
        workDirectory = Files.createDirectory(testDirectory.resolve("work"));
        for (int s = 0; s < NUM_SOURCES; s++)
            sourceVecs.add(createRandomVectors(VECTORS_PER_SOURCE, DIMENSION));
    }

    @After
    public void tearDown() {
        for (ReaderSupplier rs : readers) {
            try {
                rs.close();
            } catch (Exception ignored) {
                // best effort: the directory is deleted next
            }
        }
        TestUtil.deleteQuietly(testDirectory);
    }

    @Test
    public void testStreamedGraphMatchesFileGraph() throws Exception {
        VectorSimilarityFunction vsf = VectorSimilarityFunction.COSINE;
        List<FixedBitSet> live = liveWithDeletions();
        List<OnDiskGraphIndex> sources = loadSources(vsf, true, false, "plain");

        Path file = testDirectory.resolve("plain_out");
        new OnDiskGraphIndexCompactor(sources, live, vsf, null).compact(file);

        var out = new MemoryIndexWriter(Long.MAX_VALUE);
        out.write(new byte[PREFIX]);
        var compactor = new OnDiskGraphIndexCompactor(sources, live, vsf, null);
        compactor.compact(out, workDirectory);

        assertFalse("the caller's output stays open", out.closed);
        assertSameGraphAt(Files.readAllBytes(file), out.bytesFrom(PREFIX), PREFIX);
        assertWorkDirectoryEmpty();

        OnDiskGraphIndex streamed = load(out.toArray(), PREFIX);
        assertEquals(liveCount(live), streamed.size(0));
        try (var view = streamed.getView()) {
            VectorFloat<?> buf = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(DIMENSION);
            for (int s = 0; s < NUM_SOURCES; s++)
                for (int i = 0; i < VECTORS_PER_SOURCE; i++)
                    if (live.get(s).get(i)) {
                        view.getVectorInto(compactor.ordinalMappers().get(s).oldToNew(i), buf, 0);
                        for (int d = 0; d < DIMENSION; d++)
                            assertEquals(sourceVecs.get(s).get(i).get(d), buf.get(d), 0.0f);
                    }
        }
    }

    /** The combination a caller that stores its own vectors uses: topology-only FUSED_PQ sources, streamed out. */
    @Test
    public void testStreamedTopologyOnlyFusedPQGraphMatchesFileGraph() throws Exception {
        VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;
        List<FixedBitSet> live = liveWithDeletions();
        List<OnDiskGraphIndex> sources = loadSources(vsf, false, true, "fused");

        Path file = testDirectory.resolve("fused_out");
        var fileCompactor = new OnDiskGraphIndexCompactor(sources, sourceRavvs(), null, live, vsf, null);
        fileCompactor.compact(file);

        var out = new MemoryIndexWriter(Long.MAX_VALUE);
        out.write(new byte[PREFIX]);
        var compactor = new OnDiskGraphIndexCompactor(sources, sourceRavvs(), null, live, vsf, null);
        compactor.compact(out, workDirectory);

        assertWorkDirectoryEmpty();
        // The codebook is retrained from a random sample and steers the merge, so two compactions of
        // the same sources differ in codes and edges even to a file: same layout, same quality.
        assertEquals(Files.size(file), out.position() - PREFIX);
        OnDiskGraphIndex streamed = load(out.toArray(), PREFIX);
        assertEquals(liveCount(live), streamed.size(0));
        assertEquals(0, streamed.getDimension());
        assertNotNull(streamed.getFeatures().get(FeatureId.FUSED_PQ));
        double fileRecall = recall(load(Files.readAllBytes(file), 0), fileCompactor.ordinalMappers(), live, vsf);
        double streamedRecall = recall(streamed, compactor.ordinalMappers(), live, vsf);
        assertTrue(String.format("streamed recall %.4f vs file %.4f", streamedRecall, fileRecall), streamedRecall >= fileRecall - 0.05);
    }

    @Test
    public void testStreamedSidecarMatchesFileSidecar() throws Exception {
        VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;
        List<FixedBitSet> live = liveWithDeletions();
        List<OnDiskGraphIndex> sources = loadSources(vsf, true, false, "sidecar");
        List<CompressedVectors> sidecars = new ArrayList<>();
        for (RandomAccessVectorValues ravv : sourceRavvs()) {
            ProductQuantization pq = ProductQuantization.compute(ravv, 8, 16, true, UNWEIGHTED, ForkJoinPool.commonPool(), ForkJoinPool.commonPool());
            sidecars.add(pq.encodeAll(ravv, ForkJoinPool.commonPool()));
        }

        Path graphFile = testDirectory.resolve("sidecar_graph");
        Path pqFile = testDirectory.resolve("sidecar_pq");
        new OnDiskGraphIndexCompactor(sources, sidecars, live, vsf, null).compact(graphFile, pqFile);

        var graphOut = new MemoryIndexWriter(Long.MAX_VALUE);
        graphOut.write(new byte[PREFIX]);
        var pqOut = new MemoryIndexWriter(Long.MAX_VALUE);
        pqOut.write(new byte[PREFIX]);
        var compactor = new OnDiskGraphIndexCompactor(sources, sidecars, live, vsf, null);
        compactor.compact(graphOut, pqOut, workDirectory);

        assertSameGraphAt(Files.readAllBytes(graphFile), graphOut.bytesFrom(PREFIX), PREFIX);
        assertFalse(graphOut.closed || pqOut.closed);
        assertWorkDirectoryEmpty();

        // The sidecar codebook is retrained from a random sample: same layout, codes of its own codebook.
        assertEquals(Files.size(pqFile), pqOut.position() - PREFIX);
        var reader = new ByteBufferReader(ByteBuffer.wrap(pqOut.toArray()));
        reader.seek(PREFIX);
        PQVectors merged = PQVectors.load(reader);
        assertEquals(liveCount(live), merged.count());
        ProductQuantization mergedPq = merged.getCompressor();
        var code = VectorizationProvider.getInstance().getVectorTypeSupport().createByteSequence(mergedPq.compressedVectorSize());
        for (int s = 0; s < NUM_SOURCES; s++)
            for (int i = 0; i < VECTORS_PER_SOURCE; i++)
                if (live.get(s).get(i)) {
                    mergedPq.encodeTo(sourceVecs.get(s).get(i), code);
                    var stored = merged.get(compactor.ordinalMappers().get(s).oldToNew(i));
                    for (int b = 0; b < code.length(); b++)
                        assertEquals("source " + s + " node " + i + " code byte " + b, code.get(b), stored.get(b));
                }
    }

    /** A failing output fails the compaction and still leaves nothing behind in the work directory. */
    @Test
    public void testFailingOutputLeavesWorkDirectoryEmpty() throws Exception {
        VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;
        List<FixedBitSet> live = allLive();
        List<OnDiskGraphIndex> sources = loadSources(vsf, false, true, "failing");

        // Half the graph is well past the header (which holds the whole codebook) and inside the
        // level-0 records, while the merge holds its scratch file and its reverse-offer spill directory.
        Path file = testDirectory.resolve("failing_out");
        new OnDiskGraphIndexCompactor(sources, sourceRavvs(), null, live, vsf, null).compact(file);
        var out = new MemoryIndexWriter(Files.size(file) / 2);
        var compactor = new OnDiskGraphIndexCompactor(sources, sourceRavvs(), null, live, vsf, null);
        var e = assertThrows(RuntimeException.class, () -> compactor.compact(out, workDirectory));
        assertTrue("fails in level 0: " + e, out.position() > Files.size(file) / 4);
        assertWorkDirectoryEmpty();
    }

    @Test
    public void testStreamedOutputValidation() throws Exception {
        VectorSimilarityFunction vsf = VectorSimilarityFunction.COSINE;
        List<OnDiskGraphIndex> sources = loadSources(vsf, true, false, "validate");
        var compactor = new OnDiskGraphIndexCompactor(sources, allLive(), vsf, null);
        var out = new MemoryIndexWriter(Long.MAX_VALUE);

        assertThrows(NullPointerException.class, () -> compactor.compact((IndexWriter) null, workDirectory));
        assertThrows(IllegalArgumentException.class, () -> compactor.compact(out, testDirectory.resolve("missing")));
        assertThrows(IllegalArgumentException.class, () -> compactor.compact(out, testDirectory.resolve("validate_0")));
        // compact(graph, compressed, work) needs the sidecars it writes
        assertThrows(IllegalStateException.class, () -> compactor.compact(out, new MemoryIndexWriter(Long.MAX_VALUE), workDirectory));
        assertEquals("nothing is written on a rejected call", 0, out.position());
    }

    // ---- helpers ----

    /**
     * The streamed graph is the file graph, except for the footer's pointer to the header copy: a
     * reader seeks it as an absolute offset, so it moves with the position the graph starts at.
     */
    private static void assertSameGraphAt(byte[] file, byte[] streamed, int startOffset) {
        assertEquals("graph length", file.length, streamed.length);
        int pointer = file.length - Integer.BYTES - Long.BYTES;
        ByteBuffer expected = ByteBuffer.wrap(file.clone());
        expected.putLong(pointer, expected.getLong(pointer) + startOffset);
        assertArrayEquals(expected.array(), streamed);
    }

    /** Recall@10 of exact-scored searches against brute force over the live vectors, laid out by output ordinal. */
    private double recall(OnDiskGraphIndex compacted, List<OrdinalMapper> mappers, List<FixedBitSet> live,
                          VectorSimilarityFunction vsf) throws IOException {
        final int topK = 10, queries = 40;
        List<VectorFloat<?>> byOrdinal = new ArrayList<>(Collections.nCopies(compacted.size(0), null));
        for (int s = 0; s < NUM_SOURCES; s++)
            for (int i = 0; i < VECTORS_PER_SOURCE; i++)
                if (live.get(s).get(i))
                    byOrdinal.set(mappers.get(s).oldToNew(i), sourceVecs.get(s).get(i));
        var ravv = new ListRandomAccessVectorValues(byOrdinal, DIMENSION);
        int hits = 0;
        try (var searcher = new GraphSearcher(compacted)) {
            for (int q = 0; q < queries; q++) {
                VectorFloat<?> query = createRandomVectors(1, DIMENSION).get(0);
                NodeQueue expected = new NodeQueue(new BoundedLongHeap(topK), NodeQueue.Order.MIN_HEAP);
                for (int ord = 0; ord < byOrdinal.size(); ord++)
                    expected.push(ord, vsf.compare(query, byOrdinal.get(ord)));
                Set<Integer> truth = new HashSet<>();
                while (expected.size() > 0)
                    truth.add(expected.pop());
                for (var ns : searcher.search(DefaultSearchScoreProvider.exact(query, vsf, ravv), topK, 50, 0.0f, 0.0f, Bits.ALL).getNodes())
                    if (truth.contains(ns.node))
                        hits++;
            }
        }
        return (double) hits / (queries * topK);
    }

    private void assertWorkDirectoryEmpty() throws IOException {
        try (Stream<Path> left = Files.list(workDirectory)) {
            assertEquals("scratch and spill files left in the work directory", List.of(), left.collect(Collectors.toList()));
        }
    }

    private List<RandomAccessVectorValues> sourceRavvs() {
        List<RandomAccessVectorValues> ravvs = new ArrayList<>();
        for (List<VectorFloat<?>> vecs : sourceVecs)
            ravvs.add(new ListRandomAccessVectorValues(vecs, DIMENSION));
        return ravvs;
    }

    private List<OnDiskGraphIndex> loadSources(VectorSimilarityFunction vsf, boolean storeVectors, boolean fusedPQ, String prefix) throws IOException {
        List<OnDiskGraphIndex> sources = new ArrayList<>();
        for (int s = 0; s < NUM_SOURCES; s++) {
            Path path = CompactionSources.writeSource(testDirectory.resolve(prefix + "_" + s), sourceVecs.get(s), DIMENSION, vsf, storeVectors, fusedPQ);
            ReaderSupplier rs = ReaderSupplierFactory.open(path);
            readers.add(rs);
            sources.add(OnDiskGraphIndex.load(rs));
        }
        return sources;
    }

    private static OnDiskGraphIndex load(byte[] bytes, long offset) {
        ReaderSupplier rs = () -> new ByteBufferReader(ByteBuffer.wrap(bytes));
        return OnDiskGraphIndex.load(rs, offset);
    }

    private List<FixedBitSet> allLive() {
        List<FixedBitSet> live = new ArrayList<>();
        for (int s = 0; s < NUM_SOURCES; s++) {
            var bits = new FixedBitSet(VECTORS_PER_SOURCE);
            bits.set(0, VECTORS_PER_SOURCE);
            live.add(bits);
        }
        return live;
    }

    private List<FixedBitSet> liveWithDeletions() {
        List<FixedBitSet> live = allLive();
        for (int s = 0; s < NUM_SOURCES; s++)
            for (int i = s; i < VECTORS_PER_SOURCE; i += 10)
                live.get(s).clear(i);
        return live;
    }

    private static int liveCount(List<FixedBitSet> live) {
        int n = 0;
        for (FixedBitSet bits : live)
            n += bits.cardinality();
        return n;
    }
}
