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
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.NodeQueue;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
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
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Compaction into output ordinals the caller chooses: the merge is the one the compactor runs with its own ordinals, and
 * only the numbering of the output changes, in the records, the neighbour lists, the upper layers, the entry node and the
 * compressed sidecar.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestOnDiskGraphIndexCompactorOutputOrdinals extends RandomizedTest {
    private static final int DIMENSION = 32;
    private static final int NUM_SOURCES = 3;
    private static final int VECTORS_PER_SOURCE = 256;
    private static final int PREFIX = 13;

    private Path testDirectory;
    private Path workDirectory;
    private final List<List<VectorFloat<?>>> sourceVecs = new ArrayList<>();
    private final List<ReaderSupplier> readers = new ArrayList<>();

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory("jvector_output_ordinals_test");
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
    public void testOutputOrdinalsRenumberTheGraphTheCompactorBuilds() throws Exception {
        VectorSimilarityFunction vsf = VectorSimilarityFunction.COSINE;
        List<FixedBitSet> live = liveWithDeletions();
        List<OnDiskGraphIndex> sources = loadSources(vsf, false, "plain");

        Path own = testDirectory.resolve("own_ordinals");
        var ownCompactor = new OnDiskGraphIndexCompactor(sources, live, vsf, null);
        ownCompactor.compact(own);

        List<int[]> ordinals = shuffledOutputOrdinals(live, 42);
        Path chosen = testDirectory.resolve("chosen_ordinals");
        var compactor = new OnDiskGraphIndexCompactor(sources, live, vsf, null).withOutputOrdinals(ordinals);
        compactor.compact(chosen);

        assertMappersAre(ordinals, live, compactor.ordinalMappers());
        assertRenumbered(load(Files.readAllBytes(own), 0), ownCompactor.ordinalMappers(), load(Files.readAllBytes(chosen), 0), ordinals, live);
    }

    @Test
    public void testStreamedOutputOrdinalsMatchTheFile() throws Exception {
        VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;
        List<FixedBitSet> live = liveWithDeletions();
        List<OnDiskGraphIndex> sources = loadSources(vsf, false, "streamed");
        List<int[]> ordinals = shuffledOutputOrdinals(live, 7);

        Path file = testDirectory.resolve("streamed_file");
        new OnDiskGraphIndexCompactor(sources, live, vsf, null).withOutputOrdinals(ordinals).compact(file);

        var out = new MemoryIndexWriter(Long.MAX_VALUE);
        out.write(new byte[PREFIX]);
        new OnDiskGraphIndexCompactor(sources, live, vsf, null).withOutputOrdinals(ordinals).compact(out, workDirectory);

        assertSameGraphAt(Files.readAllBytes(file), out.bytesFrom(PREFIX), PREFIX);
        try (var left = Files.list(workDirectory)) {
            assertEquals("scratch files left in the work directory", 0, left.count());
        }
    }

    @Test
    public void testSidecarFollowsTheOutputOrdinals() throws Exception {
        VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;
        List<FixedBitSet> live = liveWithDeletions();
        List<OnDiskGraphIndex> sources = loadSources(vsf, false, "sidecar");
        List<CompressedVectors> sidecars = new ArrayList<>();
        for (RandomAccessVectorValues ravv : sourceRavvs()) {
            ProductQuantization pq = ProductQuantization.compute(ravv, 8, 16, true, UNWEIGHTED, ForkJoinPool.commonPool(), ForkJoinPool.commonPool());
            sidecars.add(pq.encodeAll(ravv, ForkJoinPool.commonPool()));
        }
        List<int[]> ordinals = shuffledOutputOrdinals(live, 11);

        Path graphFile = testDirectory.resolve("sidecar_graph");
        Path pqFile = testDirectory.resolve("sidecar_pq");
        new OnDiskGraphIndexCompactor(sources, sidecars, live, vsf, null).withOutputOrdinals(ordinals).compact(graphFile, pqFile);
        assertCodesAt(PQVectors.load(new ByteBufferReader(ByteBuffer.wrap(Files.readAllBytes(pqFile)))), ordinals, live);
        assertVectorsAt(load(Files.readAllBytes(graphFile), 0), ordinals, live);

        var graphOut = new MemoryIndexWriter(Long.MAX_VALUE);
        var pqOut = new MemoryIndexWriter(Long.MAX_VALUE);
        new OnDiskGraphIndexCompactor(sources, sidecars, live, vsf, null).withOutputOrdinals(ordinals).compact(graphOut, pqOut, workDirectory);
        assertCodesAt(PQVectors.load(new ByteBufferReader(ByteBuffer.wrap(pqOut.toArray()))), ordinals, live);
        assertArrayEquals("the streamed graph is the file graph", Files.readAllBytes(graphFile), graphOut.toArray());
    }

    /** FUSED_PQ retrains its codebook from a random sample, so the merge itself differs run to run: the vectors and the recall are compared. */
    @Test
    public void testFusedPQGraphFollowsTheOutputOrdinals() throws Exception {
        VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;
        List<FixedBitSet> live = liveWithDeletions();
        List<OnDiskGraphIndex> sources = loadSources(vsf, true, "fused");

        Path own = testDirectory.resolve("fused_own");
        var ownCompactor = new OnDiskGraphIndexCompactor(sources, live, vsf, null);
        ownCompactor.compact(own);

        List<int[]> ordinals = shuffledOutputOrdinals(live, 3);
        Path chosen = testDirectory.resolve("fused_chosen");
        new OnDiskGraphIndexCompactor(sources, live, vsf, null).withOutputOrdinals(ordinals).compact(chosen);

        OnDiskGraphIndex graph = load(Files.readAllBytes(chosen), 0);
        assertVectorsAt(graph, ordinals, live);
        double ownRecall = recall(load(Files.readAllBytes(own), 0), ownCompactor.ordinalMappers(), live, vsf);
        double chosenRecall = recall(graph, mappersOf(ordinals, live), live, vsf);
        assertTrue(String.format("recall %.4f with output ordinals vs %.4f with its own", chosenRecall, ownRecall), chosenRecall >= ownRecall - 0.05);
    }

    @Test
    public void testOutputOrdinalsValidation() throws Exception {
        VectorSimilarityFunction vsf = VectorSimilarityFunction.COSINE;
        List<FixedBitSet> live = liveWithDeletions();
        List<OnDiskGraphIndex> sources = loadSources(vsf, false, "validate");
        var compactor = new OnDiskGraphIndexCompactor(sources, live, vsf, null);

        assertThrows(NullPointerException.class, () -> compactor.withOutputOrdinals(null));
        assertThrows("one array per source", IllegalArgumentException.class,
                     () -> compactor.withOutputOrdinals(shuffledOutputOrdinals(live, 1).subList(0, NUM_SOURCES - 1)));

        List<int[]> tooShort = shuffledOutputOrdinals(live, 1);
        tooShort.set(1, Arrays.copyOf(tooShort.get(1), VECTORS_PER_SOURCE - 1));
        assertThrows("an entry per source ordinal", IllegalArgumentException.class, () -> compactor.withOutputOrdinals(tooShort));

        List<int[]> outOfRange = shuffledOutputOrdinals(live, 1);
        outOfRange.get(2)[firstLive(live.get(2))] = liveCount(live);
        assertThrows("dense over the live nodes", IllegalArgumentException.class, () -> compactor.withOutputOrdinals(outOfRange));

        List<int[]> duplicate = shuffledOutputOrdinals(live, 1);
        duplicate.get(0)[firstLive(live.get(0))] = duplicate.get(1)[firstLive(live.get(1))];
        assertThrows("each output ordinal once", IllegalArgumentException.class, () -> compactor.withOutputOrdinals(duplicate));

        // what a deleted node is given is not read
        List<int[]> deletedIgnored = shuffledOutputOrdinals(live, 1);
        for (int s = 0; s < NUM_SOURCES; s++)
            for (int i = 0; i < VECTORS_PER_SOURCE; i++)
                if (!live.get(s).get(i))
                    deletedIgnored.get(s)[i] = -7;
        compactor.withOutputOrdinals(deletedIgnored);
    }

    // ---- helpers ----

    /** The output ordinals 0..live-1, handed to the live nodes of every source in a random order. */
    private static List<int[]> shuffledOutputOrdinals(List<FixedBitSet> live, long seed) {
        int count = liveCount(live);
        int[] shuffled = new int[count];
        for (int i = 0; i < count; i++)
            shuffled[i] = i;
        Random random = new Random(seed);
        for (int i = count - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = shuffled[i];
            shuffled[i] = shuffled[j];
            shuffled[j] = t;
        }
        List<int[]> ordinals = new ArrayList<>();
        int next = 0;
        for (FixedBitSet bits : live) {
            int[] source = new int[bits.length()];
            Arrays.fill(source, OrdinalMapper.OMITTED);
            for (int i = 0; i < source.length; i++)
                if (bits.get(i))
                    source[i] = shuffled[next++];
            ordinals.add(source);
        }
        return ordinals;
    }

    private static void assertMappersAre(List<int[]> ordinals, List<FixedBitSet> live, List<OrdinalMapper> mappers) {
        assertEquals(NUM_SOURCES, mappers.size());
        for (int s = 0; s < NUM_SOURCES; s++) {
            OrdinalMapper mapper = mappers.get(s);
            assertEquals(liveCount(live) - 1, mapper.maxOrdinal());
            for (int i = 0; i < VECTORS_PER_SOURCE; i++) {
                if (live.get(s).get(i)) {
                    assertEquals(ordinals.get(s)[i], mapper.oldToNew(i));
                    assertEquals(i, mapper.newToOld(ordinals.get(s)[i]));
                } else {
                    assertEquals(OrdinalMapper.OMITTED, mapper.oldToNew(i));
                }
            }
        }
    }

    /**
     * {@code chosen} is {@code own} with every ordinal renamed: same nodes on every level, same neighbours in the same order,
     * same vectors, same entry node.
     */
    private static void assertRenumbered(OnDiskGraphIndex own, List<OrdinalMapper> ownMappers, OnDiskGraphIndex chosen,
                                         List<int[]> ordinals, List<FixedBitSet> live) throws Exception {
        int[] rename = new int[liveCount(live)];
        for (int s = 0; s < NUM_SOURCES; s++)
            for (int i = 0; i < VECTORS_PER_SOURCE; i++)
                if (live.get(s).get(i))
                    rename[ownMappers.get(s).oldToNew(i)] = ordinals.get(s)[i];

        assertEquals(own.getMaxLevel(), chosen.getMaxLevel());
        try (var ownView = own.getView(); var chosenView = chosen.getView()) {
            assertEquals(rename[ownView.entryNode().node], chosenView.entryNode().node);
            assertEquals(ownView.entryNode().level, chosenView.entryNode().level);
            for (int level = 0; level <= own.getMaxLevel(); level++) {
                assertEquals(own.size(level), chosen.size(level));
                Set<Integer> expectedNodes = new HashSet<>();
                for (NodesIterator it = own.getNodes(level); it.hasNext(); ) {
                    int node = it.next();
                    expectedNodes.add(rename[node]);
                    int[] expected = neighbours(ownView, level, node);
                    for (int k = 0; k < expected.length; k++)
                        expected[k] = rename[expected[k]];
                    assertArrayEquals("neighbours of " + node + " at level " + level, expected, neighbours(chosenView, level, rename[node]));
                }
                Set<Integer> nodes = new HashSet<>();
                for (NodesIterator it = chosen.getNodes(level); it.hasNext(); )
                    nodes.add(it.next());
                assertEquals("nodes of level " + level, expectedNodes, nodes);
            }
            VectorFloat<?> expected = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(DIMENSION);
            VectorFloat<?> actual = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(DIMENSION);
            for (int node = 0; node < rename.length; node++) {
                ownView.getVectorInto(node, expected, 0);
                chosenView.getVectorInto(rename[node], actual, 0);
                for (int d = 0; d < DIMENSION; d++)
                    assertEquals(expected.get(d), actual.get(d), 0.0f);
            }
        }
    }

    private static int[] neighbours(ImmutableGraphIndex.View view, int level, int node) {
        var it = view.getNeighborsIterator(level, node);
        int[] neighbours = new int[it.size()];
        for (int k = 0; k < neighbours.length; k++)
            neighbours[k] = it.nextInt();
        return neighbours;
    }

    private void assertVectorsAt(OnDiskGraphIndex graph, List<int[]> ordinals, List<FixedBitSet> live) throws Exception {
        assertEquals(liveCount(live), graph.size(0));
        try (var view = graph.getView()) {
            VectorFloat<?> buf = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(DIMENSION);
            for (int s = 0; s < NUM_SOURCES; s++)
                for (int i = 0; i < VECTORS_PER_SOURCE; i++)
                    if (live.get(s).get(i)) {
                        view.getVectorInto(ordinals.get(s)[i], buf, 0);
                        for (int d = 0; d < DIMENSION; d++)
                            assertEquals(sourceVecs.get(s).get(i).get(d), buf.get(d), 0.0f);
                    }
        }
    }

    private void assertCodesAt(PQVectors merged, List<int[]> ordinals, List<FixedBitSet> live) {
        assertEquals(liveCount(live), merged.count());
        ProductQuantization pq = merged.getCompressor();
        var code = VectorizationProvider.getInstance().getVectorTypeSupport().createByteSequence(pq.compressedVectorSize());
        for (int s = 0; s < NUM_SOURCES; s++)
            for (int i = 0; i < VECTORS_PER_SOURCE; i++)
                if (live.get(s).get(i)) {
                    pq.encodeTo(sourceVecs.get(s).get(i), code);
                    var stored = merged.get(ordinals.get(s)[i]);
                    for (int b = 0; b < code.length(); b++)
                        assertEquals("source " + s + " node " + i + " code byte " + b, code.get(b), stored.get(b));
                }
    }

    /** The streamed graph is the file graph, except for the footer's absolute pointer to the header copy. */
    private static void assertSameGraphAt(byte[] file, byte[] streamed, int startOffset) {
        assertEquals("graph length", file.length, streamed.length);
        int pointer = file.length - Integer.BYTES - Long.BYTES;
        ByteBuffer expected = ByteBuffer.wrap(file.clone());
        expected.putLong(pointer, expected.getLong(pointer) + startOffset);
        assertArrayEquals(expected.array(), streamed);
    }

    private static List<OrdinalMapper> mappersOf(List<int[]> ordinals, List<FixedBitSet> live) {
        List<OrdinalMapper> mappers = new ArrayList<>();
        int max = liveCount(live) - 1;
        for (int[] source : ordinals)
            mappers.add(new OrdinalMapper() {
                @Override
                public int maxOrdinal() {
                    return max;
                }

                @Override
                public int oldToNew(int oldOrdinal) {
                    return source[oldOrdinal];
                }

                @Override
                public int newToOld(int newOrdinal) {
                    throw new UnsupportedOperationException();
                }
            });
        return mappers;
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
        Random random = new Random(5);
        int hits = 0;
        try (var searcher = new GraphSearcher(compacted)) {
            for (int q = 0; q < queries; q++) {
                VectorFloat<?> query = byOrdinal.get(random.nextInt(byOrdinal.size()));
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

    private List<RandomAccessVectorValues> sourceRavvs() {
        List<RandomAccessVectorValues> ravvs = new ArrayList<>();
        for (List<VectorFloat<?>> vecs : sourceVecs)
            ravvs.add(new ListRandomAccessVectorValues(vecs, DIMENSION));
        return ravvs;
    }

    private List<OnDiskGraphIndex> loadSources(VectorSimilarityFunction vsf, boolean fusedPQ, String prefix) throws IOException {
        List<OnDiskGraphIndex> sources = new ArrayList<>();
        for (int s = 0; s < NUM_SOURCES; s++) {
            Path path = CompactionSources.writeSource(testDirectory.resolve(prefix + "_" + s), sourceVecs.get(s), DIMENSION, vsf, true, fusedPQ);
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

    private static List<FixedBitSet> liveWithDeletions() {
        List<FixedBitSet> live = new ArrayList<>();
        for (int s = 0; s < NUM_SOURCES; s++) {
            var bits = new FixedBitSet(VECTORS_PER_SOURCE);
            bits.set(0, VECTORS_PER_SOURCE);
            for (int i = s; i < VECTORS_PER_SOURCE; i += 10)
                bits.clear(i);
            live.add(bits);
        }
        return live;
    }

    private static int firstLive(FixedBitSet bits) {
        return bits.nextSetBit(0);
    }

    private static int liveCount(List<FixedBitSet> live) {
        int n = 0;
        for (FixedBitSet bits : live)
            n += bits.cardinality();
        return n;
    }
}
