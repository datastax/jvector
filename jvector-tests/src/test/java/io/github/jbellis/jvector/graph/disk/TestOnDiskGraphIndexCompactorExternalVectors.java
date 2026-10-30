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
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.NodeQueue;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
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
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Compaction of sources whose vectors are supplied by the caller rather than read from the graphs:
 * topology-only graphs (inline vectors of dimension 0) and graphs whose stored vectors the caller
 * overrides.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestOnDiskGraphIndexCompactorExternalVectors extends RandomizedTest {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final int DIMENSION = 32;
    private static final int NUM_SOURCES = 3;
    private static final int VECTORS_PER_SOURCE = 256;
    private static final int NUM_QUERIES = 40;
    private static final int TOP_K = 10;
    // The merge reads the same vectors whether they come from the graph or from the caller, so the
    // two outputs must be equally good; the slack only absorbs the scheduling of the parallel merge.
    private static final double RECALL_SLACK = 0.05;

    private final ForkJoinPool simdExecutor = ForkJoinPool.commonPool();
    private final ForkJoinPool parallelExecutor = ForkJoinPool.commonPool();
    private Path testDirectory;
    private final List<List<VectorFloat<?>>> sourceVecs = new ArrayList<>();
    private final List<ReaderSupplier> readers = new ArrayList<>();

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory("jvector_external_vectors_test");
        for (int s = 0; s < NUM_SOURCES; s++) {
            sourceVecs.add(createRandomVectors(VECTORS_PER_SOURCE, DIMENSION));
        }
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

    /**
     * Topology-only sources (inline vectors of dimension 0) merge from the caller's vectors, stay
     * topology-only in the output, and the merged graph is as good as the merge of the same sources
     * built with their vectors inline.
     */
    @Test
    public void testTopologyOnlySourcesCompactFromExternalVectors() throws Exception {
        VectorSimilarityFunction vsf = VectorSimilarityFunction.COSINE;
        List<FixedBitSet> live = liveWithDeletions();

        List<OnDiskGraphIndex> inlineSources = loadSources(vsf, true, false, "inline");
        var inlineCompactor = new OnDiskGraphIndexCompactor(inlineSources, live, vsf, null);
        Path inlineOut = testDirectory.resolve("inline_out");
        inlineCompactor.compact(inlineOut);

        List<OnDiskGraphIndex> topologySources = loadSources(vsf, false, false, "topology");
        assertEquals("a topology-only graph stores no vector components", 0, topologySources.get(0).getDimension());
        var compactor = new OnDiskGraphIndexCompactor(topologySources, sourceRavvs(), null, live, vsf, null);
        Path out = testDirectory.resolve("topology_out");
        compactor.compact(out);

        OnDiskGraphIndex compacted = load(out);
        assertEquals(liveCount(live), compacted.size(0));
        assertEquals("the output stays topology-only", 0, compacted.getDimension());
        assertEquals(0, ((InlineVectors) compacted.getFeatures().get(FeatureId.INLINE_VECTORS)).dimension());

        double inlineRecall = recall(load(inlineOut), inlineCompactor.ordinalMappers(), live, vsf);
        double externalRecall = recall(compacted, compactor.ordinalMappers(), live, vsf);
        assertTrue(String.format("external-vector merge recall %.4f vs inline %.4f", externalRecall, inlineRecall),
                   externalRecall >= inlineRecall - RECALL_SLACK);
        assertTrue(String.format("recall %.4f", externalRecall), externalRecall >= 0.8);
    }

    /**
     * Topology-only sources carrying FUSED_PQ: the codebook is retrained from the caller's vectors
     * and the output keeps FUSED_PQ next to its dimension-0 inline vectors.
     */
    @Test
    public void testTopologyOnlyFusedPQSourcesCompactFromExternalVectors() throws Exception {
        VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;
        List<FixedBitSet> live = allLive();

        List<OnDiskGraphIndex> inlineSources = loadSources(vsf, true, true, "inline_fused");
        var inlineCompactor = new OnDiskGraphIndexCompactor(inlineSources, live, vsf, null);
        Path inlineOut = testDirectory.resolve("inline_fused_out");
        inlineCompactor.compact(inlineOut);

        List<OnDiskGraphIndex> topologySources = loadSources(vsf, false, true, "topology_fused");
        var compactor = new OnDiskGraphIndexCompactor(topologySources, sourceRavvs(), null, live, vsf, null);
        Path out = testDirectory.resolve("topology_fused_out");
        compactor.compact(out);

        OnDiskGraphIndex compacted = load(out);
        assertEquals(liveCount(live), compacted.size(0));
        assertEquals(0, compacted.getDimension());
        assertNotNull("the output keeps FUSED_PQ", compacted.getFeatures().get(FeatureId.FUSED_PQ));
        assertEquals(DIMENSION, ((FusedPQ) compacted.getFeatures().get(FeatureId.FUSED_PQ)).getPQ().getOriginalDimension());

        double inlineRecall = recall(load(inlineOut), inlineCompactor.ordinalMappers(), live, vsf);
        double externalRecall = recall(compacted, compactor.ordinalMappers(), live, vsf);
        assertTrue(String.format("external-vector merge recall %.4f vs inline %.4f", externalRecall, inlineRecall),
                   externalRecall >= inlineRecall - RECALL_SLACK);
    }

    /**
     * Topology-only sources with a non-fused PQ sidecar: {@code compact(Path, Path)} retrains and
     * re-encodes from the caller's vectors.
     */
    @Test
    public void testTopologyOnlySourcesWithCompressedSidecar() throws Exception {
        VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;
        List<FixedBitSet> live = liveWithDeletions();
        List<OnDiskGraphIndex> sources = loadSources(vsf, false, false, "topology_sidecar");
        List<CompressedVectors> sidecars = new ArrayList<>();
        for (RandomAccessVectorValues ravv : sourceRavvs()) {
            ProductQuantization pq = ProductQuantization.compute(ravv, 8, 16, true, UNWEIGHTED, simdExecutor, parallelExecutor);
            sidecars.add(pq.encodeAll(ravv, simdExecutor));
        }

        var compactor = new OnDiskGraphIndexCompactor(sources, sourceRavvs(), sidecars, live, vsf, null);
        Path graphOut = testDirectory.resolve("topology_sidecar_graph");
        Path compressedOut = testDirectory.resolve("topology_sidecar_pq");
        compactor.compact(graphOut, compressedOut);

        OnDiskGraphIndex compacted = load(graphOut);
        assertEquals(liveCount(live), compacted.size(0));
        assertEquals(0, compacted.getDimension());
        try (var rs = ReaderSupplierFactory.open(compressedOut); var reader = rs.get()) {
            PQVectors merged = PQVectors.load(reader);
            assertEquals(liveCount(live), merged.count());
            ProductQuantization mergedPq = merged.getCompressor();
            var code = vectorTypeSupport.createByteSequence(mergedPq.compressedVectorSize());
            for (int s = 0; s < NUM_SOURCES; s++) {
                for (int i = 0; i < VECTORS_PER_SOURCE; i++) {
                    if (!live.get(s).get(i))
                        continue;
                    mergedPq.encodeTo(sourceVecs.get(s).get(i), code);
                    var stored = merged.get(compactor.ordinalMappers().get(s).oldToNew(i));
                    for (int b = 0; b < code.length(); b++)
                        assertEquals("source " + s + " node " + i + " code byte " + b, code.get(b), stored.get(b));
                }
            }
        }
        assertTrue(recall(compacted, compactor.ordinalMappers(), live, vsf) >= 0.8);
    }

    /**
     * When the sources store vectors and the caller supplies them too, the caller's vectors win
     * everywhere: the merge reads them and the output stores them.
     */
    @Test
    public void testExternalVectorsReplaceStoredVectors() throws Exception {
        VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;
        List<FixedBitSet> live = liveWithDeletions();
        // The graphs are built and stored with decoy vectors, the caller supplies the real ones.
        List<List<VectorFloat<?>>> decoys = new ArrayList<>();
        for (int s = 0; s < NUM_SOURCES; s++)
            decoys.add(createRandomVectors(VECTORS_PER_SOURCE, DIMENSION));
        List<OnDiskGraphIndex> sources = new ArrayList<>();
        for (int s = 0; s < NUM_SOURCES; s++)
            sources.add(load(writeSource(decoys.get(s), vsf, true, false, "decoy_" + s)));

        var compactor = new OnDiskGraphIndexCompactor(sources, sourceRavvs(), null, live, vsf, null);
        Path out = testDirectory.resolve("decoy_out");
        compactor.compact(out);

        OnDiskGraphIndex compacted = load(out);
        assertEquals(DIMENSION, compacted.getDimension());
        try (var view = compacted.getView()) {
            VectorFloat<?> buf = vectorTypeSupport.createFloatVector(DIMENSION);
            for (int s = 0; s < NUM_SOURCES; s++) {
                for (int i = 0; i < VECTORS_PER_SOURCE; i++) {
                    if (!live.get(s).get(i))
                        continue;
                    view.getVectorInto(compactor.ordinalMappers().get(s).oldToNew(i), buf, 0);
                    VectorFloat<?> expected = sourceVecs.get(s).get(i);
                    for (int d = 0; d < DIMENSION; d++)
                        assertEquals("source " + s + " node " + i + " dim " + d, expected.get(d), buf.get(d), 0.0f);
                }
            }
        }
        // Graph quality follows the caller's vectors, not the decoys the sources were built from.
        assertTrue(recall(compacted, compactor.ordinalMappers(), live, vsf) >= 0.5);
    }

    /**
     * A caller that keeps its vectors elsewhere usually cannot read a deleted one any more, so the
     * compactor never asks: a search that walks through a deleted node scores it by a live neighbour.
     */
    @Test
    public void testDeletedNodesAreNeverRead() throws Exception {
        VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;
        // A third of every source deleted, and one source left with a single live node, whose deleted
        // nodes have no live neighbour to stand in for them.
        List<FixedBitSet> live = allLive();
        for (int s = 0; s < NUM_SOURCES - 1; s++)
            for (int i = s; i < VECTORS_PER_SOURCE; i += 3)
                live.get(s).clear(i);
        live.get(NUM_SOURCES - 1).clear(0, VECTORS_PER_SOURCE);
        live.get(NUM_SOURCES - 1).set(VECTORS_PER_SOURCE / 2);

        List<OnDiskGraphIndex> inlineSources = loadSources(vsf, true, false, "never_read_inline");
        var inlineCompactor = new OnDiskGraphIndexCompactor(inlineSources, live, vsf, null);
        Path inlineOut = testDirectory.resolve("never_read_inline_out");
        inlineCompactor.compact(inlineOut);

        List<RandomAccessVectorValues> liveOnly = new ArrayList<>();
        for (int s = 0; s < NUM_SOURCES; s++)
            liveOnly.add(new LiveOnlyVectors(sourceVecs.get(s), live.get(s)));
        List<OnDiskGraphIndex> sources = loadSources(vsf, false, false, "never_read");
        var compactor = new OnDiskGraphIndexCompactor(sources, liveOnly, null, live, vsf, null);
        Path out = testDirectory.resolve("never_read_out");
        compactor.compact(out);

        OnDiskGraphIndex compacted = load(out);
        assertEquals(liveCount(live), compacted.size(0));
        double inlineRecall = recall(load(inlineOut), inlineCompactor.ordinalMappers(), live, vsf);
        double externalRecall = recall(compacted, compactor.ordinalMappers(), live, vsf);
        assertTrue(String.format("recall without the deleted vectors %.4f vs with them %.4f", externalRecall, inlineRecall),
                   externalRecall >= inlineRecall - RECALL_SLACK);
    }

    /** Topology-only sources cannot be compacted without the caller's vectors. */
    @Test
    public void testTopologyOnlySourcesRequireExternalVectors() throws Exception {
        VectorSimilarityFunction vsf = VectorSimilarityFunction.COSINE;
        List<OnDiskGraphIndex> sources = loadSources(vsf, false, false, "no_vectors");
        var e = assertThrows(IllegalArgumentException.class,
                             () -> new OnDiskGraphIndexCompactor(sources, allLive(), vsf, null));
        assertTrue(e.getMessage(), e.getMessage().contains("sourceVectors"));
    }

    @Test
    public void testExternalVectorsValidation() throws Exception {
        VectorSimilarityFunction vsf = VectorSimilarityFunction.COSINE;
        List<OnDiskGraphIndex> topology = loadSources(vsf, false, false, "validate_topology");
        List<OnDiskGraphIndex> inline = loadSources(vsf, true, false, "validate_inline");
        List<FixedBitSet> live = allLive();
        List<RandomAccessVectorValues> ravvs = sourceRavvs();

        // one entry per source
        assertThrows(IllegalArgumentException.class,
                     () -> new OnDiskGraphIndexCompactor(topology, ravvs.subList(0, 2), null, live, vsf, null));
        // no null entries
        List<RandomAccessVectorValues> withNull = new ArrayList<>(ravvs);
        withNull.set(1, null);
        assertThrows(NullPointerException.class,
                     () -> new OnDiskGraphIndexCompactor(topology, withNull, null, live, vsf, null));
        // a vector for every ordinal of the source
        List<RandomAccessVectorValues> tooFew = new ArrayList<>(ravvs);
        tooFew.set(2, new ListRandomAccessVectorValues(sourceVecs.get(2).subList(0, VECTORS_PER_SOURCE - 1), DIMENSION));
        assertThrows(IllegalArgumentException.class,
                     () -> new OnDiskGraphIndexCompactor(topology, tooFew, null, live, vsf, null));
        // one dimension across the sources
        List<RandomAccessVectorValues> mixed = new ArrayList<>(ravvs);
        mixed.set(1, new ListRandomAccessVectorValues(createRandomVectors(VECTORS_PER_SOURCE, DIMENSION / 2), DIMENSION / 2));
        assertThrows(IllegalArgumentException.class,
                     () -> new OnDiskGraphIndexCompactor(topology, mixed, null, live, vsf, null));
        // the dimension the sources store, when they store one
        List<RandomAccessVectorValues> narrower = new ArrayList<>();
        for (int s = 0; s < NUM_SOURCES; s++)
            narrower.add(new ListRandomAccessVectorValues(createRandomVectors(VECTORS_PER_SOURCE, DIMENSION / 2), DIMENSION / 2));
        assertThrows(IllegalArgumentException.class,
                     () -> new OnDiskGraphIndexCompactor(inline, narrower, null, live, vsf, null));
    }

    // ---- helpers ----

    /** Vectors of a store that has forgotten its deleted ones: reading one fails the test. */
    private static final class LiveOnlyVectors implements RandomAccessVectorValues {
        private final List<VectorFloat<?>> vecs;
        private final FixedBitSet live;

        LiveOnlyVectors(List<VectorFloat<?>> vecs, FixedBitSet live) {
            this.vecs = vecs;
            this.live = live;
        }

        @Override
        public int size() {
            return vecs.size();
        }

        @Override
        public int dimension() {
            return DIMENSION;
        }

        @Override
        public VectorFloat<?> getVector(int node) {
            if (!live.get(node))
                throw new AssertionError("the compactor read deleted node " + node);
            return vecs.get(node);
        }

        @Override
        public boolean isValueShared() {
            return false;
        }

        @Override
        public RandomAccessVectorValues copy() {
            return this;
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
        for (int s = 0; s < NUM_SOURCES; s++)
            sources.add(load(writeSource(sourceVecs.get(s), vsf, storeVectors, fusedPQ, prefix + "_" + s)));
        return sources;
    }

    private OnDiskGraphIndex load(Path path) throws IOException {
        ReaderSupplier rs = ReaderSupplierFactory.open(path);
        readers.add(rs);
        return OnDiskGraphIndex.load(rs);
    }

    private Path writeSource(List<VectorFloat<?>> vecs, VectorSimilarityFunction vsf, boolean storeVectors, boolean fusedPQ, String name) throws IOException {
        return CompactionSources.writeSource(testDirectory.resolve(name), vecs, DIMENSION, vsf, storeVectors, fusedPQ);
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

    /** Every source loses a different tenth of its nodes. */
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

    /**
     * Recall@{@link #TOP_K} of exact-scored searches over the compacted graph against brute force
     * over the live vectors. The output ordinals are the compactor's, so the scoring vectors are
     * laid out by output ordinal.
     */
    private double recall(OnDiskGraphIndex compacted, List<OrdinalMapper> mappers, List<FixedBitSet> live,
                          VectorSimilarityFunction vsf) throws IOException {
        List<VectorFloat<?>> byOrdinal = new ArrayList<>();
        for (int i = 0; i < compacted.size(0); i++)
            byOrdinal.add(null);
        for (int s = 0; s < NUM_SOURCES; s++)
            for (int i = 0; i < VECTORS_PER_SOURCE; i++)
                if (live.get(s).get(i))
                    byOrdinal.set(mappers.get(s).oldToNew(i), sourceVecs.get(s).get(i));
        for (int ord = 0; ord < byOrdinal.size(); ord++)
            assertNotNull("no vector mapped to output ordinal " + ord, byOrdinal.get(ord));
        var ravv = new ListRandomAccessVectorValues(byOrdinal, DIMENSION);

        int hits = 0;
        try (var searcher = new GraphSearcher(compacted)) {
            for (int q = 0; q < NUM_QUERIES; q++) {
                VectorFloat<?> query = createRandomVectors(1, DIMENSION).get(0);
                NodeQueue expected = new NodeQueue(new BoundedLongHeap(TOP_K), NodeQueue.Order.MIN_HEAP);
                for (int ord = 0; ord < byOrdinal.size(); ord++)
                    expected.push(ord, vsf.compare(query, byOrdinal.get(ord)));
                Set<Integer> truth = new HashSet<>();
                while (expected.size() > 0)
                    truth.add(expected.pop());
                var ssp = DefaultSearchScoreProvider.exact(query, vsf, ravv);
                for (var ns : searcher.search(ssp, TOP_K, 50, 0.0f, 0.0f, Bits.ALL).getNodes())
                    if (truth.contains(ns.node))
                        hits++;
            }
        }
        return (double) hits / (NUM_QUERIES * TOP_K);
    }
}
