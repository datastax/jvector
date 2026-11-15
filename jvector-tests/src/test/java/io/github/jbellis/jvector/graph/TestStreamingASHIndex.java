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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.quantization.ASHQueryProjection;
import io.github.jbellis.jvector.quantization.ASHVectors;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.OrdinalReservoir;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;

import static org.junit.Assert.*;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestStreamingASHIndex extends RandomizedTest {
    private static final int DIM = 32;
    private static final int PROJECTED = 24;
    private static final VectorSimilarityFunction DOT = VectorSimilarityFunction.DOT_PRODUCT;

    /** Unit vectors around a shared offset, so the mean (the landmark) is far from zero. */
    private static List<VectorFloat<?>> vectors(int n, long seed) {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var random = new Random(seed);
        var out = new ArrayList<VectorFloat<?>>(n);
        for (int i = 0; i < n; i++) {
            var v = vts.createFloatVector(DIM);
            for (int d = 0; d < DIM; d++) v.set(d, (float) random.nextGaussian() + (d < 4 ? 2f : 0f));
            VectorUtil.l2normalize(v);
            out.add(v);
        }
        return out;
    }

    @SuppressWarnings("deprecation")
    private static Function<BuildScoreProvider, GraphIndexBuilder> builders() {
        return scores -> new GraphIndexBuilder(scores, DIM, 16, 64, 1.2f, 1.2f, false, false);
    }

    private static int encodedBits(int bits) {
        return AsymmetricHashing.HEADER_BITS + PROJECTED * bits;
    }

    /** The exact top {@code k} among ordinals below {@code limit} that {@code accept} allows. */
    private static int[] exactTop(List<VectorFloat<?>> data, int limit, VectorFloat<?> q, int k, Bits accept) {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < limit; i++) if (accept.get(i)) order.add(i);
        order.sort((a, b) -> Float.compare(DOT.compare(q, data.get(b)), DOT.compare(q, data.get(a))));
        int[] top = new int[Math.min(k, order.size())];
        for (int i = 0; i < top.length; i++) top[i] = order.get(i);
        return top;
    }

    private static double recall(StreamingASHIndex index, List<VectorFloat<?>> data, List<VectorFloat<?>> queries,
                                 int k, int rerankK, Bits accept) {
        int hits = 0;
        for (var q : queries) {
            var expected = new HashSet<Integer>();
            for (int o : exactTop(data, data.size(), q, k, accept)) expected.add(o);
            for (var ns : index.search(q, k, rerankK, accept).getNodes()) if (expected.contains(ns.node)) hits++;
        }
        return hits / (double) (k * queries.size());
    }

    /** Recall of the same model's codes, encoded up front and built into a graph in one pass, as batch builds do. */
    private static double batchRecall(AsymmetricHashing model, List<VectorFloat<?>> data, List<VectorFloat<?>> queries,
                                      int k, int rerankK, Bits accept) {
        var ravv = new ListRandomAccessVectorValues(data, DIM);
        var codes = (ASHVectors) model.encodeAll(ravv);
        var scores = BuildScoreProvider.ashBuildScoreProvider(DOT, codes);
        var builder = builders().apply(scores);
        for (int i = 0; i < data.size(); i++) builder.addGraphNode(i, scores.searchProviderFor(i));
        int hits = 0;
        try (var searcher = new GraphSearcher(builder.getGraph())) {
            for (var q : queries) {
                var expected = new HashSet<Integer>();
                for (int o : exactTop(data, data.size(), q, k, accept)) expected.add(o);
                var provider = new DefaultSearchScoreProvider(codes.precomputedScoreFunctionFor(q, DOT), ravv.rerankerFor(q, DOT));
                for (var ns : searcher.search(provider, k, rerankK, 0f, 0f, accept).getNodes())
                    if (expected.contains(ns.node)) hits++;
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return hits / (double) (k * queries.size());
    }

    @Test
    public void reservoirKeepsAUniformSampleOfOrdinals() {
        var small = new OrdinalReservoir(10);
        for (int i = 0; i < 6; i++) small.offer(i);
        assertArrayEquals(new int[]{0, 1, 2, 3, 4, 5}, small.sample());

        var reservoir = new OrdinalReservoir(2000);
        for (int i = 0; i < 200_000; i++) reservoir.offer(i);
        int[] sample = reservoir.sample();
        assertEquals(200_000, reservoir.offered());
        assertTrue(sample.length > 1900 && sample.length <= 2000);
        double mean = Arrays.stream(sample).average().orElse(0);
        assertEquals(100_000, mean, 6000);
    }

    @Test
    public void coldIndexScoresExactlyThenSwitchesToASH() {
        var data = vectors(3000, 1);
        var queries = vectors(30, 2);
        var options = new StreamingASHIndex.Options();
        options.coldStartSize = 256;
        var index = StreamingASHIndex.coldStart(new ListRandomAccessVectorValues(data, DIM), DOT, encodedBits(4), 4,
                builders(), options);
        // Before the model is fixed, the graph is searched with exact scores; with a beam as large as the graph the
        // results are the exact top results, deletions excluded.
        for (int i = 0; i < 100; i++) index.add(i);
        index.markDeleted(3);
        assertFalse(index.hasModel());
        var q = queries.get(0);
        Bits notThree = node -> node != 3;
        int[] expected = exactTop(data, 100, q, 5, notThree);
        var found = index.search(q, 5, 100, Bits.ALL).getNodes();
        for (int i = 0; i < 5; i++) {
            assertEquals(expected[i], found[i].node);
            assertEquals(DOT.compare(q, data.get(expected[i])), found[i].score, 0f);
        }

        // The random projection is fixed on the pool once coldStartSize vectors have arrived; the deletion carries over.
        for (int i = 100; i < data.size(); i++) index.add(i);
        while (index.isTraining()) Thread.onSpinWait();
        assertTrue(index.hasModel());
        assertTrue(index.graph().getDeletedNodes().get(3));
        // The mean kept as vectors arrived covers both phases, deleted vectors included.
        var mean = index.mean();
        for (int d = 0; d < DIM; d++) {
            double sum = 0;
            for (var v : data) sum += v.get(d);
            assertEquals(sum / data.size(), mean.get(d), 1e-5);
        }
        assertEquals(AsymmetricHashing.RANDOM, index.model().optimizer);
        assertFalse("no executor, so no training", index.isChangingModel());
        // The deleted vector stays in the graph, marked, until cleanup removes it.
        assertEquals(data.size(), index.graph().size(0));
        double streamed = recall(index, data, queries, 10, 50, notThree);
        double batch = batchRecall(index.model(), data, queries, 10, 50, notThree);
        assertTrue("cold-started recall " + streamed + " vs batch " + batch, streamed > batch - 0.05);
    }

    @Test
    public void aCallersOwnSearcherWorksAcrossTheFirstModelAndAddsReportTheirBytes() {
        var data = vectors(3000, 11);
        var q = vectors(1, 12).get(0);
        var options = new StreamingASHIndex.Options();
        options.coldStartSize = 256;
        var index = StreamingASHIndex.coldStart(new ListRandomAccessVectorValues(data, DIM), DOT, encodedBits(2), 2,
                builders(), options);
        // A caller's searcher (one that may resume), made once and pointed at the index's current graph per search.
        var searcher = new GraphSearcher(index.graph());

        for (int i = 0; i < 100; i++) assertTrue("an exact-phase insert adds a graph node", index.add(i) > 0);
        searcher.setView(index.graph().getView());
        var exact = searcher.search(index.searchScoreProviderFor(q), 5, 100, 0f, 0f, Bits.ALL).getNodes();
        int[] expected = exactTop(data, 100, q, 5, Bits.ALL);
        for (int i = 0; i < 5; i++) assertEquals(expected[i], exact[i].node);

        int last = data.size() - 1;
        for (int i = 100; i < last; i++) index.add(i);
        while (index.isTraining()) Thread.onSpinWait();
        assertTrue(index.hasModel());
        assertTrue("an insert after the switch adds a graph node and a code",
                index.add(last) > index.codes().bytesPerVector());

        // The same searcher, now on the rescored graph with ASH scores reranked exactly: every score is exact.
        searcher.setView(index.graph().getView());
        var found = searcher.search(index.searchScoreProviderFor(q), 10, 40, 0f, 0f, Bits.ALL).getNodes();
        assertEquals(10, found.length);
        for (int i = 0; i < found.length; i++) {
            assertEquals(DOT.compare(q, data.get(found[i].node)), found[i].score, 1e-6f);
            if (i > 0) assertTrue(found[i - 1].score >= found[i].score);
        }
    }

    @Test
    public void addsReportTheHeapTheIndexHoldsAndNativeMemoryIsReportedApart() {
        var data = vectors(6000, 21);
        var options = new StreamingASHIndex.Options();
        options.coldStartSize = 256;
        options.train = true;
        var index = StreamingASHIndex.coldStart(new ListRandomAccessVectorValues(data, DIM), DOT, encodedBits(2), 2,
                builders(), options);
        long heap = 0, nativeBytes = 0;
        for (int i = 0; i < data.size() - 1; i++) {
            heap += index.add(i);
            nativeBytes += index.takeNativeBytes();
        }
        while (index.isTraining()) Thread.onSpinWait();
        // The first insert after the model is fixed reports the model and the codes of the vectors held at the switch.
        heap += index.add(data.size() - 1);
        nativeBytes += index.takeNativeBytes();
        assertTrue(index.hasModel());

        long held = index.heapBytesUsed();
        assertTrue("reported heap " + heap + " vs held " + held, Math.abs(heap - held) <= held / 10);
        assertEquals("native memory is reported apart from heap", index.nativeBytesUsed(), nativeBytes);
        assertEquals(index.ramBytesUsed(), index.heapBytesUsed() + index.nativeBytesUsed());
    }

    @Test
    public void aColdIndexTrainsITQAsItsFirstModel() {
        var data = vectors(6000, 3);
        var queries = vectors(30, 4);
        var options = new StreamingASHIndex.Options();
        options.coldStartSize = 256;
        options.train = true;
        options.reencodePerInsert = 4;
        options.queryProjection = ASHQueryProjection.FP32;
        options.queryBatchSize = 4;
        var index = StreamingASHIndex.coldStart(new ListRandomAccessVectorValues(data, DIM), DOT, encodedBits(2), 2,
                builders(), options);
        for (int i = 0; i < data.size(); i++) index.add(i);
        // As at a flush: let the training finish. The first model is trained, so no random projection was used and
        // there is no model change to finish.
        while (index.isTraining()) Thread.onSpinWait();
        assertEquals(AsymmetricHashing.ITQ, index.model().optimizer);
        index.prepareForFlush();
        // The trained model projects queries as the options say.
        assertEquals(ASHQueryProjection.FP32, index.model().getQueryProjection());
        assertEquals(4, index.model().getQueryBatchSize());
        assertFalse(index.isChangingModel());
        var codes = index.codes();
        assertEquals(1, codes.modelsInUse().length);
        assertEquals(AsymmetricHashing.ITQ, codes.model(codes.modelsInUse()[0]).optimizer);
        // Every code is now the trained model's code for its vector.
        var model = index.model();
        for (int i = 0; i < data.size(); i += 37) assertEquals(model.encode(data.get(i)), codes.get(i));
        assertEquals(data.size(), codes.snapshot().count());
        // Correctness, not quality: the index must not be worse than one built in one pass with its model. Quality is
        // measured on real data.
        double streamed = recall(index, data, queries, 10, 40, Bits.ALL);
        double floor = batchRecall(model, data, queries, 10, 40, Bits.ALL);
        assertTrue("streamed recall " + streamed + " vs one pass with the same model " + floor, streamed > floor - 0.05);
    }

    @Test
    public void concurrentAddsDeletesAndSearchesStayCorrectThroughAModelChange() throws Exception {
        var data = vectors(6000, 5);
        var queries = vectors(40, 6);
        var options = new StreamingASHIndex.Options();
        options.coldStartSize = 256;
        options.train = true;
        options.reencodePerInsert = 2;
        // The first model is fixed while inserts run, and a retrain then moves the index to a second one.
        options.retrainGrowth = 2;
        var index = StreamingASHIndex.coldStart(new ListRandomAccessVectorValues(data, DIM), DOT, encodedBits(4), 4,
                builders(), options);
        var next = new AtomicInteger();
        var added = new AtomicInteger(); // every ordinal below this has been added
        var addedSet = ConcurrentHashMap.<Integer>newKeySet();
        Set<Integer> deleted = ConcurrentHashMap.newKeySet();
        var failure = new AtomicReference<Throwable>();
        var searches = new AtomicInteger();
        var threads = new ArrayList<Thread>();
        for (int t = 0; t < 4; t++) {
            threads.add(new Thread(() -> {
                try {
                    for (int i; (i = next.getAndIncrement()) < data.size(); ) {
                        index.add(i);
                        addedSet.add(i);
                        synchronized (added) {
                            while (addedSet.contains(added.get())) added.incrementAndGet();
                        }
                    }
                } catch (Throwable e) { failure.compareAndSet(null, e); }
            }));
        }
        threads.add(new Thread(() -> {
            try {
                var random = new Random(7);
                while (next.get() < data.size()) {
                    int limit = added.get();
                    // Keep deletions to about a tenth of what has been added, as in a live index.
                    if (limit < 300 || deleted.size() >= limit / 10) {
                        Thread.onSpinWait();
                        continue;
                    }
                    int victim = random.nextInt(limit);
                    // Record a deletion only once it is done, so a search that sees it recorded began after it.
                    if (deleted.contains(victim)) continue;
                    index.markDeleted(victim);
                    deleted.add(victim);
                }
            } catch (Throwable e) { failure.compareAndSet(null, e); }
        }));
        for (int t = 0; t < 2; t++) {
            threads.add(new Thread(() -> {
                try {
                    var random = new Random();
                    while (next.get() < data.size()) {
                        var gone = new HashSet<>(deleted); // deleted before this search began
                        var q = queries.get(random.nextInt(queries.size()));
                        var nodes = index.search(q, 10, 40, Bits.ALL).getNodes();
                        float previous = Float.POSITIVE_INFINITY;
                        for (var ns : nodes) {
                            assertTrue("result " + ns.node + " was added", ns.node >= 0 && ns.node < data.size());
                            assertFalse("result " + ns.node + " was deleted before the search", gone.contains(ns.node));
                            // Reranked from the source vectors: every score is the exact score.
                            assertEquals(DOT.compare(q, data.get(ns.node)), ns.score, 0f);
                            assertTrue("results best first", ns.score <= previous);
                            previous = ns.score;
                        }
                        searches.incrementAndGet();
                    }
                } catch (Throwable e) { failure.compareAndSet(null, e); }
            }));
        }
        for (var t : threads) t.start();
        for (var t : threads) t.join();
        if (failure.get() != null) throw new AssertionError(failure.get());
        assertTrue(searches.get() > 0);
        assertTrue("deletions ran: " + deleted.size(), deleted.size() > 100 && deleted.size() <= data.size() / 10 + 1);
        while (index.isTraining()) Thread.onSpinWait();
        assertEquals(AsymmetricHashing.ITQ, index.model().optimizer);

        index.finishModelChange();
        assertEquals(1, index.codes().modelsInUse().length);
        assertCorrectResults(index, data, queries, deleted);

        // Cleanup removes the deleted nodes; searches stay correct. Graph quality is measured on real data.
        index.builder().cleanup();
        assertEquals(data.size() - deleted.size(), index.graph().size(0));
        assertCorrectResults(index, data, queries, deleted);
    }

    /** Full results, none deleted, each with its exact score, best first. */
    private static void assertCorrectResults(StreamingASHIndex index, List<VectorFloat<?>> data,
                                             List<VectorFloat<?>> queries, Set<Integer> deleted) {
        for (var q : queries) {
            var nodes = index.search(q, 10, 40, Bits.ALL).getNodes();
            assertEquals(10, nodes.length);
            float previous = Float.POSITIVE_INFINITY;
            for (var ns : nodes) {
                assertFalse(deleted.contains(ns.node));
                assertEquals(DOT.compare(q, data.get(ns.node)), ns.score, 0f);
                assertTrue(ns.score <= previous);
                previous = ns.score;
            }
        }
    }

    @Test
    public void aWarmIndexStartsWithTheTrainedModel() {
        var data = vectors(4000, 8);
        var queries = vectors(30, 9);
        var options = new StreamingASHIndex.Options();
        options.coldStartSize = 256;
        var first = StreamingASHIndex.coldStart(new ListRandomAccessVectorValues(data, DIM), DOT, encodedBits(2), 2,
                builders(), options);
        for (int i = 0; i < data.size(); i++) first.add(i);
        var model = first.trainNextModel(AsymmetricHashing.ITQ, ForkJoinPool.commonPool());
        assertEquals(AsymmetricHashing.ITQ, model.optimizer);
        var warm = StreamingASHIndex.withModel(new ListRandomAccessVectorValues(data, DIM), DOT, model, builders(),
                new StreamingASHIndex.Options());
        assertTrue(warm.hasModel());
        for (int i = 0; i < data.size(); i++) warm.add(i);
        double warmRecall = recall(warm, data, queries, 10, 40, Bits.ALL);
        double batch = batchRecall(model, data, queries, 10, 40, Bits.ALL);
        assertTrue("warm recall " + warmRecall + " vs batch " + batch, warmRecall > batch - 0.05);
    }
}
