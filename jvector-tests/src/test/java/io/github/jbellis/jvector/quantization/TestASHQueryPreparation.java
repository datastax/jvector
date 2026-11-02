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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.MockVectorValues;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.*;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.Int2ObjectHashMap;
import org.junit.Test;

import java.lang.reflect.Field;
import java.nio.file.Files;
import java.util.EnumMap;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntFunction;

import static org.junit.Assert.*;

/** Query ownership, fallback formulas, and real on-disk scratch reuse. */
public class TestASHQueryPreparation {
    private static final VectorSimilarityFunction DOT = VectorSimilarityFunction.DOT_PRODUCT;

    private static VectorFloat<?>[] inputs() {
        var type = VectorizationProvider.getInstance().getVectorTypeSupport();
        var vectors = new VectorFloat<?>[97];
        Random random = new Random(69381);
        for (int n = 0; n < vectors.length; n++) {
            vectors[n] = type.createFloatVector(64);
            for (int d = 0; d < 64; d++) vectors[n].set(d, random.nextFloat() - 0.5f);
            VectorUtil.l2normalize(vectors[n]);
        }
        return vectors;
    }

    private static AsymmetricHashing model(VectorFloat<?>[] vectors, int bits) throws Exception {
        return AsymmetricHashing.initialize(MockVectorValues.fromValues(vectors), AsymmetricHashing.RANDOM,
                AsymmetricHashing.HEADER_BITS + 33 * bits, 2, bits);
    }

    @Test
    public void fastScanSkipsGenericTermsAndStandaloneScorersStayIndependent() throws Exception {
        var vectors = inputs();
        for (int bits : new int[]{1, 2, 3, 4, 9}) {
            var ash = model(vectors, bits);
            var scorer = new ASHScorer(ash);
            var first = scorer.precomputeQuery(vectors[0]);
            boolean fastScan = bits == 2 || bits == 4;
            assertEquals(fastScan ? 0 : 66, first.tildeQPool.length);
            assertEquals(fastScan ? 0 : 2, first.sumTildeQByLandmark.length);
            float[] projection = first.qProj.clone();
            float[] landmarkDots = first.dotQMuByLandmark.clone();
            // Publicly retained prepared state must never borrow another query's arrays.
            var retained = bits == 1 || fastScan ? new ASHNeighborhoodScorer(ash, vectors[0], 8, 8) : null;
            var second = scorer.precomputeQuery(vectors[1]);
            assertNotSame(first.qProj, second.qProj);
            assertArrayEquals(projection, first.qProj, 0f);
            assertArrayEquals(landmarkDots, first.dotQMuByLandmark, 0f);
            if (retained != null) {
                float[] lut = retained.queryLut().clone();
                new ASHNeighborhoodScorer(ash, vectors[1], 8, 8);
                assertArrayEquals(lut, retained.queryLut(), 0f);
            }
        }
    }

    private static Object field(Object object, String name) throws Exception {
        Field f = object.getClass().getDeclaredField(name);
        f.setAccessible(true);
        return f.get(object);
    }

    @Test
    public void fusedScratchReusesArraysAndRejectsStaleDecoders() throws Exception {
        var vectors = inputs();
        for (int bits : new int[]{1, 2, 4}) {
            var ash = model(vectors, bits);
            var encoded = ash.encodeAll(MockVectorValues.fromValues(vectors), ForkJoinPool.commonPool());
            int degree = 8, block = 8;
            byte[] bytes = new byte[FusedASHLayout.featureSize(degree, ash.quantizedDim, bits, block)];
            for (int n = 0; n < degree; n++) FusedASHLayout.packForFused(bytes, 0, n, encoded.get(n), ash, block);
            var source = new FusedASHDecoder.PackedNeighborhoods() {
                public void readInto(int origin, byte[] dest) { System.arraycopy(bytes, 0, dest, 0, bytes.length); }
                public int maxDegree() { return degree; }
                public int featureSize() { return bytes.length; }
            };
            var cache = new Int2ObjectHashMap<FusedFeature.InlineSource>();
            cache.put(0, () -> 0L);
            var scratch = new FusedASHDecoder.QueryScratch(ash, degree, block);
            Object[] storage = {field(scratch, "input"), field(scratch, "prepared"), field(scratch, "lut"),
                    field(scratch, "neighborhoodBytes"), field(scratch, "scores")};
            FusedASHDecoder previous = null;
            for (int q = 0; q < 12; q++) {
                var decoder = FusedASHDecoder.newDecoder(source, ash, cache, ignored -> encoded.get(0),
                        vectors[q], scratch, block, DOT);
                var reference = new ASHScorer(ash).scalarScoreFunctionFor(vectors[q]);
                assertEquals(ASHScorer.toSimilarity(reference.similarityTo(encoded.get(0))), decoder.similarityTo(0), 0.00005f);
                decoder.enableSimilarityToNeighbors(1);
                for (int lane = 0; lane < degree; lane++) assertEquals(
                        ASHScorer.toSimilarity(reference.similarityTo(encoded.get(lane))), decoder.similarityToNeighbor(1, lane), 0.002f);
                // Re-entering a neighborhood for the same query is valid (search resume).
                decoder.enableSimilarityToNeighbors(2);
                decoder.enableSimilarityToNeighbors(1);
                if (previous != null) {
                    var stale = previous;
                    assertThrows(IllegalStateException.class, () -> stale.similarityTo(0));
                    assertThrows(IllegalStateException.class, () -> stale.enableSimilarityToNeighbors(1));
                    assertThrows(IllegalStateException.class, () -> stale.similarityToNeighbor(1, 0));
                }
                previous = decoder;
                String[] names = {"input", "prepared", "lut", "neighborhoodBytes", "scores"};
                for (int n = 0; n < names.length; n++) assertSame(storage[n], field(scratch, names[n]));
            }
            var last = previous;
            var wrong = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(63);
            assertThrows(IllegalArgumentException.class, () -> FusedASHDecoder.newDecoder(source, ash, cache,
                    ignored -> encoded.get(0), wrong, scratch, block, DOT));
            assertThrows(NullPointerException.class, () -> FusedASHDecoder.newDecoder(source, ash, cache,
                    ignored -> encoded.get(0), null, scratch, block, DOT));
            // Input validation happens before invalidation or scratch mutation.
            last.similarityTo(0);
            var other = model(vectors, bits);
            assertThrows(IllegalArgumentException.class, () -> FusedASHDecoder.newDecoder(source, other, cache,
                    ignored -> encoded.get(0), vectors[0], scratch, block, DOT));
            last.similarityTo(0);
            String previousMode = System.getProperty("jvector.ash.singleKernel");
            try {
                System.setProperty("jvector.ash.singleKernel", "invalid-mode");
                assertThrows(IllegalArgumentException.class, () -> FusedASHDecoder.newDecoder(source, ash, cache,
                        ignored -> encoded.get(0), vectors[0], scratch, block, DOT));
                // Failure after overwriting preparation invalidates the previous decoder.
                assertThrows(IllegalStateException.class, () -> last.similarityTo(0));
            } finally {
                if (previousMode == null) System.clearProperty("jvector.ash.singleKernel");
                else System.setProperty("jvector.ash.singleKernel", previousMode);
            }
            var recovered = FusedASHDecoder.newDecoder(source, ash, cache, ignored -> encoded.get(0),
                    vectors[0], scratch, block, DOT);
            recovered.enableSimilarityToNeighbors(1);
            recovered.similarityToNeighbor(1, 0);
        }
    }

    @Test
    public void onDiskViewsOwnIndependentScratchOnSameThread() throws Exception {
        var vectors = inputs();
        for (int bits : new int[]{1, 2, 4}) {
            var ash = model(vectors, bits);
            var encoded = ash.encodeAll(MockVectorValues.fromValues(vectors), ForkJoinPool.commonPool());
            var graph = new TestUtil.RandomlyConnectedGraphIndex(vectors.length, 8, new Random(9182));
            var path = Files.createTempFile("ash-query-scratch-", ".index");
            try {
                var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
                try (var graphView = graph.getView(); var writer = new OnDiskGraphIndexWriter.Builder(graph, path)
                        .with(new FusedASH(8, ash, 8)).with(new InlineVectors(64)).build()) {
                    suppliers.put(FeatureId.FUSED_ASH, node -> new FusedASH.State(graphView, encoded, node));
                    suppliers.put(FeatureId.INLINE_VECTORS, node -> new InlineVectors.State(vectors[node]));
                    writer.write(suppliers);
                }
                try (var reader = new SimpleMappedReader.Supplier(path); var loaded = OnDiskGraphIndex.load(reader, 0);
                     var firstView = loaded.getView(); var secondView = loaded.getView()) {
                    var first = firstView.approximateScoreFunctionFor(vectors[0], DOT);
                    first.enableSimilarityToNeighbors(0);
                    float expected = first.similarityToNeighbor(0, 0);
                    var second = secondView.approximateScoreFunctionFor(vectors[1], DOT);
                    second.enableSimilarityToNeighbors(0);
                    assertEquals(expected, first.similarityToNeighbor(0, 0), 0f);
                    first.enableSimilarityToNeighbors(1);
                    first.enableSimilarityToNeighbors(0);
                    assertEquals(expected, first.similarityToNeighbor(0, 0), 0f);
                    Object firstScratch = field(firstView, "featureScratch");
                    firstView.approximateScoreFunctionFor(vectors[2], DOT);
                    assertSame(firstScratch, field(firstView, "featureScratch"));
                    assertThrows(IllegalStateException.class, () -> first.similarityToNeighbor(0, 0));
                    second.similarityToNeighbor(0, 0);
                    try (var searcher = new GraphSearcher(loaded)) {
                        var searchView = (OnDiskGraphIndex.View) searcher.getView();
                        var approximate = searchView.approximateScoreFunctionFor(vectors[3], DOT);
                        var provider = new DefaultSearchScoreProvider(approximate,
                                searchView.rerankerFor(vectors[3], DOT), true);
                        approximate.enableSimilarityToNeighbors(0);
                        float original = approximate.similarityToNeighbor(0, 0);
                        assertEquals(5, searcher.search(provider, 5, 10, 0f, 0f, Bits.ALL).getNodes().length);
                        assertTrue(searcher.resume(5, 10).getNodes().length > 0);
                        approximate.enableSimilarityToNeighbors(0);
                        assertEquals(original, approximate.similarityToNeighbor(0, 0), 0f);
                    }
                }
            } finally { Files.deleteIfExists(path); }
        }
    }
}
