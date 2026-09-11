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
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.MockVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.*;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;
import java.nio.file.Files;
import java.util.EnumMap;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntFunction;
import static org.junit.Assert.*;

/** Search-level regression: raw negative dot products must not strand hierarchy descent. */
public class TestASHGraphSearch {
    @Test
    public void negativeEntryReachesBaseLayerWithStandaloneAndFusedScoring() throws Exception {
        String single = System.getProperty("jvector.ash.singleKernel");
        String block = System.getProperty("jvector.ash.blockKernel");
        try {
            var provider = VectorizationProvider.getInstance();
            var vts = provider.getVectorTypeSupport();
            VectorFloat<?>[] input = new VectorFloat<?>[17];
            Random random = new Random(92201);
            for (int n = 0; n < input.length; n++) {
                input[n] = vts.createFloatVector(64);
                for (int d = 0; d < 64; d++) input[n].set(d, random.nextFloat() - 0.5f);
                VectorUtil.l2normalize(input[n]);
            }
            var query = input[0].copy();
            for (int d = 0; d < query.length(); d++) query.set(d, -query.get(d));
            var values = MockVectorValues.fromValues(input);
            var graph = new TestUtil.FullyConnectedGraphIndex(0, List.of(input.length, 1)) {
                @Override public List<Integer> maxDegrees() { return List.of(16, 0); }
                @Override public int getDimension() { return 64; }
            };
            boolean simd = provider.getVectorUtilSupport().supportsAshLutScoring();
            for (int bits : new int[]{1, 2, 4}) {
                var ash = AsymmetricHashing.initialize(values, AsymmetricHashing.RANDOM,
                        AsymmetricHashing.HEADER_BITS + 33 * bits, 1, bits);
                var encoded = ash.encodeAll(values, ForkJoinPool.commonPool());
                var path = Files.createTempFile("ash-negative-entry-", ".index");
                try {
                    var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
                    try (var view = graph.getView(); var writer = new OnDiskGraphIndexWriter.Builder(graph, path)
                            .with(new FusedASH(graph.maxDegree(), ash, 32))
                            .with(new InlineVectors(64)).build()) {
                        suppliers.put(FeatureId.FUSED_ASH, n -> new FusedASH.State(view, encoded, n));
                        suppliers.put(FeatureId.INLINE_VECTORS, n -> new InlineVectors.State(input[n]));
                        writer.write(suppliers);
                    }
                    try (var reader = new SimpleMappedReader.Supplier(path);
                         var disk = OnDiskGraphIndex.load(reader, 0)) {
                        for (String mode : simd ? new String[]{"scalar", "auto", "simd"} : new String[]{"scalar", "auto"}) {
                            System.setProperty("jvector.ash.singleKernel", mode);
                            System.setProperty("jvector.ash.blockKernel", mode);
                            var raw = new ASHScorer(ash).scoreFunctionFor(query, VectorSimilarityFunction.DOT_PRODUCT);
                            assertTrue("Regression requires a negative entry score", raw.similarityTo(encoded.get(0)) < 0f);
                            var standalone = encoded.precomputedScoreFunctionFor(query, VectorSimilarityFunction.DOT_PRODUCT);
                            try (var searcher = new GraphSearcher(graph)) {
                                searcher.usePruning(false);
                                var scores = new DefaultSearchScoreProvider(standalone,
                                        n -> VectorSimilarityFunction.DOT_PRODUCT.compare(query, input[n]));
                                assertEquals(10, searcher.search(scores, 10, 10, 0f, 0f, Bits.ALL).getNodes().length);
                            }
                            for (int n = 0; n < input.length; n++) {
                                assertEquals(Math.max(0f, (1f + raw.similarityTo(encoded.get(n))) / 2f),
                                        standalone.similarityTo(n), 0.00005f);
                            }
                            try (var searcher = new GraphSearcher(disk)) {
                                searcher.usePruning(false);
                                var view = (OnDiskGraphIndex.View) searcher.getView();
                                var fused = view.approximateScoreFunctionFor(query, VectorSimilarityFunction.DOT_PRODUCT);
                                assertEquals(standalone.similarityTo(0), fused.similarityTo(0), 0.00005f);
                                var scores = new DefaultSearchScoreProvider(fused,
                                        view.rerankerFor(query, VectorSimilarityFunction.DOT_PRODUCT));
                                assertEquals(10, searcher.search(scores, 10, 10, 0f, 0f, Bits.ALL).getNodes().length);
                                fused.enableSimilarityToNeighbors(0);
                                var neighbors = view.getNeighborsIterator(0, 0);
                                for (int lane = 0; neighbors.hasNext(); lane++) {
                                    assertEquals(standalone.similarityTo(neighbors.nextInt()),
                                            fused.similarityToNeighbor(0, lane), 0.002f);
                                }
                            }
                        }
                    }
                } finally { Files.deleteIfExists(path); }
            }
        } finally {
            restore("jvector.ash.singleKernel", single);
            restore("jvector.ash.blockKernel", block);
        }
    }

    @Test
    public void dotProductSimilarityHasNonnegativeLowerBound() {
        assertEquals(0f, ASHScorer.toSimilarity(-1.25f), 0f);
        assertEquals(0f, ASHScorer.toSimilarity(-1f), 0f);
        assertEquals(0.375f, ASHScorer.toSimilarity(-0.25f), 0f);
        assertEquals(0.5f, ASHScorer.toSimilarity(0f), 0f);
        assertEquals(1f, ASHScorer.toSimilarity(1f), 0f);
    }

    private static void restore(String name, String value) {
        if (value == null) System.clearProperty(name); else System.setProperty(name, value);
    }
}
