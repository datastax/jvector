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
package io.github.jbellis.jvector.graph.similarity;

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.disk.SimpleWriter;
import io.github.jbellis.jvector.graph.FlatVectorValues;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Top-K searches stop adaptively for every similarity function a provider names, and dot-product searches know an
 * upper bound on the dot products they score, whatever the vectors' lengths.
 */
public class TestAdaptiveStoppingDefault {
    @Test
    public void providersNameTheSimilarityTheirScoresMeasure() {
        var random = new Random(3);
        var vectors = new ListRandomAccessVectorValues(List.of(TestUtil.randomVector(random, 8),
                TestUtil.randomVector(random, 8)), 8);
        var query = TestUtil.randomVector(random, 8);
        for (var similarity : VectorSimilarityFunction.values()) {
            assertEquals(similarity, DefaultSearchScoreProvider.exact(query, similarity, vectors).adaptiveSimilarity());
            var sf = DefaultSearchScoreProvider.exact(query, similarity, vectors).scoreFunction();
            assertEquals(similarity, DefaultSearchScoreProvider.forSimilarity(sf, null, similarity).adaptiveSimilarity());
            // A provider that cannot tell what its scores measure keeps the standard rule.
            assertNull(new DefaultSearchScoreProvider(sf).adaptiveSimilarity());
            assertNull(new DefaultSearchScoreProvider(sf, null).adaptiveSimilarity());
            // Graph construction keeps the standard rule for full-precision scores.
            assertNull(BuildScoreProvider.randomAccessScoreProvider(vectors, similarity)
                    .searchProviderFor(query).adaptiveSimilarity());
        }
    }

    @Test
    public void dotProductBoundsHoldForVectorsOfAnyLength() throws Exception {
        var random = new Random(17);
        int n = 500, dim = 32;
        List<VectorFloat<?>> base = new ArrayList<>();
        float maxNorm = 0;
        for (int i = 0; i < n; i++) {
            var v = TestUtil.randomVector(random, dim);
            VectorUtil.scale(v, (float) Math.pow(10, -1 + 2 * random.nextDouble()));
            base.add(v);
            maxNorm = Math.max(maxNorm, (float) Math.sqrt(VectorUtil.dotProduct(v, v)));
        }
        var ravv = new ListRandomAccessVectorValues(base, dim);
        var query = TestUtil.randomVector(random, dim);
        VectorUtil.scale(query, 3f);
        float queryNorm = (float) Math.sqrt(VectorUtil.dotProduct(query, query));
        var dot = VectorSimilarityFunction.DOT_PRODUCT;

        // A list of vectors does not track lengths, so dot-product searches over it keep the standard rule.
        assertTrue(Float.isNaN(DefaultSearchScoreProvider.exact(query, dot, ravv).dotProductBound()));

        // The flat store records the largest length as rows are added; empty, it has none.
        assertTrue(Float.isNaN(FlatVectorValues.appendable(dim).maxVectorNorm()));
        var flat = FlatVectorValues.appendable(dim);
        for (int i = 0; i < n; i++) flat.add(i, base.get(i));
        assertEquals(maxNorm, flat.maxVectorNorm(), maxNorm * 1e-6f);
        assertEquals(queryNorm * maxNorm, DefaultSearchScoreProvider.exact(query, dot, flat).dotProductBound(),
                queryNorm * maxNorm * 1e-5f);
        assertEquals(maxNorm, FlatVectorValues.copyOf(ravv).maxVectorNorm(), maxNorm * 1e-6f);

        // A caller that knows the bound can supply it.
        var exact = DefaultSearchScoreProvider.exact(query, dot, ravv).scoreFunction();
        assertEquals(queryNorm * maxNorm,
                new DefaultSearchScoreProvider(exact, null, dot, queryNorm * maxNorm).dotProductBound(), 0f);

        // PQ bounds every dot product it scores, from its reconstructions.
        for (boolean center : new boolean[] {false, true}) {
            var pq = ProductQuantization.compute(ravv, 8, 32, center);
            var pqv = pq.encodeAll(ravv, ForkJoinPool.commonPool());
            var sf = pqv.precomputedScoreFunctionFor(query, dot);
            float bound = sf.dotProductBound();
            assertTrue(Float.isFinite(bound));
            for (int i = 0; i < n; i++) {
                float estimate = 2 * sf.similarityTo(i) - 1;
                assertTrue("PQ center " + center + " node " + i + ": " + estimate + " > " + bound,
                        estimate <= bound * (1 + 1e-5f) + 1e-5f);
            }
        }

        // ASH records the largest length it encodes and keeps it through a save and load.
        var ash = AsymmetricHashing.initialize(ravv, AsymmetricHashing.RANDOM, AsymmetricHashing.HEADER_BITS + 32 * 2, 1, 2);
        var ashv = ash.encodeAll(ravv, ForkJoinPool.commonPool());
        assertEquals(maxNorm, ash.maxEncodedNorm(), maxNorm * 1e-4f);
        assertEquals(queryNorm * ash.maxEncodedNorm(), ashv.precomputedScoreFunctionFor(query, dot).dotProductBound(),
                queryNorm * maxNorm * 1e-5f);
        File file = File.createTempFile("ash-norm", ".model");
        file.deleteOnExit();
        try (var writer = new SimpleWriter(file.toPath())) {
            ash.write(writer, OnDiskGraphIndex.CURRENT_VERSION);
        }
        assertEquals(ash.compressorSize(), file.length());
        try (var readerSupplier = new SimpleMappedReader.Supplier(file.toPath())) {
            assertEquals(ash.maxEncodedNorm(), AsymmetricHashing.load(readerSupplier.get()).maxEncodedNorm(), 0f);
        }
    }
}
