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
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * A model keeps a centroid-pair table for diversity scoring only when it is small, and mutable and immutable
 * vectors over a model score the same pairs the same way with or without it.
 */
public class TestPQPairTable {
    private static final int NODES = 400;
    private static final int DIMENSION = 64;

    private static final VectorSimilarityFunction[] FUNCTIONS = {
            VectorSimilarityFunction.DOT_PRODUCT, VectorSimilarityFunction.COSINE, VectorSimilarityFunction.EUCLIDEAN};

    private static ProductQuantization model(ListRandomAccessVectorValues ravv, int subspaces) {
        return ProductQuantization.compute(ravv, subspaces, 32, false);
    }

    private static ListRandomAccessVectorValues vectors() {
        var random = new Random(3);
        var list = new ArrayList<VectorFloat<?>>();
        for (int i = 0; i < NODES; i++) list.add(TestUtil.randomVector(random, DIMENSION));
        return new ListRandomAccessVectorValues(list, DIMENSION);
    }

    @Test
    public void tableSizeCutoffAndSharing() {
        var pq = model(vectors(), 16);
        // M blocks of k * (k + 1) / 2 floats
        assertEquals(16L * 32 * 33 / 2 * Float.BYTES, pq.pairTableBytes());

        assertNull("over budget, so no table", pq.pairTableIfSmall(VectorSimilarityFunction.DOT_PRODUCT, pq.pairTableBytes() - 1));
        var dot = pq.pairTableIfSmall(VectorSimilarityFunction.DOT_PRODUCT, pq.pairTableBytes());
        assertNotNull("exactly at the budget is allowed", dot);
        assertEquals(pq.pairTableBytes() / Float.BYTES, dot.length());
        assertSame("built once", dot, pq.pairTableIfSmall(VectorSimilarityFunction.DOT_PRODUCT));
        assertSame("cosine uses the dot-product table", dot, pq.pairTableIfSmall(VectorSimilarityFunction.COSINE));
        assertNotSame(dot, pq.pairTableIfSmall(VectorSimilarityFunction.EUCLIDEAN));
    }

    @Test
    public void defaultBudgetCoversTheShapesThatBenefit() {
        // 128 subspaces of 256 clusters (cohere mFactor 8) is about 16 MB, 512 subspaces about 64 MB
        long perSubspace = 256L * 257 / 2 * Float.BYTES;
        assertTrue(128 * perSubspace <= ProductQuantization.PAIR_TABLE_MAX_BYTES);
        assertTrue(96 * perSubspace <= ProductQuantization.PAIR_TABLE_MAX_BYTES);
        assertTrue(256 * perSubspace > ProductQuantization.PAIR_TABLE_MAX_BYTES);
        assertTrue(512 * perSubspace > ProductQuantization.PAIR_TABLE_MAX_BYTES);
    }

    @Test
    public void tableCountsTowardTheModelsRetainedBytesOnceBuilt() {
        var pq = model(vectors(), 16);
        long before = pq.ramBytesUsed();
        var table = pq.pairTableIfSmall(VectorSimilarityFunction.DOT_PRODUCT);
        assertEquals(before + table.ramBytesUsed(), pq.ramBytesUsed());
    }

    @Test
    public void tableAndCodebookScoresAgreeForMutableAndImmutableVectors() {
        var ravv = vectors();
        for (int subspaces : new int[] {64, 16, 4}) {
            var pq = model(ravv, subspaces);
            var immutable = (PQVectors) pq.encodeAll(ravv, ForkJoinPool.commonPool());
            var mutable = new MutablePQVectors(pq);
            for (int i = 0; i < NODES; i++) mutable.encodeAndSet(i, ravv.getVector(i));
            var layout = pq.codebookLayout();
            int clusters = pq.getClusterCount();

            for (var vsf : FUNCTIONS) {
                assertNotNull(pq.pairTableIfSmall(vsf));
                var random = new Random(5);
                for (int trial = 0; trial < 50; trial++) {
                    int a = random.nextInt(NODES), b = random.nextInt(NODES);
                    float viaImmutable = immutable.diversityFunctionFor(a, vsf).similarityTo(b);
                    float viaMutable = mutable.diversityFunctionFor(a, vsf).similarityTo(b);
                    assertEquals(vsf + " mutable and immutable, " + subspaces + " subspaces", viaImmutable, viaMutable, 0f);

                    // the same pair scored straight from the codebooks, which a model over budget does
                    var codeA = immutable.get(a);
                    var codeB = immutable.get(b);
                    boolean dot = vsf != VectorSimilarityFunction.EUCLIDEAN;
                    float sum = VectorUtil.pqCodePairSimilarity(dot, layout.codebooks, clusters, layout.subspaceOffsets,
                            layout.subspaceWidths, layout.minWidth, layout.maxWidth, layout.dimBase, codeA, 0, codeB, 0);
                    float expected;
                    switch (vsf) {
                        case DOT_PRODUCT:
                            expected = (1 + sum) / 2;
                            break;
                        case COSINE:
                            float na = pq.reconstructedSquaredNorm(codeA, 0), nb = pq.reconstructedSquaredNorm(codeB, 0);
                            expected = (1 + sum / (float) Math.sqrt(na * nb)) / 2;
                            break;
                        default:
                            expected = 1 / (1 + sum);
                    }
                    assertEquals(vsf + " table vs codebooks, " + subspaces + " subspaces, nodes " + a + "," + b,
                            expected, viaImmutable, 1e-4f);
                }
            }
        }
    }
}
