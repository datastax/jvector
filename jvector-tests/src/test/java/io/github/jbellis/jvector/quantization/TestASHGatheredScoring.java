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

import io.github.jbellis.jvector.graph.MockVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Assume;
import org.junit.Test;

import java.util.Random;
import java.util.concurrent.ForkJoinPool;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** A batch of nodes scored as one gathered neighborhood gets the scores the nodes get one at a time. */
public class TestASHGatheredScoring {
    private static VectorFloat<?>[] unitVectors(int count, int dimension, long seed) {
        Random random = new Random(seed);
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        VectorFloat<?>[] result = new VectorFloat<?>[count];
        for (int i = 0; i < count; i++) {
            result[i] = types.createFloatVector(dimension);
            double norm = 0;
            for (int d = 0; d < dimension; d++) {
                float v = (float) random.nextGaussian();
                result[i].set(d, v);
                norm += v * v;
            }
            for (int d = 0; d < dimension; d++) result[i].set(d, (float) (result[i].get(d) / Math.sqrt(norm)));
        }
        return result;
    }

    private static final String GATHERED_SCORING = "jvector.ash.gatheredScoring";

    @Test
    public void gatheredScoringIsOffByDefault() throws Exception {
        // Gathered scoring is experimental: without the property, nodes are scored one at a time.
        var base = unitVectors(500, 64, 3);
        var ash = AsymmetricHashing.initialize(MockVectorValues.fromValues(base), AsymmetricHashing.ITQ,
                AsymmetricHashing.HEADER_BITS + 2 * 34, 1, 2);
        var codes = ash.encodeAll(MockVectorValues.fromValues(base), ForkJoinPool.commonPool());
        assertFalse(codes.scoreFunctionFor(unitVectors(1, 64, 4)[0], VectorSimilarityFunction.DOT_PRODUCT).supportsBatchScoring());
    }

    @Test
    public void gatheredBatchesScoreLikeSingleNodes() throws Exception {
        Assume.assumeTrue("needs a backend with neighborhood scoring",
                VectorizationProvider.getInstance().getVectorUtilSupport().supportsAshNeighborhoodScoring());
        System.setProperty(GATHERED_SCORING, "true");
        try {
            gatheredBatchesScoreLikeSingleNodesWhenEnabled();
        } finally {
            System.clearProperty(GATHERED_SCORING);
        }
    }

    private void gatheredBatchesScoreLikeSingleNodesWhenEnabled() throws Exception {
        for (int dimension : new int[] {64, 384, 1536}) {
            int projected = dimension / 2 + 2;
            var base = unitVectors(2000, dimension, dimension);
            var ash = AsymmetricHashing.initialize(MockVectorValues.fromValues(base), AsymmetricHashing.ITQ,
                    AsymmetricHashing.HEADER_BITS + 2 * projected, 1, 2);
            var codes = ash.encodeAll(MockVectorValues.fromValues(base), ForkJoinPool.commonPool());
            var query = unitVectors(1, dimension, 99)[0];
            var f = codes.scoreFunctionFor(query, VectorSimilarityFunction.DOT_PRODUCT);
            assertTrue(f.supportsBatchScoring());
            var random = new Random(7);
            for (int count : new int[] {1, 3, 4, 17, 32, 33, 70}) {
                int[] nodes = new int[count];
                for (int i = 0; i < count; i++) nodes[i] = random.nextInt(base.length);
                float[] batch = new float[count];
                f.similarityTo(nodes, count, batch);
                for (int i = 0; i < count; i++) {
                    assertEquals("dimension " + dimension + " count " + count + " node " + nodes[i],
                            f.similarityTo(nodes[i]), batch[i], 1e-4f);
                }
            }
        }
    }
}
