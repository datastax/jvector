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
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assume.assumeTrue;

/** Concurrent first queries on a fresh 1-bit collection must score exactly like a single query. */
public class TestASHPackedBitsRace {
    private static final String BLOCK = "jvector.ash.blockKernel";
    private static final float TOLERANCE = 1e-5f;

    @Test
    public void concurrentFirstBlockScorersSeeFullyPackedBits() throws Exception {
        assumeTrue(VectorizationProvider.getInstance().getVectorUtilSupport().supportsAshMaskedLoad());
        String old = System.getProperty(BLOCK);
        System.setProperty(BLOCK, "simd");
        ExecutorService pool = Executors.newFixedThreadPool(2);
        try {
            var type = VectorizationProvider.getInstance().getVectorTypeSupport();
            int n = 20_000;
            VectorFloat<?>[] values = new VectorFloat<?>[n];
            Random random = new Random(5512);
            for (int i = 0; i < n; i++) {
                values[i] = type.createFloatVector(64);
                for (int j = 0; j < 64; j++) values[i].set(j, random.nextFloat() - .5f);
                VectorUtil.l2normalize(values[i]);
            }
            var ash = AsymmetricHashing.initialize(MockVectorValues.fromValues(values), AsymmetricHashing.RANDOM,
                    AsymmetricHashing.HEADER_BITS + 64, 1, 1);
            var encoded = ash.encodeAll(MockVectorValues.fromValues(values), ForkJoinPool.commonPool());
            var codes = new AsymmetricHashing.QuantizedVector[n];
            for (int i = 0; i < n; i++) codes[i] = encoded.get(i);
            VectorFloat<?> query = values[7];

            int[] sizes = {16, 32};
            float[][] expected = new float[sizes.length][];
            for (int s = 0; s < sizes.length; s++) {
                expected[s] = score(new ASHVectors(ash, codes), query, sizes[s]);
            }

            for (int round = 0; round < 40; round++) {
                // Even rounds race two queries with the same block size, odd rounds with different sizes.
                var vectors = new ASHVectors(ash, codes);
                int a = 0, b = round % 2;
                var barrier = new CyclicBarrier(2);
                List<Callable<float[]>> tasks = new ArrayList<>();
                for (int s : new int[]{a, b}) {
                    tasks.add(() -> {
                        barrier.await();
                        return score(vectors, query, sizes[s]);
                    });
                }
                List<Future<float[]>> results = pool.invokeAll(tasks);
                // Vector API lane reductions do not fix their summation order, so an early (interpreted) and a later
                // (compiled) scoring of the same data can differ by an ULP. A partial or wrong-size array is off by
                // about 0.1, so this tolerance still catches the race by four orders of magnitude.
                assertArrayEquals("round " + round, expected[a], results.get(0).get(), TOLERANCE);
                assertArrayEquals("round " + round, expected[b], results.get(1).get(), TOLERANCE);
            }
        } finally {
            pool.shutdownNow();
            if (old == null) System.clearProperty(BLOCK); else System.setProperty(BLOCK, old);
        }
    }

    private static float[] score(ASHVectors vectors, VectorFloat<?> query, int blockSize) {
        var scorer = vectors.blockScorerFor(query, VectorSimilarityFunction.DOT_PRODUCT, blockSize);
        int n = vectors.count();
        float[] out = new float[n];
        float[] block = new float[blockSize];
        // Score from the last block backwards so a reader reaches unfilled blocks first.
        for (int start = (n - 1) / blockSize * blockSize; start >= 0; start -= blockSize) {
            int count = Math.min(blockSize, n - start);
            scorer.scoreRange(start, count, block);
            System.arraycopy(block, 0, out, start, count);
        }
        return out;
    }
}
