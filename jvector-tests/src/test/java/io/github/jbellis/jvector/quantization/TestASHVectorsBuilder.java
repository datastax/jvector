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
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;

import static org.junit.Assert.*;

/** Codes encoded batch by batch from several threads are the codes encodeAll gives. */
public class TestASHVectorsBuilder {
    private static final int DIMENSION = 32;

    private static VectorFloat<?>[] vectors(int count, long seed) {
        Random random = new Random(seed);
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        VectorFloat<?>[] result = new VectorFloat<?>[count];
        for (int i = 0; i < count; i++) {
            result[i] = types.createFloatVector(DIMENSION);
            float norm = 0;
            for (int d = 0; d < DIMENSION; d++) {
                float v = random.nextFloat() - .5f;
                result[i].set(d, v);
                norm += v * v;
            }
            for (int d = 0; d < DIMENSION; d++) result[i].set(d, result[i].get(d) / (float) Math.sqrt(norm));
        }
        return result;
    }

    @Test
    public void batchesFromManyThreadsMatchEncodeAll() throws Exception {
        for (int bits : new int[] {1, 2, 4}) {
            int n = 1000;
            var all = vectors(n, 17);
            var ash = AsymmetricHashing.initialize(MockVectorValues.fromValues(vectors(256, 5)), AsymmetricHashing.ITQ,
                    AsymmetricHashing.HEADER_BITS + 16 * bits, 1, bits);
            var expected = ash.encodeAll(MockVectorValues.fromValues(all), ForkJoinPool.commonPool());

            // shuffled batches of different sizes, encoded concurrently, into a builder larger than needed
            var builder = ash.vectorsBuilder(n + 100);
            List<Integer> order = new ArrayList<>();
            for (int i = 0; i < n; i++) order.add(i);
            Collections.shuffle(order, new Random(3));
            List<int[]> batches = new ArrayList<>();
            for (int start = 0, size = 1; start < n; start += size, size = size % 70 + 1) {
                int end = Math.min(n, start + size);
                int[] batch = new int[end - start];
                for (int i = start; i < end; i++) batch[i - start] = order.get(i);
                batches.add(batch);
            }
            batches.parallelStream().forEach(ordinals -> {
                VectorFloat<?>[] batch = new VectorFloat<?>[ordinals.length];
                for (int i = 0; i < ordinals.length; i++) batch[i] = all[ordinals[i]];
                builder.encode(batch, ordinals, ordinals.length);
            });
            var built = builder.build(n);

            assertEquals(n, built.count());
            assertTrue("bits " + bits, expected.codes.contentEquals(built.codes));
            assertArrayEquals(expected.symmetricOffsets, built.symmetricOffsets, 0f);

            // the live view scores the same while it is filled, and so does the built store
            var liveScores = BuildScoreProvider.ashBuildScoreProvider(VectorSimilarityFunction.DOT_PRODUCT, builder.live());
            var builtScores = BuildScoreProvider.ashBuildScoreProvider(VectorSimilarityFunction.DOT_PRODUCT, built);
            var expectedScores = BuildScoreProvider.ashBuildScoreProvider(VectorSimilarityFunction.DOT_PRODUCT, expected);
            if (bits != 1) {
                for (int a = 0; a < n; a += 97) {
                    var e = expectedScores.searchProviderFor(a).scoreFunction();
                    var l = liveScores.searchProviderFor(a).scoreFunction();
                    var b = builtScores.searchProviderFor(a).scoreFunction();
                    for (int c = 0; c < n; c += 31) {
                        assertEquals(e.similarityTo(c), l.similarityTo(c), 0f);
                        assertEquals(e.similarityTo(c), b.similarityTo(c), 0f);
                    }
                }
            }
        }
    }

    @Test
    public void ordinalsNeverEncodedKeepTheMissingVectorCode() throws Exception {
        var ash = AsymmetricHashing.initialize(MockVectorValues.fromValues(vectors(256, 5)), AsymmetricHashing.ITQ,
                AsymmetricHashing.HEADER_BITS + 16 * 2, 1, 2);
        var some = vectors(10, 9);
        var builder = ash.vectorsBuilder(20);
        builder.encode(some, new int[] {0, 2, 4, 6, 8, 10, 12, 14, 16, 18}, some.length);
        var built = builder.build(19);
        var missing = AsymmetricHashing.QuantizedVector.createEmpty(ash.quantizedDim, ash.bitsPerDimension);
        var actual = AsymmetricHashing.QuantizedVector.createEmpty(ash.quantizedDim, ash.bitsPerDimension);
        for (int hole = 1; hole < 19; hole += 2) {
            built.copyTo(hole, actual);
            assertArrayEquals(missing.extraBits, actual.extraBits);
            assertEquals(ash.centeredOffset(missing), built.symmetricOffsets[hole], 0f);
        }
        assertThrows(IndexOutOfBoundsException.class, () -> builder.encode(some, new int[] {20}, 1));
    }
}
