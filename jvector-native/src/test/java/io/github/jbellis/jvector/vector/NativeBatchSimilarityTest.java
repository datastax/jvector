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

package io.github.jbellis.jvector.vector;

import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static io.github.jbellis.jvector.vector.NativeTestSupport.requireNativeProvider;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** The batched native kernels must equal the one-pair-at-a-time native kernels, bit for bit. */
class NativeBatchSimilarityTest {
    private final NativeVectorUtilSupport support = new NativeVectorUtilSupport();
    private final MemorySegmentVectorProvider types = new MemorySegmentVectorProvider();

    private VectorFloat<?> random(Random random, int n) {
        var v = types.createFloatVector(n);
        for (int i = 0; i < n; i++) v.set(i, (float) random.nextGaussian());
        return v;
    }

    @Test
    void batchMatchesSinglePairs() {
        requireNativeProvider();
        var random = new Random(42);
        for (int dim : new int[]{1, 3, 4, 7, 8, 15, 16, 17, 33, 64, 128, 384, 1536}) {
            for (int count : new int[]{0, 1, 2, 5, 32, 65}) {
                var query = random(random, dim);
                var vectors = random(random, dim * count);
                float[] out = new float[count + 1];
                out[count] = 42f; // must stay untouched

                support.squareDistanceBatch(query, vectors, count, out);
                for (int i = 0; i < count; i++) {
                    assertEquals(support.squareDistance(query, 0, vectors, i * dim, dim), out[i], 0f, "l2 dim " + dim + " i " + i);
                }
                assertEquals(42f, out[count], 0f);

                support.dotProductBatch(query, vectors, count, out);
                for (int i = 0; i < count; i++) {
                    assertEquals(support.dotProduct(query, 0, vectors, i * dim, dim), out[i], 0f, "dot dim " + dim + " i " + i);
                }
                assertEquals(42f, out[count], 0f);

                support.cosineBatch(query, vectors, count, out);
                for (int i = 0; i < count; i++) {
                    assertEquals(support.cosine(query, 0, vectors, i * dim, dim), out[i], 0f, "cosine dim " + dim + " i " + i);
                }
                assertEquals(42f, out[count], 0f);
            }
        }
    }
}
