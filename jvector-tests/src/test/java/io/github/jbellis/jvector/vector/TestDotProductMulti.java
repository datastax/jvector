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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

/** {@link VectorUtilSupport#dotProductMulti} of the active provider against single dot products. */
public class TestDotProductMulti extends RandomizedTest {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final VectorUtilSupport active = VectorizationProvider.getInstance().getVectorUtilSupport();

    @Test
    public void testMatchesSingleDotProducts() {
        Random random = getRandom();
        for (int dim : new int[]{1, 7, 16, 33, 384}) {
            for (int count : new int[]{0, 1, 3, 8, 9, 17, 32}) {
                VectorFloat<?> v = vts.createFloatVector(dim);
                for (int i = 0; i < dim; i++) v.set(i, (float) random.nextGaussian());
                VectorFloat<?>[] queries = new VectorFloat<?>[count];
                for (int j = 0; j < count; j++) {
                    queries[j] = vts.createFloatVector(dim);
                    for (int i = 0; i < dim; i++) queries[j].set(i, (float) random.nextGaussian());
                }
                float[] out = new float[count + 1];
                out[count] = 42f;   // must stay untouched
                active.dotProductMulti(v, queries, count, out);
                for (int j = 0; j < count; j++) {
                    float expected = VectorUtil.dotProduct(v, queries[j]);
                    assertEquals("dim " + dim + " count " + count + " query " + j, expected, out[j], 1e-4f * Math.max(1f, Math.abs(expected)));
                }
                assertEquals(42f, out[count], 0f);
            }
        }
    }
}
