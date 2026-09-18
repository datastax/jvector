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
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** {@link VectorUtilSupport#quantizeTableU8} of the active provider against the scalar default. */
public class TestQuantizeTableU8 extends RandomizedTest {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final VectorUtilSupport active = VectorizationProvider.getInstance().getVectorUtilSupport();
    private static final VectorUtilSupport scalar = new DefaultVectorUtilSupport();

    @Test
    public void testMatchesScalar() {
        Random random = getRandom();
        int[] subspaceCounts = {1, 3, 48, 192};
        int[] clusterCounts = {1, 7, 16, 17, 100, 256};
        for (int m : subspaceCounts) {
            for (int k : clusterCounts) {
                for (boolean negate : new boolean[]{false, true}) {
                    VectorFloat<?> table = vts.createFloatVector(m * k);
                    for (int i = 0; i < table.length(); i++) {
                        table.set(i, (float) (random.nextGaussian() * (1 + random.nextInt(5))));
                    }
                    ByteSequence<?> a = vts.createByteSequence(m * k);
                    ByteSequence<?> b = vts.createByteSequence(m * k);
                    float[] soA = new float[2], soB = new float[2];
                    active.quantizeTableU8(table, m, k, negate, a, soA);
                    scalar.quantizeTableU8(table, m, k, negate, b, soB);
                    assertEquals(soB[0], soA[0], Math.abs(soB[0]) * 1e-6f);
                    assertEquals(soB[1], soA[1], Math.abs(soB[1]) * 1e-5f + 1e-6f);
                    for (int i = 0; i < m * k; i++) {
                        int qa = a.get(i) & 0xFF, qb = b.get(i) & 0xFF;
                        assertTrue("m=" + m + " k=" + k + " i=" + i + ": " + qa + " vs " + qb, Math.abs(qa - qb) <= 1);
                    }
                    // the affine identity: byte sum / scale + offset reproduces the float sum of one entry per subspace
                    float floatSum = 0; int byteSum = 0;
                    for (int s = 0; s < m; s++) {
                        int c = random.nextInt(k);
                        floatSum += (negate ? -1 : 1) * table.get(s * k + c);
                        byteSum += a.get(s * k + c) & 0xFF;
                    }
                    if (soA[0] > 0) {
                        float back = byteSum / soA[0] + soA[1];
                        assertEquals(floatSum, back, m * 0.6f / soA[0] + 1e-3f);
                    }
                }
            }
        }
    }
}
