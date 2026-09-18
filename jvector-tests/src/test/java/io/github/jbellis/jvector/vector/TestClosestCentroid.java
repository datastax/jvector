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
import static org.junit.Assert.assertTrue;

/** {@link VectorUtilSupport#closestCentroid} against a brute-force scan, for the active provider and the scalar default. */
public class TestClosestCentroid extends RandomizedTest {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final VectorUtilSupport active = VectorizationProvider.getInstance().getVectorUtilSupport();
    private static final VectorUtilSupport scalar = new DefaultVectorUtilSupport();

    @Test
    public void testMatchesBruteForce() {
        Random random = getRandom();
        int[] sizes = {1, 2, 3, 4, 5, 8, 9, 16};
        int[] clusterCounts = {1, 2, 3, 15, 16, 17, 31, 32, 33, 47, 64, 100, 256, 257};
        for (int size : sizes) {
            for (int clusterCount : clusterCounts) {
                int lead = random.nextInt(5);
                int trail = random.nextInt(5);
                VectorFloat<?> vector = vts.createFloatVector(lead + size + trail);
                for (int i = 0; i < vector.length(); i++) {
                    vector.set(i, (float) random.nextGaussian());
                }
                VectorFloat<?> transposed = vts.createFloatVector(size * clusterCount);
                for (int i = 0; i < transposed.length(); i++) {
                    transposed.set(i, (float) random.nextGaussian());
                }
                // a planted duplicate of the query, so the exact minimum is unambiguous
                int planted = random.nextInt(clusterCount);
                for (int i = 0; i < size; i++) {
                    transposed.set(i * clusterCount + planted, vector.get(lead + i));
                }
                assertEquals("size " + size + " clusters " + clusterCount, planted,
                             active.closestCentroid(vector, lead, transposed, size, clusterCount));
                if (vector instanceof ArrayVectorFloat) {
                    assertEquals(planted, scalar.closestCentroid(vector, lead, transposed, size, clusterCount));
                }
                // and without the plant: the chosen centroid is never farther than the brute-force minimum
                for (int i = 0; i < size; i++) {
                    transposed.set(i * clusterCount + planted, (float) random.nextGaussian());
                }
                int chosen = active.closestCentroid(vector, lead, transposed, size, clusterCount);
                assertTrue(chosen >= 0 && chosen < clusterCount);
                float chosenDistance = distance(vector, lead, transposed, size, clusterCount, chosen);
                float min = Float.MAX_VALUE;
                for (int j = 0; j < clusterCount; j++) {
                    min = Math.min(min, distance(vector, lead, transposed, size, clusterCount, j));
                }
                assertTrue("size " + size + " clusters " + clusterCount + ": " + chosenDistance + " > " + min,
                           chosenDistance <= min * (1 + 1e-5f) + 1e-6f);
            }
        }
    }

    private static float distance(VectorFloat<?> vector, int offset, VectorFloat<?> transposed, int size, int clusterCount, int j) {
        float distance = 0;
        for (int i = 0; i < size; i++) {
            float d = vector.get(offset + i) - transposed.get(i * clusterCount + j);
            distance += d * d;
        }
        return distance;
    }
}
