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
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class ClosestCentroidTest {
    @BeforeAll
    static void requireNativeProvider() {
        NativeTestSupport.requireNativeProvider();
    }

    private final VectorUtilSupport nativeSupport = new NativeVectorizationProvider().getVectorUtilSupport();
    private final MemorySegmentVectorProvider vectors = new MemorySegmentVectorProvider();

    private int previousScan(VectorFloat<?> point, int offset, VectorFloat<?> centroids, int count, int dim) {
        int nearest = 0;
        float best = Float.MAX_VALUE;
        for (int i = 0; i < count; i++) {
            float distance = nativeSupport.squareDistance(point, offset, centroids, i * dim, dim);
            if (distance < best) { best = distance; nearest = i; }
        }
        return nearest;
    }

    @Test
    void matchesExistingNativeDistancesAndScalarOracle() {
        Random random = new Random(1701);
        for (int dim : new int[]{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 24, 32, 65}) {
            for (int count : new int[]{1, 2, 17, 128, 256}) {
                float[] point = new float[dim + 7];
                float[] codebook = new float[count * dim + 5];
                for (int i = 0; i < point.length; i++) point[i] = random.nextInt(33) - 16;
                for (int i = 0; i < codebook.length; i++) codebook[i] = random.nextInt(33) - 16;
                float[] originalPoint = point.clone(), originalCodebook = codebook.clone();
                var p = vectors.createFloatVector(point);
                var c = vectors.createFloatVector(codebook);
                int expected = previousScan(p, 3, c, count, dim);
                int actual = nativeSupport.tryClosestCentroid(p, 3, c, count, dim);
                assertEquals(expected, actual, "dim=" + dim + " count=" + count);
                int scalarBest = 0;
                double minimum = Double.MAX_VALUE;
                for (int j = 0; j < count; j++) {
                    double distance = 0;
                    for (int d = 0; d < dim; d++) {
                        double diff = point[3 + d] - codebook[j * dim + d];
                        distance += diff * diff;
                    }
                    if (distance < minimum) { minimum = distance; scalarBest = j; }
                }
                assertEquals(scalarBest, actual);
                assertArrayEquals(originalPoint, point);
                assertArrayEquals(originalCodebook, codebook);
            }
        }
    }

    @Test
    void preservesFirstTieAndNonFiniteComparisonBehavior() {
        var p = vectors.createFloatVector(new float[]{0});
        for (float[] data : new float[][]{{2, -2, 3}, {Float.NaN, 1, -1},
                {Float.POSITIVE_INFINITY, Float.NaN}, {Float.MAX_VALUE, Float.MAX_VALUE},
                {-0.0f, 0.0f}, {Float.NaN, Float.NaN}}) {
            var c = vectors.createFloatVector(data);
            assertEquals(previousScan(p, 0, c, data.length, 1),
                    nativeSupport.tryClosestCentroid(p, 0, c, data.length, 1));
        }
    }

    @Test
    void rejectsInvalidRegionsBeforeNativeAccess() {
        var p = vectors.createFloatVector(8);
        var c = vectors.createFloatVector(32);
        assertThrows(IllegalArgumentException.class, () -> nativeSupport.tryClosestCentroid(p, 0, c, 0, 8));
        assertThrows(IndexOutOfBoundsException.class, () -> nativeSupport.tryClosestCentroid(p, -1, c, 4, 8));
        assertThrows(IndexOutOfBoundsException.class, () -> nativeSupport.tryClosestCentroid(p, 1, c, 4, 8));
        assertThrows(IndexOutOfBoundsException.class, () -> nativeSupport.tryClosestCentroid(p, 0, c, 5, 8));
        assertThrows(IndexOutOfBoundsException.class, () -> nativeSupport.tryClosestCentroid(p, 0, c, Integer.MAX_VALUE, 8));
    }
}
