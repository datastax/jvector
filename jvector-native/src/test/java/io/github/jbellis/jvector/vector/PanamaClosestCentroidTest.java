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

import static org.junit.jupiter.api.Assertions.*;

/** The Panama bulk centroid scan picks the same centroid as the per-centroid squareDistance loop it replaces. */
class PanamaClosestCentroidTest {
    private final PanamaVectorUtilSupport panama = new PanamaVectorUtilSupport();
    private final ArrayVectorProvider vectors = new ArrayVectorProvider();

    private int previousScan(VectorFloat<?> point, int offset, VectorFloat<?> centroids, int count, int dim) {
        int nearest = 0;
        float best = Float.MAX_VALUE;
        for (int i = 0; i < count; i++) {
            float distance = panama.squareDistance(point, offset, centroids, i * dim, dim);
            if (distance < best) { best = distance; nearest = i; }
        }
        return nearest;
    }

    @Test
    void matchesPerCentroidScan() {
        Random random = new Random(1701);
        for (int dim : new int[]{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24, 32, 65}) {
            for (int count : new int[]{1, 2, 17, 128, 256}) {
                for (boolean integral : new boolean[]{true, false}) {
                    float[] point = new float[dim + 7];
                    float[] codebook = new float[count * dim + 5];
                    for (int i = 0; i < point.length; i++) point[i] = integral ? random.nextInt(33) - 16 : (float) random.nextGaussian();
                    for (int i = 0; i < codebook.length; i++) codebook[i] = integral ? random.nextInt(33) - 16 : (float) random.nextGaussian();
                    float[] originalPoint = point.clone(), originalCodebook = codebook.clone();
                    var p = vectors.createFloatVector(point);
                    var c = vectors.createFloatVector(codebook);
                    String where = "dim=" + dim + " count=" + count + " integral=" + integral;
                    assertEquals(previousScan(p, 3, c, count, dim), panama.tryClosestCentroid(p, 3, c, count, dim), where);
                    assertArrayEquals(originalPoint, point);
                    assertArrayEquals(originalCodebook, codebook);
                }
            }
        }
    }

    @Test
    void preservesFirstTieAndNonFiniteComparisonBehavior() {
        for (int dim : new int[]{1, 2, 4, 8}) {
            var p = vectors.createFloatVector(new float[dim]);
            for (float[] values : new float[][]{{2, -2, 3}, {Float.NaN, 1, -1}, {Float.POSITIVE_INFINITY, Float.NaN},
                    {Float.MAX_VALUE, Float.MAX_VALUE}, {-0.0f, 0.0f}, {Float.NaN, Float.NaN}}) {
                // each centroid repeats its value in every dimension
                float[] data = new float[values.length * dim];
                for (int i = 0; i < data.length; i++) data[i] = values[i / dim];
                var c = vectors.createFloatVector(data);
                assertEquals(previousScan(p, 0, c, values.length, dim), panama.tryClosestCentroid(p, 0, c, values.length, dim),
                        "dim=" + dim);
            }
        }
    }

    @Test
    void rejectsInvalidRegions() {
        var p = vectors.createFloatVector(8);
        var c = vectors.createFloatVector(32);
        assertThrows(IllegalArgumentException.class, () -> panama.tryClosestCentroid(p, 0, c, 0, 8));
        assertThrows(IndexOutOfBoundsException.class, () -> panama.tryClosestCentroid(p, -1, c, 4, 8));
        assertThrows(IndexOutOfBoundsException.class, () -> panama.tryClosestCentroid(p, 1, c, 4, 8));
        assertThrows(IndexOutOfBoundsException.class, () -> panama.tryClosestCentroid(p, 0, c, 5, 8));
        assertThrows(IndexOutOfBoundsException.class, () -> panama.tryClosestCentroid(p, 0, c, Integer.MAX_VALUE, 8));
    }
}
