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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;

/** {@link VectorUtilSupport#tryClosestCentroid} of the active provider against the per-centroid scan it replaces. */
public class TestClosestCentroid extends RandomizedTest {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final VectorUtilSupport active = VectorizationProvider.getInstance().getVectorUtilSupport();
    // The Panama provider must answer for the small subspaces whose Vector API path JDK 21 can fail to intrinsify.
    private static final boolean SMALL_SUBSPACES_REQUIRED =
            VectorizationProvider.getInstance().getClass().getSimpleName().equals("PanamaVectorizationProvider");

    /** The scan ProductQuantization and KMeansPlusPlusClusterer run when the provider returns -1. */
    private static int scan(VectorFloat<?> point, int offset, VectorFloat<?> centroids, int count, int dim) {
        int nearest = 0;
        float minimum = Float.MAX_VALUE;
        for (int i = 0; i < count; i++) {
            float distance = active.squareDistance(point, offset, centroids, i * dim, dim);
            if (distance < minimum) {
                minimum = distance;
                nearest = i;
            }
        }
        return nearest;
    }

    @Test
    public void testMatchesPerCentroidScan() {
        Random random = getRandom();
        for (int dim : new int[]{0, 1, 2, 3, 4, 7, 8, 16}) {
            for (int count : new int[]{1, 2, 17, 256}) {
                for (int trial = 0; trial < 20; trial++) {
                    // Small integers make exact ties common, so the first-of-equal-distances rule is exercised.
                    boolean ties = trial % 2 == 0;
                    float[] point = new float[dim + 5];
                    float[] codebook = new float[count * dim + 3];
                    for (int i = 0; i < point.length; i++) point[i] = ties ? random.nextInt(5) - 2 : (float) random.nextGaussian();
                    for (int i = 0; i < codebook.length; i++) codebook[i] = ties ? random.nextInt(5) - 2 : (float) random.nextGaussian();
                    float[] originalPoint = point.clone(), originalCodebook = codebook.clone();
                    var p = vts.createFloatVector(point);
                    var c = vts.createFloatVector(codebook);
                    int actual = active.tryClosestCentroid(p, 3, c, count, dim);
                    if (dim <= 3 && SMALL_SUBSPACES_REQUIRED) assertNotEquals("dim " + dim, -1, actual);
                    if (actual >= 0) assertEquals("dim " + dim + " count " + count, scan(p, 3, c, count, dim), actual);
                    assertArrayEquals(originalPoint, point, 0f);
                    assertArrayEquals(originalCodebook, codebook, 0f);
                }
            }
        }
    }

    @Test
    public void testSummationOrderMatchesPerCentroidScan() {
        // Centroids that permute one difference vector have the same exact squared terms, so only the order the terms
        // are added in tells their distances apart: a kernel that sums in another order picks another centroid.
        int[][][] permutations = {{{0, 1}, {1, 0}},
                {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};
        Random random = getRandom();
        for (int[][] orders : permutations) {
            int dim = orders[0].length;
            var p = vts.createFloatVector(dim);
            for (int trial = 0; trial < 2000; trial++) {
                float[] d = new float[dim];
                for (int i = 0; i < dim; i++) d[i] = (float) random.nextGaussian();
                float[] codebook = new float[orders.length * dim];
                for (int c = 0; c < orders.length; c++)
                    for (int i = 0; i < dim; i++) codebook[c * dim + i] = d[orders[c][i]];
                var c = vts.createFloatVector(codebook);
                int actual = active.tryClosestCentroid(p, 0, c, orders.length, dim);
                if (actual >= 0) assertEquals("dim " + dim + " trial " + trial, scan(p, 0, c, orders.length, dim), actual);
            }
        }
    }

    @Test
    public void testNonFiniteDistancesMatchPerCentroidScan() {
        float[][] codebooks = {{2, -2, 3}, {Float.NaN, 1, -1}, {Float.POSITIVE_INFINITY, Float.NaN},
                {Float.MAX_VALUE, Float.MAX_VALUE}, {-0.0f, 0.0f}, {Float.NaN, Float.NaN}};
        for (int dim = 1; dim <= 3; dim++) {
            var p = vts.createFloatVector(dim);
            for (float[] values : codebooks) {
                // Repeat each value across the subspace so every centroid has the listed distance class.
                float[] data = new float[values.length * dim];
                for (int i = 0; i < data.length; i++) data[i] = values[i / dim];
                var c = vts.createFloatVector(data);
                boolean finite = true;
                for (int i = 0; i < values.length; i++) finite &= Float.isFinite(active.squareDistance(p, 0, c, i * dim, dim));
                if (!finite && active.closestCentroidChecksFiniteness() && active.getClass().desiredAssertionStatus()) {
                    // The provider's own scan must assert as the caller's scan did, or VectorUtil would skip the check.
                    int count = values.length, d = dim;
                    assertThrows(AssertionError.class, () -> active.tryClosestCentroid(p, 0, c, count, d));
                    continue;
                }
                int actual = active.tryClosestCentroid(p, 0, c, values.length, dim);
                if (actual >= 0) assertEquals(scan(p, 0, c, values.length, dim), actual);
            }
        }
    }

    @Test
    public void testRejectsInvalidRegions() {
        if (active.tryClosestCentroid(vts.createFloatVector(2), 0, vts.createFloatVector(2), 1, 2) < 0) return;
        var p = vts.createFloatVector(4);
        var c = vts.createFloatVector(8);
        assertThrows(IllegalArgumentException.class, () -> active.tryClosestCentroid(p, 0, c, 0, 2));
        assertThrows(IndexOutOfBoundsException.class, () -> active.tryClosestCentroid(p, -1, c, 4, 2));
        assertThrows(IndexOutOfBoundsException.class, () -> active.tryClosestCentroid(p, 3, c, 4, 2));
        assertThrows(IndexOutOfBoundsException.class, () -> active.tryClosestCentroid(p, 0, c, 5, 2));
        assertThrows(IndexOutOfBoundsException.class, () -> active.tryClosestCentroid(p, 0, c, Integer.MAX_VALUE, 2));
    }
}
