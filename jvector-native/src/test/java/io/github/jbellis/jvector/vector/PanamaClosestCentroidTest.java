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

/**
 * The Panama bulk centroid scan and k-means++ distance update give the same results as the per-centroid and
 * per-point squareDistance loops they replace.
 */
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

    @Test
    void kMeansUpdateMatchesPerPointLoop() {
        Random random = new Random(1702);
        for (int dim : new int[]{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24, 32, 65}) {
            for (int count : new int[]{1, 7, 33, 300}) {
                for (boolean integral : new boolean[]{true, false}) {
                    VectorFloat<?>[] points = new VectorFloat<?>[count];
                    float[][] originalPoints = new float[count][];
                    for (int i = 0; i < count; i++) {
                        originalPoints[i] = values(random, dim, integral);
                        points[i] = vectors.createFloatVector(originalPoints[i].clone());
                    }
                    float[] center = values(random, dim, integral);
                    var centroid = vectors.createFloatVector(center.clone());
                    float[] start = new float[count];
                    for (int i = 0; i < count; i++) {
                        start[i] = i % 5 == 0 ? Float.MAX_VALUE : integral ? random.nextInt(600) : (float) Math.abs(random.nextGaussian() * dim);
                    }
                    var expectedScratch = vectors.createFloatVector(count);
                    for (int i = 0; i < count; i++) expectedScratch.set(i, panama.squareDistance(points[i], 0, centroid, 0, dim));
                    var expectedMinima = vectors.createFloatVector(start.clone());
                    panama.minInPlace(expectedMinima, expectedScratch);

                    var minima = vectors.createFloatVector(start.clone());
                    var scratch = vectors.createFloatVector(count);
                    String where = "dim=" + dim + " count=" + count + " integral=" + integral;
                    assertTrue(panama.tryUpdateKMeansDistances(points, centroid, minima, scratch), where);
                    for (int i = 0; i < count; i++) {
                        closeDistance(expectedScratch.get(i), scratch.get(i), dim, where + " scratch " + i);
                        closeDistance(expectedMinima.get(i), minima.get(i), dim, where + " minima " + i);
                        for (int j = 0; j < dim; j++) assertEquals(originalPoints[i][j], points[i].get(j), 0f, where);
                    }
                    for (int j = 0; j < dim; j++) assertEquals(center[j], centroid.get(j), 0f, where);
                }
            }
        }
    }

    @Test
    void kMeansUpdateRejectsMismatchedLengths() {
        VectorFloat<?>[] points = {vectors.createFloatVector(4), vectors.createFloatVector(4)};
        var centroid = vectors.createFloatVector(4);
        assertThrows(IllegalArgumentException.class,
                () -> panama.tryUpdateKMeansDistances(points, centroid, vectors.createFloatVector(1), vectors.createFloatVector(2)));
        assertThrows(IllegalArgumentException.class,
                () -> panama.tryUpdateKMeansDistances(points, centroid, vectors.createFloatVector(2), vectors.createFloatVector(3)));
        VectorFloat<?>[] mixed = {vectors.createFloatVector(4), vectors.createFloatVector(5)};
        assertThrows(IllegalArgumentException.class,
                () -> panama.tryUpdateKMeansDistances(mixed, centroid, vectors.createFloatVector(2), vectors.createFloatVector(2)));
    }

    private static void closeDistance(float expected, float actual, int dim, String where) {
        // FloatVector ADD reductions may use any order, even within one JVM, so the per-point loop itself is not
        // bit-stable: allow one ULP per rounded addition.
        if (Float.isFinite(expected) && expected != 0.0f) {
            assertTrue(Math.abs((double) expected - actual) <= Math.max(8, dim) * Math.ulp(expected),
                    where + " expected=" + expected + " actual=" + actual);
        } else {
            assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(actual), where);
        }
    }

    private static float[] values(Random random, int dim, boolean integral) {
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++) v[i] = integral ? random.nextInt(33) - 16 : (float) random.nextGaussian();
        return v;
    }
}
