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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeTrue;

/** k-means++ start must catch non-finite points on every provider, including native updates that skip per-distance checks. */
public class TestKMeansInitialPoints extends RandomizedTest {
    // Common PQ subspace sizes are 4, 8 and 16; 8 takes the dedicated native update, and uneven splits give sizes like 3
    private static final int[] DIMENSIONS = {8, 4, 16, 3};

    @Test
    public void testNonFinitePointFails() {
        assumeTrue(KMeansPlusPlusClusterer.class.desiredAssertionStatus());
        for (int dimension : DIMENSIONS) {
            for (float bad : new float[]{Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY}) {
                var points = points(dimension, 0, 0, 1, 1);
                points[2].set(dimension - 1, bad);
                assertThrows("dimension " + dimension + ", value " + bad, AssertionError.class,
                             () -> new KMeansPlusPlusClusterer(points, 2));
            }
        }
    }

    @Test
    public void testFiniteOverflowPasses() {
        for (int dimension : DIMENSIONS) {
            // Each distance is finite (about 2e38), but any first centroid leaves two of them, whose sum overflows
            float large = (float) Math.sqrt(2e38 / dimension);
            new KMeansPlusPlusClusterer(points(dimension, 0, 0, large, large), 2);
        }
    }

    private static VectorFloat<?>[] points(int dimension, float... values) {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var points = new VectorFloat<?>[values.length];
        for (int i = 0; i < values.length; i++) {
            float[] point = new float[dimension];
            Arrays.fill(point, values[i]);
            points[i] = vts.createFloatVector(point);
        }
        return points;
    }
}
