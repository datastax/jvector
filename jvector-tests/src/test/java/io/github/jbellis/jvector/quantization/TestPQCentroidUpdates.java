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
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;
import static org.junit.Assert.*;

public class TestPQCentroidUpdates extends RandomizedTest {
    @Test
    public void testLloydIterationsMatchOriginalUpdatesBitwise() {
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        float[] values = {-10, -9, -4, -3, -2, 1, 2, 3, 4, 9, 10};
        for (int dimension : new int[]{1, 3, 8, 17, 65}) {
            VectorFloat<?>[] points = new VectorFloat<?>[values.length];
            for (int i = 0; i < points.length; i++) {
                points[i] = types.createFloatVector(dimension);
                for (int d = 0; d < dimension; d++) points[i].set(d, values[i]);
            }
            var initial = types.createFloatVector(4 * dimension);
            float[] centers = {-10, 0, 3, 10};
            for (int c = 0; c < 4; c++) for (int d = 0; d < dimension; d++) initial.set(c * dimension + d, centers[c]);
            var expected = initial.copy();
            VectorFloat<?>[] sums = new VectorFloat<?>[4];
            int[] counts = new int[4], assignments = new int[points.length];
            for (int c = 0; c < 4; c++) sums[c] = types.createFloatVector(dimension);
            for (int i = 0; i < points.length; i++) {
                int c = nearest(points[i], expected, dimension);
                assignments[i] = c; counts[c]++;
                VectorUtil.addInPlace(sums[c], points[i]);
            }
            var actual = new KMeansPlusPlusClusterer(points, initial, KMeansPlusPlusClusterer.UNWEIGHTED);
            for (int iteration = 0; iteration < 4; iteration++) {
                for (int c = 0; c < 4; c++) {
                    assertTrue("Fixture must avoid random empty-cluster reseeding", counts[c] > 0);
                    var temporary = sums[c].copy();
                    VectorUtil.scale(temporary, 1.0f / counts[c]);
                    expected.copyFrom(temporary, 0, c * dimension, dimension);
                }
                int changed = 0;
                for (int i = 0; i < points.length; i++) {
                    int old = assignments[i], next = nearest(points[i], expected, dimension);
                    if (old != next) {
                        counts[old]--;
                        VectorUtil.subInPlace(sums[old], points[i]);
                        counts[next]++;
                        VectorUtil.addInPlace(sums[next], points[i]);
                        assignments[i] = next; changed++;
                    }
                }
                if (iteration == 0) assertTrue("Fixture must exercise reassignment", changed > 0);
                assertEquals(changed, actual.clusterOnceUnweighted());
                for (int d = 0; d < expected.length(); d++)
                    assertEquals(Float.floatToRawIntBits(expected.get(d)), Float.floatToRawIntBits(actual.getCentroids().get(d)));
            }
            for (int i = 0; i < points.length; i++) for (int d = 0; d < dimension; d++) assertEquals(values[i], points[i].get(d), 0);
        }
    }

    private static int nearest(VectorFloat<?> point, VectorFloat<?> centroids, int dimension) {
        int nearest = 0;
        float minimum = Float.MAX_VALUE;
        for (int c = 0; c < 4; c++) {
            float distance = VectorUtil.squareL2Distance(point, 0, centroids, c * dimension, dimension);
            if (distance < minimum) { minimum = distance; nearest = c; }
        }
        return nearest;
    }
}
