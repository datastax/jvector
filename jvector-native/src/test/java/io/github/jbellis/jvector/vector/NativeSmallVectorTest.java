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

/** Native f32 similarity and k-means distances at lengths from one float up through several SIMD widths. */
class NativeSmallVectorTest {
    private static final int[] LENGTHS = {1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 128, 512, 768, 3072};

    @Test
    void similarityMatchesDoubleReferenceAtSmallAndOddLengths() {
        var support = NativeTestSupport.requireNativeProvider().getVectorUtilSupport();
        var types = new MemorySegmentVectorProvider();
        var random = new Random(7);
        for (int length : LENGTHS) {
            for (int offset = 0; offset < 3; offset++) {
                float[] x = new float[length + offset], y = new float[length + offset];
                for (int i = 0; i < x.length; i++) {
                    x[i] = random.nextFloat() - 0.5f;
                    y[i] = random.nextFloat() - 0.5f;
                }
                double dot = 0, l2 = 0, nx = 0, ny = 0;
                for (int i = offset; i < x.length; i++) {
                    dot += x[i] * y[i];
                    nx += x[i] * x[i];
                    ny += y[i] * y[i];
                    double d = x[i] - y[i];
                    l2 += d * d;
                }
                VectorFloat<?> a = types.createFloatVector(x), b = types.createFloatVector(y);
                String where = "length=" + length + " offset=" + offset;
                double norms = Math.sqrt(nx * ny);
                assertEquals(dot, support.dotProduct(a, offset, b, offset, length), 1e-5 * Math.max(1, norms), "dot " + where);
                assertEquals(l2, support.squareDistance(a, offset, b, offset, length), 1e-5 * Math.max(1, l2), "l2 " + where);
                assertEquals(dot / norms, support.cosine(a, offset, b, offset, length), 1e-5, "cosine " + where);
            }
        }
    }

    @Test
    void kMeansDistancesCoverEveryDimension() {
        checkKMeansDistances(NativeTestSupport.requireNativeProvider().getVectorUtilSupport());
    }

    @Test
    void kMeansDistancesCoverEveryDimensionWith128BitVectors() {
        // The 128-bit support (NEON, SSE-only x86) has its own 8-float update; others must still fall back.
        NativeTestSupport.requireNativeProvider();
        checkKMeansDistances(new NativeVectorUtilSupport128());
    }

    private static void checkKMeansDistances(VectorUtilSupport support) {
        var types = new MemorySegmentVectorProvider();
        var random = new Random(11);
        for (int dimension : new int[]{1, 2, 3, 7, 8, 9, 16, 33}) {
            int count = 37;
            VectorFloat<?>[] points = new VectorFloat<?>[count];
            for (int i = 0; i < count; i++) points[i] = randomVector(types, random, dimension);
            var centroid = randomVector(types, random, dimension);
            var minima = types.createFloatVector(count);
            var scratch = types.createFloatVector(count);
            float[] expectedMin = new float[count];
            for (int i = 0; i < count; i++) {
                float start = i % 2 == 0 ? Float.MAX_VALUE : random.nextFloat();
                minima.set(i, start);
                expectedMin[i] = start;
            }
            assertTrue(support.tryUpdateKMeansDistances(points, centroid, minima, scratch), "dimension=" + dimension);
            for (int i = 0; i < count; i++) {
                double expected = 0;
                for (int j = 0; j < dimension; j++) {
                    double d = points[i].get(j) - centroid.get(j);
                    expected += d * d;
                }
                assertEquals(expected, scratch.get(i), 1e-5 * Math.max(1, expected), "distance dimension=" + dimension);
                assertEquals(Math.min(expectedMin[i], scratch.get(i)), minima.get(i), 0f, "minimum dimension=" + dimension);
            }
        }
    }

    private static VectorFloat<?> randomVector(MemorySegmentVectorProvider types, Random random, int dimension) {
        var v = types.createFloatVector(dimension);
        for (int j = 0; j < dimension; j++) v.set(j, (random.nextFloat() - 0.5f) * 4);
        return v;
    }
}
