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
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/** Eight-dimensional KMeans++ scan for JVMs whose preferred float vectors are 128 bits. */
final class NativeKMeansInitialization128 {
    private static final Class<?> FLOAT_128_CLASS = FloatVector.SPECIES_128.vectorType();

    static boolean update(VectorFloat<?>[] points, VectorFloat<?> centroid,
                          VectorFloat<?> minima, VectorFloat<?> scratch) {
        if (!(centroid instanceof MemorySegmentVectorFloat c) || c.length() != 8
            || !(minima instanceof MemorySegmentVectorFloat m)
            || !(scratch instanceof MemorySegmentVectorFloat s)) return false;
        if (m.length() != points.length || s.length() != points.length)
            throw new IllegalArgumentException("Distance vector lengths");
        float[] center = c.array(), distances = m.array(), next = s.array();
        for (int i = 0; i < points.length; i++) {
            float[] point = ((MemorySegmentVectorFloat) points[i]).array();
            if (point.length != 8) throw new IllegalArgumentException("Point dimensions differ from centroid");
            var a0 = (FloatVector) FLOAT_128_CLASS.cast(FloatVector.fromArray(FloatVector.SPECIES_128, point, 0));
            var b0 = (FloatVector) FLOAT_128_CLASS.cast(FloatVector.fromArray(FloatVector.SPECIES_128, center, 0));
            var a1 = (FloatVector) FLOAT_128_CLASS.cast(FloatVector.fromArray(FloatVector.SPECIES_128, point, 4));
            var b1 = (FloatVector) FLOAT_128_CLASS.cast(FloatVector.fromArray(FloatVector.SPECIES_128, center, 4));
            var diff0 = (FloatVector) FLOAT_128_CLASS.cast(a0.sub(b0));
            var diff1 = (FloatVector) FLOAT_128_CLASS.cast(a1.sub(b1));
            var squared0 = (FloatVector) FLOAT_128_CLASS.cast(diff0.mul(diff0));
            var squared1 = (FloatVector) FLOAT_128_CLASS.cast(diff1.mul(diff1));
            next[i] = ((FloatVector) FLOAT_128_CLASS.cast(squared0.add(squared1))).reduceLanes(VectorOperators.ADD);
        }
        int bound = FloatVector.SPECIES_128.loopBound(points.length), i = 0;
        for (; i < bound; i += FloatVector.SPECIES_128.length()) {
            var a = (FloatVector) FLOAT_128_CLASS.cast(FloatVector.fromArray(FloatVector.SPECIES_128, distances, i));
            var b = (FloatVector) FLOAT_128_CLASS.cast(FloatVector.fromArray(FloatVector.SPECIES_128, next, i));
            ((FloatVector) FLOAT_128_CLASS.cast(a.min(b))).intoArray(distances, i);
        }
        for (; i < points.length; i++) distances[i] = Math.min(distances[i], next[i]);
        return true;
    }
}
