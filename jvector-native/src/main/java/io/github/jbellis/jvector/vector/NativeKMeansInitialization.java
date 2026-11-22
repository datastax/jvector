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

/**
 * Eight-dimensional KMeans++ initialization uses Java array SIMD while Lloyd
 * assignment and encoding retain Highway. Keeping the point loop here avoids
 * a native call per tiny distance and generic vector load/return boundaries.
 */
final class NativeKMeansInitialization {
    static boolean update(VectorFloat<?>[] points, VectorFloat<?> centroid,
                                 VectorFloat<?> minima, VectorFloat<?> scratch) {
        if (!(centroid instanceof MemorySegmentVectorFloat c) || c.length() != 8
            || !(minima instanceof MemorySegmentVectorFloat m)
            || !(scratch instanceof MemorySegmentVectorFloat s)) return false;
        if (m.length() != points.length || s.length() != points.length)
            throw new IllegalArgumentException("Distance vector lengths");
        float[] center = c.array(), distances = m.array(), next = s.array();
        var type = FloatVector.SPECIES_256.vectorType();
        // Match the Panama 8D arithmetic: subtraction, multiplication and reduceLanes.
        // Reduction order can differ from Highway's native distance kernel.
        for (int i = 0; i < points.length; i++) {
            float[] point = ((MemorySegmentVectorFloat) points[i]).array();
            if (point.length != 8) throw new IllegalArgumentException("Point dimensions differ from centroid");
            var a = (FloatVector) type.cast(FloatVector.fromArray(FloatVector.SPECIES_256, point, 0));
            var b = (FloatVector) type.cast(FloatVector.fromArray(FloatVector.SPECIES_256, center, 0));
            var diff = (FloatVector) type.cast(a.sub(b));
            next[i] = ((FloatVector) type.cast(diff.mul(diff))).reduceLanes(VectorOperators.ADD);
        }
        var preferred = FloatVector.SPECIES_PREFERRED;
        var preferredType = preferred.vectorType();
        int bound = preferred.loopBound(points.length), i = 0;
        for (; i < bound; i += preferred.length()) {
            var a = (FloatVector) preferredType.cast(FloatVector.fromArray(preferred, distances, i));
            var b = (FloatVector) preferredType.cast(FloatVector.fromArray(preferred, next, i));
            ((FloatVector) preferredType.cast(a.min(b))).intoArray(distances, i);
        }
        for (; i < points.length; i++) distances[i] = Math.min(distances[i], next[i]);
        return true;
    }
}
