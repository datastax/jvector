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

/** Array SIMD for NVQ's inherited Java arithmetic, without creating native views. */
final class NativeNVQArithmetic {
    // A constant, not a local, so every compile (OSR included) sees the exact vector class.
    private static final Class<?> PREFERRED_CLASS = FloatVector.SPECIES_PREFERRED.vectorType();

    static float min(VectorFloat<?> vector) {
        float[] data = ((MemorySegmentVectorFloat)vector).array();
        var accum = (FloatVector)PREFERRED_CLASS.cast(FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, Float.MAX_VALUE));
        int bound = FloatVector.SPECIES_PREFERRED.loopBound(data.length), i = 0;
        for (; i < bound; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = (FloatVector)PREFERRED_CLASS.cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, data, i));
            accum = (FloatVector)PREFERRED_CLASS.cast(accum.lanewise(VectorOperators.MIN, a));
        }
        float result = accum.reduceLanes(VectorOperators.MIN);
        for (; i < data.length; i++) result = Math.min(result, data[i]);
        return result;
    }

    static void scale(VectorFloat<?> vector, float multiplier) {
        float[] data = ((MemorySegmentVectorFloat)vector).array();
        var factor = (FloatVector)PREFERRED_CLASS.cast(FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, multiplier));
        int bound = FloatVector.SPECIES_PREFERRED.loopBound(data.length), i = 0;
        for (; i < bound; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = (FloatVector)PREFERRED_CLASS.cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, data, i));
            ((FloatVector)PREFERRED_CLASS.cast(a.lanewise(VectorOperators.MUL, factor))).intoArray(data, i);
        }
        for (; i < data.length; i++) data[i] *= multiplier;
    }

    static VectorFloat<?> sub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length) {
        float[] lhs = ((MemorySegmentVectorFloat)a).array(), rhs = ((MemorySegmentVectorFloat)b).array();
        var result = new MemorySegmentVectorFloat(length);
        float[] data = result.array();
        int bound = FloatVector.SPECIES_PREFERRED.loopBound(length), i = 0;
        for (; i < bound; i += FloatVector.SPECIES_PREFERRED.length()) {
            var x = (FloatVector)PREFERRED_CLASS.cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, lhs, aOffset+i));
            var y = (FloatVector)PREFERRED_CLASS.cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, rhs, bOffset+i));
            ((FloatVector)PREFERRED_CLASS.cast(x.lanewise(VectorOperators.SUB, y))).intoArray(data, i);
        }
        for (; i < length; i++) data[i] = lhs[aOffset+i] - rhs[bOffset+i];
        return result;
    }

}
