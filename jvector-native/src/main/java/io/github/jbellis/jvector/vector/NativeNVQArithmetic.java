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
    static float min(VectorFloat<?> vector) {
        float[] data = ((MemorySegmentVectorFloat)vector).array();
        var species = FloatVector.SPECIES_PREFERRED;
        var type = species.vectorType();
        var accum = (FloatVector)type.cast(FloatVector.broadcast(species, Float.MAX_VALUE));
        int bound = species.loopBound(data.length), i = 0;
        for (; i < bound; i += species.length()) {
            var a = (FloatVector)type.cast(FloatVector.fromArray(species, data, i));
            accum = (FloatVector)type.cast(accum.min(a));
        }
        float result = accum.reduceLanes(VectorOperators.MIN);
        for (; i < data.length; i++) result = Math.min(result, data[i]);
        return result;
    }

    static void scale(VectorFloat<?> vector, float multiplier) {
        float[] data = ((MemorySegmentVectorFloat)vector).array();
        var species = FloatVector.SPECIES_PREFERRED;
        var type = species.vectorType();
        var factor = (FloatVector)type.cast(FloatVector.broadcast(species, multiplier));
        int bound = species.loopBound(data.length), i = 0;
        for (; i < bound; i += species.length()) {
            var a = (FloatVector)type.cast(FloatVector.fromArray(species, data, i));
            ((FloatVector)type.cast(a.mul(factor))).intoArray(data, i);
        }
        for (; i < data.length; i++) data[i] *= multiplier;
    }

    static VectorFloat<?> sub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length) {
        float[] lhs = ((MemorySegmentVectorFloat)a).array(), rhs = ((MemorySegmentVectorFloat)b).array();
        var result = new MemorySegmentVectorFloat(length);
        float[] data = result.array();
        var species = FloatVector.SPECIES_PREFERRED;
        var type = species.vectorType();
        int bound = species.loopBound(length), i = 0;
        for (; i < bound; i += species.length()) {
            var x = (FloatVector)type.cast(FloatVector.fromArray(species, lhs, aOffset+i));
            var y = (FloatVector)type.cast(FloatVector.fromArray(species, rhs, bOffset+i));
            ((FloatVector)type.cast(x.sub(y))).intoArray(data, i);
        }
        for (; i < length; i++) data[i] = lhs[aOffset+i] - rhs[bOffset+i];
        return result;
    }

}
