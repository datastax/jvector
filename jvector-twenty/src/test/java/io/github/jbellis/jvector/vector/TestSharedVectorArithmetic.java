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

import java.util.Random;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TestSharedVectorArithmetic {
    private static final int[] LENGTHS = {0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33,
            65, 768, 1024, 1536, 3072};
    private static final float[] SPECIAL = {0f, -0f, Float.NaN, Float.POSITIVE_INFINITY,
            Float.NEGATIVE_INFINITY, Float.MIN_VALUE, -1f, 1f, 0.12345f};

    private static void assertBits(float expected, float actual) {
        assertEquals(Float.floatToIntBits(expected), Float.floatToIntBits(actual));
    }

    @Test
    void scalarArithmeticPreservesSpecialValuesAndTails() {
        var support = new PanamaVectorUtilSupport();
        for (int length : LENGTHS) {
            float[] original = new float[length];
            for (int i = 0; i < length; i++) original[i] = SPECIAL[i % SPECIAL.length];
            for (float scalar : SPECIAL) {
                var scaled = new ArrayVectorFloat(original.clone());
                var added = new ArrayVectorFloat(original.clone());
                var subtracted = new ArrayVectorFloat(original.clone());
                support.scale(scaled, scalar);
                support.addInPlace(added, scalar);
                support.subInPlace(subtracted, scalar);
                var result = support.sub(new ArrayVectorFloat(original), scalar);
                for (int i = 0; i < length; i++) {
                    assertBits(original[i] * scalar, scaled.get(i));
                    assertBits(original[i] + scalar, added.get(i));
                    assertBits(original[i] - scalar, subtracted.get(i));
                    assertBits(original[i] - scalar, result.get(i));
                }
            }
        }
    }

    @Test
    void vectorArithmeticPreservesSpecialValuesAndAliasing() {
        var support = new PanamaVectorUtilSupport();
        for (int length : LENGTHS) {
            float[] a = new float[length], b = new float[length];
            for (int i = 0; i < length; i++) {
                a[i] = SPECIAL[i % SPECIAL.length];
                b[i] = SPECIAL[(i + 3) % SPECIAL.length];
            }
            var other = new ArrayVectorFloat(b);
            var added = new ArrayVectorFloat(a.clone());
            var subtracted = new ArrayVectorFloat(a.clone());
            var minimum = new ArrayVectorFloat(a.clone());
            support.addInPlace(added, other);
            support.subInPlace(subtracted, other);
            support.minInPlace(minimum, other);
            for (int i = 0; i < length; i++) {
                assertBits(a[i] + b[i], added.get(i));
                assertBits(a[i] - b[i], subtracted.get(i));
                assertBits(Math.min(a[i], b[i]), minimum.get(i));
            }
            var alias = new ArrayVectorFloat(a.clone());
            support.addInPlace(alias, alias);
            for (int i = 0; i < length; i++) assertBits(a[i] + a[i], alias.get(i));
            alias = new ArrayVectorFloat(a.clone());
            support.subInPlace(alias, alias);
            for (int i = 0; i < length; i++) assertBits(a[i] - a[i], alias.get(i));
            alias = new ArrayVectorFloat(a.clone());
            support.minInPlace(alias, alias);
            for (int i = 0; i < length; i++) assertBits(Math.min(a[i], a[i]), alias.get(i));
        }
        var small = new ArrayVectorFloat(new float[1]);
        var large = new ArrayVectorFloat(new float[2]);
        assertThrows(IllegalArgumentException.class, () -> support.addInPlace(small, large));
        assertThrows(IllegalArgumentException.class, () -> support.subInPlace(small, large));
        assertThrows(IllegalArgumentException.class, () -> support.minInPlace(small, large));
    }

    @Test
    void outOfPlaceSubtractionPreservesOffsetsAndInputs() {
        var support = new PanamaVectorUtilSupport();
        for (int length : LENGTHS) {
            float[] a = new float[length + 11], b = new float[length + 13];
            for (int i = 0; i < a.length; i++) a[i] = SPECIAL[i % SPECIAL.length];
            for (int i = 0; i < b.length; i++) b[i] = SPECIAL[(i + 3) % SPECIAL.length];
            var av = new ArrayVectorFloat(a.clone());
            var bv = new ArrayVectorFloat(b.clone());
            var result = support.sub(av, 3, bv, 7, length);
            var aliasResult = support.sub(av, 1, av, 3, length);
            assertEquals(length, result.length());
            for (int i = 0; i < length; i++) {
                assertBits(a[i + 3] - b[i + 7], result.get(i));
                assertBits(a[i + 1] - a[i + 3], aliasResult.get(i));
            }
            for (int i = 0; i < a.length; i++) assertBits(a[i], av.get(i));
            for (int i = 0; i < b.length; i++) assertBits(b[i], bv.get(i));
        }
    }

    @Test
    void sumPreservesOriginalReductionOrder() {
        var support = new PanamaVectorUtilSupport();
        var random = new Random(981);
        for (int length : LENGTHS) {
            float[] values = new float[length];
            for (int i = 0; i < length; i++) values[i] = random.nextFloat() * 2 - 1;
            // Freeze the previous lane accumulation and reduction order.
            var sum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
            int limit = FloatVector.SPECIES_PREFERRED.loopBound(length);
            for (int i = 0; i < limit; i += FloatVector.SPECIES_PREFERRED.length())
                sum = sum.add(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, values, i));
            float expected = sum.reduceLanes(VectorOperators.ADD);
            for (int i = limit; i < length; i++) expected += values[i];
            assertEquals(Float.floatToRawIntBits(expected),
                    Float.floatToRawIntBits(support.sum(new ArrayVectorFloat(values))));
        }
    }
}
