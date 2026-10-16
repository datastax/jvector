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
import static org.junit.jupiter.api.Assertions.assertEquals;

class TestDotProduct256 {
    @Test
    void exactWidthPreservesOriginalReductionBits() {
        var support = new PanamaVectorUtilSupport();
        var random = new Random(619);
        for (int iteration = 0; iteration < 512; iteration++) {
            float[] a = new float[16], b = new float[19];
            for (int i = 0; i < a.length; i++) a[i] = random.nextFloat() * 2 - 1;
            for (int i = 0; i < b.length; i++) b[i] = random.nextFloat() * 2 - 1;
            int offset = iteration % 5;
            float expected = FloatVector.fromArray(FloatVector.SPECIES_256, a, offset)
                    .mul(FloatVector.fromArray(FloatVector.SPECIES_256, b, offset + 2))
                    .reduceLanes(VectorOperators.ADD);
            float actual = support.dotProduct256(new ArrayVectorFloat(a), offset,
                    new ArrayVectorFloat(b), offset + 2, 8);
            assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(actual));
        }
    }

    @Test
    void surroundingLengthsAndOffsetsPreserveDispatchAndTails() {
        var support = new PanamaVectorUtilSupport();
        var random = new Random(620);
        for (int length = 0; length <= 65; length++) {
            for (int offset = 0; offset < 5; offset++) {
                float[] a = new float[length + offset + 3], b = new float[length + offset + 5];
                for (int i = 0; i < a.length; i++) a[i] = random.nextInt(21) - 10;
                for (int i = 0; i < b.length; i++) b[i] = random.nextInt(21) - 10;
                float expected = 0;
                for (int i = 0; i < length; i++) expected += a[offset + i] * b[offset + 2 + i];
                float actual = support.dotProduct(new ArrayVectorFloat(a), offset,
                        new ArrayVectorFloat(b), offset + 2, length);
                // Integer inputs make every intermediate exactly representable.
                assertEquals(expected, actual, "length=" + length + " offset=" + offset);
            }
        }
    }
}
