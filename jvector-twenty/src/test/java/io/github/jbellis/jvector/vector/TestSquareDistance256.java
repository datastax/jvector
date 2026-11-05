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
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class TestSquareDistance256 {
    @Test
    void exactWidthAndSurroundingLengthsMatchScalarReference() {
        var support = new PanamaVectorUtilSupport();
        var random = new Random(941);
        for (int length = 0; length <= 65; length++) {
            for (int offset = 0; offset < 5; offset++) {
                float[] a = new float[length + offset + 3];
                float[] b = new float[length + offset + 5];
                for (int i = 0; i < a.length; i++) a[i] = random.nextInt(21) - 10;
                for (int i = 0; i < b.length; i++) b[i] = random.nextInt(21) - 10;
                float expected = 0;
                for (int i = 0; i < length; i++) {
                    float difference = a[offset + i] - b[offset + 2 + i];
                    expected += difference * difference;
                }
                float actual = support.squareDistance(
                        new ArrayVectorFloat(a), offset, new ArrayVectorFloat(b), offset + 2, length);
                // Integer inputs keep every intermediate exactly representable, including
                // the eight-coordinate kernel and its surrounding dispatch/tail cases.
                assertEquals(expected, actual, "length=" + length + " offset=" + offset);
            }
        }
    }
}
