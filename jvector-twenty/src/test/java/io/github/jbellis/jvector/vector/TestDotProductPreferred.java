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

class TestDotProductPreferred {
    @Test
    void offsetsAndTailsMatchDoublePrecisionReference() {
        var support = new PanamaVectorUtilSupport();
        var random = new Random(392);
        for (int length : new int[] {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 768, 1024, 1536, 3072}) {
            for (int offset = 0; offset < 5; offset++) {
                float[] a = new float[length + 8];
                float[] b = new float[length + 8];
                for (int i = 0; i < a.length; i++) {
                    a[i] = random.nextFloat() * 2 - 1;
                    b[i] = random.nextFloat() * 2 - 1;
                }
                double expected = 0;
                double magnitude = 0;
                for (int i = 0; i < length; i++) {
                    double product = (double) a[offset + i] * b[4 - offset + i];
                    expected += product;
                    magnitude += Math.abs(product);
                }
                float actual = support.dotProductPreferred(
                        new ArrayVectorFloat(a), offset, new ArrayVectorFloat(b), 4 - offset, length);
                assertEquals(expected, actual, Math.max(1e-6, magnitude * 2e-6),
                        "length=" + length + " offset=" + offset);
            }
        }
    }
}
