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
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TestFullPrecisionKernels {
    private static final int[] LENGTHS = {0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17,
            31, 32, 33, 65, 768, 1024, 1536, 3072};

    @Test
    void distancesMatchDoubleOraclesAcrossWidthsOffsetsAndTails() {
        var support = new PanamaVectorUtilSupport();
        var random = new Random(1825);
        for (int length : LENGTHS) {
            for (int offset : new int[]{0, 1, 7}) {
                float[] a = new float[length + offset + 9], b = new float[a.length];
                for (int i = 0; i < a.length; i++) {
                    a[i] = random.nextFloat() * 2 - 1;
                    b[i] = random.nextFloat() * 2 - 1;
                }
                double dot = 0, l2 = 0, aa = 0, bb = 0, absoluteDot = 0;
                for (int i = 0; i < length; i++) {
                    double x = a[offset + i], y = b[offset + 1 + i];
                    dot += x * y;
                    absoluteDot += Math.abs(x * y);
                    l2 += (x - y) * (x - y);
                    aa += x * x;
                    bb += y * y;
                }
                var av = new ArrayVectorFloat(a);
                var bv = new ArrayVectorFloat(b);
                for (int bits : new int[]{64, 128, 256, FloatVector.SPECIES_PREFERRED.vectorBitSize()}) {
                    float actualDot, actualL2;
                    switch (bits) {
                        case 64:
                            actualDot = support.dotProduct64(av, offset, bv, offset + 1, length);
                            actualL2 = support.squareDistance64(av, offset, bv, offset + 1, length);
                            break;
                        case 128:
                            actualDot = support.dotProduct128(av, offset, bv, offset + 1, length);
                            actualL2 = support.squareDistance128(av, offset, bv, offset + 1, length);
                            break;
                        case 256:
                            actualDot = support.dotProduct256(av, offset, bv, offset + 1, length);
                            actualL2 = support.squareDistance256(av, offset, bv, offset + 1, length);
                            break;
                        default:
                            actualDot = support.dotProductPreferred(av, offset, bv, offset + 1, length);
                            actualL2 = support.squareDistancePreferred(av, offset, bv, offset + 1, length);
                    }
                    assertEquals(dot, actualDot, Math.max(1e-7, absoluteDot * 2e-6));
                    assertEquals(l2, actualL2, Math.max(1e-7, l2 * 2e-6));
                }
                var wholeA = new ArrayVectorFloat(java.util.Arrays.copyOfRange(a, offset, offset + length));
                var wholeB = new ArrayVectorFloat(java.util.Arrays.copyOfRange(b, offset + 1, offset + 1 + length));
                assertEquals(dot, support.dotProduct(wholeA, wholeB), Math.max(1e-7, absoluteDot * 2e-6));
                assertEquals(l2, support.squareDistance(wholeA, wholeB), Math.max(1e-7, l2 * 2e-6));
                if (length == 0) assertTrue(Float.isNaN(support.cosine(wholeA, wholeB)));
                else assertEquals(dot / Math.sqrt(aa * bb), support.cosine(wholeA, wholeB), 2e-6);
                assertEquals(dot, support.dotProduct(av, offset, bv, offset + 1, length),
                        Math.max(1e-7, absoluteDot * 2e-6));
                assertEquals(l2, support.squareDistance(av, offset, bv, offset + 1, length),
                        Math.max(1e-7, l2 * 2e-6));
                float cosine = support.cosine(av, offset, bv, offset + 1, length);
                if (length == 0) assertTrue(Float.isNaN(cosine));
                else assertEquals(dot / Math.sqrt(aa * bb), cosine, 2e-6);
            }
        }
    }

    @Test
    void reductionsPreserveSpecialValuesAndEmptyInput() {
        var support = new PanamaVectorUtilSupport();
        float[] values = {0f, -0f, Float.NaN, Float.POSITIVE_INFINITY,
                Float.NEGATIVE_INFINITY, Float.MIN_VALUE, -1f, 1f};
        for (int length : LENGTHS) {
            for (float value : values) {
                float[] data = new float[length];
                float min = Float.MAX_VALUE, max = -Float.MAX_VALUE;
                for (int i = 0; i < length; i++) {
                    data[i] = value;
                    min = Math.min(min, value);
                    max = Math.max(max, value);
                }
                var vector = new ArrayVectorFloat(data);
                assertEquals(Float.floatToIntBits(min), Float.floatToIntBits(support.min(vector)));
                assertEquals(Float.floatToIntBits(max), Float.floatToIntBits(support.max(vector)));
            }
        }
    }

    @Test
    void hammingMatchesScalarForVectorBlocksAndTails() {
        var support = new PanamaVectorUtilSupport();
        var random = new Random(187);
        for (int length : new int[]{0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 48, 96}) {
            long[] a = new long[length], b = new long[length];
            int expected = 0;
            for (int i = 0; i < length; i++) {
                a[i] = random.nextLong();
                b[i] = random.nextLong();
                expected += Long.bitCount(a[i] ^ b[i]);
            }
            assertEquals(expected, support.hammingDistance(a, b));
            assertEquals(0, support.hammingDistance(a, a));
        }
    }
}
