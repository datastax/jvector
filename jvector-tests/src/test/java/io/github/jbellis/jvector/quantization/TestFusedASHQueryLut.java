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

package io.github.jbellis.jvector.quantization;

import org.junit.Test;
import java.util.Random;
import static org.junit.Assert.assertEquals;

/** The specialized query LUT builder must equal the per-entry definition bit for bit. */
public class TestFusedASHQueryLut {
    private static void reference(float[] qProj, int quantizedDim, int bits, float[] lut) {
        int groupDims = FusedASHLayout.projectionDimsPerNibble(bits);
        int groups = FusedASHLayout.codeGroups(quantizedDim, bits);
        for (int group = 0; group < groups; group++) {
            for (int nibble = 0; nibble < 16; nibble++) {
                float sum = 0f;
                for (int slot = 0; slot < groupDims; slot++) {
                    int dim = group * groupDims + slot;
                    if (dim >= quantizedDim) break;
                    int field = (nibble >>> (slot * bits)) & ((1 << bits) - 1);
                    sum += qProj[dim] * FusedASHLayout.decodeProjectionComponent(field, bits);
                }
                lut[(group << 4) + nibble] = sum;
            }
        }
    }

    @Test
    public void matchesDefinitionForEveryWidthAndTail() {
        Random random = new Random(7);
        float[] specials = {0f, -0f, Float.MIN_VALUE, -Float.MIN_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE,
                Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN, 1e-30f, -3.25f};
        for (int bits : new int[] {1, 2, 4}) {
            for (int dim : new int[] {1, 2, 3, 4, 5, 7, 8, 63, 64, 191, 192, 512, 513}) {
                for (int trial = 0; trial < 20; trial++) {
                    float[] q = new float[dim + 3]; // Values past quantizedDim must be ignored.
                    for (int i = 0; i < q.length; i++) {
                        q[i] = trial % 3 == 0 ? specials[random.nextInt(specials.length)]
                                : (float) random.nextGaussian() * (trial % 2 == 0 ? 1e-3f : 10f);
                    }
                    int size = FusedASHLayout.codeGroups(dim, bits) * 16;
                    float[] expected = new float[size + 5], actual = new float[size + 5];
                    reference(q, dim, bits, expected);
                    java.util.Arrays.fill(actual, 42f);
                    java.util.Arrays.fill(expected, size, expected.length, 42f);
                    FusedASHLayout.buildQueryLut(q, dim, bits, actual);
                    for (int i = 0; i < expected.length; i++) {
                        assertEquals("bits=" + bits + " dim=" + dim + " entry=" + i,
                                Float.floatToRawIntBits(expected[i]) == Float.floatToRawIntBits(actual[i])
                                        || (Float.isNaN(expected[i]) && Float.isNaN(actual[i])), true);
                    }
                }
            }
        }
    }

    /** Offset-binary widths: summing each 4-bit group's entry reproduces the decoded dot product. */
    @Test
    public void offsetBinaryLutSumsMatchDecodedDots() {
        Random random = new Random(11);
        for (int bits : new int[] {3, 5, 6, 7, 8, 9}) {
            for (int dim : new int[] {1, 2, 3, 5, 7, 8, 9, 63, 64, 65, 191, 192, 383}) {
                float[] q = new float[dim + 3];
                for (int i = 0; i < q.length; i++) q[i] = (float) random.nextGaussian();
                int groups = FusedASHLayout.codeGroups(dim, bits);
                float[] lut = new float[groups * 16];
                FusedASHLayout.buildQueryLut(q, dim, bits, lut);
                for (int trial = 0; trial < 20; trial++) {
                    // Random fields, plus random padding bits past the last field that must be ignored.
                    byte[] code = new byte[(groups + 1) / 2];
                    random.nextBytes(code);
                    int used = dim * bits;
                    for (int bit = used; bit < code.length * 8; bit++) code[bit >>> 3] &= (byte) ~(1 << (bit & 7));
                    double expected = 0, scale = 0;
                    for (int j = 0; j < dim; j++) {
                        int field = AsymmetricHashing.offsetBinaryField(code, 0, j, bits);
                        double value = field - ((1 << bits) - 1) / 2.0;
                        expected += q[j] * value;
                        scale += Math.abs(q[j]) * ((1 << bits) - 1);
                    }
                    float actual = 0f;
                    for (int g = 0; g < groups; g++) {
                        int nibble = ((code[g >>> 1] & 0xFF) >>> ((g & 1) * 4)) & 15;
                        actual += lut[(g << 4) + nibble];
                    }
                    assertEquals("bits=" + bits + " dim=" + dim, expected, actual, 1e-5 * Math.max(1, scale));
                    assertEquals(expected, AsymmetricHashing.dotOffsetBinaryCode(q, code, 0, dim, bits), 1e-5 * Math.max(1, scale));
                }
            }
        }
    }
}
