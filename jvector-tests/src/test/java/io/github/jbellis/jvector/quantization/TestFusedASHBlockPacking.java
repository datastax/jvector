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
import java.util.Arrays;
import java.util.Random;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class TestFusedASHBlockPacking {
    /** Independent oracle: the original nibble-by-nibble fused layout. */
    private static void reference(byte[] destination, int offset, int lane,
                                  AsymmetricHashing.QuantizedVector vector,
                                  int dimension, int bits, int blockSize) {
        int groups = FusedASHLayout.codeGroups(dimension, bits);
        for (int group = 0; group < groups; group++) {
            int nibble = bits == 1
                    ? FusedASHLayout.signNibbleFromWords(vector.binaryVector, group, dimension)
                    : FusedASHLayout.flatNibble(vector.extraBits, group);
            FusedASHLayout.setPackedNibble(destination, offset, lane, group, blockSize, nibble);
        }
        FusedASHLayout.writeLaneHeader(destination, offset, lane, dimension, bits, blockSize,
                vector.scale, vector.offset, vector.landmark);
    }

    @Test
    public void testPackingMatchesOriginalLayout() throws Exception {
        var constructor = AsymmetricHashing.QuantizedVector.class.getDeclaredConstructor(
                float.class, float.class, byte.class, long[].class, byte[].class);
        constructor.setAccessible(true);
        Random random = new Random(19);
        int[] dimensions = new int[266];
        for (int i = 0; i < 257; i++) {
            dimensions[i] = i + 1;
        }
        int[] large = {383, 384, 385, 767, 768, 769, 1536, 3072, 3073};
        System.arraycopy(large, 0, dimensions, 257, large.length);
        int cases = 0;
        for (int bits : new int[]{1, 2, 4}) {
            for (int dimension : dimensions) {
                for (int blockSize : new int[]{8, 16, 32}) {
                    var vectors = new AsymmetricHashing.QuantizedVector[blockSize + 3];
                    for (int i = 0; i < vectors.length; i++) {
                        byte[] code = new byte[FusedASHLayout.canonicalCodeBytes(dimension, bits)];
                        random.nextBytes(code);
                        long[] signs = new long[(dimension + 63) / 64];
                        for (int j = 0; j < signs.length; j++) {
                            signs[j] = random.nextLong();
                        }
                        vectors[i] = constructor.newInstance(random.nextFloat() * 3,
                                random.nextFloat() - 0.5f, (byte) random.nextInt(256), signs, code);
                    }
                    int length = FusedASHLayout.blockBytes(dimension, bits, blockSize);
                    int offset = 7; // Exercise unaligned long loads/stores and guard bytes.
                    for (int count = 0; count <= blockSize; count++) {
                        byte[] expected = new byte[length + 14];
                        Arrays.fill(expected, (byte) 87);
                        Arrays.fill(expected, offset, offset + length, (byte) 0);
                        byte[] actual = expected.clone();
                        for (int lane = 0; lane < count; lane++) {
                            reference(expected, offset, lane, vectors[3 + lane],
                                    dimension, bits, blockSize);
                        }
                        FusedASHBlockPacker.pack(actual, offset, vectors, 3, count,
                                dimension, bits, blockSize);
                        assertArrayEquals("bits=" + bits + " dimension=" + dimension
                                + " blockSize=" + blockSize + " count=" + count, expected, actual);
                        cases++;
                    }

                    // Single-vector packing must preserve other lanes and high padding nibbles.
                    byte[] expected = new byte[length + 14];
                    random.nextBytes(expected);
                    byte[] actual = expected.clone();
                    reference(expected, offset, blockSize - 1, vectors[0], dimension, bits, blockSize);
                    FusedASHLayout.packQuantizedVector(actual, offset, blockSize - 1,
                            vectors[0], dimension, bits, blockSize);
                    assertArrayEquals(expected, actual);
                    cases++;
                }
            }
        }
        assertEquals(49_476, cases);
    }
}
