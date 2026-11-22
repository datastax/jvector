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

import static org.junit.jupiter.api.Assertions.*;

/**
 * The Panama offset-binary dot (3 and 5 to 9 bits) matches an independent scalar decoder wherever the code sits in its
 * array, and gives the same bits before and after the JIT compiles it.
 */
class PanamaASHOffsetBinaryDotTest {
    private static final int[] WIDTHS = {3, 5, 6, 7, 8, 9};

    /** Field {@code j}: bits {@code [j*bits, (j+1)*bits)} of the little-endian code, minus (2^bits - 1) / 2. */
    private static double value(byte[] code, int offset, int j, int bits) {
        long field = 0;
        for (int t = 0; t < bits; t++) {
            int bit = j * bits + t;
            if ((code[offset + bit / 8] & (1 << (bit % 8))) != 0) field |= 1L << t;
        }
        return field - ((1 << bits) - 1) / 2.0;
    }

    @Test void dotMatchesScalarDecoder() {
        var panama = new PanamaVectorUtilSupport();
        if (!panama.supportsAshOffsetBinaryDot()) return;
        Random random = new Random(3591);
        for (int bits : WIDTHS) {
            for (int dims : new int[] {1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 190, 256, 380,
                    384, 512, 1536, 3073}) {
                int bytes = (dims * bits + 7) / 8;
                float[] query = new float[dims];
                for (int i = 0; i < dims; i++) query[i] = random.nextFloat() - .5f;
                // Codes at the start or middle of a page, and ending exactly at its end (the tail-copy path).
                for (int lead : new int[] {0, 5}) {
                    for (int trail : new int[] {0, 1, 70}) {
                        byte[] page = new byte[lead + bytes + trail];
                        random.nextBytes(page);
                        double expected = 0, magnitude = 0;
                        for (int i = 0; i < dims; i++) {
                            double v = value(page, lead, i, bits);
                            expected += query[i] * v;
                            magnitude += Math.abs(query[i] * v);
                        }
                        assertEquals(expected, panama.ashOffsetBinaryDot(query, page, lead, dims, bits),
                                1e-6 + magnitude * 2e-6, "bits " + bits + " dims " + dims + " lead " + lead + " trail " + trail);
                    }
                }
            }
        }
    }

    /** Sign-magnitude field {@code j} of a 2 or 4-bit code: {@code (f & m) + 1/2}, negative unless its top bit is set. */
    private static double signMagnitude(byte[] code, int offset, int j, int bits) {
        int field = (code[offset + j * bits / 8] & 255) >>> ((j * bits) % 8) & ((1 << bits) - 1);
        double m = (field & ((1 << (bits - 1)) - 1)) + .5;
        return (field >>> (bits - 1)) == 0 ? -m : m;
    }

    @Test void projectionDotMatchesScalarDecoder() {
        var panama = new PanamaVectorUtilSupport();
        if (!panama.supportsAshProjectionScoring()) return;
        Random random = new Random(4417);
        for (int bits : new int[] {2, 4}) {
            for (int dims : new int[] {1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 190, 256, 380, 384,
                    512, 1536, 3073}) {
                int bytes = (dims * bits + 7) / 8;
                float[] query = new float[dims];
                for (int i = 0; i < dims; i++) query[i] = random.nextFloat() - .5f;
                for (int lead : new int[] {0, 5}) {
                    for (int trail : new int[] {0, 1, 70}) {
                        byte[] page = new byte[lead + bytes + trail];
                        random.nextBytes(page);
                        double expected = 0, magnitude = 0;
                        for (int i = 0; i < dims; i++) {
                            double v = signMagnitude(page, lead, i, bits);
                            expected += query[i] * v;
                            magnitude += Math.abs(query[i] * v);
                        }
                        float actual = panama.ashProjectionDot(query, page, lead, dims, bits);
                        assertEquals(expected, actual, 1e-6 + magnitude * 2e-6,
                                "bits " + bits + " dims " + dims + " lead " + lead + " trail " + trail);
                    }
                }
            }
        }
    }

    /** The INT8 field-block lane dot is the exact integer sum, for every block size, lane and array end. */
    @Test void int8LaneDotIsExact() {
        var panama = new PanamaVectorUtilSupport();
        if (!panama.supportsAshOffsetBinaryInt8Dot()) return;
        Random random = new Random(1201);
        for (int bits : WIDTHS) {
            for (int units : new int[] {1, 2, 3, 5, 48, 97}) {
                byte[] digits = new byte[units * 8];
                for (int i = 0; i < digits.length; i++) digits[i] = (byte) (random.nextInt(255) - 127);
                for (int blockSize : new int[] {8, 16, 32}) {
                    int stride = blockSize * bits;
                    byte[] block = new byte[units * stride];
                    random.nextBytes(block);
                    for (int lane : new int[] {0, 1, blockSize - 1}) {
                        int offset = lane * bits;
                        long expected = 0;
                        for (int u = 0; u < units; u++) {
                            for (int k = 0; k < 8; k++) {
                                long field = 0;
                                int at = offset + u * stride;
                                for (int t = 0; t < bits; t++) {
                                    int bit = k * bits + t;
                                    if ((block[at + bit / 8] & (1 << (bit % 8))) != 0) field |= 1L << t;
                                }
                                expected += digits[u * 8 + k] * field;
                            }
                        }
                        long actual = panama.ashOffsetBinaryInt8Dot(block, offset, stride, units, bits, digits);
                        assertEquals(expected, actual,
                                "bits " + bits + " units " + units + " block " + blockSize + " lane " + lane);
                    }
                }
            }
        }
    }

    /** A code scores the same bits wherever it sits, including at the very end of its array. */
    @Test void scoresDoNotDependOnWhereTheCodeIs() {
        var panama = new PanamaVectorUtilSupport();
        if (!panama.supportsAshOffsetBinaryDot()) return;
        Random random = new Random(6604);
        for (int bits = 2; bits <= 9; bits++) {
            for (int dims : new int[] {17, 64, 190, 384, 1536}) {
                int bytes = (dims * bits + 7) / 8;
                float[] query = new float[dims];
                for (int i = 0; i < dims; i++) query[i] = random.nextFloat() - .5f;
                byte[] code = new byte[bytes];
                random.nextBytes(code);
                float reference = dot(panama, query, code, 0, dims, bits);
                for (int lead : new int[] {0, 3, 64}) {
                    for (int trail : new int[] {0, 1, 7, 33, 200}) {
                        byte[] page = new byte[lead + bytes + trail];
                        random.nextBytes(page);
                        System.arraycopy(code, 0, page, lead, bytes);
                        assertEquals(Float.floatToIntBits(reference),
                                Float.floatToIntBits(dot(panama, query, page, lead, dims, bits)),
                                "bits " + bits + " dims " + dims + " lead " + lead + " trail " + trail);
                    }
                }
            }
        }
    }

    private static float dot(PanamaVectorUtilSupport panama, float[] query, byte[] code, int offset, int dims,
                             int bits) {
        return bits == 2 || bits == 4 ? panama.ashProjectionDot(query, code, offset, dims, bits)
                : panama.ashOffsetBinaryDot(query, code, offset, dims, bits);
    }

    @Test void scoresDoNotChangeOnceCompiled() {
        var panama = new PanamaVectorUtilSupport();
        if (!panama.supportsAshOffsetBinaryDot()) return;
        Random random = new Random(8820);
        for (int bits : WIDTHS) {
            int dims = 381;
            float[] query = new float[dims];
            for (int i = 0; i < dims; i++) query[i] = random.nextFloat() - .5f;
            byte[] code = new byte[(dims * bits + 7) / 8];
            random.nextBytes(code);
            float first = panama.ashOffsetBinaryDot(query, code, 0, dims, bits);
            float last = first;
            for (int i = 0; i < 200_000; i++) last = panama.ashOffsetBinaryDot(query, code, 0, dims, bits);
            assertEquals(Float.floatToIntBits(first), Float.floatToIntBits(last), "bits " + bits);
        }
        for (int bits : new int[] {2, 4}) {
            int dims = 381;
            float[] query = new float[dims];
            for (int i = 0; i < dims; i++) query[i] = random.nextFloat() - .5f;
            byte[] code = new byte[(dims * bits + 7) / 8];
            random.nextBytes(code);
            float first = panama.ashProjectionDot(query, code, 0, dims, bits);
            float last = first;
            for (int i = 0; i < 200_000; i++) last = panama.ashProjectionDot(query, code, 0, dims, bits);
            assertEquals(Float.floatToIntBits(first), Float.floatToIntBits(last), "projection bits " + bits);
        }
    }

    @Test void rejectsOtherWidthsAndShortCodes() {
        var panama = new PanamaVectorUtilSupport();
        if (!panama.supportsAshOffsetBinaryDot()) return;
        for (int bits : new int[] {1, 2, 4, 10}) {
            assertThrows(IllegalArgumentException.class, () -> panama.ashOffsetBinaryDot(new float[8], new byte[16], 0, 8, bits));
        }
        assertThrows(IndexOutOfBoundsException.class, () -> panama.ashOffsetBinaryDot(new float[8], new byte[2], 0, 8, 3));
    }
}
