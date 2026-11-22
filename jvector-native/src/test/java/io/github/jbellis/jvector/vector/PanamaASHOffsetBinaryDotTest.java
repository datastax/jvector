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
