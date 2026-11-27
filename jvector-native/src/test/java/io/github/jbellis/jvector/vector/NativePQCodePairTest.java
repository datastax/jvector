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

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * The native PQ code-pair kernels against a plain double-precision loop, for every subspace-width class
 * (1, 2-3, 4-7, 8-15, 16+ and mixed), with uneven subspace counts so each kernel's tail paths run, and
 * with non-zero offsets into the code sequences.
 */
class NativePQCodePairTest {
    private static final int K = 256;

    private static int[] fill(int count, int width) {
        int[] widths = new int[count];
        Arrays.fill(widths, width);
        return widths;
    }

    private static int[][] shapes() {
        return new int[][] {
                // 1-float subspaces: the gather kernel, below, at and above one register of subspaces
                fill(1, 1), fill(3, 1), fill(4, 1), fill(5, 1), fill(8, 1), fill(9, 1), fill(16, 1), fill(17, 1), fill(128, 1),
                // 2- and 3-float subspaces
                fill(5, 2), fill(64, 2), fill(7, 3), {2, 3, 2, 3, 3, 2, 2, 3, 2},
                // 4 to 7 floats
                fill(4, 4), fill(9, 5), fill(8, 7), {4, 5, 6, 7, 4},
                // 8 to 15 floats
                fill(6, 8), fill(128, 8), fill(5, 12), {8, 9, 15, 8},
                // 16 floats and wider, including widths that leave a tail after the last full register
                fill(3, 16), fill(4, 20), fill(64, 16), fill(5, 33), {16, 17, 31, 16},
                // a 1-float subspace mixed with wider ones: the scalar kernel
                {1, 2, 3, 1}, {2, 1, 2, 2, 1, 2, 8, 1}, {1, 1, 1, 4},
        };
    }

    private static float reference(boolean dot, float[] codebooks, int[] offsets, int[] widths,
                                   byte[] code1, int offset1, byte[] code2, int offset2) {
        double sum = 0;
        for (int m = 0; m < widths.length; m++) {
            int base = K * offsets[m];
            int a = base + Byte.toUnsignedInt(code1[offset1 + m]) * widths[m];
            int b = base + Byte.toUnsignedInt(code2[offset2 + m]) * widths[m];
            for (int t = 0; t < widths[m]; t++) {
                double x = codebooks[a + t], y = codebooks[b + t];
                sum += dot ? x * y : (x - y) * (x - y);
            }
        }
        return (float) sum;
    }

    private static int min(int[] a) { return Arrays.stream(a).min().getAsInt(); }
    private static int max(int[] a) { return Arrays.stream(a).max().getAsInt(); }

    @Test
    void matchesPlainLoopForEveryWidthClass() {
        var support = NativeTestSupport.requireNativeProvider().getVectorUtilSupport();
        var random = new Random(42);
        for (int[] widths : shapes()) {
            int[] offsets = new int[widths.length];
            int dim = 0;
            for (int m = 0; m < widths.length; m++) {
                offsets[m] = dim;
                dim += widths[m];
            }
            float[] codebooks = new float[K * dim];
            for (int i = 0; i < codebooks.length; i++) codebooks[i] = random.nextFloat() * 2 - 1;
            var codebookVector = new MemorySegmentVectorFloat(codebooks);

            for (boolean dot : new boolean[] {true, false}) {
                for (int trial = 0; trial < 20; trial++) {
                    int offset1 = random.nextInt(9), offset2 = random.nextInt(9);
                    byte[] code1 = new byte[offset1 + widths.length + random.nextInt(4)];
                    byte[] code2 = new byte[offset2 + widths.length + random.nextInt(4)];
                    random.nextBytes(code1);
                    random.nextBytes(code2);
                    float expected = reference(dot, codebooks, offsets, widths, code1, offset1, code2, offset2);
                    float actual = support.pqCodePairSimilarity(dot, codebookVector, K, offsets, widths,
                            min(widths), max(widths), new int[dim],
                            new MemorySegmentByteSequence(code1), offset1, new MemorySegmentByteSequence(code2), offset2);
                    String where = (dot ? "dot " : "l2 ") + Arrays.toString(widths) + " trial " + trial;
                    assertEquals(expected, actual, 1e-4 * Math.max(1, Math.abs(expected)), where);
                }
            }
        }
    }

    @Test
    void identicalCodesHaveZeroSquaredDistance() {
        var support = NativeTestSupport.requireNativeProvider().getVectorUtilSupport();
        var random = new Random(7);
        for (int[] widths : new int[][] {fill(17, 1), fill(9, 3), fill(6, 8), fill(5, 20), {1, 2, 3}}) {
            int[] offsets = new int[widths.length];
            int dim = 0;
            for (int m = 0; m < widths.length; m++) {
                offsets[m] = dim;
                dim += widths[m];
            }
            float[] codebooks = new float[K * dim];
            for (int i = 0; i < codebooks.length; i++) codebooks[i] = random.nextFloat();
            byte[] code = new byte[widths.length];
            random.nextBytes(code);
            var sequence = new MemorySegmentByteSequence(code);
            assertEquals(0f, support.pqCodePairSimilarity(false, new MemorySegmentVectorFloat(codebooks), K, offsets, widths,
                    min(widths), max(widths), new int[dim], sequence, 0, sequence, 0), 0f, Arrays.toString(widths));
        }
    }

    @Test
    void batchGivesTheSingleCallResultForEveryWidthClass() {
        var support = NativeTestSupport.requireNativeProvider().getVectorUtilSupport();
        var random = new Random(11);
        for (int[] widths : shapes()) {
            int[] offsets = new int[widths.length];
            int dim = 0;
            for (int m = 0; m < widths.length; m++) {
                offsets[m] = dim;
                dim += widths[m];
            }
            float[] codebooks = new float[K * dim];
            for (int i = 0; i < codebooks.length; i++) codebooks[i] = random.nextFloat() * 2 - 1;
            var codebookVector = new MemorySegmentVectorFloat(codebooks);

            // a buffer of 40 codes laid out one after another, after a few bytes of padding
            int pad = random.nextInt(5);
            byte[] codes = new byte[pad + 40 * widths.length];
            random.nextBytes(codes);
            var sequence = new MemorySegmentByteSequence(codes);

            for (boolean dot : new boolean[] {true, false}) {
                for (int count : new int[] {1, 2, 5, 16, 40}) {
                    int[] at = new int[count];
                    for (int i = 0; i < count; i++) at[i] = pad + random.nextInt(40) * widths.length;
                    float[] out = new float[count + 3];
                    Arrays.fill(out, Float.NaN);
                    int offset1 = pad + random.nextInt(40) * widths.length;
                    support.pqCodePairSimilarityBatch(dot, codebookVector, K, offsets, widths, min(widths), max(widths),
                            new int[dim], sequence, offset1, sequence, at, count, out);
                    String where = (dot ? "dot " : "l2 ") + Arrays.toString(widths) + " count " + count;
                    for (int i = 0; i < count; i++) {
                        float single = support.pqCodePairSimilarity(dot, codebookVector, K, offsets, widths, min(widths), max(widths),
                                new int[dim], sequence, offset1, sequence, at[i]);
                        assertEquals(single, out[i], where + " pair " + i);
                    }
                    // nothing past count is written
                    for (int i = count; i < out.length; i++) assertEquals(Float.NaN, out[i], where);
                }
            }
        }
    }

    @Test
    void batchFailsInsteadOfReadingPastShortCodesOrArrays() {
        var support = NativeTestSupport.requireNativeProvider().getVectorUtilSupport();
        int[] widths = fill(8, 4), offsets = new int[8];
        for (int m = 0; m < 8; m++) offsets[m] = 4 * m;
        var codebooks = new MemorySegmentVectorFloat(new float[K * 32]);
        var code = new MemorySegmentByteSequence(new byte[16]);
        // the second code starts at 9, so it would need bytes 9..16
        assertThrows(IndexOutOfBoundsException.class, () -> support.pqCodePairSimilarityBatch(true, codebooks, K, offsets, widths,
                4, 4, new int[32], code, 0, code, new int[] {0, 9}, 2, new float[2]));
        assertThrows(IndexOutOfBoundsException.class, () -> support.pqCodePairSimilarityBatch(true, codebooks, K, offsets, widths,
                4, 4, new int[32], code, 9, code, new int[] {0}, 1, new float[1]));
        assertThrows(IndexOutOfBoundsException.class, () -> support.pqCodePairSimilarityBatch(true, codebooks, K, offsets, widths,
                4, 4, new int[32], code, 0, code, new int[] {0}, 1, new float[0]));
        assertThrows(IndexOutOfBoundsException.class, () -> support.pqCodePairSimilarityBatch(true, codebooks, K, offsets, widths,
                4, 4, new int[32], code, 0, code, new int[] {0}, 2, new float[2]));
        // an empty batch is a no-op
        support.pqCodePairSimilarityBatch(true, codebooks, K, offsets, widths, 4, 4, new int[32], code, 0, code, new int[0], 0, new float[0]);
    }

    @Test
    void shortCodesAndCodebooksFailInsteadOfReadingPastThem() {
        var support = NativeTestSupport.requireNativeProvider().getVectorUtilSupport();
        int[] widths = fill(8, 4), offsets = new int[8];
        for (int m = 0; m < 8; m++) offsets[m] = 4 * m;
        var codebooks = new MemorySegmentVectorFloat(new float[K * 32]);
        var code = new MemorySegmentByteSequence(new byte[8]);
        assertThrows(IndexOutOfBoundsException.class, () -> support.pqCodePairSimilarity(true, codebooks, K, offsets, widths,
                4, 4, new int[32], code, 1, code, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> support.pqCodePairSimilarity(true, codebooks, K, offsets, widths,
                4, 4, new int[32], code, 0, code, 1));
        assertThrows(IllegalArgumentException.class, () -> support.pqCodePairSimilarity(true,
                new MemorySegmentVectorFloat(new float[K * 31]), K, offsets, widths, 4, 4, new int[32], code, 0, code, 0));
    }
}
