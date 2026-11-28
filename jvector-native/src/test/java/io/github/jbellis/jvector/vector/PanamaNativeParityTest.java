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

import io.github.jbellis.jvector.quantization.ASHQueryProjection;
import io.github.jbellis.jvector.vector.cnative.LibraryLoader;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The Panama ASH kernels that parallel native ones: each against an independent oracle with the native arithmetic,
 * and, where the native library is loaded, against the native kernel itself.
 */
class PanamaNativeParityTest {
    private static final PanamaVectorUtilSupport PANAMA = new PanamaVectorUtilSupport();
    private static boolean nativeLoaded;

    @BeforeAll
    static void load() {
        try {
            nativeLoaded = LibraryLoader.loadJvector();
        } catch (Throwable unavailable) {
            nativeLoaded = false;
        }
    }

    private static float half(byte[] codes, int at) {
        return Float.float16ToFloat((short) (((codes[at] & 0xFF) << 8) | (codes[at + 1] & 0xFF)));
    }

    private static void writeHalf(byte[] codes, int at, float value) {
        short h = Float.floatToFloat16(value);
        codes[at] = (byte) (h >>> 8);
        codes[at + 1] = (byte) h;
    }

    /** A nibble-code neighborhood with random bodies and headers, landmark 0. */
    private static byte[] nibbleNeighborhood(Random random, int groups, int blockSize, int capacity) {
        int body = (groups + 1) / 2 * blockSize, blockBytes = body + 5 * blockSize;
        int blocks = (capacity - 1) / blockSize + 1;
        byte[] codes = new byte[blocks * blockBytes];
        for (int b = 0; b < blocks; b++) {
            int at = b * blockBytes;
            for (int i = 0; i < body; i++) codes[at + i] = (byte) random.nextInt(256);
            for (int lane = 0; lane < blockSize; lane++) {
                writeHalf(codes, at + body + 2 * lane, (float) random.nextGaussian());
                writeHalf(codes, at + body + 2 * blockSize + 2 * lane, (float) random.nextGaussian());
                codes[at + body + 4 * blockSize + lane] = 0;
            }
        }
        return codes;
    }

    @Test
    void lutNeighborhoodMatchesNativeArithmetic() {
        assumeTrue(PANAMA.supportsAshNeighborhoodScoring(), "Panama neighborhood scoring unavailable");
        Random random = new Random(21);
        for (int groups : new int[]{1, 2, 3, 15, 16, 17, 96, 192, 384}) {
            for (int blockSize : new int[]{8, 16, 32}) {
                for (int capacity : new int[]{1, 7, blockSize, 32, 33, 64}) {
                    byte[] codes = nibbleNeighborhood(random, groups, blockSize, capacity);
                    float[] lut = new float[groups * 16];
                    for (int i = 0; i < lut.length; i++) lut[i] = (float) random.nextGaussian();
                    float landmarkDot = (float) random.nextGaussian(), perEstimate = -0.75f;
                    float[] dot = new float[capacity], euclidean = new float[capacity];
                    PANAMA.ashScoreNeighborhood(codes, groups, blockSize, capacity, lut, landmarkDot, dot);
                    PANAMA.ashScoreNeighborhoodEuclidean(codes, groups, blockSize, capacity, lut, landmarkDot, perEstimate, euclidean);
                    int body = (groups + 1) / 2 * blockSize, blockBytes = body + 5 * blockSize;
                    for (int n = 0; n < capacity; n++) {
                        int b = n / blockSize, lane = n % blockSize, at = b * blockBytes;
                        float even = 0, odd = 0;
                        for (int g = 0; g < groups; g++) {
                            int nibble = (codes[at + (g / 2) * blockSize + lane] >>> (4 * (g & 1))) & 15;
                            if ((g & 1) == 0) even += lut[g * 16 + nibble]; else odd += lut[g * 16 + nibble];
                        }
                        float e = Math.fma(half(codes, at + body + 2 * lane), even + odd, landmarkDot)
                                + half(codes, at + body + 2 * blockSize + 2 * lane);
                        String where = "groups=" + groups + " block=" + blockSize + " capacity=" + capacity + " n=" + n;
                        assertEquals(Math.fma(e, 0.5f, 0.5f), dot[n], where);
                        assertEquals(1f / (1f + Math.max(perEstimate * e, 0f)), euclidean[n], where);
                    }
                    if (nativeLoaded && NativeASHKernels.isAvailable()) {
                        float[] nativeDot = new float[capacity];
                        NativeASHKernels.scoreNeighborhood(codes, groups, blockSize, capacity, lut, landmarkDot, 0f, nativeDot);
                        assertArrayEquals(nativeDot, dot, "native groups=" + groups + " block=" + blockSize);
                        if (NativeASHKernels.isEuclideanAvailable()) {
                            float[] nativeEuclidean = new float[capacity];
                            NativeASHKernels.scoreNeighborhood(codes, groups, blockSize, capacity, lut, landmarkDot,
                                    perEstimate, nativeEuclidean);
                            assertArrayEquals(nativeEuclidean, euclidean, "native EUCLIDEAN groups=" + groups);
                        }
                    }
                    codes[body + 4 * blockSize] = 3;
                    assertThrows(IllegalStateException.class, () -> PANAMA.ashScoreNeighborhood(codes, groups, blockSize,
                            capacity, lut, landmarkDot, new float[capacity]));
                }
            }
        }
    }

    @Test
    void int8NeighborhoodMatchesNative() {
        assumeTrue(PANAMA.supportsAshInt8NeighborhoodScoring(), "Panama INT8 neighborhood scoring unavailable");
        assumeTrue(nativeLoaded && NativeASHKernels.isInt8NeighborhoodAvailable(), "native INT8 neighborhood kernel unavailable");
        Random random = new Random(22);
        for (int bits : new int[]{3, 5, 6, 7, 8, 9}) {
            for (int dimensions : new int[]{8, 100, 384, 768}) {
                for (int blockSize : new int[]{8, 16, 32}) {
                    int capacity = 32, units = (dimensions - 1) / 8 + 1;
                    int body = units * bits * blockSize, blockBytes = body + 5 * blockSize;
                    int blocks = (capacity - 1) / blockSize + 1;
                    byte[] codes = new byte[blocks * blockBytes];
                    random.nextBytes(codes);
                    for (int b = 0; b < blocks; b++)
                        for (int lane = 0; lane < blockSize; lane++) {
                            writeHalf(codes, b * blockBytes + body + 2 * lane, (float) random.nextGaussian());
                            writeHalf(codes, b * blockBytes + body + 2 * blockSize + 2 * lane, (float) random.nextGaussian());
                            codes[b * blockBytes + body + 4 * blockSize + lane] = 0;
                        }
                    byte[] q = new byte[units * 8];
                    long sum = 0;
                    for (int j = 0; j < dimensions; j++) { q[j] = (byte) (random.nextInt(255) - 127); sum += q[j]; }
                    float invScale = 0.0123f, center = (float) (((1 << bits) - 1) * 0.5 * sum), landmarkDot = -0.4f;
                    float[] panama = new float[capacity], expected = new float[capacity];
                    PANAMA.ashScoreNeighborhoodInt8(codes, bits, units, blockSize, capacity, q, invScale, center, landmarkDot, panama);
                    NativeASHKernels.scoreNeighborhoodInt8(codes, bits, units, blockSize, capacity, q, invScale, center,
                            landmarkDot, 0f, expected);
                    assertArrayEquals(expected, panama, "bits=" + bits + " d=" + dimensions + " block=" + blockSize);
                }
            }
        }
    }

    @Test
    void canonicalLutDotMatchesScalar() {
        assumeTrue(PANAMA.supportsAshCanonicalLutScoring(), "Panama canonical LUT scoring unavailable");
        Random random = new Random(23);
        for (int groups : new int[]{1, 2, 3, 31, 32, 33, 64, 191, 192, 384}) {
            byte[] code = new byte[(groups + 1) / 2 + 5];
            random.nextBytes(code);
            float[] lut = new float[groups * 16];
            double scale = 0;
            for (int i = 0; i < lut.length; i++) { lut[i] = (float) random.nextGaussian(); scale += Math.abs(lut[i]); }
            float expected = ASHLutScoring.dotCanonical(code, 3, groups, lut);
            float panama = PANAMA.ashCanonicalLutDot(code, 3, groups, lut);
            assertEquals(expected, panama, 1e-6 * scale, "groups=" + groups);
            if (nativeLoaded && NativeASHKernels.isCanonicalLutAvailable())
                assertEquals(NativeASHKernels.dotCanonical(code, 3, groups, lut), panama, 1e-6 * scale, "native groups=" + groups);
        }
    }

    private static float[][] gaussian(Random random, int rows, int columns) {
        float[][] m = new float[rows][columns];
        for (float[] row : m) for (int j = 0; j < columns; j++) row[j] = (float) random.nextGaussian();
        return m;
    }

    @Test
    void fp32BatchProjectionMatchesSingle() {
        Random random = new Random(24);
        for (int[] shape : new int[][]{{1, 1}, {7, 15}, {8, 16}, {9, 17}, {384, 768}, {100, 1536}}) {
            float[][] matrix = gaussian(random, shape[0], shape[1]);
            var projection = PANAMA.prepareAshProjection(matrix);
            assertTrue(projection.supportsBatchProjection());
            for (int count : new int[]{1, 3, 4, 5, 8, 11, 32}) {
                float[][] inputs = gaussian(random, count, shape[1]);
                float[][] batch = new float[count][shape[0]];
                projection.projectBatch(inputs, batch, count);
                // Same arithmetic; the lane-reduction order may differ between the two compiled methods
                for (int i = 0; i < count; i++) {
                    float[] single = new float[shape[0]];
                    projection.project(inputs[i], single);
                    for (int r = 0; r < shape[0]; r++) {
                        double scale = 0;
                        for (int c = 0; c < shape[1]; c++) scale += Math.abs((double) matrix[r][c] * inputs[i][c]);
                        // A batch of one is projected exactly as project() is, so prepared and direct queries agree
                        assertEquals(single[r], batch[i][r], count == 1 ? 0 : 1e-6 * scale,
                                "shape " + shape[0] + "x" + shape[1] + " count=" + count + " i=" + i + " r=" + r);
                    }
                }
            }
        }
    }

    /** The native INT8 numbers, written out independently. */
    private static float[] int8Oracle(float[][] matrix, float[] query) {
        int rows = matrix.length, columns = matrix[0].length;
        float qmax = 0;
        for (float v : query) qmax = Math.max(qmax, Math.abs(v));
        float qs = qmax == 0 ? 1f : qmax / 127f;
        int[] coarse = new int[columns], fine = new int[columns];
        for (int i = 0; i < columns; i++) {
            coarse[i] = Math.max(-127, Math.min(127, (int) Math.rint(query[i] / qs)));
            fine[i] = Math.max(-127, Math.min(127, (int) Math.rint((query[i] / qs - (float) coarse[i]) * 254f)));
        }
        float[] out = new float[rows];
        for (int r = 0; r < rows; r++) {
            float wmax = 0;
            for (float v : matrix[r]) wmax = Math.max(wmax, Math.abs(v));
            float ws = wmax == 0 ? 1f : wmax / 127f;
            long c = 0, f = 0;
            for (int i = 0; i < columns; i++) {
                int w = Math.max(-127, Math.min(127, (int) Math.rint(matrix[r][i] / ws)));
                c += (long) w * coarse[i];
                f += (long) w * fine[i];
            }
            out[r] = (float) (c * 254 + f) * (1f / 254f) * qs * ws;
        }
        return out;
    }

    @Test
    void int8ProjectionMatchesNativeNumbers() {
        Random random = new Random(25);
        for (int[] shape : new int[][]{{1, 1}, {7, 15}, {16, 64}, {17, 65}, {384, 768}, {768, 1536}}) {
            float[][] matrix = gaussian(random, shape[0], shape[1]);
            var projection = PANAMA.prepareAshProjection(matrix).forQueryScoring(ASHQueryProjection.INT8);
            assertEquals("int8", projection.operandPrecision());
            int count = 10;
            float[][] inputs = gaussian(random, count, shape[1]);
            float[][] batch = new float[count][shape[0]];
            projection.projectBatch(inputs, batch, count);
            VectorUtilSupport.AshProjection nativeView = nativeLoaded && NativeASHInt8HighwayProjection.isAvailable()
                    ? NativeASHInt8HighwayProjection.prepare(java.lang.foreign.MemorySegment.ofArray(flatten(matrix)),
                            shape[0], shape[1]) : null;
            for (int i = 0; i < count; i++) {
                float[] single = new float[shape[0]];
                projection.project(inputs[i], single);
                String where = "shape " + shape[0] + "x" + shape[1] + " i=" + i;
                assertArrayEquals(int8Oracle(matrix, inputs[i]), single, where);
                assertArrayEquals(single, batch[i], where + " (batch)");
                if (nativeView != null) {
                    float[] expected = new float[shape[0]];
                    nativeView.project(inputs[i], expected);
                    assertArrayEquals(expected, single, where + " (native)");
                }
            }
            float[] bad = inputs[0].clone();
            bad[0] = Float.NaN;
            float[] out = new float[shape[0]];
            projection.project(bad, out);
            for (float v : out) assertTrue(Float.isNaN(v));
        }
        assertThrows(IllegalArgumentException.class,
                () -> PANAMA.prepareAshProjection(new float[2][65536]).forQueryScoring(ASHQueryProjection.INT8));
    }

    private static float[] flatten(float[][] matrix) {
        float[] flat = new float[matrix.length * matrix[0].length];
        for (int r = 0; r < matrix.length; r++) System.arraycopy(matrix[r], 0, flat, r * matrix[0].length, matrix[0].length);
        return flat;
    }

    /** Every finite half-precision weight decodes exactly: a one-hot query reads each weight back. */
    @Test
    void fp16ProjectionDecodesEveryHalfValue() {
        int columns = 16;
        java.util.List<Float> values = new java.util.ArrayList<>();
        for (int h = 0; h < 65536; h++) {
            float v = Float.float16ToFloat((short) h);
            if (Float.isFinite(v)) values.add(v);
        }
        int rows = (values.size() + columns - 1) / columns;
        float[][] matrix = new float[rows][columns];
        for (int i = 0; i < values.size(); i++) matrix[i / columns][i % columns] = values.get(i);
        var projection = PANAMA.prepareAshProjection(matrix).forQueryScoring(ASHQueryProjection.FP16);
        assertEquals("fp16", projection.operandPrecision());
        float[] out = new float[rows];
        for (int k = 0; k < columns; k++) {
            float[] oneHot = new float[columns];
            oneHot[k] = 1f;
            projection.project(oneHot, out);
            for (int r = 0; r < rows; r++) {
                float expected = matrix[r][k] * 1f + 0f;
                assertTrue(expected == out[r], "row " + r + " column " + k + ": " + expected + " vs " + out[r]);
            }
        }
        float[][] tooLarge = {{70000f}};
        assertThrows(IllegalArgumentException.class,
                () -> PANAMA.prepareAshProjection(tooLarge).forQueryScoring(ASHQueryProjection.FP16));
    }

    @Test
    void fp16ProjectionMatchesOracle() {
        Random random = new Random(26);
        for (int[] shape : new int[][]{{1, 1}, {7, 15}, {384, 768}}) {
            float[][] matrix = gaussian(random, shape[0], shape[1]);
            var projection = PANAMA.prepareAshProjection(matrix).forQueryScoring(ASHQueryProjection.FP16);
            float[][] inputs = gaussian(random, 5, shape[1]);
            float[][] batch = new float[5][shape[0]];
            projection.projectBatch(inputs, batch, 5);
            for (int i = 0; i < 5; i++) {
                float[] single = new float[shape[0]];
                projection.project(inputs[i], single);
                for (int r = 0; r < shape[0]; r++) {
                    double sum = 0, scale = 0;
                    for (int c = 0; c < shape[1]; c++) {
                        double p = (double) Float.float16ToFloat(Float.floatToFloat16(matrix[r][c]))
                                * Float.float16ToFloat(Float.floatToFloat16(inputs[i][c]));
                        sum += p;
                        scale += Math.abs(p);
                    }
                    assertEquals(sum, single[r], 1e-5 * scale + 1e-30, "shape " + shape[0] + "x" + shape[1] + " r=" + r);
                    assertEquals(sum, batch[i][r], 1e-5 * scale + 1e-30, "batch shape " + shape[0] + "x" + shape[1] + " r=" + r);
                }
            }
        }
    }

    /** The 2- and 4-bit direct-node dots against the scalar decode and, where loaded, native. */
    @Test
    void projectionDotsMatchScalarAndNative() {
        assumeTrue(PANAMA.supportsAshProjectionScoring(), "Panama projection scoring unavailable");
        var scalar = new DefaultVectorUtilSupport();
        Random random = new Random(27);
        for (int bits : new int[]{2, 4}) {
            for (int dimensions : new int[]{1, 3, 15, 16, 17, 63, 64, 65, 127, 128, 384, 767, 768, 1536}) {
                float[] query = new float[dimensions];
                double scale = 0;
                for (int i = 0; i < dimensions; i++) { query[i] = (float) random.nextGaussian(); scale += 8 * Math.abs(query[i]); }
                byte[] code = new byte[(dimensions * bits + 7) / 8 + 3];
                random.nextBytes(code);
                float panama = PANAMA.ashProjectionDot(query, code, 3, dimensions, bits);
                float expected = scalar.ashProjectionDot(query, code, 3, dimensions, bits);
                assertEquals(expected, panama, 1e-6 * scale, "bits=" + bits + " d=" + dimensions);
                if (nativeLoaded && (bits == 2 ? NativeASHKernels.isTwoBitDotAvailable() : NativeASHKernels.isFourBitDotAvailable())) {
                    float nat = bits == 2 ? NativeASHKernels.dotTwoBit(query, code, 3, dimensions)
                            : NativeASHKernels.dotFourBit(query, code, 3, dimensions);
                    assertEquals(nat, panama, 1e-6 * scale, "native bits=" + bits + " d=" + dimensions);
                }
            }
        }
    }

    @Test
    void panamaOffersEveryQueryPrecision() {
        for (var precision : ASHQueryProjection.values()) assertNull(PANAMA.ashQueryProjectionUnsupportedReason(precision));
    }
}
