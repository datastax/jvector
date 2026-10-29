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

import io.github.jbellis.jvector.vector.cnative.LibraryLoader;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Independent arithmetic oracles cover vector tails, both FFM modes and shared-model use. */
class NativeASHPreparationTest {
    @BeforeAll
    static void loadLibrary() {
        assumeTrue(LibraryLoader.loadJvector(), "Native library unavailable");
        assertTrue(NativeASHProjection.isAvailable());
        assertTrue(NativeASHSymmetricKernels.hasPairKernel());
        assertTrue(NativeASHSymmetricKernels.hasLutKernel());
    }

    @Test
    void projectionMatchesDoubleOracleAndIsThreadSafe() throws Exception {
        Random random = new Random(93124);
        for (int[] shape : new int[][]{{1,1},{3,7},{4,15},{5,17},{17,31},{33,129},{257,513},{1536,3072}}) {
            int rows = shape[0], columns = shape[1];
            float[][] matrix = new float[rows][columns];
            float[] query = new float[columns];
            for (float[] row : matrix) for (int j = 0; j < columns; j++) row[j] = random.nextFloat() - .5f;
            for (int j = 0; j < columns; j++) query[j] = random.nextFloat() - .5f;
            var projection = new NativeASHProjection(matrix);
            assertEquals((long) rows * columns * Float.BYTES, projection.ramBytesUsed());
            float[] expected = new float[rows];
            double[] magnitude = new double[rows];
            for (int i = 0; i < rows; i++) {
                double sum = 0;
                for (int j = 0; j < columns; j++) {
                    double product = (double) matrix[i][j] * query[j];
                    sum += product; magnitude[i] += Math.abs(product);
                }
                expected[i] = (float) sum;
            }
            float[] original = query.clone();
            try (var workers = Executors.newFixedThreadPool(4)) {
                var tasks = new ArrayList<java.util.concurrent.Callable<Void>>();
                for (int repeat = 0; repeat < 8; repeat++) tasks.add(() -> {
                    float[] out = new float[rows];
                    projection.project(query, out);
                    for (int i = 0; i < rows; i++) assertEquals(expected[i], out[i], 2e-6 * magnitude[i] + 1e-7);
                    return null;
                });
                for (var result : workers.invokeAll(tasks)) result.get();
            }
            assertArrayEquals(original, query);
        }
    }

    @Test
    void packedPairsMatchIndependentDecoderIncludingDirtyPadding() {
        Random random = new Random(55191);
        for (int bits : new int[]{1,2,4}) {
            for (int dimensions : new int[]{0,1,2,3,7,15,16,17,31,32,33,63,64,65,127,128,129,257,384,512,768,1536,65537}) {
                long[] aBits = new long[(dimensions + 63) / 64], bBits = aBits.clone();
                byte[] a = new byte[(dimensions * bits + 7) / 8], b = a.clone();
                for (int i = 0; i < aBits.length; i++) { aBits[i] = random.nextLong(); bBits[i] = random.nextLong(); }
                random.nextBytes(a); random.nextBytes(b);
                long total = 0;
                for (int j = 0; j < dimensions; j++) {
                    if (bits == 1) total += (((aBits[j/64] ^ bBits[j/64]) >>> (j%64)) & 1) == 0 ? 1 : -1;
                    else total += (long) doubled(a,j,bits) * doubled(b,j,bits);
                }
                float expected = bits == 1 ? (float) total : total * .25f;
                assertEquals(expected, NativeASHSymmetricKernels.dot(aBits,a,bBits,b,dimensions,bits), 0f);
                assertEquals(expected, NativeASHSymmetricKernels.dot(bBits,b,aBits,a,dimensions,bits), 0f);
            }
        }
    }

    private static int doubled(byte[] codes, int j, int bits) {
        int field = (codes[j * bits / 8] >>> (j * bits % 8)) & ((1 << bits) - 1);
        int sign = 1 << (bits - 1), magnitude = 2 * (field & (sign - 1)) + 1;
        return (field & sign) == 0 ? -magnitude : magnitude;
    }

    @Test
    void integerLutHandlesOffsetsTailsAndOverflow() {
        Random random = new Random(1021);
        int stride = 37, lane = 3;
        for (int groups : new int[]{0,1,2,31,63,64,65,257,4097,65539}) {
            byte[] codes = new byte[((groups+1)/2)*stride]; random.nextBytes(codes);
            short[] lut = new short[groups*32];
            for (int g = 0; g < groups; g++) for (int i = 0; i < 16; i++) {
                short value = groups == 65539 ? Short.MAX_VALUE : (short) random.nextInt();
                lut[g*32+i] = lut[g*32+i+16] = value;
            }
            for (int count : new int[]{0,1,7,16,17,32,34}) {
                float[] out = new float[count+2]; Arrays.fill(out, Float.NaN);
                NativeASHSymmetricKernels.scoreLut(codes,groups,stride,lane,count,lut,out,1);
                assertTrue(Float.isNaN(out[0])); assertTrue(Float.isNaN(out[count+1]));
                for (int i = 0; i < count; i++) {
                    long sum = 0;
                    for (int g = 0; g < groups; g++) sum += lut[g*32+((codes[(g/2)*stride+lane+i] >>> (4*(g%2))) & 15)];
                    assertEquals(sum*.25f,out[i+1],0f);
                }
            }
        }
    }

    @Test
    void rejectsUnsafeInputsBeforeNativeCall() {
        var projection = new NativeASHProjection(new float[][]{{1,2},{3,4}});
        float[] alias = {1,2};
        assertThrows(IllegalArgumentException.class, () -> projection.project(alias,alias));
        assertThrows(IllegalArgumentException.class, () -> projection.project(new float[1],new float[2]));
        assertThrows(IllegalArgumentException.class, () -> new NativeASHProjection(new float[][]{{1},{2,3}}));
        assertThrows(IndexOutOfBoundsException.class, () -> NativeASHSymmetricKernels.dot(null,new byte[1],null,new byte[1],9,2));
        assertThrows(IllegalArgumentException.class, () -> NativeASHSymmetricKernels.dot(null,null,null,null,-1,2));
        assertThrows(IllegalArgumentException.class, () -> NativeASHSymmetricKernels.dot(null,null,null,null,1,3));
        assertThrows(IndexOutOfBoundsException.class, () -> NativeASHSymmetricKernels.scoreLut(new byte[8],1,8,7,2,new short[32],new float[2],0));
        assertThrows(IndexOutOfBoundsException.class, () -> NativeASHSymmetricKernels.scoreLut(new byte[8],1,8,0,8,new short[31],new float[8],0));
    }
}
