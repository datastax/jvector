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

class TestPQAssembly {
    private final PanamaVectorUtilSupport support = new PanamaVectorUtilSupport();

    @Test
    void matchesPackedTableAcrossWidthsAndTails() {
        Random random = new Random(712903);
        for (int k : new int[] {1, 2, 16, 128, 256}) {
            for (int m : new int[] {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 128, 192, 193, 257}) {
                int block = k * (k + 1) / 2;
                float[] table = new float[m * block];
                for (int i = 0; i < table.length; i++) table[i] = random.nextFloat() * 2 - 1;
                for (int offset : new int[] {0, 1, 7}) {
                    byte[] a = new byte[offset + m];
                    byte[] b = new byte[offset + 2 + m];
                    for (int i = 0; i < m; i++) {
                        a[offset + i] = (byte) random.nextInt(k);
                        b[offset + 2 + i] = (byte) random.nextInt(k);
                    }
                    check(table, m, k, a, offset, b, offset + 2);
                }
            }
        }
    }

    @Test
    void handlesEveryUnsignedCodePairAndSymmetry() {
        int k = 256;
        int m = 17; // Both vector widths execute a SIMD group and a scalar tail.
        float[] table = new float[m * k * (k + 1) / 2];
        for (int i = 0; i < table.length; i++) table[i] = (i % 127) - 63.25f;
        byte[] a = new byte[m + 3];
        byte[] b = new byte[m + 5];
        for (int first = 0; first < 65536; first += m) {
            for (int i = 0; i < m; i++) {
                int pair = (first + i) & 65535;
                a[i + 3] = (byte) (pair >>> 8);
                b[i + 5] = (byte) pair;
            }
            check(table, m, k, a, 3, b, 5);
            check(table, m, k, b, 5, a, 3);
        }
    }

    @Test
    void concurrentCallsKeepScratchTablesIndependent() throws Exception {
        try (var executor = java.util.concurrent.Executors.newFixedThreadPool(4)) {
            var tasks = new java.util.ArrayList<java.util.concurrent.Callable<Void>>();
            for (int worker = 0; worker < 4; worker++) {
                final int id = worker;
                tasks.add(() -> {
                    int m = 193, k = 16;
                    float[] table = new float[m * k * (k + 1) / 2];
                    byte[] a = new byte[m], b = new byte[m];
                    for (int repeat = 0; repeat < 100; repeat++) {
                        for (int i = 0; i < table.length; i++) table[i] = (i + id + repeat) % 11;
                        for (int i = 0; i < m; i++) {
                            a[i] = (byte) ((i + id) % k);
                            b[i] = (byte) ((i + repeat) % k);
                        }
                        check(table, m, k, a, 0, b, 0);
                    }
                    return null;
                });
            }
            for (var result : executor.invokeAll(tasks)) result.get();
        }
    }

    private void check(float[] table, int m, int k, byte[] a, int ao, byte[] b, int bo) {
        var data = new ArrayVectorFloat(table);
        var left = new ArrayByteSequence(a);
        var right = new ArrayByteSequence(b);
        assertEquals(oracle(table, m, k, a, ao, b, bo, 1),
                support.assembleAndSumPQ_128(data, m, left, ao, right, bo, k), 1e-4f);
        assertEquals(oracle(table, m, k, a, ao, b, bo, PanamaVectorUtilSupport.PREFERRED_BIT_SIZE / 32),
                support.assembleAndSumPQ(data, m, left, ao, right, bo, k), 1e-4f);
        assertEquals(oracle(table, m, k, a, ao, b, bo, PanamaVectorUtilSupport.PREFERRED_BIT_SIZE / 32),
                support.assembleAndSumPQ(data, m, left.slice(ao, m), 0, right.slice(bo, m), 0, k), 1e-4f);
        assertEquals(oracle(table, m, k, a, ao, b, bo, 8),
                support.assembleAndSumPQ_256(data, m, left, ao, right, bo, k), 1e-4f);
        assertEquals(oracle(table, m, k, a, ao, b, bo, 16),
                support.assembleAndSumPQ_512(data, m, left, ao, right, bo, k), 1e-4f);
    }

    private static float oracle(float[] table, int m, int k,
                                byte[] a, int ao, byte[] b, int bo, int lanes) {
        float[] sums = new float[lanes];
        int complete = m / lanes * lanes;
        for (int i = 0; i < complete; i++) {
            sums[i % lanes] += lookup(table, k, i, a[ao + i] & 255, b[bo + i] & 255);
        }
        float sum = 0f;
        for (float lane : sums) sum += lane;
        for (int i = complete; i < m; i++) {
            sum += lookup(table, k, i, a[ao + i] & 255, b[bo + i] & 255);
        }
        return sum;
    }

    private static float lookup(float[] table, int k, int subspace, int a, int b) {
        int lo = Math.min(a, b), hi = Math.max(a, b);
        // Enumerate preceding rows independently of the vectorized index formula.
        int index = hi - lo;
        for (int row = 0; row < lo; row++) index += k - row;
        return table[subspace * k * (k + 1) / 2 + index];
    }
}
