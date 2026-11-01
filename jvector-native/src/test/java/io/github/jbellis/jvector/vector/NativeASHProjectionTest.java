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

import java.lang.foreign.SymbolLookup;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** FP32 batch arithmetic, caller-buffer contracts and shared-matrix lifetime. */
class NativeASHProjectionTest {
    @BeforeAll
    static void loadNative() {
        boolean available = LibraryLoader.loadJvector();
        if (Boolean.getBoolean("jvector.test.requireNative")) assertTrue(available);
        else assumeTrue(available, "Native library unavailable");
        assertTrue(SymbolLookup.loaderLookup().find("ash_project_batch_f32").isPresent(),
                "Tests must exercise the new native batch kernel, not compatibility fallback");
    }

    private static float[][] randomRows(Random random, int rows, int columns) {
        float[][] result = new float[rows][columns];
        for (float[] row : result)
            for (int i = 0; i < columns; i++) row[i] = random.nextFloat() - .5f;
        return result;
    }

    private static void checkOracle(float[][] matrix, float[][] inputs, float[][] out, int count) {
        for (int b = 0; b < count; b++) {
            for (int row = 0; row < matrix.length; row++) {
                double expected = 0, magnitude = 0;
                for (int col = 0; col < matrix[row].length; col++) {
                    double product = (double) matrix[row][col] * inputs[b][col];
                    expected += product;
                    magnitude += Math.abs(product);
                }
                assertEquals(expected, out[b][row], 5e-6 * magnitude + 1e-7,
                        "vector " + b + ", row " + row);
            }
        }
    }

    @Test
    void providerRetainsHighwayWhenOptionalFp32IsUnavailable() {
        var projection = new NativeVectorUtilSupport().prepareAshProjection(new float[][]{{1,2},{3,4}});
        if (Boolean.getBoolean("jvector.test.requireFp32OneDnn"))
            assertInstanceOf(NativeASHFP32BatchProjection.class, projection);
        if (Boolean.getBoolean("jvector.test.requireFp32Absent"))
            assertInstanceOf(NativeASHProjection.class, projection);
        float[][] outputs = new float[1][2];
        projection.projectBatch(new float[][]{{2,3}}, outputs, 1);
        assertArrayEquals(new float[]{8,18}, outputs[0]);
    }

    @Test
    void nativeAndPanamaMatchDoubleOracleAcrossBatchAndSimdTails() {
        Random random = new Random(829104);
        for (int[] shape : new int[][]{{1,1}, {3,7}, {17,31}, {33,65}, {384,768}, {1536,3072}}) {
            float[][] matrix = randomRows(random, shape[0], shape[1]);
            var nativeProjection = new NativeASHProjection(matrix);
            var panama = new PanamaVectorUtilSupport().prepareAshProjection(matrix);
            long matrixBytes = (long) shape[0] * shape[1] * Float.BYTES;
            assertEquals(matrixBytes, nativeProjection.ramBytesUsed());
            for (int count : new int[]{0,1,3,4,5,8,17,32,33,65}) {
                float[][] inputs = randomRows(random, count + 1, shape[1]);
                float[][] out = new float[count + 1][shape[0]];
                float[][] reference = new float[count + 1][shape[0]];
                float[][] saved = Arrays.stream(inputs).map(float[]::clone).toArray(float[][]::new);
                Arrays.fill(out[count], 123f);
                nativeProjection.projectBatch(inputs, out, count);
                panama.projectBatch(inputs, reference, count);
                checkOracle(matrix, inputs, out, count);
                for (int b = 0; b < count; b++) {
                    float[][] one = new float[1][shape[0]];
                    nativeProjection.projectBatch(new float[][]{inputs[b]}, one, 1);
                    assertArrayEquals(out[b], one[0], "Batch size must not change native arithmetic");
                }
                checkOracle(matrix, inputs, reference, count);
                for (int b = 0; b <= count; b++) assertArrayEquals(saved[b], inputs[b]);
                for (float value : out[count]) assertEquals(123f, value);
            }
            assertEquals(2 * matrixBytes, nativeProjection.ramBytesUsed());
        }
    }

    @Test
    void retainsFp32OperandsIncludingValuesThatRoundInFp16() {
        float weight = 1f + Math.scalb(1f, -12);
        float value = 1f + Math.scalb(1f, -13);
        var projection = new NativeASHProjection(new float[][]{{weight}});
        float[][] out = new float[5][1];
        projection.projectBatch(new float[][]{{value},{value},{value},{value},{value}}, out, 5);
        for (float[] row : out) assertEquals(weight * value, row[0], 0f);
        float[] single = new float[1];
        projection.project(new float[]{value}, single);
        assertEquals(weight * value, single[0], 0f);
        assertNotEquals(1f, out[0][0]);
    }

    @Test
    void validatesShapesCountsAndAliasingBeforeWriting() {
        for (var projection : new VectorUtilSupport.AshProjection[]{
                new NativeASHProjection(new float[][]{{1,2},{3,4}}),
                new PanamaVectorUtilSupport().prepareAshProjection(new float[][]{{1,2},{3,4}})}) {
            float[][] x = {{1,2},{3,4}}, y = {{99,99},{99,99}};
            assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(x, y, -1));
            assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(x, y, 3));
            assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(x, x, 2));
            assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(x, new float[][]{y[0],y[0]}, 2));
            assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(x, new float[][]{y[0],x[0]}, 2));
            assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(new float[][]{x[0],{1}}, y, 2));
            assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(new float[][]{{1}}, new float[][]{{1,2}}, 1));
            assertThrows(NullPointerException.class, () -> projection.projectBatch(null, y, 1));
            assertThrows(NullPointerException.class, () -> projection.projectBatch(new float[][]{null}, y, 1));
            for (float[] row : y) for (float value : row) assertEquals(99f, value);
            projection.projectBatch(new float[0][], new float[0][], 0);
            float[][] manyInputs = new float[65][2], manyOutputs = new float[65][2];
            manyOutputs[64] = manyInputs[17];
            assertThrows(IllegalArgumentException.class,
                    () -> projection.projectBatch(manyInputs, manyOutputs, 65));
            manyOutputs[64] = manyOutputs[3];
            assertThrows(IllegalArgumentException.class,
                    () -> projection.projectBatch(manyInputs, manyOutputs, 65));
        }
    }

    @Test
    void lazyPackingAndStagingRemainSafeAcrossWorkersAndModels() throws Exception {
        Random random = new Random(40215);
        float[][] a = randomRows(random, 33, 65), b = randomRows(random, 129, 257);
        var first = new NativeASHProjection(a);
        var second = new NativeASHProjection(b);
        CyclicBarrier start = new CyclicBarrier(8);
        try (var pool = Executors.newFixedThreadPool(8)) {
            var tasks = new ArrayList<java.util.concurrent.Callable<Void>>();
            for (int t = 0; t < 8; t++) tasks.add(() -> {
                float[][] inputA = randomRows(new Random(42), 35, 65);
                float[][] inputB = randomRows(new Random(43), 5, 257);
                float[][] outA = new float[35][33], outB = new float[5][129];
                start.await();
                for (int call = 0; call < 8; call++) {
                    first.projectBatch(inputA, outA, 35);
                    second.projectBatch(inputB, outB, 5);
                    checkOracle(a, inputA, outA, 35);
                    checkOracle(b, inputB, outB, 5);
                    if (call == 4) System.gc();
                }
                return null;
            });
            for (var task : pool.invokeAll(tasks)) task.get();
        }
    }
}
