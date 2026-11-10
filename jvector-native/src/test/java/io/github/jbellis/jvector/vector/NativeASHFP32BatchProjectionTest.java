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

import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.cnative.LibraryLoader;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.SymbolLookup;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Executors;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Actual optional oneDNN FP32 execution, separate from Highway's arithmetic tests. */
class NativeASHFP32BatchProjectionTest {
    @BeforeAll
    static void requireBackend() {
        boolean nativeAvailable = LibraryLoader.loadJvector();
        if (Boolean.getBoolean("jvector.test.requireNative")) assertTrue(nativeAvailable);
        else assumeTrue(nativeAvailable);
        var projection = prepared(new float[][]{{1}});
        if (Boolean.getBoolean("jvector.test.requireFp32OneDnn"))
            assertInstanceOf(NativeASHFP32BatchProjection.class, projection);
        else assumeTrue(projection instanceof NativeASHFP32BatchProjection, "Optional oneDNN FP32 unavailable");
    }

    private static VectorUtilSupport.AshProjection prepared(float[][] matrix) {
        return NativeASHFP32BatchProjection.prepare(new NativeASHProjection(matrix));
    }

    private static float[][] random(Random random, int rows, int cols) {
        float[][] result = new float[rows][cols];
        for (float[] row : result) for (int c = 0; c < cols; c++) row[c] = random.nextFloat() - .5f;
        return result;
    }

    private static void oracle(float[][] matrix, float[][] inputs, float[][] outputs, int count) {
        for (int b = 0; b < count; b++) for (int r = 0; r < matrix.length; r++) {
            double sum = 0, magnitude = 0;
            for (int c = 0; c < matrix[r].length; c++) {
                double value = (double) matrix[r][c] * inputs[b][c];
                sum += value; magnitude += Math.abs(value);
            }
            assertEquals(sum, outputs[b][r], 5e-6 * magnitude + 1e-7);
        }
    }

    @Test
    void optionalSymbolExecutesOneDnnWithoutHighwayFallback() throws Throwable {
        var descriptor = FunctionDescriptor.of(
                JAVA_INT,
                ADDRESS, ADDRESS,
                JAVA_INT, JAVA_INT,
                JAVA_INT, ADDRESS);
        var symbol = SymbolLookup.loaderLookup()
                .find("jvector_ash_fp32_project_batch").orElseThrow();
        var call = Linker.nativeLinker().downcallHandle(symbol, descriptor);
        try (var arena = Arena.ofConfined()) {
            var matrix = arena.allocate(4, 4); var input = arena.allocate(4, 4);
            var output = arena.allocate(4, 4);
            matrix.set(JAVA_FLOAT, 0, 2f);
            input.set(JAVA_FLOAT, 0, 3f);
            int status = (int) call.invokeExact(matrix, input, 1, 1, 1, output);
            assertEquals(0, status, "Actual oneDNN SGEMM must succeed, not compatibility fallback");
            assertEquals(6f, output.get(JAVA_FLOAT, 0));
            assertEquals(2, (int) call.invokeExact(matrix, input, 0, 1, 1, output));
        }
    }

    @Test
    void queryFallbackRetainsOwnerIdentityAndDoesNotDoubleChargeWeights() {
        String previous = System.getProperty("jvector.ash.queryProjection");
        try {
            System.setProperty("jvector.ash.queryProjection", "fp32");
            var projection = prepared(new float[][]{{1, 2}, {3, 4}});
            long bytes = projection.ramBytesUsed();
            assertSame(projection, projection.forQueryScoring());
            assertEquals(bytes, projection.ramBytesUsed());
            System.setProperty("jvector.ash.queryProjection", "auto");
            // Finite FP32 weights outside FP16 range force the automatic FP32 fallback.
            var fullRange = prepared(new float[][]{{70000f}});
            assertSame(fullRange, fullRange.forQueryScoring());
            float[] out = new float[1];
            fullRange.forQueryScoring().project(new float[]{2f}, out);
            assertEquals(140000f, out[0]);
            assertEquals(Float.BYTES + RamUsageEstimator.shallowSizeOfInstance(NativeASHProjection.class),
                    fullRange.ramBytesUsed());
        } finally {
            if (previous == null) System.clearProperty("jvector.ash.queryProjection");
            else System.setProperty("jvector.ash.queryProjection", previous);
        }
    }

    @Test
    void fp32QueryViewAdvertisesBatchingAndMatchesSingleQueries() {
        String previous = System.getProperty("jvector.ash.queryProjection");
        try {
            System.setProperty("jvector.ash.queryProjection", "fp32");
            Random random = new Random(5521);
            float[][] matrix = random(random, 64, 128);
            var query = prepared(matrix).forQueryScoring();
            // Queued ASH search prepares queries in batches only when the view says it can.
            assertTrue(query.supportsBatchProjection());
            float[][] inputs = random(random, 8, 128);
            float[][] batched = new float[8][64];
            query.projectBatch(inputs, batched, 8);
            for (int i = 0; i < 8; i++) {
                float[] single = new float[64];
                query.project(inputs[i], single);
                for (int r = 0; r < 64; r++) assertEquals(single[r], batched[i][r], 1e-4f);
            }
        } finally {
            if (previous == null) System.clearProperty("jvector.ash.queryProjection");
            else System.setProperty("jvector.ash.queryProjection", previous);
        }
    }

    @Test
    void fp32ArithmeticSupportsArbitraryBatchSizesAndSimdTails() {
        Random random = new Random(32145);
        for (int[] shape : new int[][]{{1,1},{3,7},{33,65},{384,768},{1536,3072}}) {
            float[][] matrix = random(random, shape[0], shape[1]);
            var projection = prepared(matrix);
            for (int count : new int[]{0,1,3,4,8,17,32,33,65}) {
                float[][] inputs = random(random, count + 1, shape[1]);
                float[][] saved = Arrays.stream(inputs).map(float[]::clone).toArray(float[][]::new);
                float[][] outputs = new float[count + 1][shape[0]];
                Arrays.fill(outputs[count], 123f);
                projection.projectBatch(inputs, outputs, count);
                oracle(matrix, inputs, outputs, count);
                for (int i = 0; i <= count; i++) assertArrayEquals(saved[i], inputs[i]);
                for (float value : outputs[count]) assertEquals(123f, value);
            }
            assertEquals((long) shape[0] * shape[1] * Float.BYTES
                    + RamUsageEstimator.shallowSizeOfInstance(NativeASHProjection.class), projection.ramBytesUsed());
        }
    }

    @Test
    void preservesFullPrecisionWeightsAndCopiesModelOnlyAtPreparation() {
        float weight = 1f + Math.scalb(1f, -12), value = 1f + Math.scalb(1f, -13);
        float[][] matrix = {{weight}};
        var projection = prepared(matrix);
        matrix[0][0] = 0;
        float[][] outputs = new float[33][1], inputs = new float[33][1];
        for (float[] input : inputs) input[0] = value;
        projection.projectBatch(inputs, outputs, 33);
        for (float[] output : outputs) assertEquals(weight * value, output[0], 0f);
        float[] singleton = new float[1];
        projection.project(new float[]{value}, singleton);
        assertEquals(weight * value, singleton[0], 0f);
        var fullRange = prepared(new float[][]{{70000f}});
        float[][] rangeOutput = new float[1][1];
        fullRange.projectBatch(new float[][]{{2f}}, rangeOutput, 1);
        assertEquals(140000f, rangeOutput[0][0], 0f);
    }

    @Test
    void validatesEveryInputBeforeWritingOutputs() {
        var projection = prepared(new float[][]{{1,2},{3,4}});
        float[][] x = {{1,2},{3,4}}, y = {{99,99},{99,99}};
        assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(x, y, -1));
        assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(x, y, 3));
        assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(x, x, 2));
        assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(x, new float[][]{y[0],x[0]}, 2));
        assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(x, new float[][]{y[0],y[0]}, 2));
        assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(new float[][]{x[0],{1}}, y, 2));
        assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(new float[][]{{1}}, new float[][]{{1,2}}, 1));
        assertThrows(NullPointerException.class, () -> projection.projectBatch(null, y, 1));
        assertThrows(NullPointerException.class, () -> projection.projectBatch(new float[][]{x[0],null}, y, 2));
        for (float[] row : y) for (float value : row) assertEquals(99f, value);
        projection.projectBatch(new float[0][], new float[0][], 0);
        float[][] largeX = new float[65][2], largeY = new float[65][2];
        largeY[64] = largeX[17];
        assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(largeX, largeY, 65));
    }

    @Test
    void concurrentCallersCanSwitchMatricesAndSurviveCollection() throws Exception {
        Random random = new Random(8419);
        float[][] a = random(random, 33, 65), b = random(random, 129, 257);
        var first = prepared(a); var second = prepared(b);
        CyclicBarrier barrier = new CyclicBarrier(8);
        try (var pool = Executors.newFixedThreadPool(8)) {
            var tasks = new ArrayList<java.util.concurrent.Callable<Void>>();
            for (int t = 0; t < 8; t++) tasks.add(() -> {
                float[][] x = random(new Random(14), 33, 65), z = random(new Random(15), 5, 257);
                float[][] y = new float[33][33], w = new float[5][129];
                barrier.await();
                for (int i = 0; i < 8; i++) {
                    first.projectBatch(x, y, 33); second.projectBatch(z, w, 5);
                    oracle(a, x, y, 33); oracle(b, z, w, 5);
                    if (i == 4) System.gc();
                }
                return null;
            });
            for (var result : pool.invokeAll(tasks)) result.get();
        }
    }
}
