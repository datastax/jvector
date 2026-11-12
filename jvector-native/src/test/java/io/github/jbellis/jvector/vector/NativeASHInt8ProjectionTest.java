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
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.parallel.ResourceLock;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** INT8 numerical, lifecycle, dispatch and compatibility regression tests. */
@ResourceLock("java.lang.System.properties")
class NativeASHInt8ProjectionTest {
    private static final String PROPERTY = "jvector.ash.queryProjection";

    @BeforeAll
    static void loadNative() {
        boolean available = LibraryLoader.loadJvector();
        if (Boolean.getBoolean("jvector.test.requireInt8")) assertTrue(available, "Native library required");
        else assumeTrue(available, "Native library unavailable");
    }

    private static final String[] PATH_FLAGS = {"fma", "avx2", "avx_vnni", "avx512f", "avx512vl", "avx512_vnni",
            "avx512_bf16", "avx512_fp16", "amx_tile", "amx_int8"};

    /** The CPU and settings that select the INT8 path, for failure messages; built only when an assertion fails. */
    private static String cpuContext() {
        String vendor = "?", model = "?", flags = "";
        try {
            for (String line : Files.readAllLines(Path.of("/proc/cpuinfo"))) {
                int colon = line.indexOf(':');
                if (colon < 0) continue;
                String key = line.substring(0, colon).trim(), value = line.substring(colon + 1).trim();
                if (key.equals("vendor_id") && vendor.equals("?")) vendor = value;
                else if (key.equals("model name") && model.equals("?")) model = value;
                else if (key.equals("flags") && flags.isEmpty()) {
                    var present = new java.util.HashSet<>(Arrays.asList(value.split("\\s+")));
                    var kept = new ArrayList<String>();
                    for (String flag : PATH_FLAGS) if (present.contains(flag)) kept.add(flag);
                    flags = String.join(" ", kept);
                }
            }
        } catch (java.io.IOException | RuntimeException e) {
            model = "unreadable /proc/cpuinfo (" + e + ")";
        }
        return " [cpu: " + vendor + ", " + model + "; flags: " + flags + "; JVECTOR_ASH_INT8_ONEDNN="
                + System.getenv("JVECTOR_ASH_INT8_ONEDNN") + "; os.arch=" + System.getProperty("os.arch") + "]";
    }

    private static VectorUtilSupport.AshProjection prepare(float[][] weights) {
        var base = new NativeASHProjection(weights);
        try {
            return NativeASHInt8Projection.prepare(base.matrix(), base.rows(), base.columns());
        } catch (UnsupportedOperationException unsupported) {
            if (Boolean.getBoolean("jvector.test.requireInt8")) throw unsupported;
            assumeTrue(false, unsupported.getMessage());
            throw unsupported;
        }
    }

    private static float scale(float[] values) {
        float max = 0;
        for (float value : values) max = Math.max(max, Math.abs(value));
        return max == 0 ? 1 : max / 127f;
    }

    private static int code(float value, float scale) {
        return Math.max(-127, Math.min(127, (int) Math.rint(value / scale)));
    }

    // The query is two INT8 digits: value / scale ~ coarse + fine / 254, with fine the coarse remainder.
    private static int fine(float value, float scale) {
        return Math.max(-127, Math.min(127, (int) Math.rint((value / scale - code(value, scale)) * 254f)));
    }

    private static float[] oracle(float[][] matrix, float[] query) {
        float queryScale = scale(query);
        float[] out = new float[matrix.length];
        for (int row = 0; row < matrix.length; row++) {
            float weightScale = scale(matrix[row]);
            long coarse = 0, fine = 0;
            for (int column = 0; column < query.length; column++) {
                int weight = code(matrix[row][column], weightScale);
                coarse += (long) code(query[column], queryScale) * weight;
                fine += (long) fine(query[column], queryScale) * weight;
            }
            out[row] = ((float) coarse + (float) fine * (1.0f / 254.0f)) * queryScale * weightScale;
        }
        return out;
    }

    private static float[][] matrix(int rows, int columns, long seed) {
        var random = new Random(seed);
        float[][] matrix = new float[rows][columns];
        for (float[] row : matrix)
            for (int i = 0; i < row.length; i++) row[i] = random.nextFloat() - .5f;
        return matrix;
    }

    @Test
    void integerOraclePaddingTailsAndSingletonAgree() {
        float[][] matrix = matrix(33, 65, 19);
        matrix[0] = new float[65];
        var projection = prepare(matrix);
        float[][] inputs = matrix(17, 65, 20), outputs = new float[18][33];
        inputs[0] = new float[65];
        Arrays.fill(outputs[17], 42);
        for (int count : new int[]{0,1,2,3,4,5,6,7,8,9,17}) {
            projection.projectBatch(inputs, outputs, count);
            for (int i = 0; i < count; i++) {
                assertArrayEquals(oracle(matrix, inputs[i]), outputs[i], 1e-7f);
                float[] singleton = new float[33];
                projection.project(inputs[i], singleton);
                assertArrayEquals(singleton, outputs[i]);
            }
            for (float value : outputs[17]) assertEquals(42f, value);
        }
    }

    @Test
    void halfStepWeightsUseNearestEvenAndQueryHalfStepsAreExact() {
        float[] ties = {.5f, 1.5f, 2.5f, 3.5f, -.5f, -1.5f, -2.5f, -3.5f};
        float[] expected = {0, 2, 2, 4, 0, -2, -2, -4};
        float[][] weights = new float[32][64];
        float[] query = new float[64];
        query[63] = 127f; // Exact unit scale for both quantization paths.
        for (int row = 0; row < ties.length; row++) {
            weights[row][row] = 127f;
            query[row] = ties[row];
            weights[row + 8][row] = ties[row];
            weights[row + 8][62] = 127f;
        }
        var projection = prepare(weights);
        float[] output = new float[32];
        projection.project(query, output);
        // The query's fine digit carries each half step (127 / 254), so queries are exact; weights still
        // round half steps to the nearest even code.
        for (int row = 0; row < ties.length; row++) {
            assertEquals(127f * ties[row], output[row], 1e-4f);
            assertEquals(ties[row] * expected[row], output[row + 8], 1e-4f);
        }
        assertArrayEquals(oracle(weights, query), output);
    }

    @Test
    void fullRangeProductsDoNotSaturateThroughAccumulatorDimensionBound() {
        for (int columns : new int[]{768, 65535}) {
            float[][] matrix = new float[32][columns];
            float[] query = new float[columns];
            Arrays.fill(query, 1f);
            for (int row = 0; row < matrix.length; row++) Arrays.fill(matrix[row], row % 2 == 0 ? 1f : -1f);
            var projection = prepare(matrix);
            float[] output = new float[32];
            projection.project(query, output);
            assertArrayEquals(oracle(matrix, query), output);
            assertEquals(columns, output[0], .02f);
            assertEquals(-columns, output[1], .02f);
        }
        assertThrows(IllegalArgumentException.class, () -> prepare(new float[32][65536]));
    }

    @Test
    void everyTileSizeMatchesSingletonsThroughAccumulatorDimensionBound() {
        // Tiles of 2-8 rows may use a different integer kernel than single rows;
        // exact int32 sums must make every tile size agree bit for bit.
        for (int columns : new int[]{3, 1023, 65535}) {
            float[][] matrix = matrix(47, columns, columns);
            for (int column = 0; column < columns; column++) matrix[1][column] = column % 2 == 0 ? 1f : -1f;
            var projection = prepare(matrix);
            float[][] inputs = matrix(8, columns, columns + 1);
            Arrays.fill(inputs[7], 1f); // Saturated codes against the alternating row.
            for (int count = 1; count <= 8; count++) {
                float[][] outputs = new float[count][47];
                projection.projectBatch(inputs, outputs, count);
                for (int i = 0; i < count; i++) {
                    float[] singleton = new float[47];
                    projection.project(inputs[i], singleton);
                    final int c = count, r = i;
                    assertArrayEquals(singleton, outputs[i], () -> "columns=" + columns + " count=" + c + " row=" + r + cpuContext());
                    if (columns < 65535) assertArrayEquals(oracle(matrix, inputs[i]), outputs[i], 1e-7f,
                            () -> "oracle, columns=" + columns + " count=" + c + " row=" + r + cpuContext());
                }
            }
        }
    }

    @Test
    void fp32ScalesMatchPrototypeAndUnderflowFailsBeforeConversion() {
        for (float weight : new float[]{0, Float.MIN_NORMAL, 1e-38f, 1f}) {
            float[][] weights = new float[32][64];
            for (float[] row : weights) Arrays.fill(row, weight);
            var projection = prepare(weights);
            for (float value : new float[]{0, Float.MIN_NORMAL, .25f}) {
                float[] query = new float[64], output = new float[32];
                Arrays.fill(query, value);
                projection.project(query, output);
                assertArrayEquals(oracle(weights, query), output);
            }
        }
        float[][] tinyWeights = new float[32][64];
        for (float[] row : tinyWeights) Arrays.fill(row, Float.MIN_VALUE);
        assertThrows(IllegalArgumentException.class, () -> prepare(tinyWeights));
        var projection = prepare(matrix(32,64,100));
        float[] query = new float[64], output = new float[32];
        Arrays.fill(query, Float.MIN_VALUE);
        Arrays.fill(output, 42f);
        assertThrows(IllegalArgumentException.class, () -> projection.project(query, output));
        for (float value : output) assertEquals(42f, value);
    }

    @Test
    void invalidInputDoesNotPublishPartialBatchResults() {
        var projection = prepare(matrix(32, 64, 1));
        float[][] inputs = matrix(9, 64, 2), outputs = new float[9][32];
        for (float[] row : outputs) Arrays.fill(row, 42);
        assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(inputs, outputs, -1));
        assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(inputs, outputs, 10));
        assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(inputs, inputs, 2));
        assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(inputs,
                new float[][]{outputs[0], outputs[0]}, 2));
        for (float invalid : new float[]{Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY}) {
            inputs[8][0] = invalid;
            assertThrows(IllegalArgumentException.class, () -> projection.projectBatch(inputs, outputs, 9));
            for (float[] row : outputs) for (float value : row) assertEquals(42f, value);
        }
        assertThrows(NullPointerException.class, () -> projection.project(null, new float[32]));
        assertThrows(NullPointerException.class, () -> projection.project(new float[64], null));
        assertThrows(IllegalArgumentException.class, () -> projection.project(new float[1], new float[32]));
        assertThrows(IllegalArgumentException.class, () -> projection.project(new float[64], new float[1]));
        float[][] invalidMatrix = matrix(32, 64, 3);
        invalidMatrix[0][0] = Float.NaN;
        assertThrows(IllegalArgumentException.class, () -> prepare(invalidMatrix));
    }

    @Test
    void fp32ResultOverflowMatchesPrototype() {
        float[][] weights = new float[32][64];
        for (int row = 0; row < weights.length; row++)
            Arrays.fill(weights[row], row % 2 == 0 ? 1f : -1f);
        var projection = prepare(weights);
        float[] input = new float[64], output = new float[32];
        Arrays.fill(input, Float.MAX_VALUE);
        projection.project(input, output);
        assertArrayEquals(oracle(weights, input), output);
        assertEquals(Float.POSITIVE_INFINITY, output[0]);
        assertEquals(Float.NEGATIVE_INFINITY, output[1]);
    }

    @Test
    void concurrentFirstUseAndAlternatingModelsAndShapes() throws Exception {
        float[][][] weights = {matrix(33,65,3), matrix(64,128,4)};
        VectorUtilSupport.AshProjection[] projections = {prepare(weights[0]), prepare(weights[1])};
        try (var pool = Executors.newFixedThreadPool(16)) {
            var work = new ArrayList<Callable<Void>>();
            for (int thread = 0; thread < 16; thread++) {
                final int seed = thread;
                work.add(() -> {
                    for (int trial = 0; trial < 16; trial++) {
                        int model = (trial + seed) % 2, count = 1 + (trial + seed) % 17;
                        float[][] inputs = matrix(count, weights[model][0].length, seed * 100L + trial);
                        float[][] outputs = new float[count][weights[model].length];
                        projections[model].projectBatch(inputs, outputs, count);
                        for (int i = 0; i < count; i++) {
                            final int m = model, c = count, r = i;
                            assertArrayEquals(oracle(weights[model], inputs[i]), outputs[i], 1e-7f,
                                    () -> "oracle, model=" + m + " count=" + c + " row=" + r + cpuContext());
                        }
                    }
                    return null;
                });
            }
            for (var task : pool.invokeAll(work)) task.get();
        }
    }

    @Test
    void packedMemoryAccountingStabilizesAfterAllBatchShapes() {
        var projection = prepare(matrix(64,128,5));
        long previous = projection.ramBytesUsed();
        for (int count = 1; count <= 8; count++) {
            projection.projectBatch(matrix(count,128,count), new float[count][64], count);
            assertTrue(projection.ramBytesUsed() >= previous);
            previous = projection.ramBytesUsed();
        }
        long prepared = projection.ramBytesUsed();
        for (int count = 8; count >= 1; count--)
            projection.projectBatch(matrix(count,128,count), new float[count][64], count);
        assertEquals(prepared, projection.ramBytesUsed());
    }

    @Test
    void precisionSelectionPreservesOriginalFp32AndAutomaticPolicy() {
        assertTrue(LibraryLoader.loadJvector());
        prepare(matrix(32,64,6)); // Require/qualify this backend before testing its selection.
        var original = new NativeASHProjection(matrix(32,64,6));
        String saved = System.getProperty(PROPERTY);
        try {
            float[] query = matrix(1,64,7)[0], before = new float[32], after = new float[32];
            original.project(query, before);
            System.setProperty(PROPERTY, "int8");
            var int8 = original.forQueryScoring();
            assertInstanceOf(NativeASHInt8Projection.class, int8);
            assertEquals("int8", int8.operandPrecision());
            original.project(query, after);
            assertArrayEquals(before, after);
            System.setProperty(PROPERTY, "fp32");
            assertSame(original, original.forQueryScoring());
            // auto prefers INT8 wherever this backend is available.
            System.setProperty(PROPERTY, "auto");
            assertInstanceOf(NativeASHInt8Projection.class, original.forQueryScoring());
        } finally {
            if (saved == null) System.clearProperty(PROPERTY); else System.setProperty(PROPERTY, saved);
        }
    }

    @Test
    void modelCachesQueryViewAndKeepsEncodingSeparate() {
        prepare(matrix(32,64,8));
        double[][] transform = new double[64][32];
        float[][] weights = matrix(32,64,9);
        for (int row = 0; row < 32; row++)
            for (int column = 0; column < 64; column++) transform[column][row] = weights[row][column];
        String saved = System.getProperty(PROPERTY);
        try {
            System.setProperty(PROPERTY,"int8");
            var model = new io.github.jbellis.jvector.quantization.AsymmetricHashing.StiefelTransform(
                    new org.apache.commons.math3.linear.Array2DRowRealMatrix(transform));
            float[] query = matrix(1,64,10)[0], encoding = new float[32], projected = new float[32];
            model.project(query,encoding);
            model.projectQueryDirect(query,projected);
            assertArrayEquals(oracle(model.AFloat,query),projected,1e-7f);
            System.setProperty(PROPERTY,"fp32");
            float[] cached = new float[32], encodedAgain = new float[32];
            model.projectQueryDirect(query,cached);
            model.project(query,encodedAgain);
            assertArrayEquals(projected,cached);
            assertArrayEquals(encoding,encodedAgain);
        } finally {
            if (saved == null) System.clearProperty(PROPERTY); else System.setProperty(PROPERTY,saved);
        }
    }

    @Test
    void abiVersionAndMissingSymbolsAreRejectedBeforeCalls() throws Exception {
        Class<?> type = Class.forName(NativeASHInt8Projection.class.getName() + "$Bindings");
        var constructor = type.getDeclaredConstructor(SymbolLookup.class);
        constructor.setAccessible(true);
        try (var arena = Arena.ofConfined()) {
            for (int version : new int[]{0,1}) {
                MemorySegment abi = Linker.nativeLinker().upcallStub(MethodHandles.constant(int.class, version),
                        FunctionDescriptor.of(JAVA_INT), arena);
                SymbolLookup lookup = name -> name.equals("jvector_ash_fp16_abi") ? Optional.of(abi) : Optional.empty();
                var failure = assertThrows(InvocationTargetException.class, () -> constructor.newInstance(lookup));
                assertInstanceOf(UnsatisfiedLinkError.class, failure.getCause());
            }
        }
    }

    @Test
    void nativeStatusesMapToTheirDocumentedJavaFailures() throws Exception {
        var failure = NativeASHInt8Projection.class.getDeclaredMethod("failure", int.class, MemorySegment.class);
        failure.setAccessible(true);
        try (var arena = Arena.ofConfined()) {
            var status = arena.allocate(260,4);
            status.setString(4,"fixture");
            assertInstanceOf(UnsupportedOperationException.class, failure.invoke(null,1,status));
            assertInstanceOf(IllegalArgumentException.class, failure.invoke(null,2,status));
            assertInstanceOf(IllegalArgumentException.class, failure.invoke(null,3,status));
            assertInstanceOf(IllegalStateException.class, failure.invoke(null,5,status));
            var error = assertThrows(InvocationTargetException.class, () -> failure.invoke(null,4,status));
            assertInstanceOf(OutOfMemoryError.class, error.getCause());
        }
    }

    @Test
    void legacyIsaCannotSilentlyRunSaturatingFallback(@TempDir Path temporary) throws Exception {
        String classpath = System.getProperty("surefire.test.class.path", System.getProperty("java.class.path"));
        Path log = temporary.resolve("legacy-isa.log");
        var builder = new ProcessBuilder(Path.of(System.getProperty("java.home"),"bin/java").toString(),
                "--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED",
                "-Djvector.test.requireInt8=true", "-cp",classpath, LegacyIsaProbe.class.getName());
        builder.environment().put("ONEDNN_MAX_CPU_ISA","AVX2");
        builder.environment().put("DNNL_MAX_CPU_ISA","AVX2");
        Process child = builder.redirectErrorStream(true).redirectOutput(log.toFile()).start();
        try {
            assertTrue(child.waitFor(30,TimeUnit.SECONDS));
            assertEquals(0,child.exitValue(),Files.readString(log));
        } finally {
            if (child.isAlive()) { child.destroyForcibly(); child.waitFor(5,TimeUnit.SECONDS); }
        }
    }

    public static final class LegacyIsaProbe {
        public static void main(String[] args) {
            assertTrue(LibraryLoader.loadJvector());
            try {
                prepare(new float[32][64]);
                throw new AssertionError("INT8 silently accepted legacy ISA");
            } catch (UnsupportedOperationException expected) { /* required mode fails clearly */ }
        }
    }
}
