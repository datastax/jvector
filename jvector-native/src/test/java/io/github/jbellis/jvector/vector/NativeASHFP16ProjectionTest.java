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

import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.vector.cnative.LibraryLoader;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.parallel.ResourceLock;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** FP16 arithmetic, query-only selection, concurrency and optional-backend compatibility. */
@ResourceLock("java.lang.System.properties")
class NativeASHFP16ProjectionTest {
    private static final String PROPERTY = "jvector.ash.queryProjection";

    @BeforeAll
    static void loadNative() {
        boolean available = LibraryLoader.loadJvector();
        if (Boolean.getBoolean("jvector.test.requireFp16"))
            assertTrue(available, "Highway library unavailable");
        else assumeTrue(available, "Highway library unavailable");
    }

    private static VectorUtilSupport.AshProjection view(NativeASHProjection original, String mode) {
        String previous = System.getProperty(PROPERTY);
        try {
            if (mode == null) System.clearProperty(PROPERTY);
            else System.setProperty(PROPERTY, mode);
            return original.forQueryScoring();
        } finally {
            if (previous == null) System.clearProperty(PROPERTY);
            else System.setProperty(PROPERTY, previous);
        }
    }

    private static VectorUtilSupport.AshProjection required(NativeASHProjection original) {
        try { return view(original, "fp16"); }
        catch (UnsupportedOperationException unsupported) {
            if (Boolean.getBoolean("jvector.test.requireFp16")) throw unsupported;
            assumeTrue(false, unsupported.getMessage());
            throw unsupported;
        }
    }

    private static float half(float value) {
        return Float.float16ToFloat(Float.floatToFloat16(value));
    }

    private static void checkOracle(float[][] matrix, float[] query, float[] actual, boolean fp16) {
        for (int row = 0; row < matrix.length; row++) {
            double expected = 0, magnitude = 0;
            for (int col = 0; col < query.length; col++) {
                double product = (double) (fp16 ? half(matrix[row][col]) : matrix[row][col])
                        * (fp16 ? half(query[col]) : query[col]);
                expected += product;
                magnitude += Math.abs(product);
            }
            assertEquals(expected, actual[row], 2e-6 * magnitude + 1e-7, "row " + row);
        }
    }

    @Test
    void matchesIndependentRoundedOperandOracleAcrossDimensions() {
        Random random = new Random(93124);
        for (int[] shape : new int[][]{{1,1},{3,7},{17,31},{65,129},{257,513},{384,768},{1536,3072}}) {
            float[][] matrix = new float[shape[0]][shape[1]];
            for (float[] row : matrix) for (int i = 0; i < row.length; i++) row[i] = random.nextFloat() - .5f;
            var fp32 = new NativeASHProjection(matrix);
            var fp16 = required(fp32);
            assertInstanceOf(NativeASHFP16Projection.class, fp16);
            assertTrue(fp16.ramBytesUsed() >= (long) shape[0] * shape[1] * 2);
            // Repeated calls exercise workspace reuse with different queries.
            for (int call = 0; call < 17; call++) {
                float[] query = new float[shape[1]], out = new float[shape[0]];
                for (int i = 0; i < query.length; i++) query[i] = random.nextFloat() - .5f;
                float[] saved = query.clone();
                fp16.project(query, out);
                checkOracle(matrix, query, out, true);
                assertArrayEquals(saved, query);
            }
            assertSame(fp32, view(fp32, "fp32"));
        }
    }

    @Test
    void halfPrecisionBoundaryRoundingMatchesTheIndependentConversion() {
        var projection = required(new NativeASHProjection(new float[][]{{1}}));
        float tie = 1f + Math.scalb(1f, -11);
        for (float value : new float[]{0f, -0f, Float.MIN_VALUE, Math.scalb(1f, -24),
                Math.scalb(1f, -14), tie, Math.nextUp(tie), -tie, 65504f, -65504f}) {
            float[] out = new float[1];
            projection.project(new float[]{value}, out);
            assertEquals(half(value), out[0], 0f);
        }
    }

    @Test
    void sharedModelsSurviveConcurrentCallsModelSwitchingAndGc() throws Exception {
        float[][] a = {{.1234567f, -.3333333f, .9f}, {-.5555555f, .0123456f, .25f}};
        float[][] b = {{.456789f, .234567f, -.75f}};
        var first = required(new NativeASHProjection(a));
        var second = required(new NativeASHProjection(b));
        try (var pool = Executors.newFixedThreadPool(8)) {
            var tasks = new ArrayList<java.util.concurrent.Callable<Void>>();
            for (int thread = 0; thread < 8; thread++) tasks.add(() -> {
                float[] query = {.234567f, -.456789f, .75f};
                float[] outA = new float[2], outB = new float[1];
                for (int call = 0; call < 40; call++) {
                    first.project(query, outA);
                    second.project(query, outB);
                    checkOracle(a, query, outA, true);
                    checkOracle(b, query, outB, true);
                    if (call == 20) System.gc();
                }
                return null;
            });
            for (var result : pool.invokeAll(tasks)) result.get();
        }
    }

    @Test
    void virtualThreadCallsKeepTheirResultsAndDiagnostics() throws Exception {
        float[][] matrix = {{.1234567f, -.3456789f}};
        var projection = required(new NativeASHProjection(matrix));
        try (var pool = Executors.newVirtualThreadPerTaskExecutor()) {
            var tasks = new ArrayList<java.util.concurrent.Callable<Void>>();
            for (int thread = 0; thread < 16; thread++) tasks.add(() -> {
                float[] out = new float[1];
                for (int call = 0; call < 16; call++) {
                    var error = assertThrows(IllegalArgumentException.class,
                            () -> projection.project(new float[]{Float.NaN, 1}, out));
                    assertTrue(error.getMessage().contains("finite FP16 range"));
                    float[] query = {.234567f, -.456789f};
                    projection.project(query, out);
                    checkOracle(matrix, query, out, true);
                    Thread.yield();
                }
                return null;
            });
            for (var result : pool.invokeAll(tasks)) result.get();
        }
    }

    @Test
    void guardsShapesAliasesAndRangeWithoutDamagingSubsequentCalls() {
        float[][] matrix = {{.1234567f, .3456789f}, {-.3333333f, .5f}};
        var fp16 = required(new NativeASHProjection(matrix));
        float[] query = {1, 2}, out = new float[2];
        assertThrows(NullPointerException.class, () -> fp16.project(null, out));
        assertThrows(NullPointerException.class, () -> fp16.project(query, null));
        assertThrows(IllegalArgumentException.class, () -> fp16.project(query, query));
        assertThrows(IllegalArgumentException.class, () -> fp16.project(new float[1], out));
        assertThrows(IllegalArgumentException.class, () -> fp16.project(query, new float[1]));
        for (float invalid : new float[]{Float.NaN, Float.POSITIVE_INFINITY, 65505f}) {
            Arrays.fill(out, 42f);
            assertThrows(IllegalArgumentException.class, () -> fp16.project(new float[]{invalid, 1}, out));
            assertArrayEquals(new float[]{42, 42}, out);
        }
        fp16.project(query, out);
        checkOracle(matrix, query, out, true);
    }

    @Test
    void defaultPrefersFp16ButFp32AndAutomaticRangeFallbackRemainAvailable() {
        var original = new NativeASHProjection(new float[][]{{.25f, -.5f}});
        assertSame(original, view(original, "fp32"));
        var explicit = required(original);
        assertInstanceOf(NativeASHFP16Projection.class, explicit);
        var automatic = view(original, null);
        assertInstanceOf(NativeASHFP16Projection.class, automatic);
        float[] query = {70000, 2}, out = new float[1], expected = new float[1];
        original.project(query, expected);
        automatic.project(query, out);
        assertArrayEquals(expected, out);
        var largeMatrix = new NativeASHProjection(new float[][]{{70000}});
        assertSame(largeMatrix, view(largeMatrix, "auto"));
        assertThrows(IllegalArgumentException.class, () -> view(largeMatrix, "fp16"));
        assertThrows(IllegalArgumentException.class, () -> view(original, "typo"));
    }

    @Test
    void encodingStaysFp32BeforeAndAfterTheQueryViewIsPrepared() {
        String previous = System.getProperty(PROPERTY);
        try {
            System.setProperty(PROPERTY, "fp16");
            double[][] w = {{.1234567, .2345678}, {-.3456789, .4567891}, {.5678912, -.6789123}};
            var transform = new AsymmetricHashing.StiefelTransform(new Array2DRowRealMatrix(w));
            float[] query = {.1234567f, -.3456789f, .9876543f};
            float[] encodingBefore = new float[2], encodingAfter = new float[2], queryOut = new float[2];
            transform.project(query, encodingBefore);
            checkOracle(transform.AFloat, query, encodingBefore, false);
            try { transform.projectQuery(query, queryOut); }
            catch (UnsupportedOperationException unsupported) {
                if (Boolean.getBoolean("jvector.test.requireFp16")) throw unsupported;
                assumeTrue(false, unsupported.getMessage());
            }
            checkOracle(transform.AFloat, query, queryOut, true);
            assertTrue(Math.abs(encodingBefore[0] - queryOut[0]) > 1e-6f);
            transform.project(query, encodingAfter);
            assertArrayEquals(encodingBefore, encodingAfter);
        } finally {
            if (previous == null) System.clearProperty(PROPERTY);
            else System.setProperty(PROPERTY, previous);
        }
    }

    @Test
    void missingOptionalLibraryFallsBackAndExplicitFp16Fails(@TempDir Path temporary) throws Exception {
        Path source = Path.of(NativeASHProjection.class.getProtectionDomain().getCodeSource().getLocation().toURI());
        assertTrue(Files.isDirectory(source));
        Path stripped = temporary.resolve("native-without-fp16.jar");
        try (var jar = new JarOutputStream(Files.newOutputStream(stripped)); var paths = Files.walk(source)) {
            for (Path p : paths.filter(Files::isRegularFile).toList()) {
                String name = source.relativize(p).toString().replace(File.separatorChar, '/');
                if (name.startsWith("libjvector_ash_fp16")) continue;
                jar.putNextEntry(new JarEntry(name));
                Files.copy(p, jar);
                jar.closeEntry();
            }
        }
        String raw = System.getProperty("surefire.test.class.path", System.getProperty("java.class.path"));
        String cp = stripped + File.pathSeparator + String.join(File.pathSeparator,
                Arrays.stream(raw.split(java.util.regex.Pattern.quote(File.pathSeparator)))
                        .filter(p -> !Path.of(p).toAbsolutePath().normalize().equals(source.toAbsolutePath().normalize()))
                        .toList());
        Path log = temporary.resolve("unavailable.log");
        Process child = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin/java").toString(),
                "--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED",
                "-Djava.library.path=" + temporary, "-cp", cp, UnavailableProbe.class.getName())
                .redirectErrorStream(true).redirectOutput(log.toFile()).start();
        try {
            assertTrue(child.waitFor(30, TimeUnit.SECONDS));
            assertEquals(0, child.exitValue(), Files.readString(log));
            assertTrue(Files.readString(log).contains("FP16_UNAVAILABLE_OK"));
        } finally {
            if (child.isAlive()) {
                child.destroyForcibly();
                child.waitFor(5, TimeUnit.SECONDS);
            }
        }
    }

    public static final class UnavailableProbe {
        public static void main(String[] args) {
            if (!LibraryLoader.loadJvector()) throw new AssertionError("Missing Highway fixture");
            var original = new NativeASHProjection(new float[][]{{1, 2}});
            if (view(original, "auto") != original) throw new AssertionError("Automatic fallback failed");
            try {
                view(original, "fp16");
                throw new AssertionError("Explicit FP16 silently fell back");
            } catch (UnsupportedOperationException expected) { /* expected */ }
            var transform = new AsymmetricHashing.StiefelTransform(new Array2DRowRealMatrix(new double[][]{{1}, {2}}));
            System.setProperty(PROPERTY, "fp16");
            float[] out = new float[1];
            transform.project(new float[]{1, 1}, out); // Encoding does not require FP16.
            if (out[0] != 3) throw new AssertionError("Encoding changed");
            System.out.println("FP16_UNAVAILABLE_OK");
        }
    }
}
