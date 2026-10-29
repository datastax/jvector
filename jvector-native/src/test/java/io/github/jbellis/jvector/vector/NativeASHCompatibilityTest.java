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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Optional ASH symbol detection must not make unrelated provider construction fail. */
class NativeASHCompatibilityTest {
    @Test
    void packagedLibraryProvidesNeighborhoodScoring() {
        String arch = System.getProperty("os.arch", "");
        assumeTrue(arch.equals("amd64") || arch.equals("x86_64"));
        assumeTrue(LibraryLoader.loadJvector(), "Native jvector library not available");
        assertTrue(new NativeVectorUtilSupport().supportsAshNeighborhoodScoring());
        assertTrue(NativeASHProjection.isAvailable());
        assertTrue(NativeASHSymmetricKernels.hasPairKernel());
        assertTrue(NativeASHSymmetricKernels.hasLutKernel());
    }

    @Test
    void missingAshSymbolRetainsJavaLutScoring() throws Exception {
        // A fresh JVM deliberately loads no native library. This tests an actual absent
        // symbol without changing the parent JVM's already loaded library or using mocks.
        Path output = Files.createTempFile("ash-missing-symbol-", ".log");
        Process child = null;
        try {
            child = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                    "--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED", "-cp",
                    System.getProperty("java.class.path"), MissingSymbolProbe.class.getName())
                            .redirectErrorStream(true)
                            .redirectOutput(output.toFile())
                            .start();
            assertTrue(child.waitFor(30, TimeUnit.SECONDS), "Missing-symbol probe timed out");
            String diagnostics = Files.readString(output);
            assertEquals(0, child.exitValue(), diagnostics);
            assertTrue(diagnostics.contains("MISSING_ASH_FALLBACK_OK"), diagnostics);
        } finally {
            if (child != null && child.isAlive()) {
                child.destroyForcibly().waitFor();
            }
            Files.deleteIfExists(output);
        }
    }

    /** Child-process entry: do not initialize LibraryLoader or the global provider. */
    public static final class MissingSymbolProbe {
        public static void main(String[] args) {
            var backend = new NativeVectorUtilSupport();
            if (backend.supportsAshNeighborhoodScoring()) {
                throw new AssertionError("ASH symbol unexpectedly available without a loaded native library");
            }
            if (NativeASHProjection.isAvailable() || NativeASHSymmetricKernels.hasPairKernel()
                    || NativeASHSymmetricKernels.hasLutKernel()) {
                throw new AssertionError("New ASH symbols unexpectedly available");
            }
            float[] projected = new float[2];
            backend.prepareAshProjection(new float[][]{{1,2,3},{4,5,6}})
                    .project(new float[]{2,3,4}, projected);
            if (projected[0] != 20f || projected[1] != 47f)
                throw new AssertionError("Java projection fallback failed");
            if (backend.ashSymmetricDot(null,new byte[]{0},null,new byte[]{0},2,4) != .5f)
                throw new AssertionError("Java symmetric pair fallback failed");
            float[] symmetric = new float[1];
            backend.ashSymmetricLutScore(new byte[1],1,1,0,1,new short[32],symmetric,0);
            if (symmetric[0] != 0f) throw new AssertionError("Java symmetric LUT fallback failed");
            float[] lut = new float[16];
            java.util.Arrays.fill(lut, .25f);
            float[] out = new float[8];
            backend.ashLutScore(new byte[8], 0, 1, 8, 0, 8, lut, out, 0);
            for (float value : out) {
                if (value != .25f) {
                    throw new AssertionError("Existing Java LUT scorer failed");
                }
            }
            try {
                backend.ashScoreNeighborhood(new byte[48], 1, 8, 8, lut, 0f, out);
                throw new AssertionError("Unavailable native kernel accepted a call");
            } catch (UnsupportedOperationException expected) {
                // Capability is optional, and direct calls report its absence clearly.
            }
            System.out.println("MISSING_ASH_FALLBACK_OK");
        }
    }
}
