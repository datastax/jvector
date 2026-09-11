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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.vector.VectorizationProvider;
import org.junit.Test;
import org.junit.Assume;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import static org.junit.Assert.*;

/** Opt-in forked-JVM matrix, because tuning constants are initialized once per JVM. */
public class TestASHKernelTuning {
    @Test
    public void allSettingsAndWidthsMatchScalarOracle() throws Exception {
        Assume.assumeTrue(Boolean.getBoolean("jvector.test.ashTuningMatrix"));
        Assume.assumeTrue(VectorizationProvider.getInstance().getVectorUtilSupport().supportsAshLutScoring());
        String architecture = System.getProperty("os.arch");
        int[] widths = architecture.equals("amd64") || architecture.equals("x86_64")
                ? new int[]{16, 32, 64} : new int[]{0};
        for (int width : widths) {
            for (int accumulators : new int[]{1, 2, 4, 8}) {
                for (int unroll : new int[]{1, 2, 4}) {
                    var command = javaCommand();
                    if (width != 0) command.add("-XX:MaxVectorSize=" + width);
                    command.add("-Djvector.test.requireAshSimd=true");
                    for (String kernel : new String[]{"lut", "projection"}) {
                        command.add("-Djvector.ash." + kernel + ".accumulators=" + accumulators);
                        command.add("-Djvector.ash." + kernel + ".unroll=" + unroll);
                    }
                    command.add("org.junit.runner.JUnitCore");
                    command.add(TestASHLutScoring.class.getName());
                    command.add(TestASHScoringDispatch.class.getName());
                    run(command, true, "width=" + width + ", accumulators=" + accumulators + ", unroll=" + unroll);
                }
            }
        }
    }

    @Test
    public void invalidTuningFailsExplicitly() throws Exception {
        Assume.assumeTrue(Boolean.getBoolean("jvector.test.ashTuningMatrix"));
        Assume.assumeTrue(VectorizationProvider.getInstance().getVectorUtilSupport().supportsAshLutScoring());
        for (String name : new String[]{"lut.accumulators", "lut.unroll", "projection.accumulators", "projection.unroll"}) {
            for (String invalid : new String[]{"0", "3", "16", "invalid"}) {
                var command = javaCommand();
                command.add("-Djvector.ash." + name + "=" + invalid);
                command.add(Probe.class.getName());
                run(command, false, "jvector.ash." + name);
            }
        }
    }

    private static List<String> javaCommand() {
        return new ArrayList<>(List.of(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "-ea", "-Xmx256m", "--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED",
                "-cp", System.getProperty("java.class.path")));
    }

    private static void run(List<String> command, boolean success, String label) throws Exception {
        // Child output is small (JUnit summary and backend diagnostics); a temporary file
        // prevents a full pipe from blocking the child while the parent waits with a timeout.
        var output = java.nio.file.Files.createTempFile("ash-tuning-test-", ".log");
        try {
            Process process = new ProcessBuilder(command).redirectErrorStream(true).redirectOutput(output.toFile()).start();
            if (!process.waitFor(60, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                fail("Timed out: " + label);
            }
            String text = java.nio.file.Files.readString(output, StandardCharsets.UTF_8);
            if (success) assertEquals(label + "\n" + text, 0, process.exitValue());
            else {
                assertNotEquals(label, 0, process.exitValue());
                assertTrue(text, text.contains(label));
            }
        } finally { java.nio.file.Files.deleteIfExists(output); }
    }

    public static class Probe {
        public static void main(String[] ignored) {
            System.out.println(VectorizationProvider.getInstance().getVectorUtilSupport().ashKernelDescription());
        }
    }
}
