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

package io.github.jbellis.jvector.example.repro;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/// Launches [MemorySafetyReproHarness] scenarios in a throwaway child JVM so that scenarios whose
/// expected outcome is a native JVM crash (SIGSEGV / SIGBUS) can be asserted on from a healthy
/// parent. The child writes its `hs_err` file into the scenario's work directory; the parent
/// collects exit code, combined stdout/stderr, and the parsed crash evidence (`siginfo` line and
/// problematic frame — the discriminators the bug doc's trigger table calls for).
public final class ChildJvm {

    /// Everything the parent needs to classify one child run.
    public static final class Result {
        public final int exitCode;
        public final String output;
        public final String hsErr;        // null when the child did not crash

        Result(int exitCode, String output, String hsErr) {
            this.exitCode = exitCode;
            this.output = output;
            this.hsErr = hsErr;
        }

        public boolean crashed() {
            return hsErr != null;
        }

        /// The `siginfo:` line from the hs_err file (signal + si_code), or "" if absent.
        public String siginfo() {
            return firstMatch("(?m)^siginfo:.*$");
        }

        /// The `# Problematic frame:` detail line from the hs_err file, or "" if absent.
        public String problematicFrame() {
            Matcher m = Pattern.compile("(?m)^# Problematic frame:\\R# (.*)$").matcher(hsErr == null ? "" : hsErr);
            return m.find() ? m.group(1).trim() : "";
        }

        /// The `Current thread` line from the hs_err file, or "" if absent.
        public String currentThread() {
            return firstMatch("(?m)^Current thread.*$");
        }

        private String firstMatch(String regex) {
            if (hsErr == null) {
                return "";
            }
            Matcher m = Pattern.compile(regex).matcher(hsErr);
            return m.find() ? m.group().trim() : "";
        }

        public boolean hsErrContains(String needle) {
            return hsErr != null && hsErr.contains(needle);
        }

        /// One-line summary for assertion messages and the verdict report.
        public String summary() {
            if (crashed()) {
                return "exit=" + exitCode + " CRASHED [" + siginfo() + "] frame=[" + problematicFrame() + "]";
            }
            String outcome = "";
            for (String line : output.split("\\R")) {
                if (line.startsWith("REPRO|OUTCOME|")) {
                    outcome = line;
                }
            }
            return "exit=" + exitCode + " " + (outcome.isEmpty() ? "(no outcome line)" : outcome);
        }
    }

    private ChildJvm() {
    }

    /// Runs `MemorySafetyReproHarness <harnessArgs...>` in a child JVM, with `hs_err` redirected
    /// into `workDir`. Blocks up to `timeoutSeconds`, then kills and fails.
    public static Result run(Path workDir, long timeoutSeconds, String... harnessArgs) throws IOException, InterruptedException {
        String javaBin = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        List<String> cmd = new ArrayList<>();
        cmd.add(javaBin);
        cmd.add("-cp");
        cmd.add(System.getProperty("java.class.path"));
        cmd.add("-ea");
        cmd.add("-Xmx1g");
        cmd.add("-XX:-CreateCoredumpOnCrash");
        cmd.add("-XX:ErrorFile=" + workDir.resolve("hs_err_pid%p.log").toAbsolutePath());
        if (Runtime.version().feature() >= 22) {
            cmd.add("--enable-native-access=ALL-UNNAMED");
        }
        cmd.add(MemorySafetyReproHarness.class.getName());
        cmd.addAll(Arrays.asList(harnessArgs));

        Process process = new ProcessBuilder(cmd).redirectErrorStream(true).start();
        StringBuilder out = new StringBuilder();
        Thread drain = new Thread(() -> {
            try (BufferedReader r = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = r.readLine()) != null) {
                    synchronized (out) {
                        out.append(line).append('\n');
                    }
                }
            } catch (IOException ignored) {
                // stream closes when the child dies; partial output is fine
            }
        }, "repro-child-drain");
        drain.start();

        if (!process.waitFor(timeoutSeconds, TimeUnit.SECONDS)) {
            process.destroyForcibly();
            drain.join(5_000);
            throw new IllegalStateException("child JVM did not finish within " + timeoutSeconds + "s; args=" + Arrays.toString(harnessArgs)
                    + "\n--- child output ---\n" + out);
        }
        drain.join(10_000);

        String hsErr = null;
        try (Stream<Path> files = Files.list(workDir)) {
            Path errFile = files.filter(p -> p.getFileName().toString().startsWith("hs_err_pid")).findFirst().orElse(null);
            if (errFile != null) {
                hsErr = Files.readString(errFile, StandardCharsets.ISO_8859_1);
            }
        }
        String output;
        synchronized (out) {
            output = out.toString();
        }
        return new Result(process.exitValue(), output, hsErr);
    }
}
