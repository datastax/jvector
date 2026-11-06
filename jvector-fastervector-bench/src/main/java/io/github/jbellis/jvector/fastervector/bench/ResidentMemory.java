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

package io.github.jbellis.jvector.fastervector.bench;

import java.io.IOException;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

/**
 * The process's resident memory on Linux, from {@code /proc/self/status}. fastervector allocates
 * outside the JVM, so the JVM's heap and off-heap figures miss it; the resident set size (RSS)
 * includes it. Elsewhere, snapshots are empty and the memory metrics are left out.
 */
final class ResidentMemory {
    private static final Path STATUS = Path.of("/proc/self/status");
    private static final Path CLEAR_REFS = Path.of("/proc/self/clear_refs");
    private static final MethodHandle MALLOC_TRIM = mallocTrim();

    private ResidentMemory() {}

    /** Current and peak resident set size, in bytes. */
    record Snapshot(long rssBytes, long peakRssBytes) {}

    static Optional<Snapshot> snapshot() {
        if (!Files.isReadable(STATUS)) {
            return Optional.empty();
        }
        try {
            return parse(Files.readAllLines(STATUS));
        } catch (IOException e) {
            return Optional.empty();
        }
    }

    /** Reads {@code VmRSS} and {@code VmHWM} (the peak) from {@code /proc/<pid>/status} lines. */
    static Optional<Snapshot> parse(List<String> statusLines) {
        long rss = -1;
        long peak = -1;
        for (String line : statusLines) {
            if (line.startsWith("VmRSS:")) {
                rss = kilobytes(line);
            } else if (line.startsWith("VmHWM:")) {
                peak = kilobytes(line);
            }
        }
        if (rss < 0 || peak < 0) {
            return Optional.empty();
        }
        return Optional.of(new Snapshot(rss * 1024, peak * 1024));
    }

    private static long kilobytes(String line) {
        String[] parts = line.substring(line.indexOf(':') + 1).trim().split("\\s+");
        return Long.parseLong(parts[0]);
    }

    /**
     * Resets the peak to the current RSS, so a later snapshot's peak covers only what follows.
     *
     * @return false if the kernel doesn't allow it; the peak then covers the whole process
     */
    static boolean resetPeak() {
        try {
            Files.writeString(CLEAR_REFS, "5");
            return true;
        } catch (IOException | SecurityException e) {
            return false;
        }
    }

    /**
     * Returns freed native memory to the OS with glibc's {@code malloc_trim(0)}, so memory a build
     * freed doesn't hide in a later baseline. Does nothing without glibc.
     */
    static void trim() {
        if (MALLOC_TRIM == null) {
            return;
        }
        try {
            int ignored = (int) MALLOC_TRIM.invokeExact(0L);
        } catch (Throwable e) {
            // Best effort: the measurement is only less precise
        }
    }

    private static MethodHandle mallocTrim() {
        try {
            Linker linker = Linker.nativeLinker();
            return linker.defaultLookup().find("malloc_trim")
                    .map(symbol -> linker.downcallHandle(symbol,
                            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG)))
                    .orElse(null);
        } catch (RuntimeException e) {
            return null;
        }
    }

    static double mb(long bytes) {
        return bytes / 1024.0 / 1024.0;
    }
}
