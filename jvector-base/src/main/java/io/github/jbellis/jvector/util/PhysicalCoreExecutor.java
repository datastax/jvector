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
package io.github.jbellis.jvector.util;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.quantization.ProductQuantization;

import java.io.Closeable;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A fork join pool which is sized to match the number of physical cores on the machine (avoiding hyper-thread count)
 * <p>
 * This is important for heavily vectorized sections of the code since it can easily saturate memory bandwidth.
 *
 * @see ProductQuantization
 * @see GraphIndexBuilder
 *
 * <p>The physical core count is resolved lazily on first use, in this order:
 * <ol>
 *   <li>the {@code jvector.physical_core_count} system property, if set (trusted as-is);</li>
 *   <li>a best-effort OS probe ({@code /proc/cpuinfo} on Linux, {@code sysctl} on macOS,
 *       {@code wmic} on Windows);</li>
 *   <li>a fallback of {@code availableProcessors() / 2}, which assumes 2-way hyper-threading.</li>
 * </ol>
 * Neither the pool nor the probe is created until {@link #pool()}, {@link #instance()} or
 * {@link #getPhysicalCoreCount()} is first called.
 *
 * <p>On first resolution the chosen count and the method that produced it are logged at
 * {@code INFO}. When no {@code jvector.physical_core_count} property is set and topology-aware
 * sizing yields a different count than the legacy {@code availableProcessors() / 2} heuristic, a
 * {@code WARNING} is also logged so the change in worker-thread count (and the throughput that
 * tracks it) is visible to an operator who expected the old value.
 */
public class PhysicalCoreExecutor implements Closeable {
    private static final Logger LOG = Logger.getLogger(PhysicalCoreExecutor.class.getName());

    /** How long to wait on a best-effort OS probe subprocess before giving up.
     * An early exit (failed probe) won't take this long, but we keep a longer window to avoid spurrious
     * race conditions on systems which are potentially busy during start-up. */
    private static final long PROBE_TIMEOUT_SECONDS = 15;

    /** Lazily resolved; {@code -1} means "not yet resolved". Guarded by the class monitor for writes. */
    private static volatile int physicalCoreCount = -1;

    /** Lazily constructed shared instance. Guarded by the class monitor for writes. */
    private static volatile PhysicalCoreExecutor instance;

    /**
     * Returns the resolved physical core count, computing it (including any OS probe) on first call
     * and caching the result. Does not create the pool.
     */
    public static int getPhysicalCoreCount() {
        int c = physicalCoreCount;
        if (c < 0) {
            synchronized (PhysicalCoreExecutor.class) {
                c = physicalCoreCount;
                if (c < 0) {
                    c = resolvePhysicalCoreCount();
                    physicalCoreCount = c;
                }
            }
        }
        return c;
    }

    /**
     * Returns the lazily-created shared instance, constructing the backing pool on first call.
     */
    public static PhysicalCoreExecutor instance() {
        PhysicalCoreExecutor i = instance;
        if (i == null) {
            synchronized (PhysicalCoreExecutor.class) {
                i = instance;
                if (i == null) {
                    i = new PhysicalCoreExecutor(getPhysicalCoreCount());
                    instance = i;
                }
            }
        }
        return i;
    }

    public static ForkJoinPool pool() {
        return instance().pool;
    }

    private final ForkJoinPool pool;

    private PhysicalCoreExecutor(int cores) {
        if (cores < 1) {
            throw new IllegalArgumentException("Physical core count must be >= 1, got: " + cores);
        }
        this.pool = new ForkJoinPool(cores);
    }

    public void execute(Runnable run) {
        pool.submit(run).join();
    }

    public <T> T submit(Supplier<T> run) {
        return pool.submit(run::get).join();
    }

    @Override
    public void close() {
        synchronized (PhysicalCoreExecutor.class) {
            pool.shutdownNow();
            // Release the singleton so a later pool()/instance() lazily creates a fresh pool.
            if (instance == this) {
                instance = null;
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Core-count resolution
    // ---------------------------------------------------------------------------------------------

    private static int resolvePhysicalCoreCount() {
        final int logical = Runtime.getRuntime().availableProcessors();
        // The pre-probe default (this class's historical sizing): assume 2-way hyper-threading.
        // Serves both as the last-resort fallback and as the baseline we warn against when
        // topology-aware sizing now selects a different number.
        final int legacyHeuristic = Math.max(1, logical / 2);

        Integer override = Integer.getInteger("jvector.physical_core_count");
        if (override != null) {
            if (override < 1) {
                throw new IllegalArgumentException(
                        "jvector.physical_core_count must be >= 1, got: " + override);
            }
            // Trust the operator: the property exists precisely to correct a wrong availableProcessors().
            LOG.log(Level.INFO,
                    "PhysicalCoreExecutor: using {0} worker threads, set explicitly via the "
                            + "jvector.physical_core_count system property (availableProcessors={1}).",
                    new Object[]{override, logical});
            return override;
        }

        int detected = detectPhysicalCores();
        final int selected;
        final String method;
        if (detected >= 1) {
            // Physical cannot exceed logical; clamping also respects cgroup CPU limits, since
            // availableProcessors() reflects the quota while /proc/cpuinfo shows all host cores.
            selected = Math.min(detected, logical);
            method = "OS topology probe on \"" + System.getProperty("os.name", "?")
                    + "\" (detected " + detected + " physical cores, clamped to availableProcessors=" + logical + ")";
        } else {
            // Unknown topology: assume 2-way hyper-threading (the historical default).
            selected = legacyHeuristic;
            method = "fallback heuristic availableProcessors()/2 (availableProcessors=" + logical
                    + "; OS topology probe unavailable)";
        }

        LOG.log(Level.INFO,
                "PhysicalCoreExecutor: using {0} worker threads, determined by {1}.",
                new Object[]{selected, method});

        // Auto-sizing now diverges from the pre-probe default; make that loud so an operator who
        // expected the old thread count (and the throughput that came with it) can pin it deliberately.
        if (selected != legacyHeuristic) {
            LOG.log(Level.WARNING,
                    "PhysicalCoreExecutor: previous auto-sized worker threads on this node would have been {0}, "
                            + "but aligning to cores now sets them to {1}. "
                            + "Set the jvector.physical_core_count system property to override.",
                    new Object[]{legacyHeuristic, selected});
        }

        return selected;
    }

    /** Best-effort physical-core probe. Returns {@code -1} when the topology can't be determined. */
    private static int detectPhysicalCores() {
        try {
            String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
            if (os.contains("linux")) {
                return detectLinuxPhysicalCores();
            }
            if (os.contains("mac") || os.contains("darwin")) {
                return parseFirstInt(runCommand("/usr/sbin/sysctl", "-n", "hw.physicalcpu"));
            }
            if (os.contains("windows")) {
                // wmic is deprecated but widely present; sum NumberOfCores across CPU packages.
                return sumInts(runCommand("wmic", "cpu", "get", "NumberOfCores"));
            }
        } catch (Throwable t) {
            // Best-effort only; fall through to the caller's heuristic default.
        }
        return -1;
    }

    /** Linux: count distinct (physical id, core id) pairs in {@code /proc/cpuinfo}. */
    private static int detectLinuxPhysicalCores() throws Exception {
        Path cpuinfo = Paths.get("/proc/cpuinfo");
        if (!Files.isReadable(cpuinfo)) {
            return -1;
        }
        Set<String> cores = new HashSet<>();
        String physicalId = null;
        String coreId = null;
        for (String line : Files.readAllLines(cpuinfo)) {
            line = line.trim();
            if (line.isEmpty()) {
                if (physicalId != null && coreId != null) {
                    cores.add(physicalId + ':' + coreId);
                }
                physicalId = null;
                coreId = null;
            } else if (line.startsWith("physical id")) {
                physicalId = valueAfterColon(line);
            } else if (line.startsWith("core id")) {
                coreId = valueAfterColon(line);
            }
        }
        // Handle a trailing block not terminated by a blank line.
        if (physicalId != null && coreId != null) {
            cores.add(physicalId + ':' + coreId);
        }
        // Topology fields are absent on many non-x86 CPUs (e.g. ARM); report unknown so the caller
        // falls back to its heuristic rather than trusting a bogus count.
        return cores.isEmpty() ? -1 : cores.size();
    }

    private static String valueAfterColon(String line) {
        int idx = line.indexOf(':');
        return idx >= 0 ? line.substring(idx + 1).trim() : "";
    }

    /** Runs a short-lived command, returning its stdout, or {@code null} on any failure/timeout. */
    private static String runCommand(String... cmd) {
        Process p = null;
        try {
            p = new ProcessBuilder(cmd).redirectErrorStream(false).start();
            // Close stdin so the child never blocks waiting for input.
            p.getOutputStream().close();
            if (!p.waitFor(PROBE_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                return null;
            }
            if (p.exitValue() != 0) {
                return null;
            }
            try (InputStream in = p.getInputStream()) {
                return new String(readAll(in), StandardCharsets.UTF_8);
            }
        } catch (Throwable t) {
            return null;
        } finally {
            if (p != null && p.isAlive()) {
                p.destroyForcibly();
            }
        }
    }

    private static byte[] readAll(InputStream in) throws Exception {
        // Probe outputs are tiny; a single bounded read is plenty.
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        byte[] buf = new byte[1024];
        int n;
        while ((n = in.read(buf)) != -1) {
            out.write(buf, 0, n);
            if (out.size() > 1 << 16) {
                break;
            }
        }
        return out.toByteArray();
    }

    private static int parseFirstInt(String s) {
        if (s == null) {
            return -1;
        }
        Matcher m = Pattern.compile("\\d+").matcher(s);
        return m.find() ? Integer.parseInt(m.group()) : -1;
    }

    private static int sumInts(String s) {
        if (s == null) {
            return -1;
        }
        Matcher m = Pattern.compile("\\d+").matcher(s);
        int sum = 0;
        boolean any = false;
        while (m.find()) {
            sum += Integer.parseInt(m.group());
            any = true;
        }
        return any ? sum : -1;
    }
}
