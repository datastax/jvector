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

package io.github.jbellis.jvector.example.reporting;

import io.github.jbellis.jvector.vector.VectorizationProvider;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * Writes sys_info.json for a run.
 *
 * Schema is intentionally simple and stable. CPU model/flags are best-effort on Linux/macOS.
 * Vector API species is best-effort (may be unavailable on Java 11).
 */
public final class SysInfoWriter {

    private SysInfoWriter() {}

    public static String writeSysInfo(Path runDir,
                                      int schemaVersion,
                                      String runId,
                                      UUID runUuid,
                                      Instant createdAt,
                                      String jvectorRef,
                                      Integer buildThreads,
                                      Integer queryThreads) throws IOException {

        Map<String, Object> root = new LinkedHashMap<>();
        root.put("schema_version", schemaVersion);
        root.put("run_id", runId);
        root.put("run_uuid", runUuid.toString());
        root.put("created_at", DateTimeFormatter.ISO_INSTANT.format(createdAt));
        root.put("jvector_ref", jvectorRef == null ? "" : jvectorRef);

        // Host / OS / JVM
        root.put("host", hostInfo());
        root.put("os", osInfo());
        root.put("jvm", jvmInfo());

        // Threads / parallelism knobs (caller supplies if known)
        Map<String, Object> threads = new LinkedHashMap<>();

        int fjp = ForkJoinPoolCommonParallelism.getInt();

        // Build executor (PhysicalCoreExecutor.pool) parallelism
        threads.put("build_executor_parallelism", buildThreads);

        // Parallel streams use the common ForkJoinPool by default
        threads.put("parallel_stream_parallelism", fjp);

        // Query benchmark characteristics
        threads.put("query_throughput_parallelism", fjp);
        threads.put("query_latency_threads", 1);

        // Raw common pool parallelism
        threads.put("fjp_common_parallelism", fjp);

        root.put("threads", threads);


        // Memory
        Map<String, Object> mem = new LinkedHashMap<>();
        mem.put("max_heap_bytes", Runtime.getRuntime().maxMemory());
        mem.putAll(physicalMemoryInfoBestEffort());
        root.put("memory", mem);

        // SIMD/species (best-effort, always available if Vector API present)
        root.put("simd", simdInfo());

        // CPU info (Linux best-effort)
        root.put("cpu", cpuInfoBestEffort());

        // system_id hash over a stable subset
        String systemId = Hashing.shortSha256Hex(canonicalSystemIdString(root));
        root.put("system_id", systemId);

        Files.createDirectories(runDir);
        Path out = runDir.resolve("sys_info.json");
        Files.writeString(out, JsonUtil.toJson(root), StandardCharsets.UTF_8);

        return systemId;
    }

    // ---------------- helpers ----------------

    private static Map<String, Object> hostInfo() {
        Map<String, Object> m = new LinkedHashMap<>();
        String name = System.getenv().getOrDefault("HOSTNAME", "");
        if (name.isEmpty()) {
            try {
                name = java.net.InetAddress.getLocalHost().getHostName();
            } catch (Throwable t) {
                // ignore
            }
        }
        m.put("name", name);
        m.put("user", System.getProperty("user.name", ""));
        return m;
    }

    private static Map<String, Object> osInfo() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("name", System.getProperty("os.name", ""));
        m.put("version", System.getProperty("os.version", ""));
        m.put("arch", System.getProperty("os.arch", ""));
        return m;
    }

    private static Map<String, Object> jvmInfo() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("vendor", System.getProperty("java.vendor", ""));
        m.put("version", System.getProperty("java.version", ""));
        m.put("vm_name", System.getProperty("java.vm.name", ""));
        m.put("runtime", System.getProperty("java.runtime.name", ""));
        return m;
    }

    private static Map<String, Object> simdInfo() {
        Map<String, Object> m = new LinkedHashMap<>();

        m.put("vectorization_provider", VectorizationProvider.getInstance().getClass().getSimpleName());

        String bitSizeProp = System.getProperty("io.github.jbellis.jvector.vector_bit_size", "");
        boolean present = !bitSizeProp.isBlank();

        m.put("simd_config_present", present);
        m.put("configured_vector_bit_size", bitSizeProp);

        Integer lanes = null;
        if (present) {
            try {
                int bits = Integer.parseInt(bitSizeProp.trim());
                lanes = bits / 32;
            } catch (NumberFormatException ignored) {
                // leave lanes null
            }
        }
        m.put("configured_float_lanes", lanes);

        return m;
    }

    private static Map<String, Object> physicalMemoryInfoBestEffort() {
        Map<String, Object> m = new LinkedHashMap<>();
        try {
            var bean = ManagementFactory.getOperatingSystemMXBean();
            if (bean instanceof com.sun.management.OperatingSystemMXBean) {
                var os = (com.sun.management.OperatingSystemMXBean) bean;
                m.put("total_physical_bytes", os.getTotalPhysicalMemorySize());
            }
        } catch (Throwable t) {
            // ignore
        }
        return m;
    }

    private static Map<String, Object> cpuInfoBestEffort() {
        Map<String, Object> m = new LinkedHashMap<>();
        String osName = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);

        // macOS best-effort via sysctl
        if (osName.contains("mac")) {
            m.put("available_processors", Runtime.getRuntime().availableProcessors());

            String model = firstNonEmpty(
                    runCmd("sysctl", "-n", "machdep.cpu.brand_string"),
                    runCmd("sysctl", "-n", "machdep.cpu.model")
            );
            if (model.isEmpty()) model = "unknown";

            String f1 = runCmd("sysctl", "-n", "machdep.cpu.features");
            String f2 = runCmd("sysctl", "-n", "machdep.cpu.leaf7_features");

            Set<String> flags = new TreeSet<>();
            for (String s : List.of(f1, f2)) {
                if (s == null || s.isEmpty() || isSysctlError(s)) {
                    continue;
                }
                flags.addAll(Arrays.asList(s.trim().split("\\s+")));
            }


            m.put("model", model);
            m.put("flags", new ArrayList<>(flags));
            return m;
        }

        // Non-Linux fallback
        if (!osName.contains("linux")) {
            m.put("available_processors", Runtime.getRuntime().availableProcessors());
            m.put("model", "unknown");
            m.put("flags", List.of());
            return m;
        }

        // Linux best-effort via /proc/cpuinfo
        m.put("available_processors", Runtime.getRuntime().availableProcessors());

        String model = "unknown";
        Set<String> flags = new TreeSet<>();
        try {
            List<String> lines = Files.readAllLines(Path.of("/proc/cpuinfo"), StandardCharsets.UTF_8);
            for (String line : lines) {
                if (model.equals("unknown") && line.startsWith("model name")) {
                    int idx = line.indexOf(':');
                    if (idx >= 0) model = line.substring(idx + 1).trim();
                }
                if (line.startsWith("flags")) {
                    int idx = line.indexOf(':');
                    if (idx >= 0) {
                        String[] parts = line.substring(idx + 1).trim().split("\\s+");
                        flags.addAll(Arrays.asList(parts));
                    }
                }
                if (!model.equals("unknown") && !flags.isEmpty()) break;
            }
        } catch (Throwable t) {
            // ignore
        }

        m.put("model", model);
        m.put("flags", new ArrayList<>(flags));
        return m;
    }

    /**
     * Canonical string for hashing. Keep it stable and avoid volatile values.
     */
    private static String canonicalSystemIdString(Map<String, Object> root) {
        // Keep a stable subset only (no timestamps, no run ids)
        Map<String, Object> subset = new LinkedHashMap<>();
        subset.put("os", root.get("os"));
        subset.put("jvm", root.get("jvm"));
        subset.put("simd", root.get("simd"));
        subset.put("threads", root.get("threads"));
        subset.put("cpu", root.get("cpu"));
        subset.put("memory", root.get("memory"));
        subset.put("jvector_ref", root.get("jvector_ref"));
        return JsonUtil.toJson(subset);
    }

    /**
     * Small helper to read common pool parallelism without hard dependency.
     */
    private static final class ForkJoinPoolCommonParallelism {
        static int getInt() {
            try {
                return java.util.concurrent.ForkJoinPool.getCommonPoolParallelism();
            } catch (Throwable t) {
                return -1;
            }
        }
    }

    private static String firstNonEmpty(String... vals) {
        for (String v : vals) {
            if (v != null && !v.isBlank()) return v.trim();
        }
        return "";
    }

    private static boolean isSysctlError(String s) {
        if (s == null) return true;
        String t = s.toLowerCase(Locale.ROOT);
        return t.contains("unknown oid") || t.startsWith("sysctl:");
    }

    private static String runCmd(String... cmd) {
        try {
            Process p = new ProcessBuilder(cmd)
                    .redirectErrorStream(true)
                    .start();
            byte[] out = p.getInputStream().readAllBytes();
            p.waitFor();
            return new String(out, StandardCharsets.UTF_8).trim();
        } catch (Throwable t) {
            return "";
        }
    }
}
