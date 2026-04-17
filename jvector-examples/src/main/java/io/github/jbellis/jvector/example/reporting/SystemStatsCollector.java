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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

/**
 * Background collector of {@code /proc} system metrics (CPU topology, load, memory, disk I/O).
 * Reads /proc files directly in Java and appends JSONL lines to a file every 30 seconds.
 */
public final class SystemStatsCollector {
    private static final Logger log = LoggerFactory.getLogger(SystemStatsCollector.class);
    private static final Path PROC_CPUINFO = Path.of("/proc/cpuinfo");
    private static final Path PROC_LOADAVG = Path.of("/proc/loadavg");
    private static final Path PROC_MEMINFO = Path.of("/proc/meminfo");
    private static final Path PROC_DISKSTATS = Path.of("/proc/diskstats");
    private static final Pattern DISK_DEVICE_PATTERN = Pattern.compile("sd[a-z]+|nvme[0-9]+n[0-9]+|vd[a-z]+|xvd[a-z]+");
    private static final DateTimeFormatter TS_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss'Z'").withZone(ZoneOffset.UTC);

    private ScheduledExecutorService scheduler;
    private BufferedWriter writer;
    private String fileName;
    private int cpuSockets;
    private int cpuCores;
    private int cpuThreads;

    public Path start(Path outputDir, String fileName) throws IOException {
        if (!Files.exists(PROC_CPUINFO)) {
            log.warn("/proc filesystem not available (not Linux?), system stats collection disabled");
            return null;
        }

        Files.createDirectories(outputDir);
        Path sysStatsPath = outputDir.resolve(fileName).toAbsolutePath();
        this.fileName = fileName;

        parseCpuTopology();

        this.writer = Files.newBufferedWriter(sysStatsPath,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);

        scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "sys-stats-collector");
            t.setDaemon(true);
            return t;
        });
        scheduler.scheduleAtFixedRate(() -> {
            try {
                String line = collectSnapshot();
                writer.write(line);
                writer.newLine();
                writer.flush();
            } catch (Exception e) {
                log.warn("Failed to collect system stats", e);
            }
        }, 0, 30, TimeUnit.SECONDS);

        log.info("System stats collection started, saving to: {}", sysStatsPath);
        return sysStatsPath;
    }

    public void stop(Path outputDir) throws InterruptedException {
        if (scheduler != null) {
            scheduler.shutdown();
            scheduler.awaitTermination(5, TimeUnit.SECONDS);
            scheduler = null;
            try {
                if (writer != null) {
                    writer.close();
                    writer = null;
                }
            } catch (IOException e) {
                log.warn("Failed to close stats writer", e);
            }
            log.info("System stats collection stopped, saved to: {}", outputDir.resolve(fileName).toAbsolutePath());
        }
    }

    public boolean isActive() {
        return scheduler != null && !scheduler.isShutdown();
    }

    public String getFileName() {
        return fileName;
    }

    private void parseCpuTopology() throws IOException {
        List<String> lines = Files.readAllLines(PROC_CPUINFO);
        int threads = 0;
        var physicalIds = new HashSet<String>();
        var coreKeys = new HashSet<String>();
        String currentPhysicalId = "0";

        for (String line : lines) {
            if (line.startsWith("processor")) {
                threads++;
            } else if (line.startsWith("physical id")) {
                currentPhysicalId = line.substring(line.indexOf(':') + 1).trim();
                physicalIds.add(currentPhysicalId);
            } else if (line.startsWith("core id")) {
                String coreId = line.substring(line.indexOf(':') + 1).trim();
                coreKeys.add(currentPhysicalId + "-" + coreId);
            }
        }

        this.cpuThreads = threads;
        this.cpuSockets = physicalIds.isEmpty() ? 1 : physicalIds.size();
        this.cpuCores = coreKeys.isEmpty() ? cpuThreads : coreKeys.size();
    }

    private String collectSnapshot() throws IOException {
        String ts = TS_FORMAT.format(Instant.now());

        // /proc/loadavg: "0.50 0.35 0.25 2/150 12345"
        String loadLine = Files.readString(PROC_LOADAVG).trim();
        String[] loadParts = loadLine.split("\\s+");
        String load1 = loadParts[0];
        String load5 = loadParts[1];
        String load15 = loadParts[2];
        String[] runProcs = loadParts[3].split("/");
        String running = runProcs[0];
        String total = runProcs[1];

        // /proc/meminfo
        long memTotal = 0, memFree = 0, memAvail = 0, buffers = 0, cached = 0, swapTotal = 0, swapFree = 0;
        for (String line : Files.readAllLines(PROC_MEMINFO)) {
            if (line.startsWith("MemTotal:"))       memTotal = parseMemValue(line);
            else if (line.startsWith("MemFree:"))    memFree = parseMemValue(line);
            else if (line.startsWith("MemAvailable:")) memAvail = parseMemValue(line);
            else if (line.startsWith("Buffers:"))    buffers = parseMemValue(line);
            else if (line.startsWith("Cached:"))     cached = parseMemValue(line);
            else if (line.startsWith("SwapTotal:"))  swapTotal = parseMemValue(line);
            else if (line.startsWith("SwapFree:"))   swapFree = parseMemValue(line);
        }

        // /proc/diskstats
        StringBuilder disks = new StringBuilder();
        for (String line : Files.readAllLines(PROC_DISKSTATS)) {
            String[] f = line.trim().split("\\s+");
            if (f.length < 14) continue;
            String dev = f[2];
            if (!DISK_DEVICE_PATTERN.matcher(dev).matches()) continue;
            if (disks.length() > 0) disks.append(',');
            disks.append(String.format(
                    "{\"device\":\"%s\",\"readsCompleted\":%s,\"readsMerged\":%s,\"sectorsRead\":%s,\"readTimeMs\":%s,"
                  + "\"writesCompleted\":%s,\"writesMerged\":%s,\"sectorsWritten\":%s,\"writeTimeMs\":%s,"
                  + "\"ioInProgress\":%s,\"ioTimeMs\":%s,\"weightedIoTimeMs\":%s}",
                    dev, f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12], f[13]));
        }

        return String.format(
                "{\"timestamp\":\"%s\",\"cpuSockets\":%d,\"cpuCores\":%d,\"cpuThreads\":%d,"
              + "\"loadAvg1\":%s,\"loadAvg5\":%s,\"loadAvg15\":%s,\"runningProcs\":%s,\"totalProcs\":%s,"
              + "\"memTotalKB\":%d,\"memFreeKB\":%d,\"memAvailableKB\":%d,\"buffersKB\":%d,\"cachedKB\":%d,"
              + "\"swapTotalKB\":%d,\"swapFreeKB\":%d,\"diskStats\":[%s]}",
                ts, cpuSockets, cpuCores, cpuThreads,
                load1, load5, load15, running, total,
                memTotal, memFree, memAvail, buffers, cached, swapTotal, swapFree,
                disks);
    }

    private static long parseMemValue(String line) {
        String[] parts = line.split("\\s+");
        return Long.parseLong(parts[1]);
    }
}
