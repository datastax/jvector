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

package io.github.jbellis.jvector.bench.benchtools;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

/**
 * Background collector of {@code /proc} system metrics (CPU topology, load, memory, disk I/O).
 * Spawns a bash process that appends JSONL lines to a file every 30 seconds.
 */
public final class SystemStatsCollector {
    private static final Logger log = LoggerFactory.getLogger(SystemStatsCollector.class);

    private static final String SCRIPT = String.join("\n",
            "cpuThreads=$(grep -c '^processor' /proc/cpuinfo)",
            "cpuSockets=$(awk '/^physical id/{print $NF}' /proc/cpuinfo | sort -u | wc -l)",
            "[ \"$cpuSockets\" -eq 0 ] && cpuSockets=1",
            "cpuCores=$(awk '/^physical id/{pid=$NF} /^core id/{print pid\"-\"$NF}' /proc/cpuinfo | sort -u | wc -l)",
            "[ \"$cpuCores\" -eq 0 ] && cpuCores=$cpuThreads",
            "while true; do",
            "  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
            "  read load1 load5 load15 runprocs rest < /proc/loadavg",
            "  IFS='/' read running total <<< \"$runprocs\"",
            "  memTotal=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)",
            "  memFree=$(awk '/^MemFree:/{print $2}' /proc/meminfo)",
            "  memAvail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)",
            "  buffers=$(awk '/^Buffers:/{print $2}' /proc/meminfo)",
            "  cached=$(awk '/^Cached:/{print $2}' /proc/meminfo)",
            "  swapTotal=$(awk '/^SwapTotal:/{print $2}' /proc/meminfo)",
            "  swapFree=$(awk '/^SwapFree:/{print $2}' /proc/meminfo)",
            "  disks=\"\"",
            "  while read maj min dev reads rmerged rsectors rtime writes wmerged wsectors wtime inprog iotime wiotime rest; do",
            "    if echo \"$dev\" | grep -qxE '(sd[a-z]+|nvme[0-9]+n[0-9]+|vd[a-z]+|xvd[a-z]+)'; then",
            "      [ -n \"$disks\" ] && disks=\"$disks,\"",
            "      disks=\"$disks{\\\"device\\\":\\\"$dev\\\",\\\"readsCompleted\\\":$reads,\\\"readsMerged\\\":$rmerged,\\\"sectorsRead\\\":$rsectors,\\\"readTimeMs\\\":$rtime,\\\"writesCompleted\\\":$writes,\\\"writesMerged\\\":$wmerged,\\\"sectorsWritten\\\":$wsectors,\\\"writeTimeMs\\\":$wtime,\\\"ioInProgress\\\":$inprog,\\\"ioTimeMs\\\":$iotime,\\\"weightedIoTimeMs\\\":$wiotime}\"",
            "    fi",
            "  done < /proc/diskstats",
            "  echo \"{\\\"timestamp\\\":\\\"$ts\\\",\\\"cpuSockets\\\":$cpuSockets,\\\"cpuCores\\\":$cpuCores,\\\"cpuThreads\\\":$cpuThreads,\\\"loadAvg1\\\":$load1,\\\"loadAvg5\\\":$load5,\\\"loadAvg15\\\":$load15,\\\"runningProcs\\\":$running,\\\"totalProcs\\\":$total,\\\"memTotalKB\\\":$memTotal,\\\"memFreeKB\\\":$memFree,\\\"memAvailableKB\\\":$memAvail,\\\"buffersKB\\\":$buffers,\\\"cachedKB\\\":$cached,\\\"swapTotalKB\\\":$swapTotal,\\\"swapFreeKB\\\":$swapFree,\\\"diskStats\\\":[$disks]}\"",
            "  sleep 30",
            "done");

    private Process process;
    private String fileName;

    /**
     * Creates the output directory, spawns the bash collector process, and returns the absolute path of the output file.
     *
     * @param outputDir directory to write the stats file into
     * @param fileName  name of the output JSONL file
     * @return the absolute path of the stats file
     * @throws IOException if the directory cannot be created or the process fails to start
     */
    public Path start(Path outputDir, String fileName) throws IOException {
        Files.createDirectories(outputDir);
        Path sysStatsPath = outputDir.resolve(fileName).toAbsolutePath();
        var pb = new ProcessBuilder("bash", "-c", SCRIPT);
        pb.redirectOutput(ProcessBuilder.Redirect.to(sysStatsPath.toFile()));
        pb.redirectErrorStream(true);
        process = pb.start();
        this.fileName = fileName;
        log.info("System stats collection started, saving to: {}", sysStatsPath);
        return sysStatsPath;
    }

    /** Destroys the process (with a 5-second wait) and logs the saved path. */
    public void stop(Path outputDir) throws InterruptedException {
        if (process != null) {
            process.destroy();
            process.waitFor(5, TimeUnit.SECONDS);
            process = null;
            log.info("System stats collection stopped, saved to: {}", outputDir.resolve(fileName).toAbsolutePath());
        }
    }

    /** Returns {@code true} if the background process is currently running. */
    public boolean isActive() {
        return process != null;
    }

    /** Returns the current file name, or {@code null} if collection has not been started. */
    public String getFileName() {
        return fileName;
    }
}
