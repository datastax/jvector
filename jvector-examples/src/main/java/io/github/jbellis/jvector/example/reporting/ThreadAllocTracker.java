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
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadInfo;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

/**
 * Periodically samples per-thread heap allocation via
 * {@link com.sun.management.ThreadMXBean#getThreadAllocatedBytes(long[])}
 * and writes JSONL output with per-thread deltas and cumulative totals.
 *
 * Lifecycle mirrors {@link SystemStatsCollector}: {@link #start(Path, String)},
 * {@link #stop()}, {@link #isActive()}, {@link #getFileName()}.
 */
public final class ThreadAllocTracker {
    private static final Logger log = LoggerFactory.getLogger(ThreadAllocTracker.class);

    private static final long DEFAULT_INTERVAL_SECONDS = 10;

    private final com.sun.management.ThreadMXBean threadMXBean;
    private final long intervalSeconds;

    private volatile Thread samplerThread;
    private volatile boolean running;
    private String fileName;

    /// Creates a tracker with the default 10-second sampling interval.
    public ThreadAllocTracker() {
        this(DEFAULT_INTERVAL_SECONDS);
    }

    /// Creates a tracker with a custom sampling interval.
    ///
    /// @param intervalSeconds seconds between each sample
    public ThreadAllocTracker(long intervalSeconds) {
        this.threadMXBean = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        this.intervalSeconds = intervalSeconds;
    }

    /// Creates the output directory, enables thread allocated memory tracking,
    /// and spawns a daemon thread that periodically writes JSONL samples.
    ///
    /// @param outputDir directory to write the JSONL file into
    /// @param fileName  name of the output file
    /// @return the absolute path of the output file
    /// @throws IOException if the directory cannot be created
    public Path start(Path outputDir, String fileName) throws IOException {
        Files.createDirectories(outputDir);
        Path outputPath = outputDir.resolve(fileName).toAbsolutePath();
        this.fileName = fileName;

        threadMXBean.setThreadAllocatedMemoryEnabled(true);

        running = true;
        samplerThread = new Thread(() -> sampleLoop(outputPath), "thread-alloc-tracker");
        samplerThread.setDaemon(true);
        samplerThread.start();

        log.info("Thread allocation tracking started, saving to: {}", outputPath);
        return outputPath;
    }

    /// Stops the sampler thread and writes a final cumulative summary line.
    public void stop() {
        running = false;
        if (samplerThread != null) {
            samplerThread.interrupt();
            try {
                samplerThread.join(5000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            samplerThread = null;
            log.info("Thread allocation tracking stopped, saved to: {}", fileName);
        }
    }

    /// Returns {@code true} if the sampler thread is currently running.
    public boolean isActive() {
        return samplerThread != null && running;
    }

    /// Returns the current file name, or {@code null} if tracking has not been started.
    public String getFileName() {
        return fileName;
    }

    private void sampleLoop(Path outputPath) {
        // Track cumulative allocations per thread (by id) for delta computation
        var previousAllocations = new HashMap<Long, Long>();

        try (var writer = Files.newBufferedWriter(outputPath)) {
            while (running) {
                try {
                    Thread.sleep(intervalSeconds * 1000);
                } catch (InterruptedException e) {
                    // On interrupt (from stop()), write final summary and exit
                    break;
                }
                writeSample(writer, previousAllocations, false);
            }
            // Write final summary with cumulative totals
            writeSample(writer, previousAllocations, true);
        } catch (IOException e) {
            log.error("Failed to write thread allocation sample", e);
        }
    }

    private void writeSample(BufferedWriter writer, Map<Long, Long> previousAllocations, boolean isSummary)
            throws IOException {
        long[] threadIds = threadMXBean.getAllThreadIds();
        long[] allocatedBytes = threadMXBean.getThreadAllocatedBytes(threadIds);
        ThreadInfo[] threadInfos = threadMXBean.getThreadInfo(threadIds);

        var sb = new StringBuilder();
        sb.append("{\"timestamp\":\"").append(Instant.now().toString()).append('"');
        if (isSummary) {
            sb.append(",\"event\":\"summary\"");
        }
        sb.append(",\"threads\":[");

        long totalAllocated = 0;
        long totalDelta = 0;
        boolean first = true;

        for (int i = 0; i < threadIds.length; i++) {
            if (threadInfos[i] == null || allocatedBytes[i] < 0) {
                continue;
            }
            long id = threadIds[i];
            long allocated = allocatedBytes[i];
            long previous = previousAllocations.getOrDefault(id, 0L);
            long delta = allocated - previous;
            previousAllocations.put(id, allocated);

            totalAllocated += allocated;
            totalDelta += delta;

            if (!first) {
                sb.append(',');
            }
            first = false;

            sb.append("{\"id\":").append(id)
              .append(",\"name\":\"").append(escapeJson(threadInfos[i].getThreadName())).append('"')
              .append(",\"allocatedBytes\":").append(allocated)
              .append(",\"deltaBytes\":").append(delta)
              .append('}');
        }

        sb.append("],\"totalAllocatedBytes\":").append(totalAllocated)
          .append(",\"totalDeltaBytes\":").append(totalDelta)
          .append('}');

        writer.write(sb.toString());
        writer.newLine();
        writer.flush();
    }

    private static String escapeJson(String value) {
        if (value == null) {
            return "";
        }
        var sb = new StringBuilder(value.length());
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"':  sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n");  break;
                case '\r': sb.append("\\r");  break;
                case '\t': sb.append("\\t");  break;
                default:   sb.append(c);
            }
        }
        return sb.toString();
    }
}
