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

package io.github.jbellis.jvector.example.benchmarks.diagnostics;

import java.lang.management.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Utility class for monitoring system resources during benchmark execution.
 * Tracks GC activity, memory usage (on-heap and off-heap), CPU load, and thread statistics.
 */
public class SystemMonitor {

    private final MemoryMXBean memoryBean;
    private final List<GarbageCollectorMXBean> gcBeans;
    private final OperatingSystemMXBean osBean;
    private final ThreadMXBean threadBean;
    private final com.sun.management.ThreadMXBean sunThreadBean;
    private final com.sun.management.OperatingSystemMXBean sunOsBean;
    private final List<MemoryPoolMXBean> memoryPoolBeans;
    private final List<BufferPoolMXBean> bufferPoolBeans;

    public SystemMonitor() {
        this.memoryBean = ManagementFactory.getMemoryMXBean();
        this.gcBeans = ManagementFactory.getGarbageCollectorMXBeans();
        this.osBean = ManagementFactory.getOperatingSystemMXBean();
        this.threadBean = ManagementFactory.getThreadMXBean();
        this.sunThreadBean = (com.sun.management.ThreadMXBean) threadBean;
        this.sunOsBean = (com.sun.management.OperatingSystemMXBean) osBean;
        this.memoryPoolBeans = ManagementFactory.getMemoryPoolMXBeans();
        this.bufferPoolBeans = ManagementFactory.getPlatformMXBeans(BufferPoolMXBean.class);

        sunThreadBean.setThreadAllocatedMemoryEnabled(true);
    }

    /**
     * Captures current system state snapshot
     */
    public SystemSnapshot captureSnapshot() {
        return new SystemSnapshot(
            System.currentTimeMillis(),
            captureGCStats(),
            captureMemoryStats(),
            captureCPUStats(),
            captureThreadStats()
        );
    }

    private GCStats captureGCStats() {
        long totalCollections = 0;
        long totalCollectionTime = 0;

        for (GarbageCollectorMXBean gcBean : gcBeans) {
            totalCollections += gcBean.getCollectionCount();
            totalCollectionTime += gcBean.getCollectionTime();
        }

        return new GCStats(totalCollections, totalCollectionTime, gcBeans.size());
    }

    private MemoryStats captureMemoryStats() {
        MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
        MemoryUsage nonHeapUsage = memoryBean.getNonHeapMemoryUsage();
        Runtime runtime = Runtime.getRuntime();

        // Calculate detailed off-heap memory usage
        long directBufferMemory = 0;
        long mappedBufferMemory = 0;
        for (BufferPoolMXBean pool : bufferPoolBeans) {
            if (pool.getName().equals("direct")) {
                directBufferMemory = pool.getMemoryUsed();
            } else if (pool.getName().equals("mapped")) {
                mappedBufferMemory = pool.getMemoryUsed();
            }
        }

        return new MemoryStats(
            heapUsage.getUsed(),
            heapUsage.getMax(),
            heapUsage.getCommitted(),
            nonHeapUsage.getUsed(),
            runtime.freeMemory(),
            runtime.totalMemory(),
            runtime.maxMemory(),
            directBufferMemory,
            mappedBufferMemory
        );
    }

    private CPUStats captureCPUStats() {
        return new CPUStats(
            sunOsBean.getSystemCpuLoad(),
            sunOsBean.getProcessCpuLoad(),
            osBean.getSystemLoadAverage(),
            osBean.getAvailableProcessors(),
            sunOsBean.getFreePhysicalMemorySize()
        );
    }

    private ThreadStats captureThreadStats() {
        long[] threadIds = sunThreadBean.getAllThreadIds();
        long[] allocatedBytes = sunThreadBean.getThreadAllocatedBytes(threadIds);
        ThreadInfo[] threadInfos = sunThreadBean.getThreadInfo(threadIds);

        var allocByThread = new HashMap<String, Long>();
        for (int i = 0; i < threadIds.length; i++) {
            if (threadInfos[i] != null && allocatedBytes[i] >= 0) {
                allocByThread.put(threadInfos[i].getThreadName(), allocatedBytes[i]);
            }
        }

        return new ThreadStats(
            threadBean.getThreadCount(),
            threadBean.getPeakThreadCount(),
            threadBean.getTotalStartedThreadCount(),
            new ThreadAllocStats(allocByThread)
        );
    }

    /**
     * Logs the difference between two snapshots
     */
    public void logDifference(String phase, SystemSnapshot before, SystemSnapshot after) {
        System.out.printf("[%s] System Changes:%n", phase);

        // GC changes
        GCStats gcDiff = after.gcStats.subtract(before.gcStats);
        if (gcDiff.totalCollections > 0) {
            System.out.printf("  GC: %d collections, %d ms total%n",
                gcDiff.totalCollections, gcDiff.totalCollectionTime);
        } else {
            System.out.printf("  GC: No collections%n");
        }

        // Memory changes
        MemoryStats memAfter = after.memoryStats;
        MemoryStats memBefore = before.memoryStats;
        System.out.printf("  Heap: %d MB used / %d MB max (change: %+d MB)%n",
            memAfter.heapUsed / 1024 / 1024,
            memAfter.heapMax / 1024 / 1024,
            (memAfter.heapUsed - memBefore.heapUsed) / 1024 / 1024);
        System.out.printf("  Off-Heap: Direct=%d MB, Mapped=%d MB (change: %+d MB, %+d MB)%n",
            memAfter.directBufferMemory / 1024 / 1024,
            memAfter.mappedBufferMemory / 1024 / 1024,
            (memAfter.directBufferMemory - memBefore.directBufferMemory) / 1024 / 1024,
            (memAfter.mappedBufferMemory - memBefore.mappedBufferMemory) / 1024 / 1024);

        // CPU stats
        CPUStats cpuAfter = after.cpuStats;
        System.out.printf("  CPU Load: %.2f%% (process: %.2f%%)%n",
            cpuAfter.systemCpuLoad * 100, cpuAfter.processCpuLoad * 100);
        System.out.printf("  System Load Average: %.2f%n", cpuAfter.systemLoadAverage);

        // Thread changes
        ThreadStats threadAfter = after.threadStats;
        System.out.printf("  Threads: %d active, %d peak%n",
            threadAfter.activeThreads, threadAfter.peakThreads);

        // Per-thread allocation deltas
        if (threadAfter.allocStats != null && before.threadStats.allocStats != null) {
            Map<String, Long> beforeAlloc = before.threadStats.allocStats.allocatedBytesByThread;
            Map<String, Long> afterAlloc = threadAfter.allocStats.allocatedBytesByThread;
            long totalDelta = 0;
            for (var entry : afterAlloc.entrySet()) {
                long prev = beforeAlloc.getOrDefault(entry.getKey(), 0L);
                long delta = entry.getValue() - prev;
                if (delta > 0) {
                    totalDelta += delta;
                }
            }
            if (totalDelta > 0) {
                System.out.printf("  Thread Allocations: %+d MB total delta%n", totalDelta / 1024 / 1024);
                // Show top 5 allocating threads
                afterAlloc.entrySet().stream()
                    .map(e -> Map.entry(e.getKey(), e.getValue() - beforeAlloc.getOrDefault(e.getKey(), 0L)))
                    .filter(e -> e.getValue() > 0)
                    .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
                    .limit(5)
                    .forEach(e -> System.out.printf("    %s: %+d MB%n", e.getKey(), e.getValue() / 1024 / 1024));
            }
        }

        System.out.printf("  Duration: %d ms%n", after.timestamp - before.timestamp);
    }

    /**
     * Logs detailed GC information
     */
    public void logDetailedGCStats(String phase) {
        System.out.printf("[%s] Detailed GC Stats:%n", phase);
        for (GarbageCollectorMXBean gcBean : gcBeans) {
            System.out.printf("  %s: %d collections, %d ms total%n",
                gcBean.getName(), gcBean.getCollectionCount(), gcBean.getCollectionTime());
        }
    }

    // Inner classes for data structures
    public static class SystemSnapshot {
        public final long timestamp;
        public final GCStats gcStats;
        public final MemoryStats memoryStats;
        public final CPUStats cpuStats;
        public final ThreadStats threadStats;

        public SystemSnapshot(long timestamp, GCStats gcStats, MemoryStats memoryStats,
                            CPUStats cpuStats, ThreadStats threadStats) {
            this.timestamp = timestamp;
            this.gcStats = gcStats;
            this.memoryStats = memoryStats;
            this.cpuStats = cpuStats;
            this.threadStats = threadStats;
        }
    }

    public static class GCStats {
        public final long totalCollections;
        public final long totalCollectionTime;
        public final int gcCount;

        public GCStats(long totalCollections, long totalCollectionTime, int gcCount) {
            this.totalCollections = totalCollections;
            this.totalCollectionTime = totalCollectionTime;
            this.gcCount = gcCount;
        }

        public GCStats subtract(GCStats other) {
            return new GCStats(
                this.totalCollections - other.totalCollections,
                this.totalCollectionTime - other.totalCollectionTime,
                this.gcCount
            );
        }
    }

    public static class MemoryStats {
        public final long heapUsed;
        public final long heapMax;
        public final long heapCommitted;
        public final long nonHeapUsed;
        public final long freeMemory;
        public final long totalMemory;
        public final long maxMemory;
        public final long directBufferMemory;
        public final long mappedBufferMemory;

        public MemoryStats(long heapUsed, long heapMax, long heapCommitted, long nonHeapUsed,
                          long freeMemory, long totalMemory, long maxMemory,
                          long directBufferMemory, long mappedBufferMemory) {
            this.heapUsed = heapUsed;
            this.heapMax = heapMax;
            this.heapCommitted = heapCommitted;
            this.nonHeapUsed = nonHeapUsed;
            this.freeMemory = freeMemory;
            this.totalMemory = totalMemory;
            this.maxMemory = maxMemory;
            this.directBufferMemory = directBufferMemory;
            this.mappedBufferMemory = mappedBufferMemory;
        }

        public long getTotalOffHeapMemory() {
            return directBufferMemory + mappedBufferMemory;
        }
    }

    public static class CPUStats {
        public final double systemCpuLoad;
        public final double processCpuLoad;
        public final double systemLoadAverage;
        public final int availableProcessors;
        public final long freePhysicalMemory;

        public CPUStats(double systemCpuLoad, double processCpuLoad, double systemLoadAverage,
                       int availableProcessors, long freePhysicalMemory) {
            this.systemCpuLoad = systemCpuLoad;
            this.processCpuLoad = processCpuLoad;
            this.systemLoadAverage = systemLoadAverage;
            this.availableProcessors = availableProcessors;
            this.freePhysicalMemory = freePhysicalMemory;
        }
    }

    public static class ThreadStats {
        public final int activeThreads;
        public final int peakThreads;
        public final long totalStartedThreads;
        public final ThreadAllocStats allocStats;

        public ThreadStats(int activeThreads, int peakThreads, long totalStartedThreads,
                          ThreadAllocStats allocStats) {
            this.activeThreads = activeThreads;
            this.peakThreads = peakThreads;
            this.totalStartedThreads = totalStartedThreads;
            this.allocStats = allocStats;
        }
    }

    /// Per-thread heap allocation snapshot captured via
    /// {@link com.sun.management.ThreadMXBean#getThreadAllocatedBytes(long[])}.
    public static class ThreadAllocStats {
        /// Map of thread name to cumulative allocated bytes.
        public final Map<String, Long> allocatedBytesByThread;

        public ThreadAllocStats(Map<String, Long> allocatedBytesByThread) {
            this.allocatedBytesByThread = allocatedBytesByThread;
        }
    }
}
