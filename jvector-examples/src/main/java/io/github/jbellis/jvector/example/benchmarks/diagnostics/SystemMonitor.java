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
import java.util.List;

/**
 * Utility class for monitoring system resources during benchmark execution.
 * Tracks GC activity, memory usage, CPU load, and thread statistics.
 */
public class SystemMonitor {

    private final MemoryMXBean memoryBean;
    private final List<GarbageCollectorMXBean> gcBeans;
    private final OperatingSystemMXBean osBean;
    private final ThreadMXBean threadBean;
    private final com.sun.management.OperatingSystemMXBean sunOsBean;

    public SystemMonitor() {
        this.memoryBean = ManagementFactory.getMemoryMXBean();
        this.gcBeans = ManagementFactory.getGarbageCollectorMXBeans();
        this.osBean = ManagementFactory.getOperatingSystemMXBean();
        this.threadBean = ManagementFactory.getThreadMXBean();
        this.sunOsBean = (com.sun.management.OperatingSystemMXBean) osBean;
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

        return new MemoryStats(
            heapUsage.getUsed(),
            heapUsage.getMax(),
            heapUsage.getCommitted(),
            nonHeapUsage.getUsed(),
            runtime.freeMemory(),
            runtime.totalMemory(),
            runtime.maxMemory()
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
        return new ThreadStats(
            threadBean.getThreadCount(),
            threadBean.getPeakThreadCount(),
            threadBean.getTotalStartedThreadCount()
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
        System.out.printf("  Heap: %d MB used / %d MB max%n",
            memAfter.heapUsed / 1024 / 1024, memAfter.heapMax / 1024 / 1024);

        // CPU stats
        CPUStats cpuAfter = after.cpuStats;
        System.out.printf("  CPU Load: %.2f%% (process: %.2f%%)%n",
            cpuAfter.systemCpuLoad * 100, cpuAfter.processCpuLoad * 100);
        System.out.printf("  System Load Average: %.2f%n", cpuAfter.systemLoadAverage);

        // Thread changes
        ThreadStats threadAfter = after.threadStats;
        System.out.printf("  Threads: %d active, %d peak%n",
            threadAfter.activeThreads, threadAfter.peakThreads);

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

        public MemoryStats(long heapUsed, long heapMax, long heapCommitted, long nonHeapUsed,
                          long freeMemory, long totalMemory, long maxMemory) {
            this.heapUsed = heapUsed;
            this.heapMax = heapMax;
            this.heapCommitted = heapCommitted;
            this.nonHeapUsed = nonHeapUsed;
            this.freeMemory = freeMemory;
            this.totalMemory = totalMemory;
            this.maxMemory = maxMemory;
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

        public ThreadStats(int activeThreads, int peakThreads, long totalStartedThreads) {
            this.activeThreads = activeThreads;
            this.peakThreads = peakThreads;
            this.totalStartedThreads = totalStartedThreads;
        }
    }
}
