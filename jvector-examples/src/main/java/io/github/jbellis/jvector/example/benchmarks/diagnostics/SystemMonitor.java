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
    /** Platform-specific OS bean for extended metrics. */
    private final com.sun.management.OperatingSystemMXBean sunOsBean;

    /**
     * Constructs a SystemMonitor that initializes connections to system management beans.
     */
    public SystemMonitor() {
        this.memoryBean = ManagementFactory.getMemoryMXBean();
        this.gcBeans = ManagementFactory.getGarbageCollectorMXBeans();
        this.osBean = ManagementFactory.getOperatingSystemMXBean();
        this.threadBean = ManagementFactory.getThreadMXBean();
        this.sunOsBean = (com.sun.management.OperatingSystemMXBean) osBean;
    }

    /**
     * Captures the current system state snapshot including GC, memory, CPU, and thread statistics.
     *
     * @return a snapshot of the current system state
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
     * Logs the difference between two snapshots to standard output.
     *
     * @param phase the name of the phase being measured
     * @param before the snapshot taken before the phase
     * @param after the snapshot taken after the phase
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
     * Logs detailed garbage collection information to standard output.
     *
     * @param phase the name of the phase to include in the output
     */
    public void logDetailedGCStats(String phase) {
        System.out.printf("[%s] Detailed GC Stats:%n", phase);
        for (GarbageCollectorMXBean gcBean : gcBeans) {
            System.out.printf("  %s: %d collections, %d ms total%n",
                gcBean.getName(), gcBean.getCollectionCount(), gcBean.getCollectionTime());
        }
    }

    /**
     * Contains a complete snapshot of system state at a point in time.
     */
    public static class SystemSnapshot {
        /** The timestamp when this snapshot was captured (milliseconds). */
        public final long timestamp;
        /** Garbage collection statistics. */
        public final GCStats gcStats;
        /** Memory usage statistics. */
        public final MemoryStats memoryStats;
        /** CPU usage statistics. */
        public final CPUStats cpuStats;
        /** Thread statistics. */
        public final ThreadStats threadStats;

        /**
         * Constructs a SystemSnapshot with the specified metrics.
         *
         * @param timestamp the timestamp when captured
         * @param gcStats garbage collection statistics
         * @param memoryStats memory usage statistics
         * @param cpuStats CPU usage statistics
         * @param threadStats thread statistics
         */
        public SystemSnapshot(long timestamp, GCStats gcStats, MemoryStats memoryStats,
                            CPUStats cpuStats, ThreadStats threadStats) {
            this.timestamp = timestamp;
            this.gcStats = gcStats;
            this.memoryStats = memoryStats;
            this.cpuStats = cpuStats;
            this.threadStats = threadStats;
        }
    }

    /**
     * Contains garbage collection statistics.
     */
    public static class GCStats {
        /** Total number of garbage collections. */
        public final long totalCollections;
        /** Total time spent in garbage collection (milliseconds). */
        public final long totalCollectionTime;
        /** Number of garbage collectors. */
        public final int gcCount;

        /**
         * Constructs GCStats with the specified metrics.
         *
         * @param totalCollections total number of collections
         * @param totalCollectionTime total time spent in collections (ms)
         * @param gcCount number of garbage collectors
         */
        public GCStats(long totalCollections, long totalCollectionTime, int gcCount) {
            this.totalCollections = totalCollections;
            this.totalCollectionTime = totalCollectionTime;
            this.gcCount = gcCount;
        }

        /**
         * Computes the difference between this and another GCStats.
         *
         * @param other the GCStats to subtract from this one
         * @return a new GCStats representing the difference
         */
        public GCStats subtract(GCStats other) {
            return new GCStats(
                this.totalCollections - other.totalCollections,
                this.totalCollectionTime - other.totalCollectionTime,
                this.gcCount
            );
        }
    }

    /**
     * Contains memory usage statistics.
     */
    public static class MemoryStats {
        /** Heap memory currently used (bytes). */
        public final long heapUsed;
        /** Maximum heap memory available (bytes). */
        public final long heapMax;
        /** Heap memory committed by the JVM (bytes). */
        public final long heapCommitted;
        /** Non-heap memory used (bytes). */
        public final long nonHeapUsed;
        /** Free memory in the runtime (bytes). */
        public final long freeMemory;
        /** Total memory in the runtime (bytes). */
        public final long totalMemory;
        /** Maximum memory the runtime can use (bytes). */
        public final long maxMemory;

        /**
         * Constructs MemoryStats with the specified metrics.
         *
         * @param heapUsed heap memory used (bytes)
         * @param heapMax maximum heap memory (bytes)
         * @param heapCommitted heap memory committed (bytes)
         * @param nonHeapUsed non-heap memory used (bytes)
         * @param freeMemory free memory (bytes)
         * @param totalMemory total memory (bytes)
         * @param maxMemory maximum memory (bytes)
         */
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

    /**
     * Contains CPU usage statistics.
     */
    public static class CPUStats {
        /** System-wide CPU load (0.0 to 1.0). */
        public final double systemCpuLoad;
        /** Process CPU load (0.0 to 1.0). */
        public final double processCpuLoad;
        /** System load average. */
        public final double systemLoadAverage;
        /** Number of available processors. */
        public final int availableProcessors;
        /** Free physical memory size (bytes). */
        public final long freePhysicalMemory;

        /**
         * Constructs CPUStats with the specified metrics.
         *
         * @param systemCpuLoad system-wide CPU load (0.0-1.0)
         * @param processCpuLoad process CPU load (0.0-1.0)
         * @param systemLoadAverage system load average
         * @param availableProcessors number of available processors
         * @param freePhysicalMemory free physical memory (bytes)
         */
        public CPUStats(double systemCpuLoad, double processCpuLoad, double systemLoadAverage,
                       int availableProcessors, long freePhysicalMemory) {
            this.systemCpuLoad = systemCpuLoad;
            this.processCpuLoad = processCpuLoad;
            this.systemLoadAverage = systemLoadAverage;
            this.availableProcessors = availableProcessors;
            this.freePhysicalMemory = freePhysicalMemory;
        }
    }

    /**
     * Contains thread statistics.
     */
    public static class ThreadStats {
        /** Number of active threads. */
        public final int activeThreads;
        /** Peak number of threads. */
        public final int peakThreads;
        /** Total number of threads started since JVM start. */
        public final long totalStartedThreads;

        /**
         * Constructs ThreadStats with the specified metrics.
         *
         * @param activeThreads number of active threads
         * @param peakThreads peak number of threads
         * @param totalStartedThreads total threads started
         */
        public ThreadStats(int activeThreads, int peakThreads, long totalStartedThreads) {
            this.activeThreads = activeThreads;
            this.peakThreads = peakThreads;
            this.totalStartedThreads = totalStartedThreads;
        }
    }
}
