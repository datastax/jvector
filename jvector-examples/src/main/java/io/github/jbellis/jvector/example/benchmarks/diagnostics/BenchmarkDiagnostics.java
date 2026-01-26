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

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * Main diagnostics coordinator for benchmarks. Provides a unified interface
 * for collecting system metrics, performance data, disk usage, and analyzing benchmark results.
 *
 * <p>This class now uses an event-driven DiskUsageMonitor that must be properly started
 * and closed. Use try-with-resources or explicitly call close() when done.
 */
public class BenchmarkDiagnostics implements AutoCloseable {

    private final DiagnosticLevel level;
    private final SystemMonitor systemMonitor;
    private final DiskUsageMonitor diskUsageMonitor;
    private final PerformanceAnalyzer performanceAnalyzer;
    private final List<SystemMonitor.SystemSnapshot> snapshots;
    private final List<DiskUsageMonitor.DiskUsageSnapshot> diskSnapshots;
    private final List<PerformanceAnalyzer.TimingAnalysis> timingAnalyses;
    private Path monitoredDirectory;
    private boolean diskMonitorStarted = false;

    public BenchmarkDiagnostics(DiagnosticLevel level) {
        this.level = level;
        this.systemMonitor = new SystemMonitor();
        this.diskUsageMonitor = new DiskUsageMonitor();
        this.performanceAnalyzer = new PerformanceAnalyzer();
        this.snapshots = new ArrayList<>();
        this.diskSnapshots = new ArrayList<>();
        this.timingAnalyses = new ArrayList<>();
        this.monitoredDirectory = null;
    }

    /**
     * Creates a BenchmarkDiagnostics instance with BASIC level diagnostics
     */
    public static BenchmarkDiagnostics createBasic() {
        return new BenchmarkDiagnostics(DiagnosticLevel.BASIC);
    }

    /**
     * Creates a BenchmarkDiagnostics instance with DETAILED level diagnostics
     */
    public static BenchmarkDiagnostics createDetailed() {
        return new BenchmarkDiagnostics(DiagnosticLevel.DETAILED);
    }

    /**
     * Creates a BenchmarkDiagnostics instance with VERBOSE level diagnostics
     */
    public static BenchmarkDiagnostics createVerbose() {
        return new BenchmarkDiagnostics(DiagnosticLevel.VERBOSE);
    }

    /**
     * Sets the directory to monitor for disk usage and starts event-driven monitoring.
     * This should be called before capturing any snapshots for optimal performance.
     *
     * @param directory the directory to monitor
     * @throws IOException if unable to start monitoring
     */
    public void setMonitoredDirectory(Path directory) throws IOException {
        this.monitoredDirectory = directory;
        if (directory != null && !diskMonitorStarted) {
            diskUsageMonitor.start(directory);
            diskMonitorStarted = true;
        }
    }

    /**
     * Captures system state before starting a benchmark phase
     */
    public void capturePrePhaseSnapshot(String phase) {
        SystemMonitor.SystemSnapshot snapshot = systemMonitor.captureSnapshot();
        snapshots.add(snapshot);

        // Capture disk usage if directory is set
        if (monitoredDirectory != null) {
            try {
                DiskUsageMonitor.DiskUsageSnapshot diskSnapshot = diskUsageMonitor.captureSnapshot(monitoredDirectory);
                diskSnapshots.add(diskSnapshot);
            } catch (IOException e) {
                if (level != DiagnosticLevel.NONE) {
                    System.err.printf("[%s] Failed to capture disk usage: %s%n", phase, e.getMessage());
                }
            }
        }

        if (level == DiagnosticLevel.VERBOSE) {
            System.out.printf("[%s] Pre-phase system snapshot captured%n", phase);
            systemMonitor.logDetailedGCStats(phase + "-Pre");
        }
    }

    /**
     * Captures system state after completing a benchmark phase and logs changes
     */
    public void capturePostPhaseSnapshot(String phase) {
        SystemMonitor.SystemSnapshot postSnapshot = systemMonitor.captureSnapshot();

        if (!snapshots.isEmpty() && level != DiagnosticLevel.NONE) {
            SystemMonitor.SystemSnapshot preSnapshot = snapshots.get(snapshots.size() - 1);
            systemMonitor.logDifference(phase, preSnapshot, postSnapshot);
        }

        snapshots.add(postSnapshot);

        // Capture and log disk usage changes
        if (monitoredDirectory != null) {
            try {
                DiskUsageMonitor.DiskUsageSnapshot postDiskSnapshot = diskUsageMonitor.captureSnapshot(monitoredDirectory);
                if (!diskSnapshots.isEmpty() && level != DiagnosticLevel.NONE) {
                    DiskUsageMonitor.DiskUsageSnapshot preDiskSnapshot = diskSnapshots.get(diskSnapshots.size() - 1);
                    diskUsageMonitor.logDifference(phase, preDiskSnapshot, postDiskSnapshot);
                }
                diskSnapshots.add(postDiskSnapshot);
            } catch (IOException e) {
                if (level != DiagnosticLevel.NONE) {
                    System.err.printf("[%s] Failed to capture disk usage: %s%n", phase, e.getMessage());
                }
            }
        }

        if (level == DiagnosticLevel.VERBOSE) {
            systemMonitor.logDetailedGCStats(phase + "-Post");
        }
    }

    /**
     * Records the execution time of a single query (for detailed timing analysis)
     */
    public void recordQueryTime(long nanoTime) {
        if (level == DiagnosticLevel.DETAILED || level == DiagnosticLevel.VERBOSE) {
            performanceAnalyzer.recordQueryTime(nanoTime);
        }
    }

    /**
     * Analyzes and logs timing data for a phase
     */
    public void analyzePhaseTimings(String phase) {
        if (level == DiagnosticLevel.DETAILED || level == DiagnosticLevel.VERBOSE) {
            PerformanceAnalyzer.TimingAnalysis analysis = performanceAnalyzer.analyzeTimings(phase);
            performanceAnalyzer.logAnalysis(analysis);
            timingAnalyses.add(analysis);
            performanceAnalyzer.reset();
        }
    }

    /**
     * Executes a benchmark phase with full diagnostic monitoring
     */
    public <T> T monitorPhase(String phase, Supplier<T> benchmarkCode) {
        capturePrePhaseSnapshot(phase);

        long startTime = System.nanoTime();
        T result = benchmarkCode.get();
        long endTime = System.nanoTime();

        capturePostPhaseSnapshot(phase);

        if (level == DiagnosticLevel.BASIC || level == DiagnosticLevel.DETAILED || level == DiagnosticLevel.VERBOSE) {
            System.out.printf("[%s] Phase completed in %.2f ms%n", phase, (endTime - startTime) / 1e6);
        }

        return result;
    }

    /**
     * Executes a benchmark phase with detailed query timing
     */
    public <T> T monitorPhaseWithQueryTiming(String phase, QueryTimingBenchmark<T> benchmarkCode) {
        capturePrePhaseSnapshot(phase);

        long startTime = System.nanoTime();
        T result = benchmarkCode.execute(this::recordQueryTime);
        long endTime = System.nanoTime();

        capturePostPhaseSnapshot(phase);
        analyzePhaseTimings(phase);

        if (level == DiagnosticLevel.BASIC || level == DiagnosticLevel.DETAILED || level == DiagnosticLevel.VERBOSE) {
            System.out.printf("[%s] Phase completed in %.2f ms%n", phase, (endTime - startTime) / 1e6);
        }

        return result;
    }

    public void console(String s) {
        if (level != DiagnosticLevel.NONE ) {
            System.out.println(s);
        }
    }

    /**
     * Gets the latest system snapshot, or null if none captured
     */
    public SystemMonitor.SystemSnapshot getLatestSystemSnapshot() {
        return snapshots.isEmpty() ? null : snapshots.get(snapshots.size() - 1);
    }

    /**
     * Gets the latest disk usage snapshot, or null if none captured
     */
    public DiskUsageMonitor.DiskUsageSnapshot getLatestDiskSnapshot() {
        return diskSnapshots.isEmpty() ? null : diskSnapshots.get(diskSnapshots.size() - 1);
    }

    /**
     * Compares performance between different phases
     */
    public void comparePhases(String baselinePhase, String currentPhase) {
        if (timingAnalyses.size() < 2) return;

        PerformanceAnalyzer.TimingAnalysis baseline = timingAnalyses.stream()
            .filter(analysis -> analysis.phase.equals(baselinePhase))
            .findFirst()
            .orElse(null);

        PerformanceAnalyzer.TimingAnalysis current = timingAnalyses.stream()
            .filter(analysis -> analysis.phase.equals(currentPhase))
            .findFirst()
            .orElse(null);

        if (baseline != null && current != null) {
            PerformanceAnalyzer.PerformanceComparison comparison =
                PerformanceAnalyzer.compareRuns(baseline, current);
            PerformanceAnalyzer.logComparison(comparison);
        }
    }

    /**
     * Logs a summary of all collected diagnostic data
     */
    public void logSummary() {
        if (level == DiagnosticLevel.NONE) return;

        System.out.println("\n=== BENCHMARK DIAGNOSTICS SUMMARY ===");
        System.out.printf("Diagnostic Level: %s%n", level);
        System.out.printf("System Snapshots: %d%n", snapshots.size());
        System.out.printf("Disk Snapshots: %d%n", diskSnapshots.size());
        System.out.printf("Timing Analyses: %d%n", timingAnalyses.size());

        if (!snapshots.isEmpty()) {
            SystemMonitor.SystemSnapshot first = snapshots.get(0);
            SystemMonitor.SystemSnapshot last = snapshots.get(snapshots.size() - 1);

            System.out.printf("Total Benchmark Duration: %d ms%n", last.timestamp - first.timestamp);

            // Overall GC impact
            SystemMonitor.GCStats totalGC = last.gcStats.subtract(first.gcStats);
            if (totalGC.totalCollections > 0) {
                System.out.printf("Total GC Impact: %d collections, %d ms%n",
                    totalGC.totalCollections, totalGC.totalCollectionTime);
            }

            // Memory usage summary
            System.out.printf("\nMemory Usage Summary:%n");
            System.out.printf("  Final Heap: %s / %s%n",
                DiskUsageMonitor.formatBytes(last.memoryStats.heapUsed),
                DiskUsageMonitor.formatBytes(last.memoryStats.heapMax));
            System.out.printf("  Final Off-Heap: Direct=%s, Mapped=%s%n",
                DiskUsageMonitor.formatBytes(last.memoryStats.directBufferMemory),
                DiskUsageMonitor.formatBytes(last.memoryStats.mappedBufferMemory));
        }

        // Disk usage summary
        if (!diskSnapshots.isEmpty()) {
            DiskUsageMonitor.DiskUsageSnapshot firstDisk = diskSnapshots.get(0);
            DiskUsageMonitor.DiskUsageSnapshot lastDisk = diskSnapshots.get(diskSnapshots.size() - 1);
            DiskUsageMonitor.DiskUsageSnapshot totalDisk = lastDisk.subtract(firstDisk);

            System.out.printf("\nDisk Usage Summary:%n");
            System.out.printf("  Total Disk Used: %s%n", DiskUsageMonitor.formatBytes(lastDisk.totalBytes));
            System.out.printf("  Total Files: %d%n", lastDisk.fileCount);
            System.out.printf("  Net Change: %s, %+d files%n",
                DiskUsageMonitor.formatBytes(totalDisk.totalBytes), totalDisk.fileCount);
        }

        // Performance variance analysis
        if (timingAnalyses.size() > 1) {
            System.out.println("\nPerformance Variance Analysis:");
            for (int i = 1; i < timingAnalyses.size(); i++) {
                comparePhases(timingAnalyses.get(0).phase, timingAnalyses.get(i).phase);
            }
        }

        System.out.println("=====================================\n");
    }

    /**
     * Checks if warmup appears to be effective based on performance stabilization
     */
    public boolean isWarmupEffective() {
        if (timingAnalyses.size() < 2) return true;

        // Compare first and last timing analyses
        PerformanceAnalyzer.TimingAnalysis first = timingAnalyses.get(0);
        PerformanceAnalyzer.TimingAnalysis last = timingAnalyses.get(timingAnalyses.size() - 1);

        double p50Improvement = calculatePercentageChange(first.p50, last.p50);

        // If performance improved by more than 20%, warmup was likely insufficient
        return Math.abs(p50Improvement) < 20.0;
    }

    /**
     * Provides recommendations based on collected diagnostic data
     */
    public void provideRecommendations() {
        if (level == DiagnosticLevel.NONE) return;

        System.out.println("\n=== PERFORMANCE RECOMMENDATIONS ===");

        // GC recommendations
        if (!snapshots.isEmpty() && snapshots.size() >= 2) {
            SystemMonitor.SystemSnapshot first = snapshots.get(0);
            SystemMonitor.SystemSnapshot last = snapshots.get(snapshots.size() - 1);
            SystemMonitor.GCStats totalGC = last.gcStats.subtract(first.gcStats);

            if (totalGC.totalCollectionTime > 1000) { // More than 1 second of GC
                System.out.println("• Consider tuning GC settings - significant GC overhead detected");
            }

            // Memory pressure check
            if (last.memoryStats.heapUsed > last.memoryStats.heapMax * 0.8) {
                System.out.println("• Consider increasing heap size - high memory pressure detected");
            }
        }

        // Warmup recommendations
        if (!isWarmupEffective()) {
            System.out.println("• Consider increasing warmup iterations - performance still improving");
        }

        // Variance recommendations
        if (!timingAnalyses.isEmpty()) {
            PerformanceAnalyzer.TimingAnalysis latest = timingAnalyses.get(timingAnalyses.size() - 1);
            if (!latest.outliers.isEmpty() && latest.outliers.size() > latest.outliers.size() * 0.05) {
                System.out.println("• High query time variance detected - investigate system load or resource contention");
            }
        }

        System.out.println("====================================\n");
    }
    
    /**
     * Closes the diagnostics system and releases resources.
     * This must be called to properly shut down the disk usage monitor.
     */
    @Override
    public void close() throws IOException {
        if (diskMonitorStarted) {
            diskUsageMonitor.close();
            diskMonitorStarted = false;
        }
    }

    /**
     * Compares performance between runs and identifies significant changes
     */
    public static PerformanceAnalyzer.PerformanceComparison compareRuns(PerformanceAnalyzer.TimingAnalysis baseline, PerformanceAnalyzer.TimingAnalysis current) {
        double p50Change = calculatePercentageChange(baseline.p50, current.p50);
        double p95Change = calculatePercentageChange(baseline.p95, current.p95);
        double p99Change = calculatePercentageChange(baseline.p99, current.p99);
        double meanChange = calculatePercentageChange(baseline.mean, current.mean);

        boolean significantRegression = Math.abs(p50Change) > 10.0 || Math.abs(p95Change) > 15.0;

        return new PerformanceAnalyzer.PerformanceComparison(
                baseline.phase, current.phase,
                p50Change, p95Change, p99Change, meanChange,
                significantRegression
        );
    }

    public static double calculatePercentageChange(long baseline, long current) {
        if (baseline == 0) return current == 0 ? 0.0 : 100.0;
        return ((double)(current - baseline) / baseline) * 100.0;
    }

    /**
     * Logs performance comparison results
     */
    public static void logComparison(PerformanceAnalyzer.PerformanceComparison comparison) {
        System.out.printf("[%s vs %s] Performance Comparison:%n",
                comparison.baselinePhase, comparison.currentPhase);
        System.out.printf("  P50 change: %+.1f%%%n", comparison.p50Change);
        System.out.printf("  P95 change: %+.1f%%%n", comparison.p95Change);
        System.out.printf("  P99 change: %+.1f%%%n", comparison.p99Change);
        System.out.printf("  Mean change: %+.1f%%%n", comparison.meanChange);

        if (comparison.significantRegression) {
            System.out.printf("  ⚠️  SIGNIFICANT PERFORMANCE CHANGE DETECTED%n");
        }
    }

    /**
     * Functional interface for benchmark code that needs query timing
     */
    @FunctionalInterface
    public interface QueryTimingBenchmark<T> {
        T execute(QueryTimeRecorder recorder);
    }

    /**
     * Functional interface for recording individual query times
     */
    @FunctionalInterface
    public interface QueryTimeRecorder {
        void recordTime(long nanoTime);
    }
}
