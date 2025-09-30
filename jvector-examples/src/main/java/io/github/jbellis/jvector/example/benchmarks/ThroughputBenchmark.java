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

package io.github.jbellis.jvector.example.benchmarks;

import io.github.jbellis.jvector.example.Grid.ConfiguredSystem;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.status.StatusTracker;
import io.github.jbellis.jvector.status.StatusUpdate;
import io.github.jbellis.jvector.status.TrackerScope;
import io.github.jbellis.jvector.status.sinks.ConsoleLoggerSink;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.example.benchmarks.diagnostics.BenchmarkDiagnostics;
import io.github.jbellis.jvector.example.benchmarks.diagnostics.DiagnosticLevel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.IntStream;
import org.apache.commons.math3.stat.StatUtils;

/**
 * Measures throughput (queries/sec) with an optional warmup phase.
 * Now includes comprehensive diagnostics to help identify performance variations.
 */
public class ThroughputBenchmark extends AbstractQueryBenchmark {
    private static final String DEFAULT_FORMAT = ".1f";

    private static volatile long SINK;

    private final int numWarmupRuns;
    private final int numTestRuns;
    private boolean computeAvgQps;
    private boolean computeMedianQps;
    private boolean computeMaxQps;
    private String formatAvgQps;
    private String formatMedianQps;
    private String formatMaxQps;
    private BenchmarkDiagnostics diagnostics;

    VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    public static ThroughputBenchmark createDefault() {
        return new ThroughputBenchmark(3, 3,
                true, false, false,
                DEFAULT_FORMAT, DEFAULT_FORMAT, DEFAULT_FORMAT,
                DiagnosticLevel.NONE);
    }

    public static ThroughputBenchmark createEmpty(int numWarmupRuns, int numTestRuns) {
        return new ThroughputBenchmark(numWarmupRuns, numTestRuns,
                false, false, false,
                DEFAULT_FORMAT, DEFAULT_FORMAT, DEFAULT_FORMAT,
                DiagnosticLevel.NONE);
    }

    private ThroughputBenchmark(int numWarmupRuns, int numTestRuns,
                                boolean computeAvgQps, boolean computeMedianQps, boolean computeMaxQps,
                                String formatAvgQps, String formatMedianQps, String formatMaxQps,
                                DiagnosticLevel diagnosticLevel) {
        this.numWarmupRuns = numWarmupRuns;
        this.numTestRuns = numTestRuns;
        this.computeAvgQps = computeAvgQps;
        this.computeMedianQps = computeMedianQps;
        this.computeMaxQps = computeMaxQps;
        this.formatAvgQps = formatAvgQps;
        this.formatMedianQps = formatMedianQps;
        this.formatMaxQps = formatMaxQps;
        this.diagnostics = new BenchmarkDiagnostics(diagnosticLevel);
    }

    public ThroughputBenchmark displayAvgQps() {
        return displayAvgQps(DEFAULT_FORMAT);
    }

    public ThroughputBenchmark displayAvgQps(String format) {
        this.computeAvgQps = true;
        this.formatAvgQps = format;
        return this;
    }

    public ThroughputBenchmark displayMedianQps() {
        return displayMedianQps(DEFAULT_FORMAT);
    }

    public ThroughputBenchmark displayMedianQps(String format) {
        this.computeMedianQps = true;
        this.formatMedianQps = format;
        return this;
    }

    public ThroughputBenchmark displayMaxQps() {
        return displayMaxQps(DEFAULT_FORMAT);
    }

    public ThroughputBenchmark displayMaxQps(String format) {
        this.computeMaxQps = true;
        this.formatMaxQps = format;
        return this;
    }

    /**
     * Configure the diagnostic level for this benchmark
     */
    public ThroughputBenchmark withDiagnostics(DiagnosticLevel level) {
        this.diagnostics = new BenchmarkDiagnostics(level);
        return this;
    }

    @Override
    public String getBenchmarkName() {
        return "ThroughputBenchmark";
    }

    @Override
    public List<Metric> runBenchmark(
            ConfiguredSystem cs,
            int topK,
            int rerankK,
            boolean usePruning,
            int queryRuns) {
        // Create a default scope if none provided
        TrackerScope defaultScope = new TrackerScope("ThroughputBenchmark",
                java.time.Duration.ofMillis(500),
                List.of(new ConsoleLoggerSink(System.out, true, false)));
        try {
            return runBenchmark(defaultScope, cs, topK, rerankK, usePruning, queryRuns);
        } finally {
            defaultScope.close();
        }
    }

    @Override
    public List<Metric> runBenchmark(
            TrackerScope parentScope,
            ConfiguredSystem cs,
            int topK,
            int rerankK,
            boolean usePruning,
            int queryRuns) {

        if (!(computeAvgQps || computeMedianQps || computeMaxQps)) {
            throw new RuntimeException("At least one metric must be displayed");
        }

        int totalQueries = cs.getDataSet().queryVectors.size();
        int dim = cs.getDataSet().getDimension();

        // Create a base task class for all benchmark phases
        class BenchmarkPhaseTask implements StatusUpdate.Provider<BenchmarkPhaseTask> {
            private volatile double progress = 0.0;
            private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;
            private final String phaseName;
            private volatile String detailedStatus = "";

            BenchmarkPhaseTask(String phaseName) {
                this.phaseName = phaseName;
            }

            @Override
            public StatusUpdate<BenchmarkPhaseTask> getTaskStatus() {
                return new StatusUpdate<>(progress, state, this);
            }

            void updateProgress(double progress) {
                this.progress = Math.min(1.0, progress);
                if (state == StatusUpdate.RunState.PENDING && progress > 0) {
                    state = StatusUpdate.RunState.RUNNING;
                }
            }

            void updateStatus(String status) {
                this.detailedStatus = status;
            }

            void start() {
                this.state = StatusUpdate.RunState.RUNNING;
                this.progress = 0.0;
            }

            void complete() {
                this.progress = 1.0;
                this.state = StatusUpdate.RunState.SUCCESS;
            }

            void fail() {
                this.state = StatusUpdate.RunState.FAILED;
            }

            @Override
            public String toString() {
                return phaseName + (detailedStatus.isEmpty() ? "" : ": " + detailedStatus);
            }
        }

        // Main benchmark task that will have child tasks for each phase
        class BenchmarkMainTask implements StatusUpdate.Provider<BenchmarkMainTask> {
            private volatile double progress = 0.0;
            private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;
            private final List<BenchmarkPhaseTask> allPhases;
            private volatile int completedPhases = 0;

            BenchmarkMainTask(int warmupRuns, int testRuns) {
                // Pre-declare all phases
                this.allPhases = new ArrayList<>();

                // Add warmup phases
                for (int i = 0; i < warmupRuns; i++) {
                    allPhases.add(new BenchmarkPhaseTask(String.format("Warmup Run %d/%d", i + 1, warmupRuns)));
                }

                // Add warmup analysis if there are warmup runs
                if (warmupRuns > 0) {
                    allPhases.add(new BenchmarkPhaseTask("Warmup Analysis"));
                }

                // Add test phases
                for (int i = 0; i < testRuns; i++) {
                    allPhases.add(new BenchmarkPhaseTask(String.format("Test Run %d/%d", i + 1, testRuns)));
                }

                // Add analysis phases
                allPhases.add(new BenchmarkPhaseTask("Performance Analysis"));
                if (testRuns > 1) {
                    allPhases.add(new BenchmarkPhaseTask("Regression Analysis"));
                }
                allPhases.add(new BenchmarkPhaseTask("Final Report"));
            }

            List<BenchmarkPhaseTask> getAllPhases() {
                return allPhases;
            }

            @Override
            public StatusUpdate<BenchmarkMainTask> getTaskStatus() {
                // Calculate progress based on phase completion
                double totalProgress = 0.0;
                for (BenchmarkPhaseTask phase : allPhases) {
                    totalProgress += phase.progress;
                }
                this.progress = allPhases.isEmpty() ? 0.0 : totalProgress / allPhases.size();
                return new StatusUpdate<>(progress, state, this);
            }

            void start() {
                this.state = StatusUpdate.RunState.RUNNING;
            }

            void phaseCompleted() {
                this.completedPhases++;
                // Progress is now calculated dynamically in getTaskStatus()
            }

            void complete() {
                this.progress = 1.0;
                this.state = StatusUpdate.RunState.SUCCESS;
            }

            @Override
            public String toString() {
                return String.format("ThroughputBenchmark (%.1f%% complete)", progress * 100);
            }
        }

        // Use the parent scope or create a child scope for this benchmark
        boolean ownScope = parentScope == null;
        TrackerScope benchmarkScope = ownScope ?
                new TrackerScope("ThroughputBenchmark",
                        java.time.Duration.ofMillis(500),
                        List.of(new ConsoleLoggerSink(System.out, true, false))) :
                parentScope.createChildScope("ThroughputBenchmark");

        // Declare statistics variables at the method level
        double avgQps = 0.0;
        double medianQps = 0.0;
        double stdDevQps = 0.0;
        double maxQps = 0.0;
        double minQps = 0.0;
        double coefficientOfVariation = 0.0;

        try {
            BenchmarkMainTask mainTask = new BenchmarkMainTask(numWarmupRuns, numTestRuns);

            // Pre-create trackers for all phases to show them in PENDING state
            List<StatusTracker<BenchmarkPhaseTask>> phaseTrackers = new ArrayList<>();
            Map<String, BenchmarkPhaseTask> phaseTaskMap = new HashMap<>();
            Map<String, StatusTracker<BenchmarkPhaseTask>> phaseTrackerMap = new HashMap<>();

            try (StatusTracker<BenchmarkMainTask> mainTracker = benchmarkScope.track(mainTask)) {
                mainTask.start();

                // Pre-declare all phase trackers as children of the main tracker
                mainTracker.executeWithContext(() -> {
                    for (BenchmarkPhaseTask phase : mainTask.getAllPhases()) {
                        StatusTracker<BenchmarkPhaseTask> tracker = mainTracker.createChild(phase);
                        phaseTrackers.add(tracker);
                        String phaseName = phase.toString();
                        phaseTaskMap.put(phaseName, phase);
                        phaseTrackerMap.put(phaseName, tracker);
                    }
                });

            // Warmup Phase with diagnostics
            double[] warmupQps = new double[numWarmupRuns];
            for (int warmupRun = 0; warmupRun < numWarmupRuns; warmupRun++) {
                // Use the pre-created task for this warmup run
                String phaseName = String.format("Warmup Run %d/%d", warmupRun + 1, numWarmupRuns);
                BenchmarkPhaseTask warmupTask = phaseTaskMap.get(phaseName);

                if (warmupTask != null) {
                    warmupTask.start();
                    warmupTask.updateStatus(String.format("Processing %d queries", totalQueries));
                    warmupTask.updateProgress(0.1);

                    String warmupPhase = "Warmup-" + warmupRun;
                    warmupQps[warmupRun] = diagnostics.monitorPhaseWithQueryTiming(warmupPhase, (recorder) -> {
                IntStream.range(0, totalQueries)
                        .parallel()
                        .forEach(k -> {
                            long queryStart = System.nanoTime();
                            
                            // Generate a random vector
                            VectorFloat<?> randQ = vts.createFloatVector(dim);
                            for (int j = 0; j < dim; j++) {
                                randQ.set(j, ThreadLocalRandom.current().nextFloat());
                            }
                            VectorUtil.l2normalize(randQ);
                            SearchResult sr = QueryExecutor.executeQuery(
                                    cs, topK, rerankK, usePruning, randQ);
                            SINK += sr.getVisitedCount();
                            
                            long queryEnd = System.nanoTime();
                            recorder.recordTime(queryEnd - queryStart);
                        });
                
                    return totalQueries / 1.0; // Return QPS placeholder
                });

                    warmupTask.updateProgress(1.0);
                    warmupTask.complete();
                }

                mainTask.phaseCompleted();
                diagnostics.console("Warmup Run " + warmupRun + ": " + warmupQps[warmupRun] + " QPS\n");
            }

            // Analyze warmup effectiveness
            if (numWarmupRuns > 1) {
                // Use the pre-created task for warmup analysis
                String analysisName = "Warmup Analysis";
                BenchmarkPhaseTask analysisTask = phaseTaskMap.get(analysisName);
                StatusTracker<BenchmarkPhaseTask> analysisTracker = phaseTrackerMap.get(analysisName);

                if (analysisTask != null && analysisTracker != null) {
                    analysisTask.start();
                    analysisTask.updateStatus("Analyzing warmup stability");

                    double warmupVariance = StatUtils.variance(warmupQps);
                    double warmupMean = StatUtils.mean(warmupQps);
                    double warmupCV = Math.sqrt(warmupVariance) / warmupMean * 100;
                    diagnostics.console("Warmup Analysis: Mean=" + warmupMean + " QPS, CV=" + warmupCV);

                    if (warmupCV > 15.0) {
                        diagnostics.console(" ⚠️  High warmup variance - consider more warmup runs\n");
                    } else {
                        diagnostics.console(" ✓ Warmup appears stable\n");
                    }

                    analysisTask.complete();
                }
                mainTask.phaseCompleted();
            }

            double[] qpsSamples = new double[numTestRuns];

            for (int testRun = 0; testRun < numTestRuns; testRun++) {
                // Use the pre-created task for this test run
                String testName = String.format("Test Run %d/%d", testRun + 1, numTestRuns);
                BenchmarkPhaseTask testTask = phaseTaskMap.get(testName);
                StatusTracker<BenchmarkPhaseTask> testTracker = phaseTrackerMap.get(testName);

                if (testTask != null && testTracker != null) {
                    testTask.start();
                    testTask.updateStatus("Performing GC");
                    testTask.updateProgress(0.1);

                    // Clear Eden and let GC complete with diagnostics monitoring
                    diagnostics.monitorPhase("GC-" + testRun, () -> {
                        System.gc();
                        System.runFinalization();
                        try {
                            Thread.sleep(500);
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                        }
                        return null;
                    });

                    testTask.updateStatus(String.format("Processing %d queries", totalQueries));
                    testTask.updateProgress(0.2);

                    String testPhase = "Test-" + testRun;

                    // Test Phase with detailed monitoring
                    qpsSamples[testRun] = diagnostics.monitorPhaseWithQueryTiming(testPhase, (recorder) -> {
                LongAdder visitedAdder = new LongAdder();
                long startTime = System.nanoTime();
                
                IntStream.range(0, totalQueries)
                        .parallel()
                        .forEach(i -> {
                            long queryStart = System.nanoTime();
                            
                            SearchResult sr = QueryExecutor.executeQuery(
                                    cs, topK, rerankK, usePruning, i);
                            // "Use" the result to prevent optimization
                            visitedAdder.add(sr.getVisitedCount());
                            
                            long queryEnd = System.nanoTime();
                            recorder.recordTime(queryEnd - queryStart);
                        });
                        
                    double elapsedSec = (System.nanoTime() - startTime) / 1e9;
                    return totalQueries / elapsedSec;
                });

                    testTask.updateProgress(1.0);
                    testTask.complete();
                }

                mainTask.phaseCompleted();
                diagnostics.console("Test Run " + testRun + ": " + qpsSamples[testRun] + " QPS\n");
            }

            // Performance variance analysis phase
            String perfAnalysisName = "Performance Analysis";
            BenchmarkPhaseTask perfAnalysisTask = phaseTaskMap.get(perfAnalysisName);
            StatusTracker<BenchmarkPhaseTask> perfTracker = phaseTrackerMap.get(perfAnalysisName);

            if (perfAnalysisTask != null && perfTracker != null) {
                perfAnalysisTask.start();
                perfAnalysisTask.updateStatus("Computing statistics");

                Arrays.sort(qpsSamples);
                medianQps = qpsSamples[numTestRuns/2];
                avgQps = StatUtils.mean(qpsSamples);
                stdDevQps = Math.sqrt(StatUtils.variance(qpsSamples));
                maxQps = StatUtils.max(qpsSamples);
                minQps = StatUtils.min(qpsSamples);
                coefficientOfVariation = (stdDevQps / avgQps) * 100;

                diagnostics.console("QPS Variance Analysis: CV=" + coefficientOfVariation + ", Range=[" + minQps + " - " + maxQps + "]\n");

                if (coefficientOfVariation > 10.0) {
                    diagnostics.console("⚠️  High performance variance detected (CV > 10%%)%n");
                }

                perfAnalysisTask.complete();
            }
            mainTask.phaseCompleted();

            // Compare test runs for performance regression detection
            if (numTestRuns > 1) {
                String regressionName = "Regression Analysis";
                BenchmarkPhaseTask regressionTask = phaseTaskMap.get(regressionName);
                StatusTracker<BenchmarkPhaseTask> regressionTracker = phaseTrackerMap.get(regressionName);

                if (regressionTask != null && regressionTracker != null) {
                    regressionTask.start();
                    regressionTask.updateStatus("Comparing test runs");
                    diagnostics.comparePhases("Test-0", "Test-" + (numTestRuns - 1));
                    regressionTask.complete();
                }
                mainTask.phaseCompleted();
            }

            // Generate final diagnostics summary and recommendations
            String finalReportName = "Final Report";
            BenchmarkPhaseTask finalTask = phaseTaskMap.get(finalReportName);
            StatusTracker<BenchmarkPhaseTask> finalTracker = phaseTrackerMap.get(finalReportName);

            if (finalTask != null && finalTracker != null) {
                finalTask.start();
                finalTask.updateStatus("Generating diagnostics summary");
                diagnostics.logSummary();
                diagnostics.provideRecommendations();
                finalTask.complete();
            }
            mainTask.phaseCompleted();

            // Mark benchmark as complete
            mainTask.complete();

            // Close all phase trackers
            for (StatusTracker<BenchmarkPhaseTask> tracker : phaseTrackers) {
                try {
                    tracker.close();
                } catch (Exception e) {
                    // Ignore errors during cleanup
                }
            }
            }  // End of mainTracker try-with-resources

            var list = new ArrayList<Metric>();
            if (computeAvgQps) {
                list.add(Metric.of("Avg QPS (of " + numTestRuns + ")", formatAvgQps, avgQps));
                list.add(Metric.of("± Std Dev", formatAvgQps, stdDevQps));
                list.add(Metric.of("CV %", ".1f", coefficientOfVariation));
            }
            if (computeMedianQps) {
                list.add(Metric.of("Median QPS (of " + numTestRuns + ")", formatMedianQps, medianQps));
            }
            if (computeMaxQps) {
                list.add(Metric.of("Max QPS (of " + numTestRuns + ")", formatMaxQps, maxQps));
                list.add(Metric.of("Min QPS (of " + numTestRuns + ")", formatMaxQps, minQps));
            }
            return list;
        } finally {
            // Clean up the benchmarkScope only if we own it
            if (ownScope) {
                benchmarkScope.close();
            }
        }
    }
}
