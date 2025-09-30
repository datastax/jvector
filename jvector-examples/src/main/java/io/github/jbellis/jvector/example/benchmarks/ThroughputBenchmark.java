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
import io.github.jbellis.jvector.status.sinks.ConsoleLoggerSink;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.example.benchmarks.diagnostics.BenchmarkDiagnostics;
import io.github.jbellis.jvector.example.benchmarks.diagnostics.DiagnosticLevel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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

        if (!(computeAvgQps || computeMedianQps || computeMaxQps)) {
            throw new RuntimeException("At least one metric must be displayed");
        }

        int totalQueries = cs.getDataSet().queryVectors.size();
        int dim = cs.getDataSet().getDimension();

        // Create a status tracking task for the benchmark process
        class BenchmarkTask implements StatusUpdate.Provider<BenchmarkTask> {
            private volatile double progress = 0.0;
            private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;
            private volatile String currentPhase = "Initializing";
            private volatile String detailedStatus = "";
            private volatile int totalSteps;
            private volatile int completedSteps = 0;
            private volatile int queriesProcessed = 0;

            BenchmarkTask(int warmupRuns, int testRuns) {
                // Calculate total steps: warmup (prep + exec per run) + warmup analysis (if > 1) +
                // test (GC + prep + exec per run) + performance analysis + regression analysis (if > 1) + final analysis
                this.totalSteps = (warmupRuns * 2) + (warmupRuns > 1 ? 1 : 0) +
                                 (testRuns * 3) + 1 + (testRuns > 1 ? 1 : 0) + 1;
            }

            @Override
            public StatusUpdate<BenchmarkTask> getTaskStatus() {
                return new StatusUpdate<>(progress, state, this);
            }

            void updatePhase(String phase, String details) {
                this.currentPhase = phase;
                this.detailedStatus = details;
                if (state == StatusUpdate.RunState.PENDING && progress > 0) {
                    state = StatusUpdate.RunState.RUNNING;
                }
            }

            void incrementProgress() {
                this.completedSteps++;
                this.progress = Math.min(1.0, (double) completedSteps / totalSteps);
                if (state == StatusUpdate.RunState.PENDING && progress > 0) {
                    state = StatusUpdate.RunState.RUNNING;
                }
            }

            void setQueriesProcessed(int queries) {
                this.queriesProcessed = queries;
            }

            void complete() {
                this.progress = 1.0;
                this.state = StatusUpdate.RunState.SUCCESS;
            }

            @Override
            public String toString() {
                return String.format("ThroughputBenchmark [%s: %s, %.1f%% complete, %d queries]",
                        currentPhase, detailedStatus, progress * 100, queriesProcessed);
            }
        }

        BenchmarkTask benchmarkTask = new BenchmarkTask(numWarmupRuns, numTestRuns);
        ConsoleLoggerSink consoleSink = new ConsoleLoggerSink(System.out, true, false);

        try (StatusTracker<BenchmarkTask> tracker = StatusTracker.withInstrumented(benchmarkTask,
                java.time.Duration.ofMillis(500),
                consoleSink)) {

            benchmarkTask.state = StatusUpdate.RunState.RUNNING;

            // Warmup Phase with diagnostics
            double[] warmupQps = new double[numWarmupRuns];
            for (int warmupRun = 0; warmupRun < numWarmupRuns; warmupRun++) {
                // Preparation phase for warmup
                benchmarkTask.updatePhase("Warmup Preparation", String.format("Run %d/%d - Initializing", warmupRun + 1, numWarmupRuns));
                benchmarkTask.incrementProgress();
                try { Thread.sleep(10); } catch (InterruptedException e) {} // Small delay to ensure status is visible

                // Execution phase for warmup
                benchmarkTask.updatePhase("Warmup Execution", String.format("Run %d/%d - Processing %d queries", warmupRun + 1, numWarmupRuns, totalQueries));
                benchmarkTask.setQueriesProcessed(totalQueries);
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

                benchmarkTask.incrementProgress();
                benchmarkTask.setQueriesProcessed(0);
                diagnostics.console("Warmup Run " + warmupRun + ": " + warmupQps[warmupRun] + " QPS\n");
            }

            // Analyze warmup effectiveness
            if (numWarmupRuns > 1) {
                benchmarkTask.updatePhase("Warmup Analysis", "Analyzing warmup stability");
                try { Thread.sleep(10); } catch (InterruptedException e) {}

                double warmupVariance = StatUtils.variance(warmupQps);
                double warmupMean = StatUtils.mean(warmupQps);
                double warmupCV = Math.sqrt(warmupVariance) / warmupMean * 100;
                diagnostics.console("Warmup Analysis: Mean=" + warmupMean + " QPS, CV=" + warmupCV);

                if (warmupCV > 15.0) {
                    diagnostics.console(" ⚠️  High warmup variance - consider more warmup runs\n");
                } else {
                    diagnostics.console(" ✓ Warmup appears stable\n");
                }
                benchmarkTask.incrementProgress();
            }

            double[] qpsSamples = new double[numTestRuns];
            for (int testRun = 0; testRun < numTestRuns; testRun++) {
                // GC phase before test run
                benchmarkTask.updatePhase("Garbage Collection", String.format("Before test run %d/%d", testRun + 1, numTestRuns));
                benchmarkTask.incrementProgress();

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

                // Test preparation phase
                benchmarkTask.updatePhase("Test Preparation", String.format("Run %d/%d - Initializing", testRun + 1, numTestRuns));
                benchmarkTask.incrementProgress();
                try { Thread.sleep(10); } catch (InterruptedException e) {}

                // Test execution phase
                benchmarkTask.updatePhase("Test Execution", String.format("Run %d/%d - Processing %d queries", testRun + 1, numTestRuns, totalQueries));
                benchmarkTask.setQueriesProcessed(totalQueries);
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

                benchmarkTask.incrementProgress();
                benchmarkTask.setQueriesProcessed(0);
                diagnostics.console("Test Run " + testRun + ": " + qpsSamples[testRun] + " QPS\n");
            }

            // Performance variance analysis phase
            benchmarkTask.updatePhase("Performance Analysis", "Computing statistics");
            try { Thread.sleep(10); } catch (InterruptedException e) {}

            Arrays.sort(qpsSamples);
            double medianQps = qpsSamples[numTestRuns/2];
            double avgQps = StatUtils.mean(qpsSamples);
            double stdDevQps = Math.sqrt(StatUtils.variance(qpsSamples));
            double maxQps = StatUtils.max(qpsSamples);
            double minQps = StatUtils.min(qpsSamples);
            double coefficientOfVariation = (stdDevQps / avgQps) * 100;

            diagnostics.console("QPS Variance Analysis: CV=" + coefficientOfVariation + ", Range=[" + minQps + " - " + maxQps + "]\n");

            if (coefficientOfVariation > 10.0) {
                diagnostics.console("⚠️  High performance variance detected (CV > 10%%)%n");
            }

            // Compare test runs for performance regression detection
            if (numTestRuns > 1) {
                benchmarkTask.updatePhase("Regression Analysis", "Comparing test runs");
                diagnostics.comparePhases("Test-0", "Test-" + (numTestRuns - 1));
            }

            benchmarkTask.incrementProgress();

            // Generate final diagnostics summary and recommendations
            benchmarkTask.updatePhase("Final Analysis", "Generating diagnostics summary");
            diagnostics.logSummary();
            diagnostics.provideRecommendations();

            benchmarkTask.incrementProgress();

            // Mark benchmark as complete
            benchmarkTask.complete();

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
        }
    }
}
