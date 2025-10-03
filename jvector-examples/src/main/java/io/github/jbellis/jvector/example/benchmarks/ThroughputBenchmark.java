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

        // Warmup Phase with diagnostics
        double[] warmupQps = new double[numWarmupRuns];
        for (int warmupRun = 0; warmupRun < numWarmupRuns; warmupRun++) {
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
            
            diagnostics.console("Warmup Run " + warmupRun + ": " + warmupQps[warmupRun] + " QPS\n");
        }

        // Analyze warmup effectiveness
        if (numWarmupRuns > 1) {
            double warmupVariance = StatUtils.variance(warmupQps);
            double warmupMean = StatUtils.mean(warmupQps);
            double warmupCV = Math.sqrt(warmupVariance) / warmupMean * 100;
            diagnostics.console("Warmup Analysis: Mean=" + warmupMean + " QPS, CV=" + warmupCV);
            
            if (warmupCV > 15.0) {
                diagnostics.console(" ⚠️  High warmup variance - consider more warmup runs\n");
            } else {
                diagnostics.console(" ✓ Warmup appears stable\n");
            }
        }

        double[] qpsSamples = new double[numTestRuns];
        for (int testRun = 0; testRun < numTestRuns; testRun++) {
            String testPhase = "Test-" + testRun;

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

            diagnostics.console("Test Run " + testRun + ": " + qpsSamples[testRun] + " QPS\n");
        }

        // Performance variance analysis
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
            diagnostics.comparePhases("Test-0", "Test-" + (numTestRuns - 1));
        }

        // Generate final diagnostics summary and recommendations
        diagnostics.logSummary();
        diagnostics.provideRecommendations();

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
