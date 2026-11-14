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

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.function.IntFunction;
import java.util.function.Consumer;
import java.util.function.Supplier;
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
    private DiagnosticLevel diagnosticLevel;
    private WarmupQueries warmupQueries = WarmupQueries.RANDOM;
    private boolean warmupUntilCompiled;
    private double minSeconds = 1.0;

    /** A warmup pass that adds at most this much JIT compilation counts as compiled. */
    private static final long COMPILED_JIT_MS = 50;
    /** Warmup passes run at most this many times while waiting for compilation to settle. */
    private static final int MAX_WARMUP_PASSES = 10;

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
        this.diagnosticLevel = diagnosticLevel;
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
     * Configures warmup and timed passes.
     *
     * @param warmupQueries  the queries to warm up with; base vectors are sampled with a fixed seed per pass
     * @param untilCompiled  keep warming past the configured passes, up to 10, until a pass adds at most 50 ms of
     *                       JIT compilation
     * @param minSeconds     minimum seconds per timed pass; a pass repeats the query set until it has run this
     *                       long (default 1.0; 0 runs the query set once)
     */
    public ThroughputBenchmark withMeasurement(WarmupQueries warmupQueries, boolean untilCompiled, double minSeconds) {
        if (!Double.isFinite(minSeconds) || minSeconds < 0)
            throw new IllegalArgumentException("Minimum seconds per pass must be finite and nonnegative: " + minSeconds);
        if (warmupQueries == WarmupQueries.NONE && untilCompiled)
            throw new IllegalArgumentException("Warming until compiled requires warmup queries");
        this.warmupQueries = Objects.requireNonNull(warmupQueries, "warmupQueries");
        this.warmupUntilCompiled = untilCompiled;
        this.minSeconds = minSeconds;
        return this;
    }

    /**
     * Configure the diagnostic level for this benchmark
     */
    public ThroughputBenchmark withDiagnostics(DiagnosticLevel level) {
        this.diagnosticLevel = level;
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

        return runWithReporting(cs, topK, rerankK, usePruning, queryRuns, null);
    }

    @Override
    public List<Metric> runBenchmark(
            ConfiguredSystem cs, int topK, int rerankK, boolean usePruning,
            int queryRuns, Consumer<Supplier<List<Metric>>> deferredReports) {
        return runWithReporting(cs, topK, rerankK, usePruning, queryRuns,
                Objects.requireNonNull(deferredReports, "deferredReports"));
    }

    private List<Metric> runWithReporting(
            ConfiguredSystem cs, int topK, int rerankK, boolean usePruning,
            int queryRuns, Consumer<Supplier<List<Metric>>> deferredReports) {
        if (cs.usesQueuedExecution()) return runConfiguredBenchmark(cs, topK, rerankK, usePruning, queryRuns, deferredReports);
        int threads = cs.directCallerThreads();
        if (threads > 1) {
            var pool = new java.util.concurrent.ForkJoinPool(threads);
            try {
                return pool.submit(() -> runConfiguredBenchmark(cs, topK, rerankK, usePruning, queryRuns, deferredReports)).join();
            } finally {
                pool.shutdown();
            }
        }
        return runConfiguredBenchmark(cs, topK, rerankK, usePruning, queryRuns, deferredReports);
    }

    /**
     * {@code count} base vectors drawn uniformly from the whole dataset, with a fixed seed for each warmup pass, so
     * every configuration warms the same index pages.
     */
    private static VectorFloat<?>[] baseSample(ConfiguredSystem cs, int count, int pass) {
        var base = cs.getDataSet().getBaseVectors();
        var random = new SplittableRandom(0x5EED_0000L + pass);
        var sample = new VectorFloat<?>[count];
        for (int i = 0; i < count; i++) sample[i] = base.get(random.nextInt(base.size())).copy();
        return sample;
    }

    /** Major page faults of this process so far, each a page read from storage; -1 where Linux /proc is absent. */
    private static long majorFaults() {
        try {
            String stat = Files.readString(Path.of("/proc/self/stat"));
            // Fields after the command name, which may contain spaces: majflt is the tenth.
            return Long.parseLong(stat.substring(stat.lastIndexOf(')') + 2).split(" ")[9]);
        } catch (IOException | RuntimeException unavailable) {
            return -1;
        }
    }

    private VectorFloat<?> randomQuery(int dim) {
        VectorFloat<?> query = vts.createFloatVector(dim);
        for (int j = 0; j < dim; j++) query.set(j, ThreadLocalRandom.current().nextFloat());
        VectorUtil.l2normalize(query);
        return query;
    }

    private static IntStream queryStream(ConfiguredSystem cs, int count) {
        IntStream stream = IntStream.range(0, count);
        return cs.directCallerThreads() == 1 ? stream : stream.parallel();
    }

    private List<Metric> runConfiguredBenchmark(ConfiguredSystem cs, int topK, int rerankK,
                                                boolean usePruning, int queryRuns,
                                                Consumer<Supplier<List<Metric>>> deferredReports) {

        if (!(computeAvgQps || computeMedianQps || computeMaxQps)) {
            throw new RuntimeException("At least one metric must be displayed");
        }
        try (BenchmarkDiagnostics diagnostics = new BenchmarkDiagnostics(diagnosticLevel);
             QueuedQueryExecution queued = cs.usesQueuedExecution()
                     ? new QueuedQueryExecution(cs, topK, rerankK, usePruning) : null) {
        if (Boolean.getBoolean("jvector.bench.verifyConcurrentResults")) {
            int count = cs.getDataSet().getQueryVectors().size();
            if (queued != null) {
                queued.verifyConcurrentResults(count, cs.getDataSet().getQueryVectors()::get);
            } else {
                // Direct callers may naturally combine projections too. Keep
                // the prepared provider fixed while checking per-thread search
                // reuse; comparing GEMV with GEMM would test rounding instead.
                queryStream(cs, count).forEach(i ->
                        QueryExecutor.verifyConcurrentQuery(cs, topK, rerankK, usePruning, i));
            }
            if (diagnosticLevel != DiagnosticLevel.NONE) {
                diagnostics.console(String.format(java.util.Locale.ROOT,
                        "[Validation] Passed concurrent-result checks for %,d queries (topK=%d, rerankK=%d)",
                        count, topK, rerankK));
            }
        }


            int totalQueries = cs.getDataSet().getQueryVectors().size();
            int dim = cs.getDataSet().getDimension();

            // Warmup, optionally continuing until JIT compilation settles.
            var compiler = ManagementFactory.getCompilationMXBean();
            boolean jitTimed = compiler != null && compiler.isCompilationTimeMonitoringSupported();
            if (warmupUntilCompiled && !jitTimed)
                throw new IllegalStateException("untilCompiled requires JIT compilation time, which this JVM does not report");
            int warmupPasses = warmupQueries == WarmupQueries.NONE ? 0 : numWarmupRuns;
            int maxWarmupPasses = warmupUntilCompiled ? Math.max(warmupPasses, MAX_WARMUP_PASSES) : warmupPasses;
            var warmupQpsList = new ArrayList<Double>();
            long lastPassJitMs = Long.MAX_VALUE;
            for (int warmupRun = 0; warmupRun < maxWarmupPasses; warmupRun++) {
                if (warmupRun >= warmupPasses && lastPassJitMs <= COMPILED_JIT_MS) break;
                String warmupPhase = "Warmup-" + warmupRun;
                VectorFloat<?>[] baseQueries = warmupQueries == WarmupQueries.BASE ? baseSample(cs, totalQueries, warmupRun) : null;
                IntFunction<VectorFloat<?>> warmupQuery =
                        baseQueries != null ? k -> baseQueries[k] : k -> randomQuery(dim);
                long jitBefore = jitTimed ? compiler.getTotalCompilationTime() : 0;
                long passStart = System.nanoTime();

                diagnostics.monitorPhaseWithQueryTiming(warmupPhase, (recorder) -> {
                    if (queued != null) {
                        SINK += queued.runPass(totalQueries, warmupQuery, recorder, false).visited();
                    } else queryStream(cs, totalQueries)
                            .forEach(k -> {
                                long queryStart = System.nanoTime();
                                SearchResult sr = QueryExecutor.executeQuery(
                                        cs, topK, rerankK, usePruning, warmupQuery.apply(k));
                                SINK += sr.getVisitedCount();

                                long queryEnd = System.nanoTime();
                                recorder.recordTime(queryEnd - queryStart);
                            });

                    return totalQueries / 1.0; // Return QPS placeholder
                });

                lastPassJitMs = jitTimed ? compiler.getTotalCompilationTime() - jitBefore : 0;
                double warmupPassQps = totalQueries / ((System.nanoTime() - passStart) / 1e9);
                warmupQpsList.add(warmupPassQps);
                diagnostics.console("Warmup Run " + warmupRun + ": " + warmupPassQps + " QPS\n");
            }
            if (warmupUntilCompiled && lastPassJitMs > COMPILED_JIT_MS) {
                System.out.printf("WARNING: JIT compilation had not settled after %d warmup passes (%d ms in the last)%n",
                        maxWarmupPasses, lastPassJitMs);
            }
            double[] warmupQps = warmupQpsList.stream().mapToDouble(Double::doubleValue).toArray();

            // Analyze warmup effectiveness
            if (warmupQps.length > 1) {
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

            long minSampleNanos = (long) (minSeconds * 1e9);
            double[] qpsSamples = new double[numTestRuns];
            if (queued != null) queued.beginMeasurements();
            // Direct searches record their own times; queued execution records response times itself.
            var directTimes = queued == null ? new ConcurrentLinkedQueue<Long>() : null;
            var timedQueries = new LongAdder();
            long faultsBefore = majorFaults();
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
                qpsSamples[testRun] = diagnostics.monitorPhaseWithQueryTiming(testPhase, (passRecorder) -> {
                    BenchmarkDiagnostics.QueryTimeRecorder recorder = directTimes == null ? passRecorder
                            : nanos -> { passRecorder.recordTime(nanos); directTimes.add(nanos); };
                    LongAdder visitedAdder = new LongAdder();
                    long startTime = System.nanoTime();

                    long completedQueries = 0;
                    do {
                        if (queued != null) {
                            visitedAdder.add(queued.runPass(totalQueries,
                                    cs.getDataSet().getQueryVectors()::get, recorder, false).visited());
                        } else queryStream(cs, totalQueries)
                                .forEach(i -> {
                                    long queryStart = System.nanoTime();

                                    SearchResult sr = QueryExecutor.executeQuery(
                                            cs, topK, rerankK, usePruning, i);
                                    // "Use" the result to prevent optimization
                                    visitedAdder.add(sr.getVisitedCount());

                                    long queryEnd = System.nanoTime();
                                    recorder.recordTime(queryEnd - queryStart);
                                });

                        completedQueries += totalQueries;
                    } while (System.nanoTime() - startTime < minSampleNanos);
                    SINK += visitedAdder.sum();
                    timedQueries.add(completedQueries);
                    double elapsedSec = (System.nanoTime() - startTime) / 1e9;
                    if (diagnosticLevel != DiagnosticLevel.NONE) {
                        diagnostics.console(String.format(java.util.Locale.ROOT,
                                "[%s] Completed %,d queries in %.3f s (%,.1f QPS)",
                                testPhase, completedQueries, elapsedSec, completedQueries / elapsedSec));
                    }
                    return completedQueries / elapsedSec;
                });
            }
            long pageReads = faultsBefore < 0 ? -1 : majorFaults() - faultsBefore;

            // Performance variance analysis
            Arrays.sort(qpsSamples);
            double medianQps = qpsSamples[numTestRuns / 2];
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
                list.add(Metric.of("search.throughput.avg_qps",
                        "Avg QPS\n (of " + numTestRuns + ")",
                        formatAvgQps,
                        avgQps));

                // Spread across passes is meaningful only with more than one pass.
                if (numTestRuns > 1) {
                    list.add(Metric.of("search.throughput.stddev_qps",
                            "± Std Dev",
                            formatAvgQps,
                            stdDevQps));

                    list.add(Metric.of("search.throughput.cv_pct",
                            "CV %",
                            ".1f",
                            coefficientOfVariation));
                }
            }
            if (computeMedianQps) {
                list.add(Metric.of("search.throughput.median_qps",
                        "Median QPS\n (of " + numTestRuns + ")",
                        formatMedianQps,
                        medianQps));
            }
            if (computeMaxQps) {
                list.add(Metric.of("search.throughput.max_qps",
                        "Max QPS\n (of " + numTestRuns + ")",
                        formatMaxQps,
                        maxQps));

                list.add(Metric.of("search.throughput.min_qps",
                        "Min QPS\n (of " + numTestRuns + ")",
                        formatMaxQps,
                        minQps));
            }
            // Execution metrics describe the timed passes; sorting waits until they are reported.
            var samples = queued != null ? queued.finishMeasurements() : null;
            long[] times = directTimes == null ? null : directTimes.stream().mapToLong(Long::longValue).toArray();
            long queriesTimed = timedQueries.sum();
            double pageReadsPerQuery = pageReads < 0 || queriesTimed == 0 ? Double.NaN : (double) pageReads / queriesTimed;
            Supplier<List<Metric>> execution = () -> {
                var metrics = new ArrayList<>(samples != null ? samples.metrics() : ResponseTimeMetrics.of(times));
                if (!Double.isNaN(pageReadsPerQuery)) {
                    metrics.add(Metric.of("search.execution.page_reads_per_query", "Page reads (per query)", ".1f",
                            pageReadsPerQuery));
                }
                return metrics;
            };
            if (deferredReports == null) list.addAll(execution.get());
            else deferredReports.accept(execution);
            return list;
        }
        catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
