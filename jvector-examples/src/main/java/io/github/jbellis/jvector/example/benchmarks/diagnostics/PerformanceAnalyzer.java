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

import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Analyzes performance timing data to detect anomalies and provide insights
 * into benchmark performance variations.
 */
public class PerformanceAnalyzer {

    private final Queue<Long> queryTimes = new ConcurrentLinkedQueue<>();
    private final AtomicLong totalQueries = new AtomicLong(0);
    private final AtomicLong totalTime = new AtomicLong(0);

    /**
     * Constructs a new PerformanceAnalyzer with empty timing data.
     */
    public PerformanceAnalyzer() {
    }

    /**
     * Records the execution time of a single query.
     *
     * @param nanoTime the query execution time in nanoseconds
     */
    public void recordQueryTime(long nanoTime) {
        queryTimes.offer(nanoTime);
        totalQueries.incrementAndGet();
        totalTime.addAndGet(nanoTime);
    }

    /**
     * Analyzes collected timing data and returns performance statistics.
     *
     * @param phase the name of the phase being analyzed
     * @return the timing analysis results
     */
    public TimingAnalysis analyzeTimings(String phase) {
        List<Long> times = new ArrayList<>(queryTimes);
        if (times.isEmpty()) {
            return new TimingAnalysis(phase, 0, 0, 0, 0, 0, 0, Collections.emptyList());
        }

        Collections.sort(times);

        long min = times.get(0);
        long max = times.get(times.size() - 1);
        long p50 = times.get(times.size() / 2);
        long p95 = times.get((int)(times.size() * 0.95));
        long p99 = times.get((int)(times.size() * 0.99));

        double mean = times.stream().mapToLong(Long::longValue).average().orElse(0.0);

        // Detect outliers (queries taking more than 3x the median)
        long outlierThreshold = p50 * 3;
        List<Long> outliers = times.stream()
            .filter(time -> time > outlierThreshold)
            .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);

        return new TimingAnalysis(phase, min, max, p50, p95, p99, (long)mean, outliers);
    }

    /**
     * Clears collected timing data
     */
    public void reset() {
        queryTimes.clear();
        totalQueries.set(0);
        totalTime.set(0);
    }

    /**
     * Compares performance between runs and identifies significant changes.
     *
     * @param baseline the baseline timing analysis
     * @param current the current timing analysis
     * @return the performance comparison results
     */
    public static PerformanceComparison compareRuns(TimingAnalysis baseline, TimingAnalysis current) {
        double p50Change = calculatePercentageChange(baseline.p50, current.p50);
        double p95Change = calculatePercentageChange(baseline.p95, current.p95);
        double p99Change = calculatePercentageChange(baseline.p99, current.p99);
        double meanChange = calculatePercentageChange(baseline.mean, current.mean);

        boolean significantRegression = Math.abs(p50Change) > 10.0 || Math.abs(p95Change) > 15.0;

        return new PerformanceComparison(
            baseline.phase, current.phase,
            p50Change, p95Change, p99Change, meanChange,
            significantRegression
        );
    }

    private static double calculatePercentageChange(long baseline, long current) {
        if (baseline == 0) return current == 0 ? 0.0 : 100.0;
        return ((double)(current - baseline) / baseline) * 100.0;
    }

    /**
     * Logs timing analysis results to standard output.
     *
     * @param analysis the timing analysis to log
     */
    public void logAnalysis(TimingAnalysis analysis) {
        System.out.printf("[%s] Query Timing Analysis:%n", analysis.phase);
        System.out.printf("  Min: %.2f ms, Max: %.2f ms%n",
            analysis.min / 1e6, analysis.max / 1e6);
        System.out.printf("  P50: %.2f ms, P95: %.2f ms, P99: %.2f ms%n",
            analysis.p50 / 1e6, analysis.p95 / 1e6, analysis.p99 / 1e6);
        System.out.printf("  Mean: %.2f ms%n", analysis.mean / 1e6);

        if (!analysis.outliers.isEmpty()) {
            System.out.printf("  Outliers: %d queries (%.1f%%) took >3x median time%n",
                analysis.outliers.size(),
                (analysis.outliers.size() * 100.0) / totalQueries.get());

            // Show worst outliers
            analysis.outliers.stream()
                .sorted(Collections.reverseOrder())
                .limit(5)
                .forEach(time -> System.out.printf("    %.2f ms%n", time / 1e6));
        }
    }

    /**
     * Logs performance comparison results to standard output.
     *
     * @param comparison the performance comparison to log
     */
    public static void logComparison(PerformanceComparison comparison) {
        System.out.printf("[%s vs %s] Performance Comparison:%n",
            comparison.baselinePhase, comparison.currentPhase);
        System.out.printf("  P50 change: %+.1f%%", comparison.p50Change);
        System.out.printf("  P95 change: %+.1f%%", comparison.p95Change);
        System.out.printf("  P99 change: %+.1f%%", comparison.p99Change);
        System.out.printf("  Mean change: %+.1f%%", comparison.meanChange);

        if (comparison.significantRegression) {
            System.out.printf("  ⚠️  SIGNIFICANT PERFORMANCE CHANGE DETECTED%n");
        }
    }

    /**
     * Contains timing analysis statistics for a benchmark phase.
     */
    public static class TimingAnalysis {
        /** The name of the phase being analyzed. */
        public final String phase;
        /** The minimum query time in nanoseconds. */
        public final long min;
        /** The maximum query time in nanoseconds. */
        public final long max;
        /** The 50th percentile (median) query time in nanoseconds. */
        public final long p50;
        /** The 95th percentile query time in nanoseconds. */
        public final long p95;
        /** The 99th percentile query time in nanoseconds. */
        public final long p99;
        /** The mean query time in nanoseconds. */
        public final long mean;
        /** List of outlier query times exceeding 3x the median. */
        public final List<Long> outliers;

        /**
         * Constructs a TimingAnalysis with the specified statistics.
         *
         * @param phase the name of the phase being analyzed
         * @param min the minimum query time in nanoseconds
         * @param max the maximum query time in nanoseconds
         * @param p50 the 50th percentile query time in nanoseconds
         * @param p95 the 95th percentile query time in nanoseconds
         * @param p99 the 99th percentile query time in nanoseconds
         * @param mean the mean query time in nanoseconds
         * @param outliers list of outlier query times
         */
        public TimingAnalysis(String phase, long min, long max, long p50, long p95, long p99,
                            long mean, List<Long> outliers) {
            this.phase = phase;
            this.min = min;
            this.max = max;
            this.p50 = p50;
            this.p95 = p95;
            this.p99 = p99;
            this.mean = mean;
            this.outliers = outliers;
        }
    }

    /**
     * Contains performance comparison results between two benchmark runs.
     */
    public static class PerformanceComparison {
        /** The name of the baseline phase. */
        public final String baselinePhase;
        /** The name of the current phase. */
        public final String currentPhase;
        /** The percentage change in 50th percentile time. */
        public final double p50Change;
        /** The percentage change in 95th percentile time. */
        public final double p95Change;
        /** The percentage change in 99th percentile time. */
        public final double p99Change;
        /** The percentage change in mean time. */
        public final double meanChange;
        /** Whether a significant performance regression was detected. */
        public final boolean significantRegression;

        /**
         * Constructs a PerformanceComparison with the specified metrics.
         *
         * @param baselinePhase the name of the baseline phase
         * @param currentPhase the name of the current phase
         * @param p50Change the percentage change in 50th percentile time
         * @param p95Change the percentage change in 95th percentile time
         * @param p99Change the percentage change in 99th percentile time
         * @param meanChange the percentage change in mean time
         * @param significantRegression whether a significant regression was detected
         */
        public PerformanceComparison(String baselinePhase, String currentPhase,
                                   double p50Change, double p95Change, double p99Change,
                                   double meanChange, boolean significantRegression) {
            this.baselinePhase = baselinePhase;
            this.currentPhase = currentPhase;
            this.p50Change = p50Change;
            this.p95Change = p95Change;
            this.p99Change = p99Change;
            this.meanChange = meanChange;
            this.significantRegression = significantRegression;
        }
    }
}
