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

import java.util.List;
import java.util.ArrayList;
import java.util.Collections;

import io.github.jbellis.jvector.example.Grid.ConfiguredSystem;
import io.github.jbellis.jvector.graph.SearchResult;

/**
 * Measures per‐query latency (mean and variance) over N runs,
 * and counts correct top‐K results.
 */
public class LatencyBenchmark
        implements QueryBenchmark<LatencyBenchmark.Summary> {

    private static volatile long SINK;

    /**
     * Holds the number of correct results, the average latency (ns),
     * and the latency variance.
     */
    public static class Summary implements BenchmarkSummary {
        private final double averageLatency;
        private final double latencyVariance;
        private final double p999Latency;

        public Summary(double averageLatency, double latencyVariance, double p999Latency) {
            this.averageLatency   = averageLatency;
            this.latencyVariance  = latencyVariance;
            this.p999Latency      = p999Latency;
        }

        @Override
        public String toString() {
            return String.format(
                    "LatencySummary{latency (AVG) = %.3fms, (VAR) = %.6fms^2, p999 = %.2fms",
                    averageLatency, latencyVariance, p999Latency
            );
        }

        public double getAverageLatency() {
            return averageLatency;
        }

        public double getLatencyVariance() {
            return latencyVariance;
        }

        public double getP999Latency() {
            return p999Latency;
        }
    }

    @Override
    public String getBenchmarkName() {
        return "LatencyBenchmark";
    }

    @Override
    public Summary runBenchmark(
            ConfiguredSystem cs,
            int topK,
            int rerankK,
            boolean usePruning,
            int queryRuns) {

        int totalQueries = cs.getDataSet().queryVectors.size();
        double mean = 0.0;
        double m2 = 0.0;
        int count = 0;

        // Collect every query latency
        List<Long> latencies = new ArrayList<>(totalQueries * queryRuns);

        // Run the full set of queries queryRuns times
        for (int run = 0; run < queryRuns; run++) {
            for (int i = 0; i < totalQueries; i++) {
                long start = System.nanoTime();
                SearchResult sr = QueryExecutor.executeQuery(
                        cs, topK, rerankK, usePruning, i);
                long duration = System.nanoTime() - start;
                // record latency for percentile computation
                latencies.add(duration);
                SINK += sr.getVisitedCount();

                // Welford’s online algorithm for variance
                count++;
                double delta = duration - mean;
                mean += delta / count;
                m2 += delta * (duration - mean);
            }
        }

        double variance = (count > 0) ? (m2 / count) / 1e12: 0.0;

        // Compute 99.9th percentile
        Collections.sort(latencies);
        int idx = (int)Math.ceil(0.999 * latencies.size()) - 1;
        if (idx < 0) idx = 0;
        if (idx >= latencies.size()) idx = latencies.size() - 1;
        long p999Latency = latencies.get(idx);

        return new Summary(mean / 1e6, variance, p999Latency / 1e6);
    }
}



