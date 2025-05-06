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
 * Measures per‐query latency (mean and standard deviation) over N runs,
 * and counts correct top‐K results.
 */
public class LatencyBenchmark extends AbstractQueryBenchmark {
    private static final String DEFAULT_FORMAT = ".3f";

    private boolean computeAvgLatency;
    private boolean computeLatencySTD;
    private boolean computeP999Latency;
    private String formatAvgLatency;
    private String formatLatencySTD;
    private String formatP999Latency;

    private static volatile long SINK;

    public static LatencyBenchmark createDefault() {
        return new LatencyBenchmark(true, false, false, DEFAULT_FORMAT, DEFAULT_FORMAT, DEFAULT_FORMAT);
    }

    public static LatencyBenchmark createEmpty() {
        return new LatencyBenchmark(false, false, false, DEFAULT_FORMAT, DEFAULT_FORMAT, DEFAULT_FORMAT);
    }

    private LatencyBenchmark(boolean computeAvgLatency, boolean computeLatencySTD, boolean computeP999Latency,
                            String formatAvgLatency, String formatLatencySTD, String formatP999Latency) {
        this.computeAvgLatency = computeAvgLatency;
        this.computeLatencySTD = computeLatencySTD;
        this.computeP999Latency = computeP999Latency;
        this.formatAvgLatency = formatAvgLatency;
        this.formatLatencySTD = formatLatencySTD;
        this.formatP999Latency = formatP999Latency;
    }

    public LatencyBenchmark displayAvgLatency() {
        return displayAvgLatency(DEFAULT_FORMAT);
    }

    public LatencyBenchmark displayAvgLatency(String format) {
        this.computeAvgLatency = true;
        this.formatAvgLatency = format;
        return this;
    }

    public LatencyBenchmark displayLatencySTD() {
        return displayLatencySTD(DEFAULT_FORMAT);
    }

    public LatencyBenchmark displayLatencySTD(String format) {
        this.computeLatencySTD = true;
        this.formatLatencySTD = format;
        return this;
    }

    public LatencyBenchmark displayP999Latency() {
        return displayP999Latency(DEFAULT_FORMAT);
    }

    public LatencyBenchmark displayP999Latency(String format) {
        this.computeP999Latency = true;
        this.formatP999Latency = format;
        return this;
    }

    @Override
    public String getBenchmarkName() {
        return "LatencyBenchmark";
    }

    @Override
    public List<Metric> runBenchmark(
            ConfiguredSystem cs,
            int topK,
            int rerankK,
            boolean usePruning,
            int queryRuns) {

        if (!(computeAvgLatency || computeLatencySTD || computeP999Latency)) {
            throw new IllegalArgumentException("At least one parameter must be set to true");
        }

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

        mean /= 1e6;
        double standardDeviation = (count > 0) ? Math.sqrt(m2 / count) / 1e6: 0.0;

        // Compute 99.9th percentile
        Collections.sort(latencies);
        int idx = (int)Math.ceil(0.999 * latencies.size()) - 1;
        if (idx < 0) idx = 0;
        if (idx >= latencies.size()) idx = latencies.size() - 1;
        double p999Latency = latencies.get(idx) / 1e6;

        var list = new ArrayList<Metric>();
        if (computeAvgLatency) {
            list.add(Metric.of("Mean Latency (ms)", formatAvgLatency, mean));
        }
        if (computeLatencySTD) {
            list.add(Metric.of("STD Latency (ms)", formatLatencySTD, standardDeviation));
        }
        if (computeP999Latency) {
            list.add(Metric.of("p999 Latency (ms)", formatP999Latency, p999Latency));
        }
        return list;
    }
}



