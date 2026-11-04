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

package io.github.jbellis.jvector.fastervector.bench;

import io.github.jbellis.jvector.example.benchmarks.Metric;
import io.github.jbellis.jvector.fastervector.IvfAshIndexSearcher;
import io.github.jbellis.jvector.fastervector.Vecs;
import io.github.jbellis.jvector.fastervector.bench.IvfConfig.SearchPoint;
import io.github.jbellis.jvector.fastervector.utils.FlatArrayVecs;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Per-query latency: one thread and one single-threaded searcher, one query per call, over the
 * query set {@code queryRuns} times. Mirrors the graph's {@code LatencyBenchmark}, including its
 * metric keys. {@code searchThreads} doesn't apply, since a single query runs on one thread.
 */
public class IvfLatencyBenchmark implements IvfQueryBenchmark {
    private static final String DEFAULT_FORMAT = ".3f";

    private final int queryRuns;
    private boolean computeAvgLatency;
    private boolean computeLatencySTD;
    private boolean computeP999Latency;

    private IvfLatencyBenchmark(int queryRuns) {
        if (queryRuns < 1) {
            throw new IllegalArgumentException("queryRuns must be at least 1");
        }
        this.queryRuns = queryRuns;
    }

    /** Mean latency over 2 passes, as the graph runner uses. */
    public static IvfLatencyBenchmark createDefault() {
        return createEmpty(2).displayAvgLatency();
    }

    public static IvfLatencyBenchmark createEmpty(int queryRuns) {
        return new IvfLatencyBenchmark(queryRuns);
    }

    public IvfLatencyBenchmark displayAvgLatency() {
        computeAvgLatency = true;
        return this;
    }

    public IvfLatencyBenchmark displayLatencySTD() {
        computeLatencySTD = true;
        return this;
    }

    public IvfLatencyBenchmark displayP999Latency() {
        computeP999Latency = true;
        return this;
    }

    @Override
    public String getBenchmarkName() {
        return "IvfLatencyBenchmark";
    }

    @Override
    public List<Metric> runBenchmark(IvfSystem system, SearchPoint point) {
        if (!(computeAvgLatency || computeLatencySTD || computeP999Latency)) {
            throw new IllegalStateException("At least one metric must be displayed");
        }
        int nq = system.queryCount();
        int dim = system.query(0).length;
        Vecs[] queries = new Vecs[nq];
        for (int i = 0; i < nq; i++) {
            queries[i] = new FlatArrayVecs(system.query(i), dim);
        }

        long[] latencies = new long[nq * queryRuns];
        try (IvfAshIndexSearcher searcher = system.searcher(point.topK(), point.nprobe(), null)) {
            int n = 0;
            for (int run = 0; run < queryRuns; run++) {
                for (Vecs query : queries) {
                    long start = System.nanoTime();
                    IvfThroughputBenchmark.consume(searcher.search(query));
                    latencies[n++] = System.nanoTime() - start;
                }
            }
        }
        return metrics(latencies);
    }

    List<Metric> metrics(long[] latenciesNanos) {
        double mean = 0;
        for (long l : latenciesNanos) {
            mean += l;
        }
        mean /= latenciesNanos.length;
        double m2 = 0;
        for (long l : latenciesNanos) {
            m2 += (l - mean) * (l - mean);
        }
        // Population standard deviation, as the graph benchmark's Welford computation gives
        double std = Math.sqrt(m2 / latenciesNanos.length);
        long[] sorted = latenciesNanos.clone();
        Arrays.sort(sorted);
        int idx = Math.min(Math.max((int) Math.ceil(0.999 * sorted.length) - 1, 0), sorted.length - 1);

        List<Metric> out = new ArrayList<>();
        if (computeAvgLatency) {
            out.add(Metric.of("search.latency.mean_ms", "Mean Latency (ms)", DEFAULT_FORMAT, mean / 1e6));
        }
        if (computeLatencySTD) {
            out.add(Metric.of("search.latency.std_ms", "STD Latency (ms)", DEFAULT_FORMAT, std / 1e6));
        }
        if (computeP999Latency) {
            out.add(Metric.of("search.latency.p999_ms", "p999 Latency (ms)", DEFAULT_FORMAT, sorted[idx] / 1e6));
        }
        return out;
    }
}
