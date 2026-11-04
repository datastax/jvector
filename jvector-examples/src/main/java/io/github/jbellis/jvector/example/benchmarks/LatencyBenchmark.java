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
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.function.LongSupplier;

import io.github.jbellis.jvector.example.Grid.ConfiguredSystem;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Measures unloaded per-query latency after random-query warmup through the
 * same execution policy used by the timed requests.
 */
public class LatencyBenchmark extends AbstractQueryBenchmark {
    private static final String DEFAULT_FORMAT = ".3f";
    private static final int WARMUP_PASSES = 3;
    private static final long MIN_WARMUP_NANOS = TimeUnit.SECONDS.toNanos(3);

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

    /**
     * The same pass and search-function instance serve warmup and measurement.
     * Separate dispatch sites can leave the measured call cold after the JVM
     * inlines the warmup site. Random query buffers are allocated once and
     * refilled between warmup sweeps, without touching measured query contents.
     */
    static long[] measure(VectorFloat<?>[] queries, int queryRuns, int warmupPasses,
                          long minimumNanos, Function<VectorFloat<?>, SearchResult> search,
                          LongSupplier clock) {
        Objects.requireNonNull(queries, "queries");
        Objects.requireNonNull(search, "search");
        Objects.requireNonNull(clock, "clock");
        if (queries.length == 0 || queryRuns <= 0 || warmupPasses <= 0 || minimumNanos < 0)
            throw new IllegalArgumentException("Require queries, positive run counts, and nonnegative warmup duration");
        int dimension = Objects.requireNonNull(queries[0], "query").length();
        if (dimension <= 0) throw new IllegalArgumentException("Query dimension must be positive");
        long[] times = new long[Math.multiplyExact(queries.length, queryRuns)];
        var vectors = VectorizationProvider.getInstance().getVectorTypeSupport();
        VectorFloat<?>[] randomInputs = new VectorFloat<?>[queries.length];
        for (int i = 0; i < randomInputs.length; i++) randomInputs[i] = vectors.createFloatVector(dimension);

        for (int phase = 0; phase <= warmupPasses; phase++) {
            boolean warming = phase < warmupPasses;
            VectorFloat<?>[] inputs = warming ? randomInputs : queries;
            int repeats = warming ? 1 : queryRuns;
            int offset = 0;
            long started = clock.getAsLong();
            do {
                if (warming) { randomize(randomInputs, dimension); offset = 0; }
                for (int run = 0; run < repeats; run++)
                    offset = runPass(inputs, search, times, offset);
            } while (warming && clock.getAsLong() - started < minimumNanos);
        }
        return times;
    }

    private static void randomize(VectorFloat<?>[] inputs, int dimension) {
        var random = ThreadLocalRandom.current();
        for (var query : inputs) {
            boolean nonZero = false;
            for (int d = 0; d < dimension; d++) {
                float value = random.nextFloat();
                query.set(d, value);
                nonZero |= value != 0;
            }
            if (!nonZero) query.set(0, 1);
            VectorUtil.l2normalize(query);
        }
    }

    /** No warmup/measurement branch or allocation at the timed call site. */
    private static int runPass(VectorFloat<?>[] queries, Function<VectorFloat<?>, SearchResult> search,
                               long[] times, int offset) {
        Objects.checkFromIndexSize(offset, queries.length, times.length);
        for (int i = 0; i < queries.length; i++) {
            long start = System.nanoTime();
            SearchResult result = search.apply(queries[i]);
            long duration = System.nanoTime() - start;
            times[offset++] = duration;
            SINK += result.getVisitedCount();
        }
        return offset;
    }

    /** Summarize only the completed measured samples; warmup overwrites no input data. */
    List<Metric> metrics(long[] latencies) {
        double mean = 0, m2 = 0;
        int count = 0;
        for (long duration : latencies) {
            count++;
            double delta = duration - mean;
            mean += delta / count;
            m2 += delta * (duration - mean);
        }
        double standardDeviation = count > 0 ? Math.sqrt(m2 / count) / 1e6 : 0;
        Arrays.sort(latencies);
        int index = Math.max(0, Math.min(latencies.length - 1, (int) Math.ceil(0.999 * latencies.length) - 1));
        var result = new ArrayList<Metric>();
        if (computeAvgLatency) result.add(Metric.of("search.latency.mean_ms", "Mean Latency (ms)", formatAvgLatency, mean / 1e6));
        if (computeLatencySTD) result.add(Metric.of("search.latency.std_ms", "STD Latency (ms)", formatLatencySTD, standardDeviation));
        if (computeP999Latency) result.add(Metric.of("search.latency.p999_ms", "p999 Latency (ms)", formatP999Latency, latencies[index] / 1e6));
        return result;
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

        VectorFloat<?>[] queries = cs.getDataSet().getQueryVectors().toArray(new VectorFloat<?>[0]);
        try (QueuedQueryExecution queued = cs.usesQueuedExecution()
                ? new QueuedQueryExecution(cs, topK, rerankK, usePruning) : null) {
            Function<VectorFloat<?>, SearchResult> search = query -> queued == null
                    ? QueryExecutor.executeQuery(cs, topK, rerankK, usePruning, query)
                    : queued.executeSingle(query);
            long[] times = measure(queries, queryRuns, WARMUP_PASSES, MIN_WARMUP_NANOS, search, System::nanoTime);
            return metrics(times);
        }
    }
}
