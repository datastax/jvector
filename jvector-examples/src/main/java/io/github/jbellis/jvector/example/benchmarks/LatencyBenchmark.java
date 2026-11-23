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
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.LongSupplier;
import java.util.function.Supplier;

import io.github.jbellis.jvector.example.Grid.ConfiguredSystem;
import io.github.jbellis.jvector.graph.SearchResult;
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
        if (queries.length > 0) {
            int dimension = Objects.requireNonNull(queries[0], "query").length();
            if (dimension <= 0) throw new IllegalArgumentException("Query dimension must be positive");
        }
        var floats = VectorizationProvider.getInstance().getVectorTypeSupport();
        int dimension = queries.length == 0 ? 0 : queries[0].length();
        return measure(queries, () -> floats.createFloatVector(dimension), FloatQuerySet::randomizeVector,
                queryRuns, warmupPasses, minimumNanos, search, clock);
    }

    /**
     * As above, for queries of any vector type: {@code newQuery} allocates a warmup buffer, once per query, and
     * {@code randomize} refills one before each warmup sweep.
     */
    static <Q> long[] measure(Q[] queries, Supplier<Q> newQuery, Consumer<Q> randomize, int queryRuns,
                              int warmupPasses, long minimumNanos, Function<Q, SearchResult> search,
                              LongSupplier clock) {
        Objects.requireNonNull(queries, "queries");
        Objects.requireNonNull(search, "search");
        Objects.requireNonNull(clock, "clock");
        if (queries.length == 0 || queryRuns <= 0 || warmupPasses <= 0 || minimumNanos < 0)
            throw new IllegalArgumentException("Require queries, positive run counts, and nonnegative warmup duration");
        Objects.requireNonNull(queries[0], "query");
        long[] times = new long[Math.multiplyExact(queries.length, queryRuns)];
        Q[] randomInputs = Arrays.copyOf(queries, queries.length);
        for (int i = 0; i < randomInputs.length; i++) randomInputs[i] = newQuery.get();

        for (int phase = 0; phase <= warmupPasses; phase++) {
            boolean warming = phase < warmupPasses;
            Q[] inputs = warming ? randomInputs : queries;
            int repeats = warming ? 1 : queryRuns;
            int offset = 0;
            long started = clock.getAsLong();
            do {
                if (warming) { for (Q query : randomInputs) randomize.accept(query); offset = 0; }
                for (int run = 0; run < repeats; run++)
                    offset = runPass(inputs, search, times, offset);
            } while (warming && clock.getAsLong() - started < minimumNanos);
        }
        return times;
    }

    /** No warmup/measurement branch or allocation at the timed call site. */
    private static <Q> int runPass(Q[] queries, Function<Q, SearchResult> search,
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

        return metrics(latencies(cs, cs.queries(), topK, rerankK, usePruning, queryRuns));
    }

    /** Each query's latency, with every query copied into an array before timing starts. */
    private static <Q> long[] latencies(ConfiguredSystem cs, QuerySet<Q> queries, int topK, int rerankK,
                                        boolean usePruning, int queryRuns) {
        Q[] inputs = queries.newArray(queries.size());
        for (int i = 0; i < inputs.length; i++) inputs[i] = queries.get(i);
        try (QueuedQueryExecution<Q> queued = cs.usesQueuedExecution()
                ? new QueuedQueryExecution<>(cs, queries, topK, rerankK, usePruning) : null) {
            Function<Q, SearchResult> search = query -> queued == null
                    ? QueryExecutor.executeQuery(cs, queries, topK, rerankK, usePruning, query)
                    : queued.executeSingle(query);
            return measure(inputs, queries::newQuery, queries::randomize, queryRuns, WARMUP_PASSES, MIN_WARMUP_NANOS,
                    search, System::nanoTime);
        }
    }
}
