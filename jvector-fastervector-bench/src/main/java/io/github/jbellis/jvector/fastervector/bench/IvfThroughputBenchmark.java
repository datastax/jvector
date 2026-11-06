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
import io.github.jbellis.jvector.fastervector.FvecThreadPool;
import io.github.jbellis.jvector.fastervector.IvfAshBatchingSearcher;
import io.github.jbellis.jvector.fastervector.IvfAshIndexSearcher;
import io.github.jbellis.jvector.fastervector.NodeScore;
import io.github.jbellis.jvector.fastervector.SearchResult;
import io.github.jbellis.jvector.fastervector.Vecs;
import io.github.jbellis.jvector.fastervector.bench.IvfConfig.SearchPoint;
import io.github.jbellis.jvector.fastervector.utils.Array2DVecs;
import io.github.jbellis.jvector.fastervector.utils.FlatArrayVecs;
import org.apache.commons.math3.stat.StatUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Queries per second, in one of three modes (Part 7, D5.3):
 * <ul>
 *   <li>{@link Mode#BATCHED}: the whole query set as one {@code search(Vecs)} call on one searcher
 *       with a {@code searchThreads} pool. This is how fastervector is meant to be used, so it
 *       reports under the graph benchmark's {@code search.throughput.*} keys.</li>
 *   <li>{@link Mode#COALESCED}: one query per call from {@code searchThreads} Java threads, through
 *       an {@link IvfAshBatchingSearcher} over one searcher with a {@code searchThreads} pool. This
 *       has the shape of the graph benchmark's QPS, a server answering single requests, and
 *       reports under {@code search.throughput.coalesced.*}.</li>
 *   <li>{@link Mode#PER_QUERY}: one query per call from {@code searchThreads} Java threads, each
 *       with its own searcher of {@code queryThreads} threads, calling fastervector directly.
 *       Reports under {@code search.throughput.per_query.*}.</li>
 * </ul>
 * Like the graph's {@code ThroughputBenchmark}, warm-up runs use random unit vectors so the measured
 * runs get no help from caches warmed by the real queries, and each measured run follows a GC pause.
 */
public class IvfThroughputBenchmark implements IvfQueryBenchmark {
    private static final Logger log = LoggerFactory.getLogger(IvfThroughputBenchmark.class);
    private static final String DEFAULT_FORMAT = ".1f";
    private static volatile long SINK;

    public enum Mode {
        BATCHED("search.throughput.", ""),
        COALESCED("search.throughput.coalesced.", " coalesced"),
        PER_QUERY("search.throughput.per_query.", " per-query");

        final String keyPrefix;
        final String headerSuffix;

        Mode(String keyPrefix, String headerSuffix) {
            this.keyPrefix = keyPrefix;
            this.headerSuffix = headerSuffix;
        }
    }

    private final Mode mode;
    private final int numWarmupRuns;
    private final int numTestRuns;
    private boolean computeAvgQps;
    private boolean computeMedianQps;
    private boolean computeMaxQps;

    private IvfThroughputBenchmark(Mode mode, int numWarmupRuns, int numTestRuns) {
        if (numTestRuns < 1) {
            throw new IllegalArgumentException("numTestRuns must be at least 1");
        }
        this.mode = mode;
        this.numWarmupRuns = numWarmupRuns;
        this.numTestRuns = numTestRuns;
    }

    /** 3 warm-up and 3 measured runs, reporting average QPS, as the graph's createDefault() does. */
    public static IvfThroughputBenchmark createDefault(Mode mode) {
        return createEmpty(mode, 3, 3).displayAvgQps();
    }

    public static IvfThroughputBenchmark createEmpty(Mode mode, int numWarmupRuns, int numTestRuns) {
        return new IvfThroughputBenchmark(mode, numWarmupRuns, numTestRuns);
    }

    public IvfThroughputBenchmark displayAvgQps() {
        computeAvgQps = true;
        return this;
    }

    public IvfThroughputBenchmark displayMedianQps() {
        computeMedianQps = true;
        return this;
    }

    public IvfThroughputBenchmark displayMaxQps() {
        computeMaxQps = true;
        return this;
    }

    @Override
    public String getBenchmarkName() {
        return switch (mode) {
            case BATCHED -> "IvfThroughputBenchmark";
            case COALESCED -> "IvfCoalescedThroughputBenchmark";
            case PER_QUERY -> "IvfPerQueryThroughputBenchmark";
        };
    }

    @Override
    public List<Metric> runBenchmark(IvfSystem system, SearchPoint point) {
        if (!(computeAvgQps || computeMedianQps || computeMaxQps)) {
            throw new IllegalStateException("At least one metric must be displayed");
        }
        int nq = system.queryCount();
        int dim = system.query(0).length;
        float[][] warmup = randomUnitVectors(nq, dim, 42);

        double[] qps = new double[numTestRuns];
        switch (mode) {
            case BATCHED -> {
                Vecs warmupVecs = new Array2DVecs(warmup);
                try (FvecThreadPool pool = FvecThreadPool.create(point.searchThreads());
                     IvfAshIndexSearcher searcher = system.searcher(point.topK(), point.nprobe(), pool)) {
                    measure(nq, () -> consume(searcher.search(warmupVecs)),
                            () -> consume(searcher.search(system.queryVecs())), qps);
                }
            }
            case COALESCED -> {
                float[][] queries = new float[nq][];
                for (int i = 0; i < nq; i++) {
                    queries[i] = system.query(i);
                }
                int clients = point.searchThreads();
                ExecutorService executor = Executors.newFixedThreadPool(clients);
                try (FvecThreadPool pool = FvecThreadPool.create(point.searchThreads());
                     IvfAshIndexSearcher searcher = system.searcher(point.topK(), point.nprobe(), pool);
                     IvfAshBatchingSearcher batcher = IvfAshBatchingSearcher.builder(searcher).build()) {
                    measure(nq, () -> runConcurrent(executor, clients, nq, (w, i) -> consumeRow(batcher.search(warmup[i]))),
                            () -> runConcurrent(executor, clients, nq, (w, i) -> consumeRow(batcher.search(queries[i]))), qps);
                    log.info("    {}: {} clients, mean batch {} queries", getBenchmarkName(), clients,
                             String.format("%.1f", (double) batcher.queryCount() / Math.max(1, batcher.batchCount())));
                } finally {
                    executor.shutdownNow();
                }
            }
            case PER_QUERY -> {
                Vecs[] queries = new Vecs[nq];
                Vecs[] warmupQueries = new Vecs[nq];
                for (int i = 0; i < nq; i++) {
                    queries[i] = new FlatArrayVecs(system.query(i), dim);
                    warmupQueries[i] = new FlatArrayVecs(warmup[i], dim);
                }
                int threads = point.searchThreads();
                List<IvfAshIndexSearcher> searchers = new ArrayList<>(threads);
                ExecutorService executor = Executors.newFixedThreadPool(threads);
                try {
                    // The searcher reads only the pool's thread count, so one pool serves them all
                    try (FvecThreadPool pool = FvecThreadPool.create(point.queryThreads())) {
                        for (int t = 0; t < threads; t++) {
                            searchers.add(system.searcher(point.topK(), point.nprobe(), pool));
                        }
                    }
                    measure(nq, () -> runConcurrent(executor, threads, nq, (w, i) -> consume(searchers.get(w).search(warmupQueries[i]))),
                            () -> runConcurrent(executor, threads, nq, (w, i) -> consume(searchers.get(w).search(queries[i]))), qps);
                } finally {
                    executor.shutdownNow();
                    searchers.forEach(IvfAshIndexSearcher::close);
                }
            }
        }
        return metrics(qps);
    }

    /** Runs the warm-up passes, then the measured passes, each after a GC pause, into {@code qps}. */
    private void measure(int nq, Runnable warmupPass, Runnable pass, double[] qps) {
        for (int i = 0; i < numWarmupRuns; i++) {
            long start = System.nanoTime();
            warmupPass.run();
            logRun("warm-up", i, numWarmupRuns, nq / ((System.nanoTime() - start) / 1e9));
        }
        for (int i = 0; i < numTestRuns; i++) {
            gcPause();
            long start = System.nanoTime();
            pass.run();
            qps[i] = nq / ((System.nanoTime() - start) / 1e9);
            logRun("run", i, numTestRuns, qps[i]);
        }
    }

    private void logRun(String kind, int i, int of, double qps) {
        log.info("    {} {} {}/{}: {} QPS", getBenchmarkName(), kind, i + 1, of, String.format("%.1f", qps));
    }

    /** One query by one worker. */
    @FunctionalInterface
    private interface Query {
        void run(int worker, int query);
    }

    /** {@code workers} threads pull query indexes from a shared counter until all {@code nq} are done. */
    private static void runConcurrent(ExecutorService executor, int workers, int nq, Query query) {
        AtomicInteger next = new AtomicInteger();
        List<Future<?>> futures = new ArrayList<>(workers);
        for (int w = 0; w < workers; w++) {
            int worker = w;
            futures.add(executor.submit(() -> {
                for (int i = next.getAndIncrement(); i < nq; i = next.getAndIncrement()) {
                    query.run(worker, i);
                }
            }));
        }
        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            } catch (java.util.concurrent.ExecutionException e) {
                throw new RuntimeException(e.getCause());
            }
        }
    }

    List<Metric> metrics(double[] qps) {
        double[] sorted = qps.clone();
        Arrays.sort(sorted);
        double avg = StatUtils.mean(sorted);
        double stdDev = sorted.length > 1 ? Math.sqrt(StatUtils.variance(sorted)) : 0.0;
        String runs = " (of " + numTestRuns + ")";
        String p = mode.keyPrefix;
        String s = mode.headerSuffix;
        List<Metric> out = new ArrayList<>();
        if (computeAvgQps) {
            out.add(Metric.of(p + "avg_qps", "Avg" + s + " QPS\n" + runs, DEFAULT_FORMAT, avg));
            out.add(Metric.of(p + "stddev_qps", "± Std Dev", DEFAULT_FORMAT, stdDev));
            out.add(Metric.of(p + "cv_pct", "CV %", ".1f", stdDev / avg * 100));
        }
        if (computeMedianQps) {
            out.add(Metric.of(p + "median_qps", "Median" + s + " QPS\n" + runs, DEFAULT_FORMAT,
                              sorted[sorted.length / 2]));
        }
        if (computeMaxQps) {
            out.add(Metric.of(p + "max_qps", "Max" + s + " QPS\n" + runs, DEFAULT_FORMAT, sorted[sorted.length - 1]));
            out.add(Metric.of(p + "min_qps", "Min" + s + " QPS\n" + runs, DEFAULT_FORMAT, sorted[0]));
        }
        return out;
    }

    static float[][] randomUnitVectors(int n, int dim, long seed) {
        Random r = new Random(seed);
        float[][] out = new float[n][dim];
        for (float[] v : out) {
            double norm = 0;
            for (int j = 0; j < dim; j++) {
                v[j] = (float) r.nextGaussian();
                norm += v[j] * v[j];
            }
            float scale = (float) (1 / Math.sqrt(norm));
            for (int j = 0; j < dim; j++) {
                v[j] *= scale;
            }
        }
        return out;
    }

    /** Clears Eden before a measured run, as the graph benchmark does. */
    static void gcPause() {
        System.gc();
        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /** Uses the result so the JIT can't drop the search. */
    static void consume(SearchResult result) {
        SINK += result.results().size();
    }

    static void consumeRow(List<NodeScore> row) {
        SINK += row.size();
    }
}
