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
import io.github.jbellis.jvector.fastervector.IvfAshIndexSearcher;
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
 * Queries per second, in one of two modes (Part 7, D5.3):
 * <ul>
 *   <li>{@link Mode#BATCHED}: the whole query set as one {@code search(Vecs)} call on one searcher
 *       with a {@code searchThreads} pool. This is how fastervector is meant to be used, so it
 *       reports under the graph benchmark's {@code search.throughput.*} keys.</li>
 *   <li>{@link Mode#PER_QUERY}: one query per call, spread over {@code searchThreads} Java threads,
 *       each with its own single-threaded searcher. This has the shape of the graph benchmark's
 *       QPS and reports under {@code search.throughput.per_query.*}.</li>
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
        return mode == Mode.BATCHED ? "IvfThroughputBenchmark" : "IvfPerQueryThroughputBenchmark";
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
        if (mode == Mode.BATCHED) {
            Vecs warmupVecs = new Array2DVecs(warmup);
            try (FvecThreadPool pool = FvecThreadPool.create(point.searchThreads());
                 IvfAshIndexSearcher searcher = system.searcher(point.topK(), point.nprobe(), pool)) {
                for (int i = 0; i < numWarmupRuns; i++) {
                    long start = System.nanoTime();
                    consume(searcher.search(warmupVecs));
                    logRun("warm-up", i, numWarmupRuns, nq / ((System.nanoTime() - start) / 1e9));
                }
                for (int i = 0; i < numTestRuns; i++) {
                    gcPause();
                    long start = System.nanoTime();
                    consume(searcher.search(system.queryVecs()));
                    qps[i] = nq / ((System.nanoTime() - start) / 1e9);
                    logRun("run", i, numTestRuns, qps[i]);
                }
            }
        } else {
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
                for (int t = 0; t < threads; t++) {
                    searchers.add(system.searcher(point.topK(), point.nprobe(), null));
                }
                for (int i = 0; i < numWarmupRuns; i++) {
                    long start = System.nanoTime();
                    runPerQuery(executor, searchers, warmupQueries);
                    logRun("warm-up", i, numWarmupRuns, nq / ((System.nanoTime() - start) / 1e9));
                }
                for (int i = 0; i < numTestRuns; i++) {
                    gcPause();
                    long start = System.nanoTime();
                    runPerQuery(executor, searchers, queries);
                    qps[i] = nq / ((System.nanoTime() - start) / 1e9);
                    logRun("run", i, numTestRuns, qps[i]);
                }
            } finally {
                executor.shutdownNow();
                searchers.forEach(IvfAshIndexSearcher::close);
            }
        }
        return metrics(qps);
    }

    private void logRun(String kind, int i, int of, double qps) {
        log.info("    {} {} {}/{}: {} QPS", getBenchmarkName(), kind, i + 1, of, String.format("%.1f", qps));
    }

    /** Each worker owns one searcher and pulls query indexes from a shared counter. */
    private static void runPerQuery(ExecutorService executor, List<IvfAshIndexSearcher> searchers, Vecs[] queries) {
        AtomicInteger next = new AtomicInteger();
        List<Future<?>> workers = new ArrayList<>(searchers.size());
        for (IvfAshIndexSearcher searcher : searchers) {
            workers.add(executor.submit(() -> {
                for (int i = next.getAndIncrement(); i < queries.length; i = next.getAndIncrement()) {
                    consume(searcher.search(queries[i]));
                }
            }));
        }
        for (Future<?> worker : workers) {
            try {
                worker.get();
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
}
