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
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.diagnostics.BenchmarkDiagnostics;
import io.github.jbellis.jvector.example.benchmarks.diagnostics.DiagnosticLevel;
import io.github.jbellis.jvector.fastervector.bench.IvfConfig.ConstructionPoint;
import io.github.jbellis.jvector.fastervector.bench.IvfConfig.SearchPoint;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * The IVF counterpart of jvector-examples' {@code Grid}: for each construction point in an
 * {@link IvfConfig}, build or reopen the index, then run every benchmark at every search point
 * and hand each row of metrics to a {@link RowSink}.
 */
public final class IvfGrid {
    private static final Logger log = LoggerFactory.getLogger(IvfGrid.class);

    private IvfGrid() {}

    /** Receives one row per construction point and search point. */
    @FunctionalInterface
    public interface RowSink {
        /** Called once per construction point, before its rows. */
        default void startConstructionPoint(IvfSystem system) {}

        void accept(IvfSystem system, SearchPoint point, List<Metric> metrics);
    }

    /**
     * Why fastervector can't run a dataset with this similarity function, if it can't.
     * fastervector scores by inner product: COSINE datasets are normalized on load and DOT_PRODUCT
     * is native, but EUCLIDEAN ordering differs from inner product.
     */
    public static Optional<String> unsupportedReason(VectorSimilarityFunction similarity) {
        if (similarity == VectorSimilarityFunction.EUCLIDEAN) {
            return Optional.of("fastervector scores by inner product only; EUCLIDEAN is not supported");
        }
        return Optional.empty();
    }

    /**
     * Runs {@code config} against {@code ds}. Skips the dataset, with a log message, if its
     * similarity function is unsupported.
     *
     * @return false if the dataset was skipped
     */
    public static boolean run(DataSet ds, IvfConfig config, List<IvfQueryBenchmark> benchmarks, RowSink sink)
            throws IOException {
        Optional<String> reason = unsupportedReason(ds.getSimilarityFunction());
        if (reason.isPresent()) {
            log.warn("Skipping dataset {}: {}", ds.getName(), reason.get());
            return false;
        }
        List<ConstructionPoint> constructionPoints = config.constructionPoints();
        List<SearchPoint> searchPoints = config.searchPoints();
        int total = constructionPoints.size() * searchPoints.size();
        int n = 0;
        try (Heartbeat heartbeat = Heartbeat.start(ds.getName())) {
            for (ConstructionPoint point : constructionPoints) {
                String index = point.subfolder(ds.getName());
                heartbeat.phase("building or opening " + index);
                try (IvfSystem system = IvfSystem.buildOrOpen(ds, config, point)) {
                    sink.startConstructionPoint(system);
                    List<Metric> constructionMetrics = system.constructionMetrics();
                    for (SearchPoint searchPoint : searchPoints) {
                        String label = String.format("%s [%d/%d] topK=%d nprobe=%d searchThreads=%d", index, ++n, total,
                                                     searchPoint.topK(), searchPoint.nprobe(), searchPoint.searchThreads());
                        long start = System.nanoTime();
                        log.info("Search point {}", label);
                        List<Metric> metrics = runBenchmarks(system, searchPoint, benchmarks, heartbeat, label);
                        log.info("Search point {} done in {}s", label, secondsSince(start));
                        metrics.addAll(constructionMetrics);
                        sink.accept(system, searchPoint, metrics);
                    }
                }
            }
        }
        return true;
    }

    /**
     * Runs the benchmarks for one search point and adds the JVM memory metrics the graph runner's
     * {@code QueryTester} reports, from snapshots before and after. fastervector's own native
     * memory isn't part of the JVM's off-heap count.
     */
    private static List<Metric> runBenchmarks(IvfSystem system, SearchPoint point, List<IvfQueryBenchmark> benchmarks,
                                              Heartbeat heartbeat, String label) throws IOException {
        List<Metric> metrics = new ArrayList<>();
        try (var diagnostics = new BenchmarkDiagnostics(DiagnosticLevel.NONE)) {
            diagnostics.capturePrePhaseSnapshot("Query");
            for (IvfQueryBenchmark benchmark : benchmarks) {
                heartbeat.phase(label + ": " + benchmark.getBenchmarkName());
                long start = System.nanoTime();
                metrics.addAll(benchmark.runBenchmark(system, point));
                log.info("  {} done in {}s", benchmark.getBenchmarkName(), secondsSince(start));
            }
            diagnostics.capturePostPhaseSnapshot("Query");
            var pre = diagnostics.getPrePhaseSystemSnapshot();
            var post = diagnostics.getLatestSystemSnapshot();
            if (pre != null && post != null) {
                long heap = Math.max(pre.memoryStats.heapUsed, post.memoryStats.heapUsed);
                long offHeap = Math.max(pre.memoryStats.getTotalOffHeapMemory(), post.memoryStats.getTotalOffHeapMemory());
                metrics.add(Metric.of("search.system.max_heap_mb", "Max heap usage (MB)", ".1f", heap / (1024.0 * 1024.0)));
                metrics.add(Metric.of("search.system.max_offheap_mb", "Max offheap usage (MB)", ".1f",
                                      offHeap / (1024.0 * 1024.0)));
            }
        }
        return metrics;
    }

    static String secondsSince(long startNanos) {
        return String.format("%.1f", (System.nanoTime() - startNanos) / 1e9);
    }

    /**
     * Logs the current phase every {@code ivf.heartbeatSeconds} seconds (default 60; 0 turns it
     * off), so a long phase, such as a single-threaded pass over the query set, isn't silent.
     */
    static final class Heartbeat implements AutoCloseable {
        static final String INTERVAL_PROPERTY = "ivf.heartbeatSeconds";

        private final ScheduledExecutorService timer;
        private final long start = System.nanoTime();
        private volatile String phase = "starting";
        private volatile long phaseStart = start;

        private Heartbeat(String name, long intervalSeconds) {
            if (intervalSeconds <= 0) {
                timer = null;
                return;
            }
            timer = Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "ivf-heartbeat");
                t.setDaemon(true);
                return t;
            });
            timer.scheduleAtFixedRate(() -> log.info("Still running {}: {} ({}s in this phase, {}s in total)",
                                                     name, phase, secondsSince(phaseStart), secondsSince(start)),
                                      intervalSeconds, intervalSeconds, TimeUnit.SECONDS);
        }

        static Heartbeat start(String name) {
            return new Heartbeat(name, Long.getLong(INTERVAL_PROPERTY, 60));
        }

        void phase(String phase) {
            this.phaseStart = System.nanoTime();
            this.phase = phase;
        }

        @Override
        public void close() {
            if (timer != null) {
                timer.shutdownNow();
            }
        }
    }
}
