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
import io.github.jbellis.jvector.status.StatusTracker;
import io.github.jbellis.jvector.status.StatusUpdate;
import io.github.jbellis.jvector.status.TrackerScope;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Measures per-query latency (mean and standard deviation) over N runs,
 * and counts correct top-K results.
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
        return runInternal(null, null, cs, topK, rerankK, usePruning, queryRuns);
    }

    @Override
    public List<Metric> runBenchmark(
            TrackerScope parentScope,
            ConfiguredSystem cs,
            int topK,
            int rerankK,
            boolean usePruning,
            int queryRuns) {
        return runInternal(parentScope, null, cs, topK, rerankK, usePruning, queryRuns);
    }

    @Override
    public List<Metric> runBenchmark(
            TrackerScope parentScope,
            StatusTracker<?> parentTracker,
            ConfiguredSystem cs,
            int topK,
            int rerankK,
            boolean usePruning,
            int queryRuns) {
        return runInternal(parentScope, parentTracker, cs, topK, rerankK, usePruning, queryRuns);
    }

    private List<Metric> runInternal(
            TrackerScope parentScope,
            StatusTracker<?> parentTracker,
            ConfiguredSystem cs,
            int topK,
            int rerankK,
            boolean usePruning,
            int queryRuns) {
        if (!(computeAvgLatency || computeLatencySTD || computeP999Latency)) {
            throw new IllegalArgumentException("At least one parameter must be set to true");
        }

        int totalQueries = cs.getDataSet().queryVectors.size();
        LatencyComputationTask task = new LatencyComputationTask(getBenchmarkName(), Math.max(1, queryRuns));

        StatusTracker<LatencyComputationTask> tracker = null;
        if (parentTracker != null) {
            tracker = parentTracker.createChild(task);
        } else if (parentScope != null) {
            tracker = parentScope.track(task);
        }

        try (StatusTracker<LatencyComputationTask> ignored = tracker) {
            if (tracker != null) {
                task.start();
            }

            double mean = 0.0;
            double m2 = 0.0;
            int count = 0;

            List<Long> latencies = new ArrayList<>(totalQueries * Math.max(1, queryRuns));

            for (int run = 0; run < queryRuns; run++) {
                for (int i = 0; i < totalQueries; i++) {
                    long start = System.nanoTime();
                    SearchResult sr = QueryExecutor.executeQuery(
                            cs, topK, rerankK, usePruning, i);
                    long duration = System.nanoTime() - start;
                    latencies.add(duration);
                    SINK += sr.getVisitedCount();

                    count++;
                    double delta = duration - mean;
                    mean += delta / count;
                    m2 += delta * (duration - mean);
                }

                if (tracker != null) {
                    task.updateProgress(run + 1);
                }
            }

            mean /= 1e6;
            double standardDeviation = (count > 0) ? Math.sqrt(m2 / count) / 1e6 : 0.0;

            Collections.sort(latencies);
            int idx = (int) Math.ceil(0.999 * latencies.size()) - 1;
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

            if (tracker != null) {
                task.complete();
            }
            return list;
        } catch (Exception e) {
            if (tracker != null) {
                task.fail();
            }
            throw e;
        }
    }

    private static final class LatencyComputationTask implements StatusUpdate.Provider<LatencyComputationTask> {
        private final String name;
        private final int totalRuns;
        private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;
        private volatile double progress = 0.0;

        LatencyComputationTask(String name, int totalRuns) {
            this.name = name;
            this.totalRuns = Math.max(1, totalRuns);
        }

        void start() {
            state = StatusUpdate.RunState.RUNNING;
            progress = totalRuns == 0 ? 1.0 : 0.0;
        }

        void updateProgress(int completedRuns) {
            if (state == StatusUpdate.RunState.PENDING) {
                state = StatusUpdate.RunState.RUNNING;
            }
            progress = Math.max(0.0, Math.min(1.0, (double) completedRuns / totalRuns));
        }

        void complete() {
            progress = 1.0;
            state = StatusUpdate.RunState.SUCCESS;
        }

        void fail() {
            state = StatusUpdate.RunState.FAILED;
        }

        @Override
        public StatusUpdate<LatencyComputationTask> getTaskStatus() {
            return new StatusUpdate<>(progress, state, this);
        }

        @Override
        public String toString() {
            return name;
        }
    }
}
