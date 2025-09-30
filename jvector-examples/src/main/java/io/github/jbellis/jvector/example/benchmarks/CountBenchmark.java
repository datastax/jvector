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
import java.util.List;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.IntStream;

/**
 * Measures average node-visit and node-expand counts over N runs.
 */
public class CountBenchmark extends AbstractQueryBenchmark {
    private static final String DEFAULT_FORMAT = ".1f";

    private boolean computeAvgNodesVisited;
    private boolean computeAvgNodesExpanded;
    private boolean computeAvgNodesExpandedBaseLayer;
    private String formatAvgNodesVisited;
    private String formatAvgNodesExpanded;
    private String formatAvgNodesExpandedBaseLayer;

    public static CountBenchmark createDefault() {
        return new CountBenchmark(true, false, false, DEFAULT_FORMAT, DEFAULT_FORMAT, DEFAULT_FORMAT);
    }

    public static CountBenchmark createEmpty() {
        return new CountBenchmark(false, false, false, DEFAULT_FORMAT, DEFAULT_FORMAT, DEFAULT_FORMAT);
    }

    private CountBenchmark(boolean computeAvgNodesVisited, boolean computeAvgNodesExpanded, boolean computeAvgNodesExpandedBaseLayer,
                           String formatAvgNodesVisited, String formatAvgNodesExpanded, String formatAvgNodesExpandedBaseLayer) {
        this.computeAvgNodesVisited = computeAvgNodesVisited;
        this.computeAvgNodesExpanded = computeAvgNodesExpanded;
        this.computeAvgNodesExpandedBaseLayer = computeAvgNodesExpandedBaseLayer;
        this.formatAvgNodesVisited = formatAvgNodesVisited;
        this.formatAvgNodesExpanded = formatAvgNodesExpanded;
        this.formatAvgNodesExpandedBaseLayer = formatAvgNodesExpandedBaseLayer;
    }

    public CountBenchmark displayAvgNodesVisited() {
        return displayAvgNodesVisited(DEFAULT_FORMAT);
    }

    public CountBenchmark displayAvgNodesVisited(String format) {
        this.computeAvgNodesVisited = true;
        this.formatAvgNodesVisited = format;
        return this;
    }

    public CountBenchmark displayAvgNodesExpanded() {
        return displayAvgNodesExpanded(DEFAULT_FORMAT);
    }

    public CountBenchmark displayAvgNodesExpanded(String format) {
        this.computeAvgNodesExpanded = true;
        this.formatAvgNodesExpanded = format;
        return this;
    }

    public CountBenchmark displayAvgNodesExpandedBaseLayer() {
        return displayAvgNodesExpandedBaseLayer(DEFAULT_FORMAT);
    }

    public CountBenchmark displayAvgNodesExpandedBaseLayer(String format) {
        this.computeAvgNodesExpandedBaseLayer = true;
        this.formatAvgNodesExpandedBaseLayer = format;
        return this;
    }

    @Override
    public String getBenchmarkName() {
        return "CountBenchmark";
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
        if (!(computeAvgNodesVisited || computeAvgNodesExpanded || computeAvgNodesExpandedBaseLayer)) {
            throw new RuntimeException("At least one metric must be displayed");
        }

        CountComputationTask task = new CountComputationTask(getBenchmarkName(), Math.max(1, queryRuns));
        StatusTracker<CountComputationTask> tracker = null;
        if (parentTracker != null) {
            tracker = parentTracker.createChild(task);
        } else if (parentScope != null) {
            tracker = parentScope.track(task);
        }

        try (StatusTracker<CountComputationTask> ignored = tracker) {
            if (tracker != null) {
                task.start();
            }

            LongAdder nodesVisited = new LongAdder();
            LongAdder nodesExpanded = new LongAdder();
            LongAdder nodesExpandedBaseLayer = new LongAdder();
            int totalQueries = cs.getDataSet().queryVectors.size();

            for (int run = 0; run < queryRuns; run++) {
                final int currentRun = run;
                IntStream.range(0, totalQueries)
                         .parallel()
                         .forEach(i -> {
                             SearchResult sr = QueryExecutor.executeQuery(
                                     cs, topK, rerankK, usePruning, i);
                             nodesVisited.add(sr.getVisitedCount());
                             nodesExpanded.add(sr.getExpandedCount());
                             nodesExpandedBaseLayer.add(sr.getExpandedCountBaseLayer());
                         });
                if (tracker != null) {
                    task.updateProgress(currentRun + 1);
                }
            }

            double avgVisited = nodesVisited.sum() / (double) (Math.max(1, queryRuns) * cs.getDataSet().queryVectors.size());
            double avgExpanded = nodesExpanded.sum() / (double) (Math.max(1, queryRuns) * cs.getDataSet().queryVectors.size());
            double avgBase = nodesExpandedBaseLayer.sum() / (double) (Math.max(1, queryRuns) * cs.getDataSet().queryVectors.size());

            var list = new ArrayList<Metric>();
            if (computeAvgNodesVisited) {
                list.add(Metric.of("Avg Visited", formatAvgNodesVisited, avgVisited));
            }
            if (computeAvgNodesExpanded) {
                list.add(Metric.of("Avg Expanded", formatAvgNodesExpanded, avgExpanded));
            }
            if (computeAvgNodesExpandedBaseLayer) {
                list.add(Metric.of("Avg Expanded Base Layer", formatAvgNodesExpandedBaseLayer, avgBase));
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

    private static final class CountComputationTask implements StatusUpdate.Provider<CountComputationTask> {
        private final String name;
        private final int totalRuns;
        private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;
        private volatile double progress = 0.0;

        CountComputationTask(String name, int totalRuns) {
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
        public StatusUpdate<CountComputationTask> getTaskStatus() {
            return new StatusUpdate<>(progress, state, this);
        }

        @Override
        public String toString() {
            return name;
        }
    }
}
