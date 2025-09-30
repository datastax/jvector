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
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.status.StatusTracker;
import io.github.jbellis.jvector.status.StatusUpdate;
import io.github.jbellis.jvector.status.TrackerScope;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Measures average recall and/or the mean average precision.
 */
public class AccuracyBenchmark extends AbstractQueryBenchmark {
    private static final String DEFAULT_FORMAT = ".2f";

    private boolean computeRecall;
    private boolean computeMAP;
    private String formatRecall;
    private String formatMAP;

    public static AccuracyBenchmark createDefault() {
        return new AccuracyBenchmark(true, false, DEFAULT_FORMAT, DEFAULT_FORMAT);
    }

    public static AccuracyBenchmark createEmpty() {
        return new AccuracyBenchmark(false, false, DEFAULT_FORMAT, DEFAULT_FORMAT);
    }

    private AccuracyBenchmark(boolean computeRecall, boolean computeMAP, String formatRecall, String formatMAP) {
        this.computeRecall = computeRecall;
        this.computeMAP = computeMAP;
        this.formatRecall = formatRecall;
        this.formatMAP = formatMAP;
    }

    public AccuracyBenchmark displayRecall() {
        return displayRecall(DEFAULT_FORMAT);
    }

    public AccuracyBenchmark displayRecall(String format) {
        this.computeRecall = true;
        this.formatRecall = format;
        return this;
    }

    public AccuracyBenchmark displayMAP() {
        return displayMAP(DEFAULT_FORMAT);
    }

    public AccuracyBenchmark displayMAP(String format) {
        this.computeMAP = true;
        this.formatMAP = format;
        return this;
    }

    @Override
    public String getBenchmarkName() {
        return "RecallBenchmark";
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
        if (!(computeRecall || computeMAP)) {
            throw new RuntimeException("At least one metric must be displayed");
        }

        AccuracyComputationTask task = new AccuracyComputationTask(getBenchmarkName());
        StatusTracker<AccuracyComputationTask> tracker = null;
        if (parentTracker != null) {
            tracker = parentTracker.createChild(task);
        } else if (parentScope != null) {
            tracker = parentScope.track(task);
        }

        try (StatusTracker<AccuracyComputationTask> ignored = tracker) {
            if (tracker != null) {
                task.start();
            }

            int totalQueries = cs.getDataSet().queryVectors.size();
            List<SearchResult> results = IntStream.range(0, totalQueries)
                    .parallel()
                    .mapToObj(i -> QueryExecutor.executeQuery(
                            cs, topK, rerankK, usePruning, i))
                    .collect(Collectors.toList());

            var list = new ArrayList<Metric>();
            if (computeRecall) {
                double recall = AccuracyMetrics.recallFromSearchResults(
                        cs.getDataSet().groundTruth, results, topK, topK);
                list.add(Metric.of("Recall@" + topK, formatRecall, recall));
            }
            if (computeMAP) {
                double map = AccuracyMetrics.meanAveragePrecisionAtK(
                        cs.getDataSet().groundTruth, results, topK);
                list.add(Metric.of("MAP@" + topK, formatMAP, map));
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

    private static final class AccuracyComputationTask implements StatusUpdate.Provider<AccuracyComputationTask> {
        private final String name;
        private volatile StatusUpdate.RunState state = StatusUpdate.RunState.PENDING;
        private volatile double progress = 0.0;

        AccuracyComputationTask(String name) {
            this.name = name;
        }

        void start() {
            state = StatusUpdate.RunState.RUNNING;
            progress = 0.0;
        }

        void complete() {
            progress = 1.0;
            state = StatusUpdate.RunState.SUCCESS;
        }

        void fail() {
            state = StatusUpdate.RunState.FAILED;
        }

        @Override
        public StatusUpdate<AccuracyComputationTask> getTaskStatus() {
            return new StatusUpdate<>(progress, state, this);
        }

        @Override
        public String toString() {
            return name;
        }
    }
}
