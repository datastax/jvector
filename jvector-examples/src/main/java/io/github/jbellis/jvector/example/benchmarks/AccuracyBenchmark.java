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

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.example.Grid.ConfiguredSystem;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.graph.SearchResult;

/**
 * Measures average recall over N runs.
 */
public class AccuracyBenchmark extends AbstractQueryBenchmark {
    private final boolean computeRecall;
    private final boolean computeMAP;

    public AccuracyBenchmark(boolean computeRecall, boolean computeMAP) {
        super(".2f");
        if (!(computeRecall || computeMAP)) {
            throw new IllegalArgumentException("At least one parameter must be set to true");
        }
        this.computeRecall = computeRecall;
        this.computeMAP = computeMAP;
    }

    public AccuracyBenchmark() {
        this(true, false);
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

        int totalQueries = cs.getDataSet().queryVectors.size();

        // execute all queries in parallel and collect results
        List<SearchResult> results = IntStream.range(0, totalQueries)
                .parallel()
                .mapToObj(i -> QueryExecutor.executeQuery(
                        cs, topK, rerankK, usePruning, i))
                .collect(Collectors.toList());

        var list = new ArrayList<Metric>();
        if (computeRecall) {
            // compute recall for this run
            double recall = AccuracyMetrics.recallFromSearchResults(
                            cs.getDataSet().groundTruth, results, topK, topK
            );
            list.add(Metric.of("Recall@" + topK, getPrintPrecision(), recall));
        }
        if (computeMAP) {
            // compute recall for this run
            double map = AccuracyMetrics.meanAveragePrecisionAtK(
                            cs.getDataSet().groundTruth, results, topK
            );
            list.add(Metric.of("MAP@" + topK, getPrintPrecision(), map));
        }
        return list;
    }
}


