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
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.example.Grid.ConfiguredSystem;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.graph.SearchResult;

/**
 * Measures average recall over N runs.
 */
public class RecallBenchmark
        implements QueryBenchmark<RecallBenchmark.Summary> {

    /**
     * Holds the averaged recall metric.
     */
    public static class Summary implements BenchmarkSummary {
        private final double averageRecall;

        public Summary(double averageRecall) {
            this.averageRecall = averageRecall;
        }

        public double getAverageRecall() {
            return averageRecall;
        }

        @Override
        public String toString() {
            return String.format(
                    "RecallSummary{recall %.4f}",
                    averageRecall
            );
        }
    }

    @Override
    public String getBenchmarkName() {
        return "RecallBenchmark";
    }

    @Override
    public Summary runBenchmark(
            ConfiguredSystem cs,
            int topK,
            int rerankK,
            boolean usePruning,
            int queryRuns) {

        double totalRecall = 0.0;
        int totalQueries = cs.getDataSet().queryVectors.size();

        for (int run = 0; run < queryRuns; run++) {
            // execute all queries in parallel and collect results
            List<SearchResult> results = IntStream.range(0, totalQueries)
                    .parallel()
                    .mapToObj(i -> QueryExecutor.executeQuery(
                            cs, topK, rerankK, usePruning, i))
                    .collect(Collectors.toList());

            // compute recall for this run
            double recall = AccuracyMetrics
                    .recallFromSearchResults(
                            cs.getDataSet().groundTruth, results, topK, topK);
            totalRecall += recall;
        }

        double avgRecall = totalRecall / queryRuns;
        return new Summary(avgRecall);
    }
}


