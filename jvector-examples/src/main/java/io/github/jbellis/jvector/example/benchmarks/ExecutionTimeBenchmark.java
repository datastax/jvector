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

import java.util.stream.IntStream;
import io.github.jbellis.jvector.example.Grid.ConfiguredSystem;
import io.github.jbellis.jvector.graph.SearchResult;

/**
 * Measures average execution time over N runs through all queries in parallel.
 */
public class ExecutionTimeBenchmark
        implements QueryBenchmark<ExecutionTimeBenchmark.Summary> {

    private static volatile long SINK;

    /**
     * Simple summary.
     */
    public static class Summary implements BenchmarkSummary {
        private final double averageRuntimeSec;

        public Summary(double averageRuntimeSec) {
            this.averageRuntimeSec = averageRuntimeSec;
        }

        @Override
        public String toString() {
            return String.format(
                    "ExecutionTimeSummary{%.2fs execution time}",
                    averageRuntimeSec
            );
        }

        public double getAvgRuntimeSec() {
            return averageRuntimeSec;
        }
    }

    @Override
    public String getBenchmarkName() {
        return "ExecutionTimeBenchmark";
    }

    @Override
    public Summary runBenchmark(
            ConfiguredSystem cs,
            int topK,
            int rerankK,
            boolean usePruning,
            int queryRuns) {

        int totalQueries = cs.getDataSet().queryVectors.size();
        double totalRuntime = 0;

        for (int run = 0; run < queryRuns; run++) {
            double startTime = System.nanoTime();

            // execute all queries in parallel
            IntStream.range(0, totalQueries)
                    .parallel()
                    .forEach(i -> {
                        SearchResult sr = QueryExecutor.executeQuery(
                                cs, topK, rerankK, usePruning, i);
                        SINK += sr.getVisitedCount();
                    });

            totalRuntime += System.nanoTime() - startTime;
        }

        double avgRuntimeSec = totalRuntime / queryRuns / 1e9;
        return new Summary(avgRuntimeSec);
    }
}



