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

import java.util.concurrent.atomic.LongAdder;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.example.Grid.ConfiguredSystem;
import io.github.jbellis.jvector.graph.SearchResult;

/**
 * Measures throughput (queries/sec) with an optional warmup phase.
 */
public class ThroughputBenchmark
        implements QueryBenchmark<ThroughputBenchmark.Summary> {

    private static volatile long SINK;

    /**
     * Holds the number of correct top‑K results and the achieved QPS.
     */
    public static class Summary implements BenchmarkSummary {
        private final double avgNodesVisited;
        private final double queriesPerSecond;

        public Summary(double queriesPerSecond, double avgNodesVisited) {
            this.avgNodesVisited = avgNodesVisited;
            this.queriesPerSecond = queriesPerSecond;
        }

        public double getQueriesPerSecond() {
            return queriesPerSecond;
        }

        public double getAvgNodesVisited() {
            return avgNodesVisited;
        }

        @Override
        public String toString() {
            return String.format("ThroughputSummary{%.2f qps, with nodes visited (AVG) %.2f}",
                    queriesPerSecond, avgNodesVisited
            );
        }
    }

    private final int warmupRuns;
    private final double warmupRatio;

    public ThroughputBenchmark(int warmupRuns, double warmupRatio) {
        this.warmupRuns = warmupRuns;
        this.warmupRatio = warmupRatio;
    }

    @Override
    public String getBenchmarkName() {
        return "ThroughputBenchmark";
    }

    @Override
    public Summary runBenchmark(
            ConfiguredSystem cs,
            int topK,
            int rerankK,
            boolean usePruning,
            int queryRuns) {

        int totalQueries = cs.getDataSet().queryVectors.size();
        int warmupCount   = (int) (totalQueries * warmupRatio);
        int testCount     = totalQueries - warmupCount;

        // Warmup Phase
        if (warmupCount > 0) {
            for (int run = 0; run < warmupRuns; run++) {
                IntStream.range(0, warmupCount)
                        .parallel()
                        .forEach(i -> {
                            SearchResult sr = QueryExecutor.executeQuery(
                                    cs, topK, rerankK, usePruning, i);
                            SINK += sr.getVisitedCount();
                        });
            }
        }

        // Test Phase
        LongAdder visitedAdder = new LongAdder();
        long startTime = System.nanoTime();

        IntStream.range(0, testCount)
                .parallel()
                .forEach(i -> {
                    int queryIndex = i + warmupCount;
                    SearchResult sr = QueryExecutor.executeQuery(
                            cs, topK, rerankK, usePruning, queryIndex);
                    // “Use” the result to prevent optimization
                    visitedAdder.add(sr.getVisitedCount());
                });

        double elapsedSec = (System.nanoTime() - startTime) / 1e9;
        double qps = testCount / elapsedSec;
        double avgVisited = (double) visitedAdder.sum() / testCount;

        return new Summary(qps, avgVisited);
    }
}
