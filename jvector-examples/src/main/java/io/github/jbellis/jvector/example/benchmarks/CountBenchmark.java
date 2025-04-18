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
 * Measures average node‐visit and node‐expand counts over N runs.
 */
public class CountBenchmark implements QueryBenchmark<CountBenchmark.Summary> {

    /**
     * Holds the averaged node‐count metrics.
     */
    public static class Summary implements BenchmarkSummary {
        private final double avgNodesVisited;
        private final double avgNodesExpanded;
        private final double avgNodesExpandedBaseLayer;

        public Summary(double avgNodesVisited,
                       double avgNodesExpanded,
                       double avgNodesExpandedBaseLayer) {
            this.avgNodesVisited = avgNodesVisited;
            this.avgNodesExpanded = avgNodesExpanded;
            this.avgNodesExpandedBaseLayer = avgNodesExpandedBaseLayer;
        }

        @Override
        public String toString() {
            return String.format(
                    "CountSummary{%.2f nodes visited (AVG), %.2f nodes expanded, and %.2f nodes expanded in base layer}",
                    avgNodesVisited,
                    avgNodesExpanded,
                    avgNodesExpandedBaseLayer
            );
        }

        public double getAvgNodesVisited() {
            return avgNodesVisited;
        }

        public double getAvgNodesExpanded() {
            return avgNodesExpanded;
        }

        public double getAvgNodesExpandedBaseLayer() {
            return avgNodesExpandedBaseLayer;
        }
    }

    @Override
    public String getBenchmarkName() {
        return "CountBenchmark";
    }

    @Override
    public Summary runBenchmark(
            ConfiguredSystem cs,
            int topK,
            int rerankK,
            boolean usePruning,
            int queryRuns) {

        LongAdder nodesVisited = new LongAdder();
        LongAdder nodesExpanded = new LongAdder();
        LongAdder nodesExpandedBaseLayer = new LongAdder();
        int totalQueries = cs.getDataSet().queryVectors.size();

        for (int run = 0; run < queryRuns; run++) {
            IntStream.range(0, totalQueries)
                    .parallel()
                    .forEach(i -> {
                        SearchResult sr = QueryExecutor.executeQuery(
                                cs, topK, rerankK, usePruning, i);
                        nodesVisited.add(sr.getVisitedCount());
                        nodesExpanded.add(sr.getExpandedCount());
                        nodesExpandedBaseLayer.add(sr.getExpandedCountBaseLayer());
                    });
        }

        double avgVisited = nodesVisited.sum() / (double) (queryRuns * totalQueries);
        double avgExpanded = nodesExpanded.sum() / (double) (queryRuns * totalQueries);
        double avgBase = nodesExpandedBaseLayer.sum() / (double) (queryRuns * totalQueries);

        return new Summary(avgVisited, avgExpanded, avgBase);
    }
}


