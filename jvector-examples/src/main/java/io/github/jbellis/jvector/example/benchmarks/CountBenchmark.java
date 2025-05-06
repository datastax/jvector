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
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.example.Grid.ConfiguredSystem;
import io.github.jbellis.jvector.graph.SearchResult;
import org.apache.commons.math3.analysis.function.Abs;

/**
 * Measures average node‐visit and node‐expand counts over N runs.
 */
public class CountBenchmark extends AbstractQueryBenchmark {
    static private final String DEFAULT_FORMAT = ".1f";

    private final boolean computeAvgNodesVisited;
    private final boolean computeAvgNodesExpanded;
    private final boolean computeAvgNodesExpandedBaseLayer;
    private final String formatAvgNodesVisited;
    private final String formatAvgNodesExpanded;
    private final String formatAvgNodesExpandedBaseLayer;

    public CountBenchmark(boolean computeAvgNodesVisited, boolean computeAvgNodesExpanded, boolean computeAvgNodesExpandedBaseLayer,
                          String formatAvgNodesVisited, String formatAvgNodesExpanded, String formatAvgNodesExpandedBaseLayer) {
        if (!(computeAvgNodesVisited || computeAvgNodesExpanded || computeAvgNodesExpandedBaseLayer)) {
            throw new IllegalArgumentException("At least one parameter must be set to true");
        }
        this.computeAvgNodesVisited = computeAvgNodesVisited;
        this.computeAvgNodesExpanded = computeAvgNodesExpanded;
        this.computeAvgNodesExpandedBaseLayer = computeAvgNodesExpandedBaseLayer;
        this.formatAvgNodesVisited = formatAvgNodesVisited;
        this.formatAvgNodesExpanded = formatAvgNodesExpanded;
        this.formatAvgNodesExpandedBaseLayer = formatAvgNodesExpandedBaseLayer;
    }

    public CountBenchmark() {
        this(true, false, false, DEFAULT_FORMAT, DEFAULT_FORMAT, DEFAULT_FORMAT);
    }

    public CountBenchmark(String formatAvgNodesVisited, String formatAvgNodesExpanded, String formatAvgNodesExpandedBaseLayer) {
        this(true, true, true, formatAvgNodesVisited, formatAvgNodesExpanded, formatAvgNodesExpandedBaseLayer);
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
        return list;
    }
}