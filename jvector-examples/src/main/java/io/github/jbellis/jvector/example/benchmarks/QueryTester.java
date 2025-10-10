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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import io.github.jbellis.jvector.example.Grid.ConfiguredSystem;

/**
 * Orchestrates running a set of QueryBenchmark instances and collects their summary results.
 * Provides a simple interface for executing multiple benchmarks sequentially and gathering
 * their metrics for analysis and comparison.
 */
public class QueryTester {
    private final List<QueryBenchmark> benchmarks;

    /**
     * Constructs a QueryTester with the specified benchmarks to execute.
     *
     * @param benchmarks the benchmarks to run, in the order provided
     */
    public QueryTester(List<QueryBenchmark> benchmarks) {
        this.benchmarks = benchmarks;
    }

    /**
     * Runs each benchmark once and returns the collected metrics.
     *
     * @param cs          the configured system under test
     * @param topK        the top-K parameter for all benchmarks
     * @param rerankK     the rerank-K parameter
     * @param usePruning  whether to enable pruning
     * @param queryRuns   number of runs for each benchmark
     * @return a list of metrics from all benchmarks
     */
    public List<Metric> run(
            ConfiguredSystem cs,
            int topK,
            int rerankK,
            boolean usePruning,
            int queryRuns) {

        List<Metric> results = new ArrayList<>();

        for (var benchmark : benchmarks) {
            var metrics = benchmark.runBenchmark(cs, topK, rerankK, usePruning, queryRuns);
            results.addAll(metrics);
        }

        return results;
    }
}

