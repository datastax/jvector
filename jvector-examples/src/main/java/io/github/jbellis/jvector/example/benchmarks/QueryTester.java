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

/**
 * Runs each of the five benchmarks on the same ConfiguredSystem
 * and returns a CompositeSummary.
 */
public class QueryTester {

    /**
     * Executes:
     *   1) ExecutionTimeBenchmark
     *   2) CountBenchmark
     *   3) RecallBenchmark
     *   4) ThroughputBenchmark (with warmupRuns &amp; warmupRatio)
     *   5) LatencyBenchmark
     *
     * @return CompositeSummary aggregating all five results.
     */
    public static CompositeSummary performQueries(
            ConfiguredSystem cs,
            int topK,
            int rerankK,
            boolean usePruning,
            int queryRuns,
            int warmupRuns,
            double warmupRatio) {

        // 1. Execution time
        ExecutionTimeBenchmark exeBench = new ExecutionTimeBenchmark();
        ExecutionTimeBenchmark.Summary exeSummary =
                exeBench.runBenchmark(cs, topK, rerankK, usePruning, queryRuns);

        // 2. Node count
        CountBenchmark countBench = new CountBenchmark();
        CountBenchmark.Summary countSummary =
                countBench.runBenchmark(cs, topK, rerankK, usePruning, queryRuns);

        // 3. Recall
        RecallBenchmark recallBench = new RecallBenchmark();
        RecallBenchmark.Summary recallSummary =
                recallBench.runBenchmark(cs, topK, rerankK, usePruning, queryRuns);

        // 4. Throughput
        ThroughputBenchmark thrBench = new ThroughputBenchmark(warmupRuns, warmupRatio);
        ThroughputBenchmark.Summary thrSummary =
                thrBench.runBenchmark(cs, topK, rerankK, usePruning, queryRuns);

        // 5. Latency
        LatencyBenchmark latBench = new LatencyBenchmark();
        LatencyBenchmark.Summary latSummary =
                latBench.runBenchmark(cs, topK, rerankK, usePruning, queryRuns);

        // Aggregate into a single CompositeSummary
        return new CompositeSummary(
                exeSummary,
                countSummary,
                recallSummary,
                thrSummary,
                latSummary
        );
    }
}

