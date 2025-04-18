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

/**
 * Aggregates the results of all benchmarks into one summary.
 */
public class CompositeSummary implements BenchmarkSummary {

    private final ExecutionTimeBenchmark.Summary executionTimeSummary;
    private final CountBenchmark.Summary countSummary;
    private final RecallBenchmark.Summary recallSummary;
    private final ThroughputBenchmark.Summary throughputSummary;
    private final LatencyBenchmark.Summary latencySummary;

    public CompositeSummary(
            ExecutionTimeBenchmark.Summary executionTimeSummary,
            CountBenchmark.Summary countSummary,
            RecallBenchmark.Summary recallSummary,
            ThroughputBenchmark.Summary throughputSummary,
            LatencyBenchmark.Summary latencySummary) {
        this.executionTimeSummary = executionTimeSummary;
        this.countSummary         = countSummary;
        this.recallSummary        = recallSummary;
        this.throughputSummary    = throughputSummary;
        this.latencySummary       = latencySummary;
    }

    public ExecutionTimeBenchmark.Summary getExecutionTimeSummary() {
        return executionTimeSummary;
    }

    public CountBenchmark.Summary getCountSummary() {
        return countSummary;
    }

    public RecallBenchmark.Summary getRecallSummary() {
        return recallSummary;
    }

    public ThroughputBenchmark.Summary getThroughputSummary() {
        return throughputSummary;
    }

    public LatencyBenchmark.Summary getLatencySummary() {
        return latencySummary;
    }

    @Override
    public String toString() {
        return "CompositeSummary {\n" +
                "  executionTime=" + executionTimeSummary + ",\n" +
                "  count="         + countSummary         + ",\n" +
                "  recall="        + recallSummary        + ",\n" +
                "  throughput="    + throughputSummary    + ",\n" +
                "  latency="       + latencySummary       + "\n" +
                "}";
    }
}


