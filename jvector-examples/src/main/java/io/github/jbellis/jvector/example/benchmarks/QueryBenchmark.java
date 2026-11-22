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

import java.util.List;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * A common interface for all search benchmarks.
 */
public interface QueryBenchmark {
    String getBenchmarkName();

    List<Metric> runBenchmark(
            ConfiguredSystem cs,
            int topK,
            int rerankK,
            boolean usePruning,
            int queryRuns
    );

    /**
     * Runs within a query-measurement group. The reporter belongs only to this
     * invocation and must not be retained after this method returns. Suppliers
     * must own completed samples, independently of any closed execution object.
     * QueryTester evaluates them after all measurements in the group succeed.
     * The original overload remains responsible for immediate standalone reports.
     */
    default List<Metric> runBenchmark(
            ConfiguredSystem cs, int topK, int rerankK, boolean usePruning,
            int queryRuns, Consumer<Supplier<List<Metric>>> deferredReports) {
        return runBenchmark(cs, topK, rerankK, usePruning, queryRuns);
    }
}


