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

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import io.github.jbellis.jvector.example.Grid.ConfiguredSystem;
import io.github.jbellis.jvector.example.benchmarks.diagnostics.BenchmarkDiagnostics;

/**
 * Orchestrates running a set of QueryBenchmark instances
 * and collects their summary results.
 */
public class QueryTester {
    private final List<QueryBenchmark> benchmarks;
    private final Path monitoredDirectory;
    private final String datasetName;

    /**
     * @param benchmarks the benchmarks to run, in the order provided
     */
    public QueryTester(List<QueryBenchmark> benchmarks) {
        this(benchmarks, null, null);
    }

    /**
     * @param benchmarks the benchmarks to run, in the order provided
     * @param monitoredDirectory optional directory to monitor for disk usage
     */
    public QueryTester(List<QueryBenchmark> benchmarks, Path monitoredDirectory) {
        this(benchmarks, monitoredDirectory, null);
    }

    /**
     * @param benchmarks the benchmarks to run, in the order provided
     * @param monitoredDirectory optional directory to monitor for disk usage
     * @param datasetName optional dataset name for retrieving build time
     */
    public QueryTester(List<QueryBenchmark> benchmarks, Path monitoredDirectory, String datasetName) {
        this.benchmarks = benchmarks;
        this.monitoredDirectory = monitoredDirectory;
        this.datasetName = datasetName;
    }

    /**
     * Run each benchmark once and return a map from each Summary class
     * to its returned summary instance.
     *
     * @param cs          the configured system under test
     * @param topK        the top‑K parameter for all benchmarks
     * @param rerankK     the rerank‑K parameter
     * @param usePruning  whether to enable pruning
     * @param queryRuns   number of runs for each benchmark
     */
    public List<Metric> run(
            ConfiguredSystem cs,
            int topK,
            int rerankK,
            boolean usePruning,
            int queryRuns) {

        List<Metric> results = new ArrayList<>();

        // Capture memory and disk usage before/after running queries
        // Use NONE level to suppress logging output that would break the table
        try (var diagnostics = new BenchmarkDiagnostics(
                io.github.jbellis.jvector.example.benchmarks.diagnostics.DiagnosticLevel.NONE)) {

            if (monitoredDirectory != null) {
                diagnostics.setMonitoredDirectory(monitoredDirectory);
            }

            diagnostics.capturePrePhaseSnapshot("Query");
            
            // Start peak memory tracking before running queries
            diagnostics.startQueryMemoryTracking();

            for (var benchmark : benchmarks) {
                var metrics = benchmark.runBenchmark(cs, topK, rerankK, usePruning, queryRuns);
                results.addAll(metrics);
            }
            
            // Stop peak memory tracking after running queries
            diagnostics.stopQueryMemoryTracking();

            // Capture memory and disk usage after running queries
            diagnostics.capturePostPhaseSnapshot("Query");

            // Add peak memory metrics to results
            var queryPeakMemory = diagnostics.getQueryPeakMemory();
            var diskSnapshot = diagnostics.getLatestDiskSnapshot();

            if (queryPeakMemory != null) {
                // Peak heap usage in MB during queries
                results.add(Metric.of("search.system.max_heap_mb", "Max heap usage (MB)", ".1f",
                        queryPeakMemory.peakHeapUsed / (1024.0 * 1024.0)));

                // Peak off-heap usage (direct + mapped) in MB during queries
                results.add(Metric.of("search.system.max_offheap_mb", "Max offheap usage (MB)", ".1f",
                        queryPeakMemory.getTotalPeakOffHeapMemory() / (1024.0 * 1024.0)));
            }

            if (diskSnapshot != null) {
                // Number of index files created
                results.add(Metric.of("search.disk.file_count", "File count", ".0f",
                        diskSnapshot.getTotalFileCount()));

                // Total size of index files created
                results.add(Metric.of("search.disk.total_file_size_mb", "Total file size (MB)", ".1f",
                        diskSnapshot.getTotalBytes() / (1024.0 * 1024.0)));
            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return results;
    }
}
