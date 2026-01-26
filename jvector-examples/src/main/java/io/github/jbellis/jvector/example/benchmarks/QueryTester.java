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

import io.github.jbellis.jvector.example.Grid;
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

        // Capture memory and disk usage before running queries
        // Use NONE level to suppress logging output that would break the table
        var diagnostics = new BenchmarkDiagnostics(io.github.jbellis.jvector.example.benchmarks.diagnostics.DiagnosticLevel.NONE);
        if (monitoredDirectory != null) {
            try {
                diagnostics.setMonitoredDirectory(monitoredDirectory);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        diagnostics.capturePrePhaseSnapshot("Query");

        for (var benchmark : benchmarks) {
            var metrics = benchmark.runBenchmark(cs, topK, rerankK, usePruning, queryRuns);
            results.addAll(metrics);
        }

        // Capture memory and disk usage after running queries
        diagnostics.capturePostPhaseSnapshot("Query");
        
        // Add memory and disk metrics to results
        var systemSnapshot = diagnostics.getLatestSystemSnapshot();
        var diskSnapshot = diagnostics.getLatestDiskSnapshot();
        
        if (systemSnapshot != null) {
            // Max heap usage in MB
            results.add(Metric.of("Max heap usage", ".1f",
                systemSnapshot.memoryStats.heapUsed / (1024.0 * 1024.0)));
            
            // Max off-heap usage (direct + mapped) in MB
            results.add(Metric.of("Max offheap usage", ".1f",
                systemSnapshot.memoryStats.getTotalOffHeapMemory() / (1024.0 * 1024.0)));
        }
        
        if (diskSnapshot != null) {
            // Total file size in MB
            results.add(Metric.of("Total file size", ".1f",
                diskSnapshot.totalBytes / (1024.0 * 1024.0)));
            
            // Number of files
            results.add(Metric.of("Number of files", ".0f",
                (double) diskSnapshot.fileCount));
        }
        
        // Add index build time if available
        if (datasetName != null && Grid.getIndexBuildTimeSeconds(datasetName) != null) {
            results.add(Metric.of("Index build time", ".2f",
                Grid.getIndexBuildTimeSeconds(datasetName)));
        }

        return results;
    }
}

