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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSetLoaderMFD;
import io.github.jbellis.jvector.example.reporting.ReportingSelectionResolver;
import io.github.jbellis.jvector.example.reporting.SearchReportingCatalog;
import io.github.jbellis.jvector.example.yaml.MultiConfig;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Tests GraphIndexes against vectors from various datasets
 */
public class HelloVectorWorld {
    public static void main(String[] args) throws IOException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());
        String datasetName = "ada002-100k";
        var ds = new DataSetLoaderMFD().loadDataSet(datasetName)
                .orElseThrow(() -> new RuntimeException("dataset " + datasetName + " not found"));
        MultiConfig config = MultiConfig.getDefaultConfig(datasetName);

        var benchmarksToCompute = config.search.benchmarks;

        var benchmarksToDisplay =
                (config.search.console != null) ? config.search.console.benchmarks : null;

        var metricsToDisplay =
                (config.search.console != null) ? config.search.console.metrics : null;

        Map<String, List<String>> benchmarksToLog =
                (config.search != null && config.search.logging != null)
                        ? config.search.logging.benchmarks
                        : null;

        var metricsToLog =
                (config.search != null && config.search.logging != null)
                        ? config.search.logging.metrics
                        : null;

        // Validate display / console selection
        ReportingSelectionResolver.validateBenchmarkSelectionSubset(
                benchmarksToCompute,
                benchmarksToDisplay,
                SearchReportingCatalog.defaultComputeBenchmarks()
        );

        ReportingSelectionResolver.validateNamedMetricSelectionNames(
                metricsToDisplay,
                SearchReportingCatalog.catalog()
        );

        // Validate logging selection
        ReportingSelectionResolver.validateBenchmarkSelectionSubset(
                benchmarksToCompute,
                benchmarksToLog,
                SearchReportingCatalog.defaultComputeBenchmarks()
        );
        ReportingSelectionResolver.validateNamedMetricSelectionNames(
                metricsToLog,
                SearchReportingCatalog.catalog()
        );

        Grid.runAll(ds,
                config.construction.useSavedIndexIfExists,
                config.construction.outDegree,
                config.construction.efConstruction,
                config.construction.neighborOverflow,
                config.construction.addHierarchy,
                config.construction.refineFinalGraph,
                config.construction.getFeatureSets(),
                config.construction.getCompressorParameters(),
                config.search.getCompressorParameters(),
                config.search.topKOverquery,
                config.search.useSearchPruning,
                benchmarksToCompute,
                benchmarksToDisplay,
                metricsToDisplay,
                benchmarksToLog,
                metricsToLog);

    }
}
