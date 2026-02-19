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

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.jbellis.jvector.example.util.BenchmarkSummarizer;
import io.github.jbellis.jvector.example.util.BenchmarkSummarizer.SummaryStats;
import io.github.jbellis.jvector.example.util.CheckpointManager;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.example.yaml.MultiConfig;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Automated benchmark runner for GitHub Actions workflow.
 * This class is specifically designed to handle the --output argument
 * for regression testing in the run-bench.yml workflow.
 * 
 * The benchmark runner supports checkpointing to allow resuming from failures.
 * It creates a checkpoint file (outputPath + ".checkpoint.json") that records
 * which tests have been fully processed. If the benchmark is restarted,
 * it will skip tests that have already been processed, allowing it to
 * continue from where it left off rather than starting over from the beginning.
 */
public class AutoBenchYAML {
    private static final Logger logger = LoggerFactory.getLogger(AutoBenchYAML.class);

    /**
     * Returns a list of all test names.
     * This includes both dataset names (which fall back to autoDefault) and specific test configuration names.
     */
    private static List<String> getAllTestNames() {
        List<String> tests = new ArrayList<>();
        // Standard datasets using autoDefault
        tests.add("cap-1M");
        tests.add("cap-6M");
        tests.add("cohere-english-v3-1M");
        tests.add("cohere-english-v3-10M");
        tests.add("dpr-1M");
        tests.add("dpr-10M");
        tests.add("ada002-100k");
        
        // Specific test configurations
        tests.add("ada002-100k-compaction");
        tests.add("colbert-1M");
        tests.add("glove-25-angular");

        return tests;
    }

    public static void main(String[] args) throws IOException {
        // Check for --output argument (required for this class)
        String outputPath = null;
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals("--output")) outputPath = args[i+1];
        }

        if (outputPath == null) {
            logger.error("Error: --output argument is required for AutoBenchYAML");
            System.exit(1);
        }

        logger.info("Heap space available is {}", Runtime.getRuntime().maxMemory());

        // Initialize checkpoint manager
        CheckpointManager checkpointManager = new CheckpointManager(outputPath);
        logger.info("Initialized checkpoint manager. Already completed tests: {}", checkpointManager.getCompletedDatasets());

        // Filter out --output, --config and their arguments from the args
        String configPath = null;
        int diagnostic_level = 0;
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals("--config")) configPath = args[i+1];
            if (args[i].equals("--diag")) diagnostic_level = Integer.parseInt(args[i+1]);
        }
        if (diagnostic_level > 0) {
            Grid.setDiagnosticLevel(diagnostic_level);
        }
        final String fOutputPath = outputPath;
        final String fConfigPath = configPath;
        String[] filteredArgs = Arrays.stream(args)
                .filter(arg -> !arg.equals("--output") && !arg.equals(fOutputPath) && 
                               !arg.equals("--config") && !arg.equals(fConfigPath))
                .toArray(String[]::new);

        // Log the filtered arguments for debugging
        logger.info("Filtered arguments: {}", Arrays.toString(filteredArgs));

        // generate a regex that matches any regex in filteredArgs, or if filteredArgs is empty/null, match everything
        var regex = filteredArgs.length == 0 ? ".*" : Arrays.stream(filteredArgs).flatMap(s -> Arrays.stream(s.split("\\s"))).map(s -> "(?:" + s + ")").collect(Collectors.joining("|"));
        logger.info("Generated regex pattern: {}", regex);

        // compile regex and do substring matching using find
        var pattern = Pattern.compile(regex);

        var testNames = getAllTestNames().stream().filter(tn -> pattern.matcher(tn).find()).collect(Collectors.toList());

        logger.info("Executing the following tests: {}", testNames);
        List<BenchResult> results = new ArrayList<>();
        // Add results from checkpoint if present
        results.addAll(checkpointManager.getCompletedResults());

        // Process tests from regex patterns
        if (!testNames.isEmpty()) {
            for (var testName : testNames) {
                // Skip already completed tests
                if (checkpointManager.isDatasetCompleted(testName)) {
                    logger.info("Skipping already completed test: {}", testName);
                    continue;
                }

                logger.info("Starting test: {}", testName);
                try {
                    MultiConfig config;
                    if (fConfigPath != null) {
                        config = MultiConfig.getConfig(fConfigPath);
                    } else {
                        // Try to load a config matching the test name, fallback to autoDefault
                        try {
                            config = MultiConfig.getConfig(testName);
                        } catch (Exception e) {
                            config = MultiConfig.getDefaultConfig("autoDefault");
                        }
                    }

                    // Resolve the actual dataset name to use
                    String dtl = (config.dataset != null && !config.dataset.isEmpty()) 
                            ? config.dataset 
                            : testName;
                    
                    // Standardize dataset name (remove .hdf5 if present)
                    if (dtl.endsWith(".hdf5")) {
                        dtl = dtl.substring(0, dtl.length() - ".hdf5".length());
                    }
                    final String datasetToLoad = dtl;

                    logger.info("Loading dataset [{}] for test [{}]", datasetToLoad, testName);
                    DataSet ds = DataSets.loadDataSet(datasetToLoad).orElseThrow(
                            () -> new RuntimeException("Dataset " + datasetToLoad + " not found")
                    );
                    logger.info("Dataset loaded: {} with {} vectors", datasetToLoad, ds.getBaseVectors().size());

                    logger.info("Using configuration: {}", config);

                                          List<BenchResult> testResults = Grid.runAllAndCollectResults(ds,
                                                  config.construction.useSavedIndexIfExists,
                                                  config.construction.outDegree, 
                                                  config.construction.efConstruction,
                                                  config.construction.neighborOverflow, 
                                                  config.construction.addHierarchy,
                                                  config.construction.refineFinalGraph,
                                                  config.construction.getFeatureSets(), 
                                                                                config.construction.getCompressorParameters(),
                                                                                config.partitions,
                                                                                config.search.getCompressorParameters(), 
                                                   
                                                  config.search.topKOverquery, 
                                                  config.search.useSearchPruning);
                                        // Tag results with the test name if it differs from dataset name
                    for (var res : testResults) {
                        if (!res.dataset.equals(testName)) {
                            res.parameters.put("testName", testName);
                        }
                    }
                    
                    results.addAll(testResults);

                    logger.info("Test completed: {}", testName);
                    // Mark test as completed and update checkpoint, passing results
                    checkpointManager.markDatasetCompleted(testName, testResults);
                } catch (Exception e) {
                    logger.error("Exception while processing test {}", testName, e);
                }
            }
        }

        // Calculate summary statistics
        try {
            SummaryStats stats = BenchmarkSummarizer.summarize(results);
            logger.info("Benchmark summary: {}", stats.toString());

            // Write results to csv file and details to json
            File detailsFile = new File(outputPath + ".json");
            ObjectMapper mapper = new ObjectMapper();
            mapper.writerWithDefaultPrettyPrinter().writeValue(detailsFile, results);

            File outputFile = new File(outputPath + ".csv");

            // Get summary statistics by dataset
            Map<String, SummaryStats> statsByDataset = BenchmarkSummarizer.summarizeByDataset(results);

            // Write CSV data
            try (FileWriter writer = new FileWriter(outputFile)) {
                // Write CSV header
                writer.write("dataset,QPS,QPS StdDev,Mean Latency,Recall@10,Index Construction Time,Avg Nodes Visited\n");

                // Write one row per dataset with average metrics
                for (Map.Entry<String, SummaryStats> entry : statsByDataset.entrySet()) {
                    String dataset = entry.getKey();
                    SummaryStats datasetStats = entry.getValue();

                    writer.write(dataset + ",");
                    writer.write(datasetStats.getAvgQps() + ",");
                    writer.write(datasetStats.getQpsStdDev() + ",");
                    writer.write(datasetStats.getAvgLatency() + ",");
                    writer.write(datasetStats.getAvgRecall() + ",");
                    writer.write(datasetStats.getIndexConstruction() + ",");
                    writer.write(datasetStats.getAvgNodesVisited() + "\n");
                }
            }

            logger.info("Benchmark results written to {} (file exists: {})", outputPath, outputFile.exists());
            // Double check that the file was created and log its size
            if (outputFile.exists()) {
                logger.info("Output file size: {} bytes", outputFile.length());
            } else {
                logger.error("Failed to create output file at {}", outputPath);
            }
        } catch (Exception e) {
            logger.error("Exception during final processing", e);
        }
    }

}
