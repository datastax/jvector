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
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.DataSetLoader;
import io.github.jbellis.jvector.example.yaml.ConstructionParameters;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.example.yaml.SearchParameters;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

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
 * which datasets have been fully processed. If the benchmark is restarted,
 * it will skip datasets that have already been processed, allowing it to
 * continue from where it left off rather than starting over from the beginning.
 */
public class AutoBenchYAML {
    private static final Logger logger = LoggerFactory.getLogger(AutoBenchYAML.class);

    /**
     * Returns a list of all dataset names.
     * This replaces the need to load datasets.yml which may not be available in all environments.
     */
    private static List<String> getAllDatasetNames() {
        List<String> allDatasets = new ArrayList<>();
        allDatasets.add("cap-1M");
        allDatasets.add("cap-6M");
        allDatasets.add("cohere-english-v3-1M");
        allDatasets.add("cohere-english-v3-10M");
        allDatasets.add("dpr-1M");
        allDatasets.add("dpr-10M");

        return allDatasets;
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
        logger.info("Initialized checkpoint manager. Already completed datasets: {}", checkpointManager.getCompletedDatasets());

        // Filter out --output, --config and their arguments from the args
        String finalOutputPath = outputPath;
        String configPath = null;
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals("--config")) configPath = args[i+1];
        }
        String finalConfigPath = configPath;
        String[] filteredArgs = Arrays.stream(args)
                .filter(arg -> !arg.equals("--output") && !arg.equals(finalOutputPath) && 
                               !arg.equals("--config") && !arg.equals(finalConfigPath))
                .toArray(String[]::new);

        // Log the filtered arguments for debugging
        logger.info("Filtered arguments: {}", Arrays.toString(filteredArgs));

        // generate a regex that matches any regex in filteredArgs, or if filteredArgs is empty/null, match everything
        var regex = filteredArgs.length == 0 ? ".*" : Arrays.stream(filteredArgs).flatMap(s -> Arrays.stream(s.split("\\s"))).map(s -> "(?:" + s + ")").collect(Collectors.joining("|"));
        logger.info("Generated regex pattern: {}", regex);

        // compile regex and do substring matching using find
        var pattern = Pattern.compile(regex);

        var datasetNames = getAllDatasetNames().stream().filter(dn -> pattern.matcher(dn).find()).collect(Collectors.toList());

        logger.info("Executing the following datasets: {}", datasetNames);
        List<BenchResult> results = new ArrayList<>();
        // Add results from checkpoint if present
        results.addAll(checkpointManager.getCompletedResults());

        // Process datasets from regex patterns
        if (!datasetNames.isEmpty()) {
            for (var datasetName : datasetNames) {
                // Skip already completed datasets
                if (checkpointManager.isDatasetCompleted(datasetName)) {
                    logger.info("Skipping already completed dataset: {}", datasetName);
                    continue;
                }

                logger.info("Loading dataset: {}", datasetName);
                try {
                    DataSet ds = DataSetLoader.loadDataSet(datasetName);
                    logger.info("Dataset loaded: {} with {} vectors", datasetName, ds.baseVectors.size());

                    String normalizedDatasetName = datasetName;
                    if (normalizedDatasetName.endsWith(".hdf5")) {
                        normalizedDatasetName = normalizedDatasetName.substring(0, normalizedDatasetName.length() - ".hdf5".length());
                    }

                    MultiConfig config = MultiConfig.getDefaultConfig("autoDefault");
                    config.dataset = normalizedDatasetName;
                    logger.info("Using configuration: {}", config);

                    List<BenchResult> datasetResults = Grid.runAllAndCollectResults(ds, 
                            config.construction.outDegree, 
                            config.construction.efConstruction,
                            config.construction.neighborOverflow, 
                            config.construction.addHierarchy,
                            config.construction.getFeatureSets(), 
                            config.construction.getCompressorParameters(),
                            config.search.getCompressorParameters(), 
                            config.search.topKOverquery, 
                            config.search.useSearchPruning);
                    results.addAll(datasetResults);

                    logger.info("Benchmark completed for dataset: {}", datasetName);
                    // Mark dataset as completed and update checkpoint, passing results
                    checkpointManager.markDatasetCompleted(datasetName, datasetResults);
                } catch (Exception e) {
                    logger.error("Exception while processing dataset {}", datasetName, e);
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
                writer.write("dataset,QPS,QPS StdDev,Mean Latency,Recall@10,Index Construction Time\n");

                // Write one row per dataset with average metrics
                for (Map.Entry<String, SummaryStats> entry : statsByDataset.entrySet()) {
                    String dataset = entry.getKey();
                    SummaryStats datasetStats = entry.getValue();

                    writer.write(dataset + ",");
                    writer.write(datasetStats.getAvgQps() + ",");
                    writer.write(datasetStats.getQpsStdDev() + ",");
                    writer.write(datasetStats.getAvgLatency() + ",");
                    writer.write(datasetStats.getAvgRecall() + ",");
                    writer.write(datasetStats.getIndexConstruction() + "\n");
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
