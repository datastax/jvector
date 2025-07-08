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
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.DataSetLoader;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Automated benchmark runner for GitHub Actions workflow.
 * This class is specifically designed to handle the --output argument
 * for regression testing in the run-bench.yml workflow.
 */
public class AutoBenchYAML {
    private static final Logger logger = LoggerFactory.getLogger(AutoBenchYAML.class);

    /**
     * Returns a list of all dataset names.
     * This replaces the need to load datasets.yml which may not be available in all environments.
     */
    private static List<String> getAllDatasetNames() {
        List<String> allDatasets = new ArrayList<>();

        // neighborhood-watch-100k datasets
//        allDatasets.add("ada002-100k");
        allDatasets.add("cohere-english-v3-100k");
//        allDatasets.add("openai-v3-small-100k");
//        allDatasets.add("gecko-100k");
//        allDatasets.add("openai-v3-large-3072-100k");
//        allDatasets.add("openai-v3-large-1536-100k");
//        allDatasets.add("e5-small-v2-100k");
//        allDatasets.add("e5-base-v2-100k");
//        allDatasets.add("e5-large-v2-100k");

        // neighborhood-watch-1M datasets
//        allDatasets.add("ada002-1M");
//        allDatasets.add("colbert-1M");
//
//        // ann-benchmarks datasets
//        allDatasets.add("glove-25-angular.hdf5");
//        allDatasets.add("glove-50-angular.hdf5");
//        allDatasets.add("lastfm-64-dot.hdf5");
//        allDatasets.add("glove-100-angular.hdf5");
//        allDatasets.add("glove-200-angular.hdf5");
//        allDatasets.add("nytimes-256-angular.hdf5");
//        allDatasets.add("sift-128-euclidean.hdf5");
//        // Large files not yet supported:
        // allDatasets.add("deep-image-96-angular.hdf5");
        // allDatasets.add("gist-960-euclidean.hdf5");

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

        // Filter out --output and its argument from the args
        String finalOutputPath = outputPath;
        String[] filteredArgs = Arrays.stream(args)
                .filter(arg -> !arg.equals("--output") && !arg.equals(finalOutputPath))
                .toArray(String[]::new);

        // generate a regex that matches any regex in filteredArgs, or if filteredArgs is empty/null, match everything
        var regex = filteredArgs.length == 0 ? ".*" : Arrays.stream(filteredArgs).flatMap(s -> Arrays.stream(s.split("\\s"))).map(s -> "(?:" + s + ")").collect(Collectors.joining("|"));
        // compile regex and do substring matching using find
        var pattern = Pattern.compile(regex);

        var datasetNames = getAllDatasetNames().stream().filter(dn -> pattern.matcher(dn).find()).collect(Collectors.toList());

        logger.info("Executing the following datasets: {}", datasetNames);
        List<BenchResult> results = new ArrayList<>();

        // Process datasets from regex patterns
        if (!datasetNames.isEmpty()) {
            for (var datasetName : datasetNames) {
                logger.info("Loading dataset: {}", datasetName);
                try {
                    DataSet ds = DataSetLoader.loadDataSet(datasetName);
                    logger.info("Dataset loaded: {} with {} vectors", datasetName, ds.baseVectors.size());

                    if (datasetName.endsWith(".hdf5")) {
                        datasetName = datasetName.substring(0, datasetName.length() - ".hdf5".length());
                    }
                    
                    MultiConfig config = MultiConfig.getDefaultConfig(datasetName);
                    logger.info("Using configuration: {}", config);

                    results.addAll(Grid.runAllAndCollectResults(ds, 
                            config.construction.outDegree, 
                            config.construction.efConstruction,
                            config.construction.neighborOverflow, 
                            config.construction.addHierarchy,
                            config.construction.getFeatureSets(), 
                            config.construction.getCompressorParameters(),
                            config.search.getCompressorParameters(), 
                            config.search.topKOverquery, 
                            config.search.useSearchPruning));
                    
                    logger.info("Benchmark completed for dataset: {}", datasetName);
                } catch (Exception e) {
                    logger.error("Exception while processing dataset {}", datasetName, e);
                }
            }
        }

        // Process YAML configuration files
        List<String> configNames = Arrays.stream(filteredArgs).filter(s -> s.endsWith(".yml")).collect(Collectors.toList());
        if (!configNames.isEmpty()) {
            for (var configName : configNames) {
                logger.info("Processing configuration file: {}", configName);
                
                try {
                    MultiConfig config = MultiConfig.getConfig(configName);
                    String datasetName = config.dataset;
                    logger.info("Configuration specifies dataset: {}", datasetName);

                    logger.info("Loading dataset: {}", datasetName);
                    DataSet ds = DataSetLoader.loadDataSet(datasetName);
                    logger.info("Dataset loaded: {} with {} vectors", datasetName, ds.baseVectors.size());

                    results.addAll(Grid.runAllAndCollectResults(ds, 
                            config.construction.outDegree, 
                            config.construction.efConstruction,
                            config.construction.neighborOverflow, 
                            config.construction.addHierarchy,
                            config.construction.getFeatureSets(), 
                            config.construction.getCompressorParameters(),
                            config.search.getCompressorParameters(), 
                            config.search.topKOverquery, 
                            config.search.useSearchPruning));
                    
                    logger.info("Benchmark completed for YAML config: {}", configName);
                } catch (Exception e) {
                    logger.error("Exception while processing YAML config {}", configName, e);
                }
            }
        }

        // Calculate summary statistics
        try {
            SummaryStats stats = BenchmarkSummarizer.summarize(results);
            logger.info("Benchmark summary: {}", stats.toString());

            // Write results to JSON file
            ObjectMapper mapper = new ObjectMapper();
            File outputFile = new File(outputPath);
            mapper.writerWithDefaultPrettyPrinter().writeValue(outputFile, results);
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
