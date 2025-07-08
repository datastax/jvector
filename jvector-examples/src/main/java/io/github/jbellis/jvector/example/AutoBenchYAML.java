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
        // Add debug info at the very start
        System.out.println("DEBUG: AutoBenchYAML starting execution");
        
        // Check for --output argument (required for this class)
        String outputPath = null;
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals("--output")) outputPath = args[i+1];
        }

        // Print all arguments for debugging
        System.out.println("DEBUG: Command line arguments:");
        for (int i = 0; i < args.length; i++) {
            System.out.println("  Arg[" + i + "]: " + args[i]);
        }

        if (outputPath == null) {
            logger.error("Error: --output argument is required for AutoBenchYAML");
            System.err.println("Error: --output argument is required for AutoBenchYAML");
            System.exit(1);
        }

        // Force System.out flush to ensure logs are written
        System.out.println("DEBUG: Output path: " + outputPath);
        System.out.flush();
        
        logger.info("Heap space available is {}", Runtime.getRuntime().maxMemory());

        // Filter out --output and its argument from the args
        String finalOutputPath = outputPath;
        String[] filteredArgs = Arrays.stream(args)
                .filter(arg -> !arg.equals("--output") && !arg.equals(finalOutputPath))
                .toArray(String[]::new);

        System.out.println("DEBUG: Filtered arguments: " + Arrays.toString(filteredArgs));
        System.out.flush();

        // generate a regex that matches any regex in filteredArgs, or if filteredArgs is empty/null, match everything
        var regex = filteredArgs.length == 0 ? ".*" : Arrays.stream(filteredArgs).flatMap(s -> Arrays.stream(s.split("\\s"))).map(s -> "(?:" + s + ")").collect(Collectors.joining("|"));
        // compile regex and do substring matching using find
        var pattern = Pattern.compile(regex);

        System.out.println("DEBUG: Using regex pattern: " + regex);
        System.out.flush();

        var datasetNames = getAllDatasetNames().stream().filter(dn -> pattern.matcher(dn).find()).collect(Collectors.toList());

        System.out.println("DEBUG: Dataset names after filtering: " + datasetNames);
        System.out.println("DEBUG: Dataset names size: " + datasetNames.size());
        System.out.flush();

        logger.info("Executing the following datasets: {}", datasetNames);
        List<BenchResult> results = new ArrayList<>();

        // Process datasets from regex patterns
        if (!datasetNames.isEmpty()) {
            System.out.println("DEBUG: Processing datasets from regex patterns");
            System.out.flush();
            for (var datasetName : datasetNames) {
                logger.info("Loading dataset: {}", datasetName);
                System.out.println("DEBUG: Attempting to load dataset: " + datasetName);
                System.out.flush();
                
                try {
                    DataSet ds = DataSetLoader.loadDataSet(datasetName);
                    System.out.println("DEBUG: Dataset loaded successfully: " + datasetName);
                    System.out.flush();
                    
                    logger.info("Dataset loaded: {} with {} vectors", datasetName, ds.baseVectors.size());
                    System.out.println("DEBUG: Dataset has " + ds.baseVectors.size() + " vectors");
                    System.out.flush();

                    if (datasetName.endsWith(".hdf5")) {
                        datasetName = datasetName.substring(0, datasetName.length() - ".hdf5".length());
                    }
                    
                    System.out.println("DEBUG: Getting default config for: " + datasetName);
                    System.out.flush();
                    
                    MultiConfig config = MultiConfig.getDefaultConfig(datasetName);
                    
                    System.out.println("DEBUG: Got config, running benchmark");
                    System.out.flush();
                    
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
                    
                    System.out.println("DEBUG: Benchmark completed for dataset: " + datasetName);
                    System.out.flush();
                } catch (Exception e) {
                    System.err.println("ERROR: Exception while processing dataset " + datasetName);
                    e.printStackTrace();
                    System.err.flush();
                }
            }
        }

        // Process YAML configuration files
        List<String> configNames = Arrays.stream(filteredArgs).filter(s -> s.endsWith(".yml")).collect(Collectors.toList());
        if (!configNames.isEmpty()) {
            System.out.println("DEBUG: Processing YAML configuration files: " + configNames);
            System.out.flush();
            
            for (var configName : configNames) {
                logger.info("Processing configuration file: {}", configName);
                System.out.println("DEBUG: Processing configuration file: " + configName);
                System.out.flush();
                
                try {
                    MultiConfig config = MultiConfig.getConfig(configName);
                    String datasetName = config.dataset;
                    logger.info("Configuration specifies dataset: {}", datasetName);
                    System.out.println("DEBUG: Configuration specifies dataset: " + datasetName);
                    System.out.flush();

                    logger.info("Loading dataset: {}", datasetName);
                    System.out.println("DEBUG: Loading dataset from YAML config: " + datasetName);
                    System.out.flush();
                    
                    DataSet ds = DataSetLoader.loadDataSet(datasetName);
                    logger.info("Dataset loaded: {} with {} vectors", datasetName, ds.baseVectors.size());
                    System.out.println("DEBUG: Dataset loaded from YAML config: " + datasetName + " with " + ds.baseVectors.size() + " vectors");
                    System.out.flush();

                    System.out.println("DEBUG: Running benchmark with YAML config");
                    System.out.flush();
                    
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
                    
                    System.out.println("DEBUG: Benchmark completed for YAML config: " + configName);
                    System.out.flush();
                } catch (Exception e) {
                    System.err.println("ERROR: Exception while processing YAML config " + configName);
                    e.printStackTrace();
                    System.err.flush();
                }
            }
        } else {
            System.out.println("DEBUG: No YAML configuration files to process");
            System.out.flush();
        }

        System.out.println("DEBUG: Benchmark processing completed. Results size: " + results.size());
        System.out.flush();

        // Calculate summary statistics
        try {
            System.out.println("DEBUG: Calculating summary statistics");
            System.out.flush();
            
            SummaryStats stats = BenchmarkSummarizer.summarize(results);
            logger.info("Benchmark summary: {}", stats.toString());
            System.out.println("DEBUG: Benchmark summary: " + stats.toString());
            System.out.flush();

            // Write results to JSON file
            System.out.println("DEBUG: Writing results to JSON file: " + outputPath);
            System.out.flush();
            
            ObjectMapper mapper = new ObjectMapper();
            File outputFile = new File(outputPath);
            mapper.writerWithDefaultPrettyPrinter().writeValue(outputFile, results);
            logger.info("Benchmark results written to {} (file exists: {})", outputPath, outputFile.exists());
            System.out.println("DEBUG: Benchmark results written to " + outputPath + " (file exists: " + outputFile.exists() + ")");
            System.out.flush();
            
            // Double check that the file was created and log its size
            if (outputFile.exists()) {
                logger.info("Output file size: {} bytes", outputFile.length());
                System.out.println("DEBUG: Output file size: " + outputFile.length() + " bytes");
            } else {
                logger.error("Failed to create output file at {}", outputPath);
                System.err.println("ERROR: Failed to create output file at " + outputPath);
            }
            System.out.flush();
        } catch (Exception e) {
            System.err.println("ERROR: Exception during final processing");
            e.printStackTrace();
            System.err.flush();
        }
        
        System.out.println("DEBUG: AutoBenchYAML execution completed");
        System.out.flush();
    }
}
