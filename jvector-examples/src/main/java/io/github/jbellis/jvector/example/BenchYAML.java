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
import io.github.jbellis.jvector.example.yaml.DatasetCollection;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Tests GraphIndexes against vectors from various datasets
 */
public class BenchYAML {
    public static void main(String[] args) throws IOException {
        // Check for --output argument
        String outputPath = null;
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals("--output")) outputPath = args[i+1];
        }

        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        // Filter out --output and its argument from the args
        String finalOutputPath = outputPath;
        String[] filteredArgs = Arrays.stream(args)
                .filter(arg -> !arg.equals("--output") && (finalOutputPath == null || !arg.equals(finalOutputPath)))
                .toArray(String[]::new);

        // generate a regex that matches any regex in filteredArgs, or if filteredArgs is empty/null, match everything
        var regex = filteredArgs.length == 0 ? ".*" : Arrays.stream(filteredArgs).flatMap(s -> Arrays.stream(s.split("\\s"))).map(s -> "(?:" + s + ")").collect(Collectors.joining("|"));
        // compile regex and do substring matching using find
        var pattern = Pattern.compile(regex);

        var datasetCollection = DatasetCollection.load();
        var datasetNames = datasetCollection.getAll().stream().filter(dn -> pattern.matcher(dn).find()).collect(Collectors.toList());

        if (outputPath == null) {
            // Original functionality when --output is not provided
            if (!datasetNames.isEmpty()) {
                System.out.println("Executing the following datasets: " + datasetNames);

                for (var datasetName : datasetNames) {
                    DataSet ds = DataSetLoader.loadDataSet(datasetName);

                    if (datasetName.endsWith(".hdf5")) {
                        datasetName = datasetName.substring(0, datasetName.length() - ".hdf5".length());
                    }
                    MultiConfig config = MultiConfig.getDefaultConfig(datasetName);

                    Grid.runAll(ds, config.construction.outDegree, config.construction.efConstruction,
                            config.construction.neighborOverflow, config.construction.addHierarchy,
                            config.construction.getFeatureSets(), config.construction.getCompressorParameters(),
                            config.search.getCompressorParameters(), config.search.topKOverquery, config.search.useSearchPruning);
                }
            }

            // get the list of YAML files from filteredArgs
            List<String> configNames = Arrays.stream(filteredArgs).filter(s -> s.endsWith(".yml")).collect(Collectors.toList());

            if (!configNames.isEmpty()) {
                for (var configName : configNames) {
                    MultiConfig config = MultiConfig.getConfig(configName);
                    String datasetName = config.dataset;

                    DataSet ds = DataSetLoader.loadDataSet(datasetName);

                    Grid.runAll(ds, config.construction.outDegree, config.construction.efConstruction,
                            config.construction.neighborOverflow, config.construction.addHierarchy,
                            config.construction.getFeatureSets(), config.construction.getCompressorParameters(),
                            config.search.getCompressorParameters(), config.search.topKOverquery, config.search.useSearchPruning);
                }
            }
        } else {
            // New functionality when --output is provided
            System.out.println("Executing the following datasets: " + datasetNames);
            List<BenchResult> results = new ArrayList<>();

            // Process datasets from regex patterns
            if (!datasetNames.isEmpty()) {
                for (var datasetName : datasetNames) {
                    DataSet ds = DataSetLoader.loadDataSet(datasetName);

                    if (datasetName.endsWith(".hdf5")) {
                        datasetName = datasetName.substring(0, datasetName.length() - ".hdf5".length());
                    }
                    MultiConfig config = MultiConfig.getDefaultConfig(datasetName);

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
                }
            }

            // Process YAML configuration files
            List<String> configNames = Arrays.stream(filteredArgs).filter(s -> s.endsWith(".yml")).collect(Collectors.toList());
            if (!configNames.isEmpty()) {
                for (var configName : configNames) {
                    MultiConfig config = MultiConfig.getConfig(configName);
                    String datasetName = config.dataset;

                    DataSet ds = DataSetLoader.loadDataSet(datasetName);

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
                }
            }

            // Calculate summary statistics
            SummaryStats stats = BenchmarkSummarizer.summarize(results);
            System.out.println(stats.toString());

            // Write results to JSON file
            ObjectMapper mapper = new ObjectMapper();
            mapper.writerWithDefaultPrettyPrinter().writeValue(new File(outputPath), results);
            System.out.println("Benchmark results written to " + outputPath);
        }
    }
}
