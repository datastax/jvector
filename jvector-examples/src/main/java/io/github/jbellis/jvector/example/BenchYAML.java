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

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.example.reporting.RunArtifacts;
import io.github.jbellis.jvector.example.reporting.SearchReportingCatalog;
import io.github.jbellis.jvector.example.yaml.DatasetCollection;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.example.yaml.RunConfig;

import java.io.IOException;
import java.security.InvalidParameterException;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Tests GraphIndexes against vectors from various datasets
 */
public class BenchYAML {

    private static final org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger(BenchYAML.class);

    public static void main(String[] args) throws IOException {
        // args is one of:
        // - a list of regexes, possibly needing to be split by whitespace.
        // - a list of YAML files
        // defensively create an argv regex, to avoid NPE from possibly malformed argv from maven exec:java
        if (args == null) {
            throw new InvalidParameterException("argv[] is null, check your maven exec config");
        }

            throw new InvalidParameterException("argv[] is null, check your maven exec config");
        }
        String regex = Arrays.stream(args)
                .filter(Objects::nonNull)
                .flatMap(s -> Arrays.stream(s.split("\\s")))
                .filter(s -> !s.isEmpty())
                .collect(Collectors.joining("|"));
        var pattern = Pattern.compile(regex.isEmpty() ? ".*" : regex);

        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        var datasetCollection = DatasetCollection.load();
        var datasetNames = datasetCollection.getAll().stream().filter(dn -> pattern.matcher(dn).find()).collect(Collectors.toList());

        List<MultiConfig> allConfigs = new ArrayList<>();

        if (!datasetNames.isEmpty()) {
            System.out.println("Executing the following datasets: " +
                    wrapList(new java.util.ArrayList<>(datasetNames), 6, "  "));

            final String HDF5 = ".hdf5";
            for (String raw : datasetNames) {
                String name = raw.endsWith(HDF5) ? raw.substring(0, raw.length() - HDF5.length()) : raw;
                allConfigs.add(MultiConfig.getDefaultConfig(name));
            }

            }

            MultiConfig.printDefaultConfigUsageSummary();
        }

        // get the list of YAML files from args
        List<String> configNames = (args == null) ? Collections.emptyList() : Arrays.stream(args)
                .filter(Objects::nonNull)
                .filter(s -> s.endsWith(".yml"))
                .collect(Collectors.toList());

        if (!configNames.isEmpty()) {
            for (var configName : configNames) {
                MultiConfig config = MultiConfig.getDefaultConfig(configName);
                allConfigs.add(config);
            }
        }

        if (allConfigs.isEmpty()) {
            System.out.println("No datasets matched. Exiting.");
            return;
        }

        // Set up reporting for run
        RunArtifacts artifacts;
        try {
            RunConfig runCfg = RunConfig.loadDefault();
            artifacts = RunArtifacts.open(runCfg, allConfigs);
        } catch (java.io.FileNotFoundException e) {
            // Legacy yamlSchemaVersion "0" behavior: no run.yml
            // - logging disabled
            // - console shows compute selection
            // - compute selection comes from legacy search.benchmarks if present, else default
            System.err.println("WARNING: run.yml not found. Falling back to deprecated legacy behavior: "
                    + "no logging, console mirrors computed benchmarks.");

            Map<String, List<String>> legacyBenchmarks = null;
            for (MultiConfig c : allConfigs) {
                if (c != null && c.legacySearchBenchmarks != null && !c.legacySearchBenchmarks.isEmpty()) {
                    legacyBenchmarks = c.legacySearchBenchmarks;
                    break;
                }
            }
            if (legacyBenchmarks == null) {
                legacyBenchmarks = SearchReportingCatalog.defaultComputeBenchmarks();
            }
            artifacts = RunArtifacts.legacyNoLogging(legacyBenchmarks);
        }


        for (var config : allConfigs) {

            String datasetName = config.dataset;
            DataSet ds = DataSets.loadDataSet(datasetName).orElseThrow(
                    () -> new RuntimeException("Could not load dataset:" + datasetName)
            );
            // Register dataset info the first time we actually load the dataset for benchmarking
            artifacts.registerDataset(datasetName, ds);

            Grid.runAll(ds,
                    config.construction.useSavedIndexIfExists,
                    config.construction.outDegree,
                    config.construction.efConstruction,
                    config.construction.neighborOverflow,
                    config.construction.addHierarchy,
                    config.construction.refineFinalGraph,
                    config.construction.getFeatureSets(),
                    config.construction.getCompressorParameters(),
                    config.compaction.numSplits,
                    config.search.getCompressorParameters(),
                    config.search.topKOverquery,
                    config.search.useSearchPruning,
                    artifacts);
        }
    }

    private static String wrapList(java.util.List<String> items, int perLine, String indent) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) sb.append(", ");
            if (i > 0 && (i % perLine) == 0) sb.append(System.lineSeparator()).append(indent);
            sb.append(items.get(i));
        }
        sb.append("]");
        return sb.toString();
    }
}
