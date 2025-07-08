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

import io.github.jbellis.jvector.example.dynamic.ClusteredDynamicDataset;
import io.github.jbellis.jvector.example.dynamic.DynamicDataset;
import io.github.jbellis.jvector.example.util.Dataset;
import io.github.jbellis.jvector.example.util.DataSetLoader;
import io.github.jbellis.jvector.example.yaml.DatasetCollection;
import io.github.jbellis.jvector.example.yaml.MultiConfig;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Tests GraphIndexes against vectors from various datasets
 */
public class BenchDynamic {
    public static void main(String[] args) throws IOException {
        // args is one of:
        // - a list of regexes, possibly needing to be split by whitespace.
        // - a list of YAML files

        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        // generate a regex that matches any regex in args, or if args is empty/null, match everything
        var regex = args.length == 0 ? ".*" : Arrays.stream(args).flatMap(s -> Arrays.stream(s.split("\\s"))).map(s -> "(?:" + s + ")").collect(Collectors.joining("|"));
        // compile regex and do substring matching using find
        var pattern = Pattern.compile(regex);

        var datasetCollection = DatasetCollection.load();
        var datasetNames = datasetCollection.getAll().stream().filter(dn -> pattern.matcher(dn).find()).collect(Collectors.toList());

        List<MultiConfig> configurations = new ArrayList<>();

        if (!datasetNames.isEmpty()) {
            for (var datasetName : datasetNames) {
                if (datasetName.endsWith(".hdf5")) {
                    datasetName = datasetName.substring(0, datasetName.length() - ".hdf5".length());
                }
                MultiConfig config = MultiConfig.getDefaultConfig(datasetName);
                configurations.add(config);
            }
        }

        // get the list of YAML files from args
        List<String> configNames = Arrays.stream(args).filter(s -> s.endsWith(".yml")).collect(Collectors.toList());

        if (!configNames.isEmpty()) {
            for (var configName : configNames) {
                MultiConfig config = MultiConfig.getConfig(configName);
                configurations.add(config);
            }
        }

        System.out.println("Executing the following datasets:" + configurations.stream().map(c -> " " + c.dataset));

        for (var config : configurations) {
            String datasetName = config.dataset;
            Dataset ds = DataSetLoader.loadDataSet(datasetName);

            DynamicDataset dds = ClusteredDynamicDataset.create(ds, 10, 1, 100);

            GridDynamic.runAll(dds, config.construction.outDegree, config.construction.efConstruction,
                    config.construction.neighborOverflow, config.construction.addHierarchy, config.construction.refineFinalGraph,
                    config.construction.getFeatureSets(), config.construction.getCompressorParameters(),
                    config.search.getCompressorParameters(), config.search.topKOverquery, config.search.useSearchPruning);
        }
    }
}
