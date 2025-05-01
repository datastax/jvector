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

import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.DownloadHelper;
import io.github.jbellis.jvector.example.util.Hdf5Loader;
import io.github.jbellis.jvector.example.yaml.DatasetCollection;
import io.github.jbellis.jvector.example.yaml.MultiConfig;

import java.io.IOException;
import java.util.Arrays;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Tests GraphIndexes against vectors from various datasets
 */
public class BenchYAML {
    public static void main(String[] args) throws IOException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        // args is list of regexes, possibly needing to be split by whitespace.
        // generate a regex that matches any regex in args, or if args is empty/null, match everything
        var regex = args.length == 0 ? ".*" : Arrays.stream(args).flatMap(s -> Arrays.stream(s.split("\\s"))).map(s -> "(?:" + s + ")").collect(Collectors.joining("|"));
        // compile regex and do substring matching using find
        var pattern = Pattern.compile(regex);

        execute(pattern);
    }

    private static void execute(Pattern pattern) throws IOException {
        var datasets = DatasetCollection.load();
        var datasetNames = datasets.getAll().stream().filter(dn -> pattern.matcher(dn).find()).collect(Collectors.toList());
        System.out.println("Executing the following datasets: " + datasetNames);

        for (var datasetName : datasetNames) {
            DataSet ds;
            if (datasetName.endsWith(".hdf5")) {
                DownloadHelper.maybeDownloadHdf5(datasetName);
                ds = Hdf5Loader.load(datasetName);
                datasetName = datasetName.substring(0, datasetName.length() - ".hdf5".length());
            } else {
                var mfd = DownloadHelper.maybeDownloadFvecs(datasetName);
                ds = mfd.load();
            }

            MultiConfig config = MultiConfig.getConfig(datasetName);

            Grid.runAll(ds, config.construction.outDegree, config.construction.efConstruction,
                    config.construction.neighborOverflow, config.construction.addHierarchy,
                    config.construction.getFeatureSets(), config.construction.getCompressorParameters(),
                    config.search.getCompressorParameters(), config.search.topKOverquery, config.search.useSearchPruning);
        }
    }
}
