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

import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.CompressorParameters.PQParameters;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.DownloadHelper;
import io.github.jbellis.jvector.example.util.Hdf5Loader;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.yaml.snakeyaml.Yaml;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

/**
 * Tests GraphIndexes against vectors from various datasets
 */
public class BenchYAML {
    public static void main(String[] args) throws IOException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        Yaml yaml = new Yaml();
        InputStream inputStream = new FileInputStream("./jvector-examples/src/main/java/io/github/jbellis/jvector/example/example.yml");
        MultiConfig config = yaml.loadAs(inputStream, MultiConfig.class);

        // args is list of regexes, possibly needing to be split by whitespace.
        // generate a regex that matches any regex in args, or if args is empty/null, match everything
        var regex = args.length == 0 ? ".*" : Arrays.stream(args).flatMap(s -> Arrays.stream(s.split("\\s"))).map(s -> "(?:" + s + ")").collect(Collectors.joining("|"));
        // compile regex and do substring matching using find
        var pattern = Pattern.compile(regex);

        // Legend for the datasets:
        // NW: large embeddings calculated by Neighborhood Watch.  100k files by default; 1M also available
        // AB: smaller vectors from ann-benchmarks:
        var coreFiles = List.of(
                "ada002-100k", // NW
                "cohere-english-v3-100k", // NW
                "openai-v3-small-100k", // NW
                "nv-qa-v4-100k", // NW
                "colbert-1M", // NW
                "gecko-100k", // NW
                "openai-v3-large-3072-100k", // NW
                "openai-v3-large-1536-100k", // NW
                "e5-small-v2-100k", // NW
                "e5-base-v2-100k", // NW
                "e5-large-v2-100k", // NW
                "glove-25-angular.hdf5", // AB
                "glove-50-angular.hdf5", // AB
                "lastfm-64-dot.hdf5", // AB
                "glove-100-angular.hdf5", // AB
                "glove-200-angular.hdf5", // AB
                "nytimes-256-angular.hdf5", // AB
                "sift-128-euclidean.hdf5" // AB
                // "deep-image-96-angular.hdf5", // AB, large files not yet supported
                // "gist-960-euclidean.hdf5", // AB, large files not yet supported
        );
        execute(coreFiles, pattern, config);
    }

    private static void execute(List<String> files, Pattern pattern, MultiConfig config) throws IOException {
        for (var datasetName : files) {
            if (pattern.matcher(datasetName).find()) {
                DataSet ds;
                if (datasetName.endsWith(".hdf5")) {
                    DownloadHelper.maybeDownloadHdf5(datasetName);
                    ds = Hdf5Loader.load(datasetName);
                } else {
                    var mfd = DownloadHelper.maybeDownloadFvecs(datasetName);
                    ds = mfd.load();
                }
                Grid.runAll(ds, config.construction.outDegree, config.construction.efConstruction,
                        config.construction.neighborOverflow, config.construction.addHierarchy,
                        config.construction.getFeatureSets(), config.construction.getCompressorParameters(),
                        config.search.getCompressorParameters(), config.search.topKOverquery, config.search.useSearchPruning);
            }
        }
    }
}
