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
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.DownloadHelper;
import io.github.jbellis.jvector.example.util.Hdf5Loader;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Tests GraphIndexes against vectors from various datasets
 */
public class StudyReachability {
    public static void main(String[] args) throws IOException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        var mGrid = List.of(32); // List.of(16, 24, 32, 48, 64, 96, 128);
        var efConstructionGrid = List.of(100); // List.of(60, 80, 100, 120, 160, 200, 400, 600, 800);
        var topKGrid = List.of(10);
        var overqueryGrid = IntStream.rangeClosed(10, 200).mapToDouble(i -> i * 0.1).boxed().collect(Collectors.toList());
        var neighborOverflowGrid = List.of(1.2f); // List.of(1.2f, 2.0f);
        var usePruningGrid = List.of(false); // List.of(false, true);
        List<Function<DataSet, CompressorParameters>> buildCompression = Arrays.asList(
                __ -> CompressorParameters.NONE
        );
        List<Function<DataSet, CompressorParameters>> searchCompression = Arrays.asList(
                __ -> CompressorParameters.NONE
        );
        List<EnumSet<FeatureId>> featureSets = Arrays.asList(
                EnumSet.of(FeatureId.INLINE_VECTORS)
        );

        // args is list of regexes, possibly needing to be split by whitespace.
        // generate a regex that matches any regex in args, or if args is empty/null, match everything
        var regex = args.length == 0 ? ".*" : Arrays.stream(args).flatMap(s -> Arrays.stream(s.split("\\s"))).map(s -> "(?:" + s + ")").collect(Collectors.joining("|"));
        // compile regex and do substring matching using find
        var pattern = Pattern.compile(regex);

        // large embeddings calculated by Neighborhood Watch.  100k files by default; 1M also available
        var files = List.of(
                "ada002-100k",
                "cohere-english-v3-100k",
                "openai-v3-small-100k",
                "nv-qa-v4-100k",
                "colbert-1M",
                "gecko-100k",
                "glove-25-angular.hdf5",
                "glove-50-angular.hdf5",
                "lastfm-64-dot.hdf5",
                "glove-100-angular.hdf5",
                "glove-200-angular.hdf5",
                "nytimes-256-angular.hdf5",
                "sift-128-euclidean.hdf5"
        );
        execute(files, pattern, buildCompression, featureSets, searchCompression, mGrid, efConstructionGrid, neighborOverflowGrid, topKGrid, overqueryGrid, usePruningGrid);
    }

    private static void execute(List<String> coreFiles, Pattern pattern, List<Function<DataSet, CompressorParameters>> buildCompression, List<EnumSet<FeatureId>> featureSets, List<Function<DataSet, CompressorParameters>> compressionGrid, List<Integer> mGrid, List<Integer> efConstructionGrid, List<Float> neighborOverflowGrid, List<Integer> topKGrid, List<Double> efSearchGrid, List<Boolean> usePruningGrid) throws IOException {
        File file;
        FileOutputStream fos;
        PrintStream ps;

        for (var datasetName : coreFiles) {
            if (pattern.matcher(datasetName).find()) {
                DataSet ds;
                if (datasetName.endsWith(".hdf5")) {
                    DownloadHelper.maybeDownloadHdf5(datasetName);
                    ds = Hdf5Loader.load(datasetName);
                } else {
                    var mfd = DownloadHelper.maybeDownloadFvecs(datasetName);
                    ds = mfd.load();
                }

                for (var addHierarchy : List.of(true, false)) {
                    var addHierarchyGrid = List.of(addHierarchy);

                    PrintStream originalOut = System.out;

                    // run navigable graph
                    file = new File("reachability_study_" + datasetName + "_hierarchy=" + addHierarchy + "_topK=10_navigable.txt");
                    fos = new FileOutputStream(file);
                    ps = new PrintStream(fos);
                    System.setOut(ps);

                    Grid.runAll(ds, mGrid, efConstructionGrid, neighborOverflowGrid, addHierarchyGrid, featureSets, buildCompression, compressionGrid, topKGrid, efSearchGrid, usePruningGrid);

                    ps.close();
                    fos.close();

                    // run reachable graph
                    file = new File("reachability_study_" + datasetName + "_hierarchy=" + addHierarchy + "_topK=10_reachable.txt");
                    fos = new FileOutputStream(file);
                    ps = new PrintStream(fos);
                    System.setOut(ps);

                    // the only difference between Grid and GridReach is that GridReach calls the Reach graphs
                    GridReach.runAll(ds, mGrid, efConstructionGrid, neighborOverflowGrid, addHierarchyGrid, featureSets, buildCompression, compressionGrid, topKGrid, efSearchGrid, usePruningGrid);

                    ps.close();
                    fos.close();

                    System.setOut(originalOut);
                }
            }
        }
    }
}
