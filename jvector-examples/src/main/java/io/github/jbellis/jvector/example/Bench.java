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
import io.github.jbellis.jvector.example.util.DataSetLoader;
import io.github.jbellis.jvector.example.yaml.DatasetCollection;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.IOException;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

/**
 * Tests GraphIndexes against vectors from various datasets
 */
public class Bench {
    public static void main(String[] args) throws IOException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        var mGrid = List.of(32); // List.of(16, 24, 32, 48, 64, 96, 128);
        var efConstructionGrid = List.of(100); // List.of(60, 80, 100, 120, 160, 200, 400, 600, 800);
        var topKGrid = Map.of(
                10, // topK
                List.of(1.0, 2.0, 5.0, 10.0), // oq
                100, // topK
                List.of(1.0, 2.0) // oq
        ); // rerankK = oq * topK
        var neighborOverflowGrid = List.of(1.2f); // List.of(1.2f, 2.0f);
        var addHierarchyGrid = List.of(true); // List.of(false, true);
        var refineFinalGraphGrid = List.of(true); // List.of(false, true);
        var usePruningGrid = List.of(true); // List.of(false, true);
        List<Function<DataSet, CompressorParameters>> buildCompression = Arrays.asList(
                ds -> new PQParameters(ds.getDimension() / 8,
                        256,
                        ds.similarityFunction == VectorSimilarityFunction.EUCLIDEAN,
                        UNWEIGHTED),
                __ -> CompressorParameters.NONE
        );
        List<Function<DataSet, CompressorParameters>> searchCompression = Arrays.asList(
                __ -> CompressorParameters.NONE,
                // ds -> new CompressorParameters.BQParameters(),
                ds -> new PQParameters(ds.getDimension() / 8,
                        256,
                        ds.similarityFunction == VectorSimilarityFunction.EUCLIDEAN,
                        UNWEIGHTED)
        );
        List<EnumSet<FeatureId>> featureSets = Arrays.asList(
                EnumSet.of(FeatureId.NVQ_VECTORS),
//                EnumSet.of(FeatureId.NVQ_VECTORS, FeatureId.FUSED_ADC),
                EnumSet.of(FeatureId.INLINE_VECTORS)
        );

        // args is list of regexes, possibly needing to be split by whitespace.
        // generate a regex that matches any regex in args, or if args is empty/null, match everything
        var regex = args.length == 0 ? ".*" : Arrays.stream(args).flatMap(s -> Arrays.stream(s.split("\\s"))).map(s -> "(?:" + s + ")").collect(Collectors.joining("|"));
        // compile regex and do substring matching using find
        var pattern = Pattern.compile(regex);

        execute(pattern, buildCompression, featureSets, searchCompression, mGrid, efConstructionGrid, neighborOverflowGrid, addHierarchyGrid, refineFinalGraphGrid, topKGrid, usePruningGrid);
    }

    private static void execute(Pattern pattern, List<Function<DataSet, CompressorParameters>> buildCompression, List<EnumSet<FeatureId>> featureSets, List<Function<DataSet, CompressorParameters>> compressionGrid, List<Integer> mGrid, List<Integer> efConstructionGrid, List<Float> neighborOverflowGrid, List<Boolean> addHierarchyGrid, List<Boolean> refineFinalGraphGrid, Map<Integer, List<Double>> topKGrid, List<Boolean> usePruningGrid) throws IOException {
        var datasetCollection = DatasetCollection.load();
        var datasetNames = datasetCollection.getAll().stream().filter(dn -> pattern.matcher(dn).find()).collect(Collectors.toList());
        System.out.println("Executing the following datasets: " + datasetNames);

        for (var datasetName : datasetNames) {
            DataSet ds = DataSetLoader.loadDataSet(datasetName);
            Grid.runAll(ds, mGrid, efConstructionGrid, neighborOverflowGrid, addHierarchyGrid, refineFinalGraphGrid, featureSets, buildCompression, compressionGrid, topKGrid, usePruningGrid);
        }
    }
}
