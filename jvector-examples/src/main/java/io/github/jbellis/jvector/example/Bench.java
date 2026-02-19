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
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.example.yaml.DatasetCollection;
import io.github.jbellis.jvector.example.yaml.TestDataPartition;
import io.github.jbellis.jvector.example.yaml.TestDataPartition.Distribution;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.IOException;
import java.util.*;
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
        var parsedArgs = parseArgs(args);

        // When enabled, caches built indices for reuse in future runs.
        // Useful for large indexes and repeated testing.
        boolean enableIndexCache = false;

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
                        ds.getSimilarityFunction() == VectorSimilarityFunction.EUCLIDEAN,
                        UNWEIGHTED),
                __ -> CompressorParameters.NONE
        );
        List<Function<DataSet, CompressorParameters>> searchCompression = Arrays.asList(
                __ -> CompressorParameters.NONE,
                // ds -> new CompressorParameters.BQParameters(),
                ds -> new PQParameters(ds.getDimension() / 8,
                        256,
                        ds.getSimilarityFunction() == VectorSimilarityFunction.EUCLIDEAN,
                        UNWEIGHTED)
        );
        List<EnumSet<FeatureId>> featureSets = Arrays.asList(
                EnumSet.of(FeatureId.NVQ_VECTORS),
                EnumSet.of(FeatureId.NVQ_VECTORS, FeatureId.FUSED_PQ),
                EnumSet.of(FeatureId.INLINE_VECTORS)
        );
        execute(parsedArgs.datasetPattern, enableIndexCache, buildCompression, featureSets, parsedArgs.partitions, searchCompression, mGrid, efConstructionGrid, neighborOverflowGrid, addHierarchyGrid, refineFinalGraphGrid, topKGrid, usePruningGrid);
    }

    private static void execute(Pattern pattern, boolean enableIndexCache, List<Function<DataSet, CompressorParameters>> buildCompression, List<EnumSet<FeatureId>> featureSets, TestDataPartition partitions, List<Function<DataSet, CompressorParameters>> compressionGrid, List<Integer> mGrid, List<Integer> efConstructionGrid, List<Float> neighborOverflowGrid, List<Boolean> addHierarchyGrid, List<Boolean> refineFinalGraphGrid, Map<Integer, List<Double>> topKGrid, List<Boolean> usePruningGrid) throws IOException {
        var datasetCollection = DatasetCollection.load();
        var datasetNames = datasetCollection.getAll().stream().filter(dn -> pattern.matcher(dn).find()).collect(Collectors.toList());
        System.out.println("Executing the following datasets: " + datasetNames);

        for (var datasetName : datasetNames) {
            DataSet ds = DataSets.loadDataSet(datasetName).orElseThrow(
                    () -> new RuntimeException("Dataset " + datasetName + " not found")
            );
            Grid.runAll(ds, enableIndexCache, mGrid, efConstructionGrid, neighborOverflowGrid, addHierarchyGrid, refineFinalGraphGrid, featureSets, buildCompression, partitions, compressionGrid, topKGrid, usePruningGrid);
        }
    }

    static ParsedArgs parseArgs(String[] args) {
        List<String> tokens = Arrays.stream(args == null ? new String[0] : args)
                .filter(Objects::nonNull)
                .flatMap(s -> Arrays.stream(s.split("\\s+")))
                .filter(s -> !s.isEmpty())
                .collect(Collectors.toList());

        var regexTokens = new ArrayList<String>();
        var splitCounts = new ArrayList<Integer>();
        var splitDistributions = new ArrayList<Distribution>();

        for (int i = 0; i < tokens.size(); i++) {
            String token = tokens.get(i);
            switch (token) {
                case "--num-splits":
                    i = requireAndParseSplitCounts(tokens, i, splitCounts);
                    break;
                case "--split-distribution":
                    i = requireAndParseSplitDistributions(tokens, i, splitDistributions);
                    break;
                default:
                    regexTokens.add(token);
            }
        }

        var regex = regexTokens.isEmpty()
                ? ".*"
                : regexTokens.stream().map(s -> "(?:" + s + ")").collect(Collectors.joining("|"));

        var partitions = new TestDataPartition();
        if (!splitCounts.isEmpty()) {
            partitions.numSplits = splitCounts;
        }
        if (!splitDistributions.isEmpty()) {
            partitions.splitDistribution = splitDistributions;
        }

        return new ParsedArgs(Pattern.compile(regex), partitions);
    }

    private static int requireAndParseSplitCounts(List<String> tokens, int optionIndex, List<Integer> splitCounts) {
        if (optionIndex + 1 >= tokens.size()) {
            throw new IllegalArgumentException("Missing value for --num-splits");
        }
        int sizeBefore = splitCounts.size();
        String value = tokens.get(optionIndex + 1);
        for (String raw : value.split(",")) {
            String trimmed = raw.trim();
            if (trimmed.isEmpty()) {
                continue;
            }
            int parsed = Integer.parseInt(trimmed);
            if (parsed <= 0) {
                throw new IllegalArgumentException("--num-splits values must be positive: " + parsed);
            }
            splitCounts.add(parsed);
        }
        if (splitCounts.size() == sizeBefore) {
            throw new IllegalArgumentException("No valid values provided for --num-splits");
        }
        return optionIndex + 1;
    }

    private static int requireAndParseSplitDistributions(List<String> tokens, int optionIndex, List<Distribution> splitDistributions) {
        if (optionIndex + 1 >= tokens.size()) {
            throw new IllegalArgumentException("Missing value for --split-distribution");
        }
        int sizeBefore = splitDistributions.size();
        String value = tokens.get(optionIndex + 1);
        for (String raw : value.split(",")) {
            String trimmed = raw.trim();
            if (trimmed.isEmpty()) {
                continue;
            }
            try {
                splitDistributions.add(Distribution.valueOf(trimmed.toUpperCase(Locale.ROOT)));
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("Invalid --split-distribution value '" + trimmed
                        + "'. Expected one of " + Arrays.toString(Distribution.values()), e);
            }
        }
        if (splitDistributions.size() == sizeBefore) {
            throw new IllegalArgumentException("No valid values provided for --split-distribution");
        }
        return optionIndex + 1;
    }

    static final class ParsedArgs {
        final Pattern datasetPattern;
        final TestDataPartition partitions;

        ParsedArgs(Pattern datasetPattern, TestDataPartition partitions) {
            this.datasetPattern = datasetPattern;
            this.partitions = partitions;
        }
    }
}
