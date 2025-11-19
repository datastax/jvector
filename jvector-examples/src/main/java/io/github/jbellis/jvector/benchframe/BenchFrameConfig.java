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

package io.github.jbellis.jvector.benchframe;

import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.CompressorParameters.PQParameters;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.util.*;
import java.util.function.Function;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

/**
 * Typesafe configuration class for benchmark execution. Provides a unified, immutable configuration
 * model that encapsulates all parameters needed to run a benchmark including graph construction
 * parameters, search parameters, and feature sets.
 * <p>
 * This class can be used programmatically through its {@link Builder} or constructed from
 * YAML-based {@link MultiConfig} using {@link #fromMultiConfig(MultiConfig)}.
 * <p>
 * All collections returned by getter methods are unmodifiable to maintain immutability.
 *
 * <h2>Usage Examples</h2>
 * <pre>{@code
 * // Create from YAML MultiConfig
 * MultiConfig yaml = MultiConfig.getDefaultConfig("dataset-name");
 * BenchFrameConfig config = BenchFrameConfig.fromMultiConfig(yaml);
 *
 * // Create with Builder
 * BenchFrameConfig config = new BenchFrameConfig.Builder()
 *     .withDatasetName("my-dataset")
 *     .withMGrid(List.of(16, 32, 64))
 *     .withEfConstructionGrid(List.of(100, 200))
 *     .build();
 *
 * // Use default Bench-style configuration
 * BenchFrameConfig defaults = BenchFrameConfig.createBenchDefaults();
 * }</pre>
 *
 * @see MultiConfig
 * @see BenchFrame
 */
public class BenchFrameConfig {
    // Dataset identification
    private final String datasetName;

    // Graph construction parameters
    private final List<Integer> mGrid;
    private final List<Integer> efConstructionGrid;
    private final List<Float> neighborOverflowGrid;
    private final List<Boolean> addHierarchyGrid;
    private final List<Boolean> refineFinalGraphGrid;
    private final List<? extends Set<FeatureId>> featureSets;
    private final List<Function<DataSet, CompressorParameters>> buildCompressors;

    // Search parameters
    private final List<Function<DataSet, CompressorParameters>> searchCompressors;
    private final Map<Integer, List<Double>> topKOverqueryGrid;
    private final List<Boolean> usePruningGrid;

    // Benchmark selection
    private final Map<String, List<String>> benchmarkSpec;

    // Result collection mode
    private final boolean collectResults;

    private BenchFrameConfig(Builder builder) {
        this.datasetName = builder.datasetName;
        this.mGrid = Collections.unmodifiableList(builder.mGrid);
        this.efConstructionGrid = Collections.unmodifiableList(builder.efConstructionGrid);
        this.neighborOverflowGrid = Collections.unmodifiableList(builder.neighborOverflowGrid);
        this.addHierarchyGrid = Collections.unmodifiableList(builder.addHierarchyGrid);
        this.refineFinalGraphGrid = Collections.unmodifiableList(builder.refineFinalGraphGrid);
        this.featureSets = Collections.unmodifiableList(builder.featureSets);
        this.buildCompressors = Collections.unmodifiableList(builder.buildCompressors);
        this.searchCompressors = Collections.unmodifiableList(builder.searchCompressors);
        this.topKOverqueryGrid = Collections.unmodifiableMap(builder.topKOverqueryGrid);
        this.usePruningGrid = Collections.unmodifiableList(builder.usePruningGrid);
        this.benchmarkSpec = builder.benchmarkSpec == null ? null : Collections.unmodifiableMap(builder.benchmarkSpec);
        this.collectResults = builder.collectResults;
    }

    /**
     * Returns the dataset name associated with this configuration.
     *
     * @return the dataset name, may be null if not specified
     */
    public String getDatasetName() { return datasetName; }

    /**
     * Returns the grid of M (max connections per node) values to test.
     *
     * @return unmodifiable list of M values
     */
    public List<Integer> getMGrid() { return mGrid; }

    /**
     * Returns the grid of efConstruction values to test during graph construction.
     *
     * @return unmodifiable list of efConstruction values
     */
    public List<Integer> getEfConstructionGrid() { return efConstructionGrid; }

    /**
     * Returns the grid of neighbor overflow multipliers to test. This controls how many
     * candidate neighbors are considered relative to M during graph construction.
     *
     * @return unmodifiable list of neighbor overflow multipliers
     */
    public List<Float> getNeighborOverflowGrid() { return neighborOverflowGrid; }

    /**
     * Returns the grid of add hierarchy boolean values indicating whether to use hierarchical
     * graph construction.
     *
     * @return unmodifiable list of boolean values
     */
    public List<Boolean> getAddHierarchyGrid() { return addHierarchyGrid; }

    /**
     * Returns the grid of refine final graph boolean values indicating whether to perform
     * final graph refinement after construction.
     *
     * @return unmodifiable list of boolean values
     */
    public List<Boolean> getRefineFinalGraphGrid() { return refineFinalGraphGrid; }

    /**
     * Returns the feature sets to test. Each set contains {@link FeatureId}s that enable
     * specific features like inline vectors or NVQ vectors.
     *
     * @return unmodifiable list of feature sets
     */
    public List<? extends Set<FeatureId>> getFeatureSets() { return featureSets; }

    /**
     * Returns the compressor functions to use during graph construction. Each function takes
     * a {@link DataSet} and returns appropriate {@link CompressorParameters}.
     *
     * @return unmodifiable list of compressor parameter functions
     */
    public List<Function<DataSet, CompressorParameters>> getBuildCompressors() { return buildCompressors; }

    /**
     * Returns the compressor functions to use during search. Each function takes
     * a {@link DataSet} and returns appropriate {@link CompressorParameters}.
     *
     * @return unmodifiable list of compressor parameter functions
     */
    public List<Function<DataSet, CompressorParameters>> getSearchCompressors() { return searchCompressors; }

    /**
     * Returns the grid of topK overquery multipliers mapped by K value. For example,
     * a map entry of (10, [1.0, 2.0, 5.0]) means for top-10 queries, test overquery
     * factors of 1.0x, 2.0x, and 5.0x.
     *
     * @return unmodifiable map of K values to overquery multipliers
     */
    public Map<Integer, List<Double>> getTopKOverqueryGrid() { return topKOverqueryGrid; }

    /**
     * Returns the grid of boolean values indicating whether to use search pruning.
     *
     * @return unmodifiable list of boolean values
     */
    public List<Boolean> getUsePruningGrid() { return usePruningGrid; }

    /**
     * Returns the benchmark specification mapping benchmark types to their configurations.
     * A null value indicates all default benchmarks should be run.
     *
     * @return unmodifiable map of benchmark specifications, or null for default benchmarks
     */
    public Map<String, List<String>> getBenchmarkSpec() { return benchmarkSpec; }

    /**
     * Returns whether results should be collected and returned from benchmark execution.
     *
     * @return true if results should be collected, false otherwise
     */
    public boolean shouldCollectResults() { return collectResults; }

    /**
     * Creates a new {@link Builder} initialized with this configuration's values.
     * This is useful for creating modified copies of existing configurations.
     *
     * @return a new Builder with this configuration's values
     */
    public Builder toBuilder() {
        return new Builder()
            .withDatasetName(datasetName)
            .withMGrid(mGrid)
            .withEfConstructionGrid(efConstructionGrid)
            .withNeighborOverflowGrid(neighborOverflowGrid)
            .withAddHierarchyGrid(addHierarchyGrid)
            .withRefineFinalGraphGrid(refineFinalGraphGrid)
            .withFeatureSets(featureSets)
            .withBuildCompressors(buildCompressors)
            .withSearchCompressors(searchCompressors)
            .withTopKOverqueryGrid(topKOverqueryGrid)
            .withUsePruningGrid(usePruningGrid)
            .withBenchmarkSpec(benchmarkSpec)
            .collectResults(collectResults);
    }

    /**
     * Creates a BenchFrameConfig from a YAML-based {@link MultiConfig}. This factory method
     * provides compatibility with the existing YAML configuration system.
     *
     * @param config the MultiConfig to convert
     * @return a new BenchFrameConfig with values from the MultiConfig
     */
    public static BenchFrameConfig fromMultiConfig(MultiConfig config) {
        return new Builder()
            .withDatasetName(config.dataset)
            .withMGrid(config.construction.outDegree)
            .withEfConstructionGrid(config.construction.efConstruction)
            .withNeighborOverflowGrid(config.construction.neighborOverflow)
            .withAddHierarchyGrid(config.construction.addHierarchy)
            .withRefineFinalGraphGrid(config.construction.refineFinalGraph)
            .withFeatureSets(config.construction.getFeatureSets())
            .withBuildCompressors(config.construction.getCompressorParameters())
            .withSearchCompressors(config.search.getCompressorParameters())
            .withTopKOverqueryGrid(config.search.topKOverquery)
            .withUsePruningGrid(config.search.useSearchPruning)
            .withBenchmarkSpec(config.search.benchmarks)
            .build();
    }

    /**
     * Creates a default configuration matching the original Bench.java's hardcoded parameters.
     * This provides a baseline configuration suitable for most benchmark scenarios.
     * <p>
     * Default values include:
     * <ul>
     *   <li>M: 32</li>
     *   <li>efConstruction: 100</li>
     *   <li>neighborOverflow: 1.2</li>
     *   <li>addHierarchy: true</li>
     *   <li>refineFinalGraph: true</li>
     *   <li>usePruning: true</li>
     *   <li>topK overquery: 10 -&gt; [1.0, 2.0, 5.0, 10.0], 100 -&gt; [1.0, 2.0]</li>
     *   <li>Feature sets: NVQ_VECTORS and INLINE_VECTORS</li>
     *   <li>Compressors: PQ for build, both none and PQ for search</li>
     * </ul>
     *
     * @return a new BenchFrameConfig with default Bench.java values
     */
    public static BenchFrameConfig createBenchDefaults() {
        return new Builder()
            .withMGrid(List.of(32))
            .withEfConstructionGrid(List.of(100))
            .withNeighborOverflowGrid(List.of(1.2f))
            .withAddHierarchyGrid(List.of(true))
            .withRefineFinalGraphGrid(List.of(true))
            .withUsePruningGrid(List.of(true))
            .withTopKOverqueryGrid(Map.of(
                10, List.of(1.0, 2.0, 5.0, 10.0),
                100, List.of(1.0, 2.0)
            ))
            .withFeatureSets(Arrays.asList(
                EnumSet.of(FeatureId.NVQ_VECTORS),
                EnumSet.of(FeatureId.INLINE_VECTORS)
            ))
            .withBuildCompressors(Arrays.asList(
                ds -> new PQParameters(ds.getDimension() / 8,
                        256,
                        ds.getSimilarityFunction() == VectorSimilarityFunction.EUCLIDEAN,
                        UNWEIGHTED),
                __ -> CompressorParameters.NONE
            ))
            .withSearchCompressors(Arrays.asList(
                __ -> CompressorParameters.NONE,
                ds -> new PQParameters(ds.getDimension() / 8,
                        256,
                        ds.getSimilarityFunction() == VectorSimilarityFunction.EUCLIDEAN,
                        UNWEIGHTED)
            ))
            .build();
    }

    /**
     * Builder for fluent BenchFrameConfig construction. All builder methods return the builder
     * instance for method chaining. Collections provided to builder methods are defensively
     * copied to prevent external modification.
     * <p>
     * Default values provide sensible single-value grids:
     * <ul>
     *   <li>mGrid: [32]</li>
     *   <li>efConstructionGrid: [100]</li>
     *   <li>neighborOverflowGrid: [1.2]</li>
     *   <li>addHierarchyGrid: [true]</li>
     *   <li>refineFinalGraphGrid: [true]</li>
     *   <li>featureSets: [INLINE_VECTORS]</li>
     *   <li>buildCompressors: [NONE]</li>
     *   <li>searchCompressors: [NONE]</li>
     *   <li>topKOverqueryGrid: {10: [1.0]}</li>
     *   <li>usePruningGrid: [true]</li>
     *   <li>benchmarkSpec: null (use default benchmarks)</li>
     *   <li>collectResults: false</li>
     * </ul>
     */
    public static class Builder {
        private String datasetName;
        private List<Integer> mGrid = List.of(32);
        private List<Integer> efConstructionGrid = List.of(100);
        private List<Float> neighborOverflowGrid = List.of(1.2f);
        private List<Boolean> addHierarchyGrid = List.of(true);
        private List<Boolean> refineFinalGraphGrid = List.of(true);
        private List<? extends Set<FeatureId>> featureSets = List.of(EnumSet.of(FeatureId.INLINE_VECTORS));
        private List<Function<DataSet, CompressorParameters>> buildCompressors =
            List.of(__ -> CompressorParameters.NONE);
        private List<Function<DataSet, CompressorParameters>> searchCompressors =
            List.of(__ -> CompressorParameters.NONE);
        private Map<Integer, List<Double>> topKOverqueryGrid = Map.of(10, List.of(1.0));
        private List<Boolean> usePruningGrid = List.of(true);
        private Map<String, List<String>> benchmarkSpec = null; // null means use default benchmarks
        private boolean collectResults = false;

        /**
         * Sets the dataset name.
         *
         * @param datasetName the dataset name to associate with this configuration
         * @return this builder for method chaining
         */
        public Builder withDatasetName(String datasetName) {
            this.datasetName = datasetName;
            return this;
        }

        /**
         * Sets the grid of M (max connections per node) values to test.
         *
         * @param mGrid list of M values, defensively copied
         * @return this builder for method chaining
         */
        public Builder withMGrid(List<Integer> mGrid) {
            this.mGrid = new ArrayList<>(mGrid);
            return this;
        }

        /**
         * Sets the grid of efConstruction values to test during graph construction.
         *
         * @param efConstructionGrid list of efConstruction values, defensively copied
         * @return this builder for method chaining
         */
        public Builder withEfConstructionGrid(List<Integer> efConstructionGrid) {
            this.efConstructionGrid = new ArrayList<>(efConstructionGrid);
            return this;
        }

        /**
         * Sets the grid of neighbor overflow multipliers to test.
         *
         * @param neighborOverflowGrid list of overflow multipliers, defensively copied
         * @return this builder for method chaining
         */
        public Builder withNeighborOverflowGrid(List<Float> neighborOverflowGrid) {
            this.neighborOverflowGrid = new ArrayList<>(neighborOverflowGrid);
            return this;
        }

        /**
         * Sets the grid of add hierarchy boolean values.
         *
         * @param addHierarchyGrid list of boolean values, defensively copied
         * @return this builder for method chaining
         */
        public Builder withAddHierarchyGrid(List<Boolean> addHierarchyGrid) {
            this.addHierarchyGrid = new ArrayList<>(addHierarchyGrid);
            return this;
        }

        /**
         * Sets the grid of refine final graph boolean values.
         *
         * @param refineFinalGraphGrid list of boolean values, defensively copied
         * @return this builder for method chaining
         */
        public Builder withRefineFinalGraphGrid(List<Boolean> refineFinalGraphGrid) {
            this.refineFinalGraphGrid = new ArrayList<>(refineFinalGraphGrid);
            return this;
        }

        /**
         * Sets the feature sets to test.
         *
         * @param featureSets list of feature sets, defensively copied
         * @return this builder for method chaining
         */
        public Builder withFeatureSets(List<? extends Set<FeatureId>> featureSets) {
            this.featureSets = new ArrayList<>(featureSets);
            return this;
        }

        /**
         * Sets the compressor functions to use during graph construction.
         *
         * @param buildCompressors list of compressor parameter functions, defensively copied
         * @return this builder for method chaining
         */
        public Builder withBuildCompressors(List<Function<DataSet, CompressorParameters>> buildCompressors) {
            this.buildCompressors = new ArrayList<>(buildCompressors);
            return this;
        }

        /**
         * Sets the compressor functions to use during search.
         *
         * @param searchCompressors list of compressor parameter functions, defensively copied
         * @return this builder for method chaining
         */
        public Builder withSearchCompressors(List<Function<DataSet, CompressorParameters>> searchCompressors) {
            this.searchCompressors = new ArrayList<>(searchCompressors);
            return this;
        }

        /**
         * Sets the grid of topK overquery multipliers.
         *
         * @param topKOverqueryGrid map of K values to overquery multipliers, defensively copied
         * @return this builder for method chaining
         */
        public Builder withTopKOverqueryGrid(Map<Integer, List<Double>> topKOverqueryGrid) {
            this.topKOverqueryGrid = new HashMap<>(topKOverqueryGrid);
            return this;
        }

        /**
         * Sets the grid of use pruning boolean values.
         *
         * @param usePruningGrid list of boolean values, defensively copied
         * @return this builder for method chaining
         */
        public Builder withUsePruningGrid(List<Boolean> usePruningGrid) {
            this.usePruningGrid = new ArrayList<>(usePruningGrid);
            return this;
        }

        /**
         * Sets the benchmark specification. A null value indicates default benchmarks should be used.
         *
         * @param benchmarkSpec map of benchmark specifications, defensively copied if not null
         * @return this builder for method chaining
         */
        public Builder withBenchmarkSpec(Map<String, List<String>> benchmarkSpec) {
            this.benchmarkSpec = benchmarkSpec == null ? null : new HashMap<>(benchmarkSpec);
            return this;
        }

        /**
         * Sets whether to collect results.
         *
         * @param collectResults true to collect results, false otherwise
         * @return this builder for method chaining
         */
        public Builder collectResults(boolean collectResults) {
            this.collectResults = collectResults;
            return this;
        }

        /**
         * Builds and returns a configured BenchFrameConfig instance with immutable collections.
         *
         * @return a new BenchFrameConfig with the configured values
         */
        public BenchFrameConfig build() {
            return new BenchFrameConfig(this);
        }
    }
}
