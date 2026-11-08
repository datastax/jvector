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
import io.github.jbellis.jvector.example.diagnostics.ASHBuildScoreExperiment;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Function;

/**
 * Everything {@link Grid} runs for one dataset: the construction sweep, the feature sets written for each
 * graph, the search sweep, and whether built indexes are cached.
 *
 * <p>The construction sweep is the cartesian product of its {@link SweepDimension}s, with the first
 * dimension outermost. To sweep a new construction parameter, add it to {@link BuildConfig} and add one
 * {@code sweep(...)} line to {@link #fromGrids}.</p>
 */
public final class GridSpec {
    final List<SweepDimension<?>> buildSweep;
    final List<? extends Set<FeatureId>> featureSets;
    final SearchConfig search;
    final boolean enableIndexCache;
    private final List<Boolean> fusedGraph;

    private GridSpec(Builder b) {
        this.buildSweep = List.copyOf(b.buildSweep);
        this.featureSets = b.featureSets;
        this.search = b.search;
        this.enableIndexCache = b.enableIndexCache;
        this.fusedGraph = b.fusedGraph;
    }

    /** The sweep a dataset's YAML config describes. */
    public static GridSpec fromConfig(MultiConfig config) {
        var construction = config.construction;
        var search = config.search;
        if (config.usesSchema3()) {
            boolean onDisk = "onDisk".equals(construction.type);
            var writerMode = onDisk ? Objects.requireNonNullElse(construction.writerMode, "parallel") : null;
            return graphSweeps(construction.addHierarchy, construction.refineFinalGraph, construction.outDegree,
                    construction.neighborOverflow, construction.efConstruction)
                    .sweep("construction.compression", construction.compression, (point, compressor) -> point
                            .buildCompressor(compressor.getCompressorParameters())
                            .ashConstructionMode("ASH".equals(compressor.type)
                                    ? Objects.requireNonNullElseGet(compressor.ashConstructionMode(),
                                            ASHBuildScoreExperiment::defaultMode)
                                    : null)
                            .indexType(construction.type).writerMode(writerMode).rerankingType(construction.rerankingType()))
                    .featureSets(List.of())
                    .fusedGraph(construction.fusedGraph == null ? List.of(false) : construction.fusedGraph)
                    .search(SearchConfig.builder().compressionEntries(search.compression)
                            .topKGrid(search.topKOverquery).usePruningGrid(search.useSearchPruning).build())
                    .enableIndexCache(onDisk && Boolean.TRUE.equals(construction.useSavedIndexIfExists))
                    .build();
        }
        return fromGrids(construction.addHierarchy,
                construction.refineFinalGraph,
                construction.outDegree,
                construction.neighborOverflow,
                construction.efConstruction,
                construction.getCompressorParameters(),
                construction.getFeatureSets(),
                SearchConfig.builder()
                        .searchCompressors(search.getCompressorParameters())
                        .topKGrid(search.topKOverquery)
                        .usePruningGrid(search.useSearchPruning)
                        .build(),
                Objects.requireNonNull(construction.useSavedIndexIfExists,
                        "construction.useSavedIndexIfExists must be set"));
    }

    /**
     * The sweep over the given construction grids. This is the one place that sets the construction
     * sweep order; arguments are listed in that order, the first varying slowest.
     */
    static GridSpec fromGrids(List<Boolean> addHierarchyGrid,
                              List<Boolean> refineFinalGraphGrid,
                              List<Integer> mGrid,
                              List<Float> neighborOverflowGrid,
                              List<Integer> efConstructionGrid,
                              List<Function<DataSet, CompressorParameters>> buildCompressors,
                              List<? extends Set<FeatureId>> featureSets,
                              SearchConfig search,
                              boolean enableIndexCache) {
        return graphSweeps(addHierarchyGrid, refineFinalGraphGrid, mGrid, neighborOverflowGrid, efConstructionGrid)
                .sweep("construction.compression", buildCompressors, BuildConfig.Builder::buildCompressor)
                .featureSets(featureSets)
                .search(search)
                .enableIndexCache(enableIndexCache)
                .build();
    }

    private static Builder graphSweeps(List<Boolean> hierarchy, List<Boolean> refinement, List<Integer> degree,
                                       List<Float> overflow, List<Integer> efConstruction) {
        return builder()
                .sweep("addHierarchy", hierarchy, BuildConfig.Builder::addHierarchy)
                .sweep("refineFinalGraph", refinement, BuildConfig.Builder::refineFinalGraph)
                .sweep("outDegree", degree, BuildConfig.Builder::M)
                .sweep("neighborOverflow", overflow, BuildConfig.Builder::neighborOverflow)
                .sweep("efConstruction", efConstruction, BuildConfig.Builder::efConstruction);
    }

    List<? extends Set<FeatureId>> featuresFor(BuildConfig build, SearchConfig searchCase) {
        if (build.indexType == null) return featureSets;
        if ("inMemory".equals(build.indexType)) return List.of(Set.of());

        // The file format requires a vector feature. Source vectors also provide primary
        // traversal scores for uncompressed search, independently of the reranking policy.
        var base = EnumSet.noneOf(FeatureId.class);
        if ("NVQ".equals(build.rerankingType)) base.add(FeatureId.NVQ_VECTORS);
        if (base.isEmpty() || "None".equals(searchCase.compression.type)) base.add(FeatureId.INLINE_VECTORS);
        var sets = new ArrayList<Set<FeatureId>>();
        for (boolean fused : fusedGraph) {
            var features = EnumSet.copyOf(base);
            if (fused) {
                features.add("ASH".equals(searchCase.compression.type) ? FeatureId.FUSED_ASH : FeatureId.FUSED_PQ);
            }
            if (!sets.contains(features)) sets.add(features);
        }
        return sets;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Every point of the construction sweep, in iteration order.
     *
     * @throws IllegalStateException if a point leaves a required {@link BuildConfig} parameter unset
     */
    List<BuildConfig> buildPoints() {
        List<BuildConfig> points = new ArrayList<>();
        int dims = buildSweep.size();
        for (var dim : buildSweep) {
            if (dim.size() == 0) {
                return points;
            }
        }
        int[] index = new int[dims];
        while (true) {
            var builder = BuildConfig.builder();
            for (int d = 0; d < dims; d++) {
                buildSweep.get(d).apply(builder, index[d]);
            }
            points.add(builder.build());

            // Advance the last dimension fastest, carrying into earlier ones.
            int d = dims - 1;
            while (d >= 0 && ++index[d] == buildSweep.get(d).size()) {
                index[d] = 0;
                d--;
            }
            if (d < 0) {
                return points;
            }
        }
    }

    public static final class Builder {
        private final List<SweepDimension<?>> buildSweep = new ArrayList<>();
        private List<? extends Set<FeatureId>> featureSets;
        private SearchConfig search;
        private boolean enableIndexCache;
        private List<Boolean> fusedGraph;

        private Builder() {}

        /** Adds a construction dimension after those already added, so it varies faster than them. */
        public <T> Builder sweep(String name, List<T> values, BiConsumer<BuildConfig.Builder, T> setter) {
            buildSweep.add(SweepDimension.of(name, values, setter));
            return this;
        }

        public Builder featureSets(List<? extends Set<FeatureId>> featureSets) { this.featureSets = featureSets; return this; }
        public Builder search(SearchConfig search) { this.search = search; return this; }
        public Builder enableIndexCache(boolean enableIndexCache) { this.enableIndexCache = enableIndexCache; return this; }
        Builder fusedGraph(List<Boolean> fusedGraph) { this.fusedGraph = List.copyOf(fusedGraph); return this; }

        /**
         * @throws IllegalStateException if the feature sets or search config were not set
         * @throws IllegalArgumentException if a feature set names a feature with no {@link FeatureProvider}
         */
        public GridSpec build() {
            if (featureSets == null) {
                throw new IllegalStateException("GridSpec.featureSets must be set");
            }
            // Fail before the first build rather than write indexes without the feature.
            for (var features : featureSets) {
                for (var id : features) {
                    FeatureProviders.require(id);
                }
            }
            if (search == null) {
                throw new IllegalStateException("GridSpec.search must be set");
            }
            return new GridSpec(this);
        }
    }
}
