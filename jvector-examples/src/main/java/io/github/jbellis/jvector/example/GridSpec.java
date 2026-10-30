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

import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.BiConsumer;

/**
 * Everything {@link Grid} runs for one dataset: the construction sweep, the feature sets written for each
 * graph, the search sweep, and whether built indexes are cached.
 *
 * <p>The construction sweep is the cartesian product of its {@link SweepDimension}s, with the first
 * dimension outermost. To sweep a new construction parameter, add it to {@link BuildConfig} and add one
 * {@code sweep(...)} line to {@link #fromConfig}.</p>
 */
public final class GridSpec {
    final List<SweepDimension<?>> buildSweep;
    final List<? extends Set<FeatureId>> featureSets;
    final SearchConfig search;
    final boolean enableIndexCache;

    private GridSpec(Builder b) {
        this.buildSweep = List.copyOf(b.buildSweep);
        this.featureSets = b.featureSets;
        this.search = b.search;
        this.enableIndexCache = b.enableIndexCache;
    }

    /** The sweep a dataset's YAML config describes. */
    public static GridSpec fromConfig(MultiConfig config) {
        var construction = config.construction;
        var search = config.search;
        return builder()
                // Order sets the output order: the first dimension varies slowest.
                .sweep("addHierarchy", construction.addHierarchy, BuildConfig.Builder::addHierarchy)
                .sweep("refineFinalGraph", construction.refineFinalGraph, BuildConfig.Builder::refineFinalGraph)
                .sweep("outDegree", construction.outDegree, BuildConfig.Builder::M)
                .sweep("neighborOverflow", construction.neighborOverflow, BuildConfig.Builder::neighborOverflow)
                .sweep("efConstruction", construction.efConstruction, BuildConfig.Builder::efConstruction)
                .sweep("construction.compression", construction.getCompressorParameters(), BuildConfig.Builder::buildCompressor)
                .featureSets(construction.getFeatureSets())
                .search(SearchConfig.builder()
                        .searchCompressors(search.getCompressorParameters())
                        .topKGrid(search.topKOverquery)
                        .usePruningGrid(search.useSearchPruning)
                        .build())
                .enableIndexCache(Objects.requireNonNull(construction.useSavedIndexIfExists,
                        "construction.useSavedIndexIfExists must be set"))
                .build();
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

        private Builder() {}

        /** Adds a construction dimension after those already added, so it varies faster than them. */
        public <T> Builder sweep(String name, List<T> values, BiConsumer<BuildConfig.Builder, T> setter) {
            buildSweep.add(SweepDimension.of(name, values, setter));
            return this;
        }

        public Builder featureSets(List<? extends Set<FeatureId>> featureSets) { this.featureSets = featureSets; return this; }
        public Builder search(SearchConfig search) { this.search = search; return this; }
        public Builder enableIndexCache(boolean enableIndexCache) { this.enableIndexCache = enableIndexCache; return this; }

        /** @throws IllegalStateException if the feature sets or search config were not set */
        public GridSpec build() {
            if (featureSets == null) {
                throw new IllegalStateException("GridSpec.featureSets must be set");
            }
            if (search == null) {
                throw new IllegalStateException("GridSpec.search must be set");
            }
            return new GridSpec(this);
        }
    }
}
