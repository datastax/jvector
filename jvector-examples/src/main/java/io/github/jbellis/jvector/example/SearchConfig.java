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
import io.github.jbellis.jvector.example.util.CompressorParameters;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * The search-side sweep {@link Grid} runs against every index it builds: each search compressor,
 * each topK with its overquery factors, each pruning setting, and optionally each adaptive termination gamma.
 *
 * <p>To add a search parameter, add a field, a {@link Builder} setter, and its use in {@link Grid}.</p>
 */
public final class SearchConfig {
    /** Compressors used for approximate scoring at search time; one returning NONE searches full precision. */
    public final List<Function<DataSet, CompressorParameters>> searchCompressors;
    /** topK mapped to the overquery factors to try; each factor gives rerankK = topK * factor. */
    public final Map<Integer, List<Double>> topKGrid;
    public final List<Boolean> usePruningGrid;
    /**
     * Adaptive termination gammas (see {@link io.github.jbellis.jvector.graph.AdaptiveTermination}) run for
     * every topK/overquery cell in addition to the classic beam search; empty runs beam search only.
     */
    public final List<Float> adaptiveGammaGrid;

    private SearchConfig(Builder b) {
        this.searchCompressors = b.searchCompressors;
        this.topKGrid = b.topKGrid;
        this.usePruningGrid = b.usePruningGrid;
        this.adaptiveGammaGrid = b.adaptiveGammaGrid;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private List<Function<DataSet, CompressorParameters>> searchCompressors;
        private Map<Integer, List<Double>> topKGrid;
        private List<Boolean> usePruningGrid;
        private List<Float> adaptiveGammaGrid = List.of();

        private Builder() {}

        public Builder searchCompressors(List<Function<DataSet, CompressorParameters>> searchCompressors) { this.searchCompressors = searchCompressors; return this; }
        public Builder topKGrid(Map<Integer, List<Double>> topKGrid) { this.topKGrid = topKGrid; return this; }
        public Builder usePruningGrid(List<Boolean> usePruningGrid) { this.usePruningGrid = usePruningGrid; return this; }
        /** Optional; null or empty runs beam search only. */
        public Builder adaptiveGammaGrid(List<Float> adaptiveGammaGrid) { this.adaptiveGammaGrid = adaptiveGammaGrid == null ? List.of() : List.copyOf(adaptiveGammaGrid); return this; }

        /**
         * @throws IllegalStateException if a parameter was not set
         * @throws IllegalArgumentException if an adaptive gamma is not finite and non-negative
         */
        public SearchConfig build() {
            require(searchCompressors, "searchCompressors");
            require(topKGrid, "topKGrid");
            require(usePruningGrid, "usePruningGrid");
            // Fail before the first build rather than after an index is written.
            for (float gamma : adaptiveGammaGrid) {
                if (!(gamma >= 0) || Float.isInfinite(gamma)) {
                    throw new IllegalArgumentException("search.adaptiveGamma values must be finite and >= 0, got " + gamma);
                }
            }
            return new SearchConfig(this);
        }

        private static void require(Object value, String name) {
            if (value == null) {
                throw new IllegalStateException("SearchConfig." + name + " must be set");
            }
        }
    }
}
