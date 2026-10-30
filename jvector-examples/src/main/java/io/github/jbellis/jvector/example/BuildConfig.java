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

import java.util.function.Function;

/**
 * One point in the {@link Grid} index-construction sweep: the parameters used to build a single graph,
 * which is then written once per configured feature set.
 *
 * <p>To add a construction parameter, add a field, a {@link Builder} setter, and its use in {@link Grid}.
 * Keeping each parameter on its own lines lets concurrent changes merge without touching shared signatures.</p>
 */
public final class BuildConfig {
    /** Maximum out-degree of the graph. */
    public final int M;
    public final int efConstruction;
    public final float neighborOverflow;
    /** Diversity pruning factor; Grid has always used 1.2. */
    public final float alpha;
    public final boolean addHierarchy;
    public final boolean refineFinalGraph;
    /** Compression used for graph construction, or null to build from full-precision vectors. */
    public final Function<DataSet, CompressorParameters> buildCompressor;

    private BuildConfig(Builder b) {
        this.M = b.M;
        this.efConstruction = b.efConstruction;
        this.neighborOverflow = b.neighborOverflow;
        this.alpha = b.alpha;
        this.addHierarchy = b.addHierarchy;
        this.refineFinalGraph = b.refineFinalGraph;
        this.buildCompressor = b.buildCompressor;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Integer M;
        private Integer efConstruction;
        private Float neighborOverflow;
        private float alpha = 1.2f;
        private Boolean addHierarchy;
        private Boolean refineFinalGraph;
        private Function<DataSet, CompressorParameters> buildCompressor;

        private Builder() {}

        public Builder M(int M) { this.M = M; return this; }
        public Builder efConstruction(int efConstruction) { this.efConstruction = efConstruction; return this; }
        public Builder neighborOverflow(float neighborOverflow) { this.neighborOverflow = neighborOverflow; return this; }
        public Builder alpha(float alpha) { this.alpha = alpha; return this; }
        public Builder addHierarchy(boolean addHierarchy) { this.addHierarchy = addHierarchy; return this; }
        public Builder refineFinalGraph(boolean refineFinalGraph) { this.refineFinalGraph = refineFinalGraph; return this; }
        public Builder buildCompressor(Function<DataSet, CompressorParameters> buildCompressor) { this.buildCompressor = buildCompressor; return this; }

        /** @throws IllegalStateException if a parameter without a default was not set */
        public BuildConfig build() {
            require(M, "M");
            require(efConstruction, "efConstruction");
            require(neighborOverflow, "neighborOverflow");
            require(addHierarchy, "addHierarchy");
            require(refineFinalGraph, "refineFinalGraph");
            return new BuildConfig(this);
        }

        private static void require(Object value, String name) {
            if (value == null) {
                throw new IllegalStateException("BuildConfig." + name + " must be set");
            }
        }
    }
}
