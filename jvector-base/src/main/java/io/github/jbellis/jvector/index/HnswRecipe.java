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

package io.github.jbellis.jvector.index;

import io.github.jbellis.jvector.annotations.Experimental;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Named, recommended starting configurations for a graph/HNSW index, applied via a graph index
 * builder's {@code applyRecipe(HnswRecipe)}.
 * <p>
 * Each recipe carries a set of <em>ingredients</em>: builder parameter values keyed by parameter
 * name (see {@link Param}). The builder looks each one up by name with {@link #get(String)} and
 * sets it; parameters a recipe does not mention are left as the builder has them.
 * <p>
 * Only {@link #DEFAULT} is defined so far, and it simply restates JVector's own defaults, so applying
 * it is effectively a no-op on a fresh builder. {@link #HIGH_RECALL} and {@link #HIGH_PERFORMANCE}
 * have no ingredients yet, and {@code applyRecipe} refuses them at runtime rather than guess.
 */
@Experimental
public enum HnswRecipe {
    /** JVector's default value for every builder parameter that has one. */
    DEFAULT(Map.ofEntries(
            Map.entry(Param.COMPRESSION_TYPE, "NONE"),
            Map.entry(Param.MAX_DEGREES, List.of(32)),
            Map.entry(Param.BEAM_WIDTH, 100),
            Map.entry(Param.NEIGHBOR_OVERFLOW, 1.2f),
            Map.entry(Param.ALPHA, 1.2f),
            Map.entry(Param.ADD_HIERARCHY, true),
            Map.entry(Param.REFINE_FINAL_GRAPH, true),
            Map.entry(Param.PQ_SUBSPACES, 0),
            Map.entry(Param.PQ_GLOBAL_CENTERING, false),
            Map.entry(Param.PQ_ANISOTROPIC_THRESHOLD, -1.0f),
            Map.entry(Param.ASH_PROJECTED_DIMENSIONS, 0),
            Map.entry(Param.ASH_BITS_PER_DIMENSION, 2))),
    /** Favors recall over build and search speed. Not defined yet. */
    HIGH_RECALL(Map.of()),
    /** Favors build and search speed over recall. Not defined yet. */
    HIGH_PERFORMANCE(Map.of());

    /**
     * Names of the builder parameters a recipe can set. Values are stored with the builder's own
     * Java types, except {@link #COMPRESSION_TYPE}, which is the name of a {@code CompressionType}
     * constant because that enum lives outside this module.
     */
    @Experimental
    public static final class Param {
        /** {@code String}: name of a {@code CompressionType} constant. */
        public static final String COMPRESSION_TYPE = "compressionType";
        /** {@code List<Integer>}: max degree per layer. */
        public static final String MAX_DEGREES = "maxDegrees";
        /** {@code Integer}. */
        public static final String BEAM_WIDTH = "beamWidth";
        /** {@code Float}. */
        public static final String NEIGHBOR_OVERFLOW = "neighborOverflow";
        /** {@code Float}. */
        public static final String ALPHA = "alpha";
        /** {@code Boolean}. */
        public static final String ADD_HIERARCHY = "addHierarchy";
        /** {@code Boolean}. */
        public static final String REFINE_FINAL_GRAPH = "refineFinalGraph";
        /** {@code Integer}: PQ subspaces; {@code 0} means the builder's default for the vector dimension. */
        public static final String PQ_SUBSPACES = "pqSubspaces";
        /** {@code Boolean}. */
        public static final String PQ_GLOBAL_CENTERING = "pqGlobalCentering";
        /** {@code Float}: {@code -1.0} is unweighted. */
        public static final String PQ_ANISOTROPIC_THRESHOLD = "pqAnisotropicThreshold";
        /** {@code Integer}: ASH projected dimensions; {@code 0} means the builder's default, the vector dimension. */
        public static final String ASH_PROJECTED_DIMENSIONS = "ashProjectedDimensions";
        /** {@code Integer}: ASH bits per projected dimension. */
        public static final String ASH_BITS_PER_DIMENSION = "ashBitsPerDimension";

        private Param() {
        }
    }

    private final Map<String, Object> ingredients;

    HnswRecipe(Map<String, Object> ingredients) {
        this.ingredients = ingredients;
    }

    /** Whether this recipe has any ingredients defined yet. */
    public boolean isDefined() {
        return !ingredients.isEmpty();
    }

    /** Whether this recipe sets the named parameter. */
    public boolean has(String param) {
        return ingredients.containsKey(param);
    }

    /** The names of the parameters this recipe sets. */
    public Set<String> params() {
        return ingredients.keySet();
    }

    /**
     * The value this recipe sets for the named parameter, cast to the caller's expected type
     * (see {@link Param} for each parameter's type), or {@code null} if the recipe does not set it.
     */
    @SuppressWarnings("unchecked")
    public <T> T get(String param) {
        return (T) ingredients.get(param);
    }
}
