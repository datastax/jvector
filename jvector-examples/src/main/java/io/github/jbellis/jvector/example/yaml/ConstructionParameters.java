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

package io.github.jbellis.jvector.example.yaml;

import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.util.EnumSet;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Construction parameters for graph index building.
 */
public class ConstructionParameters extends CommonParameters {
    /** List of out-degree values. */
    public List<Integer> outDegree;
    /** List of efConstruction values. */
    public List<Integer> efConstruction;
    /** List of neighbor overflow values. */
    public List<Float> neighborOverflow;
    /** List of add hierarchy flags. */
    public List<Boolean> addHierarchy;
    /** List of refine final graph flags. */
    public List<Boolean> refineFinalGraph;
    /** List of reranking strategies. */
    public List<String> reranking;
    /** Flag to use saved index if exists. */
    public Boolean useSavedIndexIfExists;

    /**
     * Constructs a ConstructionParameters.
     */
    public ConstructionParameters() {}

    /**
     * Gets the feature sets based on reranking strategies.
     * @return the list of feature sets
     */
    public List<EnumSet<FeatureId>> getFeatureSets() {
        return reranking.stream().map(item -> {
            switch (item) {
                case "FP":
                    return EnumSet.of(FeatureId.INLINE_VECTORS);
                case "NVQ":
                    return EnumSet.of(FeatureId.NVQ_VECTORS);
                default:
                    throw new IllegalArgumentException("Only 'FP' and 'NVQ' are supported");
            }
        }).collect(Collectors.toList());
    }
}