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

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;


public class ConstructionParameters extends CommonParameters {
    public String type;
    public String writerMode;
    public List<Integer> outDegree;
    public List<Integer> efConstruction;
    public List<Float> neighborOverflow;
    public List<Boolean> addHierarchy;
    public List<Boolean> refineFinalGraph;
    /** Schema 2 accepts a list of features. Schema 3 accepts a single map with a type. */
    public Object reranking;
    public List<Boolean> fusedGraph;
    public Boolean useSavedIndexIfExists;

    void validateSchema3() {
        if (type == null || !List.of("inMemory", "onDisk").contains(type)) {
            throw new IllegalArgumentException("construction.type must be inMemory or onDisk");
        }
        if (writerMode != null && !List.of("sequential", "parallel").contains(writerMode)) {
            throw new IllegalArgumentException("construction.writerMode must be sequential or parallel");
        }
        if ("inMemory".equals(type) && (writerMode != null || fusedGraph != null || useSavedIndexIfExists != null)) {
            throw new IllegalArgumentException("construction.writerMode, fusedGraph, and useSavedIndexIfExists require onDisk");
        }
        validateSweep("outDegree", outDegree, value -> value > 0);
        validateSweep("efConstruction", efConstruction, value -> value > 0);
        validateSweep("neighborOverflow", neighborOverflow, value -> Float.isFinite(value) && value >= 1);
        validateSweep("addHierarchy", addHierarchy, value -> true);
        validateSweep("refineFinalGraph", refineFinalGraph, value -> true);
        if (fusedGraph != null) validateSweep("fusedGraph", fusedGraph, value -> true);
        rerankingType();
        validateCompression("construction", false);
    }

    private static <T> void validateSweep(String name, List<T> values, Predicate<T> valid) {
        if (values == null || values.isEmpty() || values.stream().anyMatch(value -> value == null || !valid.test(value))) {
            throw new IllegalArgumentException("construction." + name + " requires a nonempty list of valid values");
        }
    }

    public String rerankingType() {
        if (reranking == null) return "None";
        if (!(reranking instanceof Map<?, ?>)) {
            throw new IllegalArgumentException("construction.reranking must be a map with a type in schema 3");
        }
        var fields = (Map<?, ?>) reranking;
        if (fields.size() != 1 || !(fields.get("type") instanceof String)
                || !List.of("None", "FP32", "NVQ").contains(fields.get("type"))) {
            throw new IllegalArgumentException("construction.reranking.type must be None, FP32, or NVQ");
        }
        return (String) fields.get("type");
    }

    public boolean usesNVQReranking() {
        if (reranking instanceof List<?>) return ((List<?>) reranking).contains("NVQ");
        return "NVQ".equals(rerankingType());
    }

    public List<EnumSet<FeatureId>> getFeatureSets() {
        var featureSets = new LinkedHashSet<EnumSet<FeatureId>>();
        boolean hasPQConstruction = hasConstructionCompressor("PQ");
        boolean hasASHConstruction = hasConstructionCompressor("ASH");

        for (var fusedItem : fusedGraph) {
            if (!(reranking instanceof List<?>)) {
                throw new IllegalArgumentException("construction.reranking must be a list in schema 2");
            }
            for (var item : (List<?>) reranking) {
                EnumSet<FeatureId> baseFeatures;

                switch (String.valueOf(item)) {
                    case "FP":
                        baseFeatures = EnumSet.of(FeatureId.INLINE_VECTORS);
                        break;
                    case "NVQ":
                        baseFeatures = EnumSet.of(FeatureId.NVQ_VECTORS);
                        break;
                    default:
                        throw new IllegalArgumentException("Only 'FP' and 'NVQ' are supported");
                }

                if (!fusedItem) {
                    featureSets.add(baseFeatures);
                    continue;
                }

                boolean addedFused = false;
                if (hasPQConstruction) {
                    var features = EnumSet.copyOf(baseFeatures);
                    features.add(FeatureId.FUSED_PQ);
                    featureSets.add(features);
                    addedFused = true;
                }
                if (hasASHConstruction) {
                    var features = EnumSet.copyOf(baseFeatures);
                    features.add(FeatureId.FUSED_ASH);
                    featureSets.add(features);
                    addedFused = true;
                }

                if (!addedFused) {
                    throw new IllegalArgumentException(
                            "fusedGraph=Yes requires construction compression type PQ or ASH");
                }
            }
        }

        return new ArrayList<>(featureSets);
    }

    private boolean hasConstructionCompressor(String type) {
        if (compression == null) {
            return false;
        }

        for (Compression c : compression) {
            if (c != null && c.type != null && c.type.equalsIgnoreCase(type)) {
                return true;
            }
        }
        return false;
    }
}
