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


public class ConstructionParameters extends CommonParameters {
    public List<Integer> outDegree;
    public List<Integer> efConstruction;
    public List<Float> neighborOverflow;
    public List<Boolean> addHierarchy;
    public List<Boolean> refineFinalGraph;
    public List<String> reranking;
    public List<Boolean> fusedGraph;
    public Boolean useSavedIndexIfExists;

    public List<EnumSet<FeatureId>> getFeatureSets() {
        var featureSets = new LinkedHashSet<EnumSet<FeatureId>>();
        boolean hasPQConstruction = hasConstructionCompressor("PQ");
        boolean hasASHConstruction = hasConstructionCompressor("ASH");

        for (var fusedItem : fusedGraph) {
            for (var item : reranking) {
                EnumSet<FeatureId> baseFeatures;

                switch (item) {
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