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


public class ConstructionParameters extends CommonParameters {
    public List<Integer> outDegree;
    public List<Integer> efConstruction;
    public List<Float> neighborOverflow;
    public List<Boolean> addHierarchy;
    public List<Boolean> refineFinalGraph;
    public List<String> reranking;
    public Boolean useSavedIndexIfExists;

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