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