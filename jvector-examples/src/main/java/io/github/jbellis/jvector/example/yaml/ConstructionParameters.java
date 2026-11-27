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
    /** onDisk only: pad base-layer records so each starts on this page boundary (a power of two); omit for none. */
    public Integer pageAlignment;
    public List<Integer> outDegree;
    /**
     * Schema 3: maximum out-degrees of the layers above the base layer to sweep; omit for {@code outDegree} on every
     * layer. With it, {@code outDegree} is the base layer's degree. Needs {@code addHierarchy} Yes.
     */
    public List<Integer> upperLayerDegree;
    public List<Integer> efConstruction;
    public List<Float> neighborOverflow;
    public List<Boolean> addHierarchy;
    public List<Boolean> refineFinalGraph;
    /** Schema 2 accepts a list of features. Schema 3 accepts a single map with a type. */
    public Object reranking;
    public List<Boolean> fusedGraph;
    public Boolean useSavedIndexIfExists;
    /** Schema 3: threads for training, encoding, graph insertion and writing; an integer or max. */
    public Object workers;
    /**
     * Schema 3: insert the base vectors in arrival order into a streaming index, which encodes each as it arrives,
     * instead of encoding them all first. An {@code onDisk} streaming index is written and reloaded, as a flush does.
     */
    public Boolean streaming;
    /**
     * ASH streaming only: vectors scored exactly before a cold index fixes a random-projection model, or 0 to start
     * with the construction compressor's model. PQ and None streaming builds omit it or use 0.
     */
    public List<Integer> coldStartSize;
    /** ASH streaming only: whether a cold index trains its first model on its sample while vectors arrive. */
    public List<Boolean> trainWhileStreaming;

    /** The construction worker count, or null when the YAML leaves it to the platform default. */
    public Integer resolvedWorkers() {
        return workers == null ? null : Workers.resolve(workers, "construction.workers");
    }

    void validateSchema3() {
        if (type == null || !List.of("inMemory", "onDisk").contains(type)) {
            throw new IllegalArgumentException("construction.type must be inMemory or onDisk");
        }
        if (writerMode != null && !List.of("sequential", "parallel").contains(writerMode)) {
            throw new IllegalArgumentException("construction.writerMode must be sequential or parallel");
        }
        if (!"onDisk".equals(type) && (writerMode != null || fusedGraph != null || pageAlignment != null)) {
            throw new IllegalArgumentException("construction.writerMode, pageAlignment, and fusedGraph require onDisk");
        }
        if (Boolean.TRUE.equals(streaming)) {
            if (compression == null || compression.size() != 1 || compression.get(0) == null
                    || !List.of("ASH", "PQ", "None").contains(compression.get(0).type)) {
                throw new IllegalArgumentException("construction.streaming requires a single ASH, PQ or None construction.compression");
            }
            if ("ASH".equals(compression.get(0).type)) {
                validateSweep("coldStartSize", coldStartSize, value -> value >= 0);
            } else if ((coldStartSize != null && coldStartSize.stream().anyMatch(value -> value == null || value != 0))
                    || (trainWhileStreaming != null && trainWhileStreaming.contains(Boolean.TRUE))) {
                throw new IllegalArgumentException("Only ASH streaming builds start cold or train while streaming:"
                        + " use coldStartSize [0] and trainWhileStreaming No, or omit them");
            }
            if ("inMemory".equals(type) && "NVQ".equals(rerankingType())) {
                throw new IllegalArgumentException("An inMemory streaming index reranks with the source vectors: use reranking FP32 or None");
            }
            if (fusedGraph != null && fusedGraph.contains(Boolean.TRUE)) {
                throw new IllegalArgumentException("construction.streaming writes its codes beside the index: use fusedGraph No");
            }
            if (trainWhileStreaming != null) validateSweep("trainWhileStreaming", trainWhileStreaming, value -> true);
        } else if (coldStartSize != null || trainWhileStreaming != null) {
            throw new IllegalArgumentException("construction.coldStartSize and trainWhileStreaming require construction.streaming Yes");
        }
        if (pageAlignment != null && (pageAlignment < 512 || pageAlignment > (1 << 20) || Integer.bitCount(pageAlignment) != 1)) {
            throw new IllegalArgumentException("construction.pageAlignment must be a power of two from 512 to 1048576");
        }
        validateSweep("outDegree", outDegree, value -> value > 0);
        if (upperLayerDegree != null) {
            validateSweep("upperLayerDegree", upperLayerDegree, value -> value > 0);
            if (addHierarchy == null || addHierarchy.contains(Boolean.FALSE))
                throw new IllegalArgumentException("construction.upperLayerDegree requires addHierarchy Yes");
            if (Boolean.TRUE.equals(streaming))
                throw new IllegalArgumentException("construction.upperLayerDegree is not supported for streaming builds");
        }
        validateSweep("efConstruction", efConstruction, value -> value > 0);
        validateSweep("neighborOverflow", neighborOverflow, value -> Float.isFinite(value) && value >= 1);
        validateSweep("addHierarchy", addHierarchy, value -> true);
        validateSweep("refineFinalGraph", refineFinalGraph, value -> true);
        if (fusedGraph != null) validateSweep("fusedGraph", fusedGraph, value -> true);
        rerankingType();
        resolvedWorkers();
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
