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

package io.github.jbellis.jvector.example.benchmarks.datasets;

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.yaml.snakeyaml.Yaml;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * The essential properties of a vector dataset which make it valid for indexing, querying, and testing in general.
 * These properties describe the base facet of a whole dataset, and are not defined for query vectors, distances, etc.
 * For facet-by-facet (base, query, distance) properties, a different interface will be provided.
 */
public interface DataSetProperties {

    /// Canonical key for the dataset name ({@link String}).
    String KEY_NAME = "name";

    /// Canonical key for the similarity function ({@link VectorSimilarityFunction} or its name as a {@link String}).
    String KEY_SIMILARITY_FUNCTION = "similarity_function";

    /// Canonical key for the number of base vectors ({@link Integer}).
    String KEY_NUM_VECTORS = "num_vectors";

    /// Canonical key for whether the dataset is normalized ({@link Boolean}).
    String KEY_IS_NORMALIZED = "is_normalized";

    /// Canonical key for whether the dataset is free of zero vectors ({@link Boolean}).
    String KEY_IS_ZERO_VECTOR_FREE = "is_zero_vector_free";

    /// Canonical key for whether the dataset is free of duplicate vectors ({@link Boolean}).
    String KEY_IS_DUPLICATE_VECTOR_FREE = "is_duplicate_vector_free";

    /// Canonical key for how benchmark loaders should treat the dataset at load time.
    String KEY_LOAD_BEHAVIOR = "load_behavior";

    /**
     * Controls benchmark-loader behavior for this dataset.
     *
     * <p>LEGACY_SCRUB preserves current behavior (to be deprecated).
     * NO_SCRUB loads the dataset exactly as provided, without load-time scrubbing
     * or ground-truth remapping.
     */
    enum LoadBehavior {
        LEGACY_SCRUB,
        NO_SCRUB
    }

    /**
     * Returns the similarity function for this dataset.
     *
     * @return the similarity function, or empty if not configured
     */
    Optional<VectorSimilarityFunction> similarityFunction();

    /**
     * Get the number of (base) vectors in this dataset.
     * @return the number of base vectors in this dataset.
     */
    public int numVectors();

    /**
     * Get the name of the dataset
     * @return the name of the dataset
     */
    public String getName();

    /**
     * Has this dataset been normalized?
     * It is an error for this to return true when the vectors are not normalized.
     * It is acceptable to return false when the dataset is not <em>known</em> to be normalized.
     * @return true if this dataset has been normalized, false otherwise.
     */
    public boolean isNormalized();

    /**
     * Has this dataset been verified or corrected to contain no zero-vectors, i.e., vectors with all components set to zero?
     * Vectors with some zero components are deemed valid, but vectors with all components set to zero are not.
     * It is an error for this to return true when there is a single zero-vector.
     * It is acceptable for it to return false when the dataset is not <em>known</em> to be zero vector free.
     * @return true if the dataset is known to have no zero vectors
     */
    public boolean isZeroVectorFree();

    /**
     * Vectors in this dataset must be distinct or this method should return false.
     * Datasets which have duplicate values are assumed to have other issues with embedding, process controls, etc.
     * Further, graph construction algorithms are sensitive to vector identity being stable between ordinals and values.
     * It is an error for this to return true when there are duplicate vectors.
     * It is acceptable for it to return false when the dataset is not <em>known</em> to be duplicate vector free.
     * @return true, if all vectors in this dataset are distinct.
     */
    public boolean isDuplicateVectorFree();

    /**
     * Returns how benchmark loaders should treat this dataset at load time.
     *
     * <p>This is a loader policy, not a statement of dataset quality.
     * The default preserves legacy behavior.
     *
     * @return the benchmark loader behavior for this dataset
     */
    default LoadBehavior loadBehavior() {
        return LoadBehavior.LEGACY_SCRUB;
    }

    /**
     * A convenience method to capture the notion of a valid dataset.
     * As any additional qualifiers are added to this data carrier, this method should be updated accordingly.
     * @return true, if the dataset is known to be valid for indexing and querying.
     */
    default boolean isValid() {
        return isZeroVectorFree() && isDuplicateVectorFree();
    }

    /// A {@link DataSetProperties} implementation backed by a {@code Map<String, Object>}.
    ///
    /// Property keys use the {@code KEY_*} constants defined on {@link DataSetProperties}.
    /// Missing or null values fall back to safe defaults (empty optional, zero, or false).
    ///
    /// A {@code PropertyMap} can be constructed directly from a map, or loaded from a YAML
    /// file with an optional document key to select a named top-level entry.
    ///
    /// ### Examples
    /// ```java
    /// // From a map
    /// var props = new DataSetProperties.PropertyMap(Map.of(
    ///     DataSetProperties.KEY_NAME, "ada002-100k",
    ///     DataSetProperties.KEY_SIMILARITY_FUNCTION, VectorSimilarityFunction.COSINE
    /// ));
    ///
    /// // From a YAML file, selecting a named entry
    /// var props = new DataSetProperties.PropertyMap("dataset_metadata.yml", "ada002-100k");
    ///
    /// // From a flat YAML file (no top-level key)
    /// var props = new DataSetProperties.PropertyMap("my_dataset.yml", null);
    /// ```
    class PropertyMap implements DataSetProperties {

        private final Map<String, Object> properties;

        /// Creates a new instance backed by the given map.
        ///
        /// @param properties the property map; keys should use the {@code KEY_*} constants
        ///                   from {@link DataSetProperties}
        public PropertyMap(Map<String, Object> properties) {
            this.properties = properties;
        }

        /// Loads properties from a YAML file.
        ///
        /// If {@code documentKey} is non-null and non-empty, the YAML document is expected
        /// to be a map of maps, and the entry at that key is used as the properties. The
        /// document key is also set as the {@link DataSetProperties#KEY_NAME} if no explicit
        /// name is present.
        ///
        /// If {@code documentKey} is null or empty, the entire YAML document is treated as
        /// the property map.
        ///
        /// @param yamlFile    path to a {@code .yml} or {@code .yaml} file
        /// @param documentKey the top-level key to select, or null/empty to use the whole document
        /// @throws IllegalArgumentException if the file does not end in {@code .yml} or {@code .yaml}
        /// @throws RuntimeException         if the file cannot be read or parsed
        @SuppressWarnings("unchecked")
        public PropertyMap(String yamlFile, String documentKey) {
            if (!yamlFile.endsWith(".yml") && !yamlFile.endsWith(".yaml")) {
                throw new IllegalArgumentException("Expected a .yml or .yaml file, got: " + yamlFile);
            }
            Map<String, Object> loaded;
            try (InputStream in = new FileInputStream(yamlFile)) {
                loaded = new Yaml().load(in);
            } catch (IOException e) {
                throw new RuntimeException("Failed to load YAML from " + yamlFile, e);
            }
            if (documentKey != null && !documentKey.isEmpty()) {
                Object entry = loaded.get(documentKey);
                if (entry == null) {
                    throw new IllegalArgumentException("No entry found for key '" + documentKey + "' in " + yamlFile);
                }
                if (!(entry instanceof Map)) {
                    throw new IllegalArgumentException("Entry for key '" + documentKey + "' in " + yamlFile + " is not a map");
                }
                var props = new HashMap<>((Map<String, Object>) entry);
                props.putIfAbsent(KEY_NAME, documentKey);
                this.properties = props;
            } else {
                this.properties = loaded != null ? loaded : Map.of();
            }
        }

        @Override
        public Optional<VectorSimilarityFunction> similarityFunction() {
            var value = properties.get(KEY_SIMILARITY_FUNCTION);
            if (value instanceof VectorSimilarityFunction) {
                return Optional.of((VectorSimilarityFunction) value);
            }
            if (value instanceof String) {
                return Optional.of(VectorSimilarityFunction.valueOf((String) value));
            }
            return Optional.empty();
        }

        @Override
        public int numVectors() {
            var value = properties.get(KEY_NUM_VECTORS);
            if (value instanceof Number) {
                return ((Number) value).intValue();
            }
            return 0;
        }

        @Override
        public String getName() {
            var value = properties.get(KEY_NAME);
            return value != null ? value.toString() : "";
        }

        @Override
        public boolean isNormalized() {
            return Boolean.TRUE.equals(properties.get(KEY_IS_NORMALIZED));
        }

        @Override
        public boolean isZeroVectorFree() {
            return Boolean.TRUE.equals(properties.get(KEY_IS_ZERO_VECTOR_FREE));
        }

        @Override
        public boolean isDuplicateVectorFree() {
            return Boolean.TRUE.equals(properties.get(KEY_IS_DUPLICATE_VECTOR_FREE));
        }

        @Override
        public LoadBehavior loadBehavior() {
            var value = properties.get(KEY_LOAD_BEHAVIOR);
            if (value instanceof LoadBehavior) {
                return (LoadBehavior) value;
            }
            if (value instanceof String) {
                return LoadBehavior.valueOf((String) value);
            }
            return LoadBehavior.LEGACY_SCRUB;
        }
    }
}
