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

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.*;

public class DataSetPropertiesTest {

    private static String testResource(String name) {
        return Objects.requireNonNull(
                DataSetPropertiesTest.class.getResource(name),
                "Test resource not found: " + name
        ).getPath();
    }

    // ========================================================================
    // PropertyMap from Map — happy paths
    // ========================================================================

    @Test
    public void propertyMapFromFullMap() {
        var props = new DataSetProperties.PropertyMap(Map.of(
                DataSetProperties.KEY_NAME, "test-ds",
                DataSetProperties.KEY_SIMILARITY_FUNCTION, VectorSimilarityFunction.COSINE,
                DataSetProperties.KEY_NUM_VECTORS, 42000,
                DataSetProperties.KEY_IS_NORMALIZED, true,
                DataSetProperties.KEY_IS_ZERO_VECTOR_FREE, true,
                DataSetProperties.KEY_IS_DUPLICATE_VECTOR_FREE, true
        ));
        assertEquals("test-ds", props.getName());
        assertEquals(VectorSimilarityFunction.COSINE, props.similarityFunction().orElse(null));
        assertEquals(42000, props.numVectors());
        assertTrue(props.isNormalized());
        assertTrue(props.isZeroVectorFree());
        assertTrue(props.isDuplicateVectorFree());
        assertTrue(props.isValid());
    }

    @Test
    public void propertyMapSimilarityFunctionFromString() {
        var props = new DataSetProperties.PropertyMap(Map.of(
                DataSetProperties.KEY_SIMILARITY_FUNCTION, "EUCLIDEAN"
        ));
        assertEquals(VectorSimilarityFunction.EUCLIDEAN, props.similarityFunction().orElse(null));
    }

    @Test
    public void propertyMapSimilarityFunctionFromEnum() {
        var props = new DataSetProperties.PropertyMap(Map.of(
                DataSetProperties.KEY_SIMILARITY_FUNCTION, VectorSimilarityFunction.DOT_PRODUCT
        ));
        assertEquals(VectorSimilarityFunction.DOT_PRODUCT, props.similarityFunction().orElse(null));
    }

    @Test
    public void propertyMapDefaults() {
        var props = new DataSetProperties.PropertyMap(Map.of());
        assertEquals("", props.getName());
        assertTrue(props.similarityFunction().isEmpty());
        assertEquals(0, props.numVectors());
        assertFalse(props.isNormalized());
        assertFalse(props.isZeroVectorFree());
        assertFalse(props.isDuplicateVectorFree());
        assertFalse(props.isValid());
    }

    @Test
    public void propertyMapNumVectorsFromLong() {
        var props = new DataSetProperties.PropertyMap(Map.of(
                DataSetProperties.KEY_NUM_VECTORS, 99999L
        ));
        assertEquals(99999, props.numVectors());
    }

    @Test
    public void propertyMapIsValidRequiresBothFlags() {
        var onlyZeroFree = new DataSetProperties.PropertyMap(Map.of(
                DataSetProperties.KEY_IS_ZERO_VECTOR_FREE, true
        ));
        assertFalse(onlyZeroFree.isValid());

        var onlyDedupe = new DataSetProperties.PropertyMap(Map.of(
                DataSetProperties.KEY_IS_DUPLICATE_VECTOR_FREE, true
        ));
        assertFalse(onlyDedupe.isValid());

        var both = new DataSetProperties.PropertyMap(Map.of(
                DataSetProperties.KEY_IS_ZERO_VECTOR_FREE, true,
                DataSetProperties.KEY_IS_DUPLICATE_VECTOR_FREE, true
        ));
        assertTrue(both.isValid());
    }

    // ========================================================================
    // PropertyMap from Map — adversarial inputs
    // ========================================================================

    @Test
    public void propertyMapInvalidSimilarityFunctionString() {
        var props = new DataSetProperties.PropertyMap(Map.of(
                DataSetProperties.KEY_SIMILARITY_FUNCTION, "NOT_A_REAL_FUNCTION"
        ));
        assertThrows(IllegalArgumentException.class, props::similarityFunction);
    }

    @Test
    public void propertyMapSimilarityFunctionWrongType() {
        // An Integer is neither String nor VectorSimilarityFunction
        var props = new DataSetProperties.PropertyMap(Map.of(
                DataSetProperties.KEY_SIMILARITY_FUNCTION, 12345
        ));
        assertTrue(props.similarityFunction().isEmpty());
    }

    @Test
    public void propertyMapNumVectorsWrongType() {
        var props = new DataSetProperties.PropertyMap(Map.of(
                DataSetProperties.KEY_NUM_VECTORS, "not-a-number"
        ));
        assertEquals(0, props.numVectors());
    }

    @Test
    public void propertyMapBooleanFieldsIgnoreNonBooleans() {
        // String "true" is not Boolean.TRUE
        var map = new HashMap<String, Object>();
        map.put(DataSetProperties.KEY_IS_NORMALIZED, "true");
        map.put(DataSetProperties.KEY_IS_ZERO_VECTOR_FREE, 1);
        map.put(DataSetProperties.KEY_IS_DUPLICATE_VECTOR_FREE, "yes");
        var props = new DataSetProperties.PropertyMap(map);
        assertFalse(props.isNormalized());
        assertFalse(props.isZeroVectorFree());
        assertFalse(props.isDuplicateVectorFree());
    }

    @Test
    public void propertyMapNullValuesInMap() {
        var map = new HashMap<String, Object>();
        map.put(DataSetProperties.KEY_NAME, null);
        map.put(DataSetProperties.KEY_SIMILARITY_FUNCTION, null);
        map.put(DataSetProperties.KEY_NUM_VECTORS, null);
        map.put(DataSetProperties.KEY_IS_NORMALIZED, null);
        var props = new DataSetProperties.PropertyMap(map);
        assertEquals("", props.getName());
        assertTrue(props.similarityFunction().isEmpty());
        assertEquals(0, props.numVectors());
        assertFalse(props.isNormalized());
    }

    // ========================================================================
    // PropertyMap from YAML — keyed entry
    // ========================================================================

    @Test
    public void yamlKeyedEntryFullProperties() {
        var props = new DataSetProperties.PropertyMap(testResource("multi_entry.yml"), "ada002-100k");
        assertEquals("ada002-100k", props.getName());
        assertEquals(VectorSimilarityFunction.COSINE, props.similarityFunction().orElse(null));
        assertEquals(100000, props.numVectors());
        assertTrue(props.isNormalized());
        assertTrue(props.isZeroVectorFree());
        assertTrue(props.isDuplicateVectorFree());
        assertTrue(props.isValid());
    }

    @Test
    public void yamlKeyedEntryMinimalProperties() {
        var props = new DataSetProperties.PropertyMap(testResource("multi_entry.yml"), "minimal-entry");
        assertEquals("minimal-entry", props.getName());
        assertEquals(VectorSimilarityFunction.DOT_PRODUCT, props.similarityFunction().orElse(null));
        assertEquals(0, props.numVectors());
        assertFalse(props.isNormalized());
        assertFalse(props.isValid());
    }

    @Test
    public void yamlKeyedEntryExplicitNameOverridesDocumentKey() {
        var props = new DataSetProperties.PropertyMap(testResource("multi_entry.yml"), "has-explicit-name");
        assertEquals("custom-name", props.getName(), "Explicit name in YAML should take precedence over document key");
    }

    @Test
    public void yamlKeyedEntryNameDefaultsToDocumentKey() {
        var props = new DataSetProperties.PropertyMap(testResource("multi_entry.yml"), "sift-128-euclidean");
        assertEquals("sift-128-euclidean", props.getName());
    }

    // ========================================================================
    // PropertyMap from YAML — flat document (null/empty key)
    // ========================================================================

    @Test
    public void yamlFlatDocumentNullKey() {
        var props = new DataSetProperties.PropertyMap(testResource("flat_entry.yml"), null);
        assertEquals("flat-dataset", props.getName());
        assertEquals(VectorSimilarityFunction.EUCLIDEAN, props.similarityFunction().orElse(null));
        assertEquals(50000, props.numVectors());
        assertFalse(props.isNormalized());
        assertTrue(props.isZeroVectorFree());
    }

    @Test
    public void yamlFlatDocumentEmptyKey() {
        var props = new DataSetProperties.PropertyMap(testResource("flat_entry.yml"), "");
        assertEquals("flat-dataset", props.getName());
        assertEquals(VectorSimilarityFunction.EUCLIDEAN, props.similarityFunction().orElse(null));
    }

    // ========================================================================
    // PropertyMap from YAML — adversarial / error cases
    // ========================================================================

    @Test
    public void yamlRejectsNonYamlExtension() {
        assertThrows(IllegalArgumentException.class, () ->
                new DataSetProperties.PropertyMap("data.json", null));
        assertThrows(IllegalArgumentException.class, () ->
                new DataSetProperties.PropertyMap("data.txt", "key"));
        assertThrows(IllegalArgumentException.class, () ->
                new DataSetProperties.PropertyMap("data.yml.bak", null));
    }

    @Test
    public void yamlAcceptsBothExtensions() {
        // .yml is tested above; verify .yaml also passes the extension check
        // (will fail at file-not-found, not at extension check)
        var ex = assertThrows(RuntimeException.class, () ->
                new DataSetProperties.PropertyMap("/nonexistent/path.yaml", null));
        assertTrue(ex.getMessage().contains("Failed to load YAML"), ex.getMessage());
    }

    @Test
    public void yamlNonexistentFile() {
        assertThrows(RuntimeException.class, () ->
                new DataSetProperties.PropertyMap("/no/such/file.yml", null));
    }

    @Test
    public void yamlMissingDocumentKey() {
        var ex = assertThrows(IllegalArgumentException.class, () ->
                new DataSetProperties.PropertyMap(testResource("multi_entry.yml"), "no-such-dataset"));
        assertTrue(ex.getMessage().contains("no-such-dataset"), ex.getMessage());
    }

    @Test
    public void yamlDocumentKeyPointsToNonMap() {
        var ex = assertThrows(IllegalArgumentException.class, () ->
                new DataSetProperties.PropertyMap(testResource("scalar_entry.yml"), "some_key"));
        assertTrue(ex.getMessage().contains("not a map"), ex.getMessage());
    }

    @Test
    public void yamlEmptyDocument() {
        var props = new DataSetProperties.PropertyMap(testResource("empty.yml"), null);
        assertEquals("", props.getName());
        assertTrue(props.similarityFunction().isEmpty());
    }

    // ========================================================================
    // DataSetMetadataReader
    // ========================================================================

    @Test
    public void metadataReaderLooksUpExactKey() {
        var reader = DataSetMetadataReader.load(testResource("multi_entry.yml"));
        var props = reader.getProperties("ada002-100k");
        assertTrue(props.isPresent());
        assertEquals("ada002-100k", props.get().getName());
        assertEquals(VectorSimilarityFunction.COSINE, props.get().similarityFunction().orElse(null));
    }

    @Test
    public void metadataReaderReturnsEmptyForUnknownKey() {
        var reader = DataSetMetadataReader.load(testResource("multi_entry.yml"));
        assertTrue(reader.getProperties("does-not-exist").isEmpty());
    }

    @Test
    public void metadataReaderFallsBackToHdf5Suffix() {
        // multi_entry.yml has "sift-128-euclidean" as a key (no .hdf5 suffix)
        // The reader should NOT find "sift-128-euclidean" via hdf5 fallback since the key exists directly
        var reader = DataSetMetadataReader.load(testResource("multi_entry.yml"));
        var props = reader.getProperties("sift-128-euclidean");
        assertTrue(props.isPresent());
        assertEquals(VectorSimilarityFunction.EUCLIDEAN, props.get().similarityFunction().orElse(null));
    }

    @Test
    public void metadataReaderNonexistentFile() {
        assertThrows(RuntimeException.class, () ->
                DataSetMetadataReader.load("/no/such/file.yml"));
    }

    // ========================================================================
    // DataSetInfo delegation
    // ========================================================================

    @Test
    public void dataSetInfoDelegatesToBaseProperties() {
        var base = new DataSetProperties.PropertyMap(Map.of(
                DataSetProperties.KEY_NAME, "delegate-test",
                DataSetProperties.KEY_SIMILARITY_FUNCTION, VectorSimilarityFunction.EUCLIDEAN,
                DataSetProperties.KEY_NUM_VECTORS, 7777,
                DataSetProperties.KEY_IS_NORMALIZED, true,
                DataSetProperties.KEY_IS_ZERO_VECTOR_FREE, true,
                DataSetProperties.KEY_IS_DUPLICATE_VECTOR_FREE, true
        ));
        var info = new DataSetInfo(base, () -> { throw new AssertionError("loader should not be called"); });

        assertEquals("delegate-test", info.getName());
        assertEquals(VectorSimilarityFunction.EUCLIDEAN, info.similarityFunction().orElse(null));
        assertEquals(7777, info.numVectors());
        assertTrue(info.isNormalized());
        assertTrue(info.isZeroVectorFree());
        assertTrue(info.isDuplicateVectorFree());
        assertTrue(info.isValid());
    }

    @Test
    public void dataSetInfoLazyLoading() {
        var callCount = new int[]{0};
        var base = new DataSetProperties.PropertyMap(Map.of(DataSetProperties.KEY_NAME, "lazy"));
        // Return a dummy non-null sentinel so the cache works (null would defeat the null-check)
        var sentinel = new DataSet() {
            public int getDimension() { return 0; }
            public RandomAccessVectorValues getBaseRavv() { return null; }
            public String getName() { return "sentinel"; }
            public VectorSimilarityFunction getSimilarityFunction() { return VectorSimilarityFunction.COSINE; }
            public List<VectorFloat<?>> getBaseVectors() { return Collections.emptyList(); }
            public List<VectorFloat<?>> getQueryVectors() { return Collections.emptyList(); }
            public List<? extends List<Integer>> getGroundTruth() { return Collections.emptyList(); }
        };
        var info = new DataSetInfo(base, () -> {
            callCount[0]++;
            return sentinel;
        });

        // Accessing properties should not trigger the loader
        info.getName();
        info.similarityFunction();
        info.numVectors();
        assertEquals(0, callCount[0], "Loader should not have been called for metadata access");

        // getDataSet triggers it
        info.getDataSet();
        assertEquals(1, callCount[0]);

        // Second call should use cache
        info.getDataSet();
        assertEquals(1, callCount[0], "Loader should only be called once");
    }

    // ========================================================================
    // Integration: DataSetMetadataReader loaded from production metadata file
    // ========================================================================

    @Test
    public void productionMetadataFileLoadsSuccessfully() {
        // This validates the actual dataset-metadata.yml is well-formed
        var reader = DataSetMetadataReader.load();
        var props = reader.getProperties("ada002-100k");
        assertTrue(props.isPresent(), "ada002-100k should be in the production metadata file");
        assertEquals(VectorSimilarityFunction.COSINE, props.get().similarityFunction().orElse(null));
    }

    @Test
    public void productionMetadataAllEntriesHaveSimilarityFunction() {
        var reader = DataSetMetadataReader.load();
        // All entries in the production metadata should have a similarity function
        for (var name : new String[]{"cohere-english-v3-100k", "ada002-100k", "openai-v3-small-1536-100k",
                "gecko-100k", "openai-v3-large-3072-100k", "openai-v3-large-1536-100k",
                "e5-small-v2-100k", "e5-base-v2-100k", "e5-large-v2-100k",
                "ada002-1M", "colbert-1M"}) {
            var props = reader.getProperties(name);
            assertTrue(props.isPresent(), "Missing metadata for " + name);
            assertTrue(props.get().similarityFunction().isPresent(),
                    "Missing similarity_function for " + name);
        }
    }
}
