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

import org.yaml.snakeyaml.Yaml;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/// Reads dataset metadata from a multi-entry YAML file and provides keyed lookups
/// for {@link DataSetProperties}.
///
/// This is used by loaders such as {@link DataSetLoaderMFD} and {@link DataSetLoaderHDF5}
/// that do not have an intrinsic way to determine the similarity function from the dataset
/// name or file format alone.
///
/// The YAML file maps dataset keys to their metadata properties using the same key names
/// as the {@code KEY_*} constants on {@link DataSetProperties}:
/// ```yaml
/// ada002-100k:
///   similarity_function: COSINE
///   is_normalized: true
/// ```
///
/// For single-entry lookups without caching, use
/// {@link DataSetProperties.PropertyMap#PropertyMap(String, String)} directly.
/// This class is useful when the same file is queried repeatedly for different keys.
///
/// Keys may or may not include file extensions (e.g. {@code .hdf5}). The lookup tries
/// the exact key first, then falls back to the key with {@code .hdf5} appended.
public class DataSetMetadataReader {

    private static final String DEFAULT_FILE = "jvector-examples/yaml-configs/dataset_metadata.yml";

    private final Map<String, Map<String, Object>> metadata;

    private DataSetMetadataReader(Map<String, Map<String, Object>> metadata) {
        this.metadata = metadata != null ? metadata : Map.of();
    }

    /// Loads dataset metadata from the default file ({@code jvector-examples/yaml-configs/dataset_metadata.yml}).
    ///
    /// @return the loaded metadata
    /// @throws RuntimeException if the file cannot be read
    public static DataSetMetadataReader load() {
        return load(DEFAULT_FILE);
    }

    /// Loads dataset metadata from the specified file.
    ///
    /// @param file path to the YAML metadata file
    /// @return the loaded metadata
    /// @throws RuntimeException if the file cannot be read
    @SuppressWarnings("unchecked")
    public static DataSetMetadataReader load(String file) {
        try (InputStream inputStream = new FileInputStream(file)) {
            Map<String, Map<String, Object>> data = new Yaml().load(inputStream);
            return new DataSetMetadataReader(data);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load dataset metadata from " + file, e);
        }
    }

    /// Looks up the {@link DataSetProperties} for a dataset by key.
    ///
    /// The lookup first tries the exact key. If that is not found, it also tries the
    /// corresponding key with or without the {@code .hdf5} suffix so that callers may
    /// use either form.
    ///
    /// The matched YAML entry is wrapped in a {@link DataSetProperties.PropertyMap}
    /// with the requested dataset key injected as the dataset name when no explicit
    /// name is present. Properties not present in the YAML default to empty/false/zero.
    ///
    /// @param datasetKey the dataset name or filename to look up
    /// @return the dataset properties if an entry exists, or empty if no entry is found
    public Optional<DataSetProperties> getProperties(String datasetKey) {
        return findEntry(datasetKey).map(entry -> {
            var props = new HashMap<>(entry);
            props.putIfAbsent(DataSetProperties.KEY_NAME, datasetKey);
            return new DataSetProperties.PropertyMap(props);
        });
    }

    private Optional<Map<String, Object>> findEntry(String datasetKey) {
        Map<String, Object> entry = metadata.get(datasetKey);
        if (entry != null) {
            return Optional.of(entry);
        }

        if (datasetKey.endsWith(".hdf5")) {
            return Optional.ofNullable(metadata.get(datasetKey.substring(0, datasetKey.length() - ".hdf5".length())));
        }

        return Optional.ofNullable(metadata.get(datasetKey + ".hdf5"));
    }
}
