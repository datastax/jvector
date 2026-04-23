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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Pattern;

/// Reads dataset metadata from a multi-entry YAML file and provides keyed lookups
/// for {@link DataSetProperties}.
///
/// This is used by loaders such as {@link DataSetLoaderSimpleMFD}
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
///
/// Keys may also contain glob wildcards (`*` matches any run of characters, `?` matches a single
/// character) which are tried as a fallback after exact and hdf5-suffix lookups fail. For example,
/// a YAML key `"sift1m:label_*"` will match lookups of `sift1m:label_00`, `sift1m:label_01`, etc.
/// A single glob key is expected to match many concrete dataset names — that is the point. However,
/// if two or more glob keys in the same metadata file both match the same concrete lookup, the
/// glob set is ambiguous for that lookup and {@link IllegalStateException} is thrown so the
/// ambiguity can be resolved in the metadata file.
public class DataSetMetadataReader {

    private static final String DEFAULT_FILE = "jvector-examples/yaml-configs/dataset-metadata.yml";
    private static final String MODULE_RELATIVE_DEFAULT_FILE = "yaml-configs/dataset-metadata.yml";

    private final Map<String, Map<String, Object>> metadata;
    private final List<GlobEntry> globEntries;

    private DataSetMetadataReader(Map<String, Map<String, Object>> metadata) {
        this.metadata = metadata != null ? metadata : Map.of();
        this.globEntries = new ArrayList<>();
        for (Map.Entry<String, Map<String, Object>> e : this.metadata.entrySet()) {
            if (containsGlob(e.getKey())) {
                this.globEntries.add(new GlobEntry(e.getKey(), globToPattern(e.getKey()), e.getValue()));
            }
        }
    }

    /// Loads dataset metadata from the default file ({@code jvector-examples/yaml-configs/dataset-metadata.yml}).
    ///
    /// @return the loaded metadata
    /// @throws RuntimeException if the file cannot be read
    public static DataSetMetadataReader load() {
        Path defaultPath = Paths.get(DEFAULT_FILE);
        if (Files.isRegularFile(defaultPath)) {
            return load(defaultPath.toString());
        }

        Path moduleRelativePath = Paths.get(MODULE_RELATIVE_DEFAULT_FILE);
        if (Files.isRegularFile(moduleRelativePath)) {
            return load(moduleRelativePath.toString());
        }

        throw new RuntimeException(
                "Failed to load dataset metadata from default locations: "
                        + defaultPath.toAbsolutePath().normalize()
                        + " or "
                        + moduleRelativePath.toAbsolutePath().normalize());
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

        String alternate = datasetKey.endsWith(".hdf5")
                ? datasetKey.substring(0, datasetKey.length() - ".hdf5".length())
                : datasetKey + ".hdf5";
        entry = metadata.get(alternate);
        if (entry != null) {
            return Optional.of(entry);
        }

        // Collect every glob key that matches this one concrete lookup.
        // A glob is expected to match many different concrete keys across calls; what is NOT
        // acceptable is having >1 glob claim ownership of the same concrete key on one lookup.
        List<GlobEntry> matches = new ArrayList<>();
        for (GlobEntry g : globEntries) {
            if (g.pattern.matcher(datasetKey).matches() || g.pattern.matcher(alternate).matches()) {
                matches.add(g);
            }
        }
        if (matches.size() > 1) {
            List<String> globs = new ArrayList<>();
            for (GlobEntry g : matches) globs.add(g.glob);
            throw new IllegalStateException(
                    "Ambiguous glob match for dataset key '" + datasetKey + "': " + globs);
        }
        if (matches.size() == 1) {
            return Optional.of(matches.get(0).value);
        }

        return Optional.empty();
    }

    private static boolean containsGlob(String s) {
        return s.indexOf('*') >= 0 || s.indexOf('?') >= 0;
    }

    /// Converts a glob pattern (supporting `*` and `?`) to a regex that matches whole strings.
    /// All other characters are treated as literals.
    static Pattern globToPattern(String glob) {
        StringBuilder sb = new StringBuilder(glob.length() + 8);
        int i = 0;
        int literalStart = 0;
        while (i < glob.length()) {
            char c = glob.charAt(i);
            if (c == '*' || c == '?') {
                if (i > literalStart) {
                    sb.append(Pattern.quote(glob.substring(literalStart, i)));
                }
                sb.append(c == '*' ? ".*" : ".");
                literalStart = i + 1;
            }
            i++;
        }
        if (literalStart < glob.length()) {
            sb.append(Pattern.quote(glob.substring(literalStart)));
        }
        return Pattern.compile(sb.toString());
    }

    private static final class GlobEntry {
        final String glob;
        final Pattern pattern;
        final Map<String, Object> value;

        GlobEntry(String glob, Pattern pattern, Map<String, Object> value) {
            this.glob = glob;
            this.pattern = pattern;
            this.value = value;
        }
    }
}
