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

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSetSpec;
import org.yaml.snakeyaml.Yaml;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/// The named sections of `datasets.yml`, each a list of datasets to benchmark.
///
/// A list entry is either a plain string in {@link DataSetSpec} sugared form, e.g.
/// `cohere-english-v3-100k` or `cohere-english-v3-100k(mmap)`, or a structured map:
/// ```yaml
/// regression-tests:
///   - cap-1M
///   - name: cohere-english-v3-1M
///     profile: default
///     wrappers:
///       - mmap
///       - lru: { grain: 4096, capacityMb: 512 }
/// ```
/// Both forms are exposed as their canonical sugared string, which {@link
/// io.github.jbellis.jvector.example.benchmarks.datasets.DataSets} parses back into a spec, so
/// existing name-based filtering and configuration lookups keep working.
public class DatasetCollection {
    private static final String defaultFile = "./jvector-examples/yaml-configs/datasets.yml";

    public final Map<String, List<String>> datasetNames;

    private DatasetCollection(Map<String, List<String>> datasetNames) {
        this.datasetNames = datasetNames;
    }

    public static DatasetCollection load() throws IOException  {
        return load(defaultFile);
    }

    public static DatasetCollection load(String file) throws IOException  {
        try (InputStream inputStream = new FileInputStream(file)) {
            Yaml yaml = new Yaml();
            Map<String, List<Object>> raw = yaml.load(inputStream);
            return new DatasetCollection(canonicalize(raw));
        }
    }

    /// Converts each section's entries to canonical sugared dataset spec strings.
    ///
    /// @param raw the parsed YAML: section name to list of string or map entries
    /// @return section name to list of canonical spec strings; null sections are preserved as null
    static Map<String, List<String>> canonicalize(Map<String, List<Object>> raw) {
        Map<String, List<String>> result = new LinkedHashMap<>();
        if (raw == null) {
            return result;
        }
        for (var section : raw.entrySet()) {
            List<Object> entries = section.getValue();
            if (entries == null) {
                result.put(section.getKey(), null);
                continue;
            }
            List<String> specs = new ArrayList<>(entries.size());
            for (Object entry : entries) {
                try {
                    specs.add(DataSetSpec.from(entry).toString());
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException("Invalid dataset entry in section '" + section.getKey() + "': " + e.getMessage(), e);
                }
            }
            result.put(section.getKey(), specs);
        }
        return result;
    }

    public List<String> getAll() {
        List<String> allDatasetNames = new ArrayList<>();
        for (var key : datasetNames.keySet()) {
            var subList = datasetNames.get(key);
            if (subList != null) {
                allDatasetNames.addAll(subList);
            }
        }
        return allDatasetNames;
    }

    public List<String> getSection(String section) {
        List<String> sectionDatasetNames = new ArrayList<>();
        for (var key : datasetNames.keySet()) {
            if (key.equals(section)) {
                var subList = datasetNames.get(key);
                if (subList != null) {
                    sectionDatasetNames.addAll(subList);
                }
            }
        }
        return sectionDatasetNames;
    }
}
