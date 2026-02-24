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

import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import org.yaml.snakeyaml.LoaderOptions;
import org.yaml.snakeyaml.constructor.Constructor;
import org.yaml.snakeyaml.Yaml;

import java.io.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class MultiConfig {
    public ConstructionParameters construction;
    public SearchParameters search;
    public String dataset;

    private static final String defaultDirectory = "jvector-examples/yaml-configs/";
    private static final java.util.regex.Pattern YAML_SCHEMA_VERSION_KEY =
            java.util.regex.Pattern.compile("(?m)^\\s*yamlSchemaVersion\\s*:");

    private int yamlSchemaVersion;
    private int onDiskIndexVersion;

    private static final java.util.concurrent.ConcurrentSkipListSet<String> DEFAULT_USED_FOR_DATASETS =
            new java.util.concurrent.ConcurrentSkipListSet<>();

    private static final java.util.concurrent.atomic.AtomicReference<File> DEFAULT_FILE_USED =
            new java.util.concurrent.atomic.AtomicReference<>();

    public static MultiConfig getDefaultConfig(String datasetName) throws FileNotFoundException {
        var name = defaultDirectory + datasetName;
        if (!name.endsWith(".yml")) {
            name += ".yml";
        }
        File configFile = new File(name);
        boolean useDefault = !configFile.exists();
        if (useDefault) {
            configFile = new File(defaultDirectory + "default.yml");

            // Record fallback usage for a compact summary print later
            DEFAULT_FILE_USED.compareAndSet(null, configFile);
            DEFAULT_USED_FOR_DATASETS.add(datasetName);
        }

        var config = getConfig(configFile);

        if (useDefault) {
            config.dataset = datasetName;
        }

        return config;
    }

    public static MultiConfig getConfig(String configName) throws FileNotFoundException {
        File configFile = new File(configName);
        // If the file doesn't exist as an absolute path, try relative to the default directory
        if (!configFile.exists() && !configName.startsWith("/") && !configName.contains(":")) {
            configFile = new File(defaultDirectory + configName);
        }
        return getConfig(configFile);
    }

    static MultiConfig getConfig(File configFile) throws FileNotFoundException {
        if (!configFile.exists()) {
            throw new FileNotFoundException(configFile.getAbsolutePath());
        }

        final String text;
        try {
            text = Files.readString(configFile.toPath(), java.nio.charset.StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException("Failed to read config: " + configFile.getAbsolutePath(), e);
        }

        // New schema: strict parsing (and validation)
        if (YAML_SCHEMA_VERSION_KEY.matcher(text).find()) {
            try (InputStream inputStream = new FileInputStream(configFile)) {
                LoaderOptions loaderOptions = new LoaderOptions();
                Constructor ctor = new Constructor(MultiConfig.class, loaderOptions);
                Yaml yaml = new Yaml(ctor);

                // Checks onDiskIndexVersion is valid (if present), among other things
                MultiConfig cfg = yaml.loadAs(inputStream, MultiConfig.class);

                // Check if onDiskIndexVersion was provided at all
                if (cfg.getOnDiskIndexVersion() == 0) {
                    throw new IllegalArgumentException("Required field 'onDiskIndexVersion' is missing in "
                            + configFile.getAbsolutePath());
                }

                // Validate supported versions
                if (cfg.getYamlSchemaVersion() != 1) {
                    throw new IllegalArgumentException("Unsupported yamlSchemaVersion="
                            + cfg.getYamlSchemaVersion() + " in " + configFile.getAbsolutePath());
                }

                return cfg;
            } catch (IOException e) {
                throw new RuntimeException("Failed to read config: " + configFile.getAbsolutePath(), e);
            }
        }


        // Legacy yamlSchemaVersion "0": lenient parsing (ignore unknown fields like search.benchmarks)
        if (WARNED_LEGACY.compareAndSet(false, true)) {
            System.err.println("WARNING: Deprecated legacy YAML schema detected (no yamlSchemaVersion). "
                    + "Unknown fields will be ignored. Please migrate configs to yamlSchemaVersion: 1 and run.yml.");
        }

        try {
            LoaderOptions loaderOptions = new LoaderOptions();
            Constructor ctor = new Constructor(MultiConfig.class, loaderOptions);

            // Ignore unknown properties
            ctor.getPropertyUtils().setSkipMissingProperties(true);

            // Legacy load
            Yaml yaml = new Yaml(ctor);

            MultiConfig cfg;
            try (InputStream inputStream = new FileInputStream(configFile)) {
                cfg = yaml.loadAs(inputStream, MultiConfig.class);
            }

            // Optional: extract legacy search.benchmarks if present, without reintroducing the field.
            cfg.legacySearchBenchmarks = extractLegacySearchBenchmarks(text);

            // Make yamlSchemaVersion real for legacy loads too
            cfg.setYamlSchemaVersion(0);

            return cfg;
        } catch (IOException e) {
            throw new RuntimeException("Failed to read config: " + configFile.getAbsolutePath(), e);
        }
    }

    // Legacy schema (yamlSchemaVersion absent).
    // Not part of the v1 schema; safe to remove later.
    public transient Map<String, List<String>> legacySearchBenchmarks;

    private static final java.util.concurrent.atomic.AtomicBoolean WARNED_LEGACY =
            new java.util.concurrent.atomic.AtomicBoolean(false);


    /** Back-compat for old yaml files that still use `version:`. */
    @Deprecated
    public void setVersion(int version) {
        setOnDiskIndexVersion(version);
    }

    @Deprecated
    public int getVersion() {
        return getOnDiskIndexVersion();
    }

    public int getYamlSchemaVersion() {
        return yamlSchemaVersion;
    }

    public void setYamlSchemaVersion(int yamlSchemaVersion) {
        this.yamlSchemaVersion = yamlSchemaVersion;
    }

    public int getOnDiskIndexVersion() {
        return onDiskIndexVersion;
    }

    public void setOnDiskIndexVersion(int onDiskIndexVersion) {
        if (onDiskIndexVersion != OnDiskGraphIndex.CURRENT_VERSION) {
            throw new IllegalArgumentException("YAML config specifies invalid onDiskIndexVersion: " + onDiskIndexVersion
            + " (version " + OnDiskGraphIndex.CURRENT_VERSION + " expected)");
        }
        this.onDiskIndexVersion = onDiskIndexVersion;
    }

    public static void printDefaultConfigUsageSummary() {
        File f = DEFAULT_FILE_USED.get();
        if (f == null) return;

        var datasets = new java.util.ArrayList<>(DEFAULT_USED_FOR_DATASETS);
        if (datasets.isEmpty()) return;

        // Print a wrapped bracket-list similar to "Executing the following datasets"
        System.out.println("Default YAML used for datasets: " + wrapList(datasets, 6, "  "));
    }

    private static String wrapList(java.util.List<String> items, int perLine, String indent) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) sb.append(", ");
            if (i > 0 && (i % perLine) == 0) sb.append(System.lineSeparator()).append(indent);
            sb.append(items.get(i));
        }
        sb.append("]");
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, List<String>> extractLegacySearchBenchmarks(String yamlText) {
        try {
            Object rootObj = new Yaml().load(yamlText);
            if (!(rootObj instanceof Map)) return null;

            Map<String, Object> root = (Map<String, Object>) rootObj;
            Object searchObj = root.get("search");
            if (!(searchObj instanceof Map)) return null;

            Map<String, Object> search = (Map<String, Object>) searchObj;
            Object benchObj = search.get("benchmarks");
            if (!(benchObj instanceof Map)) return null;

            Map<String, Object> bench = (Map<String, Object>) benchObj;
            Map<String, List<String>> out = new LinkedHashMap<>();

            for (var e : bench.entrySet()) {
                String k = String.valueOf(e.getKey());
                Object v = e.getValue();
                if (v instanceof List) {
                    List<?> rawList = (List<?>) v;
                    List<String> strs = new ArrayList<>(rawList.size());
                    for (Object item : rawList) strs.add(String.valueOf(item));
                    out.put(k, strs);
                }
            }
            return out.isEmpty() ? null : out;
        } catch (Throwable t) {
            return null; // best-effort; legacy support only
        }
    }
}
