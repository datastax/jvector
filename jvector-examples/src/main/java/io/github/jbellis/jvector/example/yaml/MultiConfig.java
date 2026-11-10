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
    /** Opt into bounded bulk input loading; omitted settings preserve resident loading. */
    public boolean useDatasetPrefetching = false;
    /** Optional; when [Yes], AutoBenchYAML also runs CompactionBench. Absent defaults to [No]. */
    public List<Boolean> withCompaction;

    private static final String defaultDirectory = "jvector-examples/yaml-configs/index-parameters/";
    private static final java.util.regex.Pattern YAML_SCHEMA_VERSION_KEY =
            java.util.regex.Pattern.compile("(?m)^\\s*yamlSchemaVersion\\s*:");

    private int yamlSchemaVersion;
    private int onDiskIndexVersion;

    /**
     * Returns the default index config for a dataset: {@code index-parameters/<dataset>.yml}, named exactly
     * like the dataset and declaring that dataset. Subdirectories hold examples and are never searched.
     * A dataset without its own file uses {@code default.yml}.
     */
    public static MultiConfig getDefaultConfig(String datasetName) throws FileNotFoundException {
        return getDefaultConfig(new File(defaultDirectory), datasetName);
    }

    static MultiConfig getDefaultConfig(File directory, String datasetName) throws FileNotFoundException {
        String fileName = datasetName.endsWith(".yml") ? datasetName : datasetName + ".yml";
        String[] names = directory.list();
        if (names == null) throw new FileNotFoundException("Index config directory not found: " + directory.getPath());
        // Compare names exactly. File.exists ignores case on some file systems, so lookups would
        // otherwise find different files on macOS and Linux.
        String caseOnly = null;
        for (String name : names) {
            if (name.equals(fileName)) {
                File file = new File(directory, name);
                MultiConfig config = getConfig(file);
                if (config.dataset != null && !config.dataset.equals(datasetName)) {
                    throw new IllegalArgumentException("Index config " + file.getPath() + " declares dataset: "
                            + config.dataset + "." + System.lineSeparator()
                            + "A default config's file name must match its dataset field exactly.");
                }
                return config;
            }
            if (name.equalsIgnoreCase(fileName) && new File(directory, name).isFile()) caseOnly = name;
        }
        if (caseOnly != null) {
            throw new IllegalArgumentException("No index config is named exactly " + fileName + "; found"
                    + System.lineSeparator() + "  " + new File(directory, caseOnly).getPath()
                    + System.lineSeparator() + "File names must match the dataset name exactly, including case.");
        }
        File fallback = new File(directory, "default.yml");
        System.err.println("WARNING: No index config named " + fileName + "; using " + fallback.getPath() + "."
                + System.lineSeparator() + "Dataset defaults are named <dataset>.yml in " + directory.getPath() + ".");
        MultiConfig config = getConfig(fallback);
        config.dataset = datasetName;
        return config;
    }

    public static MultiConfig getConfig(String configName) throws FileNotFoundException {
        File configFile = new File(configName);
        // If the file doesn't exist as an absolute path, try relative to the default directory
        if (!configFile.exists() && !configName.startsWith("/") && !configName.contains(":")) {
            File relative = new File(defaultDirectory + configName);
            if (!relative.exists()) throw new FileNotFoundException("Index config not found: " + configName);
            configFile = relative;
        }
        return getConfig(configFile);
    }

    static MultiConfig getConfig(File configFile) throws FileNotFoundException {
        if (!configFile.exists()) {
            throw new FileNotFoundException("Index config not found: " + configFile.getPath());
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
                if (cfg.getYamlSchemaVersion() != 2 && cfg.getYamlSchemaVersion() != 3) {
                    throw new IllegalArgumentException("Unsupported yamlSchemaVersion="
                            + cfg.getYamlSchemaVersion() + " in " + configFile.getAbsolutePath());
                }
                try {
                    if (cfg.usesSchema3()) cfg.validateSchema3();
                    else cfg.validateSchema2();
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException(configFile.getAbsolutePath() + ": " + e.getMessage(), e);
                }

                return cfg;
            } catch (IOException e) {
                throw new RuntimeException("Failed to read config: " + configFile.getAbsolutePath(), e);
            }
        }


        // Legacy yamlSchemaVersion "0": lenient parsing (ignore unknown fields like search.benchmarks)
        if (WARNED_LEGACY.compareAndSet(false, true)) {
            System.err.println("WARNING: Deprecated legacy YAML schema detected (no yamlSchemaVersion). "
                    + "Unknown fields will be ignored. Please migrate configs to yamlSchemaVersion: 1 and run-config.yml.");
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

    /** Returns true only if withCompaction is present and set to Yes; defaults to false. */
    public boolean isCompactionEnabled() {
        return withCompaction != null && withCompaction.contains(Boolean.TRUE);
    }

    public int getYamlSchemaVersion() {
        return yamlSchemaVersion;
    }

    public boolean usesSchema3() {
        return yamlSchemaVersion == 3;
    }

    private void validateSchema2() {
        if (construction != null && (construction.type != null || construction.writerMode != null
                || (construction.reranking != null && !(construction.reranking instanceof List<?>)))) {
            throw new IllegalArgumentException("construction.type, writerMode, and reranking.type require schema 3");
        }
        if ((construction != null && construction.workers != null) || (search != null && search.workers != null)) {
            throw new IllegalArgumentException("construction.workers and search.workers require schema 3");
        }
        if (hasSchema3CompressionFields(construction) || hasSchema3CompressionFields(search)) {
            throw new IllegalArgumentException("compression scoring and ASH query settings require schema 3");
        }
    }

    private static boolean hasSchema3CompressionFields(CommonParameters section) {
        if (section == null || section.compression == null) return false;
        for (var entry : section.compression) {
            if (entry == null) continue;
            if (entry.scoring != null) return true;
            if ("ASH".equals(entry.type) && entry.parameters != null
                    && (entry.parameters.containsKey("queryExecution")
                    || entry.parameters.containsKey("maxBatchSize")
                    || entry.parameters.containsKey("queryProjection"))) return true;
        }
        return false;
    }

    private void validateSchema3() {
        if (dataset == null || dataset.isBlank() || construction == null || search == null) {
            throw new IllegalArgumentException("schema 3 requires dataset, construction, and search");
        }
        construction.validateSchema3();
        search.validateSchema3();
        var reranking = construction.rerankingType();
        boolean fused = construction.fusedGraph != null && construction.fusedGraph.contains(Boolean.TRUE);
        if (fused && "None".equals(reranking)) {
            throw new IllegalArgumentException("fusedGraph requires construction.reranking.type FP32 or NVQ");
        }
        for (var entry : search.compression) {
            if (fused && !List.of("ASH", "PQ").contains(entry.type)) {
                throw new IllegalArgumentException("fusedGraph requires ASH or PQ search compression");
            }
            if ("None".equals(entry.type) && !"None".equals(reranking)) {
                throw new IllegalArgumentException("Search type None scores with full-precision vectors, so "
                        + reranking + " reranking does not apply. Use reranking.type None");
            }
        }
    }

    /**
     * The pipeline this schema 3 config selects, inferred from its fields, for example
     * {@code onDisk: build ASH-2b symmetric -> search fused ASH-2b -> rerank NVQ}.
     * Sweeps list each value. Returns null for older schemas.
     */
    public String describePipeline() {
        return pipeline(true);
    }

    /** As {@link #describePipeline()} without quantizer settings, so configs with one pipeline compare equal. */
    public String pipelineSignature() {
        return pipeline(false);
    }

    private String pipeline(boolean detailed) {
        if (!usesSchema3()) return null;
        String fused = "";
        if (construction.fusedGraph != null) {
            boolean yes = construction.fusedGraph.contains(Boolean.TRUE);
            boolean no = construction.fusedGraph.contains(Boolean.FALSE);
            fused = yes && no ? "fused or separate " : yes ? "fused " : "";
        }
        return construction.type
                + ": build " + describe(construction.compression, detailed)
                + " -> search " + fused + describe(search.compression, detailed)
                + " -> rerank " + construction.rerankingType();
    }

    private static String describe(List<Compression> entries, boolean detailed) {
        var parts = new ArrayList<String>();
        for (var entry : entries) {
            String part = entry.type;
            if (detailed && "ASH".equals(entry.type)) {
                Map<String, ?> parameters = entry.parameters == null ? Map.of() : entry.parameters;
                Object bits = parameters.get("bitsPerDimension");
                part += "-" + (bits == null ? "1" : bits) + "b";
                Object landmarks = parameters.get("landmarkCount");
                if (landmarks != null && !"1".equals(String.valueOf(landmarks))) part += " " + landmarks + " landmarks";
            }
            if (detailed && entry.scoring != null) part += " " + entry.scoring;
            parts.add(part);
        }
        return String.join(" | ", parts);
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
