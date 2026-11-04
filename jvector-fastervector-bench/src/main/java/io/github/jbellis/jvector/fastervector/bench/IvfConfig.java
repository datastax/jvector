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

package io.github.jbellis.jvector.fastervector.bench;

import org.yaml.snakeyaml.LoaderOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.Constructor;
import org.yaml.snakeyaml.error.YAMLException;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * IVF benchmark parameters for one dataset, loaded from
 * {@code jvector-fastervector-bench/yaml-configs/ivf-parameters/<dataset>.yml}, or {@code default.yml}
 * when the dataset has no file of its own. Lists are grids: every combination is run.
 * Paths resolve from the repository root, as in jvector-examples.
 */
public class IvfConfig {
    static final String DEFAULT_DIRECTORY = "jvector-fastervector-bench/yaml-configs/ivf-parameters/";

    public int yamlSchemaVersion;
    public String dataset;
    public Construction construction;
    public Search search;

    /** Build parameters. Each {@code nlist} x {@code storageFactor} pair is a construction point. */
    public static class Construction {
        public List<Integer> nlist;
        public List<Float> storageFactor;
        /** Build thread pool size; 0 uses all CPUs. */
        public int buildThreads = 0;
        /** One directory per SSD. */
        public List<String> ssdPaths;
        public boolean useSavedIndexIfExists = false;
    }

    /** Search parameters, run against every construction point. */
    public static class Search {
        /** topK to the nprobe values to run with it. */
        public Map<Integer, List<Integer>> topKNprobe;
        public List<Integer> searchThreads;
    }

    /** One index to build. */
    public record ConstructionPoint(int nlist, float storageFactor) {
        /** Index directory under each SSD path, so construction points don't overwrite each other. */
        public String subfolder(String datasetName) {
            return datasetName + "/nlist" + nlist + "-sf" + storageFactor;
        }
    }

    /** One search configuration to measure. */
    public record SearchPoint(int topK, int nprobe, int searchThreads) {}

    /**
     * Loads the config for {@code datasetName}, falling back to {@code default.yml}. With the
     * fallback, {@link #dataset} is set to {@code datasetName}.
     */
    public static IvfConfig getDefaultConfig(String datasetName) throws FileNotFoundException {
        String name = datasetName.endsWith(".yml") ? datasetName : datasetName + ".yml";
        File configFile = new File(DEFAULT_DIRECTORY + name);
        if (configFile.exists()) {
            return getConfig(configFile);
        }
        IvfConfig config = getConfig(new File(DEFAULT_DIRECTORY + "default.yml"));
        config.dataset = datasetName;
        return config;
    }

    /** Loads a config by path, or by file name within the default directory. */
    public static IvfConfig getConfig(String configName) throws FileNotFoundException {
        File configFile = new File(configName);
        if (!configFile.exists() && !configFile.isAbsolute()) {
            configFile = new File(DEFAULT_DIRECTORY + configName);
        }
        return getConfig(configFile);
    }

    static IvfConfig getConfig(File configFile) throws FileNotFoundException {
        if (!configFile.exists()) {
            throw new FileNotFoundException(configFile.getAbsolutePath());
        }
        IvfConfig config;
        try (InputStream in = new FileInputStream(configFile)) {
            Yaml yaml = new Yaml(new Constructor(IvfConfig.class, new LoaderOptions()));
            config = yaml.loadAs(in, IvfConfig.class);
        } catch (IOException e) {
            throw new RuntimeException("Failed to read config: " + configFile.getAbsolutePath(), e);
        } catch (YAMLException e) {
            throw new IllegalArgumentException("Invalid config " + configFile.getAbsolutePath() + ": " + e.getMessage(), e);
        }
        if (config == null) {
            throw new IllegalArgumentException("Empty config: " + configFile.getAbsolutePath());
        }
        config.applyOverrides(System.getProperties());
        config.validate(configFile.getAbsolutePath());
        return config;
    }

    /** Machine-specific settings, overridable on the command line so one set of dataset files serves every machine. */
    static final String SSD_PATHS_PROPERTY = "ivf.ssdPaths";
    static final String BUILD_THREADS_PROPERTY = "ivf.buildThreads";
    static final String SEARCH_THREADS_PROPERTY = "ivf.searchThreads";

    /**
     * Replaces the machine-specific settings with any of these properties that are set:
     * {@code ivf.ssdPaths} (comma-separated), {@code ivf.buildThreads}, {@code ivf.searchThreads}
     * (comma-separated). Blank values are ignored.
     */
    void applyOverrides(java.util.Properties props) {
        if (construction == null || search == null) {
            return;   // validate() reports the missing section
        }
        String ssdPaths = props.getProperty(SSD_PATHS_PROPERTY);
        if (ssdPaths != null && !ssdPaths.isBlank()) {
            construction.ssdPaths = splitList(ssdPaths);
            System.out.println("Override " + SSD_PATHS_PROPERTY + ": " + construction.ssdPaths);
        }
        String buildThreads = props.getProperty(BUILD_THREADS_PROPERTY);
        if (buildThreads != null && !buildThreads.isBlank()) {
            construction.buildThreads = parseInt(BUILD_THREADS_PROPERTY, buildThreads.trim());
            System.out.println("Override " + BUILD_THREADS_PROPERTY + ": " + construction.buildThreads);
        }
        String searchThreads = props.getProperty(SEARCH_THREADS_PROPERTY);
        if (searchThreads != null && !searchThreads.isBlank()) {
            search.searchThreads = splitList(searchThreads).stream()
                    .map(s -> parseInt(SEARCH_THREADS_PROPERTY, s))
                    .toList();
            System.out.println("Override " + SEARCH_THREADS_PROPERTY + ": " + search.searchThreads);
        }
    }

    private static List<String> splitList(String value) {
        return java.util.Arrays.stream(value.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .toList();
    }

    private static int parseInt(String property, String value) {
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("-D" + property + " must be an integer, got '" + value + "'");
        }
    }

    void validate(String source) {
        if (yamlSchemaVersion != 1) {
            fail(source, "yamlSchemaVersion must be 1, got " + yamlSchemaVersion);
        }
        if (construction == null) {
            fail(source, "construction is missing");
        }
        if (search == null) {
            fail(source, "search is missing");
        }
        requireNonEmpty(source, "construction.nlist", construction.nlist);
        construction.nlist.forEach(n -> requirePositive(source, "construction.nlist", n));
        requireNonEmpty(source, "construction.storageFactor", construction.storageFactor);
        construction.storageFactor.forEach(f -> {
            if (f == null || !(f > 0)) {
                fail(source, "construction.storageFactor values must be positive, got " + f);
            }
        });
        if (construction.buildThreads < 0) {
            fail(source, "construction.buildThreads must be 0 (all CPUs) or positive, got " + construction.buildThreads);
        }
        requireNonEmpty(source, "construction.ssdPaths", construction.ssdPaths);
        if (search.topKNprobe == null || search.topKNprobe.isEmpty()) {
            fail(source, "search.topKNprobe must not be empty");
        }
        search.topKNprobe.forEach((topK, nprobes) -> {
            requirePositive(source, "search.topKNprobe topK", topK);
            requireNonEmpty(source, "search.topKNprobe[" + topK + "]", nprobes);
            nprobes.forEach(p -> requirePositive(source, "search.topKNprobe[" + topK + "] nprobe", p));
        });
        requireNonEmpty(source, "search.searchThreads", search.searchThreads);
        search.searchThreads.forEach(t -> requirePositive(source, "search.searchThreads", t));
    }

    /** Every {@code nlist} x {@code storageFactor} combination, in config order. */
    public List<ConstructionPoint> constructionPoints() {
        List<ConstructionPoint> points = new ArrayList<>();
        for (int nlist : construction.nlist) {
            for (float storageFactor : construction.storageFactor) {
                points.add(new ConstructionPoint(nlist, storageFactor));
            }
        }
        return points;
    }

    /** Every {@code topK} x {@code nprobe} x {@code searchThreads} combination, in config order. */
    public List<SearchPoint> searchPoints() {
        List<SearchPoint> points = new ArrayList<>();
        search.topKNprobe.forEach((topK, nprobes) -> {
            for (int nprobe : nprobes) {
                for (int threads : search.searchThreads) {
                    points.add(new SearchPoint(topK, nprobe, threads));
                }
            }
        });
        return points;
    }

    public List<Path> ssdPaths() {
        return construction.ssdPaths.stream().map(Path::of).toList();
    }

    private static void requireNonEmpty(String source, String field, List<?> values) {
        if (values == null || values.isEmpty()) {
            fail(source, field + " must not be empty");
        }
    }

    private static void requirePositive(String source, String field, Integer value) {
        if (value == null || value <= 0) {
            fail(source, field + " values must be positive, got " + value);
        }
    }

    private static void fail(String source, String message) {
        throw new IllegalArgumentException("Invalid config " + source + ": " + message);
    }

    // Null-safe: SnakeYAML calls this on a partly built config when it reports a parse error
    @Override
    public String toString() {
        String c = construction == null ? "null"
                : "{nlist=" + construction.nlist
                  + ", storageFactor=" + construction.storageFactor
                  + ", buildThreads=" + construction.buildThreads
                  + ", ssdPaths=" + construction.ssdPaths
                  + ", useSavedIndexIfExists=" + construction.useSavedIndexIfExists + "}";
        String s = search == null ? "null"
                : "{topKNprobe=" + search.topKNprobe + ", searchThreads=" + search.searchThreads + "}";
        return "IvfConfig{dataset=" + dataset + ", construction=" + c + ", search=" + s + "}";
    }
}
