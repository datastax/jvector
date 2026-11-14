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

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Collectors;

import static org.junit.Assert.*;

/** Dataset default lookup, the checked-in index parameters, and the example pipelines. */
public class IndexParametersTest {
    private static final Path INDEX_PARAMETERS = Path.of("jvector-examples/yaml-configs/index-parameters");
    private static final String NL = System.lineSeparator();
    @Rule public final TemporaryFolder directory = new TemporaryFolder();

    private static String config(String dataset) {
        return "yamlSchemaVersion: 3\nonDiskIndexVersion: 7\ndataset: " + dataset + "\n"
                + "construction:\n  type: inMemory\n  outDegree: [16]\n  efConstruction: [50]\n"
                + "  neighborOverflow: [1.2f]\n  addHierarchy: [No]\n  refineFinalGraph: [Yes]\n"
                + "  compression:\n    - type: None\n  reranking:\n    type: None\n"
                + "search:\n  topKOverquery:\n    10: [1.0]\n  useSearchPruning: [No]\n"
                + "  compression:\n    - type: None\n";
    }

    private File write(String name, String dataset) throws Exception {
        var file = new File(directory.getRoot(), name);
        Files.writeString(file.toPath(), config(dataset));
        return file;
    }

    @Test
    public void fullPrecisionSearchRejectsAnyReranker() throws Exception {
        for (var reranker : List.of("NVQ", "FP32")) {
            var file = new File(directory.getRoot(), "rerank-" + reranker + ".yml");
            Files.writeString(file.toPath(), config("test").replace("reranking:\n    type: None", "reranking:\n    type: " + reranker));
            var e = assertThrows(IllegalArgumentException.class, () -> MultiConfig.getConfig(file.getPath()));
            assertTrue(e.getMessage(), e.getMessage().endsWith("Search type None scores with full-precision vectors, so "
                    + reranker + " reranking does not apply. Use reranking.type None"));
        }
    }

    @Test
    public void exactNameFindsTheDatasetDefault() throws Exception {
        write("mpnet-1m.yml", "mpnet-1m");
        write("default.yml", "default");
        assertEquals("mpnet-1m", MultiConfig.getDefaultConfig(directory.getRoot(), "mpnet-1m").dataset);
    }

    @Test
    public void mismatchedDatasetFieldIsRejected() throws Exception {
        var file = write("cap-6m.yml", "cap-1M");
        var e = assertThrows(IllegalArgumentException.class,
                () -> MultiConfig.getDefaultConfig(directory.getRoot(), "cap-6m"));
        assertEquals("Index config " + file.getPath() + " declares dataset: cap-1M." + NL
                + "A default config's file name must match its dataset field exactly.", e.getMessage());
    }

    @Test
    public void caseOnlyMatchIsRejectedOnEveryFileSystem() throws Exception {
        var file = write("mpnet-1M.yml", "mpnet-1m");
        write("default.yml", "default");
        var e = assertThrows(IllegalArgumentException.class,
                () -> MultiConfig.getDefaultConfig(directory.getRoot(), "mpnet-1m"));
        assertEquals("No index config is named exactly mpnet-1m.yml; found" + NL + "  " + file.getPath() + NL
                + "File names must match the dataset name exactly, including case.", e.getMessage());
    }

    @Test
    public void subdirectoriesAreNotSearchedAndMissingDefaultsWarn() throws Exception {
        assertTrue(new File(directory.getRoot(), "in-memory-fp").mkdir());
        write("in-memory-fp/mpnet-5m.yml", "mpnet-5m");
        var fallback = write("default.yml", "default");
        var err = new ByteArrayOutputStream();
        var previous = System.err;
        MultiConfig config;
        try {
            System.setErr(new PrintStream(err, true, StandardCharsets.UTF_8));
            config = MultiConfig.getDefaultConfig(directory.getRoot(), "mpnet-5m");
        } finally {
            System.setErr(previous);
        }
        assertEquals("mpnet-5m", config.dataset);
        assertEquals("WARNING: No index config named mpnet-5m.yml; using " + fallback.getPath() + "." + NL
                + "Dataset defaults are named <dataset>.yml in " + directory.getRoot().getPath() + "." + NL,
                err.toString(StandardCharsets.UTF_8));
    }

    @Test
    public void missingExplicitConfigNamesThePath() {
        var e = assertThrows(FileNotFoundException.class,
                () -> MultiConfig.getConfig("ash-options/no-such-example.yml"));
        assertEquals("Index config not found: ash-options/no-such-example.yml", e.getMessage());
    }

    @Test
    public void checkedInDefaultsAreFoundByTheirExactDatasetNames() throws Exception {
        List<String> names;
        try (var files = Files.list(INDEX_PARAMETERS)) {
            names = files.map(p -> p.getFileName().toString())
                    .filter(n -> n.endsWith(".yml") && !n.equals("default.yml")).sorted().collect(Collectors.toList());
        }
        assertTrue(names.size() > 20);
        for (String name : names) {
            String dataset = name.substring(0, name.length() - ".yml".length());
            assertEquals(name, dataset, MultiConfig.getDefaultConfig(INDEX_PARAMETERS.toFile(), dataset).dataset);
        }
    }

    private static Map<Path, MultiConfig> allConfigs() throws Exception {
        var configs = new TreeMap<Path, MultiConfig>();
        try (var files = Files.walk(INDEX_PARAMETERS)) {
            for (var path : files.filter(p -> p.toString().endsWith(".yml")).collect(Collectors.toList())) {
                configs.put(INDEX_PARAMETERS.relativize(path), MultiConfig.getConfig(path.toString()));
            }
        }
        return configs;
    }

    @Test
    public void everyExampleInAFolderUsesOnePipeline() throws Exception {
        var byFolder = new TreeMap<String, Map<String, List<Path>>>();
        for (var entry : allConfigs().entrySet()) {
            Path relative = entry.getKey();
            if (relative.getNameCount() != 2 || relative.startsWith("studies")) continue;
            byFolder.computeIfAbsent(relative.getName(0).toString(), k -> new TreeMap<>())
                    .computeIfAbsent(entry.getValue().pipelineSignature(), k -> new ArrayList<>()).add(relative);
        }
        assertTrue(byFolder.size() >= 10);
        for (var folder : byFolder.entrySet()) {
            if (folder.getValue().size() == 1) continue;
            var listing = new StringBuilder();
            folder.getValue().forEach((pipeline, files) ->
                    files.forEach(f -> listing.append(NL).append("  ").append(f).append(": ").append(pipeline)));
            fail("Files in " + folder.getKey() + "/ select different pipelines:" + listing + NL
                    + "Every file in an example folder must use that folder's pipeline.");
        }
    }

    @Test
    public void everyConfigurationOptionHasAnExample() throws Exception {
        var used = new LinkedHashMap<String, Set<String>>();
        for (var config : allConfigs().values()) {
            var c = config.construction;
            add(used, "construction.type", c.type);
            add(used, "construction.writerMode", c.writerMode);
            add(used, "construction.reranking.type", c.rerankingType());
            if (c.fusedGraph != null) c.fusedGraph.forEach(v -> add(used, "construction.fusedGraph", v ? "Yes" : "No"));
            for (var entry : c.compression) {
                add(used, "construction.compression.type", entry.type);
                add(used, "construction.compression.scoring", entry.scoring);
                if (entry.parameters != null) add(used, "ASH optimizer", entry.parameters.get("optimizer"));
            }
            add(used, "search.queryExecution", config.search.queryExecution());
            for (var entry : config.search.compression) {
                add(used, "search.compression.type", entry.type);
                if ("ASH".equals(entry.type) && entry.parameters != null) {
                    add(used, "ASH queryProjection", entry.parameters.get("queryProjection"));
                }
            }
        }
        var required = new LinkedHashMap<String, List<String>>();
        required.put("construction.type", List.of("inMemory", "onDisk"));
        required.put("construction.writerMode", List.of("sequential", "parallel"));
        required.put("construction.fusedGraph", List.of("Yes", "No"));
        required.put("construction.compression.type", List.of("None", "ASH", "PQ", "BQ"));
        required.put("construction.compression.scoring", List.of("symmetric", "asymmetric", "mixed"));
        required.put("search.compression.type", List.of("None", "ASH", "PQ", "BQ"));
        required.put("construction.reranking.type", List.of("None", "FP32", "NVQ"));
        required.put("ASH optimizer", List.of("ITQ", "RANDOM"));
        required.put("ASH queryProjection", List.of("auto", "fp32", "fp16", "int8"));
        required.put("search.queryExecution", List.of("direct", "queued"));
        var missing = new ArrayList<String>();
        required.forEach((field, values) -> values.stream()
                .filter(v -> !used.getOrDefault(field, Set.of()).contains(v))
                .forEach(v -> missing.add("No index config uses " + field + ": " + v + ".")));
        assertTrue(String.join(NL, missing), missing.isEmpty());
    }

    private static void add(Map<String, Set<String>> used, String field, Object value) {
        if (value != null) used.computeIfAbsent(field, k -> new TreeSet<>()).add(String.valueOf(value));
    }

    @Test
    public void pipelineIsInferredFromTheFields() throws Exception {
        var compressed = MultiConfig.getConfig(INDEX_PARAMETERS.resolve("compressed-build-fused-ash-nvq/cap-1M-2bit.yml").toString());
        assertEquals("onDisk: build ASH-2b symmetric -> search fused ASH-2b -> rerank NVQ", compressed.describePipeline());
        assertEquals("onDisk: build ASH -> search fused ASH -> rerank NVQ", compressed.pipelineSignature());
        var memory = MultiConfig.getConfig(INDEX_PARAMETERS.resolve("in-memory-ash/mpnet-1m.yml").toString());
        assertEquals("inMemory: build FP32 -> search ASH-2b -> rerank FP32", memory.describePipeline());
        var uncompressed = MultiConfig.getConfig(INDEX_PARAMETERS.resolve("glove-100-angular.yml").toString());
        assertEquals("onDisk: build FP32 -> search FP32 -> rerank None", uncompressed.describePipeline());
        var landmarks = MultiConfig.getConfig(INDEX_PARAMETERS.resolve("ash-options/ada002-1M-4-landmarks.yml").toString());
        assertEquals("onDisk: build ASH-2b 4 landmarks asymmetric -> search fused ASH-2b 4 landmarks -> rerank NVQ",
                landmarks.describePipeline());
    }
}
