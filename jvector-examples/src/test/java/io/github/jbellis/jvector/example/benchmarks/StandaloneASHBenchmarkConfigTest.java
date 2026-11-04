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
package io.github.jbellis.jvector.example.benchmarks;

import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.example.yaml.RunConfig;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.yaml.snakeyaml.Yaml;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

/** Fair code budgets and independent selection of checked-in run/index YAML. */
public class StandaloneASHBenchmarkConfigTest {
    private static final Path CONFIGS = Path.of("jvector-examples/yaml-configs/ash-pq");
    @Rule public final TemporaryFolder directory = new TemporaryFolder();

    @Test
    public void explicitRunFileDoesNotRequireEditingTheDefaultYaml() throws Exception {
        var path = directory.newFile("run.yml").toPath().toAbsolutePath();
        Files.writeString(path, "yamlSchemaVersion: 1\nqueryExecution:\n  mode: queued\n");
        String previous = System.getProperty("jvector.bench.runConfig");
        try {
            System.setProperty("jvector.bench.runConfig", path.toString());
            assertEquals("queued", RunConfig.loadDefault().queryExecution.mode);
            Files.writeString(path, "");
            assertThrows(IllegalArgumentException.class, RunConfig::loadDefault);
        } finally {
            if (previous == null) System.clearProperty("jvector.bench.runConfig");
            else System.setProperty("jvector.bench.runConfig", previous);
        }
    }

    @Test
    public void yamlPathsAreNotRegexesAndMissingExplicitRunFileFailsBeforeDataLoading() throws Exception {
        var index = directory.newFile("index with [literal.yml").toPath();
        Files.copy(CONFIGS.resolve("mpnet-2bit-ash.yml"), index,
                java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        var missing = directory.getRoot().toPath().resolve("missing-run.yml");
        String previous = System.getProperty("jvector.bench.runConfig");
        try {
            System.setProperty("jvector.bench.runConfig", missing.toString());
            var failure = assertThrows(java.io.FileNotFoundException.class,
                    () -> io.github.jbellis.jvector.example.BenchYAML.main(new String[]{index.toString()}));
            assertTrue(failure.getMessage().contains("missing-run.yml"));
        } finally {
            if (previous == null) System.clearProperty("jvector.bench.runConfig");
            else System.setProperty("jvector.bench.runConfig", previous);
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> yaml(Path path) throws Exception {
        return new Yaml().load(Files.readString(path));
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> compression(Map<String, Object> section) {
        return ((List<Map<String, Object>>) section.get("compression")).get(0);
    }

    @Test
    @SuppressWarnings("unchecked")
    public void allPairsMatchBytesAndUseFusedGraphsNvqAndTheSameSearchParameters() throws Exception {
        var dimensions = Map.of("mpnet", 768, "cohere", 1024, "ada002", 1536, "openai3072", 3072);
        for (var entry : dimensions.entrySet()) {
            for (int bits : new int[]{2, 4}) {
                Map<String, Object> previousConstruction = null;
                String previousDataset = null;
                for (String algorithm : List.of("ash", "pq")) {
                    Path path = CONFIGS.resolve(entry.getKey() + "-" + bits + "bit-" + algorithm + ".yml");
                    assertNotNull(MultiConfig.getConfig(path.toAbsolutePath().toString()));
                    var root = yaml(path);
                    var construction = (Map<String, Object>) root.get("construction");
                    var search = (Map<String, Object>) root.get("search");
                    assertEquals(List.of(true), construction.get("fusedGraph"));
                    assertEquals(List.of("NVQ"), construction.get("reranking"));
                    assertEquals(List.of(32), construction.get("outDegree"));
                    assertEquals(List.of(100), construction.get("efConstruction"));
                    assertEquals(compression(construction), compression(search));
                    assertEquals(Map.of(10, List.of(1.0, 2.0)), search.get("topKOverquery"));
                    var parameters = (Map<String, Object>) compression(construction).get("parameters");
                    int expectedBits = 40 + entry.getValue() / 2 * bits;
                    if (algorithm.equals("ash")) {
                        assertEquals(expectedBits, parameters.get("encodedBits"));
                        assertEquals(1, parameters.get("landmarkCount"));
                        assertEquals(bits, parameters.get("bitsPerDimension"));
                        assertEquals("ITQ", parameters.get("optimizer"));
                    } else {
                        assertEquals(expectedBits / 8, parameters.get("m"));
                        assertEquals(256, parameters.get("k"));
                        assertEquals(previousDataset, root.get("dataset"));
                        for (String field : List.of("outDegree", "efConstruction", "neighborOverflow",
                                "addHierarchy", "refineFinalGraph", "fusedGraph", "reranking", "useSavedIndexIfExists"))
                            assertEquals(previousConstruction.get(field), construction.get(field));
                    }
                    previousDataset = (String) root.get("dataset");
                    previousConstruction = construction;
                }
            }
        }
    }

    @Test
    public void runModesKeepStandardBenchmarkSelectionsAndQueuedBatchDefaults() throws Exception {
        for (String mode : List.of("construction", "direct", "queued")) {
            var config = RunConfig.load(CONFIGS.resolve("run-" + mode + ".yml").toString());
            assertEquals(mode.equals("queued") ? "queued" : "direct", config.queryExecution.mode);
            assertEquals(8, config.queryExecution.maxBatchSize);
            assertEquals(List.of("recall"), config.benchmarks.get("accuracy"));
            if (!mode.equals("construction")) {
                assertEquals(List.of("AVG"), config.benchmarks.get("throughput"));
                assertEquals(List.of("AVG", "STD", "P999"), config.benchmarks.get("latency"));
            }
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    public void provenanceRecordsTheActualSourceAndSelectedQueryParallelism() throws Exception {
        String oldThreads = System.getProperty("jvector.bench.queryThreads");
        String oldRef = System.getProperty("jvector.bench.sourceRef");
        try {
            System.setProperty("jvector.bench.queryThreads", "7");
            System.setProperty("jvector.bench.sourceRef", "test-source-reference");
            var config = new RunConfig();
            config.logging = new RunConfig.RunLogging();
            config.logging.dir = directory.getRoot().getAbsolutePath();
            config.logging.jvectorRef = "old-label";
            for (String mode : List.of("direct", "queued")) {
                config.queryExecution.mode = mode;
                config.logging.runId = mode;
                var reporting = io.github.jbellis.jvector.example.reporting.RunReporting.open(config);
                var info = yaml(reporting.run().runDir().resolve("sys_info.json"));
                assertEquals("test-source-reference", info.get("jvector_ref"));
                var threads = (Map<String, Object>) info.get("threads");
                int expected = mode.equals("direct") ? 7 : Math.max(1,
                        Math.min(Runtime.getRuntime().availableProcessors(),
                                io.github.jbellis.jvector.util.PhysicalCoreExecutor.getPhysicalCoreCount()));
                assertEquals(expected, threads.get("query_throughput_parallelism"));
            }
        } finally {
            if (oldThreads == null) System.clearProperty("jvector.bench.queryThreads");
            else System.setProperty("jvector.bench.queryThreads", oldThreads);
            if (oldRef == null) System.clearProperty("jvector.bench.sourceRef");
            else System.setProperty("jvector.bench.sourceRef", oldRef);
        }
    }
}
