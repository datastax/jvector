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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.example.benchmarks.datasets.SimpleDataSet;
import io.github.jbellis.jvector.example.reporting.RunArtifacts;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.OnDiskGraphIndexCache;
import io.github.jbellis.jvector.example.yaml.MetricSelection;
import io.github.jbellis.jvector.example.yaml.RunConfig;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.example.yaml.SearchParameters;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import org.junit.Test;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import static org.junit.Assert.*;

/** The refactored strategy, feature provider, queued queries and CSV listener must work together. */
public class TestGridASHIntegration {
    @Rule public final TemporaryFolder temporary = new TemporaryFolder();

    @Test(timeout = 120_000)
    public void gridSpecBuildsFusedASHAndReportsPrecisionMetadata() throws Exception {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var random = new Random(19);
        var vectors = new ArrayList<io.github.jbellis.jvector.vector.types.VectorFloat<?>>();
        for (int i = 0; i < 128; i++) {
            var v = vts.createFloatVector(32);
            for (int d = 0; d < 32; d++) v.set(d, random.nextFloat() - 0.5f);
            vectors.add(v);
        }
        var data = new SimpleDataSet("grid-ash-merge-" + UUID.randomUUID(), VectorSimilarityFunction.DOT_PRODUCT,
                vectors, List.of(vectors.get(0)), List.of(List.of(0)));
        var parameters = new CompressorParameters.ASHParameters(AsymmetricHashing.RANDOM,
                AsymmetricHashing.HEADER_BITS + 16 * 2, 1, 2);
        var spec = GridSpec.fromGrids(List.of(false), List.of(false), List.of(8), List.of(1.2f), List.of(32),
                List.of(ignored -> parameters), List.of(EnumSet.of(FeatureId.INLINE_VECTORS, FeatureId.FUSED_ASH)),
                SearchConfig.builder().searchCompressors(List.of(ignored -> CompressorParameters.NONE))
                        .topKGrid(Map.of(1, List.of(1.0, 2.0))).usePruningGrid(List.of(false)).build(), false);
        var config = new RunConfig();
        config.benchmarks = Map.of("accuracy", List.of("recall"));
        config.logging = new RunConfig.RunLogging();
        config.logging.type = "csv";
        config.logging.dir = temporary.getRoot().toString();
        config.logging.benchmarks = config.benchmarks;
        String previous = System.getProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY);
        var originalOut = System.out;
        var output = new ByteArrayOutputStream();
        try {
            System.setProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY, "fp32");
            var indexConfig = new MultiConfig();
            indexConfig.search = new SearchParameters();
            indexConfig.search.topKOverquery = Map.of(1, List.of(1.0, 2.0));
            var artifacts = RunArtifacts.open(config, List.of(indexConfig));
            System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
            Grid.run(data, spec, artifacts);
            var lines = Files.readAllLines(artifacts.experimentsWriter().path());
            assertEquals(3, lines.size());
            var headers = Arrays.asList(lines.get(0).split(","));
            for (var line : lines.subList(1, lines.size())) {
                var cells = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1);
                assertEquals(headers.size(), cells.length);
                assertEquals("fp32", cells[headers.indexOf("ash_query_projection_requested")]);
                assertEquals("fp32", cells[headers.indexOf("ash_query_projection_effective")]);
                assertTrue(cells[headers.indexOf("feature_set")].contains("FUSED_ASH"));
            }
            var console = output.toString(StandardCharsets.UTF_8);
            assertTrue(console, console.contains("executionMode        direct"));
        } finally {
            System.setOut(originalOut);
            if (previous == null) System.clearProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY);
            else System.setProperty(SearchParameters.ASH_QUERY_PROJECTION_PROPERTY, previous);
            Files.deleteIfExists(Path.of("compressor_cache").resolve(parameters.idStringFor(data)));
        }
    }
    @Test(timeout = 120_000)
    public void schema3FlushesSearchCodesOverSourceScoredGraph() throws Exception {
        var data = schema3Data();
        String callerThreads = expectedDirectThreadCount();
        for (String codec : List.of("ASH", "PQ")) {
            var search = "ASH".equals(codec)
                    ? String.join("\n", "- type: ASH", "  parameters:", "    optimizer: RANDOM",
                            "    encodedBits: 72", "    bitsPerDimension: 2", "    queryProjection: fp32",
                            "    queryExecution: queued", "    maxBatchSize: 4")
                    : "- type: PQ\n  parameters: {m: 4, k: 256}";
            var config = schema3Config("onDisk", "ASH".equals(codec) ? "sequential" : "parallel",
                    true, "NVQ", search);
            var spec = GridSpec.fromConfig(config);
            try (var projection = new SearchParameters.AshQueryProjectionSetting()) {
                projection.apply(List.of(config));
                var artifacts = RunArtifacts.open(csvRunConfig(), List.of(config));
                try {
                    Grid.run(data, spec, artifacts);
                    var lines = Files.readAllLines(artifacts.experimentsWriter().path());
                    assertEquals(2, lines.size());
                    assertTrue(lines.get(0), lines.get(0).contains("index_type,writer_mode,reranking_type"));
                    assertTrue(lines.get(1), lines.get(1).contains("FUSED_" + codec));
                    assertTrue(lines.get(1), lines.get(1).matches("ASH".equals(codec)
                            ? ".*,onDisk,sequential,NVQ,,queued,[0-9]+,4,.*"
                            : ".*,onDisk,parallel,NVQ,,direct," + callerThreads + ",,.*"));
                    assertTrue(lines.get(1), lines.get(1).contains("NVQ_VECTORS"));
                    assertFalse(lines.get(1), lines.get(1).contains("INLINE_VECTORS"));
                } finally {
                    var params = config.search.getCompressorParameters().get(0).apply(data);
                    Files.deleteIfExists(Path.of("compressor_cache").resolve(params.idStringFor(data)));
                }
            }
        }
    }

    @Test(timeout = 120_000)
    public void schema3SourceOnlySearchUsesNoReranker() throws Exception {
        var data = schema3Data();
        for (String type : List.of("inMemory", "onDisk")) {
            var config = schema3Config(type, "sequential", false, "None", "- type: None");
            var spec = GridSpec.fromConfig(config);
            var build = spec.buildPoints().get(0);
            var features = spec.featuresFor(build, spec.search.cases().get(0));
            var directory = temporary.newFolder().toPath();
            IndexBuildStrategy strategy = "inMemory".equals(type)
                    ? new InMemoryBuild(directory)
                    : new CompressedOnDiskBuild(
                            OnDiskGraphIndexCache.disabled(),
                            directory, null, new ConstructionMetrics(true), null, "test");
            var built = strategy.build(features, build, data);
            var featureSet = features.get(0);
            assertEquals("onDisk".equals(type) ? Set.of(FeatureId.INLINE_VECTORS) : Set.of(), featureSet);
            try (var index = built.indexes.get(featureSet);
                 var cs = new Grid.ConfiguredSystem(data, index, null, featureSet, "None", null);
                 var view = index.getView()) {
                if (index instanceof OnDiskGraphIndex) {
                    assertEquals(featureSet, ((OnDiskGraphIndex) index).getFeatureSet());
                }
                var scorer = cs.scoreProviderFor(data.getQueryVectors().get(0), view);
                assertTrue(scorer.scoreFunction().isExact());
                assertNull(scorer.reranker());
                var searcher = cs.getSearcher();
                var query = data.getQueryVectors().get(0);
                var result = searcher.search(cs.scoreProviderFor(query, searcher.getView()),
                        1, 2, 0.0f, 0.0f, Bits.ALL);
                assertEquals(1, result.getNodes().length);
                assertEquals(0, result.getRerankedCount());
                var best = result.getNodes()[0];
                assertEquals(data.getSimilarityFunction().compare(query, data.getBaseVectors().get(best.node)),
                        best.score, 0.000001f);
            }
            var artifacts = RunArtifacts.open(csvRunConfig(), List.of(config));
            Grid.run(data, spec, artifacts);
            assertEquals(2, Files.readAllLines(artifacts.experimentsWriter().path()).size());
        }
    }

    @Test(timeout = 120_000)
    public void schema3SupportsOptionalRerankingForNonFusedCompressedSearch() throws Exception {
        var data = schema3Data();
        String callerThreads = expectedDirectThreadCount();
        for (String reranker : List.of("None", "NVQ")) {
            var config = schema3Config("inMemory", null, false, reranker, "- type: BQ");
            var run = csvRunConfig();
            run.logging.metrics = new MetricSelection();
            run.logging.metrics.put("construction", List.of("index_quant_time_s"));
            var artifacts = RunArtifacts.open(run, List.of(config));
            Grid.run(data, GridSpec.fromConfig(config), artifacts);
            var lines = Files.readAllLines(artifacts.experimentsWriter().path());
            assertEquals(2, lines.size());
            assertTrue(lines.get(1), lines.get(1).matches(".*,inMemory,," + reranker + ",,direct," + callerThreads + ",,.*"));
            assertEquals("NVQ".equals(reranker),
                    lines.get(0).contains("construction.index_quant_time_s.NVQ.encoding_time_s"));
        }
    }

    @Test(timeout = 120_000)
    public void schema3BuildsAndFlushesWithTheSameAshSettings() throws Exception {
        var data = schema3Data();
        String ash = "- type: ASH\n  parameters: {optimizer: RANDOM, encodedBits: 72, bitsPerDimension: 2}";
        var config = schema3Config("onDisk", "parallel", true, "FP32",
                ash.replace("  parameters:", "  scoring: symmetric\n  parameters:"), ash);
        var params = config.search.getCompressorParameters().get(0).apply(data);
        try {
            var artifacts = RunArtifacts.open(csvRunConfig(), List.of(config));
            Grid.run(data, GridSpec.fromConfig(config), artifacts);
            var lines = Files.readAllLines(artifacts.experimentsWriter().path());
            assertEquals(2, lines.size());
            assertTrue(lines.get(1), lines.get(1).contains("FUSED_ASH"));
            assertTrue(lines.get(1), lines.get(1).matches(".*,onDisk,parallel,FP32,symmetric-all,direct,"
                    + expectedDirectThreadCount() + ",,.*"));
        } finally {
            Files.deleteIfExists(Path.of("compressor_cache").resolve(params.idStringFor(data)));
        }
    }

    @Test
    public void schema3CacheHitSkipsBuildAndFlush() throws Exception {
        var data = schema3Data();
        var config = schema3Config("onDisk", "parallel", false, "None", "- type: None");
        var spec = GridSpec.fromConfig(config);
        var build = spec.buildPoints().get(0);
        var features = spec.featuresFor(build, spec.search.cases().get(0));
        var cache = OnDiskGraphIndexCache.initialize(temporary.newFolder().toPath());
        var work = temporary.newFolder().toPath();
        var fresh = new CompressedOnDiskBuild(cache, work, () -> null, new ConstructionMetrics(true),
                null, "schema3:None").build(features, build, data);
        var featureSet = features.get(0);
        try (var index = fresh.indexes.get(featureSet)) {
            assertNotNull(fresh.buildTimeS);
            assertEquals(data.getBaseRavv().size(), index.size(0));
        }

        var reused = new CompressedOnDiskBuild(cache, work, () -> {
            throw new AssertionError("cache hit tried to prepare a build compressor");
        }, new ConstructionMetrics(true), null, "schema3:None").build(features, build, data);
        try (var index = reused.indexes.get(featureSet)) {
            assertNull(reused.buildTimeS);
            assertEquals(fresh.fileSizes, reused.fileSizes);
            assertEquals(data.getBaseRavv().size(), index.size(0));
        }
    }

    @Test
    public void schema3RecordsGraphBuildFlushAndReopenTimes() throws Exception {
        var data = schema3Data();
        var config = schema3Config("onDisk", "parallel", false, "None", "- type: None");
        var spec = GridSpec.fromConfig(config);
        var build = spec.buildPoints().get(0);
        var features = spec.featuresFor(build, spec.search.cases().get(0));
        var cache = OnDiskGraphIndexCache.initialize(temporary.newFolder().toPath());
        var work = temporary.newFolder().toPath();

        var fresh = new ConstructionMetrics(true);
        var built = new CompressedOnDiskBuild(cache, work, () -> null, fresh, null, "schema3:None")
                .build(features, build, data);
        built.indexes.values().forEach(index -> { try { index.close(); } catch (Exception e) { throw new RuntimeException(e); } });
        assertTrue(fresh.graphBuildTimeS > 0);
        assertTrue(fresh.flushTimeS > 0);
        assertTrue(fresh.reopenTimeS > 0);
        assertEquals(built.buildTimeS, fresh.graphBuildTimeS + fresh.flushTimeS, 1e-9);
        var recorded = new ArrayList<io.github.jbellis.jvector.example.benchmarks.Metric>();
        fresh.appendTo(recorded);
        var keys = recorded.stream().map(io.github.jbellis.jvector.example.benchmarks.Metric::getKey)
                .collect(java.util.stream.Collectors.toSet());
        assertTrue(keys.toString(), keys.containsAll(List.of("construction.graph_build_time_s",
                "construction.flush_time_s", "construction.reopen_time_s")));

        // A cache hit loads the cached index without building, flushing or reopening it.
        var reused = new ConstructionMetrics(true);
        var cached = new CompressedOnDiskBuild(cache, work, () -> null, reused, null, "schema3:None")
                .build(features, build, data);
        cached.indexes.values().forEach(index -> { try { index.close(); } catch (Exception e) { throw new RuntimeException(e); } });
        assertNull(cached.buildTimeS);
        assertNull(reused.graphBuildTimeS);
        assertNull(reused.flushTimeS);
        assertNull(reused.reopenTimeS);
    }

    @Test
    public void schema3RejectsFusedSearchWithoutReranking() {
        var error = assertThrows(IllegalArgumentException.class,
                () -> schema3Config("onDisk", "parallel", true, "None",
                        "- type: PQ\n  parameters: {m: 4, k: 256}"));
        assertTrue(error.getMessage(), error.getMessage().contains("fusedGraph requires construction.reranking.type"));
    }

    private SimpleDataSet schema3Data() {
        var support = VectorizationProvider.getInstance().getVectorTypeSupport();
        var random = new Random(91);
        var vectors = new ArrayList<io.github.jbellis.jvector.vector.types.VectorFloat<?>>();
        for (int i = 0; i < 288; i++) {
            var vector = support.createFloatVector(32);
            for (int d = 0; d < 32; d++) vector.set(d, random.nextFloat() - 0.5f);
            VectorUtil.l2normalize(vector);
            vectors.add(vector);
        }
        return new SimpleDataSet("grid-schema3-" + UUID.randomUUID(), VectorSimilarityFunction.DOT_PRODUCT,
                vectors, List.of(vectors.get(0)), List.of(List.of(0)));
    }

    private static String expectedDirectThreadCount() {
        int threads = Integer.getInteger("jvector.bench.queryThreads", 0);
        return threads == 0 ? "" : Integer.toString(threads);
    }

    private RunConfig csvRunConfig() throws Exception {
        var config = new RunConfig();
        config.benchmarks = Map.of("accuracy", List.of("recall"));
        config.logging = new RunConfig.RunLogging();
        config.logging.type = "csv";
        config.logging.dir = temporary.newFolder().toString();
        config.logging.benchmarks = config.benchmarks;
        return config;
    }

    private MultiConfig schema3Config(String type, String writer, boolean fused, String reranker, String search)
            throws Exception {
        return schema3Config(type, writer, fused, reranker, "- type: None", search);
    }

    private MultiConfig schema3Config(String type, String writer, boolean fused, String reranker,
                                      String construction, String search) throws Exception {
        var path = temporary.newFile().toPath();
        String diskSettings = "onDisk".equals(type)
                ? "  writerMode: " + writer + "\n  fusedGraph: [" + fused + "]\n"
                : "";
        String yaml = String.join("\n",
                "yamlSchemaVersion: 3",
                "onDiskIndexVersion: " + OnDiskGraphIndex.CURRENT_VERSION,
                "dataset: test",
                "construction:",
                "  type: " + type,
                diskSettings + "  outDegree: [8]",
                "  efConstruction: [32]",
                "  neighborOverflow: [1.2f]",
                "  addHierarchy: [No]",
                "  refineFinalGraph: [No]",
                "  compression:",
                "    " + construction.replace("\n", "\n    "),
                "  reranking:",
                "    type: " + reranker,
                "search:",
                "  topKOverquery:",
                "    1: [2.0]",
                "  useSearchPruning: [No]",
                "  compression:",
                "    " + search.replace("\n", "\n    "));
        Files.writeString(path, yaml);
        return MultiConfig.getConfig(path.toString());
    }
}
