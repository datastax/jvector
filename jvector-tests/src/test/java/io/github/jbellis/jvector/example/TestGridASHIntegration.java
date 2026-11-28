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
import io.github.jbellis.jvector.example.yaml.BenchmarkSelection;
import io.github.jbellis.jvector.example.yaml.Measurement;
import io.github.jbellis.jvector.example.yaml.MetricSelection;
import io.github.jbellis.jvector.example.yaml.RunConfig;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.example.yaml.SearchParameters;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.MutablePQVectors;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
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
                            "    maxBatchSize: 4")
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

    @Test(timeout = 120_000)
    public void upperLayerDegreeSweepsTheUpperLayersDegree() throws Exception {
        // outDegree stays the base layer's degree; each upperLayerDegree is one build with that upper-layer degree
        var data = schema3Data();
        var config = schema3Config("onDisk", "parallel", false, "None", "- type: None");
        config.construction.addHierarchy = List.of(true);
        config.construction.upperLayerDegree = List.of(4, 8);
        var spec = GridSpec.fromConfig(config);
        var degrees = spec.buildPoints().stream().map(BuildConfig::layerDegrees).collect(java.util.stream.Collectors.toList());
        assertEquals(List.of(List.of(8, 4), List.of(8, 8)), degrees);
        var artifacts = RunArtifacts.open(csvRunConfig(), List.of(config));
        Grid.run(data, spec, artifacts);
        assertEquals(3, Files.readAllLines(artifacts.experimentsWriter().path()).size());
    }

    @Test
    public void schema3AcceptsAnAshTrainingWindow() throws Exception {
        var data = schema3Data();
        String ash = "- type: ASH\n  parameters: {optimizer: RANDOM, encodedBits: 72, bitsPerDimension: 2,"
                + " trainingStart: 8, trainingCount: 64}";
        var config = schema3Config("inMemory", "parallel", false, "FP32",
                ash.replace("  parameters:", "  scoring: symmetric\n  parameters:"), ash);
        var build = config.construction.getCompressorParameters().get(0).apply(data);
        var search = config.search.getCompressorParameters().get(0).apply(data);
        assertTrue(build.idStringFor(data), build.idStringFor(data).endsWith("_train8+64"));
        assertEquals(build.idStringFor(data), search.idStringFor(data));
    }

    @Test(timeout = 120_000)
    public void streamingBuildsWarmAndColdAndSearchesWithItsOwnCodes() throws Exception {
        var data = schema3Data();
        String ash = "- type: ASH\n  parameters: {optimizer: ITQ, encodedBits: 72, bitsPerDimension: 2}";
        var config = schema3Config("streaming", null, false, "FP32",
                ash.replace("  parameters:", "  scoring: symmetric\n  parameters:"), ash);
        var params = config.construction.getCompressorParameters().get(0).apply(data);
        var out = new java.io.ByteArrayOutputStream();
        var console = System.out;
        System.setOut(new java.io.PrintStream(out, true));
        try {
            var artifacts = RunArtifacts.open(csvRunConfig(), List.of(config));
            Grid.run(data, GridSpec.fromConfig(config), artifacts);
            var lines = Files.readAllLines(artifacts.experimentsWriter().path());
            assertEquals("A warm and a cold build, one search each", 3, lines.size());
        } finally {
            System.setOut(console);
            Files.deleteIfExists(Path.of("compressor_cache").resolve(params.idStringFor(data)));
        }
        String text = out.toString();
        assertTrue(text, text.contains("streamed 288 vectors"));
        assertTrue(text, text.contains("an inherited model"));
        assertTrue(text, text.contains("a cold-start (after 64 vectors) model"));
        assertEquals(text, 2, text.split(java.util.regex.Pattern.quote("codes the index built as vectors arrived ["), -1).length - 1);
    }

    @Test(timeout = 120_000)
    public void streamingPQEncodesEachVectorOnArrivalAsOnePassEncodingDoes() {
        var data = schema3Data();
        var pq = ProductQuantization.compute(data.getBaseRavv(), 4, 32, false);
        var build = BuildConfig.builder().M(8).efConstruction(32).neighborOverflow(1.2f).alpha(1.2f)
                .addHierarchy(false).refineFinalGraph(false).indexType("inMemory").streaming(true).workers(4).build();
        var params = new CompressorParameters.PQParameters(4, 32, false, -1.0f);
        var index = StreamingBuild.stream(build, data, params, () -> pq, true);
        var expected = (PQVectors) pq.encodeAll(data.getBaseRavv());
        var codes = (PQVectors) index.codes();
        assertTrue(codes instanceof MutablePQVectors);
        assertEquals(expected.count(), codes.count());
        for (int i = 0; i < expected.count(); i++) assertEquals("code " + i, expected.get(i), codes.get(i));
        assertEquals(data.getBaseRavv().size(), index.graph().size(0));
        assertNotNull("NVQ needs the mean kept as vectors arrive", index.mean());

        var exact = StreamingBuild.stream(build, data, CompressorParameters.NONE, () -> {
            throw new AssertionError("an exact streaming build trains no model");
        }, false);
        assertNull(exact.codes());
        assertNull(exact.mean());
        assertEquals(data.getBaseRavv().size(), exact.graph().size(0));
    }

    @Test(timeout = 120_000)
    public void streamingPQRefinesItsModelAndKeepsTheGraphNavigable() {
        var data = schema3Data();
        var vectors = data.getBaseRavv();
        var pq = ProductQuantization.compute(vectors, 4, 32, false);
        var build = BuildConfig.builder().M(8).efConstruction(32).neighborOverflow(1.2f).alpha(1.2f)
                .addHierarchy(false).refineFinalGraph(false).indexType("inMemory").streaming(true).workers(4)
                .refineAfter(144).build();
        var index = StreamingBuild.stream(build, data, new CompressorParameters.PQParameters(4, 32, false, -1.0f),
                () -> pq, false);
        assertTrue(index.start(), index.start().contains("refined after 144 vectors"));
        var codes = (PQVectors) index.codes();
        var refined = (ProductQuantization) codes.getCompressor();
        assertNotSame("the model was refined", pq, refined);
        // Every vector, before and after the refinement, carries the refined model's code.
        var expected = (PQVectors) refined.encodeAll(vectors);
        for (int i = 0; i < vectors.size(); i++) assertEquals("code " + i, expected.get(i), codes.get(i));
        // Nodes placed before the refinement stay reachable: base vectors find themselves about as often as in the
        // same build without a refinement, both those placed before it and those placed after.
        var unrefined = StreamingBuild.stream(BuildConfig.builder().M(8).efConstruction(32).neighborOverflow(1.2f)
                        .alpha(1.2f).addHierarchy(false).refineFinalGraph(false).indexType("inMemory").streaming(true)
                        .workers(4).build(), data, new CompressorParameters.PQParameters(4, 32, false, -1.0f),
                () -> pq, false);
        int[] refinedFound = selfFound(index.graph(), data, 144);
        int[] baselineFound = selfFound(unrefined.graph(), data, 144);
        String found = "refined " + Arrays.toString(refinedFound) + " vs unrefined " + Arrays.toString(baselineFound);
        for (int half = 0; half < 2; half++) assertTrue(found, refinedFound[half] >= baselineFound[half] - 15);
    }

    /** Base vectors among their own 10 nearest results, for ordinals before {@code split} and from it on. */
    private static int[] selfFound(io.github.jbellis.jvector.graph.ImmutableGraphIndex graph, SimpleDataSet data,
                                   int split) {
        var vectors = data.getBaseRavv();
        int[] found = new int[2];
        for (int i = 0; i < vectors.size(); i++) {
            var result = io.github.jbellis.jvector.graph.GraphSearcher.search(vectors.getVector(i), 10, vectors,
                    data.getSimilarityFunction(), graph, Bits.ALL);
            for (var node : result.getNodes()) if (node.node == i) found[i < split ? 0 : 1]++;
        }
        return found;
    }

    @Test
    public void storedCodesKeepPQCodesOnlyForIndexesThatBuiltThem() throws Exception {
        var data = schema3Data();
        var pq = ProductQuantization.compute(data.getBaseRavv(), 4, 32, false);
        var codes = (PQVectors) pq.encodeAll(data.getBaseRavv());
        assertFalse("batch builds keep storing only ASH codes", StoredCodes.supports(codes, false));
        assertTrue(StoredCodes.supports(codes, true));
        var index = temporary.newFolder().toPath().resolve("graph");
        StoredCodes.write(codes, index);
        var loaded = StoredCodes.load(index, codes.count());
        assertNotNull(loaded);
        assertTrue(loaded.codes instanceof PQVectors);
        for (int i = 0; i < codes.count(); i++) assertEquals(codes.get(i), ((PQVectors) loaded.codes).get(i));
    }

    @Test(timeout = 120_000)
    public void streamingPQAndFullPrecisionBuildAndSearch() throws Exception {
        var data = schema3Data();
        String pq = "- type: PQ\n  parameters: {m: 4, k: 32}";
        for (String type : List.of("streamingWarm", "streamingWarmOnDisk")) {
            for (String construction : List.of(pq, "- type: None")) {
                // PQ refines its model halfway, as a compaction that inherits a codebook does.
                var config = schema3Config(type, "parallel", false, type.endsWith("OnDisk") ? "NVQ" : "FP32",
                        construction, pq, construction.equals(pq) ? "  refineAfter: [144]\n" : "");
                var out = new ByteArrayOutputStream();
                var console = System.out;
                System.setOut(new PrintStream(out, true, StandardCharsets.UTF_8));
                try {
                    var artifacts = RunArtifacts.open(csvRunConfig(), List.of(config));
                    Grid.run(data, GridSpec.fromConfig(config), artifacts);
                    var lines = Files.readAllLines(artifacts.experimentsWriter().path());
                    assertEquals("One build, one search", 2, lines.size());
                } finally {
                    System.setOut(console);
                    Files.deleteIfExists(Path.of("compressor_cache").resolve(
                            config.search.getCompressorParameters().get(0).apply(data).idStringFor(data)));
                }
                String text = out.toString(StandardCharsets.UTF_8);
                String label = type + " " + construction;
                assertTrue(label + "\n" + text, text.contains("streamed 288 vectors"));
                boolean streamedPQ = construction.equals(pq);
                assertTrue(label + "\n" + text, text.contains(streamedPQ ? "refined after 144 vectors" : "with exact scores"));
                // A PQ index is searched with the codes it built; on disk, they are stored beside it and reloaded.
                assertEquals(label + "\n" + text, streamedPQ, text.contains(type.endsWith("OnDisk")
                        ? "codes the index built as vectors arrived, loaded with the index"
                        : "codes the index built as vectors arrived ["));
            }
        }
    }

    @Test
    public void streamingExamplesParse() throws Exception {
        assertStreamingExample("streaming-pq", 2);
        assertStreamingExample("streaming-fp", 1);
    }

    private static void assertStreamingExample(String folder, int buildPoints) throws Exception {
        Path path = Path.of("jvector-examples", "yaml-configs", "index-parameters", folder, "cohere-english-v3-1M.yml");
        if (!Files.exists(path)) path = Path.of("..").resolve(path);
        var spec = GridSpec.fromConfig(MultiConfig.getConfig(path.toAbsolutePath().toString()));
        assertEquals(folder, buildPoints, spec.buildPoints().size());
        assertTrue(folder, spec.buildPoints().stream().allMatch(build -> build.streaming && build.coldStartSize == null));
    }

    @Test
    public void streamingRejectsColdStartsOutsideASHAndUnsupportedCodecs() {
        var pq = assertThrows(IllegalArgumentException.class, () -> schema3Config("streaming", null, false, "FP32",
                "- type: PQ\n  parameters: {m: 4, k: 32}", "- type: None"));
        assertTrue(pq.getMessage(), pq.getMessage().contains("Only ASH streaming builds start cold"));
        var bq = assertThrows(IllegalArgumentException.class, () -> schema3Config("streamingWarm", null, false, "FP32",
                "- type: BQ", "- type: None"));
        assertTrue(bq.getMessage(), bq.getMessage().contains("requires a single ASH, PQ or None"));
        String ash = "- type: ASH\n  parameters: {optimizer: ITQ, encodedBits: 72, bitsPerDimension: 2}";
        var refineASH = assertThrows(IllegalArgumentException.class, () -> schema3Config("streaming", null, false,
                "FP32", ash, ash, "  refineAfter: [64]\n"));
        assertTrue(refineASH.getMessage(), refineASH.getMessage().contains("refineAfter requires PQ"));
        String pqCodes = "- type: PQ\n  parameters: {m: 4, k: 32}";
        var tooFew = assertThrows(IllegalArgumentException.class, () -> schema3Config("streamingWarm", null, false,
                "FP32", pqCodes, pqCodes, "  refineAfter: [0, 16]\n"));
        assertTrue(tooFew.getMessage(), tooFew.getMessage().contains("at least the PQ k (32)"));
    }

    @Test
    public void savedInMemoryIndexesReloadIntoMemoryWithTheirCodes() throws Exception {
        var data = schema3Data();
        var build = BuildConfig.builder().M(8).efConstruction(32).neighborOverflow(1.2f).alpha(1.2f)
                .addHierarchy(true).refineFinalGraph(false).indexType("inMemory").build();
        var ash = AsymmetricHashing.initialize(data.getBaseRavv(), VectorSimilarityFunction.DOT_PRODUCT, AsymmetricHashing.ITQ, 72, 1, 2);
        var codes = ash.encodeAll(data.getBaseRavv());
        var cache = OnDiskGraphIndexCache.initialize(temporary.newFolder().toPath());
        var builds = new java.util.concurrent.atomic.AtomicInteger();
        IndexBuildStrategy fresh = (featureSets, b, ds) -> {
            builds.incrementAndGet();
            return new IndexBuildStrategy.BuiltIndexes(Map.of(Set.of(), GraphBuilderFactory.buildGraph(ds, b, codes)),
                    Map.of(), 1.0);
        };
        var first = new SavedInMemoryBuild(cache, "test", () -> fresh, () -> codes, false)
                .build(List.of(Set.of()), build, data);
        var second = new SavedInMemoryBuild(cache, "test", () -> fresh, () -> codes, false)
                .build(List.of(Set.of()), build, data);
        assertEquals("the second run reloads instead of building", 1, builds.get());
        assertNull(second.buildTimeS);
        var built = first.indexes.get(Set.of());
        var loaded = second.indexes.get(Set.of());
        assertTrue(loaded instanceof io.github.jbellis.jvector.graph.OnHeapGraphIndex);
        assertEquals(built.getMaxLevel(), loaded.getMaxLevel());
        try (var a = built.getView(); var b = loaded.getView()) {
            assertEquals(a.entryNode(), b.entryNode());
            for (int level = 0; level <= built.getMaxLevel(); level++) {
                for (int node = 0; node < data.getBaseRavv().size(); node++) {
                    if (((io.github.jbellis.jvector.graph.OnHeapGraphIndex) built).getMaxLevelForNode(node) < level) continue;
                    var x = a.getNeighborsIterator(level, node);
                    var y = b.getNeighborsIterator(level, node);
                    assertEquals(x.size(), y.size());
                    while (x.hasNext()) assertEquals(x.nextInt(), y.nextInt());
                }
            }
        }
        var stored = second.storedCodes.get(Set.of());
        assertFalse(stored.builtWithIndex);
        for (int i = 0; i < codes.count(); i++) assertEquals(((io.github.jbellis.jvector.quantization.ASHVectors) codes).get(i),
                ((io.github.jbellis.jvector.quantization.ASHVectors) stored.codes).get(i));
    }

    @Test(timeout = 120_000)
    public void streamingOnDiskWritesReloadsAndSearchesTheFlushedIndex() throws Exception {
        var data = schema3Data();
        String ash = "- type: ASH\n  parameters: {optimizer: ITQ, encodedBits: 72, bitsPerDimension: 2}";
        var config = schema3Config("streamingOnDisk", "parallel", false, "NVQ",
                ash.replace("  parameters:", "  scoring: symmetric\n  parameters:"), ash);
        var params = config.construction.getCompressorParameters().get(0).apply(data);
        var out = new java.io.ByteArrayOutputStream();
        var console = System.out;
        System.setOut(new java.io.PrintStream(out, true));
        try {
            var artifacts = RunArtifacts.open(csvRunConfig(), List.of(config));
            Grid.run(data, GridSpec.fromConfig(config), artifacts);
            var lines = Files.readAllLines(artifacts.experimentsWriter().path());
            assertEquals("A warm and a cold build, one search each", 3, lines.size());
        } finally {
            System.setOut(console);
            Files.deleteIfExists(Path.of("compressor_cache").resolve(params.idStringFor(data)));
        }
        String text = out.toString();
        assertTrue(text, text.contains("streamed 288 vectors"));
        // Each flushed index is reloaded and searched with the codes written beside it.
        assertEquals(text, 2, text.split("codes the index built as vectors arrived, loaded with the index", -1).length - 1);
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
            assertEquals(fresh.files, reused.files);
            assertEquals(fresh.fileSize(featureSet), reused.fileSize(featureSet));
            assertEquals(data.getBaseRavv().size(), index.size(0));
        }
    }

    @Test(timeout = 120_000)
    public void nonFusedIndexStoresItsCodesAndSearchesLoadThem() throws Exception {
        var data = schema3Data();
        String ash = "- type: ASH\n  parameters: {optimizer: RANDOM, encodedBits: 72, bitsPerDimension: 2}";
        var config = schema3Config("onDisk", "parallel", false, "NVQ",
                ash.replace("  parameters:", "  scoring: symmetric\n  parameters:"), ash);
        var spec = GridSpec.fromConfig(config);
        var build = spec.buildPoints().get(0);
        var features = spec.featuresFor(build, spec.search.cases().get(0));
        var featureSet = features.get(0);
        var params = config.search.getCompressorParameters().get(0).apply(data);
        var cache = OnDiskGraphIndexCache.initialize(temporary.newFolder().toPath());
        var work = temporary.newFolder().toPath();
        try {
            var freshMetrics = new ConstructionMetrics(true);
            var fresh = new CompressedOnDiskBuild(cache, work, () -> params.computeCompressor(data),
                    freshMetrics, null, "schema3:stored-codes").build(features, build, data);
            assertNotNull(fresh.storedCodes.get(featureSet));
            var stored = fresh.storedCodes.get(featureSet).codes;
            try (var index = fresh.indexes.get(featureSet)) {
                // Writing the codes is part of the flush.
                assertTrue(freshMetrics.codesWriteTimeS > 0);
                assertTrue(freshMetrics.codesWriteTimeS <= freshMetrics.flushTimeS);
                assertNotNull(fresh.codesFileSize(featureSet));
                assertTrue(Files.exists(StoredCodes.pathFor(fresh.files.get(featureSet))));
                assertEquals(data.getBaseRavv().size(), stored.count());
                assertTrue(params.describes(stored.getCompressor()));
                // The stored codes are the ones the model encodes from the base vectors.
                assertEquals(stored.getCompressor().encodeAll(data.getBaseRavv()), stored);
            }

            var reusedMetrics = new ConstructionMetrics(true);
            var reused = new CompressedOnDiskBuild(cache, work, () -> {
                throw new AssertionError("cache hit tried to prepare a build compressor");
            }, reusedMetrics, null, "schema3:stored-codes").build(features, build, data);
            try (var index = reused.indexes.get(featureSet)) {
                assertNull(reused.buildTimeS);
                assertNull(reusedMetrics.codesWriteTimeS);
                assertEquals(stored, reused.storedCodes.get(featureSet).codes);
            }

            // A search with the same ASH settings uses the codes stored with the index and encodes nothing,
            // and the CSV reports loading them where it would report encoding them.
            var run = csvRunConfig();
            run.logging.metrics = new MetricSelection();
            run.logging.metrics.put("construction", List.of("codes_write_time_s", "codes_file_size_mb",
                    "search_quant_time_s"));
            var artifacts = RunArtifacts.open(run, List.of(config));
            var console = new ByteArrayOutputStream();
            var stdout = System.out;
            try {
                System.setOut(new PrintStream(console, true, StandardCharsets.UTF_8));
                Grid.run(data, spec, artifacts);
            } finally {
                System.setOut(stdout);
            }
            var output = console.toString(StandardCharsets.UTF_8);
            assertTrue(output, output.contains("traversal codes stored with the index"));
            assertFalse(output, output.contains(" encoded " + data.getBaseRavv().size() + " vectors"));
            var lines = Files.readAllLines(artifacts.experimentsWriter().path());
            assertEquals(2, lines.size());
            var headers = Arrays.asList(lines.get(0).split(","));
            var cells = lines.get(1).split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1);
            for (var key : List.of("search.search_quant_time_s.ASH.load_time_s",
                    "construction.codes_write_time_s", "construction.codes_file_size_mb")) {
                assertTrue(lines.get(0), headers.contains(key));
                assertFalse(key, cells[headers.indexOf(key)].isEmpty());
            }
            String encoding = "search.search_quant_time_s.ASH.encoding_time_s";
            assertTrue(lines.get(0), headers.contains(encoding));
            assertTrue(encoding, cells[headers.indexOf(encoding)].isEmpty());
        } finally {
            Files.deleteIfExists(Path.of("compressor_cache").resolve(params.idStringFor(data)));
        }
    }

    @Test(timeout = 120_000)
    public void searchWorkersSweepQueriesOneIndexWithEachWorkerCount() throws Exception {
        var data = schema3Data();
        var search = String.join("\n", "- type: ASH", "  parameters:", "    optimizer: RANDOM",
                "    encodedBits: 72", "    bitsPerDimension: 2", "    queryProjection: fp32",
                "    maxBatchSize: 4");
        var config = schema3Config("onDisk", "parallel", true, "NVQ", search);
        config.construction.workers = 2;
        config.search.workers = List.of(1, 2);
        var params = config.search.getCompressorParameters().get(0).apply(data);
        try (var projection = new SearchParameters.AshQueryProjectionSetting()) {
            projection.apply(List.of(config));
            var artifacts = RunArtifacts.open(csvRunConfig(), List.of(config));
            var console = new java.io.ByteArrayOutputStream();
            var stdout = System.out;
            try {
                System.setOut(new java.io.PrintStream(console, true, java.nio.charset.StandardCharsets.UTF_8));
                Grid.run(data, GridSpec.fromConfig(config), artifacts);
            } finally {
                System.setOut(stdout);
            }
            var output = console.toString(java.nio.charset.StandardCharsets.UTF_8);
            assertEquals(output, 1, output.split("Building graph index", -1).length - 1);
            var lines = Files.readAllLines(artifacts.experimentsWriter().path());
            // One build, then one row per search worker count, each recording its own workers.
            assertEquals(3, lines.size());
            assertTrue(lines.get(1), lines.get(1).matches(".*,onDisk,parallel,NVQ,,queued,1,4,.*"));
            assertTrue(lines.get(2), lines.get(2).matches(".*,onDisk,parallel,NVQ,,queued,2,4,.*"));
        } finally {
            Files.deleteIfExists(Path.of("compressor_cache").resolve(params.idStringFor(data)));
        }
    }

    @Test(timeout = 120_000)
    public void measurementSettingsTimeOneColdPass() throws Exception {
        var data = schema3Data();
        var config = schema3Config("onDisk", "parallel", false, "None", "- type: None");
        boolean linux = System.getProperty("os.name").equals("Linux");
        // Eviction calls the OS through the foreign function API, final from JDK 22.
        boolean evict = linux && Runtime.version().feature() >= 22;
        var run = csvRunConfig();
        run.benchmarks = Map.of("throughput", List.of("AVG"), "accuracy", List.of("recall"));
        run.measurement = new Measurement();
        run.measurement.evictIndex = evict;
        run.measurement.throughput = new Measurement.Throughput();
        run.measurement.throughput.warmup = "base";
        run.measurement.throughput.untilCompiled = true;
        run.measurement.throughput.passes = 1;
        run.measurement.throughput.minSeconds = 0.0;
        run.console = new BenchmarkSelection();
        run.console.benchmarks = run.benchmarks;
        run.console.metrics = new MetricSelection();
        run.console.metrics.put("execution", List.of("response_p50_ms", "response_p999_ms", "page_reads_per_query"));
        var console = new ByteArrayOutputStream();
        var stdout = System.out;
        try {
            Grid.setMeasurement(run.measurement);
            var artifacts = RunArtifacts.open(run, List.of(config));
            System.setOut(new PrintStream(console, true, StandardCharsets.UTF_8));
            Grid.run(data, GridSpec.fromConfig(config), artifacts);
        } finally {
            System.setOut(stdout);
            Grid.setMeasurement(null);
        }
        var output = console.toString(StandardCharsets.UTF_8);
        // One timed pass, with direct execution reporting its search times.
        assertTrue(output, output.contains("(of 1)"));
        // Column headers wrap, so their units print on the next line.
        assertTrue(output, output.contains("Loaded p50"));
        assertTrue(output, output.contains("Loaded p99.9"));
        assertEquals(output, evict, output.contains("index evicted from the page cache"));
        assertEquals(output, linux, output.contains("Page reads"));
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
        return schema3Config(type, writer, fused, reranker, construction, search, "");
    }

    /** {@code settings} adds construction lines, each indented and ending in a newline. */
    private MultiConfig schema3Config(String type, String writer, boolean fused, String reranker,
                                      String construction, String search, String settings) throws Exception {
        var path = temporary.newFile().toPath();
        // "streaming" streams into an in-memory index; "streamingOnDisk" also writes and reloads it, as a flush does.
        // "streamingWarm" and "streamingWarmOnDisk" set no coldStartSize, as PQ and None streaming builds may.
        boolean streaming = type.startsWith("streaming");
        boolean warmOnly = type.startsWith("streamingWarm");
        if (streaming) type = type.endsWith("OnDisk") ? "onDisk" : "inMemory";
        String diskSettings = ("onDisk".equals(type)
                ? "  writerMode: " + writer + "\n  fusedGraph: [" + fused + "]\n" : "")
                + (streaming ? "  streaming: Yes\n" + (warmOnly ? "" : "  coldStartSize: [0, 64]\n") : "") + settings;
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
                // Batched ASH search runs queued; every other case keeps the direct default.
                search.contains("maxBatchSize") ? "  queryExecution: queued" : "",
                "  compression:",
                "    " + search.replace("\n", "\n    "));
        Files.writeString(path, yaml);
        return MultiConfig.getConfig(path.toString());
    }
}
