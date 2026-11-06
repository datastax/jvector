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

import io.github.jbellis.jvector.example.benchmarks.Metric;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.SimpleDataSet;
import io.github.jbellis.jvector.fastervector.FastervectorLibrary;
import io.github.jbellis.jvector.fastervector.bench.IvfConfig.SearchPoint;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Runs the IVF grid end to end on a small synthetic dataset. Skipped unless the fastervector
 * native libraries load; {@code -Djvector.fastervector.test.requireNative=true} makes that a failure.
 */
class IvfGridNativeTest {
    private static final int N = 2000;
    private static final int DIM = 32;
    private static final int NQ = 20;
    private static final int CLUSTERS = 16;
    // Every list probed, so recall reflects only ASH compression, as in IvfAshIndexNativeTest
    private static final double MIN_RECALL = 0.5;

    private static DataSet ds;

    @TempDir
    Path dir;

    @BeforeAll
    static void loadNativeAndMakeData() {
        try {
            FastervectorLibrary.load();
        } catch (UnsupportedOperationException | UnsatisfiedLinkError e) {
            if (Boolean.getBoolean("jvector.fastervector.test.requireNative")) {
                throw e;
            }
            assumeTrue(false, "fastervector native libraries unavailable: " + e.getMessage());
        }

        VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        Random r = new Random(42);
        float[][] centers = new float[CLUSTERS][];
        for (int c = 0; c < CLUSTERS; c++) {
            centers[c] = gaussian(r, 1.0f, null);
        }
        float[][] base = new float[N][];
        for (int i = 0; i < N; i++) {
            base[i] = gaussian(r, 0.3f, centers[r.nextInt(CLUSTERS)]);
        }
        List<VectorFloat<?>> baseVectors = new ArrayList<>();
        for (float[] v : base) {
            baseVectors.add(vts.createFloatVector(v));
        }
        List<VectorFloat<?>> queryVectors = new ArrayList<>();
        List<List<Integer>> groundTruth = new ArrayList<>();
        for (int q = 0; q < NQ; q++) {
            float[] query = gaussian(r, 0.3f, centers[r.nextInt(CLUSTERS)]);
            queryVectors.add(vts.createFloatVector(query));
            groundTruth.add(IntStream.range(0, N).boxed()
                    .sorted(Comparator.comparingDouble((Integer i) -> dot(query, base[i])).reversed())
                    .limit(100)
                    .collect(Collectors.toList()));
        }
        ds = new SimpleDataSet("synthetic", VectorSimilarityFunction.DOT_PRODUCT, baseVectors, queryVectors, groundTruth);
    }

    @Test
    void runsEveryPointAndReportsMetrics() throws IOException {
        IvfConfig config = config(false);
        var rows = run(config);

        assertEquals(config.constructionPoints().size() * config.searchPoints().size(), rows.size());
        for (Row row : rows) {
            Map<String, Double> m = row.metrics();
            String recallKey = "search.accuracy.recall_at_" + row.point().topK();
            assertTrue(m.containsKey(recallKey), "missing " + recallKey);
            assertTrue(m.containsKey("construction.index_build_time_s"));
            assertTrue(m.get("construction.index_file_size_mb") > 0);
            assertTrue(m.get("search.disk.file_count") > 0);
            assertTrue(m.containsKey("search.system.max_heap_mb"));
            // Native tests run only on Linux, where /proc/self/status gives resident memory
            assertTrue(m.containsKey(IvfSystem.INDEX_RSS_KEY), "missing " + IvfSystem.INDEX_RSS_KEY);
            assertTrue(m.containsKey(IvfGrid.SEARCH_PEAK_RSS_KEY), "missing " + IvfGrid.SEARCH_PEAK_RSS_KEY);
            if (row.point().nprobe() == CLUSTERS) {
                assertTrue(m.get(recallKey) >= MIN_RECALL, recallKey + " = " + m.get(recallKey));
            }
        }
        // Each construction point builds into its own directory
        for (var point : config.constructionPoints()) {
            assertTrue(Files.exists(IvfSystem.manifestPath(config.ssdPaths(), point.subfolder(ds.getName()))));
        }
    }

    @Test
    void savedIndexIsReopenedWithSameRecall() throws IOException {
        var built = run(config(false));
        var reopened = run(config(true));
        assertEquals(built.size(), reopened.size());
        for (int i = 0; i < built.size(); i++) {
            assertFalse(reopened.get(i).metrics().containsKey("construction.index_build_time_s"),
                        "a reopened index has no build time");
            String recallKey = "search.accuracy.recall_at_" + built.get(i).point().topK();
            assertEquals(built.get(i).metrics().get(recallKey), reopened.get(i).metrics().get(recallKey));
        }
    }

    @Test
    void timingBenchmarksReportPositiveValues() throws IOException {
        List<IvfQueryBenchmark> benchmarks = List.of(
                IvfThroughputBenchmark.createEmpty(IvfThroughputBenchmark.Mode.BATCHED, 1, 2).displayAvgQps(),
                IvfThroughputBenchmark.createEmpty(IvfThroughputBenchmark.Mode.COALESCED, 1, 2).displayAvgQps(),
                IvfThroughputBenchmark.createEmpty(IvfThroughputBenchmark.Mode.PER_QUERY, 1, 2).displayAvgQps(),
                IvfLatencyBenchmark.createEmpty(1).displayAvgLatency().displayP999Latency());
        // Each measured run starts with a 500 ms GC pause, so keep this grid to one point per thread count
        IvfConfig config = config(false);
        config.construction.storageFactor = List.of(1.0f);
        config.search.topKNprobe = Map.of(10, List.of(CLUSTERS));
        config.search.queryThreads = List.of(2);
        var rows = run(config, benchmarks);
        assertEquals(2, rows.size());
        assertFalse(rows.isEmpty());
        for (Row row : rows) {
            for (String key : List.of("search.throughput.avg_qps", "search.throughput.coalesced.avg_qps",
                                      "search.throughput.per_query.avg_qps",
                                      "search.latency.mean_ms", "search.latency.p999_ms")) {
                assertTrue(row.metrics().get(key) > 0, key + " = " + row.metrics().get(key) + " at " + row.point());
            }
        }
    }

    @Test
    void regressionOutputHasOneCsvRowPerDataset() throws IOException {
        // The regression benchmarks run 3 + 3 throughput passes per mode, so keep this to one row
        IvfConfig config = config(false);
        config.construction.storageFactor = List.of(1.0f);
        config.search.topKNprobe = Map.of(10, List.of(CLUSTERS));
        config.search.searchThreads = List.of(2);
        List<io.github.jbellis.jvector.example.BenchResult> results = new ArrayList<>();
        IvfGrid.run(ds, config, AutoBenchIvfYAML.defaultBenchmarks(),
                    (system, point, metrics) -> results.add(AutoBenchIvfYAML.toBenchResult(system, config, point, metrics)));
        assertEquals(1, results.size());
        assertEquals(16, results.get(0).parameters.get("nlist"));
        assertTrue((Double) results.get(0).metrics.get("QPS") > 0);
        assertTrue((Double) results.get(0).metrics.get("Coalesced QPS") > 0);
        assertTrue((Double) results.get(0).metrics.get("Per-query QPS") > 0);

        Path csv = dir.resolve("ivf.csv");
        AutoBenchIvfYAML.writeCsv(results, csv);
        List<String> lines = Files.readAllLines(csv);
        assertEquals(2, lines.size());
        assertTrue(lines.get(1).startsWith("synthetic,"), lines.get(1));
    }

    private record Row(SearchPoint point, Map<String, Double> metrics) {}

    private List<Row> run(IvfConfig config) throws IOException {
        return run(config, List.of(IvfAccuracyBenchmark.createDefault().displayMAP()));
    }

    private List<Row> run(IvfConfig config, List<IvfQueryBenchmark> benchmarks) throws IOException {
        List<Row> rows = new ArrayList<>();
        assertTrue(IvfGrid.run(ds, config, benchmarks,
                (system, point, metrics) -> rows.add(new Row(point, metrics.stream()
                        .collect(Collectors.toMap(Metric::getKey, Metric::getValue))))));
        return rows;
    }

    private IvfConfig config(boolean useSaved) {
        IvfConfig config = new IvfConfig();
        config.yamlSchemaVersion = 1;
        config.dataset = ds.getName();
        config.construction = new IvfConfig.Construction();
        config.construction.nlist = List.of(CLUSTERS);
        config.construction.storageFactor = List.of(1.0f, 2.0f);
        config.construction.buildThreads = 2;
        config.construction.ssdPaths = List.of(dir.toString());
        config.construction.useSavedIndexIfExists = useSaved;
        config.search = new IvfConfig.Search();
        config.search.topKNprobe = Map.of(10, List.of(4, CLUSTERS));
        config.search.searchThreads = List.of(1, 2);
        config.validate("test");
        return config;
    }

    private static float[] gaussian(Random r, float sigma, float[] center) {
        float[] v = new float[DIM];
        for (int i = 0; i < DIM; i++) {
            v[i] = (center == null ? 0 : center[i]) + (float) r.nextGaussian() * sigma;
        }
        return v;
    }

    private static float dot(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }
}
