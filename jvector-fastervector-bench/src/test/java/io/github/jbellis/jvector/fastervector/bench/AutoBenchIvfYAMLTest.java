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

import io.github.jbellis.jvector.example.BenchResult;
import io.github.jbellis.jvector.example.benchmarks.Metric;
import io.github.jbellis.jvector.fastervector.bench.IvfThroughputBenchmark.Mode;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** Arguments, result naming, and the CSV, without the native libraries. */
class AutoBenchIvfYAMLTest {
    /** AutoBenchYAML's header, copied from jvector-examples; the IVF CSV must start with it. */
    private static final String AUTOBENCH_HEADER = "dataset,QPS,QPS StdDev,Mean Latency,Recall@10,Index Construction Time,"
            + "Avg Nodes Visited,Build Heap Used (MB),Build Off-Heap (MB),Search Heap Used (MB),Search Off-Heap (MB)";

    @TempDir
    Path dir;

    @Test
    void parsesAutoBenchYamlArguments() {
        var args = AutoBenchIvfYAML.parseArgs(new String[]{"--output", "out", "--diag", "2", "cap-1M dpr", "--config", "c.yml"});
        assertEquals("out", args.output());
        assertEquals("c.yml", args.config());
        assertEquals(List.of("cap-1M", "dpr"), args.regexes());
        assertNull(AutoBenchIvfYAML.parseArgs(new String[]{"cap"}).output());
    }

    @Test
    void regexesMatchAnyPattern() {
        var p = AutoBenchIvfYAML.regexFor(List.of("cap-1M", "dpr"));
        assertTrue(p.matcher("cap-1M").find());
        assertTrue(p.matcher("dpr-gemma-1m").find());
        assertFalse(p.matcher("cohere-english-v3-1M").find());
        assertTrue(AutoBenchIvfYAML.regexFor(List.of()).matcher("anything").find());
    }

    @Test
    void metricsUseTheNamesTheSummarizerReads() {
        List<Metric> metrics = new ArrayList<>();
        metrics.addAll(IvfThroughputBenchmark.createDefault(Mode.BATCHED).metrics(new double[]{100, 100, 100}));
        metrics.addAll(IvfThroughputBenchmark.createDefault(Mode.COALESCED).metrics(new double[]{700, 700, 700}));
        metrics.addAll(IvfThroughputBenchmark.createDefault(Mode.PER_QUERY).metrics(new double[]{900, 900, 900}));
        metrics.addAll(IvfLatencyBenchmark.createDefault().metrics(new long[]{2_000_000}));
        metrics.add(Metric.of("search.accuracy.recall_at_10", "Recall@10", ".2f", 0.9));
        metrics.add(Metric.of("construction.index_build_time_s", "Index build time (s)", ".2f", 12));
        metrics.add(Metric.of("construction.index_file_size_mb", "Index file size (MB)", ".2f", 50));
        metrics.add(Metric.of("search.disk.total_file_size_mb", "Total file size (MB)", ".1f", 50));
        metrics.add(Metric.of("search.system.max_heap_mb", "Max heap usage (MB)", ".1f", 300));

        Map<String, Object> m = AutoBenchIvfYAML.resultMetrics(metrics);
        assertEquals(100.0, m.get("QPS"));
        assertEquals(0.0, m.get("± Std Dev"));
        assertEquals(700.0, m.get("Coalesced QPS"));
        assertEquals(900.0, m.get("Per-query QPS"));
        assertEquals(2.0, m.get("Mean Latency (ms)"));
        assertEquals(0.9, m.get("Recall@10"));
        assertEquals(12.0, m.get("Index Build Time"));
        assertEquals(50.0, m.get("Disk Usage (MB)"));
        assertEquals(300.0, m.get("Max heap usage (MB)"));
        assertFalse(m.containsKey("Total file size (MB)"), "the duplicate disk size is dropped");
    }

    @Test
    void csvStartsWithAutoBenchYamlColumnsAndUsesBatchedQps() throws IOException {
        List<BenchResult> results = List.of(
                result("cap-1M", 100, 900, 0.8, 10, 40),
                result("cap-1M", 300, 1100, 0.9, 20, 40),
                result("dpr-gemma-1m", 50, 60, 0.7, 30, 80));
        Path csv = dir.resolve("ivf.csv");
        AutoBenchIvfYAML.writeCsv(results, csv);

        List<String> lines = Files.readAllLines(csv);
        assertTrue(lines.get(0).startsWith(AUTOBENCH_HEADER + ","), lines.get(0));
        assertEquals(3, lines.size());
        String[] cap = lines.get(1).split(",", -1);
        assertEquals("cap-1M", cap[0]);
        assertEquals(200.0, Double.parseDouble(cap[1]), 1e-9, "QPS is the batched average, not per-query");
        assertEquals(0.85, Double.parseDouble(cap[4]), 1e-9);
        assertEquals("", cap[6], "Avg Nodes Visited is blank for IVF");
        assertEquals(500.0, Double.parseDouble(cap[11]), 1e-9, "Coalesced QPS");
        assertEquals(1000.0, Double.parseDouble(cap[13]), 1e-9, "Per-query QPS");
        assertEquals(40.0, Double.parseDouble(cap[15]), 1e-9);
        assertEquals(10.0, Double.parseDouble(cap[16]), 1e-9, "Index Memory (MB)");
        assertEquals(20.0, Double.parseDouble(cap[17]), 1e-9, "Search Peak Memory (MB)");
        assertEquals("", cap[18], "Build Peak Memory (MB) is blank when no row has it");
        assertEquals(19, cap.length);
        assertTrue(lines.get(2).startsWith("dpr-gemma-1m,50.0,"), lines.get(2));
    }

    private static BenchResult result(String dataset, double qps, double perQueryQps, double recall,
                                      double buildSeconds, double diskMb) {
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("QPS", qps);
        metrics.put("± Std Dev", 1.0);
        metrics.put("Coalesced QPS", perQueryQps / 2);
        metrics.put("Coalesced QPS StdDev", 3.0);
        metrics.put("Per-query QPS", perQueryQps);
        metrics.put("Per-query QPS StdDev", 2.0);
        metrics.put("Mean Latency (ms)", 1.5);
        metrics.put("Recall@10", recall);
        metrics.put("Index Build Time", buildSeconds);
        metrics.put("Disk Usage (MB)", diskMb);
        metrics.put("Index Memory (MB)", diskMb / 4);
        metrics.put("Search Peak Memory (MB)", diskMb / 2);
        return new BenchResult(dataset, Map.of("nprobe", 16), metrics);
    }
}
