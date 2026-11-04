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
import io.github.jbellis.jvector.example.reporting.SearchSelection;
import io.github.jbellis.jvector.example.yaml.RunConfig;
import io.github.jbellis.jvector.fastervector.bench.IvfThroughputBenchmark.Mode;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/** run-config.yml handling, console selection, logging columns, the table, and argument parsing. */
class IvfReportingTest {
    @TempDir
    Path dir;

    @Test
    void defaultBenchmarksMeasureBothThroughputModesAndSkipCount() {
        List<IvfQueryBenchmark> benchmarks = IvfBenchmarks.fromSpec(null);
        assertEquals(List.of("IvfThroughputBenchmark", "IvfPerQueryThroughputBenchmark",
                             "IvfLatencyBenchmark", "IvfRecallBenchmark"),
                     benchmarks.stream().map(IvfQueryBenchmark::getBenchmarkName).collect(Collectors.toList()));
    }

    @Test
    void unknownStatIsRejected() {
        assertThrows(IllegalArgumentException.class,
                     () -> IvfBenchmarks.fromSpec(Map.of("throughput", List.of("P50"))));
    }

    @Test
    void perQueryThroughputFollowsItsBatchedCounterpart() {
        var selection = SearchSelection.forConsole(Map.of("throughput", List.of("AVG"), "accuracy", List.of("recall")), null);
        List<Metric> results = Stream.of(
                        "search.throughput.avg_qps", "search.throughput.per_query.avg_qps",
                        "search.throughput.median_qps", "search.throughput.per_query.median_qps",
                        "search.latency.mean_ms", "search.accuracy.recall_at_10")
                .map(k -> Metric.of(k, k, ".1f", 1)).collect(Collectors.toList());
        var shown = IvfReporting.select(results, selection.resolveForTopK(10)).stream().map(Metric::getKey).toList();
        assertEquals(List.of("search.throughput.avg_qps", "search.throughput.per_query.avg_qps",
                             "search.accuracy.recall_at_10"), shown);
    }

    @Test
    void emptySelectionShowsEverything() {
        var selection = SearchSelection.forConsole(null, null);
        List<Metric> results = List.of(Metric.of("search.throughput.per_query.avg_qps", "x", ".1f", 1));
        assertEquals(results, IvfReporting.select(results, selection.resolveForTopK(10)));
    }

    @Test
    void loggingColumnsFollowTheSelectionPerTopK() {
        var selection = SearchSelection.forLogging(Map.of("throughput", List.of("AVG"), "accuracy", List.of("recall")), null);
        assertEquals(List.of(
                        "search.throughput.avg_qps", "search.throughput.stddev_qps", "search.throughput.cv_pct",
                        "search.throughput.per_query.avg_qps", "search.throughput.per_query.stddev_qps",
                        "search.throughput.per_query.cv_pct",
                        "search.accuracy.recall_at_10", "search.accuracy.recall_at_100",
                        "construction.index_build_time_s", "construction.index_file_size_mb"),
                     IvfReporting.loggingColumns(selection, Set.of(10, 100)));
    }

    @Test
    void openWithoutRunConfigComputesDefaults() throws IOException {
        IvfReporting reporting = IvfReporting.open(null, List.of(IvfConfig.getDefaultConfig("x")));
        assertEquals(4, reporting.benchmarks().size());
        List<Metric> results = List.of(Metric.of("search.latency.mean_ms", "x", ".1f", 1));
        assertEquals(results, reporting.forConsole(results, 10));
    }

    @Test
    void loggingWritesIvfColumnsInTheRunDirectory() throws IOException {
        RunConfig runCfg = new RunConfig();
        runCfg.benchmarks = Map.of("throughput", List.of("AVG"), "accuracy", List.of("recall"));
        runCfg.logging = new RunConfig.RunLogging();
        runCfg.logging.dir = dir.toString();
        runCfg.logging.type = "csv";
        runCfg.logging.runId = "ivf-test";
        runCfg.logging.benchmarks = Map.of("accuracy", List.of("recall"));
        IvfReporting.open(runCfg, List.of(IvfConfig.getDefaultConfig("x")));

        // The run directory is the runId plus a timestamp
        Path runDir;
        try (Stream<Path> runs = Files.list(dir)) {
            runDir = runs.filter(p -> p.getFileName().toString().startsWith("ivf-test")).findFirst().orElseThrow();
        }
        String header = Files.readAllLines(runDir.resolve("experiments.csv")).get(0);
        assertTrue(header.startsWith(String.join(",", IvfReporting.FIXED_COLUMNS)), header);
        assertTrue(header.contains("index_type") && header.contains("nprobe"), header);
        assertTrue(header.endsWith("search.accuracy.recall_at_10,search.accuracy.recall_at_100,"
                                   + "construction.index_build_time_s,construction.index_file_size_mb"), header);
    }

    @Test
    void tableHasNprobeAndThreadColumns() {
        var bytes = new ByteArrayOutputStream();
        var table = new IvfTablePrinter(new PrintStream(bytes, true, StandardCharsets.UTF_8));
        List<Metric> metrics = IvfThroughputBenchmark.createDefault(Mode.PER_QUERY).metrics(new double[]{1234.5});
        table.printRow(16, 4, metrics);
        table.printRow(32, 4, metrics);
        String[] lines = bytes.toString(StandardCharsets.UTF_8).strip().split("\n");
        assertTrue(lines[0].startsWith("nprobe") && lines[0].contains("Threads") && lines[0].contains("Avg per-query"),
                   lines[0]);
        assertTrue(lines[1].contains("(of 3)"), "second header line: " + lines[1]);   // createDefault's run count
        assertTrue(lines[2].matches("-+"), lines[2]);
        assertTrue(lines[3].startsWith("16") && lines[3].contains("1234.5"), lines[3]);
        assertTrue(lines[4].startsWith("32"), lines[4]);
        assertEquals(5, lines.length, "one header for both rows");
    }

    @Test
    void argumentsSelectDatasetsAndConfigFiles() throws IOException {
        List<String> all = List.of("cap-1M", "cap-6M", "glove-100-angular.hdf5", "dpr-gemma-1m");
        assertEquals(List.of("cap-1M", "cap-6M"), datasets(BenchIvfYAML.configsFor(new String[]{"cap"}, all)));
        assertEquals(List.of("glove-100-angular"), datasets(BenchIvfYAML.configsFor(new String[]{"glove"}, all)));
        assertEquals(List.of("cap-1M", "dpr-gemma-1m"),
                     datasets(BenchIvfYAML.configsFor(new String[]{"cap-1M dpr"}, all)));
        assertEquals(all.size(), BenchIvfYAML.configsFor(new String[]{}, all).size());

        Path file = dir.resolve("mine.yml");
        Files.copy(Path.of(IvfConfig.DEFAULT_DIRECTORY, "default.yml"), file);
        // A config file alone runs only that file
        assertEquals(List.of("default"), datasets(BenchIvfYAML.configsFor(new String[]{file.toString()}, all)));
        assertThrows(FileNotFoundException.class,
                     () -> BenchIvfYAML.configsFor(new String[]{dir.resolve("absent.yml").toString()}, all));
    }

    private static List<String> datasets(List<IvfConfig> configs) {
        return configs.stream().map(c -> c.dataset).collect(Collectors.toList());
    }
}
