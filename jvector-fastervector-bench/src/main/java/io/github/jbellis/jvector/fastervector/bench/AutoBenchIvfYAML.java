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

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.jbellis.jvector.example.BenchResult;
import io.github.jbellis.jvector.example.benchmarks.Metric;
import io.github.jbellis.jvector.example.benchmarks.datasets.BaseVectorLoading;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSetInfo;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.example.util.BenchmarkSummarizer;
import io.github.jbellis.jvector.example.util.CheckpointManager;
import io.github.jbellis.jvector.example.yaml.DatasetCollection;
import io.github.jbellis.jvector.fastervector.bench.IvfConfig.SearchPoint;
import io.github.jbellis.jvector.fastervector.bench.IvfThroughputBenchmark.Mode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * IVF regression runner, the counterpart of jvector-examples' {@code AutoBenchYAML}, with the same
 * arguments and output files so the two can be compared side by side:
 * <pre>
 *   AutoBenchIvfYAML --output &lt;prefix&gt; [--config &lt;ivf-config.yml&gt;] [dataset regex ...]
 * </pre>
 * Datasets come from the {@code regression-tests} section of datasets.yml. Writes
 * {@code <prefix>.json} (every result as a {@link BenchResult}), {@code <prefix>.csv} (one row per
 * dataset, its first 11 columns exactly AutoBenchYAML's), and {@code <prefix>.checkpoint.json}, so
 * an interrupted run resumes. Run it from the repository root.
 */
public class AutoBenchIvfYAML {
    private static final Logger logger = LoggerFactory.getLogger(AutoBenchIvfYAML.class);
    private static final String REGRESSION_TEST_KEY = "regression-tests";

    /** AutoBenchYAML's columns, then the IVF-only ones. */
    static final String CSV_HEADER = "dataset,QPS,QPS StdDev,Mean Latency,Recall@10,Index Construction Time,Avg Nodes Visited,"
            + "Build Heap Used (MB),Build Off-Heap (MB),Search Heap Used (MB),Search Off-Heap (MB),"
            + "Coalesced QPS,Coalesced QPS StdDev,Per-query QPS,Per-query QPS StdDev,Index Size (MB),Index Memory (MB),Search Peak Memory (MB),"
            + "Build Peak Memory (MB)";

    /** Metric keys to the BenchResult names BenchmarkSummarizer reads; anything else keeps its header. */
    private static final Map<String, String> RESULT_NAMES = Map.ofEntries(
            // "QPS" exactly, so the summarizer picks the batched number before its "contains QPS" fallback
            Map.entry("search.throughput.avg_qps", "QPS"),
            Map.entry("search.throughput.stddev_qps", "± Std Dev"),
            Map.entry("search.throughput.cv_pct", "QPS CV %"),
            Map.entry("search.throughput.coalesced.avg_qps", "Coalesced QPS"),
            Map.entry("search.throughput.coalesced.stddev_qps", "Coalesced QPS StdDev"),
            Map.entry("search.throughput.coalesced.cv_pct", "Coalesced QPS CV %"),
            Map.entry("search.throughput.per_query.avg_qps", "Per-query QPS"),
            Map.entry("search.throughput.per_query.stddev_qps", "Per-query QPS StdDev"),
            Map.entry("search.throughput.per_query.cv_pct", "Per-query QPS CV %"),
            Map.entry("construction.index_build_time_s", "Index Build Time"),
            Map.entry("construction.index_file_size_mb", "Disk Usage (MB)"),
            Map.entry("construction.build_heap_mb", "Heap Memory Used (MB)"),
            Map.entry("construction.build_offheap_mb", "Total Off-Heap (MB)"),
            Map.entry("search.disk.total_file_size_mb", ""),   // same value as Disk Usage (MB)
            Map.entry("search.system.max_heap_mb", "Max heap usage (MB)"),
            Map.entry("search.system.max_offheap_mb", "Max offheap usage (MB)"),
            Map.entry(IvfSystem.INDEX_RSS_KEY, "Index Memory (MB)"),
            Map.entry(IvfGrid.SEARCH_PEAK_RSS_KEY, "Search Peak Memory (MB)"),
            Map.entry(IvfSystem.BUILD_PEAK_RSS_KEY, "Build Peak Memory (MB)"));

    record Args(String output, String config, List<String> regexes) {}

    public static void main(String[] args) throws IOException {
        Args parsed = parseArgs(args);
        if (parsed.output() == null) {
            logger.error("Error: --output argument is required for AutoBenchIvfYAML");
            System.exit(1);
        }
        logger.info("Heap space available is {}", Runtime.getRuntime().maxMemory());

        CheckpointManager checkpoint = new CheckpointManager(parsed.output());
        logger.info("Already completed datasets: {}", checkpoint.getCompletedDatasets());

        Pattern pattern = regexFor(parsed.regexes());
        List<String> datasets = DatasetCollection.load().getSection(REGRESSION_TEST_KEY).stream()
                .filter(name -> pattern.matcher(name).find())
                .collect(Collectors.toList());
        if (datasets.isEmpty()) {
            throw new RuntimeException("No datasets matched the given patterns, nothing to do");
        }
        logger.info("Executing the following datasets: {}", datasets);

        List<BenchResult> results = new ArrayList<>(checkpoint.getCompletedResults());
        for (String datasetName : datasets) {
            if (checkpoint.isDatasetCompleted(datasetName)) {
                logger.info("Skipping already completed dataset: {}", datasetName);
                continue;
            }
            try {
                List<BenchResult> datasetResults = runDataset(datasetName, parsed.config());
                results.addAll(datasetResults);
                checkpoint.markDatasetCompleted(datasetName, datasetResults);
            } catch (Exception e) {
                logger.error("Exception while processing dataset {}", datasetName, e);
            }
        }

        new ObjectMapper().writerWithDefaultPrettyPrinter().writeValue(new File(parsed.output() + ".json"), results);
        Path csv = Path.of(parsed.output() + ".csv");
        writeCsv(results, csv);
        logger.info("Benchmark results written to {} and {}.json", csv, parsed.output());
    }

    /** --output and --config take a value; --diag (AutoBenchYAML's) is accepted and ignored; the rest are regexes. */
    static Args parseArgs(String[] args) {
        String output = null;
        String config = null;
        List<String> regexes = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--output" -> output = i + 1 < args.length ? args[++i] : null;
                case "--config" -> config = i + 1 < args.length ? args[++i] : null;
                case "--diag" -> i++;
                default -> {
                    for (String word : args[i].split("\\s")) {
                        if (!word.isEmpty()) {
                            regexes.add(word);
                        }
                    }
                }
            }
        }
        return new Args(output, config, regexes);
    }

    static Pattern regexFor(List<String> regexes) {
        if (regexes.isEmpty()) {
            return Pattern.compile(".*");
        }
        return Pattern.compile(regexes.stream().map(r -> "(?:" + r + ")").collect(Collectors.joining("|")));
    }

    private static List<BenchResult> runDataset(String datasetName, String configPath) throws IOException {
        String name = datasetName.endsWith(".hdf5")
                ? datasetName.substring(0, datasetName.length() - ".hdf5".length())
                : datasetName;
        IvfConfig config = configPath != null ? IvfConfig.getConfig(configPath) : IvfConfig.getDefaultConfig(name);
        config.dataset = name;
        logger.info("Using configuration: {}", config);

        DataSetInfo info = DataSets.loadDataSet(datasetName, BaseVectorLoading.forBenchmark(false))
                .orElseThrow(() -> new RuntimeException("Dataset " + datasetName + " not found"));
        Optional<String> unsupported = info.similarityFunction().flatMap(IvfGrid::unsupportedReason);
        if (unsupported.isPresent()) {
            logger.warn("Skipping dataset {}: {}", datasetName, unsupported.get());
            return List.of();
        }

        List<BenchResult> results = new ArrayList<>();
        try (DataSet ds = info.getDataSet()) {
            logger.info("Dataset loaded: {} with {} vectors", datasetName, ds.getBaseVectors().size());
            IvfGrid.run(ds, config, defaultBenchmarks(),
                        (system, point, metrics) -> results.add(toBenchResult(system, config, point, metrics)));
        }
        logger.info("Benchmark completed for dataset: {}", datasetName);
        return results;
    }

    /**
     * Fixed, as AutoBenchYAML's are: average QPS batched, coalesced and per-query (per-query
     * unless {@code -Divf.perQuery=false}), mean latency, recall.
     */
    static List<IvfQueryBenchmark> defaultBenchmarks() {
        return defaultBenchmarks(IvfBenchmarks.perQueryEnabled());
    }

    static List<IvfQueryBenchmark> defaultBenchmarks(boolean perQuery) {
        List<IvfQueryBenchmark> out = new ArrayList<>();
        for (Mode mode : IvfBenchmarks.throughputModes(perQuery)) {
            out.add(IvfThroughputBenchmark.createDefault(mode));
        }
        out.add(IvfLatencyBenchmark.createDefault());
        out.add(IvfAccuracyBenchmark.createDefault());
        return out;
    }

    static BenchResult toBenchResult(IvfSystem system, IvfConfig config, SearchPoint point, List<Metric> metrics) {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("indexType", "ivf-ash");
        params.put("nlist", system.constructionPoint().nlist());
        params.put("storageFactor", system.constructionPoint().storageFactor());
        params.put("buildThreads", config.construction.buildThreads);
        params.put("reopened", system.reopened());
        params.put("topK", point.topK());
        params.put("nprobe", point.nprobe());
        params.put("searchThreads", point.searchThreads());
        params.put("queryThreads", point.queryThreads());
        return new BenchResult(system.dataSet().getName(), params, resultMetrics(metrics));
    }

    /** The metrics under the names BenchmarkSummarizer and AutoBenchYAML's JSON use. */
    static Map<String, Object> resultMetrics(List<Metric> metrics) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Metric m : metrics) {
            String name = RESULT_NAMES.getOrDefault(m.getKey(), m.getHeader());
            if (!name.isEmpty()) {
                out.put(name, m.getValue());
            }
        }
        return out;
    }

    /**
     * One row per dataset, in first-seen order. The shared columns come from jvector-examples'
     * {@link BenchmarkSummarizer}, so they're computed exactly as AutoBenchYAML computes them; node
     * counts don't apply to IVF and are left blank.
     */
    static void writeCsv(List<BenchResult> results, Path csv) throws IOException {
        Map<String, BenchmarkSummarizer.SummaryStats> stats = BenchmarkSummarizer.summarizeByDataset(results);
        Set<String> datasets = new LinkedHashSet<>();
        results.forEach(r -> datasets.add(r.dataset));
        try (Writer w = Files.newBufferedWriter(csv, StandardCharsets.UTF_8)) {
            w.write(CSV_HEADER + "\n");
            for (String dataset : datasets) {
                var s = stats.get(dataset);
                List<BenchResult> rows = results.stream().filter(r -> r.dataset.equals(dataset)).toList();
                w.write(String.join(",",
                        dataset,
                        Double.toString(s.getAvgQps()),
                        Double.toString(s.getQpsStdDev()),
                        Double.toString(s.getAvgLatency()),
                        Double.toString(s.getAvgRecall()),
                        Double.toString(s.getIndexConstruction()),
                        "",
                        Double.toString(s.getAvgBuildHeapMB()),
                        Double.toString(s.getAvgBuildOffHeapMB()),
                        Double.toString(s.getAvgSearchHeapMB()),
                        Double.toString(s.getAvgSearchOffHeapMB()),
                        average(rows, "Coalesced QPS"),
                        average(rows, "Coalesced QPS StdDev"),
                        average(rows, "Per-query QPS"),
                        average(rows, "Per-query QPS StdDev"),
                        average(rows, "Disk Usage (MB)"),
                        average(rows, "Index Memory (MB)"),
                        average(rows, "Search Peak Memory (MB)"),
                        average(rows, "Build Peak Memory (MB)")) + "\n");
            }
        }
    }

    private static String average(List<BenchResult> rows, String metric) {
        double sum = 0;
        int n = 0;
        for (BenchResult r : rows) {
            if (r.metrics != null && r.metrics.get(metric) instanceof Number v) {
                sum += v.doubleValue();
                n++;
            }
        }
        return n == 0 ? "" : Double.toString(sum / n);
    }
}
