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
import io.github.jbellis.jvector.example.reporting.DatasetInfoWriter;
import io.github.jbellis.jvector.example.reporting.ExperimentsCsvWriter;
import io.github.jbellis.jvector.example.reporting.ReportingSelectionResolver.ResolvedSelection;
import io.github.jbellis.jvector.example.reporting.RunContext;
import io.github.jbellis.jvector.example.reporting.RunReporting;
import io.github.jbellis.jvector.example.reporting.SearchSelection;
import io.github.jbellis.jvector.example.yaml.RunConfig;
import io.github.jbellis.jvector.fastervector.bench.IvfConfig.SearchPoint;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * run-config.yml applied to IVF results: which benchmarks to compute, which metrics the console
 * shows, and, when logging is enabled, an {@code experiments.csv} with IVF columns in the run's own
 * logging directory. Selections resolve through jvector-examples' {@link SearchSelection}, so the
 * graph names apply; a per-query or coalesced throughput metric is shown wherever its batched
 * counterpart is.
 */
final class IvfReporting {
    static final String PER_QUERY_PREFIX = "search.throughput.per_query.";
    static final String COALESCED_PREFIX = "search.throughput.coalesced.";
    private static final String BATCHED_PREFIX = "search.throughput.";

    static final List<String> FIXED_COLUMNS = List.of(
            "schema_version", "run_id", "run_uuid", "system_id", "index_type", "dataset_name",
            "nlist", "storageFactor", "buildThreads", "reopened", "topK", "nprobe", "searchThreads",
            "queryThreads");

    private static final List<String> THROUGHPUT_STATS =
            List.of("avg_qps", "stddev_qps", "cv_pct", "median_qps", "max_qps", "min_qps");

    private final Map<String, List<String>> benchmarksToCompute;
    private final SearchSelection consoleSelection;
    private final SearchSelection logSelection;
    private final RunContext run;
    private final ExperimentsCsvWriter experiments;
    private final DatasetInfoWriter datasetInfo;

    private IvfReporting(Map<String, List<String>> benchmarksToCompute, SearchSelection consoleSelection,
                         SearchSelection logSelection, RunContext run, ExperimentsCsvWriter experiments,
                         DatasetInfoWriter datasetInfo) {
        this.benchmarksToCompute = benchmarksToCompute;
        this.consoleSelection = consoleSelection;
        this.logSelection = logSelection;
        this.run = run;
        this.experiments = experiments;
        this.datasetInfo = datasetInfo;
    }

    /** {@code runCfg} null means no run-config.yml: default benchmarks, everything on the console, no logging. */
    static IvfReporting open(RunConfig runCfg, List<IvfConfig> configs) throws IOException {
        Map<String, List<String>> compute = runCfg == null ? null : runCfg.benchmarks;
        var console = SearchSelection.forConsole(
                runCfg == null || runCfg.console == null ? null : runCfg.console.benchmarks,
                runCfg == null || runCfg.console == null ? null : runCfg.console.metrics);
        var log = SearchSelection.forLogging(
                runCfg == null || runCfg.logging == null ? null : runCfg.logging.benchmarks,
                runCfg == null || runCfg.logging == null ? null : runCfg.logging.metrics);
        console.validate(compute);
        log.validate(compute);

        boolean loggingEnabled = runCfg != null && runCfg.logging != null
                && runCfg.logging.type != null && !runCfg.logging.type.isBlank();
        if (!loggingEnabled) {
            return new IvfReporting(compute, console, log, null, null, null);
        }
        RunContext run = RunReporting.open(runCfg).run();
        System.out.println("Logging enabled: " + run.runDir().toAbsolutePath() + ".\n"
                + "Delete this directory to remove run artifacts and free up disk space.");
        var experiments = new ExperimentsCsvWriter(run, FIXED_COLUMNS, loggingColumns(log, topKs(configs)));
        return new IvfReporting(compute, console, log, run, experiments, DatasetInfoWriter.open(run));
    }

    List<IvfQueryBenchmark> benchmarks() {
        return IvfBenchmarks.fromSpec(benchmarksToCompute);
    }

    /** The console's selection of {@code results}, warning once about selected metrics IVF doesn't produce. */
    List<Metric> forConsole(List<Metric> results, int topK) {
        ResolvedSelection resolved = consoleSelection.resolveForTopK(topK);
        consoleSelection.warnMissing(results, resolved);
        return select(results, resolved);
    }

    void registerDataset(String name, DataSet ds) throws IOException {
        if (datasetInfo != null) {
            datasetInfo.register(DatasetInfoWriter.fromDataSet(name, "", "", "", ds));
        }
    }

    void log(IvfSystem system, IvfConfig config, SearchPoint point, List<Metric> results) throws IOException {
        if (experiments == null) {
            return;
        }
        Map<String, String> fixed = new HashMap<>();
        fixed.put("schema_version", Integer.toString(run.schemaVersion()));
        fixed.put("run_id", run.runId());
        fixed.put("run_uuid", run.runUuid().toString());
        fixed.put("system_id", run.systemId());
        fixed.put("index_type", "ivf-ash");
        fixed.put("dataset_name", system.dataSet().getName());
        fixed.put("nlist", Integer.toString(system.constructionPoint().nlist()));
        fixed.put("storageFactor", Float.toString(system.constructionPoint().storageFactor()));
        fixed.put("buildThreads", Integer.toString(config.construction.buildThreads));
        fixed.put("reopened", Boolean.toString(system.reopened()));
        fixed.put("topK", Integer.toString(point.topK()));
        fixed.put("nprobe", Integer.toString(point.nprobe()));
        fixed.put("searchThreads", Integer.toString(point.searchThreads()));
        fixed.put("queryThreads", Integer.toString(point.queryThreads()));
        experiments.appendRow(fixed, results);
    }

    /**
     * Keeps metrics the selection matches, plus per-query and coalesced throughput wherever their
     * batched counterpart matches, and always the resident-memory metrics.
     */
    static List<Metric> select(List<Metric> results, ResolvedSelection resolved) {
        if (resolved.keys().isEmpty() && resolved.prefixes().isEmpty()) {
            return results;
        }
        List<Metric> out = new ArrayList<>(results.size());
        for (Metric m : results) {
            if (selected(m.getKey(), resolved) || MEMORY_KEYS.contains(m.getKey())) {
                out.add(m);
            }
        }
        return out;
    }

    /** Resident-memory metrics: the footprint comparison is a reason to run IVF, so they're always shown and logged. */
    static final List<String> MEMORY_KEYS = List.of(IvfSystem.INDEX_RSS_KEY, IvfGrid.SEARCH_PEAK_RSS_KEY,
                                                    IvfSystem.BUILD_PEAK_RSS_KEY);

    private static boolean selected(String key, ResolvedSelection resolved) {
        for (String prefix : List.of(PER_QUERY_PREFIX, COALESCED_PREFIX)) {
            if (key.startsWith(prefix)) {
                return resolved.matchesKey(BATCHED_PREFIX + key.substring(prefix.length()));
            }
        }
        return resolved.matchesKey(key);
    }

    /**
     * Metric columns for experiments.csv, in a fixed order: every IVF metric the logging selection
     * picks for any of {@code topKs}. Build time, index size and resident memory are always logged.
     */
    static List<String> loggingColumns(SearchSelection logSelection, Set<Integer> topKs) {
        Set<String> columns = new LinkedHashSet<>();
        for (int topK : new TreeSet<>(topKs)) {
            ResolvedSelection resolved = logSelection.resolveForTopK(topK);
            for (String key : allKeys(topK)) {
                if (selected(key, resolved) || (resolved.keys().isEmpty() && resolved.prefixes().isEmpty())) {
                    columns.add(key);
                }
            }
        }
        columns.add("construction.index_build_time_s");
        columns.add("construction.index_file_size_mb");
        columns.addAll(MEMORY_KEYS);
        return List.copyOf(columns);
    }

    /** Every metric key the IVF benchmarks can produce for one topK, in column order. */
    static List<String> allKeys(int topK) {
        List<String> keys = new ArrayList<>();
        for (String stat : THROUGHPUT_STATS) {
            keys.add(BATCHED_PREFIX + stat);
        }
        for (String stat : THROUGHPUT_STATS) {
            keys.add(COALESCED_PREFIX + stat);
        }
        for (String stat : THROUGHPUT_STATS) {
            keys.add(PER_QUERY_PREFIX + stat);
        }
        keys.add("search.latency.mean_ms");
        keys.add("search.latency.std_ms");
        keys.add("search.latency.p999_ms");
        keys.add("search.accuracy.recall_at_" + topK);
        keys.add("search.accuracy.map_at_" + topK);
        keys.add("construction.index_build_time_s");
        keys.add("construction.index_file_size_mb");
        keys.add("search.disk.total_file_size_mb");
        keys.add("search.disk.file_count");
        keys.add("search.system.max_heap_mb");
        keys.add("search.system.max_offheap_mb");
        keys.add("construction.build_heap_mb");
        keys.add("construction.build_offheap_mb");
        return keys;
    }

    static Set<Integer> topKs(List<IvfConfig> configs) {
        Set<Integer> topKs = new TreeSet<>();
        for (IvfConfig config : configs) {
            topKs.addAll(config.search.topKNprobe.keySet());
        }
        return topKs;
    }
}
