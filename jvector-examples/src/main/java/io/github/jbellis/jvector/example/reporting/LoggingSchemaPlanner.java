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

package io.github.jbellis.jvector.example.reporting;

import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.example.yaml.RunConfig;

import java.util.*;

public final class LoggingSchemaPlanner {

    private LoggingSchemaPlanner() {}

    public static List<String> unionLoggingMetricKeys(RunConfig runCfg, List<MultiConfig> allConfigs) {
        // No logging configured => no output-key columns
        if (runCfg == null || runCfg.logging == null) {
            return List.of();
        }

        // 1) Compute union of keys that may appear in this run
        Set<String> union = new HashSet<>();
        Set<Integer> allTopKs = new HashSet<>();

        for (MultiConfig cfg : allConfigs) {
            if (cfg == null || cfg.search == null) {
                continue;
            }
            if (cfg.search.topKOverquery != null) {
                allTopKs.addAll(cfg.search.topKOverquery.keySet()); // map key IS topK
            }
        }

        // Resolve per topK because accuracy keys depend on topK
        for (int topK : allTopKs) {
            var ctx = ReportingSelectionResolver.Context.of("topK", Integer.toString(topK));
            var resolved = ReportingSelectionResolver.resolve(runCfg.logging, SearchReportingCatalog.catalog(), ctx);
            union.addAll(resolved.keys());
        }

        // 2) Emit keys in canonical order, only if present in union
        List<Integer> sortedTopKs = new ArrayList<>(allTopKs);
        Collections.sort(sortedTopKs);

        List<String> ordered = new ArrayList<>();

        // throughput
        addIfPresent(union, ordered, "search.throughput.avg_qps");
        addIfPresent(union, ordered, "search.throughput.stddev_qps");
        addIfPresent(union, ordered, "search.throughput.cv_pct");
        addIfPresent(union, ordered, "search.throughput.median_qps");
        addIfPresent(union, ordered, "search.throughput.max_qps");
        addIfPresent(union, ordered, "search.throughput.min_qps");

        // latency
        addIfPresent(union, ordered, "search.latency.mean_ms");
        addIfPresent(union, ordered, "search.latency.std_ms");
        addIfPresent(union, ordered, "search.latency.p999_ms");

        // count
        addIfPresent(union, ordered, "search.count.avg_visited");
        addIfPresent(union, ordered, "search.count.avg_expanded");
        addIfPresent(union, ordered, "search.count.avg_expanded_base_layer");

        // accuracy (topK-dependent)
        for (int topK : sortedTopKs) {
            addIfPresent(union, ordered, "search.accuracy.recall_at_" + topK);
            addIfPresent(union, ordered, "search.accuracy.map_at_" + topK);
        }

        // named metrics
        addIfPresent(union, ordered, "search.system.max_heap_mb");
        addIfPresent(union, ordered, "search.system.max_offheap_mb");
        addIfPresent(union, ordered, "search.disk.total_file_size_mb");
        addIfPresent(union, ordered, "search.disk.file_count");
        addIfPresent(union, ordered, "construction.index_build_time_s");

        // Defensive: append any remaining keys (should be none) in sorted order
        if (!union.isEmpty()) {
            List<String> rest = new ArrayList<>(union);
            Collections.sort(rest);
            ordered.addAll(rest);
        }

        return ordered;
    }

    private static void addIfPresent(Set<String> union, List<String> ordered, String key) {
        if (union.remove(key)) {
            ordered.add(key);
        }
    }
}
