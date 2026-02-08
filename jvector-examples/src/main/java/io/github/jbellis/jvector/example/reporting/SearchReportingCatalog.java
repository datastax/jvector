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

import java.util.List;
import java.util.Map;

/**
 * Search-phase reporting catalog:
 * - Default compute spec (when search.benchmarks is omitted)
 * - Mapping from YAML benchmark stat selections -> Metric.key templates
 * - Mapping from YAML named metrics -> Metric.key
 *
 * This is sink-neutral: messages refer to "benchmarks.*" and "metrics.*".
 * Console/logging can both reuse it without a class matrix.
 */
public final class SearchReportingCatalog {

    private SearchReportingCatalog() {}

    /** Default search.benchmarks when YAML omits the benchmarks section entirely. */
    public static Map<String, List<String>> defaultComputeBenchmarks() {
        return Map.of(
                "throughput", List.of("AVG"),
                "latency",    List.of("AVG"),
                "count",      List.of("visited"),
                "accuracy",   List.of("recall")
        );
    }

    /** Catalog used by ReportingSelectionResolver for the search phase. */
    public static ReportingSelectionResolver.Catalog catalog() {
        return new ReportingSelectionResolver.Catalog(
                benchmarkKeyTemplates(),
                namedMetricKeys(),
                "benchmarks",  // sink-neutral selector prefix
                "metrics"      // sink-neutral selector prefix
        );
    }

    private static Map<String, Map<String, List<String>>> benchmarkKeyTemplates() {
        return Map.of(
                "throughput", Map.of(
                        "AVG",    List.of("search.throughput.avg_qps",
                                "search.throughput.stddev_qps",
                                "search.throughput.cv_pct"),
                        "MEDIAN", List.of("search.throughput.median_qps"),
                        "MAX",    List.of("search.throughput.max_qps",
                                "search.throughput.min_qps")
                ),
                "latency", Map.of(
                        "AVG",  List.of("search.latency.mean_ms"),
                        "STD",  List.of("search.latency.std_ms"),
                        "P999", List.of("search.latency.p999_ms")
                ),
                "count", Map.of(
                        "visited",             List.of("search.count.avg_visited"),
                        "expanded",            List.of("search.count.avg_expanded"),
                        "expanded base layer", List.of("search.count.avg_expanded_base_layer")
                ),
                "accuracy", Map.of(
                        // topK-dependent templates:
                        "recall", List.of("search.accuracy.recall_at_{topK}"),
                        "MAP",    List.of("search.accuracy.map_at_{topK}")
                )
        );
    }

    private static Map<String, Map<String, String>> namedMetricKeys() {
        return Map.of(
                "system", Map.of(
                        "max_heap_mb",    "search.system.max_heap_mb",
                        "max_offheap_mb", "search.system.max_offheap_mb"
                ),
                "disk", Map.of(
                        "total_file_size_mb", "search.disk.total_file_size_mb",
                        "file_count",         "search.disk.file_count"
                ),
                "construction", Map.of(
                        "index_build_time_s", "construction.index_build_time_s"
                )
        );
    }
}
