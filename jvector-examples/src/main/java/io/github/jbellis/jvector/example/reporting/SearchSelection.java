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

import io.github.jbellis.jvector.example.benchmarks.Metric;
import io.github.jbellis.jvector.example.yaml.BenchmarkSelection;
import io.github.jbellis.jvector.example.yaml.MetricSelection;

import java.util.*;

/**
 * Encapsulates selection + resolution + warning + application for a single sink (console/logging)
 * in the search phase, using {@link ReportingSelectionResolver} and {@link SearchReportingCatalog}.
 *
 * This prevents call-site ordering mistakes (validate -> resolve -> warn -> apply).
 */
public final class SearchSelection {
    private final String purpose; // "console" or "logging" (used in warnings)
    private final BenchmarkSelection selection;
    private final Set<String> warnedMissingKeys = new HashSet<>();

    private SearchSelection(String purpose, Map<String, List<String>> benchmarks, MetricSelection metrics) {
        this.purpose = purpose;
        this.selection = new BenchmarkSelection();
        this.selection.benchmarks = benchmarks;
        this.selection.metrics = metrics;
    }

    public static SearchSelection forConsole(Map<String, List<String>> benchmarks, MetricSelection metrics) {
        return new SearchSelection("console", benchmarks, metrics);
    }

    public static SearchSelection forLogging(Map<String, List<String>> benchmarks, MetricSelection metrics) {
        return new SearchSelection("logging", benchmarks, metrics);
    }

    /** Fail-fast validations that are cheap and should be done before index construction. */
    public void validate(Map<String, List<String>> benchmarksToCompute) {
        ReportingSelectionResolver.validateBenchmarkSelectionSubset(
                benchmarksToCompute,
                selection.benchmarks,
                SearchReportingCatalog.defaultComputeBenchmarks()
        );
        ReportingSelectionResolver.validateNamedMetricSelectionNames(
                selection.metrics,
                SearchReportingCatalog.catalog()
        );
    }

    /** Resolve the selection to concrete Metric.key strings for a particular topK. */
    public ReportingSelectionResolver.ResolvedSelection resolveForTopK(int topK) {
        var ctx = ReportingSelectionResolver.Context.of("topK", Integer.toString(topK));
        return ReportingSelectionResolver.resolve(selection, SearchReportingCatalog.catalog(), ctx);
    }

    /**
     * Warn once per missing selected key (best-effort availability at runtime).
     * Uses YAML-ish selectors for readability, and includes the underlying key.
     */
    public void warnMissing(List<Metric> results, ReportingSelectionResolver.ResolvedSelection resolved) {
        if (resolved.keys().isEmpty()) {
            return;
        }

        Set<String> present = new HashSet<>();
        for (Metric m : results) {
            present.add(m.getKey());
        }

        for (String k : resolved.keys()) {
            if (!present.contains(k) && warnedMissingKeys.add(k)) {
                System.err.println("WARNING: selected " + purpose + " output not available; skipping "
                        + resolved.selectorForKey(k) + " (" + k + ")");
            }
        }
    }

    /** Apply the resolved selection as an intersection over Metric.key, preserving original order. */
    public List<Metric> apply(List<Metric> results, ReportingSelectionResolver.ResolvedSelection resolved) {
        if (resolved.keys().isEmpty()) {
            return results;
        }

        Set<String> allowed = resolved.keys();
        List<Metric> out = new ArrayList<>(results.size());
        for (Metric m : results) {
            if (allowed.contains(m.getKey())) {
                out.add(m);
            }
        }
        return out;
    }
}
