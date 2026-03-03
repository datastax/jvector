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
 * Selections are typically run-level (from run.yml via {@link io.github.jbellis.jvector.example.yaml.RunConfig}).
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
     *
     * Prefix selections (@prefix:...) are treated as a family: we warn if expected sub-keys
     * (compute_time_s / encoding_time_s) are missing for any quantType that appears under that prefix.
     */
    public void warnMissing(List<Metric> results, ReportingSelectionResolver.ResolvedSelection resolved) {
        if ((resolved.keys().isEmpty()) && (resolved.prefixes().isEmpty())) {
            return;
        }

        Set<String> present = new HashSet<>();
        for (Metric m : results) {
            present.add(m.getKey());
        }

        // Warn for missing exact keys
        for (String k : resolved.keys()) {
            if (!present.contains(k) && warnedMissingKeys.add(k)) {
                System.err.println("WARNING: selected " + purpose + " output not available; skipping "
                        + resolved.selectorForKey(k) + " (" + k + ")");
            }
        }

        // Warn for missing sub-keys under each selected prefix, per quantType observed at runtime.
        for (String p : resolved.prefixes()) {
            String pDot = p + ".";
            Map<String, boolean[]> seen = new HashMap<>(); // qt -> [hasCompute, hasAnyEncoding]

            for (String k : present) {
                if (!k.startsWith(pDot)) continue;

                int qtStart = pDot.length();
                int qtEnd = k.indexOf('.', qtStart);
                if (qtEnd < 0) continue;

                String qt = k.substring(qtStart, qtEnd);
                String rest = k.substring(qtEnd + 1);

                boolean[] flags = seen.computeIfAbsent(qt, __ -> new boolean[2]);
                if (rest.equals("compute_time_s")) {
                    flags[0] = true;
                } else if (rest.equals("encoding_time_s") || rest.startsWith("encoding_time_s.")) {
                    flags[1] = true;
                }
            }

            // For each quant type that showed up under the prefix, warn if compute/encoding missing
            for (var e : seen.entrySet()) {
                String qt = e.getKey();
                boolean hasCompute = e.getValue()[0];
                boolean hasEnc = e.getValue()[1];

                String sel = resolved.selectorForPrefix(p);

                if (!hasCompute) {
                    String missingKey = p + "." + qt + ".compute_time_s";
                    if (warnedMissingKeys.add(missingKey)) {
                        System.err.println("WARNING: selected " + purpose + " output not available; skipping "
                                + sel + " (" + missingKey + ")");
                    }
                }
                // We don't measure encoding time for NVQ since it is encoded incrementally.
                boolean expectEncoding = "PQ".equals(qt) || "BQ".equals(qt);
                if (expectEncoding && !hasEnc) {
                    String missingKey = p + "." + qt + ".encoding_time_s";
                    if (warnedMissingKeys.add(missingKey)) {
                        System.err.println("WARNING: selected " + purpose + " output not available; skipping "
                                + sel + " (" + missingKey + ")");
                    }
                }
            }
        }
    }

    /** Apply the resolved selection as an intersection over Metric.key, preserving original order.
     * Keep metrics whose key matches exact OR prefix.
     * */
    public List<Metric> apply(List<Metric> results, ReportingSelectionResolver.ResolvedSelection resolved) {
        if ((resolved.keys().isEmpty()) && (resolved.prefixes().isEmpty())) {
            return results;
        }

        List<Metric> out = new ArrayList<>(results.size());
        for (Metric m : results) {
            if (resolved.matchesKey(m.getKey())) {
                out.add(m);
            }
        }
        return out;
    }
}
