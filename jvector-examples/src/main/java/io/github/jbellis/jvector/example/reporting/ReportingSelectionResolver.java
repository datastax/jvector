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

import io.github.jbellis.jvector.example.yaml.BenchmarkSelection;
import io.github.jbellis.jvector.example.yaml.MetricSelection;

import java.util.*;

/**
 * Generic selection resolver for reporting sinks (console/logging/etc).
 *
 * Inputs:
 *  - computeBenchmarks: authoritative benchmark compute spec (YAML names)
 *  - selection: sink selection (benchmarks subset + named metrics)
 *  - catalog: mapping from YAML names -> Metric.key templates/keys
 *  - ctx: context for template substitution (e.g., topK)
 *
 * Rules:
 *  - benchmark stat selection must be a subset of computed stats (fail fast)
 *  - named metric selection must use recognized names (fail fast)
 *  - runtime availability is best effort: caller can warn+omit missing produced keys
 */
public final class ReportingSelectionResolver {

    private ReportingSelectionResolver() {}

    /** Simple substitution context, e.g. {"topK":"10"} used for key templates. */
    public static final class Context {
        private final Map<String, String> vars;

        private Context(Map<String, String> vars) {
            this.vars = Collections.unmodifiableMap(new HashMap<>(vars));
        }

        public static Context of(String k, String v) {
            return new Context(Map.of(k, v));
        }

        public static Context of(Map<String, String> vars) {
            return new Context(vars);
        }

        public String get(String k) {
            return vars.get(k);
        }

        public Map<String, String> vars() {
            return vars;
        }
    }

    /**
     * Catalog for a phase (search/construction/etc). This is data, not sink-specific.
     *
     * - benchmarkKeyTemplates: type -> stat -> list of Metric.key templates (may contain "{topK}" etc)
     * - namedMetricKeys: category -> name -> Metric.key (may also contain templates, if desired)
     * - benchmarkYamlPrefix / metricsYamlPrefix: used for user-facing selectors in warnings/errors
     */
    public static final class Catalog {
        public final Map<String, Map<String, List<String>>> benchmarkKeyTemplates;
        public final Map<String, Map<String, String>> namedMetricKeys;
        public final String benchmarkYamlPrefix; // e.g. "search.console.benchmarks"
        public final String metricsYamlPrefix;   // e.g. "search.console.metrics"

        public Catalog(Map<String, Map<String, List<String>>> benchmarkKeyTemplates,
                       Map<String, Map<String, String>> namedMetricKeys,
                       String benchmarkYamlPrefix,
                       String metricsYamlPrefix) {
            this.benchmarkKeyTemplates = benchmarkKeyTemplates;
            this.namedMetricKeys = namedMetricKeys;
            this.benchmarkYamlPrefix = benchmarkYamlPrefix;
            this.metricsYamlPrefix = metricsYamlPrefix;
        }
    }

    /** Result: concrete keys + a YAML-ish label per key for warnings/errors. */
    public static final class ResolvedSelection {
        private final Set<String> keys;
        private final Map<String, String> keyToSelector;

        public ResolvedSelection(Set<String> keys, Map<String, String> keyToSelector) {
            this.keys = Collections.unmodifiableSet(new HashSet<>(keys));
            this.keyToSelector = Collections.unmodifiableMap(new HashMap<>(keyToSelector));
        }

        public Set<String> keys() { return keys; }

        /** For warnings/errors: returns something like "search.console.benchmarks.latency.P999". */
        public String selectorForKey(String key) {
            return keyToSelector.getOrDefault(key, key);
        }
    }

    // -----------------------------
    // Validation (pre-build)
    // -----------------------------

    /** If compute spec is null/empty, treat as defaultCompute. */
    public static Map<String, List<String>> effectiveComputeSpec(Map<String, List<String>> compute,
                                                                 Map<String, List<String>> defaultCompute) {
        return (compute == null || compute.isEmpty()) ? defaultCompute : compute;
    }

    /** Fail-fast: benchmark stat selection must be a subset of computed stats (YAML names). */
    public static void validateBenchmarkSelectionSubset(Map<String, List<String>> computeBenchmarks,
                                                        Map<String, List<String>> selectedBenchmarks,
                                                        Map<String, List<String>> defaultCompute) {
        if (selectedBenchmarks == null || selectedBenchmarks.isEmpty()) {
            return;
        }

        Map<String, List<String>> compute = effectiveComputeSpec(computeBenchmarks, defaultCompute);

        for (var e : selectedBenchmarks.entrySet()) {
            String type = e.getKey();
            List<String> wanted = e.getValue();

            List<String> available = compute.get(type);
            if (available == null) {
                throw new IllegalArgumentException(
                        "Selection requests benchmark type not computed: " + type + "\n" +
                                "Computed types: " + compute.keySet()
                );
            }
            for (String stat : wanted) {
                if (!available.contains(stat)) {
                    throw new IllegalArgumentException(
                            "Selection requests stat not computed: benchmarks." + type + "." + stat + "\n" +
                                    "Computed stats for '" + type + "': " + available
                    );
                }
            }
        }
    }

    /** Fail-fast: named metrics must be recognized (availability is runtime/best-effort). */
    public static void validateNamedMetricSelectionNames(MetricSelection metricsToSelect, Catalog catalog) {
        if (metricsToSelect == null || metricsToSelect.isEmpty()) {
            return;
        }

        for (var e : metricsToSelect.entrySet()) {
            String category = e.getKey();
            Map<String, String> nameToKey = catalog.namedMetricKeys.get(category);
            if (nameToKey == null) {
                throw new IllegalArgumentException(
                        "Unknown " + catalog.metricsYamlPrefix + " category: " + category + "\n" +
                                "Valid categories: " + catalog.namedMetricKeys.keySet()
                );
            }
            for (String name : e.getValue()) {
                if (!nameToKey.containsKey(name)) {
                    throw new IllegalArgumentException(
                            "Unknown " + catalog.metricsYamlPrefix + "." + category + " value: " + name + "\n" +
                                    "Valid names for '" + category + "': " + nameToKey.keySet()
                    );
                }
            }
        }
    }

    // -----------------------------
    // Resolution (to concrete Metric.key set)
    // -----------------------------

    public static ResolvedSelection resolve(BenchmarkSelection selection, Catalog catalog, Context ctx) {
        if (selection == null) {
            return new ResolvedSelection(Set.of(), Map.of());
        }

        Set<String> keys = new HashSet<>();
        Map<String, String> keyToSelector = new HashMap<>();

        // Benchmarks (type/stat -> templates -> keys)
        if (selection.benchmarks != null && !selection.benchmarks.isEmpty()) {
            for (var e : selection.benchmarks.entrySet()) {
                String type = e.getKey();
                List<String> stats = e.getValue();

                for (String stat : stats) {
                    List<String> expanded = expandBenchmarkStat(type, stat, catalog, ctx);
                    String selector = catalog.benchmarkYamlPrefix + "." + type + "." + stat;
                    for (String k : expanded) {
                        keys.add(k);
                        keyToSelector.putIfAbsent(k, selector);
                    }
                }
            }
        }

        // Named metrics (category/name -> key)
        if (selection.metrics != null && !selection.metrics.isEmpty()) {
            // caller should validate names pre-build, but keep this defensive
            validateNamedMetricSelectionNames(selection.metrics, catalog);

            for (var e : selection.metrics.entrySet()) {
                String category = e.getKey();
                for (String name : e.getValue()) {
                    String k = substitute(catalog.namedMetricKeys.get(category).get(name), ctx);
                    keys.add(k);
                    keyToSelector.putIfAbsent(k, catalog.metricsYamlPrefix + "." + category + "." + name);
                }
            }
        }

        return new ResolvedSelection(keys, keyToSelector);
    }

    private static List<String> expandBenchmarkStat(String type, String stat, Catalog catalog, Context ctx) {
        Map<String, List<String>> byStat = catalog.benchmarkKeyTemplates.get(type);
        if (byStat == null) {
            throw new IllegalArgumentException("Unknown benchmark type in selection: " + type);
        }
        List<String> templates = byStat.get(stat);
        if (templates == null) {
            throw new IllegalArgumentException("Unknown stat in selection: " + type + "." + stat);
        }
        List<String> out = new ArrayList<>(templates.size());
        for (String t : templates) {
            out.add(substitute(t, ctx));
        }
        return out;
    }

    private static String substitute(String template, Context ctx) {
        if (template == null) return null;
        String s = template;
        for (var e : ctx.vars().entrySet()) {
            s = s.replace("{" + e.getKey() + "}", e.getValue());
        }
        return s;
    }
}
