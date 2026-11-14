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

package io.github.jbellis.jvector.example.yaml;

import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.Objects;

public class SearchParameters extends CommonParameters {
    /** ASH query arithmetic; omitted settings preserve JVM properties and auto. */
    public String queryProjection;

    public static final String ASH_QUERY_PROJECTION_PROPERTY = "jvector.ash.queryProjection";

    /** Resolve one precision before loading models; cached views cannot change mid-JVM. */
    public static final class AshQueryProjectionSetting implements AutoCloseable {
        private final String original = System.getProperty(ASH_QUERY_PROJECTION_PROPERTY);
        private boolean applied;

        public void apply(List<MultiConfig> configs) {
            String selected = null;
            boolean explicitYaml = false;
            List<String> settings = new ArrayList<>();
            boolean usesAsh = false;
            for (MultiConfig config : configs) {
                boolean configUsesAsh = hasAsh(config.construction) || hasAsh(config.search);
                usesAsh |= configUsesAsh;
                if (config.usesSchema3()) {
                    collectProjectionSettings(config.construction, settings);
                    collectProjectionSettings(config.search, settings);
                } else {
                    String legacy = config.search == null ? null : config.search.queryProjection;
                    if (legacy != null) validatePrecision(legacy);
                    if (configUsesAsh) settings.add(legacy);
                }
            }
            if (usesAsh && original != null) validatePrecision(original);
            for (var yaml : settings) {
                if (yaml != null) validatePrecision(yaml);
                String mode = original != null ? original : yaml == null ? "auto" : yaml;
                validatePrecision(mode);
                if (selected != null && !selected.equals(mode))
                    throw new IllegalArgumentException("Different ASH queryProjection settings require separate BenchYAML JVMs");
                selected = mode;
                explicitYaml |= yaml != null;
            }
            if (original == null && explicitYaml && selected != null) {
                System.setProperty(ASH_QUERY_PROJECTION_PROPERTY, selected);
                applied = true;
            }
        }

        private static void collectProjectionSettings(CommonParameters section, List<String> settings) {
            if (section == null || section.compression == null) return;
            for (var entry : section.compression) {
                if (entry != null && "ASH".equals(entry.type) && entry.parameters != null) {
                    String value = entry.parameters.get("queryProjection");
                    if (value != null) settings.add(value);
                }
            }
        }

        private static boolean hasAsh(CommonParameters parameters) {
            return parameters != null && parameters.compression != null
                    && parameters.compression.stream().anyMatch(c -> c != null && "ASH".equals(c.type));
        }

        static void validatePrecision(String value) {
            if (!List.of("auto", "fp32", "fp16", "int8").contains(value))
                throw new IllegalArgumentException("queryProjection must be auto, fp32, fp16 or int8: " + value);
        }

        @Override public void close() {
            if (applied) {
                System.clearProperty(ASH_QUERY_PROJECTION_PROPERTY);
                applied = false;
            }
        }
    }

    public Map<Integer, List<Double>> topKOverquery;
    public List<Boolean> useSearchPruning;
    /**
     * Schema 3: a sweep of search worker counts, each an integer or max. Direct search uses this many
     * calling threads; queued ASH search uses this many workers to prepare and execute queries.
     */
    public List<Object> workers;
    /** Schema 3: direct (default) or queued, for every compression in this search. */
    public String queryExecution;
    /**
     * Schema 3, queued only: queries admitted per worker, counting the one it is searching. The queue holds this
     * times the worker count, never less than one batch. Defaults to the larger of 8 and the batch size.
     */
    public Integer requestsPerWorker;

    /** direct unless the YAML selects queued. */
    public String queryExecution() {
        return queryExecution == null ? "direct" : queryExecution;
    }

    /** One resolved count per sweep value, or null when the YAML leaves it to the platform default. */
    public List<Integer> resolvedWorkers() {
        if (workers == null) return null;
        if (workers.isEmpty()) throw new IllegalArgumentException("search.workers must list at least one worker count");
        var counts = new ArrayList<Integer>();
        for (var value : workers) counts.add(Workers.resolve(value, "search.workers"));
        return counts;
    }

    void validateSchema3() {
        if (queryProjection != null) {
            throw new IllegalArgumentException("place queryProjection in the ASH compression parameters in schema 3");
        }
        validateCompression("search", true);
        resolvedWorkers();
        if (!List.of("direct", "queued").contains(queryExecution())) {
            throw new IllegalArgumentException("search.queryExecution must be direct or queued");
        }
        if (requestsPerWorker != null && !"queued".equals(queryExecution())) {
            throw new IllegalArgumentException("search.requestsPerWorker requires queued queryExecution");
        }
        if (requestsPerWorker != null && requestsPerWorker <= 0) {
            throw new IllegalArgumentException("search.requestsPerWorker must be positive");
        }
        if (!"queued".equals(queryExecution()) && compression != null
                && compression.stream().anyMatch(entry -> entry != null && entry.hasMaxBatchSize())) {
            throw new IllegalArgumentException("ASH maxBatchSize requires search.queryExecution: queued");
        }
        if (topKOverquery == null || topKOverquery.isEmpty() || useSearchPruning == null || useSearchPruning.isEmpty()
                || useSearchPruning.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException("search.topKOverquery and search.useSearchPruning must be set");
        }
        for (var entry : topKOverquery.entrySet()) {
            var k = entry.getKey();
            if (k == null || k <= 0 || entry.getValue() == null || entry.getValue().isEmpty()) {
                throw new IllegalArgumentException("search.topKOverquery requires positive K and a list of multipliers");
            }
            for (var multiplier : entry.getValue()) {
                if (multiplier == null || !Double.isFinite(multiplier) || multiplier < 1 || multiplier * k > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException("search.topKOverquery multipliers must be finite, at least 1, and fit an integer candidate budget");
                }
            }
        }
    }

    // NOTE: benchmark compute + console/logging selection are now run-level (run-config.yml)
    // and are no longer recognized in dataset configs.
}
