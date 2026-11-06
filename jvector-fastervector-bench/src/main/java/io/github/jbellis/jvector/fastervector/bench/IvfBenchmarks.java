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

import io.github.jbellis.jvector.example.reporting.SearchReportingCatalog;
import io.github.jbellis.jvector.fastervector.bench.IvfThroughputBenchmark.Mode;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Turns run-config.yml's {@code benchmarks} section into IVF benchmarks, the way the graph
 * {@code Grid} turns it into its own. Throughput is measured batched, coalesced and, unless
 * {@code -Divf.perQuery=false}, per-query. {@code count} has no IVF meaning and is skipped with a
 * warning.
 */
final class IvfBenchmarks {
    static final String PER_QUERY_PROPERTY = "ivf.perQuery";

    private IvfBenchmarks() {}

    /**
     * Whether to measure direct per-query throughput: {@code -Divf.perQuery=false} turns it off. At
     * the pinned fastervector commit, concurrent single-query searchers share one scan pool and one
     * set of I/O rings, so per-query QPS barely scales with threads and takes most of a run's time.
     * Coalesced throughput, which answers the same single queries through
     * {@code IvfAshBatchingSearcher}, is always measured.
     */
    static boolean perQueryEnabled() {
        String value = System.getProperty(PER_QUERY_PROPERTY, "true").trim().toLowerCase(Locale.ROOT);
        return switch (value) {
            case "true" -> true;
            case "false" -> false;
            default -> throw new IllegalArgumentException(PER_QUERY_PROPERTY + " must be true or false, got " + value);
        };
    }

    /** The throughput modes to measure. */
    static List<Mode> throughputModes(boolean perQuery) {
        if (!perQuery) {
            System.out.println("Per-query throughput off (-D" + PER_QUERY_PROPERTY + "=false)");
            return List.of(Mode.BATCHED, Mode.COALESCED);
        }
        return List.of(Mode.values());
    }

    static List<IvfQueryBenchmark> fromSpec(Map<String, List<String>> spec) {
        return fromSpec(spec, perQueryEnabled());
    }

    static List<IvfQueryBenchmark> fromSpec(Map<String, List<String>> spec, boolean perQuery) {
        if (spec == null || spec.isEmpty()) {
            spec = SearchReportingCatalog.defaultComputeBenchmarks();
        }
        List<IvfQueryBenchmark> out = new ArrayList<>();
        // Fixed order regardless of how the YAML lists them, as in Grid
        if (spec.containsKey("throughput")) {
            for (Mode mode : throughputModes(perQuery)) {
                var bench = IvfThroughputBenchmark.createEmpty(mode, 3, 3);
                for (String stat : spec.get("throughput")) {
                    switch (stat) {
                        case "AVG" -> bench.displayAvgQps();
                        case "MEDIAN" -> bench.displayMedianQps();
                        case "MAX" -> bench.displayMaxQps();
                        default -> throw new IllegalArgumentException("Unknown throughput stat: " + stat);
                    }
                }
                out.add(bench);
            }
        }
        if (spec.containsKey("latency")) {
            var bench = IvfLatencyBenchmark.createEmpty(2);
            for (String stat : spec.get("latency")) {
                switch (stat) {
                    case "AVG" -> bench.displayAvgLatency();
                    case "STD" -> bench.displayLatencySTD();
                    case "P999" -> bench.displayP999Latency();
                    default -> throw new IllegalArgumentException("Unknown latency stat: " + stat);
                }
            }
            out.add(bench);
        }
        if (spec.containsKey("count")) {
            System.err.println("WARNING: benchmarks.count (graph node visits) has no IVF equivalent; skipping it");
        }
        if (spec.containsKey("accuracy")) {
            var bench = IvfAccuracyBenchmark.createEmpty();
            for (String stat : spec.get("accuracy")) {
                switch (stat) {
                    case "recall" -> bench.displayRecall();
                    case "MAP" -> bench.displayMAP();
                    default -> throw new IllegalArgumentException("Unknown accuracy stat: " + stat);
                }
            }
            out.add(bench);
        }
        return out;
    }
}
