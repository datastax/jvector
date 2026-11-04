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
import java.util.Map;

/**
 * Turns run-config.yml's {@code benchmarks} section into IVF benchmarks, the way the graph
 * {@code Grid} turns it into its own. Throughput is always measured both batched and per-query.
 * {@code count} has no IVF meaning and is skipped with a warning.
 */
final class IvfBenchmarks {
    private IvfBenchmarks() {}

    static List<IvfQueryBenchmark> fromSpec(Map<String, List<String>> spec) {
        if (spec == null || spec.isEmpty()) {
            spec = SearchReportingCatalog.defaultComputeBenchmarks();
        }
        List<IvfQueryBenchmark> out = new ArrayList<>();
        // Fixed order regardless of how the YAML lists them, as in Grid
        if (spec.containsKey("throughput")) {
            for (Mode mode : Mode.values()) {
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
