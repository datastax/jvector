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

package io.github.jbellis.jvector.example.benchmarks;

import java.util.Arrays;
import java.util.List;

/**
 * Per-query time distribution of the throughput benchmark's timed passes: the time each queued query waits and
 * runs, or each direct search's own time. Reported under {@code metrics.execution}.
 */
final class ResponseTimeMetrics {
    private ResponseTimeMetrics() {}

    /** Mean and percentiles of {@code nanos}, which is sorted in place. */
    static List<Metric> of(long[] nanos) {
        Arrays.sort(nanos);
        double sum = 0;
        for (long time : nanos) sum += time;
        return List.of(
                Metric.of("search.execution.response_mean_ms", "Loaded mean (ms)", ".3f", nanos.length == 0 ? 0 : sum / nanos.length / 1e6),
                Metric.of("search.execution.response_p50_ms", "Loaded p50 (ms)", ".3f", percentile(nanos, .50)),
                Metric.of("search.execution.response_p95_ms", "Loaded p95 (ms)", ".3f", percentile(nanos, .95)),
                Metric.of("search.execution.response_p99_ms", "Loaded p99 (ms)", ".3f", percentile(nanos, .99)),
                Metric.of("search.execution.response_p999_ms", "Loaded p99.9 (ms)", ".3f", percentile(nanos, .999)));
    }

    private static double percentile(long[] sorted, double p) {
        return sorted.length == 0 ? 0 : sorted[Math.min(sorted.length - 1, (int) Math.ceil(sorted.length * p) - 1)] / 1e6;
    }
}
