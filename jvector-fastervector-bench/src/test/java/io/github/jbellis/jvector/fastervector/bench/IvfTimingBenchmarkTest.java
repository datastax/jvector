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
import io.github.jbellis.jvector.fastervector.bench.IvfThroughputBenchmark.Mode;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/** The statistics and metric keys of the timing benchmarks, without the native libraries. */
class IvfTimingBenchmarkTest {

    @Test
    void batchedThroughputUsesTheGraphKeys() {
        var m = byKey(IvfThroughputBenchmark.createEmpty(Mode.BATCHED, 0, 3)
                .displayAvgQps().displayMedianQps().displayMaxQps()
                .metrics(new double[]{300, 100, 200}));
        assertEquals(200, m.get("search.throughput.avg_qps"), 1e-9);
        assertEquals(100, m.get("search.throughput.stddev_qps"), 1e-9);   // sample std dev
        assertEquals(50, m.get("search.throughput.cv_pct"), 1e-9);
        assertEquals(200, m.get("search.throughput.median_qps"), 1e-9);
        assertEquals(300, m.get("search.throughput.max_qps"), 1e-9);
        assertEquals(100, m.get("search.throughput.min_qps"), 1e-9);
    }

    @Test
    void perQueryThroughputHasItsOwnKeysAndHeaders() {
        List<Metric> metrics = IvfThroughputBenchmark.createDefault(Mode.PER_QUERY).metrics(new double[]{10, 10, 10});
        var m = byKey(metrics);
        assertEquals(10, m.get("search.throughput.per_query.avg_qps"), 1e-9);
        assertEquals(0, m.get("search.throughput.per_query.stddev_qps"), 1e-9);
        assertFalse(m.containsKey("search.throughput.avg_qps"));
        assertTrue(metrics.get(0).getHeader().contains("per-query"));
    }

    @Test
    void singleRunHasZeroStdDev() {
        var m = byKey(IvfThroughputBenchmark.createDefault(Mode.BATCHED).metrics(new double[]{42}));
        assertEquals(42, m.get("search.throughput.avg_qps"), 1e-9);
        assertEquals(0, m.get("search.throughput.stddev_qps"), 1e-9);
    }

    @Test
    void latencyStatisticsMatchTheGraphBenchmark() {
        long[] nanos = new long[1000];
        for (int i = 0; i < nanos.length; i++) {
            nanos[i] = (i + 1) * 1_000_000L;   // 1..1000 ms
        }
        var m = byKey(IvfLatencyBenchmark.createEmpty(1)
                .displayAvgLatency().displayLatencySTD().displayP999Latency()
                .metrics(nanos));
        assertEquals(500.5, m.get("search.latency.mean_ms"), 1e-9);
        assertEquals(Math.sqrt((1000.0 * 1000 - 1) / 12), m.get("search.latency.std_ms"), 1e-6);
        assertEquals(999, m.get("search.latency.p999_ms"), 1e-9);
    }

    @Test
    void warmupQueriesAreDeterministicUnitVectors() {
        float[][] a = IvfThroughputBenchmark.randomUnitVectors(5, 16, 42);
        float[][] b = IvfThroughputBenchmark.randomUnitVectors(5, 16, 42);
        assertArrayEquals(a, b);
        for (float[] v : a) {
            double norm = 0;
            for (float x : v) {
                norm += x * x;
            }
            assertEquals(1.0, norm, 1e-5);
        }
    }

    @Test
    void benchmarksNeedSomethingToReport() {
        assertThrows(IllegalStateException.class,
                     () -> IvfThroughputBenchmark.createEmpty(Mode.BATCHED, 0, 1).runBenchmark(null, null));
        assertThrows(IllegalStateException.class,
                     () -> IvfLatencyBenchmark.createEmpty(1).runBenchmark(null, null));
        assertThrows(IllegalArgumentException.class, () -> IvfThroughputBenchmark.createEmpty(Mode.BATCHED, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> IvfLatencyBenchmark.createEmpty(0));
    }

    private static Map<String, Double> byKey(List<Metric> metrics) {
        return metrics.stream().collect(Collectors.toMap(Metric::getKey, Metric::getValue));
    }
}
