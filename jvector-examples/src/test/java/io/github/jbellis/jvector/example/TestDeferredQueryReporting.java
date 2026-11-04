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
package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.example.benchmarks.*;
import io.github.jbellis.jvector.example.benchmarks.datasets.SimpleDataSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import org.junit.Test;
import java.util.*;
import java.util.function.Function;
import static org.junit.Assert.*;

public class TestDeferredQueryReporting {
    private Grid.ConfiguredSystem system() {
        var v = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(new float[]{1});
        return new Grid.ConfiguredSystem(new SimpleDataSet("deferred-report", VectorSimilarityFunction.DOT_PRODUCT,
                List.of(v), List.of(v), List.of(List.of(0))), null, null, Set.of());
    }
    private QueryBenchmark probe(String name, Function<Grid.ConfiguredSystem,List<Metric>> operation) {
        return new QueryBenchmark() {
            public String getBenchmarkName() { return name; }
            public List<Metric> runBenchmark(Grid.ConfiguredSystem cs, int k, int r, boolean p, int runs) {
                return operation.apply(cs);
            }
        };
    }
    @Test
    public void reportingRunsAfterEveryMeasurementAndRetainsItsValues() throws Exception {
        var order = new ArrayList<String>();
        try (var cs = system()) {
            var producer = probe("throughput", c -> {
                order.add("throughput");
                c.deferQueryMetrics(() -> { order.add("report"); return List.of(Metric.of("test.report", "report", ".1f", 7)); });
                return List.of();
            });
            var latency = probe("latency", c -> { assertFalse(order.contains("report")); order.add("latency"); return List.of(); });
            var accuracy = probe("accuracy", c -> { assertFalse(order.contains("report")); order.add("accuracy"); return List.of(); });
            var metrics = new QueryTester(List.of(producer, latency, accuracy)).run(cs, 1, 1, false, 1);
            assertEquals(List.of("throughput", "latency", "accuracy", "report"), order);
            assertEquals(7, metrics.stream().filter(m -> m.getKey().equals("test.report")).findFirst().orElseThrow().getValue(), 0);
            var later = new ArrayList<Metric>(); cs.appendDeferredQueryMetrics(later); assertTrue(later.isEmpty());
        }
    }
    @Test
    public void failedMeasurementDiscardsPendingReporting() throws Exception {
        var order = new ArrayList<String>();
        try (var cs = system()) {
            var producer = probe("throughput", c -> { c.deferQueryMetrics(() -> { order.add("report"); return List.of(); }); return List.of(); });
            var failure = probe("latency", c -> { throw new IllegalStateException("measurement failed"); });
            assertThrows(IllegalStateException.class, () -> new QueryTester(List.of(producer, failure)).run(cs, 1, 1, false, 1));
            var later = new ArrayList<Metric>(); cs.appendDeferredQueryMetrics(later);
            assertTrue(order.isEmpty()); assertTrue(later.isEmpty());
        }
    }
}
