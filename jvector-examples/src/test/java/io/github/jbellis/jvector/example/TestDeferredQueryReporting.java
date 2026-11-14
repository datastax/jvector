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

import io.github.jbellis.jvector.example.benchmarks.Metric;
import io.github.jbellis.jvector.example.benchmarks.QueryBenchmark;
import io.github.jbellis.jvector.example.benchmarks.QueryTester;
import io.github.jbellis.jvector.example.benchmarks.QueuedQueryExecution;
import io.github.jbellis.jvector.example.benchmarks.datasets.SimpleDataSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Supplier;

import static org.junit.Assert.*;

/** Reporting belongs to one measurement invocation and owns completed samples. */
public class TestDeferredQueryReporting {
    private Grid.ConfiguredSystem system() {
        var vector = VectorizationProvider.getInstance().getVectorTypeSupport()
                .createFloatVector(new float[]{1});
        var data = new SimpleDataSet("deferred-report", VectorSimilarityFunction.DOT_PRODUCT,
                List.of(vector), List.of(vector), List.of(List.of(0)));
        return new Grid.ConfiguredSystem(data, null, null, Set.of()) {
            @Override public int queryWorkerCount() { return 1; }
        };
    }

    private QueryBenchmark probe(String name,
            BiFunction<Grid.ConfiguredSystem, Consumer<Supplier<List<Metric>>>, List<Metric>> operation) {
        return new QueryBenchmark() {
            public String getBenchmarkName() { return name; }

            public List<Metric> runBenchmark(Grid.ConfiguredSystem cs, int k, int r,
                    boolean pruning, int runs) {
                var reports = new ArrayList<Supplier<List<Metric>>>();
                var result = new ArrayList<>(operation.apply(cs, reports::add));
                for (var report : reports) result.addAll(report.get());
                return result;
            }

            public List<Metric> runBenchmark(Grid.ConfiguredSystem cs, int k, int r,
                    boolean pruning, int runs, Consumer<Supplier<List<Metric>>> reports) {
                return operation.apply(cs, reports);
            }
        };
    }

    private static Metric report() {
        return Metric.of("test.report", "report", ".1f", 7);
    }

    @Test
    public void reportingRunsAfterEveryMeasurementAndRetainsItsValues() throws Exception {
        var order = new ArrayList<String>();
        try (var cs = system()) {
            var producer = probe("throughput", (c, reports) -> {
                order.add("throughput");
                reports.accept(() -> { order.add("report"); return List.of(report()); });
                return List.of();
            });
            var latency = probe("latency", (c, reports) -> {
                assertFalse(order.contains("report")); order.add("latency"); return List.of();
            });
            var accuracy = probe("accuracy", (c, reports) -> {
                assertFalse(order.contains("report")); order.add("accuracy"); return List.of();
            });
            var metrics = new QueryTester(List.of(producer, latency, accuracy)).run(cs, 1, 1, false, 1);
            assertEquals(List.of("throughput", "latency", "accuracy", "report"), order);
            assertEquals(1, metrics.stream().filter(m -> m.getKey().equals("test.report")).count());
            assertEquals(7, metrics.stream().filter(m -> m.getKey().equals("test.report"))
                    .findFirst().orElseThrow().getValue(), 0);
            assertTrue(new QueryTester(List.of()).run(cs, 1, 1, false, 1).stream()
                    .noneMatch(m -> m.getKey().equals("test.report")));
        }
    }

    @Test
    public void failedMeasurementCannotLeakReportingIntoTheNextInvocation() throws Exception {
        var order = new ArrayList<String>();
        try (var cs = system()) {
            var producer = probe("throughput", (c, reports) -> {
                reports.accept(() -> { order.add("report"); return List.of(report()); });
                return List.of();
            });
            var failure = probe("latency", (c, reports) -> {
                throw new IllegalStateException("measurement failed");
            });
            assertThrows(IllegalStateException.class,
                    () -> new QueryTester(List.of(producer, failure)).run(cs, 1, 1, false, 1));
            var later = new QueryTester(List.of()).run(cs, 1, 1, false, 1);
            assertTrue(order.isEmpty());
            assertTrue(later.stream().noneMatch(m -> m.getKey().equals("test.report")));
        }
    }

    @Test
    public void standaloneReportingDoesNotAttachStateToTheConfiguredSystem() throws Exception {
        try (var cs = system()) {
            var benchmark = probe("standalone", (c, reports) -> {
                reports.accept(() -> List.of(report()));
                return List.of();
            });
            assertEquals(1, benchmark.runBenchmark(cs, 1, 1, false, 1).size());
            assertTrue(new QueryTester(List.of()).run(cs, 1, 1, false, 1).stream()
                    .noneMatch(m -> m.getKey().equals("test.report")));
        }
    }

    @Test
    public void legacyBenchmarkOverloadStillRunsInQueryTester() throws Exception {
        try (var cs = system()) {
            var benchmark = new QueryBenchmark() {
                public String getBenchmarkName() { return "legacy"; }
                public List<Metric> runBenchmark(Grid.ConfiguredSystem system, int k, int r,
                        boolean pruning, int runs) { return List.of(report()); }
            };
            assertTrue(new QueryTester(List.of(benchmark)).run(cs, 1, 1, false, 1).stream()
                    .anyMatch(m -> m.getKey().equals("test.report")));
        }
    }

    @Test(timeout=10000)
    public void rejectedSubmissionReturnsItsPooledRequestAndInputGuardsPrecedeAllocation() throws Exception {
        try (var cs = system()) {
            assertThrows(IllegalArgumentException.class, () -> new QueuedQueryExecution(cs, 0, 1, false));
            assertThrows(IllegalArgumentException.class, () -> new QueuedQueryExecution(cs, 2, 1, false));
            var queued = new QueuedQueryExecution(cs, 1, 1, false);
            assertThrows(IllegalArgumentException.class, () -> queued.runPass(-1, i -> null, null, false));
            assertThrows(NullPointerException.class, () -> queued.runPass(0, null, null, false));
            assertThrows(NullPointerException.class, () -> queued.executeSingle(null));
            queued.close();
            var query = cs.getDataSet().getQueryVectors().get(0);
            for (int i = 0; i < 100; i++)
                assertThrows(java.util.concurrent.RejectedExecutionException.class, () -> queued.executeSingle(query));
        }
    }

    @Test
    public void completedSnapshotSurvivesExecutorCloseAndCanOnlyBeTransferredOnce() throws Exception {
        try (var cs = system()) {
            QueuedQueryExecution.Measurements samples;
            try (var queued = new QueuedQueryExecution(cs, 1, 1, false)) {
                assertThrows(IllegalStateException.class, queued::finishMeasurements);
                queued.beginMeasurements();
                assertThrows(IllegalStateException.class, queued::beginMeasurements);
                queued.runPass(0, i -> { throw new AssertionError("Empty pass fetched a query"); }, null, false);
                samples = queued.finishMeasurements();
                assertThrows(IllegalStateException.class, queued::finishMeasurements);
                assertThrows(IllegalStateException.class, queued::beginMeasurements);
            }
            var first = samples.metrics();
            var second = samples.metrics();
            // Mean and p50, p95, p99 and p99.9 response times, then the mean batch size.
            assertEquals(6, first.size());
            for (int i = 0; i < first.size(); i++) {
                assertEquals(0, first.get(i).getValue(), 0);
                assertEquals(first.get(i).getValue(), second.get(i).getValue(), 0);
            }
        }
    }
}
