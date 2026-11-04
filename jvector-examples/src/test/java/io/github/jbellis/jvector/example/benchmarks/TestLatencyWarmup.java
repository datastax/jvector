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

import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.HashSet;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import static org.junit.Assert.*;

/** Checks the shared pass, random input ownership, timing floor and metric isolation. */
public class TestLatencyWarmup {
    private static final SearchResult RESULT = new SearchResult(new SearchResult.NodeScore[0], 1, 0, 0, 0, 0);

    private VectorFloat<?>[] queries(int count) {
        var result = new VectorFloat<?>[count];
        for (int i = 0; i < count; i++)
            result[i] = VectorizationProvider.getInstance().getVectorTypeSupport()
                    .createFloatVector(new float[]{1,0,0,0,0,0,0,0});
        return result;
    }

    @Test
    public void warmupAndMeasurementUseTheSameOperationAndExcludeWarmupSamples() {
        var measured = queries(3);
        var originals = Collections.newSetFromMap(new IdentityHashMap<Object, Boolean>());
        Collections.addAll(originals, measured);
        var warmIdentities = Collections.newSetFromMap(new IdentityHashMap<Object, Boolean>());
        var values = new HashSet<Float>();
        var clock = new AtomicLong();
        var warmCalls = new AtomicInteger();
        var measuredCalls = new AtomicInteger();
        long[] samples = LatencyBenchmark.measure(measured, 2, 2, 10, query -> {
            if (originals.contains(query)) {
                measuredCalls.incrementAndGet();
                assertEquals(12, warmCalls.get());
                assertEquals(1, query.get(0), 0);
                assertEquals(0, query.get(1), 0);
            } else {
                warmCalls.incrementAndGet();
                warmIdentities.add(query);
                values.add(query.get(0));
                double norm = 0;
                for (int d = 0; d < 8; d++) norm += query.get(d) * query.get(d);
                assertEquals(1, norm, 1e-5);
            }
            clock.addAndGet(2);
            return RESULT;
        }, clock::get);
        assertEquals(12, warmCalls.get());
        assertEquals(6, measuredCalls.get());
        assertEquals(6, samples.length);
        assertEquals(3, warmIdentities.size()); // Reused across every warmup sweep.
        assertTrue(values.size() > 3);
        for (long value : samples) assertTrue(value >= 0);
    }

    @Test
    public void minimumQueryCoverageAppliesWithoutATimeFloor() {
        var calls = new AtomicInteger();
        long[] samples = LatencyBenchmark.measure(queries(7), 2, 3, 0, query -> {
            calls.incrementAndGet(); return RESULT;
        }, () -> 0);
        assertEquals(35, calls.get());
        assertEquals(14, samples.length);
    }

    @Test
    public void statisticsUseOnlySuppliedMeasuredSamples() {
        var metrics = LatencyBenchmark.createDefault().displayLatencySTD().displayP999Latency()
                .metrics(new long[]{3_000_000, 1_000_000, 2_000_000});
        assertEquals(2, metrics.stream().filter(m -> m.getKey().equals("search.latency.mean_ms")).findFirst().orElseThrow().getValue(), 1e-9);
        assertEquals(Math.sqrt(2.0 / 3), metrics.stream().filter(m -> m.getKey().equals("search.latency.std_ms")).findFirst().orElseThrow().getValue(), 1e-9);
        assertEquals(3, metrics.stream().filter(m -> m.getKey().equals("search.latency.p999_ms")).findFirst().orElseThrow().getValue(), 1e-9);
    }

    @Test
    public void failedWarmupPropagatesBeforeAnyMeasuredSearch() {
        var error = new IllegalStateException("search failed");
        assertSame(error, assertThrows(IllegalStateException.class,
                () -> LatencyBenchmark.measure(queries(1), 1, 1, 0, query -> { throw error; }, () -> 0)));
    }

    @Test
    public void invalidInputsFailBeforeSearch() {
        var calls = new AtomicInteger();
        for (int[] args : new int[][]{{0,1,1,0}, {1,0,1,0}, {1,1,0,0}, {1,1,1,-1}})
            assertThrows(IllegalArgumentException.class, () -> LatencyBenchmark.measure(
                    queries(args[0]), args[1], args[2], args[3], query -> {calls.incrementAndGet();return RESULT;}, () -> 0));
        assertEquals(0, calls.get());
        assertThrows(NullPointerException.class, () -> LatencyBenchmark.measure(queries(1),1,1,0,null,()->0));
        assertThrows(NullPointerException.class, () -> LatencyBenchmark.measure(queries(1),1,1,0,query->RESULT,null));
    }
}
