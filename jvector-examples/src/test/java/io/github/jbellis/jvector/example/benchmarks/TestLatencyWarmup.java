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

import org.junit.Test;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.HashSet;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import static org.junit.Assert.*;

/** Exercises the actual warmup coverage, input ownership, timing and failure contract. */
public class TestLatencyWarmup {
    @Test
    public void completePassesMeetTheDurationFloorAndReuseOnlyOneRandomInputBuffer() {
        var clock = new AtomicLong();
        var identities = Collections.newSetFromMap(new IdentityHashMap<Object, Boolean>());
        var samples = new HashSet<Float>();
        long completed = LatencyBenchmark.warmup(8, 3, 2, 10, query -> {
            identities.add(query);
            samples.add(query.get(0));
            double norm = 0;
            for (int d = 0; d < 8; d++) norm += query.get(d) * query.get(d);
            assertEquals(1.0, norm, 1e-5);
            clock.addAndGet(2);
        }, clock::get);
        assertEquals(12, completed); // Two complete 3-query sweeps per timed pass.
        assertEquals(24, clock.get());
        assertEquals(1, identities.size());
        assertTrue(samples.size() > 1);
    }

    @Test
    public void minimumQueryCoverageStillAppliesWithNoTimeFloor() {
        var calls = new AtomicInteger();
        assertEquals(21, LatencyBenchmark.warmup(8, 7, 3, 0,
                query -> calls.incrementAndGet(), () -> 0));
        assertEquals(21, calls.get());
    }

    @Test
    public void failedWarmupPropagatesInsteadOfProducingLatencyMetrics() {
        var error = new IllegalStateException("search failed");
        assertSame(error, assertThrows(IllegalStateException.class,
                () -> LatencyBenchmark.warmup(8, 1, 1, 0, query -> { throw error; }, () -> 0)));
    }

    @Test
    public void invalidInputsFailBeforeWarmupWork() {
        var calls = new AtomicInteger();
        for (int[] args : new int[][]{{0,1,1,0}, {8,0,1,0}, {8,1,0,0}, {8,1,1,-1}}) {
            assertThrows(IllegalArgumentException.class, () -> LatencyBenchmark.warmup(
                    args[0], args[1], args[2], args[3], query -> calls.incrementAndGet(), () -> 0));
        }
        assertEquals(0, calls.get());
        assertThrows(NullPointerException.class, () -> LatencyBenchmark.warmup(8,1,1,0,null,()->0));
        assertThrows(NullPointerException.class, () -> LatencyBenchmark.warmup(8,1,1,0,query->{},null));
    }
}
