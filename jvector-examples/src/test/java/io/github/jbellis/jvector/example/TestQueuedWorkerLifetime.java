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

import io.github.jbellis.jvector.example.benchmarks.datasets.SimpleDataSet;
import io.github.jbellis.jvector.util.BatchingExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import org.junit.Test;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import static org.junit.Assert.*;

/** Warmup and later benchmark phases must use the same workers and thread-owned state. */
public class TestQueuedWorkerLifetime {
    @Test(timeout=20000)
    public void workerStateSurvivesClosingAnExecutionPhase() throws Exception {
        var vector = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(new float[]{1});
        var data = new SimpleDataSet("worker-lifetime", VectorSimilarityFunction.DOT_PRODUCT,
                List.of(vector), List.of(vector), List.of(List.of(0)));
        var system = new Grid.ConfiguredSystem(data, null, null, Set.of());
        var pool = system.queryWorkers();
        var scratch = new ThreadLocal<Integer>();
        int count = system.queryWorkerCount();
        try {
            for (int phase = 0; phase < 2; phase++) {
                int expected = phase;
                var entered = new CountDownLatch(count);
                var completed = new CountDownLatch(count);
                var failure = new AtomicReference<Throwable>();
                try (var queue = new BatchingExecutor<Integer>(count, count, 1, pool,
                        new BatchingExecutor.Operation<Integer>() {
                    public Object batchKey(Integer query) { return null; }
                    public void prepareBatch(List<Integer> queries) {}
                    public void execute(Integer query) throws Exception {
                        entered.countDown();
                        if (!entered.await(5, TimeUnit.SECONDS)) throw new IllegalStateException("Workers did not all start");
                        if (expected == 0) scratch.set(1);
                        else assertEquals(Integer.valueOf(1), scratch.get());
                    }
                    public void complete(Integer query, Throwable error) {
                        if (error != null) failure.compareAndSet(null, error);
                        completed.countDown();
                    }
                })) {
                    for (int i = 0; i < count; i++) assertTrue(queue.trySubmit(i));
                    assertTrue(completed.await(10, TimeUnit.SECONDS));
                    assertNull(failure.get());
                }
                assertSame(pool, system.queryWorkers());
                assertFalse(pool.isShutdown());
            }
        } finally { system.close(); }
        assertTrue(pool.isTerminated());
        assertThrows(IllegalStateException.class, system::queryWorkers);
    }
}
