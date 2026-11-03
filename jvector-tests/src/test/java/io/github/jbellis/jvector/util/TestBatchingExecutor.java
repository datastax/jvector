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
package io.github.jbellis.jvector.util;

import org.junit.Test;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import static org.junit.Assert.*;

/** Tests real queued submission, bounded admission, failures, handoff and close. */
public class TestBatchingExecutor {
    private abstract static class IntOperation implements BatchingExecutor.Operation<Integer> {
        public Object batchKey(Integer value) { return this; }
        public void prepareBatch(List<Integer> values) throws Exception {}
        public void execute(Integer value) throws Exception {}
    }

    @Test(timeout=15000)
    public void oneProducerAndOneWorkerPrepareEightWithoutEightCallerThreads() throws Exception {
        var executor = Executors.newSingleThreadExecutor();
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var done = new CountDownLatch(9);
        var maximumBatch = new AtomicInteger();
        var failure = new AtomicReference<Throwable>();
        try (var service = new BatchingExecutor<Integer>(1, 9, 8, executor, new IntOperation() {
            public void prepareBatch(List<Integer> values) {
                maximumBatch.accumulateAndGet(values.size(), Math::max);
                assertTrue(values.size() <= 8);
            }
            public void execute(Integer value) throws Exception {
                if (value == 0) { entered.countDown(); release.await(); }
            }
            public void complete(Integer value, Throwable error) {
                if (error != null) failure.compareAndSet(null, error);
                done.countDown();
            }
        })) {
            assertTrue(service.trySubmit(0));
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            for (int i = 1; i < 9; i++) assertTrue(service.trySubmit(i));
            assertEquals(9, service.outstanding());
            assertFalse(service.trySubmit(10));
            release.countDown();
            assertTrue(done.await(5, TimeUnit.SECONDS));
            assertNull(failure.get());
            assertEquals(8, maximumBatch.get());
            assertEquals(1, service.batchHistogram()[1]);
            assertEquals(1, service.batchHistogram()[8]);
        } finally { release.countDown(); executor.shutdownNow(); }
    }

    @Test(timeout=15000)
    public void drainingABatchDoesNotAccumulateDuplicateWakeups() throws Exception {
        var executor = Executors.newSingleThreadExecutor();
        var firstEntered = new CountDownLatch(1);
        var firstRelease = new CountDownLatch(1);
        var batchEntered = new CountDownLatch(1);
        var batchRelease = new CountDownLatch(1);
        var done = new CountDownLatch(9);
        var failure = new AtomicReference<Throwable>();
        try (var service = new BatchingExecutor<Integer>(1, 9, 8, executor, new IntOperation() {
            public void execute(Integer value) throws Exception {
                if (value == 0) { firstEntered.countDown(); firstRelease.await(); }
                if (value == 1) { batchEntered.countDown(); batchRelease.await(); }
            }
            public void complete(Integer value, Throwable error) {
                if (error != null) failure.compareAndSet(null, error);
                done.countDown();
            }
        })) {
            try {
                assertTrue(service.trySubmit(0));
                assertTrue(firstEntered.await(5, TimeUnit.SECONDS));
                for (int i = 1; i < 9; i++) assertTrue(service.trySubmit(i));
                firstRelease.countDown();
                assertTrue(batchEntered.await(5, TimeUnit.SECONDS));
                // Seven prepared requests remain runnable. Counting their original
                // pending signals again would grow notifications under continuous load.
                var field = BatchingExecutor.class.getDeclaredField("work");
                field.setAccessible(true);
                assertEquals(7, ((Semaphore) field.get(service)).availablePermits());
                batchRelease.countDown();
                assertTrue(done.await(5, TimeUnit.SECONDS));
                assertNull(failure.get());
            } finally { firstRelease.countDown(); batchRelease.countDown(); }
        } finally { firstRelease.countDown(); batchRelease.countDown(); executor.shutdownNow(); }
    }

    @Test(timeout=15000)
    public void isolatedRequestsStartImmediatelyAtCapacityEightAndRecordsAreReused() throws Exception {
        var executor = Executors.newFixedThreadPool(2);
        var completed = new ArrayBlockingQueue<Integer>(1);
        var completedCount = new AtomicInteger();
        try (var service = new BatchingExecutor<Integer>(2, 8, 8, executor, new IntOperation() {
            public void complete(Integer value, Throwable error) {
                assertNull(error); completedCount.incrementAndGet(); completed.add(value);
            }
        })) {
            for (int i = 0; i < 100; i++) {
                assertTrue(service.trySubmit(i));
                assertEquals(Integer.valueOf(i), completed.poll(5, TimeUnit.SECONDS));
            }
            assertEquals(100, completedCount.get());
            assertEquals(100, service.batchHistogram()[1]);
        } finally { executor.shutdownNow(); }
    }

    @Test(timeout=15000)
    public void prepareAndExecutionFailuresCompleteRequestsAndCloseDrainsAcceptedWork() throws Exception {
        var executor = Executors.newFixedThreadPool(3);
        var counts = new AtomicIntegerArray(64);
        var errors = new AtomicInteger();
        var service = new BatchingExecutor<Integer>(3, 64, 8, executor, new IntOperation() {
            public Object batchKey(Integer value) { return null; }
            public void prepareBatch(List<Integer> values) {
                if (values.get(0) == 7) throw new IllegalArgumentException("prepare failure");
            }
            public void execute(Integer value) {
                if (value == 31) throw new IllegalStateException("search failure");
            }
            public void complete(Integer value, Throwable error) {
                counts.incrementAndGet(value);
                if (error != null) errors.incrementAndGet();
            }
        });
        try {
            for (int i = 0; i < 64; i++) assertTrue(service.trySubmit(i));
            service.close();
            assertEquals(0, service.outstanding());
            for (int i = 0; i < 64; i++) assertEquals(1, counts.get(i));
            assertEquals(2, errors.get());
            assertThrows(RejectedExecutionException.class, () -> service.trySubmit(0));
            service.close();
        } finally { executor.shutdownNow(); }
    }

    @Test(timeout=15000)
    public void differentModelsNeverSharePreparationAndPreparedWorkCanChangeWorkers() throws Exception {
        var executor = Executors.newFixedThreadPool(4);
        Object[] keys = {new Object(), new Object()};
        var done = new CountDownLatch(128);
        var failure = new AtomicReference<Throwable>();
        var preparedThread = new ConcurrentHashMap<Integer, Thread>();
        var changedWorker = new AtomicBoolean();
        try (var service = new BatchingExecutor<Integer>(4, 128, 8, executor, new IntOperation() {
            public Object batchKey(Integer value) { return keys[(value / 16) & 1]; }
            public void prepareBatch(List<Integer> values) {
                Object key = batchKey(values.get(0));
                for (Integer value : values) {
                    assertSame(key, batchKey(value));
                    preparedThread.put(value, Thread.currentThread());
                }
            }
            public void execute(Integer value) {
                if (preparedThread.get(value) != Thread.currentThread()) changedWorker.set(true);
            }
            public void complete(Integer value, Throwable error) {
                if (error != null) failure.compareAndSet(null, error);
                done.countDown();
            }
        })) {
            for (int i = 0; i < 128; i++) assertTrue(service.trySubmit(i));
            assertTrue(done.await(5, TimeUnit.SECONDS));
            assertNull(failure.get());
            assertEquals(128, preparedThread.size());
            // Handoff is allowed rather than mandatory; scheduling determines the owner.
        } finally { executor.shutdownNow(); }
    }
}
