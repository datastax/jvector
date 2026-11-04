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
    public void drainingABatchDoesNotRetainIdleSharedExecutorThreads() throws Exception {
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
                batchRelease.countDown();
                assertTrue(done.await(5, TimeUnit.SECONDS));
                assertNull(failure.get());
                assertEquals(Integer.valueOf(42), executor.submit(() -> 42).get(5, TimeUnit.SECONDS));
            } finally { firstRelease.countDown(); batchRelease.countDown(); }
        } finally { firstRelease.countDown(); batchRelease.countDown(); executor.shutdownNow(); }
    }

    @Test(timeout=15000)
    public void sharedExecutorNeedsNoPermanentlyReservedThreads() throws Exception {
        var executor = Executors.newSingleThreadExecutor();
        var completed = new CountDownLatch(1);
        try (var service = new BatchingExecutor<Integer>(4, 8, 8, executor, new IntOperation() {
            public void complete(Integer request, Throwable error) { assertNull(error); completed.countDown(); }
        })) {
            assertTrue(service.trySubmit(1));
            assertTrue(completed.await(5, TimeUnit.SECONDS));
            assertEquals(Integer.valueOf(7), executor.submit(() -> 7).get(5, TimeUnit.SECONDS));
        } finally { executor.shutdownNow(); }
    }

    @Test(timeout=15000)
    public void completionFailureIsContainedAndReportedWithoutPoisoningLaterRequests() throws Exception {
        var executor = Executors.newSingleThreadExecutor();
        var error = new IllegalStateException("callback failure");
        var second = new CountDownLatch(1);
        var completed = new AtomicInteger();
        var service = new BatchingExecutor<Integer>(1, 8, 8, executor, new IntOperation() {
            public void complete(Integer request, Throwable failure) {
                assertNull(failure);
                completed.incrementAndGet();
                if (request == 0) throw error;
                second.countDown();
            }
        });
        try {
            assertTrue(service.trySubmit(0));
            assertTrue(service.trySubmit(1));
            assertTrue(second.await(5, TimeUnit.SECONDS));
            assertEquals(2, completed.get());
            assertEquals(1, service.completionFailureCount());
            assertSame(error, service.firstCompletionFailure());
            var failure = assertThrows(IllegalStateException.class, service::close);
            assertSame(error, failure.getCause());
            assertEquals(0, service.outstanding());
        } finally { executor.shutdownNow(); }
    }

    @Test(timeout=15000)
    public void keyFailureCompletionDoesNotHoldInternalLock() throws Exception {
        var executor = Executors.newSingleThreadExecutor();
        var submitter = Executors.newSingleThreadExecutor();
        var done = new CountDownLatch(2);
        var first = new IllegalArgumentException("key failure");
        var errors = new AtomicReference<Throwable>();
        var reference = new AtomicReference<BatchingExecutor<Integer>>();
        var service = new BatchingExecutor<Integer>(1, 8, 8, executor, new IntOperation() {
            public Object batchKey(Integer request) { if (request == 0) throw first; return null; }
            public void complete(Integer request, Throwable failure) {
                try {
                    if (request == 0) {
                        assertSame(first, failure);
                        assertTrue(submitter.submit(() -> reference.get().trySubmit(1)).get(5, TimeUnit.SECONDS));
                    } else assertNull(failure);
                } catch (Throwable error) { errors.compareAndSet(null, error); }
                finally { done.countDown(); }
            }
        });
        reference.set(service);
        try (service) {
            assertTrue(service.trySubmit(0));
            assertTrue(done.await(7, TimeUnit.SECONDS));
            assertNull(errors.get());
        } finally { submitter.shutdownNow(); executor.shutdownNow(); }
    }

    @Test(timeout=15000)
    public void rejectedDispatchCompletesAcceptedWorkAndLaterSubmissionCanSucceed() throws Exception {
        var executor = Executors.newSingleThreadExecutor();
        var launches = new AtomicInteger();
        Executor sometimesRejects = work -> {
            if (launches.getAndIncrement() == 0) throw new RejectedExecutionException("busy host");
            executor.execute(work);
        };
        var failures = new AtomicInteger();
        var completed = new CountDownLatch(2);
        try (var service = new BatchingExecutor<Integer>(1, 8, 8, sometimesRejects, new IntOperation() {
            public void complete(Integer request, Throwable failure) {
                if (failure != null) { assertTrue(failure instanceof RejectedExecutionException); failures.incrementAndGet(); }
                completed.countDown();
            }
        })) {
            assertTrue(service.trySubmit(0));
            assertTrue(service.trySubmit(1));
            assertTrue(completed.await(5, TimeUnit.SECONDS));
            assertEquals(1, failures.get());
            assertEquals(0, service.completionFailureCount());
        } finally { executor.shutdownNow(); }
    }

    @Test(timeout=15000)
    public void directExecutorDoesNotMakeAsynchronousSubmissionExecuteInline() {
        var executions = new AtomicInteger();
        var error = new AtomicReference<Throwable>();
        try (var service = new BatchingExecutor<Integer>(1, 8, 8, Runnable::run, new IntOperation() {
            public void execute(Integer request) { executions.incrementAndGet(); }
            public void complete(Integer request, Throwable failure) { error.set(failure); }
        })) {
            assertTrue(service.trySubmit(0));
            assertEquals(0, executions.get());
            assertTrue(error.get() instanceof RejectedExecutionException);
            assertEquals(0, service.outstanding());
        }
    }

    @Test
    public void knownDiscardAndCallerRunsPoliciesAreRejectedBeforeAdmission() {
        for (var policy : List.of(new ThreadPoolExecutor.DiscardPolicy(),
                new ThreadPoolExecutor.DiscardOldestPolicy(), new ThreadPoolExecutor.CallerRunsPolicy())) {
            var executor = new ThreadPoolExecutor(1, 1, 0, TimeUnit.MILLISECONDS,
                    new ArrayBlockingQueue<Runnable>(1), policy);
            try {
                assertThrows(IllegalArgumentException.class,
                        () -> new BatchingExecutor<Integer>(1, 8, 8, executor, new IntOperation() {
                            public void complete(Integer request, Throwable failure) {}
                        }));
            } finally { executor.shutdownNow(); }
        }
    }

    @Test(timeout=15000)
    public void deferredTaskMayLaterRunOnItsOriginalSubmittingThread() {
        var scheduled = new ArrayBlockingQueue<Runnable>(1);
        var executions = new AtomicInteger();
        var completed = new AtomicInteger();
        try (var service = new BatchingExecutor<Integer>(1, 8, 8, scheduled::add, new IntOperation() {
            public void execute(Integer request) { executions.incrementAndGet(); }
            public void complete(Integer request, Throwable failure) { assertNull(failure); completed.incrementAndGet(); }
        })) {
            assertTrue(service.trySubmit(1));
            assertEquals(0, executions.get());
            scheduled.remove().run();
            assertEquals(1, executions.get());
            assertEquals(1, completed.get());
        }
    }

    @Test(timeout=15000)
    public void incompatibleQueuedRequestsUseTheFullWorkerBudget() throws Exception {
        var executor = Executors.newFixedThreadPool(4);
        var scheduled = new ArrayBlockingQueue<Runnable>(4);
        var entered = new CountDownLatch(4);
        var done = new CountDownLatch(4);
        var failure = new AtomicReference<Throwable>();
        try (var service = new BatchingExecutor<Integer>(4, 8, 8, scheduled::add, new IntOperation() {
            public Object batchKey(Integer request) { return null; }
            public void execute(Integer request) throws Exception {
                entered.countDown();
                if (!entered.await(5, TimeUnit.SECONDS)) throw new IllegalStateException("Budget underutilized");
            }
            public void complete(Integer request, Throwable error) {
                if (error != null) failure.compareAndSet(null, error);
                done.countDown();
            }
        })) {
            for (int i = 0; i < 4; i++) assertTrue(service.trySubmit(i));
            assertEquals(4, scheduled.size());
            for (int i = 0; i < 4; i++) executor.execute(scheduled.remove());
            assertTrue(done.await(7, TimeUnit.SECONDS));
            assertNull(failure.get());
        } finally { executor.shutdownNow(); }
    }

    @Test(timeout=15000)
    public void closeWaitsForCallbacksAndPreservesInterruptStatus() throws Exception {
        var executor = Executors.newSingleThreadExecutor();
        var closer = Executors.newSingleThreadExecutor();
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var service = new BatchingExecutor<Integer>(1, 8, 8, executor, new IntOperation() {
            public void complete(Integer request, Throwable failure) {
                entered.countDown();
                boolean interrupted = false;
                for (;;) {
                    try { release.await(); break; }
                    catch (InterruptedException error) { interrupted = true; }
                }
                if (interrupted) Thread.currentThread().interrupt();
            }
        });
        try {
            assertTrue(service.trySubmit(0));
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            assertEquals(0, service.outstanding());
            var closed = closer.submit(() -> {
                Thread.currentThread().interrupt();
                service.close();
                return Thread.currentThread().isInterrupted();
            });
            assertThrows(TimeoutException.class, () -> closed.get(20, TimeUnit.MILLISECONDS));
            release.countDown();
            assertTrue(closed.get(5, TimeUnit.SECONDS));
        } finally { release.countDown(); service.close(); executor.shutdownNow(); closer.shutdownNow(); }
    }

    @Test(timeout=15000)
    public void concurrentSubmitAndCloseCompleteEveryAcceptedRequestExactlyOnce() throws Exception {
        var executor = Executors.newFixedThreadPool(4);
        var producers = Executors.newFixedThreadPool(4);
        var accepted = new AtomicInteger();
        var completions = new AtomicIntegerArray(2000);
        var first = new CountDownLatch(1);
        var start = new CountDownLatch(1);
        var failure = new AtomicReference<Throwable>();
        var service = new BatchingExecutor<Integer>(4, 32, 8, executor, new IntOperation() {
            public void complete(Integer request, Throwable error) {
                if (error != null) failure.compareAndSet(null, error);
                if (completions.incrementAndGet(request) != 1)
                    failure.compareAndSet(null, new AssertionError("Duplicate completion"));
            }
        });
        try {
            var futures = new java.util.ArrayList<Future<?>>();
            for (int producer = 0; producer < 4; producer++) {
                int offset = producer * 500;
                futures.add(producers.submit(() -> {
                    start.await();
                    for (int i = 0; i < 500; i++) {
                        try {
                            while (!service.trySubmit(offset + i)) Thread.yield();
                            accepted.incrementAndGet(); first.countDown();
                        } catch (RejectedExecutionException closed) { return null; }
                    }
                    return null;
                }));
            }
            start.countDown();
            assertTrue(first.await(5, TimeUnit.SECONDS));
            service.close();
            for (var future : futures) future.get(5, TimeUnit.SECONDS);
            assertNull(failure.get());
            int completed = 0;
            for (int i = 0; i < 2000; i++) completed += completions.get(i);
            assertEquals(accepted.get(), completed);
        } finally { start.countDown(); service.close(); producers.shutdownNow(); executor.shutdownNow(); }
    }

    @Test(timeout=15000)
    public void boundedRingsWrapUnderConcurrentMixedModelTraffic() throws Exception {
        int count = 20000;
        var executor = Executors.newFixedThreadPool(4);
        var producers = Executors.newFixedThreadPool(4);
        Object[] keys = {new Object(), new Object()};
        var done = new CountDownLatch(count);
        var completed = new AtomicIntegerArray(count);
        var failure = new AtomicReference<Throwable>();
        try (var service = new BatchingExecutor<Integer>(4, 32, 8, executor, new IntOperation() {
            public Object batchKey(Integer request) { return keys[(request / 3) & 1]; }
            public void prepareBatch(List<Integer> values) {
                Object key = batchKey(values.get(0));
                for (Integer value : values) assertSame(key, batchKey(value));
            }
            public void complete(Integer request, Throwable error) {
                if (error != null) failure.compareAndSet(null, error);
                if (completed.incrementAndGet(request) != 1)
                    failure.compareAndSet(null, new AssertionError("Duplicate completion"));
                done.countDown();
            }
        })) {
            var futures = new java.util.ArrayList<Future<?>>();
            for (int producer = 0; producer < 4; producer++) {
                int offset = producer * (count / 4);
                futures.add(producers.submit(() -> {
                    for (int i = 0; i < count / 4; i++)
                        while (!service.trySubmit(offset + i)) Thread.yield();
                }));
            }
            for (var future : futures) future.get(10, TimeUnit.SECONDS);
            assertTrue(done.await(3, TimeUnit.SECONDS));
            assertNull(failure.get());
            long queries = 0;
            var histogram = service.batchHistogram();
            for (int i = 1; i < histogram.length; i++) queries += i * histogram[i];
            assertEquals(count, queries);
            for (int i = 0; i < count; i++) assertEquals(1, completed.get(i));
        } finally { executor.shutdownNow(); producers.shutdownNow(); }
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
