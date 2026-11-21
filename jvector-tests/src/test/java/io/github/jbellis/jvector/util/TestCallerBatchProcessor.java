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
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import static org.junit.Assert.*;

public class TestCallerBatchProcessor {
    private static void awaitWaiting(CallerBatchProcessor<?, ?> processor, int expected) {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (processor.waitingCount() != expected && System.nanoTime() < deadline) Thread.yield();
        assertEquals(expected, processor.waitingCount());
    }

    private abstract static class IntegerOperation implements CallerBatchProcessor.Operation<Integer, int[]> {
        public void applyBatch(List<Integer> input, List<int[]> output) throws Exception {
            for (int i = 0; i < input.size(); i++) apply(input.get(i), output.get(i));
        }
    }

    @Test(timeout=10000)
    public void loneCallsRunImmediatelyOnTheCallerAtBothBatchLimits() {
        for (int limit : new int[]{1,8}) {
            Thread owner = Thread.currentThread();
            var processor = new CallerBatchProcessor<>(2, limit, new IntegerOperation() {
                public void apply(Integer value, int[] output) {
                    assertSame(owner, Thread.currentThread());
                    output[0] = value;
                }
            });
            int[] out = new int[1];
            for (int i = 0; i < 100; i++) { processor.process(i, out); assertEquals(i, out[0]); }
            assertEquals(100, processor.batchHistogram()[1]);
            assertEquals(0, processor.waitingCount());
        }
    }

    @Test(timeout=15000)
    public void concurrentLeadersCombineOnlyExistingFollowersAndPublishOwnedOutputs() throws Exception {
        var entered = new CountDownLatch(2);
        var release = new CountDownLatch(1);
        var active = new AtomicInteger();
        var maximum = new AtomicInteger();
        var processor = new CallerBatchProcessor<>(2, 3, new CallerBatchProcessor.Operation<Integer,int[]>() {
            private void begin() { maximum.accumulateAndGet(active.incrementAndGet(), Math::max); }
            public void apply(Integer value, int[] out) throws Exception {
                begin();
                try { if (value < 2) { entered.countDown(); release.await(); } out[0] = value * 7; }
                finally { active.decrementAndGet(); }
            }
            public void applyBatch(List<Integer> input, List<int[]> output) {
                begin();
                try {
                    assertTrue(input.size() <= 3);
                    for (int i = 0; i < input.size(); i++) output.get(i)[0] = input.get(i) * 7;
                    try { input.clear(); fail("Batch input view must be read-only"); }
                    catch (UnsupportedOperationException expected) { }
                } finally { active.decrementAndGet(); }
            }
        });
        var executor = Executors.newFixedThreadPool(9);
        try {
            var futures = new ArrayList<Future<?>>();
            for (int i = 0; i < 2; i++) {
                final int value = i;
                futures.add(executor.submit(() -> { int[] out = new int[1]; processor.process(value, out); assertEquals(value * 7, out[0]); }));
            }
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            assertEquals(2, maximum.get());
            for (int i = 2; i < 9; i++) {
                final int value = i;
                futures.add(executor.submit(() -> {
                    Thread original = Thread.currentThread();
                    int[] out = new int[1]; processor.process(value, out);
                    assertSame(original, Thread.currentThread());
                    assertEquals(value * 7, out[0]);
                }));
            }
            awaitWaiting(processor, 7);
            release.countDown();
            for (var future : futures) future.get(5, TimeUnit.SECONDS);
            long[] histogram = processor.batchHistogram();
            assertEquals(2, maximum.get());
            assertTrue(histogram[2] + histogram[3] > 0);
            assertEquals(9, histogram[1] + 2 * histogram[2] + 3 * histogram[3]);
        } finally { release.countDown(); executor.shutdownNow(); }
    }

    @Test(timeout=20000)
    public void repeatedSlotReuseDoesNotLoseOrMisrouteRequests() throws Exception {
        var processor = new CallerBatchProcessor<>(2, 8, new IntegerOperation() {
            public void apply(Integer value, int[] out) { Thread.yield(); out[0] = value; }
        });
        var executor = Executors.newFixedThreadPool(16);
        var start = new CountDownLatch(1);
        try {
            var futures = new ArrayList<Future<?>>();
            for (int thread = 0; thread < 16; thread++) {
                final int prefix = thread * 1000;
                futures.add(executor.submit(() -> {
                    start.await();
                    int[] output = new int[1];
                    for (int i = 0; i < 500; i++) {
                        processor.process(prefix + i, output);
                        assertEquals(prefix + i, output[0]);
                    }
                    return null;
                }));
            }
            start.countDown();
            for (var future : futures) future.get(10, TimeUnit.SECONDS);
            long[] histogram = processor.batchHistogram();
            long requests = 0;
            for (int i = 1; i < histogram.length; i++) requests += i * histogram[i];
            assertEquals(8000, requests);
            assertEquals(0, processor.waitingCount());
        } finally { start.countDown(); executor.shutdownNow(); }
    }

    @Test(timeout=15000)
    public void batchFailureIsPublishedAndDoesNotStrandTheNextLeader() throws Exception {
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var processor = new CallerBatchProcessor<>(1, 2, new IntegerOperation() {
            public void apply(Integer value, int[] out) throws Exception {
                if (value == 0) { entered.countDown(); release.await(); }
                if (value == 2) throw new IllegalArgumentException("batch failure");
                out[0] = value;
            }
        });
        var executor = Executors.newFixedThreadPool(4);
        try {
            var first = executor.submit(() -> processor.process(0, new int[1]));
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            var one = executor.submit(() -> processor.process(1, new int[1]));
            awaitWaiting(processor, 1);
            var two = executor.submit(() -> processor.process(2, new int[1]));
            awaitWaiting(processor, 2);
            var three = executor.submit(() -> { int[] out = new int[1]; processor.process(3, out); return out[0]; });
            awaitWaiting(processor, 3);
            release.countDown(); first.get(5, TimeUnit.SECONDS);
            for (var failed : List.of(one, two)) {
                try { failed.get(5, TimeUnit.SECONDS); fail("Missing batch failure"); }
                catch (ExecutionException expected) { assertTrue(expected.getCause() instanceof IllegalArgumentException); }
            }
            assertEquals(Integer.valueOf(3), three.get(5, TimeUnit.SECONDS));
        } finally { release.countDown(); executor.shutdownNow(); }
    }

    @Test(timeout=15000)
    public void interruptedFollowerKeepsItsBuffersAliveUntilCompletion() throws Exception {
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var completed = new CountDownLatch(1);
        var interrupted = new AtomicBoolean();
        var processor = new CallerBatchProcessor<>(1, 8, new IntegerOperation() {
            public void apply(Integer value, int[] out) throws Exception {
                if (value == 0) { entered.countDown(); release.await(); }
                out[0] = value;
            }
        });
        var executor = Executors.newSingleThreadExecutor();
        Thread follower = new Thread(() -> {
            int[] output = new int[1]; processor.process(17, output);
            assertEquals(17, output[0]);
            interrupted.set(Thread.currentThread().isInterrupted()); completed.countDown();
        });
        try {
            var leader = executor.submit(() -> processor.process(0, new int[1]));
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            follower.start(); awaitWaiting(processor, 1); follower.interrupt();
            assertFalse(completed.await(50, TimeUnit.MILLISECONDS));
            release.countDown(); leader.get(5, TimeUnit.SECONDS);
            assertTrue(completed.await(5, TimeUnit.SECONDS)); assertTrue(interrupted.get());
        } finally { release.countDown(); follower.join(5000); executor.shutdownNow(); }
    }

    @Test(timeout=10000)
    public void reentrantOperationFailsWithoutLeakingLeaderCapacity() {
        var reference = new AtomicReference<CallerBatchProcessor<Integer,int[]>>();
        var processor = new CallerBatchProcessor<>(1, 8, new IntegerOperation() {
            public void apply(Integer value, int[] out) {
                if (value == 0) reference.get().process(1, out);
                out[0] = value;
            }
        });
        reference.set(processor);
        try { processor.process(0, new int[1]); fail("Reentrant operation must fail"); }
        catch (IllegalStateException expected) { }
        int[] out = new int[1]; processor.process(2, out); assertEquals(2, out[0]);
    }
    @Test
    public void repeatedSingletonDispatchDoesNotAllocateRequestObjects() {
        var threadBean = java.lang.management.ManagementFactory.getThreadMXBean();
        org.junit.Assume.assumeTrue(threadBean instanceof com.sun.management.ThreadMXBean);
        var bean = (com.sun.management.ThreadMXBean) threadBean;
        org.junit.Assume.assumeTrue(bean.isThreadAllocatedMemorySupported());
        if (!bean.isThreadAllocatedMemoryEnabled()) bean.setThreadAllocatedMemoryEnabled(true);
        var processor = CallerBatchProcessor.opportunistic(8, 8, new IntegerOperation() {
            public void apply(Integer value, int[] output) { output[0] = value; }
        });
        int[] output = new int[1];
        for (int i = 0; i < 50000; i++) processor.process(1, output);
        long thread = Thread.currentThread().getId();
        long before = bean.getThreadAllocatedBytes(thread);
        for (int i = 0; i < 10000; i++) processor.process(1, output);
        long allocated = bean.getThreadAllocatedBytes(thread) - before;
        assertEquals(0, allocated);
        assertEquals(1, output[0]);
    }

    @Test(timeout=15000)
    public void concurrentHistogramSnapshotsDoNotChangeDispatchOrFinalCounts() throws Exception {
        // Also exercises leader counts above a machine-word bitmap's capacity.
        var processor = new CallerBatchProcessor<>(65, 8, new IntegerOperation() {
            public void apply(Integer value, int[] output) { output[0] = value; }
        });
        var pool = Executors.newFixedThreadPool(8);
        var stop = new AtomicBoolean();
        var observer = new Thread(() -> {
            while (!stop.get()) {
                processor.batchHistogram();
                Thread.yield();
            }
        });
        observer.start();
        try {
            var work = new ArrayList<Callable<Void>>();
            for (int worker = 0; worker < 8; worker++) work.add(() -> {
                int[] out = new int[1];
                for (int i = 0; i < 1000; i++) {
                    processor.process(1, out);
                    assertEquals(1, out[0]);
                }
                return null;
            });
            for (var future : pool.invokeAll(work)) future.get();
            assertEquals(8000, processor.batchHistogram()[1]);
            assertEquals(0, processor.waitingCount());
        } finally {
            stop.set(true);
            observer.join(5000);
            pool.shutdownNow();
        }
    }


    @Test(timeout=15000)
    public void pendingInputsAreServedInFifoOrderRegardlessOfCombinerThread() throws Exception {
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var order = new ArrayList<Integer>();
        var processor = new CallerBatchProcessor<>(1, 3, new IntegerOperation() {
            public void apply(Integer value, int[] output) throws Exception {
                if (value == 0) { entered.countDown(); release.await(); }
                order.add(value);
                output[0] = value;
            }
        });
        var pool = Executors.newFixedThreadPool(8);
        try {
            var work = new ArrayList<Future<?>>();
            work.add(pool.submit(() -> processor.process(0, new int[1])));
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            for (int i = 1; i < 8; i++) {
                final int value = i;
                work.add(pool.submit(() -> {
                    int[] output = new int[1];
                    processor.process(value, output);
                    assertEquals(value, output[0]);
                }));
                awaitWaiting(processor, i);
            }
            release.countDown();
            for (var future : work) future.get(5, TimeUnit.SECONDS);
            assertEquals(List.of(0, 1, 2, 3, 4, 5, 6, 7), order);
        } finally { release.countDown(); pool.shutdownNow(); }
    }

    @Test(timeout=20000)
    public void contendedDispatchReusesRequestsAndBatchViewsWithoutJavaAllocation() throws Exception {
        var threadBean = java.lang.management.ManagementFactory.getThreadMXBean();
        org.junit.Assume.assumeTrue(threadBean instanceof com.sun.management.ThreadMXBean);
        var bean = (com.sun.management.ThreadMXBean) threadBean;
        org.junit.Assume.assumeTrue(bean.isThreadAllocatedMemorySupported());
        if (!bean.isThreadAllocatedMemoryEnabled()) bean.setThreadAllocatedMemoryEnabled(true);
        var processor = new CallerBatchProcessor<>(1, 8, new IntegerOperation() {
            public void apply(Integer value, int[] output) {
                for (int i = 0; i < 100; i++) Thread.onSpinWait();
                output[0] = value;
            }
        });
        var pool = Executors.newFixedThreadPool(8);
        try {
            var work = new ArrayList<Callable<Long>>();
            for (int worker = 0; worker < 8; worker++) work.add(() -> {
                int[] output = new int[1];
                for (int i = 0; i < 20000; i++) processor.process(1, output);
                // A JIT deoptimization inside a window can make the JVM's own VarHandle linkage allocate, so measure
                // up to three windows and require one to be allocation-free. A real allocation shows up in every one.
                long tid = Thread.currentThread().getId();
                long bytes = Long.MAX_VALUE;
                for (int window = 0; window < 3 && bytes != 0; window++) {
                    long before = bean.getThreadAllocatedBytes(tid);
                    for (int i = 0; i < 2000; i++) processor.process(1, output);
                    bytes = Math.min(bytes, bean.getThreadAllocatedBytes(tid) - before);
                }
                assertEquals(1, output[0]);
                return bytes;
            });
            for (var result : pool.invokeAll(work)) assertEquals(0L, result.get().longValue());
            long[] histogram = processor.batchHistogram();
            long combined = 0;
            for (int size = 2; size < histogram.length; size++) combined += histogram[size];
            assertTrue("Must exercise contended batch dispatch", combined > 0);
        } finally { pool.shutdownNow(); }
    }

    @Test(timeout=20000)
    public void completionRacingWithParkingNeverLosesTheWakeup() throws Exception {
        var processor = CallerBatchProcessor.opportunistic(8, 8, new IntegerOperation() {
            public void apply(Integer value, int[] output) { output[0] = value; }
        });
        var pool = Executors.newFixedThreadPool(16);
        var ready = new CountDownLatch(16);
        var start = new CountDownLatch(1);
        try {
            var work = new ArrayList<Callable<Void>>();
            for (int worker = 0; worker < 16; worker++) work.add(() -> {
                int[] output = new int[1];
                ready.countDown();
                start.await();
                for (int i = 0; i < 10000; i++) {
                    processor.process(1, output);
                    assertEquals(1, output[0]);
                }
                return null;
            });
            var futures = new ArrayList<Future<Void>>();
            for (var task : work) futures.add(pool.submit(task));
            assertTrue(ready.await(5, TimeUnit.SECONDS));
            start.countDown();
            for (var future : futures) future.get(10, TimeUnit.SECONDS);
            long total = 0;
            long[] histogram = processor.batchHistogram();
            for (int size = 1; size < histogram.length; size++) total += size * histogram[size];
            assertEquals(160000, total);
        } finally { start.countDown(); pool.shutdownNow(); }
    }

    @Test(timeout=20000)
    public void opportunisticBypassUsesTheCallerBudgetAndDrainsOverflowBeforeReturningToDirect() throws Exception {
        var directEntered = new CountDownLatch(4);
        var directRelease = new CountDownLatch(1);
        var overflowEntered = new CountDownLatch(1);
        var overflowRelease = new CountDownLatch(1);
        var active = new AtomicInteger();
        var maximum = new AtomicInteger();
        var batches = new AtomicInteger();
        var lastCaller = new AtomicReference<Thread>();
        var processor = CallerBatchProcessor.opportunistic(4, 8,
                new CallerBatchProcessor.Operation<Integer, int[]>() {
                    public void apply(Integer value, int[] output) throws Exception {
                        maximum.accumulateAndGet(active.incrementAndGet(), Math::max);
                        try {
                            if (value < 4) { directEntered.countDown(); directRelease.await(); }
                            if (value == 4) { overflowEntered.countDown(); overflowRelease.await(); }
                            if (value == 13) lastCaller.set(Thread.currentThread());
                            output[0] = value;
                        } finally { active.decrementAndGet(); }
                    }
                    public void applyBatch(List<Integer> input, List<int[]> output) {
                        maximum.accumulateAndGet(active.incrementAndGet(), Math::max);
                        try {
                            batches.incrementAndGet();
                            for (int i = 0; i < input.size(); i++) output.get(i)[0] = input.get(i);
                        } finally { active.decrementAndGet(); }
                    }
                });
        var pool = Executors.newFixedThreadPool(13);
        try {
            var work = new ArrayList<Future<?>>();
            for (int value = 0; value < 4; value++) {
                final int v = value;
                work.add(pool.submit(() -> processor.process(v, new int[1])));
            }
            assertTrue(directEntered.await(5, TimeUnit.SECONDS));
            assertEquals(4, maximum.get()); // No premature serialization at the combiner limit of one.
            work.add(pool.submit(() -> processor.process(4, new int[1])));
            assertTrue(overflowEntered.await(5, TimeUnit.SECONDS));
            assertEquals(5, maximum.get()); // Four direct calls plus one combining call.
            for (int value = 5; value < 12; value++) {
                final int v = value;
                work.add(pool.submit(() -> {
                    int[] output = new int[1]; processor.process(v, output); assertEquals(v, output[0]);
                }));
            }
            awaitWaiting(processor, 7);
            directRelease.countDown();
            for (int k = 0; k < 4; k++) work.get(k).get(5, TimeUnit.SECONDS);
            work.add(pool.submit(() -> processor.process(12, new int[1])));
            awaitWaiting(processor, 8); // Released direct slots cannot barge ahead of queued work.
            overflowRelease.countDown();
            for (var result : work) result.get(5, TimeUnit.SECONDS);
            assertTrue(batches.get() > 0);
            assertTrue(maximum.get() <= 5);
            int[] output = new int[1];
            processor.process(13, output);
            assertEquals(13, output[0]);
            assertSame(Thread.currentThread(), lastCaller.get());
            long total = 0;
            long[] histogram = processor.batchHistogram();
            for (int b = 1; b < histogram.length; b++) total += b * histogram[b];
            assertEquals(14, total);
        } finally { directRelease.countDown(); overflowRelease.countDown(); pool.shutdownNow(); }
    }

    @Test(timeout=15000)
    public void freeDirectCapacityDoesNotWaitForAnExecutingOverflowOperation() throws Exception {
        var directEntered = new CountDownLatch(1);
        var directRelease = new CountDownLatch(1);
        var overflowEntered = new CountDownLatch(1);
        var overflowRelease = new CountDownLatch(1);
        var processor = CallerBatchProcessor.opportunistic(1, 8, new IntegerOperation() {
            public void apply(Integer value, int[] output) throws Exception {
                if (value == 0) { directEntered.countDown(); directRelease.await(); }
                if (value == 1) { overflowEntered.countDown(); overflowRelease.await(); }
                output[0] = value;
            }
        });
        var pool = Executors.newFixedThreadPool(3);
        try {
            var direct = pool.submit(() -> processor.process(0, new int[1]));
            assertTrue(directEntered.await(5, TimeUnit.SECONDS));
            var overflow = pool.submit(() -> processor.process(1, new int[1]));
            assertTrue(overflowEntered.await(5, TimeUnit.SECONDS));
            directRelease.countDown(); direct.get(5, TimeUnit.SECONDS);
            assertEquals(0, processor.waitingCount());
            // The earlier overflow call is already executing, not waiting. It
            // must not make a new call wait with an available direct slot.
            var next = pool.submit(() -> {
                int[] output = new int[1]; processor.process(2, output); return output[0];
            });
            assertEquals(Integer.valueOf(2), next.get(1, TimeUnit.SECONDS));
            overflowRelease.countDown(); overflow.get(5, TimeUnit.SECONDS);
            assertEquals(3, processor.batchHistogram()[1]);
        } finally { directRelease.countDown(); overflowRelease.countDown(); pool.shutdownNow(); }
    }

    @Test(timeout=15000)
    public void realBatchesRetainSharingUntilTheirBorrowedInputsComplete() throws Exception {
        var directEntered = new CountDownLatch(1);
        var directRelease = new CountDownLatch(1);
        var overflowEntered = new CountDownLatch(1);
        var overflowRelease = new CountDownLatch(1);
        var batchEntered = new CountDownLatch(1);
        var batchRelease = new CountDownLatch(1);
        var processor = CallerBatchProcessor.opportunistic(1, 8, new IntegerOperation() {
            public void apply(Integer value, int[] output) throws Exception {
                if (value == 0) { directEntered.countDown(); directRelease.await(); }
                if (value == 1) { overflowEntered.countDown(); overflowRelease.await(); }
                output[0] = value;
            }
            public void applyBatch(List<Integer> input, List<int[]> output) throws Exception {
                assertEquals(2, input.size());
                batchEntered.countDown(); batchRelease.await();
                super.applyBatch(input, output);
            }
        });
        var pool = Executors.newFixedThreadPool(5);
        try {
            var first = pool.submit(() -> processor.process(0, new int[1]));
            assertTrue(directEntered.await(5, TimeUnit.SECONDS));
            var overflow = pool.submit(() -> processor.process(1, new int[1]));
            assertTrue(overflowEntered.await(5, TimeUnit.SECONDS));
            var two = pool.submit(() -> processor.process(2, new int[1]));
            awaitWaiting(processor, 1);
            var three = pool.submit(() -> processor.process(3, new int[1]));
            awaitWaiting(processor, 2);
            overflowRelease.countDown(); overflow.get(5, TimeUnit.SECONDS);
            assertTrue(batchEntered.await(5, TimeUnit.SECONDS));
            directRelease.countDown(); first.get(5, TimeUnit.SECONDS);
            var next = pool.submit(() -> {
                int[] output = new int[1]; processor.process(4, output); return output[0];
            });
            awaitWaiting(processor, 1);
            batchRelease.countDown();
            two.get(5, TimeUnit.SECONDS); three.get(5, TimeUnit.SECONDS);
            assertEquals(Integer.valueOf(4), next.get(5, TimeUnit.SECONDS));
            assertEquals(3, processor.batchHistogram()[1]);
            assertEquals(1, processor.batchHistogram()[2]);
        } finally {
            directRelease.countDown(); overflowRelease.countDown(); batchRelease.countDown(); pool.shutdownNow();
        }
    }

    @Test(timeout=10000)
    public void opportunisticBypassReleasesCapacityAfterFailureAndRejectsInvalidBudgets() {
        var operation = new IntegerOperation() {
            public void apply(Integer value, int[] output) {
                if (value == 0) throw new IllegalArgumentException("failure");
                output[0] = value;
            }
        };
        for (int parallelism : new int[]{1, 4, 8}) {
            var processor = CallerBatchProcessor.opportunistic(parallelism, 8, operation);
            for (int i = 0; i < parallelism + 2; i++) {
                try { processor.process(0, new int[1]); fail("Missing failure"); }
                catch (IllegalArgumentException expected) { }
            }
            int[] output = new int[1]; processor.process(7, output); assertEquals(7, output[0]);
        }
        try { CallerBatchProcessor.opportunistic(0, 8, operation); fail("Invalid budget"); }
        catch (IllegalArgumentException expected) { }
    }

}
