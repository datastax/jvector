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

package io.github.jbellis.jvector.graph;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class TestParallelExecutor {

    // ---- over(...) argument validation ----

    @Test(expected = NullPointerException.class)
    public void overRejectsNullExecutor() {
        ParallelExecutor.over(null, 1);
    }

    @Test
    public void overRejectsNonPositiveParallelism() {
        ExecutorService es = Executors.newSingleThreadExecutor();
        try {
            for (int bad : new int[]{0, -1}) {
                try {
                    ParallelExecutor.over(es, bad);
                    fail("expected IllegalArgumentException for parallelism " + bad);
                } catch (IllegalArgumentException expected) {
                    // ok
                }
            }
        } finally {
            es.shutdown();
        }
    }

    // ---- completeness: every element exactly once ----

    @Test
    public void chunkingForEachIntCoversRangeExactlyOnce() {
        ExecutorService es = Executors.newFixedThreadPool(3);
        try {
            ParallelExecutor pe = ParallelExecutor.over(es, 3);
            AtomicIntegerArray hits = new AtomicIntegerArray(10_000);
            pe.forEachInt(10_000, hits::incrementAndGet);
            for (int i = 0; i < hits.length(); i++) {
                assertEquals("index " + i, 1, hits.get(i));
            }
            // Zero-size range: no calls, no error.
            AtomicInteger calls = new AtomicInteger();
            pe.forEachInt(0, i -> calls.incrementAndGet());
            assertEquals(0, calls.get());
        } finally {
            es.shutdown();
        }
    }

    @Test
    public void chunkingForEachIntStreamCoversAllElements() {
        ExecutorService es = Executors.newFixedThreadPool(2);
        try {
            ParallelExecutor pe = ParallelExecutor.over(es, 2);
            // A filtered (non-SIZED) stream exercises traversal-side batching.
            Set<Integer> expected = IntStream.range(0, 5_000).filter(i -> i % 3 == 0)
                    .boxed().collect(Collectors.toSet());
            Set<Integer> seen = new ConcurrentSkipListSet<>();
            pe.forEach(IntStream.range(0, 5_000).filter(i -> i % 3 == 0), seen::add);
            assertEquals(expected, seen);
        } finally {
            es.shutdown();
        }
    }

    @Test
    public void chunkingForEachGenericStreamCoversAllElements() {
        ExecutorService es = Executors.newFixedThreadPool(2);
        try {
            ParallelExecutor pe = ParallelExecutor.over(es, 2);
            Set<String> expected = IntStream.range(0, 1_000).mapToObj(Integer::toString)
                    .collect(Collectors.toSet());
            Set<String> seen = new ConcurrentSkipListSet<>();
            pe.forEach(IntStream.range(0, 1_000).mapToObj(Integer::toString), seen::add);
            assertEquals(expected, seen);
        } finally {
            es.shutdown();
        }
    }

    // ---- thread placement: what is (and is not) distributed ----

    @Test
    public void chunkingTraversesSourceOnCallingThreadAndRunsBodyOnWorkers() {
        ExecutorService es = Executors.newFixedThreadPool(2);
        try {
            ParallelExecutor pe = ParallelExecutor.over(es, 2);
            Thread caller = Thread.currentThread();
            Set<Thread> traversalThreads = ConcurrentHashMap.newKeySet();
            Set<Thread> bodyThreads = ConcurrentHashMap.newKeySet();
            IntStream source = IntStream.range(0, 2_000).map(i -> {
                traversalThreads.add(Thread.currentThread());
                return i;
            });
            pe.forEach(source, i -> bodyThreads.add(Thread.currentThread()));
            assertEquals("upstream stages run on the calling thread only",
                    Set.of(caller), traversalThreads);
            assertFalse("the body must not run on the calling thread", bodyThreads.contains(caller));
        } finally {
            es.shutdown();
        }
    }

    @Test
    public void overForkJoinPoolDelegatesToStreamDecomposition() {
        ForkJoinPool pool = new ForkJoinPool(2);
        try {
            ParallelExecutor pe = ParallelExecutor.over(pool, 5);
            Thread caller = Thread.currentThread();
            AtomicBoolean traversedOnCaller = new AtomicBoolean(false);
            IntStream source = IntStream.range(0, 2_000).map(i -> {
                if (Thread.currentThread() == caller) {
                    traversedOnCaller.set(true);
                }
                return i;
            });
            AtomicInteger count = new AtomicInteger();
            pe.forEach(source, i -> count.incrementAndGet());
            assertEquals(2_000, count.get());
            assertFalse("a ForkJoinPool must delegate to forkJoin(): upstream stages run inside the pool",
                    traversedOnCaller.get());
        } finally {
            pool.shutdown();
        }
    }

    @Test
    public void callerRunsExecutesInOrderOnCallingThread() {
        ParallelExecutor pe = ParallelExecutor.callerRuns();
        Thread caller = Thread.currentThread();
        List<Integer> order = new ArrayList<>();
        pe.forEachInt(100, i -> {
            assertSame(caller, Thread.currentThread());
            order.add(i);
        });
        assertEquals(IntStream.range(0, 100).boxed().collect(Collectors.toList()), order);
    }

    // ---- failure and interrupt semantics ----

    @Test
    public void chunkingBodyFailurePropagatesWithNothingLeftRunning() {
        ExecutorService es = Executors.newFixedThreadPool(4);
        try {
            ParallelExecutor pe = ParallelExecutor.over(es, 4);
            RuntimeException marker = new RuntimeException("marker");
            AtomicInteger active = new AtomicInteger();
            try {
                pe.forEachInt(100, i -> {
                    active.incrementAndGet();
                    try {
                        if (i == 41) {
                            throw marker;
                        }
                        Thread.sleep(5);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        active.decrementAndGet();
                    }
                });
                fail("expected the body failure to propagate");
            } catch (RuntimeException e) {
                assertSame("the first observed failure must propagate as-is", marker, e);
            }
            assertEquals("no body may still be running once the failure unwinds", 0, active.get());
        } finally {
            es.shutdown();
        }
    }

    /**
     * The drain guarantee on the fork/join path. A parallel stream propagates a body failure up
     * through join() as soon as one branch throws, so the naive implementation let the caller
     * resume while sibling bodies were still running on pool workers — and an embedder that frees
     * the memory those bodies are reading (unmapping a file once the call returns) gets a SIGSEGV
     * rather than an exception. Bodies must therefore all be finished when the call returns.
     */
    @Test(timeout = 30_000)
    public void forkJoinBodyFailurePropagatesWithNothingLeftRunning() {
        ForkJoinPool pool = new ForkJoinPool(4);
        try {
            ParallelExecutor pe = ParallelExecutor.forkJoin(pool);
            RuntimeException marker = new RuntimeException("marker");
            AtomicInteger active = new AtomicInteger();
            AtomicInteger peakActive = new AtomicInteger();
            try {
                pe.forEachInt(100, i -> {
                    int now = active.incrementAndGet();
                    peakActive.accumulateAndGet(now, Math::max);
                    try {
                        if (i == 41) {
                            throw marker;
                        }
                        Thread.sleep(5);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        active.decrementAndGet();
                    }
                });
                fail("expected the body failure to propagate");
            } catch (RuntimeException e) {
                assertSame("the first observed failure must propagate as-is", marker, e);
            }
            assertEquals("no body may still be running once the failure unwinds", 0, active.get());
            // Without real concurrency the assertion above would hold trivially.
            assertTrue("bodies should have run concurrently, peak was " + peakActive.get(),
                       peakActive.get() > 1);
        } finally {
            pool.shutdownNow();
        }
    }

    /** Same guarantee on the generic-stream path. */
    @Test(timeout = 30_000)
    public void forkJoinStreamBodyFailurePropagatesWithNothingLeftRunning() {
        ForkJoinPool pool = new ForkJoinPool(4);
        try {
            ParallelExecutor pe = ParallelExecutor.forkJoin(pool);
            RuntimeException marker = new RuntimeException("marker");
            AtomicInteger active = new AtomicInteger();
            AtomicInteger peakActive = new AtomicInteger();
            List<Integer> items = IntStream.range(0, 100).boxed().collect(Collectors.toList());
            try {
                pe.forEach(items.stream(), i -> {
                    int now = active.incrementAndGet();
                    peakActive.accumulateAndGet(now, Math::max);
                    try {
                        if (i == 41) {
                            throw marker;
                        }
                        Thread.sleep(5);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        active.decrementAndGet();
                    }
                });
                fail("expected the body failure to propagate");
            } catch (RuntimeException e) {
                assertSame("the first observed failure must propagate as-is", marker, e);
            }
            assertEquals("no body may still be running once the failure unwinds", 0, active.get());
            assertTrue("bodies should have run concurrently, peak was " + peakActive.get(),
                       peakActive.get() > 1);
        } finally {
            pool.shutdownNow();
        }
    }

    /** A failure must not stop the iteration from settling every element that had already begun. */
    @Test(timeout = 30_000)
    public void forkJoinSkipsRemainingElementsAfterAFailure() {
        ForkJoinPool pool = new ForkJoinPool(4);
        try {
            ParallelExecutor pe = ParallelExecutor.forkJoin(pool);
            AtomicInteger completed = new AtomicInteger();
            try {
                pe.forEachInt(10_000, i -> {
                    if (i == 0) {
                        throw new IllegalStateException("boom");
                    }
                    completed.incrementAndGet();
                });
                fail("expected the body failure to propagate");
            } catch (IllegalStateException expected) {
                // expected
            }
            assertTrue("elements after the failure should be skipped, not all 10000 run, saw "
                               + completed.get(),
                       completed.get() < 10_000);
        } finally {
            pool.shutdownNow();
        }
    }

    @Test(timeout = 10_000)
    public void chunkingInterruptDrainsStartedWorkAndRestoresFlag() throws Exception {
        ExecutorService es = Executors.newFixedThreadPool(2);
        try {
            ParallelExecutor pe = ParallelExecutor.over(es, 2);
            AtomicInteger active = new AtomicInteger();
            AtomicReference<Throwable> caught = new AtomicReference<>();
            AtomicBoolean flagRestored = new AtomicBoolean();
            AtomicInteger activeAtUnwind = new AtomicInteger(-1);
            Thread runner = new Thread(() -> {
                try {
                    pe.forEachInt(64, i -> {
                        active.incrementAndGet();
                        try {
                            Thread.sleep(20);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        } finally {
                            active.decrementAndGet();
                        }
                    });
                } catch (Throwable t) {
                    caught.set(t);
                    flagRestored.set(Thread.currentThread().isInterrupted());
                    activeAtUnwind.set(active.get());
                }
            }, "interrupted-orchestrator");
            runner.start();
            Thread.sleep(100);
            runner.interrupt();
            runner.join(8_000);

            assertFalse("orchestrator must not hang", runner.isAlive());
            assertTrue("expected a RuntimeException, got " + caught.get(),
                    caught.get() instanceof RuntimeException);
            assertTrue("cause must be the InterruptedException, got " + caught.get().getCause(),
                    caught.get().getCause() instanceof InterruptedException);
            assertTrue("interrupt flag must be restored before unwinding", flagRestored.get());
            assertEquals("no body may still be running once the interrupt unwinds", 0, activeAtUnwind.get());
        } finally {
            es.shutdown();
        }
    }

    @Test
    public void chunkingRejectedExecutionPropagates() {
        ExecutorService es = Executors.newSingleThreadExecutor();
        es.shutdown();
        ParallelExecutor pe = ParallelExecutor.over(es, 1);
        try {
            pe.forEachInt(10, i -> { });
            fail("expected RejectedExecutionException");
        } catch (RuntimeException e) {
            assertTrue("expected RejectedExecutionException, got " + e,
                    e instanceof java.util.concurrent.RejectedExecutionException);
        }
    }

    // ---- reentrancy ----

    @Test(timeout = 10_000)
    public void chunkingNestedUseRunsInlineWithoutDeadlock() {
        ExecutorService es = Executors.newFixedThreadPool(2);
        try {
            ParallelExecutor pe = ParallelExecutor.over(es, 2);
            AtomicInteger inner = new AtomicInteger();
            // Without the inline-degrade guard, every worker blocks awaiting sub-chunks that can
            // never be scheduled on the bounded pool; the timeout turns that hang into a failure.
            pe.forEachInt(8, i -> pe.forEachInt(10, j -> inner.incrementAndGet()));
            assertEquals(80, inner.get());
        } finally {
            es.shutdown();
        }
    }
}
