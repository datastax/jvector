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

package io.github.jbellis.jvector.graph.disk;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/// Repro for PR #747: an interrupted parallel write keeps running until every record is written.
///
/// ## The bug
///
/// `ParallelGraphWriter.writeL0Records` submits one task per ordinal range, then
/// `awaitTasks` waits on each future in turn. When the waiting thread is interrupted,
/// `awaitTasks` records the interruption and calls `future.get()` again. When a task fails, it
/// keeps the failure and moves on to the next future. Either way it waits for every task. No
/// task is cancelled, so not-yet-started ranges still run and every running supplier runs to
/// the end of its range.
///
/// Joining before the channel and suppliers are released is right. But nothing stops the work
/// being joined. So an interrupted or already-failed write runs the rest of the final
/// serialization, encoding every remaining record, and only then reports the interruption or
/// failure. For a 10M-node graph that's the whole final write. Before #747, `writeL0Records`
/// threw on interrupt immediately, without joining.
///
/// ## The test
///
/// Writes a 6,000-node graph with two writer workers and an `InlineVectors` supplier that takes
/// 1 ms per record, so an uninterrupted write takes about 3 s. After the first 50 records it
/// interrupts the writing thread, then expects the write to fail within 1 s and to have supplied
/// fewer than all records. At `ec42a94b` the write runs to completion, 3.2 s after the
/// interrupt, and supplies every record.
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestParallelWriterInterrupt extends LuceneTestCase {
    private static final int SIZE = 6000, DIMENSIONS = 17, WORKERS = 2;
    private static final long MAX_STOP_MILLIS = 1000;

    @Test(timeout = 120_000)
    public void interruptedWriteStopsPromptly() throws Exception {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var random = new Random(42);
        var list = new ArrayList<VectorFloat<?>>();
        for (int n = 0; n < SIZE; n++) {
            var v = vts.createFloatVector(DIMENSIONS);
            for (int d = 0; d < DIMENSIONS; d++) v.set(d, random.nextFloat() - 0.5f);
            list.add(v);
        }
        var values = new ListRandomAccessVectorValues(list, DIMENSIONS);
        var path = Files.createTempFile("parallel-writer-interrupt", ".idx");
        try (var builder = new GraphIndexBuilder(values, VectorSimilarityFunction.DOT_PRODUCT,
                8, 32, 1.2f, 1.2f, true)) {
            var graph = TestUtil.buildSequentially(builder, values);
            var supplied = new AtomicInteger();
            var started = new CountDownLatch(50);
            var failure = new AtomicReference<Throwable>();
            var interruptedAt = new AtomicReference<Long>();
            var finishedAt = new AtomicReference<Long>();

            var writing = new Thread(() -> {
                try (var writer = new OnDiskParallelGraphIndexWriter.Builder(graph, path)
                        .withParallelWorkerThreads(WORKERS)
                        .with(new InlineVectors(DIMENSIONS))
                        .build()) {
                    writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS, ordinal -> {
                        try {
                            Thread.sleep(1);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            throw new IllegalStateException("supplier interrupted", e);
                        }
                        supplied.incrementAndGet();
                        started.countDown();
                        return new InlineVectors.State(values.getVector(ordinal));
                    }));
                } catch (Throwable t) {
                    failure.set(t);
                } finally {
                    finishedAt.set(System.nanoTime());
                }
            }, "parallel-writer-under-test");
            writing.start();
            assertTrue("writer did not start supplying records", started.await(30, TimeUnit.SECONDS));
            interruptedAt.set(System.nanoTime());
            writing.interrupt();
            writing.join(60_000);

            assertNotNull("write did not finish", finishedAt.get());
            long stopMillis = TimeUnit.NANOSECONDS.toMillis(finishedAt.get() - interruptedAt.get());
            assertTrue("interrupted write took " + stopMillis + " ms to stop and supplied "
                            + supplied.get() + " of " + SIZE + " records",
                    stopMillis <= MAX_STOP_MILLIS && supplied.get() < SIZE);
            assertNotNull("interrupted write reported success", failure.get());
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test(timeout = 60_000)
    public void laterWorkerFailureStopsQueuedRangesBeforeEarlierTaskFinishes() throws Exception {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var list = new ArrayList<VectorFloat<?>>();
        for (int n = 0; n < 512; n++) list.add(vts.createFloatVector(new float[] {n + 1f, 1f}));
        var values = new ListRandomAccessVectorValues(list, 2);
        int tasks = Math.min(Runtime.getRuntime().availableProcessors(), values.size());
        // An earlier blocked task and a later failed task require at least two ranges.
        org.junit.Assume.assumeTrue(tasks > 1);
        int chunk = (values.size() + tasks - 1) / tasks;
        int submittedTasks = (values.size() + chunk - 1) / chunk;
        var submitted = new CountDownLatch(submittedTasks);
        var workers = new ThreadPoolExecutor(2, 2, 0, TimeUnit.MILLISECONDS, new LinkedBlockingQueue<>()) {
            @Override public void execute(Runnable task) {
                super.execute(task);
                submitted.countDown();
            }
        };
        var caller = Executors.newSingleThreadExecutor();
        var release = new CountDownLatch(1);
        var blocked = new CountDownLatch(1);
        var failed = new CountDownLatch(1);
        var laterRecords = new AtomicInteger();
        var path = Files.createTempFile("parallel-later-failure", ".idx");
        try (var builder = new GraphIndexBuilder(values, VectorSimilarityFunction.DOT_PRODUCT,
                4, 16, 1.2f, 1.2f, false)) {
            var graph = TestUtil.buildSequentially(builder, values);
            try (var out = new io.github.jbellis.jvector.disk.BufferedRandomAccessWriter(path);
                 var writer = new ParallelGraphWriter(out, graph, java.util.List.of(new InlineVectors(2)),
                         new ParallelGraphWriter.Config(2, false, 1, 64), path, workers)) {
                var writing = caller.submit(() -> {
                    writer.writeL0Records(new OrdinalMapper.IdentityMapper(values.size() - 1),
                            java.util.List.of(new InlineVectors(2)),
                            Feature.singleStateFactory(FeatureId.INLINE_VECTORS, ordinal -> {
                                if (ordinal == 0) {
                                    blocked.countDown();
                                    try { if (!release.await(20, TimeUnit.SECONDS)) throw new AssertionError("release timeout"); }
                                    catch (InterruptedException e) { throw new AssertionError(e); }
                                } else if (ordinal == chunk) {
                                    try { if (!blocked.await(20, TimeUnit.SECONDS)) throw new AssertionError("first task did not block"); }
                                    catch (InterruptedException e) { throw new AssertionError(e); }
                                    failed.countDown();
                                    throw new IllegalStateException("later worker failure");
                                } else if (failed.getCount() == 0) {
                                    laterRecords.incrementAndGet();
                                }
                                return new InlineVectors.State(values.getVector(ordinal));
                            }), 0);
                    return null;
                });
                assertTrue(failed.await(20, TimeUnit.SECONDS));
                assertTrue(submitted.await(20, TimeUnit.SECONDS));
                // The free worker reaches this sentinel only after draining the queued ranges.
                assertEquals(42, workers.submit(() -> 42).get(20, TimeUnit.SECONDS).intValue());
                assertEquals("queued ranges encoded after another worker failed", 0, laterRecords.get());
                assertFalse("writer released resources while the first supplier was still running", writing.isDone());
                release.countDown();
                var error = assertThrows(ExecutionException.class, () -> writing.get(20, TimeUnit.SECONDS));
                assertTrue(error.getCause() instanceof IllegalStateException);
                assertEquals("later worker failure", error.getCause().getMessage());
                assertEquals(42, workers.submit(() -> 42).get().intValue());
            }
        } finally {
            release.countDown();
            caller.shutdownNow(); workers.shutdownNow();
            caller.awaitTermination(20, TimeUnit.SECONDS); workers.awaitTermination(20, TimeUnit.SECONDS);
            Files.deleteIfExists(path);
        }
    }
}
