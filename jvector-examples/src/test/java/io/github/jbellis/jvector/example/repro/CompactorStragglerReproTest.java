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

package io.github.jbellis.jvector.example.repro;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor;
import io.github.jbellis.jvector.util.work.ProgressLimiter;
import io.github.jbellis.jvector.util.work.WorkStage;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.Assume;
import org.junit.Test;

import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/// Guard tests for the F1 drain-on-unwind fix in `OnDiskGraphIndexCompactor` (prescription:
/// `local/memory_safety_fix_plan.md` F1; the defect these tests originally *proved* is documented
/// with its evidence in `local/memory_safety_design_brief.md` §4): when a merge batch fails or the
/// orchestrating thread is interrupted, `compact()` must block until every in-flight batch has
/// finished before the exception escapes. Otherwise control returns to a caller who is entitled
/// to release the source graphs while worker tasks are still reading them — natively fatal for
/// NIO-mapped sources; the pre-fix form of T4b reproduced the production core dump exactly that
/// way (SIGSEGV/`SEGV_MAPERR` in the merge read lineage).
///
/// - T4a checks the drained unwind deterministically, in-process: a poisoned source read fails
///   one batch while another worker's source read is provably parked in flight; `compact()` must
///   not throw until that read has completed.
/// - T4b runs the production sequence in a child JVM: after `compact()` throws, a
///   contract-compliant host closes its host-style mapped sources — which must now be crash-free.
/// - T5 interrupts the orchestrator mid-refine (the realistic host trigger: nodetool stop /
///   shutdown / cancellation) while one refine task is provably parked in flight; `compact()`
///   must not throw until that task has finished.
/// - T5b (the F5 fix) interrupts the orchestrator during the fused pre-encode fan-out
///   (`invokeAll` on the strategies' executor adapter) while one pre-encode task is provably
///   parked in flight. Stock `invokeAll` cancels-with-interrupt and returns immediately, so
///   pre-F5 that task — which reads source graphs and writes the code cache that
///   `onAfterClose` unmaps — outlived `compact()`.
public class CompactorStragglerReproTest {

    private static void assumeLinux() {
        Assume.assumeTrue("child-JVM crash detection relies on Linux mmap semantics",
                System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("linux"));
    }

    private static void verdict(String theory, String text) {
        System.out.println("VERDICT|" + theory + "|" + text);
    }

    private static ThreadFactory workerFactory() {
        return new WorkerThreadFactory();
    }

    private static final class WorkerThreadFactory implements ThreadFactory {
        private final AtomicInteger n = new AtomicInteger();

        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r, GatedReaderSupplier.Gate.WORKER_PREFIX + n.getAndIncrement());
            t.setDaemon(true);
            return t;
        }
    }

    private static boolean chainContains(Throwable t, String needle) {
        for (Throwable c = t; c != null; c = c.getCause()) {
            if (String.valueOf(c.getMessage()).contains(needle) || c.getClass().getName().contains(needle)) {
                return true;
            }
        }
        return false;
    }

    private static String chain(Throwable t) {
        StringBuilder sb = new StringBuilder();
        for (Throwable c = t; c != null; c = c.getCause()) {
            if (sb.length() > 0) {
                sb.append(" <- ");
            }
            sb.append(c.getClass().getSimpleName()).append('(').append(c.getMessage()).append(')');
        }
        return sb.toString();
    }

    /// F1 guard: `compact()` must not return control while source reads are in flight. One batch
    /// is poisoned while another worker's source read is parked — provably in flight — and the
    /// drain must complete that read (released by the [GatedReaderSupplier.AutoReleaser] ~1s
    /// later) before the failure escapes. Pre-F1 this test proved the inverse: `compact()` threw
    /// with 3 source reads still executing and thousands more completing after the unwind.
    @Test(timeout = 240_000)
    public void t4a_compactDrainsInflightSourceReadsBeforeUnwinding() throws Exception {
        Path dir = ReproGraphs.newWorkDir("t4a-straggler");
        final int dimension = 24;
        final int perSource = 220;
        Path s0 = ReproGraphs.buildInlineGraph(dir.resolve("s0.graph"), ReproGraphs.randomVectors(perSource, dimension, 301), dimension);
        Path s1 = ReproGraphs.buildInlineGraph(dir.resolve("s1.graph"), ReproGraphs.randomVectors(perSource, dimension, 302), dimension);

        try (ReaderSupplier r0 = ReaderSupplierFactory.open(s0);
             ReaderSupplier r1 = ReaderSupplierFactory.open(s1)) {
            GatedReaderSupplier.Gate gate = new GatedReaderSupplier.Gate();
            GatedReaderSupplier g0 = new GatedReaderSupplier(r0, gate, GatedReaderSupplier.Role.POISON);
            GatedReaderSupplier g1 = new GatedReaderSupplier(r1, gate, GatedReaderSupplier.Role.PARK);
            OnDiskGraphIndex src0 = OnDiskGraphIndex.load(g0);
            OnDiskGraphIndex src1 = OnDiskGraphIndex.load(g1);

            ExecutorService pool = Executors.newFixedThreadPool(4, workerFactory());
            try {
                var compactor = new OnDiskGraphIndexCompactor(
                        List.of(src0, src1),
                        ReproGraphs.allLive(perSource, perSource),
                        ReproGraphs.stackedRemappers(perSource, perSource),
                        VectorSimilarityFunction.EUCLIDEAN,
                        pool, 4);

                gate.arm();
                Thread releaser = new Thread(new GatedReaderSupplier.AutoReleaser(gate, 1_000), "repro-gate-releaser");
                releaser.setDaemon(true);
                releaser.start();

                Throwable thrown = null;
                int activeReadsAtThrow = -1;
                try {
                    compactor.compact(dir.resolve("out.graph"));
                } catch (Throwable t) {
                    gate.mark();
                    activeReadsAtThrow = gate.activeWorkerReads();
                    thrown = t;
                }

                assertNotNull("the poisoned batch must make compact() throw", thrown);
                assertTrue("compact() must have failed because of the injected poison, but failed with: " + chain(thrown),
                        chainContains(thrown, "REPRO_POISON"));
                assertNotNull("the repro choreography must have parked a source read across the failure",
                        gate.parkedThread());
                assertEquals("F1: compact() must drain every in-flight source read before unwinding",
                        0, activeReadsAtThrow);
                assertFalse("the parked read must complete before compact() throws, never after",
                        gate.parkedReadResumedAfterMark());

                pool.shutdown();
                assertTrue("workers must quiesce", pool.awaitTermination(120, TimeUnit.SECONDS));
                assertEquals("no source read may complete after compact() has thrown",
                        0, gate.workerReadsCompletedAfterMark());
                verdict("T4a", "FIX HOLDS: compact() drained all in-flight source reads before unwinding"
                        + " (0 active at throw, 0 completed after)");
            } finally {
                gate.releaseParked();
                pool.shutdownNow();
            }
        }
    }

    /// F1 guard: the production sequence must be crash-free end to end. In a child JVM, after
    /// `compact()` throws, the host — correctly, per the documented contract — closes its
    /// host-style mapped sources (a raw munmap). The drain guarantees nothing is still reading
    /// them, so the child must reach the orderly NO_CRASH outcome (exit 45). Pre-F1 this exact
    /// sequence died with SIGSEGV/`SEGV_MAPERR` on a worker thread inside the merge read lineage
    /// — the observed production core dump.
    @Test(timeout = 360_000)
    public void t4b_contractCompliantSourceCloseAfterCompactThrowsIsCrashFree() throws Exception {
        assumeLinux();
        Path dir = ReproGraphs.newWorkDir("t4b-straggler-crash");
        ChildJvm.Result r = ChildJvm.run(dir, 300, "compactor-straggler", dir.toString());

        assertTrue("child must reach the compact() failure: " + r.summary(), r.output.contains("STAGE|COMPACT_THREW"));
        assertNotEquals("compact() unexpectedly succeeded despite the poisoned batch", 44, r.exitCode);
        assertTrue("the drain must leave zero in-flight source reads at the throw: " + r.summary(),
                r.output.contains("ACTIVE_WORKER_READS=0"));
        assertTrue("child must close sources only after compact() threw: " + r.summary(),
                r.output.contains("STAGE|SOURCES_CLOSED"));
        assertFalse("F1: contract-compliant source close after compact() throws must be crash-free: " + r.summary(),
                r.crashed());
        assertEquals("child must survive to the orderly no-crash outcome: " + r.summary(), 45, r.exitCode);
        assertTrue("no source read may complete after the unwind: " + r.summary(),
                r.output.contains("readsCompletedAfterThrow=0"));
        verdict("T4b", "FIX HOLDS: contract-compliant close after compact() threw is crash-free: " + r.summary());
    }

    /// F1 guard: the refine window loop must drain its in-flight tasks before an interrupt-driven
    /// unwind escapes. One refine task is parked — provably in flight — when the orchestrator is
    /// interrupted; the 1s hold before release is the regression window in which the pre-F1 code
    /// threw (abandoning the parked task and up to a window of others, which then raced the
    /// unwind's closing of the output supplier and channel). Post-F1, `compact()` must not throw
    /// until the parked task has finished.
    @Test(timeout = 600_000)
    public void t5_refineLoopDrainsInflightTasksOnInterrupt() throws Exception {
        Path dir = ReproGraphs.newWorkDir("t5-refine-interrupt");
        final int dimension = 16;
        final int perSource = 1000;
        Path s0 = ReproGraphs.buildInlineGraph(dir.resolve("s0.graph"), ReproGraphs.randomVectors(perSource, dimension, 401), dimension);
        Path s1 = ReproGraphs.buildInlineGraph(dir.resolve("s1.graph"), ReproGraphs.randomVectors(perSource, dimension, 402), dimension);

        try (ReaderSupplier r0 = ReaderSupplierFactory.open(s0);
             ReaderSupplier r1 = ReaderSupplierFactory.open(s1)) {
            OnDiskGraphIndex src0 = OnDiskGraphIndex.load(r0);
            OnDiskGraphIndex src1 = OnDiskGraphIndex.load(r1);

            ExecutorService pool = Executors.newFixedThreadPool(4, workerFactory());
            CountingExecutor counting = new CountingExecutor(pool);
            try {
                var compactor = new OnDiskGraphIndexCompactor(
                        List.of(src0, src1),
                        ReproGraphs.allLive(perSource, perSource),
                        ReproGraphs.stackedRemappers(perSource, perSource),
                        VectorSimilarityFunction.EUCLIDEAN,
                        counting, 4);
                RefineSignal refineSignal = new RefineSignal();
                compactor.setProgressLimiter(refineSignal);
                // Refinement is opt-in (default off); this F1 guard parks a refine task and
                // interrupts mid-refine, so the refinement pass must be enabled explicitly.
                compactor.setRefineAfterCompaction(true);

                AtomicReference<Throwable> thrown = new AtomicReference<>();
                AtomicInteger inFlightAtThrow = new AtomicInteger(-1);
                AtomicLong thrownAtNanos = new AtomicLong();
                Thread orchestrator = new Thread(
                        new CompactRun(compactor, dir.resolve("out.graph"), counting, thrown, inFlightAtThrow, thrownAtNanos),
                        "repro-orchestrator");
                orchestrator.start();

                assertTrue("refine phase must begin (merge finished)", refineSignal.refineStarted.await(480, TimeUnit.SECONDS));
                counting.armParkOnce();
                assertTrue("a refine task must park in flight", counting.awaitParked(120, TimeUnit.SECONDS));
                orchestrator.interrupt();
                // Regression window: pre-F1, compact() throws during this hold with the parked
                // task still in flight; post-F1 the drain blocks on it instead.
                Thread.sleep(1_000);
                counting.releaseParked();

                orchestrator.join(TimeUnit.SECONDS.toMillis(120));
                assertFalse("orchestrator must unwind after interrupt", orchestrator.isAlive());
                assertNotNull("interrupt during refine must make compact() throw", thrown.get());
                assertTrue("unwind must be caused by the interrupt, was: " + chain(thrown.get()),
                        chainContains(thrown.get(), "InterruptedException"));

                pool.shutdown();
                assertTrue("workers must quiesce", pool.awaitTermination(120, TimeUnit.SECONDS));

                long graceNanos = TimeUnit.MILLISECONDS.toNanos(250);
                long parkedEndMinusThrow = counting.parkedTaskEndNanos() - thrownAtNanos.get();
                assertTrue("F1: compact() must not throw while a refine task is in flight"
                                + " (parked task ended " + TimeUnit.NANOSECONDS.toMillis(parkedEndMinusThrow)
                                + "ms after the throw; expected at/before it)",
                        counting.parkedTaskEndNanos() <= thrownAtNanos.get() + graceNanos);
                verdict("T5", "FIX HOLDS: refine unwind drained in-flight tasks; parked task finished "
                        + TimeUnit.NANOSECONDS.toMillis(Math.max(0, -parkedEndMinusThrow))
                        + "ms before compact() threw (inFlightAtThrow=" + inFlightAtThrow.get() + ")");
            } finally {
                counting.releaseParked();
                pool.shutdownNow();
            }
        }
    }

    /// F5 guard: the strategies' `invokeAll` fan-out — fused pre-encode here — must drain its
    /// started tasks before an interrupt-driven unwind escapes `compact()`. Pre-encode tasks
    /// open source views (`getVectorInto` per live node) and write the pre-encode code cache
    /// that `compact()`'s finally unmaps, so pre-F5 abandonment was doubly fatal: the caller
    /// could unmap sources under a live read, and jvector itself unmapped the cache under a
    /// live write. The parked task ignores interrupts, exactly like real encode work, so stock
    /// `invokeAll`'s cancel-with-interrupt cannot end it — only a real drain can.
    @Test(timeout = 600_000)
    public void t5b_preEncodeFanoutDrainsInflightTasksOnInterrupt() throws Exception {
        Path dir = ReproGraphs.newWorkDir("t5b-preencode-interrupt");
        final int dimension = 32;
        final int perSource = 256;
        Path s0 = ReproGraphs.buildFusedGraph(dir.resolve("s0.graph"), ReproGraphs.randomVectors(perSource, dimension, 501), dimension);
        Path s1 = ReproGraphs.buildFusedGraph(dir.resolve("s1.graph"), ReproGraphs.randomVectors(perSource, dimension, 502), dimension);

        try (ReaderSupplier r0 = ReaderSupplierFactory.open(s0);
             ReaderSupplier r1 = ReaderSupplierFactory.open(s1)) {
            OnDiskGraphIndex src0 = OnDiskGraphIndex.load(r0);
            OnDiskGraphIndex src1 = OnDiskGraphIndex.load(r1);

            ExecutorService pool = Executors.newFixedThreadPool(4, workerFactory());
            CountingExecutor counting = new CountingExecutor(pool);
            try {
                var compactor = new OnDiskGraphIndexCompactor(
                        List.of(src0, src1),
                        ReproGraphs.allLive(perSource, perSource),
                        ReproGraphs.stackedRemappers(perSource, perSource),
                        VectorSimilarityFunction.COSINE,
                        counting, 4);

                AtomicReference<Throwable> thrown = new AtomicReference<>();
                AtomicInteger inFlightAtThrow = new AtomicInteger(-1);
                AtomicLong thrownAtNanos = new AtomicLong();
                // In a fused compaction the first task through the executor is a pre-encode
                // chunk (strategy.onAfterHeader -> precomputeCodes -> invokeAll, before any
                // merge batch), so arming the one-shot park before compact() pins a pre-encode
                // task in flight.
                counting.armParkOnce();
                Thread orchestrator = new Thread(
                        new CompactRun(compactor, dir.resolve("out.graph"), counting, thrown, inFlightAtThrow, thrownAtNanos),
                        "repro-orchestrator");
                orchestrator.start();

                assertTrue("a pre-encode task must park in flight", counting.awaitParked(120, TimeUnit.SECONDS));
                orchestrator.interrupt();
                // Regression window: with stock invokeAll, compact() throws during this hold —
                // cancel-with-interrupt does not end the parked task; with the F5 drain it
                // blocks until the release below.
                Thread.sleep(1_000);
                counting.releaseParked();

                orchestrator.join(TimeUnit.SECONDS.toMillis(120));
                assertFalse("orchestrator must unwind after interrupt", orchestrator.isAlive());
                assertNotNull("interrupt during pre-encode must make compact() throw", thrown.get());
                assertTrue("unwind must be caused by the interrupt, was: " + chain(thrown.get()),
                        chainContains(thrown.get(), "InterruptedException"));

                pool.shutdown();
                assertTrue("workers must quiesce", pool.awaitTermination(120, TimeUnit.SECONDS));

                long graceNanos = TimeUnit.MILLISECONDS.toNanos(250);
                long parkedEndMinusThrow = counting.parkedTaskEndNanos() - thrownAtNanos.get();
                assertTrue("F5: compact() must not throw while a pre-encode task is in flight"
                                + " (parked task ended " + TimeUnit.NANOSECONDS.toMillis(parkedEndMinusThrow)
                                + "ms after the throw; expected at/before it)",
                        counting.parkedTaskEndNanos() <= thrownAtNanos.get() + graceNanos);
                verdict("T5b", "FIX HOLDS: pre-encode fan-out drained before unwind; parked task finished "
                        + TimeUnit.NANOSECONDS.toMillis(Math.max(0, -parkedEndMinusThrow))
                        + "ms before compact() threw (inFlightAtThrow=" + inFlightAtThrow.get() + ")");
            } finally {
                counting.releaseParked();
                pool.shutdownNow();
            }
        }
    }

    /// Runs `compact()` and records the unwind instant plus how many pool tasks were mid-flight.
    private static final class CompactRun implements Runnable {
        private final OnDiskGraphIndexCompactor compactor;
        private final Path out;
        private final CountingExecutor counting;
        private final AtomicReference<Throwable> thrown;
        private final AtomicInteger inFlightAtThrow;
        private final AtomicLong thrownAtNanos;

        CompactRun(OnDiskGraphIndexCompactor compactor, Path out, CountingExecutor counting,
                   AtomicReference<Throwable> thrown, AtomicInteger inFlightAtThrow, AtomicLong thrownAtNanos) {
            this.compactor = compactor;
            this.out = out;
            this.counting = counting;
            this.thrown = thrown;
            this.inFlightAtThrow = inFlightAtThrow;
            this.thrownAtNanos = thrownAtNanos;
        }

        @Override
        public void run() {
            try {
                compactor.compact(out);
            } catch (Throwable t) {
                thrownAtNanos.set(System.nanoTime());
                inFlightAtThrow.set(counting.inFlight());
                thrown.set(t);
            }
        }
    }

    /// Signals when the compactor reports the first REFINE progress event (orchestrator-side).
    private static final class RefineSignal implements ProgressLimiter {
        final CountDownLatch refineStarted = new CountDownLatch(1);

        @Override
        public void onProgress(WorkStage stage, long completed, long total) {
            if (stage == OnDiskGraphIndexCompactor.Phase.REFINE) {
                refineStarted.countDown();
            }
        }
    }

    /// Wraps the pool so the test can observe compactor-submitted tasks — how many are mid-flight
    /// and when each finished — and, for the F1 guard, park a single task (one-shot) provably in
    /// flight until released.
    private static final class CountingExecutor implements Executor {
        private final Executor delegate;
        private final AtomicInteger inFlight = new AtomicInteger();
        private final AtomicLong lastTaskEndNanos = new AtomicLong();
        private final AtomicBoolean parkArmed = new AtomicBoolean();
        private final CountDownLatch parkedLatch = new CountDownLatch(1);
        private final CountDownLatch parkRelease = new CountDownLatch(1);
        private final AtomicLong parkedTaskEndNanos = new AtomicLong();

        CountingExecutor(Executor delegate) {
            this.delegate = delegate;
        }

        int inFlight() {
            return inFlight.get();
        }

        long lastTaskEndNanos() {
            return lastTaskEndNanos.get();
        }

        long parkedTaskEndNanos() {
            return parkedTaskEndNanos.get();
        }

        /// Arms the one-shot park: the next task to start executing parks until [#releaseParked()].
        void armParkOnce() {
            parkArmed.set(true);
        }

        boolean awaitParked(long timeout, TimeUnit unit) throws InterruptedException {
            return parkedLatch.await(timeout, unit);
        }

        void releaseParked() {
            parkRelease.countDown();
        }

        @Override
        public void execute(Runnable command) {
            delegate.execute(new CountedTask(command));
        }

        private final class CountedTask implements Runnable {
            private final Runnable command;

            CountedTask(Runnable command) {
                this.command = command;
            }

            @Override
            public void run() {
                boolean parkedTask = parkArmed.compareAndSet(true, false);
                inFlight.incrementAndGet();
                try {
                    if (parkedTask) {
                        parkedLatch.countDown();
                        // Uninterruptible on purpose: this models a real merge/encode chunk —
                        // CPU + mmap work that never polls the interrupt flag — so a
                        // cancel-with-interrupt (stock invokeAll's unwind behavior) cannot make
                        // it "finish" early. Interrupts are remembered and re-asserted.
                        boolean sawInterrupt = false;
                        boolean released = false;
                        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(60);
                        while (!released) {
                            long remaining = deadline - System.nanoTime();
                            if (remaining <= 0) {
                                throw new IllegalStateException("parked repro task was never released");
                            }
                            try {
                                released = parkRelease.await(remaining, TimeUnit.NANOSECONDS);
                            } catch (InterruptedException e) {
                                sawInterrupt = true;
                            }
                        }
                        if (sawInterrupt) {
                            Thread.currentThread().interrupt();
                        }
                    }
                    command.run();
                } finally {
                    inFlight.decrementAndGet();
                    long now = System.nanoTime();
                    lastTaskEndNanos.accumulateAndGet(now, Math::max);
                    if (parkedTask) {
                        parkedTaskEndNanos.set(now);
                    }
                }
            }
        }
    }
}
