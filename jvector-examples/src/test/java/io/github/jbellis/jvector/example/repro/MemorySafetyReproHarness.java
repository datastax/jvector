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

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.lang.reflect.Constructor;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/// Child-JVM entry point for the memory-safety reproductions. Every scenario that may end in a
/// native JVM crash runs here, launched by [ChildJvm] from the actual tests, so the crash kills
/// this process instead of the test suite.
///
/// Reader-level scenarios (`reader <kind> <trigger> <workDir>`) spin reader threads over a raw
/// big-endian float file through one of three [ReaderSupplier] implementations, then invalidate
/// the mapping underneath them:
///
/// - kind `arena`: jvector-native's `MemorySegmentReader` (Arena-managed mapping, loaded
///   reflectively exactly like `ReaderSupplierFactory` does)
/// - kind `hoststyle`: [HostStyleMappedReader] (raw NIO mapping, bulk byte-swapping reads, no
///   liveness handshake — models a Cassandra `FileHandle`-style adapter)
/// - kind `simplemapped`: jvector's shipped [SimpleMappedReader] (raw NIO mapping, per-element
///   reads, `invokeCleaner` on supplier close)
///
/// and trigger `close` (clean supplier close / unmap) or `truncate` (shrink the file in place).
///
/// The compactor scenario (`compactor-straggler <workDir>`) exercises the full production stack:
/// a cross-source merge on a worker pool whose batches read sources through [GatedReaderSupplier];
/// one batch is poisoned so `compact()` throws, and a contract-compliant "host" then closes the
/// source suppliers (raw unmap). Before the F1 drain-on-unwind fix this crashed the JVM (the
/// parked straggler read resumed into the dead mapping — SIGSEGV/SEGV_MAPERR, see
/// `local/memory_safety_design_brief.md` T4b); with the fix, `compact()` drains the parked read
/// before throwing and the scenario must end at `OUTCOME|NO_CRASH` (exit 45).
///
/// Exit codes: 0 = readers stopped with a Java exception; 42 = trigger had no effect;
/// 43 = close refused with an exception while reads continued; 44 = compact() returned normally;
/// 45 = no crash after sources closed; 99 = harness failure. A native crash never reaches an
/// orderly exit — the parent detects it via the exit status and the hs_err file.
public final class MemorySafetyReproHarness {
    private static final String ARENA_SUPPLIER_CLASS = "io.github.jbellis.jvector.disk.MemorySegmentReader$Supplier";

    private MemorySafetyReproHarness() {
    }

    public static void main(String[] args) {
        try {
            String mode = args[0];
            int code;
            if ("reader".equals(mode)) {
                code = readerScenario(args[1], args[2], Path.of(args[3]));
            } else if ("compactor-straggler".equals(mode)) {
                code = compactorStragglerScenario(Path.of(args[1]));
            } else {
                throw new IllegalArgumentException("unknown mode: " + mode);
            }
            System.exit(code);
        } catch (Throwable t) {
            t.printStackTrace(System.out);
            stage("FATAL|" + t);
            System.exit(99);
        }
    }

    private static void stage(String message) {
        System.out.println("REPRO|" + message);
        System.out.flush();
    }

    /// True when the arena-based reader (jvector-native, JDK 22+) is loadable in this JVM.
    public static boolean arenaReaderAvailable() {
        try {
            Class.forName(ARENA_SUPPLIER_CLASS);
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    static ReaderSupplier openSupplier(String kind, Path file) throws Exception {
        switch (kind) {
            case "arena": {
                Constructor<?> ctor = Class.forName(ARENA_SUPPLIER_CLASS).getConstructor(Path.class);
                return (ReaderSupplier) ctor.newInstance(file);
            }
            case "hoststyle":
                return new HostStyleMappedReader.Supplier(file);
            case "simplemapped":
                return new SimpleMappedReader.Supplier(file);
            default:
                throw new IllegalArgumentException("unknown reader kind: " + kind);
        }
    }

    // ---------------------------------------------------------------- reader-level scenarios

    private static int readerScenario(String kind, String trigger, Path workDir) throws Exception {
        final int dimension = 64;
        final int count = 100_000;
        Path file = ReproGraphs.writeBigEndianFloats(workDir.resolve("floats.bin"), count, dimension, 42L);
        ReaderSupplier supplier = openSupplier(kind, file);

        final AtomicLong reads = new AtomicLong();
        final List<Throwable> readerErrors = Collections.synchronizedList(new ArrayList<>());
        final int readerCount = 2;
        final CountDownLatch stopped = new CountDownLatch(readerCount);

        for (int i = 0; i < readerCount; i++) {
            final long seed = 1000 + i;
            Thread t = new Thread(new ReaderLoop(supplier, count, dimension, seed, reads, readerErrors, stopped),
                    "repro-reader-" + i);
            t.setDaemon(true);
            t.start();
        }

        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while (reads.get() < 30_000 && System.nanoTime() < deadline && stopped.getCount() == readerCount) {
            Thread.sleep(10);
        }
        stage("WARMED|reads=" + reads.get());

        Throwable closeError = null;
        if ("close".equals(trigger)) {
            try {
                supplier.close();
            } catch (Throwable t) {
                closeError = t;
            }
        } else if ("truncate".equals(trigger)) {
            try (FileChannel ch = FileChannel.open(file, StandardOpenOption.WRITE)) {
                ch.truncate(4096);
            }
        } else {
            throw new IllegalArgumentException("unknown trigger: " + trigger);
        }
        stage("TRIGGERED|" + trigger + (closeError == null ? "" : "|closeThrew=" + closeError.getClass().getName()));

        boolean allStopped = stopped.await(8, TimeUnit.SECONDS);
        if (allStopped) {
            Throwable first = readerErrors.isEmpty() ? null : readerErrors.get(0);
            stage("OUTCOME|SURVIVED_EXCEPTION|" + describe(first));
            return 0;
        }
        long before = reads.get();
        Thread.sleep(300);
        boolean stillFlowing = reads.get() > before;
        if (closeError != null) {
            stage("OUTCOME|CLOSE_REFUSED|" + describe(closeError) + "|readsStillFlowing=" + stillFlowing);
            return 43;
        }
        stage("OUTCOME|NO_EFFECT|readsStillFlowing=" + stillFlowing);
        return 42;
    }

    /// The hot read loop: random-position bulk float reads, the access pattern of a graph search.
    private static final class ReaderLoop implements Runnable {
        private final ReaderSupplier supplier;
        private final int count;
        private final int dimension;
        private final long seed;
        private final AtomicLong reads;
        private final List<Throwable> errors;
        private final CountDownLatch stopped;

        ReaderLoop(ReaderSupplier supplier, int count, int dimension, long seed,
                   AtomicLong reads, List<Throwable> errors, CountDownLatch stopped) {
            this.supplier = supplier;
            this.count = count;
            this.dimension = dimension;
            this.seed = seed;
            this.reads = reads;
            this.errors = errors;
            this.stopped = stopped;
        }

        @Override
        public void run() {
            try {
                RandomAccessReader reader = supplier.get();
                float[] dst = new float[dimension];
                Random rnd = new Random(seed);
                while (true) {
                    reader.seek((long) rnd.nextInt(count) * dimension * Float.BYTES);
                    reader.read(dst, 0, dimension);
                    reads.incrementAndGet();
                }
            } catch (Throwable t) {
                errors.add(t);
            } finally {
                stopped.countDown();
            }
        }
    }

    private static String describe(Throwable t) {
        if (t == null) {
            return "none";
        }
        String msg = String.valueOf(t.getMessage());
        return t.getClass().getName() + ": " + (msg.length() > 160 ? msg.substring(0, 160) : msg);
    }

    // ---------------------------------------------------------------- compactor-level scenario

    /// Exercises the production crash sequence end to end. Sequence markers narrate each step so
    /// the parent can assert ordering around COMPACT_THREW and SOURCES_CLOSED — the actions of a
    /// host that followed the documented lifecycle contract ("keep sources alive until compact()
    /// returns or throws"). Pre-F1 this crashed after SOURCES_CLOSED; post-F1 the drain leaves
    /// nothing in flight at the throw and the scenario must survive to exit 45.
    private static int compactorStragglerScenario(Path workDir) throws Exception {
        final int dimension = 24;
        final int perSource = 220;

        Path s0 = ReproGraphs.buildInlineGraph(workDir.resolve("src0.graph"),
                ReproGraphs.randomVectors(perSource, dimension, 101), dimension);
        Path s1 = ReproGraphs.buildInlineGraph(workDir.resolve("src1.graph"),
                ReproGraphs.randomVectors(perSource, dimension, 202), dimension);
        stage("STAGE|SOURCES_BUILT");

        HostStyleMappedReader.Supplier h0 = new HostStyleMappedReader.Supplier(s0);
        HostStyleMappedReader.Supplier h1 = new HostStyleMappedReader.Supplier(s1);
        GatedReaderSupplier.Gate gate = new GatedReaderSupplier.Gate();
        GatedReaderSupplier g0 = new GatedReaderSupplier(h0, gate, GatedReaderSupplier.Role.POISON);
        GatedReaderSupplier g1 = new GatedReaderSupplier(h1, gate, GatedReaderSupplier.Role.PARK);

        OnDiskGraphIndex src0 = OnDiskGraphIndex.load(g0);
        OnDiskGraphIndex src1 = OnDiskGraphIndex.load(g1);

        ExecutorService pool = Executors.newFixedThreadPool(4, new WorkerThreadFactory());
        var compactor = new OnDiskGraphIndexCompactor(
                List.of(src0, src1),
                ReproGraphs.allLive(perSource, perSource),
                ReproGraphs.stackedRemappers(perSource, perSource),
                VectorSimilarityFunction.EUCLIDEAN,
                pool, 4);

        gate.arm();
        // Releases the parked read ~1s after it parks: long enough that the poison provably
        // fires (and the unwind begins) while the read is in flight, short enough that the F1
        // drain — which blocks compact() on the parked read — completes promptly.
        Thread releaser = new Thread(new GatedReaderSupplier.AutoReleaser(gate, 1_000), "repro-gate-releaser");
        releaser.setDaemon(true);
        releaser.start();
        stage("STAGE|COMPACT_START");
        try {
            compactor.compact(workDir.resolve("out.graph"));
            stage("OUTCOME|COMPACT_RETURNED_NORMALLY");
            return 44;
        } catch (Throwable t) {
            gate.mark();
            stage("STAGE|COMPACT_THREW|" + rootChain(t));
        }

        stage("STAGE|PARKED=" + (gate.parkedThread() != null) + "|ACTIVE_WORKER_READS=" + gate.activeWorkerReads());

        // The contract-compliant host: compact() has thrown, so the caller now releases its
        // sources. For a FileHandle-style mapping that release is a raw munmap.
        g0.close();
        g1.close();
        stage("STAGE|SOURCES_CLOSED");

        gate.releaseParked();
        stage("STAGE|GATE_RELEASED");

        // Pre-F1 the abandoned straggler resumed into the unmapped pages here and the JVM died
        // within this window; with the drain in place nothing is in flight and it must pass
        // quietly.
        for (int i = 0; i < 5; i++) {
            Thread.sleep(1_000);
        }
        stage("OUTCOME|NO_CRASH|activeWorkerReads=" + gate.activeWorkerReads()
                + "|readsCompletedAfterThrow=" + gate.workerReadsCompletedAfterMark());
        return 45;
    }

    /// Names pool threads with the prefix [GatedReaderSupplier.Gate#WORKER_PREFIX] so the gates
    /// only engage on merge workers, never on the orchestrator or the graph-loading main thread.
    private static final class WorkerThreadFactory implements ThreadFactory {
        private final AtomicInteger n = new AtomicInteger();

        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r, GatedReaderSupplier.Gate.WORKER_PREFIX + n.getAndIncrement());
            t.setDaemon(true);
            return t;
        }
    }

    private static String rootChain(Throwable t) {
        StringBuilder sb = new StringBuilder();
        for (Throwable c = t; c != null; c = c.getCause()) {
            if (sb.length() > 0) {
                sb.append(" <- ");
            }
            sb.append(c.getClass().getSimpleName());
        }
        return sb.toString();
    }
}
