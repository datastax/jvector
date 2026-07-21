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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/// A [ReaderSupplier] wrapper that instruments every read a compaction worker performs against a
/// source graph, so a test can determine — deterministically — whether `compact()` can unwind
/// while source reads are still in flight. It originally proved the straggler-abandonment defect
/// (see `local/memory_safety_design_brief.md`, T4); since the F1 drain-on-unwind fix it guards
/// the fixed invariant: `compact()` returns or throws only after every worker read has finished.
///
/// Two roles drive the choreography, coordinated through a shared [Gate]:
///
/// - [Role#PARK]: the first worker-thread read against this source parks (before touching the
///   delegate) until the gate is released. This pins a provably in-flight source read across the
///   moment `compact()` throws.
/// - [Role#POISON]: once a reader has parked, the next worker-thread read against this source on
///   a *different* thread throws, failing that merge batch. The orchestrator sees the failure at
///   `ExecutorCompletionService.take().get()` and `compact()` unwinds — while the parked read (and
///   any other in-flight batches) are still executing.
///
/// Gating applies only on threads named with [Gate#WORKER_PREFIX] and only after [Gate#arm()], so
/// graph loading and orchestrator-side reads pass through untouched.
public final class GatedReaderSupplier implements ReaderSupplier {

    /// What this source's reads do once the gate is armed.
    public enum Role {
        NONE, PARK, POISON
    }

    /// Shared coordination state between the parked source, the poisoned source, and the test.
    public static final class Gate {
        public static final String WORKER_PREFIX = "repro-worker-";

        final CountDownLatch parked = new CountDownLatch(1);
        final CountDownLatch release = new CountDownLatch(1);
        final AtomicReference<Thread> parkedThread = new AtomicReference<>();
        final AtomicBoolean poisonFired = new AtomicBoolean();
        final AtomicInteger activeWorkerReads = new AtomicInteger();
        final AtomicInteger workerReadsCompletedAfterMark = new AtomicInteger();
        final AtomicBoolean parkedReadResumedAfterMark = new AtomicBoolean();
        volatile boolean armed;
        volatile boolean marked;

        /// Starts gating worker reads; call after sources are loaded, right before `compact()`.
        public void arm() {
            armed = true;
        }

        /// Records the observation point (typically: `compact()` just threw); reads that complete
        /// after this are counted as post-unwind stragglers.
        public void mark() {
            marked = true;
        }

        /// True once a worker read is parked inside the wrapped source.
        public boolean awaitParked(long timeout, TimeUnit unit) throws InterruptedException {
            return parked.await(timeout, unit);
        }

        /// Lets the parked read proceed into the delegate.
        public void releaseParked() {
            release.countDown();
        }

        /// Worker reads currently inside a wrapped read call (parked ones included).
        public int activeWorkerReads() {
            return activeWorkerReads.get();
        }

        /// Worker reads that ran to completion after [#mark()].
        public int workerReadsCompletedAfterMark() {
            return workerReadsCompletedAfterMark.get();
        }

        /// True if the parked read resumed and finished after [#mark()].
        public boolean parkedReadResumedAfterMark() {
            return parkedReadResumedAfterMark.get();
        }

        public Thread parkedThread() {
            return parkedThread.get();
        }

        static boolean onWorkerThread() {
            return Thread.currentThread().getName().startsWith(WORKER_PREFIX);
        }
    }

    /// Gate choreography for the post-F1 world: waits for a read to park, holds it briefly (so a
    /// concurrently injected failure provably begins the unwind while the read is in flight),
    /// then releases it. The F1 drain in `compact()` blocks on the parked read until this fires,
    /// so the read completes strictly before `compact()` throws.
    public static final class AutoReleaser implements Runnable {
        private final Gate gate;
        private final long holdMillis;

        public AutoReleaser(Gate gate, long holdMillis) {
            this.gate = gate;
            this.holdMillis = holdMillis;
        }

        @Override
        public void run() {
            try {
                if (gate.awaitParked(60, TimeUnit.SECONDS)) {
                    Thread.sleep(holdMillis);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                gate.releaseParked();
            }
        }
    }

    private final ReaderSupplier delegate;
    private final Gate gate;
    private final Role role;

    public GatedReaderSupplier(ReaderSupplier delegate, Gate gate, Role role) {
        this.delegate = delegate;
        this.gate = gate;
        this.role = role;
    }

    @Override
    public RandomAccessReader get() throws IOException {
        return new GatedReader(delegate.get());
    }

    @Override
    public void close() throws IOException {
        delegate.close();
    }

    private interface IoOp {
        void run() throws IOException;
    }

    private final class GatedReader implements RandomAccessReader {
        private final RandomAccessReader in;

        GatedReader(RandomAccessReader in) {
            this.in = in;
        }

        private void gated(IoOp op) throws IOException {
            if (!gate.armed || !Gate.onWorkerThread()) {
                op.run();
                return;
            }
            gate.activeWorkerReads.incrementAndGet();
            try {
                applyRole();
                op.run();
                if (gate.marked) {
                    gate.workerReadsCompletedAfterMark.incrementAndGet();
                    if (Thread.currentThread() == gate.parkedThread.get()) {
                        gate.parkedReadResumedAfterMark.set(true);
                    }
                }
            } finally {
                gate.activeWorkerReads.decrementAndGet();
            }
        }

        private void applyRole() {
            if (role == Role.PARK) {
                if (gate.parkedThread.compareAndSet(null, Thread.currentThread())) {
                    gate.parked.countDown();
                    try {
                        // Backstop only: the choreography (AutoReleaser) releases the gate long
                        // before this; with the F1 drain, compact() blocks until it does.
                        if (!gate.release.await(15, TimeUnit.SECONDS)) {
                            throw new IllegalStateException("repro gate was never released");
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException("parked repro read interrupted", e);
                    }
                }
            } else if (role == Role.POISON) {
                if (gate.parked.getCount() == 0
                        && Thread.currentThread() != gate.parkedThread.get()
                        && gate.poisonFired.compareAndSet(false, true)) {
                    throw new RuntimeException("REPRO_POISON: injected merge batch failure");
                }
            }
        }

        @Override
        public void seek(long offset) throws IOException {
            in.seek(offset);
        }

        @Override
        public long getPosition() throws IOException {
            return in.getPosition();
        }

        @Override
        public int readInt() throws IOException {
            int[] out = new int[1];
            gated(() -> out[0] = in.readInt());
            return out[0];
        }

        @Override
        public float readFloat() throws IOException {
            float[] out = new float[1];
            gated(() -> out[0] = in.readFloat());
            return out[0];
        }

        @Override
        public long readLong() throws IOException {
            long[] out = new long[1];
            gated(() -> out[0] = in.readLong());
            return out[0];
        }

        @Override
        public void readFully(byte[] bytes) throws IOException {
            gated(() -> in.readFully(bytes));
        }

        @Override
        public void readFully(ByteBuffer buffer) throws IOException {
            gated(() -> in.readFully(buffer));
        }

        @Override
        public void readFully(float[] floats) throws IOException {
            gated(() -> in.readFully(floats));
        }

        @Override
        public void readFully(long[] vector) throws IOException {
            gated(() -> in.readFully(vector));
        }

        @Override
        public void read(int[] ints, int offset, int count) throws IOException {
            gated(() -> in.read(ints, offset, count));
        }

        @Override
        public void read(float[] floats, int offset, int count) throws IOException {
            gated(() -> in.read(floats, offset, count));
        }

        @Override
        public void close() throws IOException {
            in.close();
        }

        @Override
        public long length() throws IOException {
            return in.length();
        }
    }
}
