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

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.locks.LockSupport;

/**
 * Synchronously combines work from concurrent callers, using only their threads.
 * The strict constructor permits at most {@code maxLeaders} operations at once;
 * {@link #opportunistic} additionally supplies a direct pass-through budget.
 * An available
 * leader starts immediately; it never waits for a batch to fill. Waiting requests
 * are taken in FIFO order by a caller that acquires free capacity. Capacity is
 * never reserved for a parked caller. Reused request records are published
 * without a blocking queue lock; only batch removal needs brief exclusion.
 *
 * <p>Input and output belong to the submitting caller but must be accessible to
 * other threads until {@link #process} returns. The operation must not retain
 * them or reenter this processor. A batch finishes before any of its callers
 * returns; subsequent application work is entirely outside this processor.
 * Interrupts are preserved, but do not abandon an input that another caller may
 * already be using. A failed operation fails its batch and still hands off work.
 *
 * <p>Request slots, batch lists and callbacks are reused. This class creates no
 * executor, worker, completion future, timer, or per-operation request object.
 * A slot is allocated on each participating thread's first use of this instance.
 */
public final class CallerBatchProcessor<I, O> implements Accountable {
    /** An operation that writes caller-owned outputs before returning. */
    public interface Operation<I, O> {
        /** Handles a single input without wrapping it in a batch. */
        void apply(I input, O output) throws Exception;

        /** Borrowed read-only lists; neither the lists nor their rows may escape. */
        void applyBatch(List<I> inputs, List<O> outputs) throws Exception;
    }

    private static final int WAITING = 0, ASSIGNED = 1, DONE = 2;
    private static final int CLAIM_BUSY = 0, CLAIM_SKIPPED = 1, CLAIM_READY = 2;
    private static final VarHandle INCOMING, DRAINING;
    static {
        try {
            var lookup = MethodHandles.lookup();
            INCOMING = lookup.findVarHandle(CallerBatchProcessor.class, "incoming", Request.class);
            DRAINING = lookup.findVarHandle(CallerBatchProcessor.class, "draining", int.class);
        } catch (ReflectiveOperationException failure) {
            throw new ExceptionInInitializerError(failure);
        }
    }
    private final int maxBatchSize;
    private final Operation<I, O> operation;
    private final Context<I, O>[] contexts;
    private final DirectSlot[] directSlots;
    private final ThreadLocal<Request<I, O>> local = ThreadLocal.withInitial(Request::new);
    // Producers publish their reusable records with one successful CAS. A
    // consumer reverses a detached chain once, preserving FIFO publication order.
    // ready and incoming are disjoint; only the drain owner modifies ready.
    private volatile Request<I, O> incoming, ready;
    private volatile int draining;
    private final long retainedBytes;

    /**
     * @param maxLeaders maximum concurrent operations, without creating threads
     * @param maxBatchSize maximum existing requests claimed by one leader
     * @param operation computation; all its methods must support concurrent calls
     */
    public CallerBatchProcessor(int maxLeaders, int maxBatchSize, Operation<I, O> operation) {
        this(maxLeaders, maxBatchSize, 0, operation);
    }

    /**
     * Creates an opportunistic processor for a caller-owned CPU budget. Up to
     * {@code parallelism} overlapping calls pass through directly while no work
     * is queued or executing as a multi-input batch. Only overlap beyond that budget starts
     * combining; it does not throttle ordinary calls to the combiner limit.
     *
     * <p>The fixed internal limit is one combiner per four budgeted CPUs, with
     * a minimum of one. No thread is created. Direct calls are outside that
     * combiner limit; total concurrent operations cannot exceed the direct budget
     * plus the combiner limit. Emptying the queue and completing its multi-input
     * batches restores
     * the direct path. An overflowing singleton never suppresses free direct
     * capacity on its own. There are no timing thresholds,
     * learned decisions, dataset-dependent rules or batch-fill delays.
     *
     * @param parallelism CPU capacity available to the callers, positive
     * @param maxBatchSize maximum ready inputs passed to one batch operation
     * @param operation operation whose methods support concurrent calls
     */
    public static <I, O> CallerBatchProcessor<I, O> opportunistic(
            int parallelism, int maxBatchSize, Operation<I, O> operation) {
        if (parallelism < 1) throw new IllegalArgumentException("Parallelism must be positive");
        return new CallerBatchProcessor<>(Math.max(1, parallelism / 4), maxBatchSize, parallelism, operation);
    }

    @SuppressWarnings("unchecked")
    private CallerBatchProcessor(int maxLeaders, int maxBatchSize, int directParallelism,
                                 Operation<I, O> operation) {
        if (maxLeaders < 1 || maxBatchSize < 1 || maxBatchSize == Integer.MAX_VALUE)
            throw new IllegalArgumentException("Leader and batch limits must be positive");
        this.maxBatchSize = maxBatchSize;
        this.operation = Objects.requireNonNull(operation);
        directSlots = new DirectSlot[directParallelism];
        for (int i = 0; i < directSlots.length; i++) directSlots[i] = new DirectSlot(i);
        contexts = (Context<I, O>[]) new Context<?, ?>[maxLeaders];
        for (int i = 0; i < maxLeaders; i++) contexts[i] = new Context<>(maxBatchSize, i);
        long bytes = RamUsageEstimator.shallowSizeOf(this)
                + RamUsageEstimator.shallowSizeOf(local) + RamUsageEstimator.shallowSizeOf(contexts)
                + RamUsageEstimator.shallowSizeOf(directSlots);
        for (DirectSlot slot : directSlots) bytes = Math.addExact(bytes, RamUsageEstimator.shallowSizeOf(slot));
        for (Context<I, O> context : contexts) {
            bytes = Math.addExact(bytes, RamUsageEstimator.shallowSizeOf(context)
                    + listBytes(context.requests, maxBatchSize)
                    + RamUsageEstimator.shallowSizeOf(context.inputs)
                    + RamUsageEstimator.shallowSizeOf(context.outputs)
                    + RamUsageEstimator.shallowSizeOf(context.histogram));
        }
        retainedBytes = bytes;
    }

    private static long listBytes(ArrayList<?> list, int capacity) {
        return RamUsageEstimator.shallowSizeOf(list)
                + RamUsageEstimator.alignObjectSize(RamUsageEstimator.NUM_BYTES_ARRAY_HEADER
                        + (long) capacity * RamUsageEstimator.NUM_BYTES_OBJECT_REF);
    }

    /** Owned coordinator storage; the operation and caller-owned request slots are excluded. */
    @Override
    public long ramBytesUsed() { return retainedBytes; }

    private static final class Request<I, O> {
        final Thread thread = Thread.currentThread();
        volatile int state;
        volatile boolean parked;
        boolean inUse;
        int preferred = (int) (thread.getId() ^ (thread.getId() >>> 32)) & Integer.MAX_VALUE;
        int preferredDirect = preferred;
        I input;
        O output;
        Throwable failure;
        Request<I, O> next;
    }

    private static final class DirectSlot {
        private static final VarHandle CLAIMED, COUNT;
        static {
            try {
                var lookup = MethodHandles.lookup();
                CLAIMED = lookup.findVarHandle(DirectSlot.class, "claimed", int.class);
                COUNT = lookup.findVarHandle(DirectSlot.class, "calls", long.class);
            } catch (ReflectiveOperationException failure) {
                throw new ExceptionInInitializerError(failure);
            }
        }
        final int index;
        volatile int claimed;
        private long calls;
        DirectSlot(int index) { this.index = index; }
        void record() { COUNT.setOpaque(this, (long) COUNT.getOpaque(this) + 1); }
    }

    // Static slots/contexts do not retain the processor (and its model) in an
    // idle thread's ThreadLocal value after the application releases the model.
    private static final class Context<I, O> {
        private static final VarHandle CLAIMED;
        private static final VarHandle COUNT = MethodHandles.arrayElementVarHandle(long[].class);
        static {
            try {
                CLAIMED = MethodHandles.lookup().findVarHandle(Context.class, "claimed", int.class);
            } catch (ReflectiveOperationException failure) {
                throw new ExceptionInInitializerError(failure);
            }
        }
        volatile int claimed;
        final int slot;
        final long[] histogram;
        final ArrayList<Request<I, O>> requests;
        final List<I> inputs;
        final List<O> outputs;
        Context(int capacity, int slot) {
            this.slot = slot;
            histogram = new long[capacity + 1];
            requests = new ArrayList<>(capacity);
            inputs = new AbstractList<>() {
                public int size() { return requests.size(); }
                public I get(int i) { return requests.get(i).input; }
            };
            outputs = new AbstractList<>() {
                public int size() { return requests.size(); }
                public O get(int i) { return requests.get(i).output; }
            };
        }

        // Only the owning leader writes these counts. Opaque access makes the
        // diagnostic snapshot non-tearing without a shared atomic increment.
        void record(int size) {
            COUNT.setOpaque(histogram, size, (long) COUNT.getOpaque(histogram, size) + 1);
        }
    }

    /** Executes synchronously, preserving interruption and caller buffer lifetime. */
    public void process(I input, O output) {
        Objects.requireNonNull(input); Objects.requireNonNull(output);
        Request<I, O> request = local.get();
        if (request.inUse) throw new IllegalStateException("Reentrant batch processing is unsupported");
        request.inUse = true;
        boolean interrupted = Thread.interrupted();
        try {
            if (directSlots.length != 0 && !hasPending() && !isBatching()) {
                DirectSlot slot = acquireDirect(request.preferredDirect % directSlots.length);
                if (slot != null) {
                    try {
                        // Honor older requests published while capacity was acquired.
                        // A singleton overflow must not suppress free direct
                        // capacity. Real batches retain sharing until completion.
                        if (!hasPending() && !isBatching()) {
                            request.preferredDirect = slot.index;
                            slot.record();
                            operation.apply(input, output);
                            return;
                        }
                    } finally { DirectSlot.CLAIMED.setVolatile(slot, 0); }
                }
            }
            // An idle caller executes directly. Only saturated callers enter the
            // FIFO; no monitor, list mutation or request publication is needed.
            if (!hasPending()) {
                Context<I, O> context = acquire(request.preferred % contexts.length);
                if (context != null) {
                    if (!hasPending()) {
                        request.preferred = context.slot;
                        context.record(1);
                        try { operation.apply(input, output); }
                        finally { release(context); }
                        return;
                    }
                    // A waiter arrived while we acquired capacity; give it priority.
                    release(context);
                }
            }
            request.input = input;
            request.output = output;
            request.failure = null;
            request.state = WAITING;
            publish(request);
            for (;;) {
                int state = request.state; // Acquire publishes outputs and failures.
                if (state == DONE) break;
                if (state == WAITING) {
                    Context<I, O> context = acquire(request.preferred % contexts.length);
                    if (context != null) {
                        request.preferred = context.slot;
                        int claim = take(context, request);
                        if (claim == CLAIM_READY) { run(context); continue; }
                        if (claim == CLAIM_SKIPPED) { release(context); continue; }
                        // The drain owner will signal pending work after release.
                        // Publish this slot first, then recheck the drain gate to
                        // cover an unlock that observed the slot still occupied.
                        free(context);
                        if ((int) DRAINING.getVolatile(this) == 0) continue;
                    }
                }
                request.parked = true;
                // Completion publishes DONE before checking parked. Rechecking
                // after this volatile store ensures an early completion either
                // prevents parking or supplies the unpark permit.
                int latest = request.state;
                if (latest != DONE && (latest != WAITING
                        || (int) DRAINING.getVolatile(this) != 0 || !hasFreeCapacity())) {
                    LockSupport.park(this);
                    if (Thread.interrupted()) interrupted = true;
                }
                request.parked = false;
            }
            if (request.failure instanceof Error) throw (Error) request.failure;
            if (request.failure instanceof RuntimeException) throw (RuntimeException) request.failure;
            if (request.failure != null) throw new RuntimeException("Batch operation failed", request.failure);
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Exception failure) {
            if (failure instanceof InterruptedException) interrupted = true;
            throw new RuntimeException("Batch operation failed", failure);
        } finally {
            request.input = null;
            request.output = null;
            request.failure = null;
            request.parked = false;
            request.inUse = false;
            if (interrupted) Thread.currentThread().interrupt();
        }
    }

    private DirectSlot acquireDirect(int preferred) {
        int index = preferred;
        do {
            DirectSlot slot = directSlots[index];
            if ((int) DirectSlot.CLAIMED.getVolatile(slot) == 0
                    && DirectSlot.CLAIMED.compareAndSet(slot, 0, 1)) return slot;
            if (++index == directSlots.length) index = 0;
        } while (index != preferred);
        return null;
    }

    private boolean isBatching() {
        for (Context<I, O> context : contexts)
            if ((int) Context.CLAIMED.getVolatile(context) > 1) return true;
        return false;
    }

    // Each slot has its own admission word. Callers start at their last slot,
    // avoiding a global CAS counter and repeated competition for the first slot.
    private Context<I, O> acquire(int preferred) {
        int slot = preferred;
        do {
            Context<I, O> context = contexts[slot];
            if ((int) Context.CLAIMED.getVolatile(context) == 0
                    && Context.CLAIMED.compareAndSet(context, 0, 1)) return context;
            if (++slot == contexts.length) slot = 0;
        } while (slot != preferred);
        return null;
    }

    private void free(Context<I, O> context) {
        // Volatile publication orders the subsequent waiter check after freeing
        // capacity, closing the enqueue-versus-release lost-wakeup race.
        Context.CLAIMED.setVolatile(context, 0);
    }

    private boolean hasFreeCapacity() {
        for (Context<I, O> context : contexts)
            if ((int) Context.CLAIMED.getVolatile(context) == 0) return true;
        return false;
    }

    private boolean hasPending() { return ready != null || incoming != null; }

    private void publish(Request<I, O> request) {
        Request<I, O> previous;
        do {
            previous = incoming;
            request.next = previous;
        } while (!INCOMING.compareAndSet(this, previous, request));
    }

    private void signalPending() {
        Request<I, O> next = ready;
        if (next == null) next = incoming;
        // Any pending caller can act as combiner; take() still serves FIFO.
        // A concurrent completion can make this a harmless spurious wakeup.
        if (next != null) LockSupport.unpark(next.thread);
    }

    private void release(Context<I, O> context) {
        free(context);
        if (hasPending()) signalPending();
    }

    private void releaseDrain() {
        // A failed drainer frees its slot before checking this publication. The
        // two checks ensure it either retries or receives a wakeup, never sleeps
        // with free capacity and no owner responsible for further progress.
        DRAINING.setVolatile(this, 0);
        if (hasPending() && hasFreeCapacity()) signalPending();
    }

    @SuppressWarnings("unchecked")
    private int take(Context<I, O> context, Request<I, O> owner) {
        if (!DRAINING.compareAndSet(this, 0, 1)) return CLAIM_BUSY;
        try {
            if (owner.state != WAITING) return CLAIM_SKIPPED;
            Request<I, O> first = ready;
            if (first == null) {
                Request<I, O> stack = (Request<I, O>) INCOMING.getAndSet(this, null);
                while (stack != null) {
                    Request<I, O> next = stack.next;
                    stack.next = first;
                    first = stack;
                    stack = next;
                }
            }
            if (first == null) return CLAIM_SKIPPED;
            do {
                Request<I, O> next = first.next;
                first.next = null;
                first.state = ASSIGNED;
                context.requests.add(first);
                first = next;
            } while (first != null && context.requests.size() < maxBatchSize);
            ready = first;
            return CLAIM_READY;
        } finally { releaseDrain(); }
    }

    private void run(Context<I, O> context) {
        int size = context.requests.size();
        // Admission distinguishes a true shared computation from a singleton
        // overflow. Zero is free; one is reserved/singleton; larger values are
        // executing batch widths. All occupied states retain the same limit.
        if (size > 1) Context.CLAIMED.setVolatile(context, size);
        context.record(size);
        Throwable failure = null;
        try {
            if (context.requests.size() == 1) {
                Request<I, O> only = context.requests.get(0);
                operation.apply(only.input, only.output);
            } else operation.applyBatch(context.inputs, context.outputs);
        } catch (Throwable error) {
            failure = error;
            if (error instanceof InterruptedException) Thread.currentThread().interrupt();
        } finally {
            // This context is exclusively owned until release. Publish results
            // and wake callers without holding the drain gate.
            for (int i = 0; i < context.requests.size(); i++) {
                Request<I, O> request = context.requests.get(i);
                request.failure = failure;
                request.state = DONE; // Release before followers access their output.
                if (request.parked) LockSupport.unpark(request.thread);
            }
            context.requests.clear();
            release(context);
        }
    }

    /**
     * Diagnostic snapshot: index is batch size, value is batch count. Concurrent
     * calls may observe an in-progress operation; totals are exact when idle.
     */
    public long[] batchHistogram() {
        long[] snapshot = new long[maxBatchSize + 1];
        for (DirectSlot slot : directSlots) snapshot[1] += (long) DirectSlot.COUNT.getOpaque(slot);
        for (Context<I, O> context : contexts)
            for (int i = 1; i < snapshot.length; i++)
                snapshot[i] += (long) Context.COUNT.getOpaque(context.histogram, i);
        return snapshot;
    }

    /**
     * Diagnostic snapshot of waiting requests, excluding executing batches. The
     * short drain exclusion is used only by this explicit inspection method.
     */
    public int waitingCount() {
        while (!DRAINING.compareAndSet(this, 0, 1)) Thread.yield();
        try {
            int count = 0;
            for (Request<I, O> r = ready; r != null; r = r.next) count = Math.incrementExact(count);
            // Producers can prepend records, but cannot change this chain while
            // the drain gate is held. This is a snapshot at the incoming read.
            for (Request<I, O> r = incoming; r != null; r = r.next) count = Math.incrementExact(count);
            return count;
        } finally { releaseDrain(); }
    }
}
