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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Bounded two-stage execution on a caller-supplied executor. Ready work is
 * prepared in groups without a batch-fill delay, then each request executes
 * independently. A worker can execute requests prepared by another worker.
 *
 * <p>Exactly {@code workers} long-lived tasks are submitted to the executor; it
 * must have that much available execution capacity. This class creates no
 * threads and does not shut down the supplied executor. Use one instance for a
 * shared CPU budget rather than reserving worker sets for every model.
 *
 * <p>Queue records, batch lists and row views are allocated at construction.
 * Each accepted request is handed back through its completion callback, which
 * may recycle it after its last use. No request is discarded on close. Close drains accepted
 * work and must precede shutting down the supplied executor.
 */
public final class BatchingExecutor<T> implements AutoCloseable {
    /** Thread-safe operations over independently owned requests. */
    public interface Operation<T> {
        /** Identity key for compatible preparation; null requests are prepared alone. */
        Object batchKey(T request);
        /** Borrowed read-only list, valid only for this call; never retained. */
        void prepareBatch(List<T> requests) throws Exception;
        /** Runs one prepared request, without any batch-wide completion barrier. */
        void execute(T request) throws Exception;
        /** Called exactly once per accepted request; must not throw. */
        void complete(T request, Throwable failure);
    }

    private static final long CLOSED = Long.MIN_VALUE;
    private static final class Slot<T> { T request; }
    private final ArrayBlockingQueue<Slot<T>> available, pending, ready;
    private final ReentrantLock drain = new ReentrantLock();
    private final Semaphore work = new Semaphore(0);
    private final AtomicLong admission = new AtomicLong();
    private final AtomicReference<Throwable> callbackFailure = new AtomicReference<>();
    private final CountDownLatch stopped;
    private final Operation<T> operation;
    private final int capacity, workerCount, batchSize;
    private final List<Worker> workers;
    private final ThreadLocal<Boolean> inWorker = new ThreadLocal<>();

    /**
     * @param capacity total accepted requests, including preparing and executing
     * @param batchSize maximum currently available compatible requests per preparation
     * @param executor caller-owned asynchronous execution capacity; completion
     *                 callbacks run on these workers and must return promptly
     */
    public BatchingExecutor(int workerCount, int capacity, int batchSize,
                            Executor executor, Operation<T> operation) {
        if (workerCount <= 0 || capacity <= 0 || batchSize <= 0 || batchSize > capacity)
            throw new IllegalArgumentException("Workers, capacity and batch size must be positive; batch size <= capacity");
        this.workerCount = workerCount;
        this.capacity = capacity;
        this.batchSize = batchSize;
        this.operation = Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(executor, "executor");
        available = new ArrayBlockingQueue<>(capacity);
        pending = new ArrayBlockingQueue<>(capacity);
        ready = new ArrayBlockingQueue<>(capacity);
        for (int i = 0; i < capacity; i++) available.add(new Slot<>());
        stopped = new CountDownLatch(workerCount);
        workers = new ArrayList<>(workerCount);
        for (int i = 0; i < workerCount; i++) workers.add(new Worker());
        int submitted = 0;
        try {
            for (Worker worker : workers) { executor.execute(worker); submitted++; }
        } catch (RuntimeException | Error failure) {
            admission.getAndUpdate(value -> value | CLOSED);
            for (int i = submitted; i < workerCount; i++) stopped.countDown();
            work.release(submitted);
            awaitStopped();
            throw failure;
        }
    }

    /**
     * Accepts without waiting, returning false at the total admission limit.
     * Rejection leaves ownership with the caller. Close rejects further requests.
     * Inputs must not be mutated or submitted twice before completion.
     */
    public boolean trySubmit(T request) {
        Objects.requireNonNull(request, "request");
        if (callbackFailure.get() != null)
            throw new IllegalStateException("A completion callback failed", callbackFailure.get());
        for (;;) {
            long count = admission.get();
            if (count < 0) throw new RejectedExecutionException("Batching executor is closed");
            if (count == capacity) return false;
            if (admission.compareAndSet(count, count + 1)) break;
        }
        // Completion returns the slot before releasing its admission count.
        Slot<T> slot = available.remove();
        slot.request = request;
        pending.add(slot);
        work.release();
        return true;
    }

    private final class Worker implements Runnable {
        final ArrayList<Slot<T>> slots = new ArrayList<>(batchSize);
        final ArrayList<T> requests = new ArrayList<>(batchSize);
        final List<T> borrowedRequests = Collections.unmodifiableList(requests);
        final long[] histogram = new long[batchSize + 1];

        @Override
        public void run() {
            inWorker.set(Boolean.TRUE);
            try {
                for (;;) {
                    if (admission.get() == CLOSED) return;
                    work.acquireUninterruptibly();
                    Slot<T> next = ready.poll();
                    if (next != null) {
                        Throwable failure = null;
                        try { operation.execute(next.request); }
                        catch (Throwable error) { failure = error; }
                        finish(next, failure);
                        continue;
                    }
                    drain.lock();
                    try {
                        next = pending.poll();
                        if (next == null) continue;
                        slots.add(next);
                        requests.add(next.request);
                        Object key = operation.batchKey(next.request);
                        while (key != null && slots.size() < batchSize) {
                            Slot<T> head = pending.peek();
                            if (head == null || operation.batchKey(head.request) != key) break;
                            next = pending.remove();
                            // Each submission signaled one pending request. This worker
                            // already consumed the first signal; retire the additional
                            // ones as their requests are folded into this preparation.
                            // Another worker may already have consumed a signal and be
                            // waiting for drain, so acquisition must never block here.
                            work.tryAcquire();
                            slots.add(next);
                            requests.add(next.request);
                        }
                    } catch (Throwable failure) {
                        // A key callback failure cannot strand already removed work.
                        for (Slot<T> slot : slots) finish(slot, failure);
                        slots.clear(); requests.clear();
                        continue;
                    } finally { drain.unlock(); }
                    Throwable failure = null;
                    try { operation.prepareBatch(borrowedRequests); histogram[slots.size()]++; }
                    catch (Throwable error) { failure = error; }
                    if (failure == null) {
                        int size = slots.size();
                        if (size == 1) {
                            Slot<T> slot = slots.get(0);
                            try { operation.execute(slot.request); }
                            catch (Throwable error) { failure = error; }
                            finish(slot, failure);
                        } else {
                            for (Slot<T> slot : slots) ready.add(slot);
                            work.release(size);
                        }
                    } else {
                        for (Slot<T> slot : slots) finish(slot, failure);
                    }
                    slots.clear(); requests.clear();
                }
            } finally {
                inWorker.remove();
                stopped.countDown();
            }
        }
    }

    private void finish(Slot<T> slot, Throwable failure) {
        T request = slot.request;
        slot.request = null;
        available.add(slot);
        long count = admission.decrementAndGet();
        try { operation.complete(request, failure); }
        catch (Throwable error) { callbackFailure.compareAndSet(null, error); }
        if (count == CLOSED) work.release(workerCount);
    }

    /** Snapshot of successful preparation groups; exact when all requests have completed. */
    public long[] batchHistogram() {
        long[] result = new long[batchSize + 1];
        for (Worker worker : workers)
            for (int i = 1; i < result.length; i++) result[i] += worker.histogram[i];
        return result;
    }

    /** Accepted requests in either queue or preparation/execution, excluding completion callbacks. */
    public int outstanding() { return (int) (admission.get() & Long.MAX_VALUE); }

    @Override
    public void close() {
        if (Boolean.TRUE.equals(inWorker.get()))
            throw new IllegalStateException("Cannot close a batching executor from its worker callback");
        admission.getAndUpdate(value -> value | CLOSED);
        work.release(workerCount);
        awaitStopped();
        if (callbackFailure.get() != null)
            throw new IllegalStateException("A completion callback failed", callbackFailure.get());
    }

    private void awaitStopped() {
        boolean interrupted = false;
        for (;;) {
            try { stopped.await(); break; }
            catch (InterruptedException e) { interrupted = true; }
        }
        if (interrupted) Thread.currentThread().interrupt();
    }
}
