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

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Bounded asynchronous execution with optional preparation of compatible requests
 * in batches. Batches contain only requests already waiting; there is no fill timer.
 * Prepared work takes priority, and the preparing worker executes its first request
 * while other workers may execute the rest.
 *
 * <p>The supplied executor must schedule asynchronously, reject explicitly when it
 * cannot accept work, and remain operational until this service closes. Direct
 * executors, caller-runs rejection policies and silent-discard policies are not
 * supported. At most {@code workerCount} reusable drain tasks are submitted at a
 * time. Each exits when the queues are empty, releasing shared execution capacity.
 * This class creates no threads and does not shut down the supplied executor.
 *
 * <p>Requests and their buffers remain caller-owned. An accepted request must not
 * be modified or submitted again until its completion callback. Rejection at the
 * admission limit leaves ownership with the caller. Accepted work completes once,
 * including preparation or dispatch failure. Completion may run on a submitting
 * thread if the executor rejects dispatch; successful work never runs inline.
 * Queue arrays, worker runnables and batch lists are allocated at construction.
 *
 * <p>Close rejects new submissions and waits for accepted work and callbacks.
 * Do not close from a callback or block a host executor thread needed to drain this
 * service. The host must not cancel or discard already accepted drain tasks.
 */
public final class BatchingExecutor<T> implements AutoCloseable {
    /** Callbacks must be thread-safe. Completion must return promptly. */
    public interface Operation<T> {
        /**
         * Immutable identity key for compatible preparation, or null to prepare
         * alone. Called on the submitting thread, outside internal locks.
         */
        Object batchKey(T request);
        /** Borrowed read-only list, valid only during this call; never retain it. */
        void prepareBatch(List<T> requests) throws Exception;
        /** Executes one prepared request, without a batch-wide completion barrier. */
        void execute(T request) throws Exception;
        /**
         * Called exactly once for accepted work. The request may be recycled here.
         * Exceptions are retained for inspection and close, without poisoning work.
         */
        void complete(T request, Throwable failure);
    }

    // Keep the entry on host threads but clear its value on exit. Removing it
    // would allocate a fresh ThreadLocalMap entry for every isolated drain task.
    private static final ThreadLocal<BatchingExecutor<?>> CURRENT = new ThreadLocal<>();

    private final ReentrantLock lock = new ReentrantLock();
    private final Condition drained = lock.newCondition();
    private final Executor executor;
    private final Operation<T> operation;
    private final int capacity, batchSize, workerLimit;
    private final Object[] pending, pendingKeys, ready;
    private final Throwable[] pendingFailures;
    private int pendingHead, pendingTail, pendingSize;
    private int pendingGroups, tailGroupSize;
    private int readyHead, readyTail, readySize;
    private final ArrayDeque<Worker> idleWorkers;
    private final List<Worker> workers;
    private int activeWorkers, busyWorkers, callbacks;
    private volatile int outstanding;
    private volatile boolean closed;
    private final AtomicLong callbackFailures = new AtomicLong();
    private final AtomicReference<Throwable> firstCallbackFailure = new AtomicReference<>();

    /**
     * @param workerCount maximum concurrent preparation/execution drain tasks
     * @param capacity maximum requests queued, preparing or executing; excludes callbacks
     * @param batchSize maximum currently waiting compatible requests per preparation
     * @param executor caller-owned asynchronous execution capacity
     * @param operation request preparation, execution and completion operations
     */
    public BatchingExecutor(int workerCount, int capacity, int batchSize,
                            Executor executor, Operation<T> operation) {
        if (workerCount <= 0 || capacity <= 0 || batchSize <= 0
                || batchSize > capacity || batchSize == Integer.MAX_VALUE)
            throw new IllegalArgumentException("Workers, capacity and batch size must be positive; "
                    + "batch size must fit its histogram and not exceed capacity");
        this.executor = Objects.requireNonNull(executor, "executor");
        this.operation = Objects.requireNonNull(operation, "operation");
        if (executor instanceof ThreadPoolExecutor) {
            var policy = ((ThreadPoolExecutor) executor).getRejectedExecutionHandler();
            if (policy instanceof ThreadPoolExecutor.DiscardPolicy
                    || policy instanceof ThreadPoolExecutor.DiscardOldestPolicy
                    || policy instanceof ThreadPoolExecutor.CallerRunsPolicy)
                throw new IllegalArgumentException("Executor must reject explicitly without discarding or running inline");
        }
        this.capacity = capacity;
        this.batchSize = batchSize;
        this.workerLimit = Math.min(workerCount, capacity);
        pending = new Object[capacity];
        pendingKeys = new Object[capacity];
        pendingFailures = new Throwable[capacity];
        ready = new Object[capacity];
        idleWorkers = new ArrayDeque<>(workerLimit);
        workers = new ArrayList<>(workerLimit);
        for (int i = 0; i < workerLimit; i++) {
            Worker worker = new Worker();
            workers.add(worker);
            idleWorkers.addLast(worker);
        }
    }

    /**
     * Accepts without waiting for capacity. Returns false at the admission limit;
     * throws after close. A true return transfers ownership until completion.
     */
    public boolean trySubmit(T request) {
        Objects.requireNonNull(request, "request");
        if (closed) throw new RejectedExecutionException("Batching executor is closed");
        if (outstanding == capacity) return false;
        Object key = null;
        Throwable failure = null;
        try { key = operation.batchKey(request); }
        catch (Throwable error) { failure = error; }
        Worker launches;
        lock.lock();
        try {
            if (closed) throw new RejectedExecutionException("Batching executor is closed");
            if (outstanding == capacity) return false;
            int previousTail = pendingTail == 0 ? capacity - 1 : pendingTail - 1;
            if (pendingSize != 0 && failure == null && key != null
                    && pendingFailures[previousTail] == null && pendingKeys[previousTail] == key
                    && tailGroupSize < batchSize) {
                tailGroupSize++;
            } else {
                pendingGroups++;
                tailGroupSize = 1;
            }
            pending[pendingTail] = request;
            pendingKeys[pendingTail] = key;
            pendingFailures[pendingTail] = failure;
            pendingTail = advance(pendingTail);
            pendingSize++;
            outstanding++;
            launches = reserveWorkers();
        } finally { lock.unlock(); }
        launch(launches);
        return true;
    }

    private int advance(int position) { return position == capacity - 1 ? 0 : position + 1; }

    /** Called under lock; links reusable tasks rather than allocating a launch list. */
    private Worker reserveWorkers() {
        // Track compatible groups at admission rather than scanning the queue or
        // treating unrelated models as one batch and underutilizing the budget.
        int target = (int) Math.min(workerLimit, (long) busyWorkers + readySize + pendingGroups);
        Worker first = null, last = null;
        while (activeWorkers < target) {
            Worker worker = idleWorkers.removeFirst();
            activeWorkers++;
            if (last == null) first = worker;
            else last.nextLaunch = worker;
            last = worker;
        }
        return first;
    }

    /** Executor calls and operation callbacks are always outside lock. */
    private void launch(Worker first) {
        while (first != null) {
            Worker worker = first;
            first = worker.nextLaunch;
            worker.nextLaunch = null;
            worker.launchingThread.set(Thread.currentThread());
            try { executor.execute(worker); }
            catch (RuntimeException | Error failure) { dispatchFailed(worker, failure); }
            finally {
                // A host worker may later dequeue a task that it submitted.
                // Only execution on the submitting call stack is inline. CAS
                // cannot clear the marker of a reused task launched elsewhere.
                worker.launchingThread.compareAndSet(Thread.currentThread(), null);
            }
        }
    }

    private void dispatchFailed(Worker worker, Throwable failure) {
        // Allocation is confined to dispatch failure, never ordinary submission.
        List<T> failed = null;
        lock.lock();
        try {
            worker.launchingThread.set(null);
            activeWorkers--;
            idleWorkers.addLast(worker);
            if (activeWorkers == 0) {
                failed = new ArrayList<>(pendingSize + readySize);
                while (readySize != 0) failed.add(takeReady());
                while (pendingSize != 0) failed.add(takePending());
                pendingGroups = 0;
            }
            signalDrained();
        } finally { lock.unlock(); }
        if (failed != null) for (T request : failed) finish(request, failure);
        // Another accepted drain task, if present, will process the queues.
    }

    @SuppressWarnings("unchecked")
    private T takePending() {
        T value = (T) pending[pendingHead];
        pending[pendingHead] = null;
        pendingKeys[pendingHead] = null;
        pendingFailures[pendingHead] = null;
        pendingHead = advance(pendingHead);
        pendingSize--;
        if (pendingSize == 0) tailGroupSize = 0;
        return value;
    }

    @SuppressWarnings("unchecked")
    private T takeReady() {
        T value = (T) ready[readyHead];
        ready[readyHead] = null;
        readyHead = advance(readyHead);
        readySize--;
        return value;
    }

    private final class Worker implements Runnable {
        final ArrayList<T> requests = new ArrayList<>(batchSize);
        final List<T> borrowedRequests = Collections.unmodifiableList(requests);
        final AtomicLongArray histogram = new AtomicLongArray(batchSize + 1);
        Worker nextLaunch;
        final AtomicReference<Thread> launchingThread = new AtomicReference<>();
        boolean busy, alreadyPrepared;
        Throwable preparationFailure;

        /** Acquires the next work, or atomically releases this task's worker lease. */
        boolean takeWork() {
            lock.lock();
            try {
                if (busy) { busyWorkers--; busy = false; }
                requests.clear();
                preparationFailure = null;
                if (readySize != 0) {
                    requests.add(takeReady());
                    alreadyPrepared = true;
                } else if (pendingSize != 0) {
                    Object key = pendingKeys[pendingHead];
                    preparationFailure = pendingFailures[pendingHead];
                    requests.add(takePending());
                    alreadyPrepared = false;
                    while (preparationFailure == null && key != null
                            && requests.size() < batchSize && pendingSize != 0
                            && pendingKeys[pendingHead] == key && pendingFailures[pendingHead] == null)
                        requests.add(takePending());
                    pendingGroups--;
                } else {
                    activeWorkers--;
                    idleWorkers.addLast(this);
                    signalDrained();
                    return false;
                }
                busyWorkers++;
                busy = true;
                return true;
            } finally { lock.unlock(); }
        }

        @Override
        public void run() {
            if (Thread.currentThread() == launchingThread.get())
                throw new RejectedExecutionException("Executor must schedule work asynchronously");
            launchingThread.set(null);
            BatchingExecutor<?> previous = CURRENT.get();
            CURRENT.set(BatchingExecutor.this);
            try {
                while (takeWork()) {
                    Throwable failure = preparationFailure;
                    if (!alreadyPrepared && failure == null) {
                        try {
                            operation.prepareBatch(borrowedRequests);
                            histogram.incrementAndGet(requests.size());
                        } catch (Throwable error) { failure = error; }
                    }
                    if (failure != null) {
                        for (T request : requests) finish(request, failure);
                        continue;
                    }
                    if (requests.size() > 1) {
                        Worker launches;
                        lock.lock();
                        try {
                            for (int i = 1; i < requests.size(); i++) {
                                ready[readyTail] = requests.get(i);
                                readyTail = advance(readyTail);
                                readySize++;
                            }
                            launches = reserveWorkers();
                        } finally { lock.unlock(); }
                        launch(launches);
                    }
                    T first = requests.get(0);
                    try { operation.execute(first); }
                    catch (Throwable error) { failure = error; }
                    finish(first, failure);
                }
            } finally {
                CURRENT.set(previous); // Do not retain the service/model on host threads.
            }
        }
    }

    private void finish(T request, Throwable failure) {
        lock.lock();
        try {
            // Release admission before the callback may recycle the request.
            callbacks++;
            outstanding--;
        } finally { lock.unlock(); }
        BatchingExecutor<?> previous = CURRENT.get();
        boolean entering = previous != this;
        if (entering) CURRENT.set(this);
        try { operation.complete(request, failure); }
        catch (Throwable error) {
            firstCallbackFailure.compareAndSet(null, error);
            callbackFailures.incrementAndGet();
        } finally {
            if (entering) CURRENT.set(previous);
            lock.lock();
            try { callbacks--; signalDrained(); }
            finally { lock.unlock(); }
        }
    }

    private void signalDrained() {
        if (closed && outstanding == 0 && activeWorkers == 0 && callbacks == 0)
            drained.signalAll();
    }

    /** Atomic snapshot; exact after all accepted requests have completed. */
    public long[] batchHistogram() {
        long[] result = new long[batchSize + 1];
        for (Worker worker : workers)
            for (int i = 1; i < result.length; i++) result[i] += worker.histogram.get(i);
        return result;
    }

    /** Accepted work, excluding completion callbacks. */
    public int outstanding() { return outstanding; }

    /** Number of completion callbacks that threw, without blocking later submissions. */
    public long completionFailureCount() { return callbackFailures.get(); }

    /** First callback exception for monitoring, or null when no callback failed. */
    public Throwable firstCompletionFailure() { return firstCallbackFailure.get(); }

    /** Drains accepted work, preserves interrupts, then reports any callback failure. */
    @Override
    public void close() {
        if (CURRENT.get() == this)
            throw new IllegalStateException("Cannot close a batching executor from its operation or callback");
        lock.lock();
        try {
            closed = true;
            while (outstanding != 0 || activeWorkers != 0 || callbacks != 0)
                drained.awaitUninterruptibly();
        } finally { lock.unlock(); }
        Throwable failure = firstCallbackFailure.get();
        if (failure != null)
            throw new IllegalStateException(completionFailureCount() + " completion callbacks failed", failure);
    }
}
