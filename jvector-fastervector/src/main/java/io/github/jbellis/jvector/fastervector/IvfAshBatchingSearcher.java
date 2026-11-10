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

package io.github.jbellis.jvector.fastervector;

import io.github.jbellis.jvector.fastervector.utils.FlatArrayVecs;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

/**
 * Answers single queries from many threads by combining them into batched searches.
 * <p>
 * fastervector is built to search a batch of queries per call: it scans each probed list once
 * for every query in the batch that probes it. Concurrent single-query calls don't scale, since
 * at the pinned fastervector commit every searcher shares the same I/O rings and scan pool. This
 * class turns single queries into batches: callers submit one query each, and one dispatcher
 * thread searches whatever has queued, up to {@code maxBatch} queries, in one call on the
 * wrapped searcher. Queries that arrive during a search form the next batch, so under load the
 * batches grow with the number of callers without any added wait.
 * <p>
 * The queue holds at most {@code maxQueued} queries. When it is full, a new query is rejected
 * with {@link RejectedExecutionException}, at once by default or after waiting up to
 * {@code enqueueTimeout} for space, so overload shows up as rejections rather than as latency
 * that grows without bound. A query may also carry a timeout: if its deadline passes while it is
 * queued, it is dropped without being searched and its future fails with
 * {@link TimeoutException}. Cancelled futures are dropped the same way.
 * <p>
 * Give the wrapped searcher a thread pool sized for the whole machine; it runs one batch at a
 * time. This class doesn't own the searcher: close this first, then the searcher.
 * <p>
 * Futures are completed on the dispatcher thread, so stages chained onto them without an
 * executor run there and delay the next batch; keep them short. A future that times out is
 * failed by {@link CompletableFuture#orTimeout}'s shared timer thread instead.
 */
public final class IvfAshBatchingSearcher implements AutoCloseable {
    /** Default {@link Builder#maxBatch(int)}. */
    public static final int DEFAULT_MAX_BATCH = 1024;
    /** Default {@link Builder#maxQueued(int)}. */
    public static final int DEFAULT_MAX_QUEUED = 4 * DEFAULT_MAX_BATCH;
    // Keeps deadline arithmetic on System.nanoTime() from overflowing (about 146 years)
    private static final long MAX_TIMEOUT_NANOS = Long.MAX_VALUE / 2;

    private final Function<Vecs, SearchResult> search;
    /** Query dimension the searcher checks, or 0 if it doesn't know it. */
    private final int dim;
    private final int maxBatch;
    private final int maxQueued;
    private final long maxDelayNanos;
    /** How long a query waits for queue space; 0 rejects at once. */
    private final long enqueueTimeoutNanos;
    private final LinkedBlockingQueue<Request> queue;
    private final Thread dispatcher;
    private final AtomicLong batches = new AtomicLong();
    private final AtomicLong queries = new AtomicLong();
    private final AtomicLong rejected = new AtomicLong();
    private final AtomicLong expired = new AtomicLong();
    private volatile boolean closed;

    /** {@code deadline} is a {@link System#nanoTime()} value, used only if {@code timed}. */
    private record Request(float[] query, CompletableFuture<List<NodeScore>> result, boolean timed, long deadline) {}

    IvfAshBatchingSearcher(Function<Vecs, SearchResult> search, int dim, int maxBatch, Duration maxDelay) {
        this(search, dim, maxBatch, maxDelay, DEFAULT_MAX_QUEUED, Duration.ZERO);
    }

    IvfAshBatchingSearcher(Function<Vecs, SearchResult> search, int dim, int maxBatch, Duration maxDelay,
                           int maxQueued, Duration enqueueTimeout) {
        if (maxBatch < 1) {
            throw new IllegalArgumentException("maxBatch must be at least 1, got " + maxBatch);
        }
        if (maxDelay.isNegative()) {
            throw new IllegalArgumentException("maxDelay must not be negative, got " + maxDelay);
        }
        if (maxQueued < 1) {
            throw new IllegalArgumentException("maxQueued must be at least 1, got " + maxQueued);
        }
        if (enqueueTimeout.isNegative()) {
            throw new IllegalArgumentException("enqueueTimeout must not be negative, got " + enqueueTimeout);
        }
        this.search = search;
        this.dim = dim;
        this.maxBatch = maxBatch;
        this.maxQueued = maxQueued;
        this.maxDelayNanos = nanos(maxDelay);
        this.enqueueTimeoutNanos = nanos(enqueueTimeout);
        this.queue = new LinkedBlockingQueue<>(maxQueued);
        this.dispatcher = new Thread(this::dispatch, "ivfash-batching-searcher");
        dispatcher.setDaemon(true);
        dispatcher.start();
    }

    /** A builder that wraps {@code searcher}, which must stay open until this is closed. */
    public static Builder builder(IvfAshIndexSearcher searcher) {
        return new Builder(searcher);
    }

    /**
     * Searches one query and waits for its result, however long that takes.
     *
     * @param query the query vector; copied, so the caller may reuse the array
     * @return up to {@code topk} neighbours, best first
     * @throws IllegalArgumentException   if the query's dimension doesn't match the index's
     * @throws RejectedExecutionException if the queue stays full (see {@link Builder#enqueueTimeout})
     * @throws IllegalStateException      if this is closed, or closes before the query is searched
     * @throws FvecException              if the native search fails
     */
    public List<NodeScore> search(float[] query) {
        try {
            return searchAsync(query).join();
        } catch (CompletionException e) {
            throw unwrap(e);
        }
    }

    /**
     * Searches one query, giving up once {@code timeout} has passed. Waiting for queue space
     * counts against the timeout.
     *
     * @param query   the query vector; copied, so the caller may reuse the array
     * @param timeout how long the caller will wait for the result; must be positive
     * @return up to {@code topk} neighbours, best first
     * @throws TimeoutException           if the timeout passes before the result arrives
     * @throws IllegalArgumentException   if the query's dimension doesn't match the index's,
     *                                    or the timeout isn't positive
     * @throws RejectedExecutionException if the queue stays full (see {@link Builder#enqueueTimeout})
     * @throws IllegalStateException      if this is closed, or closes before the query is searched
     * @throws FvecException              if the native search fails
     */
    public List<NodeScore> search(float[] query, Duration timeout) throws TimeoutException {
        try {
            return searchAsync(query, timeout).join();
        } catch (CompletionException e) {
            if (e.getCause() instanceof TimeoutException timedOut) {
                throw timedOut;
            }
            throw unwrap(e);
        }
    }

    /**
     * Queues one query with no deadline. The future fails with the exceptions
     * {@link #search(float[])} throws once the query is queued; cancelling it before the query
     * is searched removes the query from the next batch.
     *
     * @param query the query vector; copied, so the caller may reuse the array
     * @throws IllegalArgumentException   if the query's dimension doesn't match the index's
     * @throws RejectedExecutionException if the queue stays full (see {@link Builder#enqueueTimeout})
     * @throws IllegalStateException      if this is closed
     */
    public CompletableFuture<List<NodeScore>> searchAsync(float[] query) {
        return enqueue(query, false, 0);
    }

    /**
     * Queues one query with a deadline {@code timeout} from now. Waiting for queue space counts
     * against it. When the deadline passes the future fails with {@link TimeoutException}: a
     * query still queued is dropped without being searched, and the result of a query already
     * being searched is discarded.
     *
     * @param query   the query vector; copied, so the caller may reuse the array
     * @param timeout time until the deadline; must be positive
     * @throws IllegalArgumentException   if the query's dimension doesn't match the index's,
     *                                    or the timeout isn't positive
     * @throws RejectedExecutionException if the queue stays full (see {@link Builder#enqueueTimeout})
     * @throws IllegalStateException      if this is closed
     */
    public CompletableFuture<List<NodeScore>> searchAsync(float[] query, Duration timeout) {
        Objects.requireNonNull(timeout, "timeout");
        if (timeout.isNegative() || timeout.isZero()) {
            throw new IllegalArgumentException("timeout must be positive, got " + timeout);
        }
        return enqueue(query, true, Math.min(nanos(timeout), MAX_TIMEOUT_NANOS));
    }

    private CompletableFuture<List<NodeScore>> enqueue(float[] query, boolean timed, long timeoutNanos) {
        if (dim != 0 && query.length != dim) {
            throw new IllegalArgumentException("query dim mismatch: index dim=" + dim + ", query dim=" + query.length);
        }
        if (closed) {
            throw new IllegalStateException("IvfAshBatchingSearcher is already closed");
        }
        long deadline = System.nanoTime() + timeoutNanos;
        Request request = new Request(query.clone(), new CompletableFuture<>(), timed, deadline);
        boolean queued;
        try {
            queued = offer(request);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            rejected.incrementAndGet();
            throw new RejectedExecutionException("interrupted while waiting for queue space", e);
        }
        if (!queued) {
            rejected.incrementAndGet();
            throw new RejectedExecutionException("IvfAshBatchingSearcher queue is full (" + maxQueued + " queries)");
        }
        // close() may have drained the queue just before the add; if so, nothing will search it
        if (closed && queue.remove(request)) {
            request.result().completeExceptionally(closedException());
        }
        if (timed) {
            // Fails the caller's future at the deadline even if the query is already in a batch
            request.result().orTimeout(Math.max(0, deadline - System.nanoTime()), TimeUnit.NANOSECONDS);
        }
        return request.result();
    }

    private boolean offer(Request request) throws InterruptedException {
        if (enqueueTimeoutNanos == 0) {
            return queue.offer(request);
        }
        long wait = enqueueTimeoutNanos;
        if (request.timed()) {
            wait = Math.min(wait, request.deadline() - System.nanoTime());
        }
        return queue.offer(request, wait, TimeUnit.NANOSECONDS);
    }

    /** Number of native search calls made so far. */
    public long batchCount() {
        return batches.get();
    }

    /** Number of queries searched so far; divided by {@link #batchCount()}, the mean batch size. */
    public long queryCount() {
        return queries.get();
    }

    /** Number of queries rejected so far because the queue was full. */
    public long rejectedCount() {
        return rejected.get();
    }

    /** Number of queries dropped so far because their deadline passed before they were searched. */
    public long expiredCount() {
        return expired.get();
    }

    /** Number of queries waiting for a batch, including dropped ones the dispatcher hasn't reached yet. */
    public int queuedCount() {
        return queue.size();
    }

    /**
     * Stops the dispatcher after the batch in progress. Queries still queued fail with
     * {@link IllegalStateException}. Doesn't close the wrapped searcher. Idempotent.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        dispatcher.interrupt();
        boolean interrupted = false;
        while (dispatcher.isAlive()) {
            try {
                dispatcher.join();
            } catch (InterruptedException e) {
                interrupted = true;
            }
        }
        failAll(drain());
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    private void dispatch() {
        List<Request> batch = new ArrayList<>();
        try {
            while (!closed) {
                batch.add(queue.take());
                fill(batch);
                if (maxDelayNanos > 0) {
                    long deadline = System.nanoTime() + maxDelayNanos;
                    while (batch.size() < maxBatch) {
                        Request next = queue.poll(deadline - System.nanoTime(), TimeUnit.NANOSECONDS);
                        if (next == null) {
                            break;
                        }
                        batch.add(next);
                        fill(batch);
                    }
                }
                prune(batch);
                if (!batch.isEmpty()) {
                    execute(batch);
                }
                batch.clear();
            }
        } catch (InterruptedException e) {
            // close() stops the dispatcher this way
        } finally {
            failAll(batch);
        }
    }

    /** Drops dead queries from the batch and tops it up from the queue, up to {@code maxBatch}. */
    private void fill(List<Request> batch) {
        prune(batch);
        while (batch.size() < maxBatch && queue.drainTo(batch, maxBatch - batch.size()) > 0) {
            prune(batch);
        }
    }

    /**
     * Removes queries that shouldn't be searched: those past their deadline, which fail with
     * {@link TimeoutException}, and those whose future is already done, such as cancelled ones.
     */
    private void prune(List<Request> batch) {
        long now = System.nanoTime();
        batch.removeIf(r -> {
            if (r.timed() && now - r.deadline() >= 0) {
                r.result().completeExceptionally(new TimeoutException("query deadline passed before it was searched"));
                expired.incrementAndGet();
                return true;
            }
            return r.result().isDone();
        });
    }

    /** Searches the batch in one call. Without a known dim, the first query's length sets it. */
    private void execute(List<Request> batch) {
        int d = batch.get(0).query().length;
        List<Request> searched = new ArrayList<>(batch.size());
        for (Request request : batch) {
            if (request.query().length == d) {
                searched.add(request);
            } else {
                request.result().completeExceptionally(new IllegalArgumentException(
                        "query dim mismatch: batch dim=" + d + ", query dim=" + request.query().length));
            }
        }
        float[] flat = new float[searched.size() * d];
        for (int i = 0; i < searched.size(); i++) {
            System.arraycopy(searched.get(i).query(), 0, flat, i * d, d);
        }
        SearchResult result;
        try {
            result = search.apply(new FlatArrayVecs(flat, d));
        } catch (RuntimeException e) {
            searched.forEach(r -> r.result().completeExceptionally(e));
            return;
        }
        batches.incrementAndGet();
        queries.addAndGet(searched.size());
        for (int i = 0; i < searched.size(); i++) {
            // A no-op for a query whose deadline passed during the search
            searched.get(i).result().complete(result.results().get(i));
        }
    }

    private List<Request> drain() {
        List<Request> left = new ArrayList<>();
        queue.drainTo(left);
        return left;
    }

    private static void failAll(List<Request> requests) {
        requests.forEach(r -> r.result().completeExceptionally(closedException()));
    }

    private static IllegalStateException closedException() {
        return new IllegalStateException("IvfAshBatchingSearcher closed before the query was searched");
    }

    private static RuntimeException unwrap(CompletionException e) {
        return e.getCause() instanceof RuntimeException cause ? cause : e;
    }

    /** {@code d} in nanoseconds, saturating instead of overflowing. */
    private static long nanos(Duration d) {
        try {
            return d.toNanos();
        } catch (ArithmeticException e) {
            return d.isNegative() ? Long.MIN_VALUE : Long.MAX_VALUE;
        }
    }

    /** Builder for {@link IvfAshBatchingSearcher}. Obtain via {@link #builder(IvfAshIndexSearcher)}. */
    public static final class Builder {
        private final IvfAshIndexSearcher searcher;
        private int maxBatch = DEFAULT_MAX_BATCH;
        private Duration maxDelay = Duration.ZERO;
        private int maxQueued = DEFAULT_MAX_QUEUED;
        private Duration enqueueTimeout = Duration.ZERO;

        private Builder(IvfAshIndexSearcher searcher) {
            this.searcher = searcher;
        }

        /** Most queries per search call. Default {@value #DEFAULT_MAX_BATCH}. */
        public Builder maxBatch(int maxBatch) {
            this.maxBatch = maxBatch;
            return this;
        }

        /**
         * How long to wait for more queries before searching a batch that isn't full. Default
         * zero: search what has queued as soon as the previous search ends. A wait adds up to
         * this much latency to every query, but can make batches larger at moderate load.
         */
        public Builder maxDelay(Duration maxDelay) {
            this.maxDelay = Objects.requireNonNull(maxDelay, "maxDelay");
            return this;
        }

        /**
         * Most queries waiting for a batch; past this, new queries are rejected. Default
         * {@value #DEFAULT_MAX_QUEUED}. Size it from the latency callers will accept: a query
         * that joins a full queue waits about {@code maxQueued / maxBatch} searches.
         */
        public Builder maxQueued(int maxQueued) {
            this.maxQueued = maxQueued;
            return this;
        }

        /**
         * How long a query waits for queue space before it is rejected. Default zero: reject at
         * once when the queue is full. A query's own timeout, if shorter, caps the wait.
         */
        public Builder enqueueTimeout(Duration enqueueTimeout) {
            this.enqueueTimeout = Objects.requireNonNull(enqueueTimeout, "enqueueTimeout");
            return this;
        }

        /** Starts the dispatcher thread. */
        public IvfAshBatchingSearcher build() {
            return new IvfAshBatchingSearcher(searcher::search, searcher.dim(), maxBatch, maxDelay,
                                              maxQueued, enqueueTimeout);
        }
    }
}
