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
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
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
 * Give the wrapped searcher a thread pool sized for the whole machine; it runs one batch at a
 * time. This class doesn't own the searcher: close this first, then the searcher.
 * <p>
 * Futures are completed on the dispatcher thread, so stages chained onto them without an
 * executor run there and delay the next batch; keep them short.
 */
public final class IvfAshBatchingSearcher implements AutoCloseable {
    /** Default {@link Builder#maxBatch(int)}. */
    public static final int DEFAULT_MAX_BATCH = 1024;

    private final Function<Vecs, SearchResult> search;
    /** Query dimension the searcher checks, or 0 if it doesn't know it. */
    private final int dim;
    private final int maxBatch;
    private final long maxDelayNanos;
    private final LinkedBlockingQueue<Request> queue = new LinkedBlockingQueue<>();
    private final Thread dispatcher;
    private final AtomicLong batches = new AtomicLong();
    private final AtomicLong queries = new AtomicLong();
    private volatile boolean closed;

    private record Request(float[] query, CompletableFuture<List<NodeScore>> result) {}

    IvfAshBatchingSearcher(Function<Vecs, SearchResult> search, int dim, int maxBatch, Duration maxDelay) {
        if (maxBatch < 1) {
            throw new IllegalArgumentException("maxBatch must be at least 1, got " + maxBatch);
        }
        if (maxDelay.isNegative()) {
            throw new IllegalArgumentException("maxDelay must not be negative, got " + maxDelay);
        }
        this.search = search;
        this.dim = dim;
        this.maxBatch = maxBatch;
        this.maxDelayNanos = maxDelay.toNanos();
        this.dispatcher = new Thread(this::dispatch, "ivfash-batching-searcher");
        dispatcher.setDaemon(true);
        dispatcher.start();
    }

    /** A builder that wraps {@code searcher}, which must stay open until this is closed. */
    public static Builder builder(IvfAshIndexSearcher searcher) {
        return new Builder(searcher);
    }

    /**
     * Searches one query and waits for its result.
     *
     * @param query the query vector; copied, so the caller may reuse the array
     * @return up to {@code topk} neighbours, best first
     * @throws IllegalArgumentException if the query's dimension doesn't match the index's
     * @throws IllegalStateException    if this is closed, or closes before the query is searched
     * @throws FvecException            if the native search fails
     */
    public List<NodeScore> search(float[] query) {
        try {
            return searchAsync(query).join();
        } catch (CompletionException e) {
            if (e.getCause() instanceof RuntimeException cause) {
                throw cause;
            }
            throw e;
        }
    }

    /**
     * Queues one query. The future fails with the exceptions {@link #search(float[])} throws.
     *
     * @param query the query vector; copied, so the caller may reuse the array
     * @throws IllegalArgumentException if the query's dimension doesn't match the index's
     * @throws IllegalStateException    if this is closed
     */
    public CompletableFuture<List<NodeScore>> searchAsync(float[] query) {
        if (dim != 0 && query.length != dim) {
            throw new IllegalArgumentException("query dim mismatch: index dim=" + dim + ", query dim=" + query.length);
        }
        if (closed) {
            throw new IllegalStateException("IvfAshBatchingSearcher is already closed");
        }
        Request request = new Request(query.clone(), new CompletableFuture<>());
        queue.add(request);
        // close() may have drained the queue just before the add; if so, nothing will search it
        if (closed && queue.remove(request)) {
            request.result().completeExceptionally(closedException());
        }
        return request.result();
    }

    /** Number of native search calls made so far. */
    public long batchCount() {
        return batches.get();
    }

    /** Number of queries searched so far; divided by {@link #batchCount()}, the mean batch size. */
    public long queryCount() {
        return queries.get();
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
                queue.drainTo(batch, maxBatch - batch.size());
                if (maxDelayNanos > 0) {
                    long deadline = System.nanoTime() + maxDelayNanos;
                    while (batch.size() < maxBatch) {
                        Request next = queue.poll(deadline - System.nanoTime(), TimeUnit.NANOSECONDS);
                        if (next == null) {
                            break;
                        }
                        batch.add(next);
                        queue.drainTo(batch, maxBatch - batch.size());
                    }
                }
                execute(batch);
                batch.clear();
            }
        } catch (InterruptedException e) {
            // close() stops the dispatcher this way
        } finally {
            failAll(batch);
        }
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

    /** Builder for {@link IvfAshBatchingSearcher}. Obtain via {@link #builder(IvfAshIndexSearcher)}. */
    public static final class Builder {
        private final IvfAshIndexSearcher searcher;
        private int maxBatch = DEFAULT_MAX_BATCH;
        private Duration maxDelay = Duration.ZERO;

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
            this.maxDelay = maxDelay;
            return this;
        }

        /** Starts the dispatcher thread. */
        public IvfAshBatchingSearcher build() {
            return new IvfAshBatchingSearcher(searcher::search, searcher.dim(), maxBatch, maxDelay);
        }
    }
}
