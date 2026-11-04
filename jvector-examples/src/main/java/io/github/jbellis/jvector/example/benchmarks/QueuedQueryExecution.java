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
package io.github.jbellis.jvector.example.benchmarks;

import io.github.jbellis.jvector.example.Grid.ConfiguredSystem;
import io.github.jbellis.jvector.example.benchmarks.diagnostics.BenchmarkDiagnostics.QueryTimeRecorder;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.similarity.PreparedQuery;
import io.github.jbellis.jvector.graph.similarity.QueryPreparer;
import io.github.jbellis.jvector.util.BatchingExecutor;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.LongArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.IntFunction;

/**
 * The queued execution policy used by the standard BenchYAML benchmarks. It
 * supplies the same full passes and query sources as the calling benchmark;
 * warmup, sampling, GC policy and reporting remain in those benchmarks.
 */
public final class QueuedQueryExecution implements AutoCloseable {
    private static final QueryTimeRecorder NO_TIMING = nanos -> {};
    private final ConfiguredSystem system;
    private final int topK, rerankK;
    private final boolean usePruning;
    private final QueryPreparer preparer;
    private final BatchingExecutor<Request> execution;
    private final ArrayBlockingQueue<Request> free;
    private LongArrayList measuredResponseTimes = new LongArrayList();
    private long[] initialHistogram;
    private boolean measuring;

    private static final class Rows {
        final VectorFloat<?>[] queries;
        final PreparedQuery[] states;
        Rows(int count) { queries = new VectorFloat<?>[count]; states = new PreparedQuery[count]; }
    }

    private static final class Request {
        final PreparedQuery state;
        final Semaphore completed = new Semaphore(0);
        Pass pass;
        int ordinal;
        long submittedAt;
        VectorFloat<?> query;
        SearchResult result;
        Throwable failure;
        Request(PreparedQuery state) { this.state = state; }
    }

    public static final class Pass {
        final IntFunction<VectorFloat<?>> inputs;
        final QueryTimeRecorder recorder;
        final CountDownLatch completed;
        final AtomicReference<Throwable> failure = new AtomicReference<>();
        final LongAdder visited = new LongAdder();
        final long[] responseTimes;
        public final SearchResult[] results;
        Pass(int count, IntFunction<VectorFloat<?>> inputs, QueryTimeRecorder recorder, boolean capture) {
            this.inputs = inputs; this.recorder = recorder;
            completed = new CountDownLatch(count);
            responseTimes = new long[count];
            results = capture ? new SearchResult[count] : null;
        }
        public long visited() { return visited.sum(); }
    }

    public QueuedQueryExecution(ConfiguredSystem system, int topK, int rerankK, boolean usePruning) {
        if (topK <= 0 || rerankK < topK)
            throw new IllegalArgumentException("Require 0 < topK <= rerankK");
        this.system = Objects.requireNonNull(system, "system");
        this.topK = topK; this.rerankK = rerankK; this.usePruning = usePruning;
        this.preparer = system.queryPreparer();
        int cores = system.queryWorkerCount();
        int batchSize = system.queryBatchSize();
        if (batchSize <= 0) throw new IllegalArgumentException("Benchmark query batch size must be positive");
        // One full default batch per worker bounds waiting while permitting
        // genuine GEMM preparation. Keep the same admission limit for B=1
        // controls; larger explicitly selected batches remain supported.
        int capacity = Math.multiplyExact(Math.max(8, batchSize), cores);
        free = new ArrayBlockingQueue<>(capacity);
        for (int i = 0; i < capacity; i++) free.add(new Request(preparer == null ? null : preparer.newState()));
        var rows = ThreadLocal.withInitial(() -> new Rows(batchSize));
        var workers = system.queryWorkers();
        execution = new BatchingExecutor<>(cores, capacity, batchSize, workers,
                    new BatchingExecutor.Operation<Request>() {
                public Object batchKey(Request request) {
                    return preparer != null && preparer.supportsBatchPreparation() ? preparer : null;
                }
                public void prepareBatch(List<Request> requests) {
                    for (Request request : requests)
                        if (request.pass != null) request.query = request.pass.inputs.apply(request.ordinal);
                    if (preparer == null) return;
                    Rows buffers = rows.get();
                    try {
                        for (int i = 0; i < requests.size(); i++) {
                            buffers.queries[i] = requests.get(i).query;
                            buffers.states[i] = requests.get(i).state;
                        }
                        preparer.prepareBatch(buffers.queries, buffers.states, requests.size());
                    } finally {
                        Arrays.fill(buffers.queries, 0, requests.size(), null);
                        Arrays.fill(buffers.states, 0, requests.size(), null);
                    }
                }
                public void execute(Request request) {
                    request.result = request.state == null
                            ? QueryExecutor.executeQuery(system, topK, rerankK, usePruning, request.query)
                            : QueryExecutor.executePreparedQuery(system, topK, rerankK, usePruning, request.state);
                }
                public void complete(Request request, Throwable failure) {
                    long responseTime = System.nanoTime() - request.submittedAt;
                    request.failure = failure;
                    Pass pass = request.pass;
                    if (pass == null) { request.completed.release(); return; }
                    try {
                        if (failure == null) {
                            pass.visited.add(request.result.getVisitedCount());
                            pass.responseTimes[request.ordinal] = responseTime;
                            pass.recorder.recordTime(responseTime);
                            if (pass.results != null) pass.results[request.ordinal] = request.result;
                        } else pass.failure.compareAndSet(null, failure);
                    } catch (Throwable error) { pass.failure.compareAndSet(null, error); }
                    finally {
                        release(request);
                        pass.completed.countDown();
                    }
                }
            });
        System.out.printf("QUERY_EXECUTION mode=queued workers=%d capacity=%d max_batch_size=%d optimized_batch=%s%n",
                cores, capacity, batchSize, preparer != null && preparer.supportsBatchPreparation());
    }

    private Request acquire() {
        boolean interrupted = false;
        for (;;) {
            try { Request request = free.take(); if (interrupted) Thread.currentThread().interrupt(); return request; }
            catch (InterruptedException e) { interrupted = true; }
        }
    }

    private void release(Request request) {
        request.query = null; request.result = null; request.failure = null; request.pass = null;
        free.add(request);
    }

    private void submit(Request request) {
        boolean accepted = false;
        try {
            accepted = execution.trySubmit(request);
            if (!accepted) throw new IllegalStateException("Admission accounting mismatch");
        } finally {
            // A throwing or rejected submission never transferred ownership.
            if (!accepted) release(request);
        }
    }

    public Pass runPass(int count, IntFunction<VectorFloat<?>> source, QueryTimeRecorder recorder, boolean capture) {
        if (count < 0) throw new IllegalArgumentException("Query count must be nonnegative");
        Objects.requireNonNull(source, "source");
        Pass pass = new Pass(count, source, recorder == null ? NO_TIMING : recorder, capture);
        for (int i = 0; i < count; i++) {
            long submittedAt = System.nanoTime(); // Includes waiting for admission.
            Request request = acquire();
            request.pass = pass; request.ordinal = i; request.submittedAt = submittedAt;
            submit(request);
        }
        boolean interrupted = false;
        for (;;) {
            try { pass.completed.await(); break; }
            catch (InterruptedException e) { interrupted = true; }
        }
        if (interrupted) Thread.currentThread().interrupt();
        rethrow(pass.failure.get());
        if (measuring) for (long time : pass.responseTimes) measuredResponseTimes.addLong(time);
        return pass;
    }

    /** One submitted request with normal queue/batch capacity; not a singleton setting. */
    public SearchResult executeSingle(VectorFloat<?> query) {
        Objects.requireNonNull(query, "query");
        long submittedAt = System.nanoTime();
        Request request = acquire();
        request.pass = null; request.query = query; request.submittedAt = submittedAt;
        submit(request);
        request.completed.acquireUninterruptibly();
        try { rethrow(request.failure); return request.result; }
        finally { release(request); }
    }

    private static void rethrow(Throwable error) {
        if (error instanceof Error) throw (Error) error;
        if (error instanceof RuntimeException) throw (RuntimeException) error;
        if (error != null) throw new RuntimeException("Queued query failed", error);
    }

    public void beginMeasurements() {
        if (measuring || measuredResponseTimes == null)
            throw new IllegalStateException("Measurements already started or transferred");
        initialHistogram = execution.batchHistogram();
        measuring = true;
    }

    /**
     * Transfers completed samples without copying the large response buffer.
     * Call after measured passes complete and before closing this execution.
     * The returned value does not retain this executor, requests, or model.
     */
    public Measurements finishMeasurements() {
        if (!measuring) throw new IllegalStateException("Measurements are not active");
        long[] histogram = execution.batchHistogram();
        for (int i = 1; i < histogram.length; i++) histogram[i] -= initialHistogram[i];
        var snapshot = new Measurements(measuredResponseTimes, histogram);
        measuredResponseTimes = null;
        measuring = false;
        return snapshot;
    }

    /** Completed samples, safe to aggregate after the execution has closed. */
    public static final class Measurements {
        private final LongArrayList responseTimes;
        private final long[] histogram;

        private Measurements(LongArrayList responseTimes, long[] histogram) {
            this.responseTimes = responseTimes;
            this.histogram = histogram;
        }

        /** Sorting and formatting occur only when the owner requests reporting. */
        public List<Metric> metrics() {
            long[] times = responseTimes.toLongArray();
            Arrays.sort(times);
            double sum = 0;
            for (long time : times) sum += time;
            long groups = 0, queries = 0;
            for (int i = 1; i < histogram.length; i++) {
                groups += histogram[i];
                queries += i * histogram[i];
            }
            double meanBatch = groups == 0 ? 0 : (double) queries / groups;
            System.out.printf("QUERY_BATCHES queries=%d groups=%d mean=%.4f histogram=%s%n",
                    queries, groups, meanBatch, Arrays.toString(histogram));
            return List.of(
                    Metric.of("search.execution.response_mean_ms", "Loaded mean (ms)", ".3f", times.length == 0 ? 0 : sum / times.length / 1e6),
                    Metric.of("search.execution.response_p95_ms", "Loaded p95 (ms)", ".3f", percentile(times, .95)),
                    Metric.of("search.execution.response_p99_ms", "Loaded p99 (ms)", ".3f", percentile(times, .99)),
                    Metric.of("search.execution.mean_batch_size", "Mean batch", ".3f", meanBatch));
        }
    }

    private static double percentile(long[] sorted, double p) {
        return sorted.length == 0 ? 0 : sorted[Math.min(sorted.length - 1, (int) Math.ceil(sorted.length * p) - 1)] / 1e6;
    }

    @Override
    public void close() {
        execution.close(); // ConfiguredSystem owns the workers and their reusable search state.
    }
}
