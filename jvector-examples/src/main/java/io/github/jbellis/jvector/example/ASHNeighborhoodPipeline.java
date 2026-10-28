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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.quantization.ASHNeighborhoodScorer;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import java.util.ArrayDeque;
import java.util.Objects;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Bounded benchmark pipeline: one coordinator prepares queries while a persistent pool scores.
 *
 * <p>Submit, measure and close belong to the coordinator thread. At most two scans are submitted
 * by {@link #measure}; every scan owns its prepared query and every pool thread owns its scratch.
 * Timing includes preparation, scheduling, score consumption, pipeline fill and complete drain.
 * Average elapsed time per completed scan is a throughput metric, not per-query latency.</p>
 */
final class ASHNeighborhoodPipeline implements AutoCloseable {
    private static final int MAX_IN_FLIGHT = 2;
    private static volatile double sink;
    private final ASHNeighborhoodFixture fixture;
    private final int[] order;
    private final int threads;
    private final ThreadPoolExecutor pool;
    private final ThreadLocal<float[]> output;
    private long preparationNanos;

    ASHNeighborhoodPipeline(ASHNeighborhoodFixture fixture, int[] order, int threads) {
        if (threads < 1 || threads > order.length)
            throw new IllegalArgumentException("Worker count must be in [1, neighborhood count]");
        this.fixture = Objects.requireNonNull(fixture, "fixture");
        validateOrder(order, fixture.neighborhoods.length);
        this.order = order.clone();
        this.threads = threads;
        this.output = ThreadLocal.withInitial(() -> new float[fixture.capacity]);
        var workerIds = new AtomicInteger();
        pool = (ThreadPoolExecutor) Executors.newFixedThreadPool(threads, task -> {
            var worker = new Thread(task, "ash-neighborhood-" + workerIds.incrementAndGet());
            worker.setDaemon(true); // An aborted benchmark must not leave its process alive.
            return worker;
        });
        pool.prestartAllCoreThreads();
    }

    /** Prepare exactly once, then publish the immutable scorer with each partition task. */
    Scan submit(VectorFloat<?> query) {
        if (pool.isShutdown()) {
            throw new IllegalStateException("ASH pipeline is closed");
        }
        long start = System.nanoTime();
        var scorer = new ASHNeighborhoodScorer(fixture.ash, query, fixture.capacity, fixture.blockSize);
        preparationNanos += System.nanoTime() - start;
        var scan = new Scan(threads);
        for (int worker = 0; worker < threads; worker++) {
            final int slot = worker;
            final int first = (int) ((long) worker * order.length / threads);
            final int last = (int) ((long) (worker + 1) * order.length / threads);
            var partition = new FutureTask<Void>(() -> {
                float[] scores = output.get();
                double checksum = 0;
                for (int n = first; n < last; n++) {
                    scorer.score(fixture.neighborhoods[order[n]], scores);
                    for (float score : scores)
                        checksum += score;
                }
                scan.checksums[slot] = checksum;
                return null;
            }) {
                @Override
                protected void done() {
                    // Also complete the scan when shutdown cancels a queued partition.
                    try {
                        get(); // Already terminal: no waiting here.
                    } catch (ExecutionException error) {
                        scan.failure.compareAndSet(null, error.getCause());
                    } catch (CancellationException error) {
                        scan.failure.compareAndSet(null, error);
                    } catch (InterruptedException error) {
                        Thread.currentThread().interrupt();
                        scan.failure.compareAndSet(null, error);
                    } finally {
                        scan.done.countDown();
                    }
                }
            };
            pool.execute(partition);
        }
        return scan;
    }

    @Override
    public void close() {
        pool.shutdown();
        boolean interrupted = false;
        try {
            while (!pool.isTerminated()) {
                try {
                    pool.awaitTermination(1, TimeUnit.SECONDS);
                } catch (InterruptedException error) {
                    interrupted = true;
                    for (Runnable queued : pool.shutdownNow()) {
                        ((Future<?>) queued).cancel(false);
                    }
                }
            }
        } finally {
            if (interrupted)
                Thread.currentThread().interrupt();
        }
    }

    private static void validateOrder(int[] order, int count) {
        if (order.length != count) {
            throw new IllegalArgumentException("A full scan must include every neighborhood exactly once");
        }
        boolean[] seen = new boolean[count];
        for (int index : order) {
            if (index < 0 || index >= count || seen[index]) {
                throw new IllegalArgumentException("Neighborhood order must be a permutation");
            }
            seen[index] = true;
        }
    }

    /** Completion state for one full scan; the latch publishes each partition's checksum. */
    static final class Scan {
        private final CountDownLatch done;
        private final double[] checksums;
        private final AtomicReference<Throwable> failure = new AtomicReference<>();

        Scan(int threads) {
            done = new CountDownLatch(threads);
            checksums = new double[threads];
        }

        double await() {
            try {
                done.await();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("Interrupted while scoring", e);
            }
            if (failure.get() != null)
                throw new IllegalStateException("Neighborhood scoring worker failed", failure.get());
            double sum = 0;
            for (double checksum : checksums)
                sum += checksum;
            return sum;
        }
    }

    static final class Timing {
        final double nanosPerScan, preparationNanosPerScan, checksum;
        Timing(double nanos, double preparation, double checksum) {
            this.nanosPerScan = nanos;
            this.preparationNanosPerScan = preparation;
            this.checksum = checksum;
        }
    }

    Timing measure(VectorFloat<?> query, double seconds) {
        var pending = new ArrayDeque<Scan>(MAX_IN_FLIGHT);
        long preparationStart = preparationNanos;
        long start = System.nanoTime();
        long completed = 0;
        double checksum = 0;
        do {
            if (pending.size() == MAX_IN_FLIGHT) {
                checksum += pending.removeFirst().await();
                completed++;
            }
            // Preparation of this scan overlaps scoring of earlier submitted scans.
            pending.addLast(submit(query));
        } while (System.nanoTime() - start < seconds * 1e9);
        while (!pending.isEmpty()) {
            checksum += pending.removeFirst().await();
            completed++;
        }
        long elapsed = System.nanoTime() - start;
        sink = checksum / completed;
        return new Timing(
                (double) elapsed / completed, (double) (preparationNanos - preparationStart) / completed, sink);
    }
}
