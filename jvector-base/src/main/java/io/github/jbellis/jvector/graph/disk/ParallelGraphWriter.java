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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.disk.RandomAccessWriter;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.AsynchronousFileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

import java.util.EnumSet;
import java.util.List;
import java.util.Objects;
import java.util.Map;
import java.util.ArrayList;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntFunction;

/**
 * Orchestrates parallel writing of L0 node records to disk using asynchronous file I/O.
 * <p>
 * This class manages:
 * - A thread pool for building node records in parallel
 * - Per-thread ImmutableGraphIndex.View instances for thread-safe neighbor iteration
 * - A buffer pool to avoid excessive allocation
 * - Asynchronous file channel writes using position-based writes
 * <p>
 * The new architecture writes directly to the AsynchronousFileChannel using position-based
 * writes, which allows it to handle pre-written features the same way as the sequential
 * writer: by checking for null suppliers and skipping those features.
 * <p>
 * Usage:
 * <pre>
 * try (var parallelWriter = new ParallelGraphWriter(writer, graph, config, filePath)) {
 *     parallelWriter.writeL0Records(ordinalMapper, inlineFeatures, featureStateSuppliers, baseOffset);
 * }
 * </pre>
 */
class ParallelGraphWriter implements AutoCloseable {
    private final RandomAccessWriter writer;
    private final ImmutableGraphIndex graph;
    private final ExecutorService executor;
    private final ThreadLocal<ImmutableGraphIndex.View> viewPerThread;
    private final ThreadLocal<ByteBuffer> bufferPerThread;
    private final CopyOnWriteArrayList<ImmutableGraphIndex.View> allViews = new CopyOnWriteArrayList<>();
    private final int recordSize;
    private final Path filePath;
    private final int taskMultiplier;
    private final int maxInFlightWrites;
    private static final AtomicInteger threadCounter = new AtomicInteger(0);

    /**
     * Configuration for parallel writing.
     */
    static class Config {
        final int workerThreads;
        final boolean useDirectBuffers;
        final int taskMultiplier;
        final int maxInFlightWrites;

        /**
         * @param workerThreads number of worker threads for building records (0 = use available processors)
         * @param useDirectBuffers whether to use direct ByteBuffers (can be faster for large records)
         * @param taskMultiplier multiplier for number of tasks relative to worker threads
         *                       (4x = good balance for most use cases, higher = more fine-grained parallelism)
         * @param maxInFlightWrites maximum number of concurrent async write operations (0 = use cores * 100)
         */
        public Config(int workerThreads, boolean useDirectBuffers, int taskMultiplier, int maxInFlightWrites) {
            this.workerThreads = workerThreads <= 0 ? Runtime.getRuntime().availableProcessors() : workerThreads;
            this.useDirectBuffers = useDirectBuffers;
            this.taskMultiplier = taskMultiplier <= 0 ? 4 : taskMultiplier;
            this.maxInFlightWrites = maxInFlightWrites <= 0 ? Runtime.getRuntime().availableProcessors() * 100 : maxInFlightWrites;
        }

        /**
         * Convenience constructor with default maxInFlightWrites.
         *
         * @param workerThreads number of worker threads for building records (0 = use available processors)
         * @param useDirectBuffers whether to use direct ByteBuffers (can be faster for large records)
         * @param taskMultiplier multiplier for number of tasks relative to worker threads
         */
        public Config(int workerThreads, boolean useDirectBuffers, int taskMultiplier) {
            this(workerThreads, useDirectBuffers, taskMultiplier, 0);
        }

        /**
         * Returns a default configuration suitable for most use cases.
         * Uses available CPU cores, heap buffers, 4x task multiplier, and cores * 100 max in-flight writes.
         *
         * @return default configuration
         */
        public static Config defaultConfig() {
            return new Config(0, false, 4, 0);
        }
    }

    /**
     * Creates a parallel writer.
     *
     * @param writer the underlying writer
     * @param graph the graph being written
     * @param inlineFeatures the inline features to write
     * @param config parallelization configuration
     * @param filePath file path for async writes (required, cannot be null)
     */
    public ParallelGraphWriter(RandomAccessWriter writer,
                               ImmutableGraphIndex graph,
                               List<Feature> inlineFeatures,
                               Config config,
                               Path filePath) {
        this.writer = writer;
        this.graph = graph;
        this.filePath = Objects.requireNonNull(filePath);
        this.taskMultiplier = config.taskMultiplier;
        this.maxInFlightWrites = config.maxInFlightWrites;
        this.executor = Executors.newFixedThreadPool(config.workerThreads,
            r -> {
                Thread t = new Thread(r);
                t.setName("ParallelGraphWriter-Worker-" + threadCounter.getAndIncrement());
                t.setDaemon(false);
                return t;
            });

        // Compute fixed record size for L0
        this.recordSize = Integer.BYTES // node ordinal
            + inlineFeatures.stream().mapToInt(Feature::featureSize).sum()
            + Integer.BYTES // neighbor count
            + graph.getDegree(0) * Integer.BYTES; // neighbors + padding

        // Thread-local views for safe neighbor iteration
        // CopyOnWriteArrayList handles concurrent additions safely
        this.viewPerThread = ThreadLocal.withInitial(() -> {
            var view = graph.getView();
            allViews.add(view);
            return view;
        });

        // Thread-local buffers to avoid allocation overhead
        // Use BIG_ENDIAN to match Java DataOutput specification
        final int bufferSize = recordSize;
        final boolean useDirect = config.useDirectBuffers;
        this.bufferPerThread = ThreadLocal.withInitial(() -> {
            ByteBuffer buffer = useDirect ? ByteBuffer.allocateDirect(bufferSize) : ByteBuffer.allocate(bufferSize);
            buffer.order(java.nio.ByteOrder.BIG_ENDIAN);
            return buffer;
        });
    }

    /**
     * Writes all L0 node records in parallel using truly asynchronous file I/O with range-based task batching.
     * <p>
     * This implementation separates record building (CPU-bound) from I/O operations (I/O-bound) to achieve
     * true asynchronous execution. Worker threads build records in memory and submit them for async writing
     * without blocking, allowing the thread to continue building more records while previous writes are in flight.
     * <p>
     * The implementation uses:
     * - AsyncWriteCoordinator: Manages async writes with backpressure control
     * - AsyncNodeRecordTask: Builds records and submits writes without blocking
     * - CompletableFuture: Tracks write completion across all tasks
     * <p>
     * The number of tasks is determined by available CPU cores multiplied by a configurable multiplier.
     * This provides good load balancing while minimizing task creation and management overhead.
     * <p>
     * Features that have been pre-written (via writeInline) are detected by null suppliers and skipped
     * during the write process, similar to how the sequential writer handles this case.
     *
     * @param ordinalMapper maps between old and new ordinals
     * @param inlineFeatures the inline features to write
     * @param featureStateSuppliers suppliers for feature state (null = feature already written)
     * @param baseOffset the file offset where L0 records start
     * @throws IOException if an IO error occurs
     */
    public void writeL0Records(OrdinalMapper ordinalMapper,
                               List<Feature> inlineFeatures,
                               Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers,
                               long baseOffset) throws IOException {
        int maxOrdinal = ordinalMapper.maxOrdinal();
        int totalOrdinals = maxOrdinal + 1;

        // Calculate optimal number of tasks based on cores and task multiplier
        int numCores = Runtime.getRuntime().availableProcessors();
        int numTasks = Math.max(1, Math.min(numCores * taskMultiplier, totalOrdinals));

        // Calculate ordinals per task (ceiling division to cover all ordinals)
        int ordinalsPerTask = (totalOrdinals + numTasks - 1) / numTasks;

        // Open the async file channel for position-based writes
        // Important: Don't pass our executor to the channel - it would cause deadlock since
        // worker threads would block waiting for writes that need threads from the same pool.
        // Using the default system thread pool for I/O operations avoids this issue.
        var opts = new java.util.HashSet<java.nio.file.OpenOption>();
        opts.add(StandardOpenOption.WRITE);
        opts.add(StandardOpenOption.READ);
        
        try (var channel = AsynchronousFileChannel.open(filePath, opts, null);
             var writeCoordinator = new AsyncWriteCoordinator(channel,
                 this.maxInFlightWrites)) {
            
            List<Future<List<CompletableFuture<Void>>>> taskFutures = new ArrayList<>(numTasks);

            // Submit range-based tasks that build records and submit async writes
            for (int i = 0; i < numTasks; i++) {
                int startOrdinal = i * ordinalsPerTask;
                int endOrdinal = Math.min(startOrdinal + ordinalsPerTask, totalOrdinals);

                // Skip if range is empty (can happen with final task)
                if (startOrdinal >= totalOrdinals) {
                    break;
                }

                final int start = startOrdinal;
                final int end = endOrdinal;

                Future<List<CompletableFuture<Void>>> future = executor.submit(() -> {
                    var view = viewPerThread.get();
                    var buffer = bufferPerThread.get();

                    var task = new AsyncNodeRecordTask(
                            start,                    // Start of range (inclusive)
                            end,                      // End of range (exclusive)
                            ordinalMapper,
                            graph,
                            view,
                            inlineFeatures,
                            featureStateSuppliers,
                            recordSize,
                            baseOffset,               // Base offset (task calculates per-ordinal offsets)
                            writeCoordinator,         // Async write coordinator
                            buffer                    // Thread-local buffer
                    );

                    return task.call();
                });

                taskFutures.add(future);
            }

            // Collect all write futures from all tasks
            List<CompletableFuture<Void>> allWriteFutures = new ArrayList<>();
            for (Future<List<CompletableFuture<Void>>> taskFuture : taskFutures) {
                try {
                    List<CompletableFuture<Void>> writeFutures = taskFuture.get();
                    allWriteFutures.addAll(writeFutures);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IOException("Interrupted while building records", e);
                } catch (ExecutionException e) {
                    throw unwrapExecutionException(e);
                }
            }

            // Wait for all writes to complete
            try {
                CompletableFuture.allOf(allWriteFutures.toArray(new CompletableFuture[0])).join();
            } catch (Exception e) {
                // Find and throw the first exception
                for (CompletableFuture<Void> future : allWriteFutures) {
                    if (future.isCompletedExceptionally()) {
                        try {
                            future.join(); // This will throw
                        } catch (Exception ex) {
                            Throwable cause = ex.getCause();
                            if (cause instanceof IOException) {
                                throw (IOException) cause;
                            } else if (cause instanceof RuntimeException) {
                                throw (RuntimeException) cause;
                            } else {
                                throw new IOException("Write operation failed", cause);
                            }
                        }
                    }
                }
                // If we get here, rethrow the original exception
                throw new IOException("Write operation failed", e);
            }

            // Force all writes to disk before closing the channel.
            // This guarantees data durability and prevents intermittent test failures where
            // subsequent reads might see stale data or zeros.
            channel.force(true);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted during async write coordination", e);
        }
    }

    /**
     * Unwraps ExecutionException to throw the underlying cause.
     * Handles IOException, RuntimeException, and wraps other exceptions.
     *
     * @param e the execution exception to unwrap
     * @return an IOException wrapping the cause
     * @throws RuntimeException if the cause is a RuntimeException
     */
    private IOException unwrapExecutionException(ExecutionException e) {
        Throwable cause = e.getCause();
        if (cause instanceof IOException) {
            return (IOException) cause;
        } else if (cause instanceof RuntimeException) {
            throw (RuntimeException) cause;
        } else {
            throw new RuntimeException("Error building node record", cause);
        }
    }

    /**
     * Returns the computed record size for L0 nodes.
     */
    public int getRecordSize() {
        return recordSize;
    }

    @Override
    public void close() throws IOException {
        try {
            // Shutdown executor
            executor.shutdown();
            try {
                if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }

            // Close all views (CopyOnWriteArrayList is safe for concurrent iteration)
            for (var view : allViews) {
                view.close();
            }
            allViews.clear();
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Error closing parallel writer", e);
        }
    }
}

