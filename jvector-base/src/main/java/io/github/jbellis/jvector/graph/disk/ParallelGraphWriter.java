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
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntFunction;

/**
 * Orchestrates parallel writing of L0 node records to disk using asynchronous file I/O.
 * <p>
 * This class manages:
 * - A thread pool for building node records in parallel
 * - Per-thread ImmutableGraphIndex.View instances for thread-safe neighbor iteration
 * - A buffer pool to avoid excessive allocation
 * - Asynchronous file channel writes that maintain correct ordering
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
    private static final AtomicInteger threadCounter = new AtomicInteger(0);

    /**
     * Configuration for parallel writing.
     */
    static class Config {
        final int workerThreads;
        final boolean useDirectBuffers;

        /**
         * @param workerThreads number of worker threads for building records (0 = use available processors)
         * @param useDirectBuffers whether to use direct ByteBuffers (can be faster for large records)
         */
        public Config(int workerThreads, boolean useDirectBuffers) {
            this.workerThreads = workerThreads <= 0 ? Runtime.getRuntime().availableProcessors() : workerThreads;
            this.useDirectBuffers = useDirectBuffers;
        }

        /**
         * Returns a default configuration suitable for most use cases.
         * Uses available CPU cores and heap buffers.
         *
         * @return default configuration
         */
        public static Config defaultConfig() {
            return new Config(0, false);
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
     * Writes all L0 node records in parallel using asynchronous file I/O with memory-aware backpressure.
     * Records are written in order to maintain index correctness. The implementation uses a streaming
     * approach with bounded memory usage to prevent heap exhaustion on large graphs.
     * <p>
     * The method builds records in chunks sized according to available heap memory. When a chunk is full,
     * it is written to disk before building the next chunk. This provides natural backpressure and prevents
     * memory issues when processing very large graphs.
     *
     * @param ordinalMapper maps between old and new ordinals
     * @param inlineFeatures the inline features to write
     * @param featureStateSuppliers suppliers for feature state
     * @param baseOffset the file offset where L0 records start
     * @throws IOException if an IO error occurs
     */
    public void writeL0Records(OrdinalMapper ordinalMapper,
                               List<Feature> inlineFeatures,
                               Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers,
                               long baseOffset) throws IOException {
        // Calculate optimal buffer size based on available heap memory
        int bufferSize = calculateOptimalBufferSize();
        int maxOrdinal = ordinalMapper.maxOrdinal();
        List<Future<NodeRecordTask.Result>> futures = new ArrayList<>(bufferSize);

        // Build and write records in chunks to manage memory usage
        for (int newOrdinal = 0; newOrdinal <= maxOrdinal; newOrdinal++) {
            final int ordinal = newOrdinal;
            final long fileOffset = baseOffset + (long) ordinal * recordSize;

            // Submit task to build this record
            Future<NodeRecordTask.Result> future = executor.submit(() -> {
                var view = viewPerThread.get();
                var buffer = bufferPerThread.get();

                var task = new NodeRecordTask(
                        ordinal,
                        ordinalMapper,
                        graph,
                        view,
                        inlineFeatures,
                        featureStateSuppliers,
                        recordSize,
                        fileOffset,
                        buffer
                );

                return task.call();
            });

            futures.add(future);

            // When buffer is full, write this batch and clear for next batch
            if (futures.size() >= bufferSize) {
                writeRecordsAsync(futures);
                futures.clear();
            }
        }

        // Write any remaining records
        if (!futures.isEmpty()) {
            writeRecordsAsync(futures);
        }
    }

    /**
     * Calculates the optimal buffer size based on available heap memory.
     * Uses a conservative approach to prevent memory exhaustion while maintaining
     * good parallelism and throughput.
     * <p>
     * The calculation considers:
     * - Current heap usage and maximum heap size
     * - Estimated memory per record (data + Future/Result overhead)
     * - Conservative allocation (20% of available memory)
     * - Reasonable bounds (min 100, max 1000000 records)
     *
     * @return the number of records to buffer before writing
     */
    private int calculateOptimalBufferSize() {
        Runtime runtime = Runtime.getRuntime();
        long maxMemory = runtime.maxMemory();
        long usedMemory = runtime.totalMemory() - runtime.freeMemory();
        long availableMemory = maxMemory - usedMemory;

        // Use 20% of available memory for buffering to be conservative
        long memoryForBuffer = (long) (availableMemory * 0.2);

        // Estimate memory per record:
        // - ByteBuffer copy (recordSize bytes)
        // - Future wrapper and Result object overhead (~1KB)
        int estimatedMemoryPerRecord = recordSize + 1024;

        int calculatedBufferSize = (int) (memoryForBuffer / estimatedMemoryPerRecord);

        // Ensure buffer size is reasonable:
        // - Minimum 100 records to ensure some parallelism
        // - Maximum 1000000 records to cap memory usage
        return Math.max(100, Math.min(calculatedBufferSize, 1000000));
    }

    /**
     * Writes records asynchronously using AsynchronousFileChannel for improved throughput.
     * Records are written in sequential order by iterating through the futures list, which
     * ensures that even though record building is parallelized, writes occur in the correct order.
     * Creates a dedicated thread pool for async I/O operations and properly cleans up resources.
     *
     * @param futures the completed record building tasks
     * @throws IOException if an I/O error occurs
     */
    private void writeRecordsAsync(List<Future<NodeRecordTask.Result>> futures) throws IOException {
        var opts = EnumSet.of(StandardOpenOption.WRITE, StandardOpenOption.READ);
        int numThreads = Math.min(Runtime.getRuntime().availableProcessors(), 32);
        ExecutorService fileWritePool = null;

        try {
            fileWritePool = new ThreadPoolExecutor(
                    numThreads, numThreads,
                    0L, TimeUnit.MILLISECONDS,
                    new LinkedBlockingQueue<>(),
                    r -> {
                        var t = new Thread(r, "graphnode-writer");
                        t.setDaemon(true);
                        return t;
                    });

            // Use a bounded list to allow multiple concurrent async writes while providing backpressure
            // Buffer size is 2x the I/O thread pool size to keep the pipeline full
            int maxConcurrentWrites = numThreads * 2;
            List<Future<Integer>> pendingWrites = new ArrayList<>(maxConcurrentWrites);

            try (var afc = AsynchronousFileChannel.open(filePath, opts, fileWritePool)) {
                for (Future<NodeRecordTask.Result> future : futures) {
                    NodeRecordTask.Result result = future.get();

                    // Submit async write and track the future
                    // result.data is already a copy made in NodeRecordTask to avoid
                    // race conditions with thread-local buffer reuse
                    Future<Integer> writeFuture = afc.write(result.data, result.fileOffset);
                    pendingWrites.add(writeFuture);

                    // When buffer is full, wait for all pending writes to complete
                    if (pendingWrites.size() >= maxConcurrentWrites) {
                        for (Future<Integer> wf : pendingWrites) {
                            wf.get(); // Wait for write completion
                        }
                        pendingWrites.clear();
                    }
                }

                // Wait for any remaining pending writes
                for (Future<Integer> wf : pendingWrites) {
                    wf.get();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted while writing records", e);
            } catch (ExecutionException e) {
                throw unwrapExecutionException(e);
            }
        } finally {
            if (fileWritePool != null) {
                fileWritePool.shutdown();
                try {
                    if (!fileWritePool.awaitTermination(60, TimeUnit.SECONDS)) {
                        fileWritePool.shutdownNow();
                    }
                } catch (InterruptedException e) {
                    fileWritePool.shutdownNow();
                    Thread.currentThread().interrupt();
                }
            }
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
