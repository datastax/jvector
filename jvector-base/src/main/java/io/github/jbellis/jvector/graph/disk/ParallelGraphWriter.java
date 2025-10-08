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

import io.github.jbellis.jvector.disk.BufferedRandomAccessWriter;
import io.github.jbellis.jvector.disk.RandomAccessWriter;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.AsynchronousFileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;
import java.util.function.IntFunction;

/**
 * Orchestrates parallel writing of L0 node records to disk.
 * <p>
 * This class manages:
 * - A thread pool for building node records in parallel
 * - Per-thread ImmutableGraphIndex.View instances for thread-safe neighbor iteration
 * - A buffer pool to avoid excessive allocation
 * - A write dispatcher for serialized IO
 * <p>
 * Usage:
 * <pre>
 * try (var parallelWriter = new ParallelGraphWriter(writer, graph, config)) {
 *     parallelWriter.writeL0Records(ordinalMapper, inlineFeatures, featureStateSuppliers, baseOffset);
 * }
 * </pre>
 */
class ParallelGraphWriter implements AutoCloseable {
    private final RandomAccessWriter writer;
    private final ImmutableGraphIndex graph;
    private final ExecutorService executor;
    private final GraphIndexWriteDispatcher dispatcher;
    private final ThreadLocal<ImmutableGraphIndex.View> viewPerThread;
    private final ThreadLocal<ByteBuffer> bufferPerThread;
    private final List<ImmutableGraphIndex.View> allViews = new ArrayList<>();
    private final int recordSize;
    private final Path filePath; // Optional: for async writes

    /**
     * Configuration for parallel writing.
     */
    static class Config {
        final int workerThreads;
        final int dispatcherQueueCapacity;
        final boolean useDirectBuffers;

        /**
         * @param workerThreads number of worker threads for building records (0 = use available processors)
         * @param dispatcherQueueCapacity max records to buffer before blocking (for backpressure)
         * @param useDirectBuffers whether to use direct ByteBuffers (can be faster for large records)
         */
        public Config(int workerThreads, int dispatcherQueueCapacity, boolean useDirectBuffers) {
            this.workerThreads = workerThreads <= 0 ? Runtime.getRuntime().availableProcessors() : workerThreads;
            this.dispatcherQueueCapacity = dispatcherQueueCapacity;
            this.useDirectBuffers = useDirectBuffers;
        }

        /**
         * Returns a default configuration suitable for most use cases.
         * Uses available CPU cores, a queue capacity of 1024, and heap buffers.
         *
         * @return default configuration
         */
        public static Config defaultConfig() {
            return new Config(0, 1024, false);
        }
    }

    /**
     * Creates a parallel writer.
     *
     * @param writer the underlying writer
     * @param graph the graph being written
     * @param inlineFeatures the inline features to write
     * @param config parallelization configuration
     * @param filePath optional file path for async writes (can be null)
     */
    public ParallelGraphWriter(RandomAccessWriter writer,
                               ImmutableGraphIndex graph,
                               List<Feature> inlineFeatures,
                               Config config,
                               Path filePath) {
        this.writer = writer;
        this.graph = graph;
        this.filePath = filePath;
        this.executor = Executors.newFixedThreadPool(config.workerThreads,
            r -> {
                Thread t = new Thread(r);
                t.setName("ParallelGraphWriter-Worker");
                t.setDaemon(false);
                return t;
            });
        this.dispatcher = new GraphIndexWriteDispatcher(writer, config.dispatcherQueueCapacity);

        // Compute fixed record size for L0
        this.recordSize = Integer.BYTES // node ordinal
            + inlineFeatures.stream().mapToInt(Feature::featureSize).sum()
            + Integer.BYTES // neighbor count
            + graph.getDegree(0) * Integer.BYTES; // neighbors + padding

        // Thread-local views for safe neighbor iteration
        this.viewPerThread = ThreadLocal.withInitial(() -> {
            var view = graph.getView();
            synchronized (allViews) {
                allViews.add(view);
            }
            return view;
        });

        // Thread-local buffers to avoid allocation overhead
        final int bufferSize = recordSize;
        final boolean useDirect = config.useDirectBuffers;
        this.bufferPerThread = ThreadLocal.withInitial(() ->
            useDirect ? ByteBuffer.allocateDirect(bufferSize) : ByteBuffer.allocate(bufferSize)
        );
    }

    /**
     * Writes all L0 node records in parallel.
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
        // Build all records in parallel
        List<Future<NodeRecordTask.Result>> futures = buildRecords(
                ordinalMapper, inlineFeatures, featureStateSuppliers, baseOffset);

        // Write results using appropriate strategy
        if (filePath != null && writer instanceof BufferedRandomAccessWriter) {
            writeRecordsAsync(futures);
        } else {
            writeRecordsSync(futures);
        }
    }

    /**
     * Builds all node records in parallel using the worker thread pool.
     * Each worker thread obtains its own graph view and buffer from thread-local storage.
     *
     * @param ordinalMapper maps between old and new ordinals
     * @param inlineFeatures the inline features to write
     * @param featureStateSuppliers suppliers for feature state
     * @param baseOffset the file offset where L0 records start
     * @return list of futures for all record building tasks
     */
    private List<Future<NodeRecordTask.Result>> buildRecords(
            OrdinalMapper ordinalMapper,
            List<Feature> inlineFeatures,
            Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers,
            long baseOffset) {
        int maxOrdinal = ordinalMapper.maxOrdinal();
        List<Future<NodeRecordTask.Result>> futures = new ArrayList<>(maxOrdinal + 1);

        for (int newOrdinal = 0; newOrdinal <= maxOrdinal; newOrdinal++) {
            final int ordinal = newOrdinal;
            final long fileOffset = baseOffset + (long) ordinal * recordSize;

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
        }

        return futures;
    }

    /**
     * Writes records synchronously via the write dispatcher.
     * The dispatcher serializes writes through the RandomAccessWriter to ensure thread safety.
     *
     * @param futures the completed record building tasks
     * @throws IOException if an I/O error occurs
     */
    private void writeRecordsSync(List<Future<NodeRecordTask.Result>> futures) throws IOException {
        try {
            for (Future<NodeRecordTask.Result> future : futures) {
                NodeRecordTask.Result result = future.get();
                dispatcher.submit(result);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while writing records", e);
        } catch (ExecutionException e) {
            throw unwrapExecutionException(e);
        }
    }

    /**
     * Writes records asynchronously using AsynchronousFileChannel for improved throughput.
     * Creates a dedicated thread pool for async I/O operations and properly cleans up resources.
     * Requires filePath to be set.
     *
     * @param futures the completed record building tasks
     * @throws IOException if an I/O error occurs
     */
    private void writeRecordsAsync(List<Future<NodeRecordTask.Result>> futures) throws IOException {
        var opts = EnumSet.of(StandardOpenOption.WRITE, StandardOpenOption.READ);
        int numThreads = Math.min(Runtime.getRuntime().availableProcessors(), 32);
        ExecutorService fileWritePool = new ThreadPoolExecutor(
                numThreads, numThreads,
                0L, TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<>(),
                r -> {
                    var t = new Thread(r, "graphnode-writer");
                    t.setDaemon(true);
                    return t;
                });

        try (var afc = AsynchronousFileChannel.open(filePath, opts, fileWritePool)) {
            try {
                for (Future<NodeRecordTask.Result> future : futures) {
                    NodeRecordTask.Result result = future.get();

                    // Copy buffer to avoid issues with thread-local buffer reuse
                    // The result.data buffer is thread-local and may be reused by another task
                    // before the async write completes, causing corruption
                    // Note: result.data has position at end of written data, need to copy from 0
                    ByteBuffer source = result.data.duplicate();
                    source.flip(); // Sets limit to position, position to 0
                    ByteBuffer bufferCopy = ByteBuffer.allocate(source.remaining());
                    bufferCopy.put(source);
                    bufferCopy.flip();

                    afc.write(bufferCopy, result.fileOffset).get();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted while writing records", e);
            } catch (ExecutionException e) {
                throw unwrapExecutionException(e);
            }
        } finally {
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

            // Close dispatcher (waits for all writes to complete)
            dispatcher.close();

            // Close all views
            synchronized (allViews) {
                for (var view : allViews) {
                    view.close();
                }
                allViews.clear();
            }
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Error closing parallel writer", e);
        }
    }
}
