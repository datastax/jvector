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
import java.util.ArrayList;
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
    private final WriteDispatcher dispatcher;
    private final ThreadLocal<ImmutableGraphIndex.View> viewPerThread;
    private final ThreadLocal<ByteBuffer> bufferPerThread;
    private final List<ImmutableGraphIndex.View> allViews = new ArrayList<>();
    private final int recordSize;

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
     */
    public ParallelGraphWriter(RandomAccessWriter writer,
                               ImmutableGraphIndex graph,
                               List<Feature> inlineFeatures,
                               Config config) {
        this.writer = writer;
        this.graph = graph;
        this.executor = Executors.newFixedThreadPool(config.workerThreads,
            r -> {
                Thread t = new Thread(r);
                t.setName("ParallelGraphWriter-Worker");
                t.setDaemon(false);
                return t;
            });
        this.dispatcher = new WriteDispatcher(writer, config.dispatcherQueueCapacity);

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
        int maxOrdinal = ordinalMapper.maxOrdinal();
        List<Future<NodeRecordTask.Result>> futures = new ArrayList<>(maxOrdinal + 1);

        // Submit all tasks
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

        // Collect results and submit to dispatcher
        // We process futures in order to maintain some locality, but the dispatcher
        // can handle out-of-order writes since each record has its own offset
        try {
            for (Future<NodeRecordTask.Result> future : futures) {
                NodeRecordTask.Result result = future.get();
                dispatcher.submit(result);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while writing records", e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof IOException) {
                throw (IOException) cause;
            } else if (cause instanceof RuntimeException) {
                throw (RuntimeException) cause;
            } else {
                throw new RuntimeException("Error building node record", cause);
            }
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
