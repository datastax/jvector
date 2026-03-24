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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.AsynchronousFileChannel;
import java.nio.channels.CompletionHandler;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Coordinates asynchronous write operations to a file channel with backpressure control.
 * <p>
 * This class manages a pool of in-flight write operations, ensuring that the system doesn't
 * overwhelm memory with too many pending writes. It uses a semaphore to limit the number
 * of concurrent writes and tracks all pending operations for completion.
 * <p>
 * Key features:
 * - Backpressure control via semaphore (limits in-flight writes)
 * - Automatic retry for partial writes
 * - Tracks all pending writes for batch completion
 * - Thread-safe for concurrent submissions
 * <p>
 * Usage:
 * <pre>
 * try (var coordinator = new AsyncWriteCoordinator(channel, 1000)) {
 *     CompletableFuture<Void> future = coordinator.submitWrite(buffer, offset);
 *     // ... submit more writes ...
 *     coordinator.awaitCompletion(); // Wait for all writes
 * }
 * </pre>
 */
class AsyncWriteCoordinator implements AutoCloseable {
    private final AsynchronousFileChannel channel;
    private final Semaphore writePermits;
    private final ConcurrentLinkedQueue<CompletableFuture<Void>> pendingWrites;
    private final AtomicInteger activeWrites;
    private volatile boolean closed = false;

    /**
     * Creates a new async write coordinator.
     *
     * @param channel the file channel to write to
     * @param maxInFlightWrites maximum number of concurrent write operations
     */
    public AsyncWriteCoordinator(AsynchronousFileChannel channel, int maxInFlightWrites) {
        this.channel = channel;
        this.writePermits = new Semaphore(maxInFlightWrites);
        this.pendingWrites = new ConcurrentLinkedQueue<>();
        this.activeWrites = new AtomicInteger(0);
    }

    /**
     * Submits an asynchronous write operation.
     * <p>
     * This method returns immediately after submitting the write. The actual I/O happens
     * asynchronously. If the maximum number of in-flight writes has been reached, this
     * method will block until a permit becomes available (backpressure control).
     * <p>
     * The buffer must be positioned at the start of the data to write. The buffer will
     * be fully written even if multiple I/O operations are required.
     *
     * @param buffer the buffer to write (must be ready to read)
     * @param position the file position to write at
     * @return a CompletableFuture that completes when the write finishes
     * @throws InterruptedException if interrupted while waiting for a write permit
     * @throws IllegalStateException if the coordinator has been closed
     */
    public CompletableFuture<Void> submitWrite(ByteBuffer buffer, long position) throws InterruptedException {
        if (closed) {
            throw new IllegalStateException("AsyncWriteCoordinator has been closed");
        }

        // Acquire permit (blocks if too many writes in flight)
        writePermits.acquire();
        activeWrites.incrementAndGet();

        CompletableFuture<Void> future = new CompletableFuture<>();
        pendingWrites.add(future);

        // Start the async write with completion handler
        writeWithRetry(buffer, position, future);

        return future;
    }

    /**
     * Writes a buffer fully to the channel, retrying if necessary for partial writes.
     * Uses a CompletionHandler to handle async completion without blocking.
     *
     * @param buffer the buffer to write
     * @param position the current file position
     * @param future the future to complete when done
     */
    private void writeWithRetry(ByteBuffer buffer, long position, CompletableFuture<Void> future) {
        channel.write(buffer, position, null, new CompletionHandler<Integer, Void>() {
            @Override
            public void completed(Integer bytesWritten, Void attachment) {
                if (bytesWritten < 0) {
                    failed(new IOException("Channel closed while writing"), null);
                    return;
                }

                if (buffer.hasRemaining()) {
                    // Partial write - continue with remaining bytes
                    long newPosition = position + bytesWritten;
                    writeWithRetry(buffer, newPosition, future);
                } else {
                    // Write complete
                    writePermits.release();
                    activeWrites.decrementAndGet();
                    future.complete(null);
                }
            }

            @Override
            public void failed(Throwable exc, Void attachment) {
                writePermits.release();
                activeWrites.decrementAndGet();
                future.completeExceptionally(exc);
            }
        });
    }

    /**
     * Waits for all pending writes to complete.
     * <p>
     * This method blocks until all writes submitted via {@link #submitWrite} have finished.
     * If any write failed, this method will throw an IOException with the first failure.
     *
     * @throws IOException if any write operation failed
     * @throws InterruptedException if interrupted while waiting
     */
    public void awaitCompletion() throws IOException, InterruptedException {
        // Convert queue to array to avoid concurrent modification
        CompletableFuture<Void>[] futures = pendingWrites.toArray(new CompletableFuture[0]);
        
        try {
            // Wait for all writes to complete
            CompletableFuture.allOf(futures).join();
        } catch (Exception e) {
            // Find and throw the first exception
            for (CompletableFuture<Void> future : futures) {
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
    }

    /**
     * Returns the number of currently active (in-flight) write operations.
     *
     * @return number of active writes
     */
    public int getActiveWriteCount() {
        return activeWrites.get();
    }

    /**
     * Returns the total number of pending writes (both active and queued).
     *
     * @return number of pending writes
     */
    public int getPendingWriteCount() {
        return pendingWrites.size();
    }

    @Override
    public void close() throws IOException {
        closed = true;
        try {
            awaitCompletion();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while waiting for writes to complete", e);
        }
        pendingWrites.clear();
    }
}

// Made with Bob
