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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Single-threaded IO dispatcher that drains a queue of records and writes them
 * to a RandomAccessWriter at precomputed offsets.
 * <p>
 * This ensures thread-safe access to the underlying writer while allowing
 * parallel record building in worker threads.
 */
class WriteDispatcher implements AutoCloseable {
    private static final NodeRecordTask.Result POISON_PILL = 
        new NodeRecordTask.Result(-1, -1, ByteBuffer.allocate(0));
    
    private final RandomAccessWriter writer;
    private final BlockingQueue<NodeRecordTask.Result> queue;
    private final Thread dispatcherThread;
    private final AtomicReference<Throwable> error = new AtomicReference<>();
    private volatile boolean closed = false;

    /**
     * Creates and starts a write dispatcher.
     * 
     * @param writer the underlying writer to write to
     * @param queueCapacity maximum number of records to buffer (for backpressure)
     */
    public WriteDispatcher(RandomAccessWriter writer, int queueCapacity) {
        this.writer = writer;
        this.queue = new LinkedBlockingQueue<>(queueCapacity);
        this.dispatcherThread = new Thread(this::run, "WriteDispatcher");
        this.dispatcherThread.setDaemon(false);
        this.dispatcherThread.start();
    }

    /**
     * Submits a record to be written. Blocks if the queue is full (backpressure).
     * 
     * @param result the record to write
     * @throws IOException if an IO error occurred in the dispatcher thread
     * @throws InterruptedException if interrupted while waiting for queue space
     */
    public void submit(NodeRecordTask.Result result) throws IOException, InterruptedException {
        checkError();
        queue.put(result);
    }

    /**
     * Main dispatcher loop - runs in dedicated thread.
     */
    private void run() {
        try {
            while (true) {
                NodeRecordTask.Result result = queue.take();
                
                if (result == POISON_PILL) {
                    break;
                }
                
                // Seek to the correct offset and write the record
                writer.seek(result.fileOffset);
                
                // Write the buffer contents
                byte[] bytes = new byte[result.data.remaining()];
                result.data.get(bytes);
                writer.write(bytes);
            }
        } catch (Throwable t) {
            error.set(t);
        }
    }

    /**
     * Checks if an error occurred in the dispatcher thread and throws it.
     */
    private void checkError() throws IOException {
        Throwable t = error.get();
        if (t != null) {
            if (t instanceof IOException) {
                throw (IOException) t;
            } else if (t instanceof RuntimeException) {
                throw (RuntimeException) t;
            } else {
                throw new RuntimeException("Error in write dispatcher", t);
            }
        }
    }

    /**
     * Signals the dispatcher to finish processing queued records and stop.
     * Blocks until all records are written.
     * 
     * @throws IOException if an IO error occurred
     */
    @Override
    public void close() throws IOException {
        if (closed) {
            return;
        }
        closed = true;
        
        try {
            // Send poison pill to stop the dispatcher
            queue.put(POISON_PILL);
            
            // Wait for dispatcher to finish
            dispatcherThread.join();
            
            // Check for errors
            checkError();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while closing dispatcher", e);
        }
    }

    /**
     * Returns the number of records currently queued for writing.
     */
    public int queueSize() {
        return queue.size();
    }
}
