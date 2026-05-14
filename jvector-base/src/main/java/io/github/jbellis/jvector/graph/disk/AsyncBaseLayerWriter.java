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

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.AsynchronousFileChannel;
import java.nio.channels.CompletionHandler;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor.WriteResult;

/**
 * Decouples base-layer disk I/O from the compute threads that build node records.
 *
 * <p>Workers borrow a pre-allocated direct {@link ByteBuffer} from this writer's pool,
 * fill it via {@link CompactWriter#writeInlineNodeRecord}, and hand the resulting
 * {@link WriteResult} back via {@link #submit}. Writes are issued on an
 * {@link AsynchronousFileChannel} so the calling thread (typically the main compaction
 * loop) returns immediately without blocking on kernel I/O.  Pool buffers are recycled
 * automatically in the completion handler.
 *
 * <p>Borrowing uses {@link ForkJoinPool#managedBlock} so ForkJoinPool worker threads
 * that need to wait for a buffer do not starve the pool when the buffer supply is
 * temporarily exhausted.
 *
 * <p>Thread safety: {@link #borrowBuffer} and {@link #submit} may be called from any
 * thread concurrently. {@link #awaitCompletion} must be called from a single thread
 * after all {@link #submit} calls have been issued.
 */
final class AsyncBaseLayerWriter implements Closeable {

    // Aim for enough buffers so workers never have to wait:
    // up to taskWindowSize workers filling a buffer simultaneously, plus the same
    // count already submitted and in-flight for async write, times a 2× safety margin.
    private static final int POOL_MULTIPLIER = 4;

    private final AsynchronousFileChannel channel;
    private final LinkedBlockingQueue<ByteBuffer> bufferPool;
    private final int recordSize;
    private final AtomicInteger pendingWrites = new AtomicInteger(0);
    private final AtomicReference<Throwable> writeError = new AtomicReference<>();

    AsyncBaseLayerWriter(Path outputPath, int recordSize, int taskWindowSize, int maxNodesPerBatch)
            throws IOException {
        this.recordSize = recordSize;
        this.channel = AsynchronousFileChannel.open(outputPath, StandardOpenOption.WRITE);

        int poolSize = Math.max(256, taskWindowSize * maxNodesPerBatch * POOL_MULTIPLIER);
        this.bufferPool = new LinkedBlockingQueue<>(poolSize);
        for (int i = 0; i < poolSize; i++) {
            ByteBuffer buf = ByteBuffer.allocateDirect(recordSize);
            buf.order(ByteOrder.BIG_ENDIAN);
            bufferPool.add(buf);
        }
    }

    /**
     * Borrows a cleared, write-ready direct buffer from the pool.
     *
     * <p>If no buffer is immediately available, the call blocks in a ForkJoinPool-aware
     * manner (using {@link ForkJoinPool#managedBlock}) so that the pool can compensate
     * with an extra thread while this one waits.
     *
     * @throws InterruptedException if the calling thread is interrupted while waiting
     * @throws IOException          if a previous async write has failed
     */
    ByteBuffer borrowBuffer() throws InterruptedException, IOException {
        checkError();
        final ByteBuffer[] holder = {null};
        ForkJoinPool.managedBlock(new ForkJoinPool.ManagedBlocker() {
            @Override
            public boolean block() throws InterruptedException {
                holder[0] = bufferPool.take();
                return true;
            }

            @Override
            public boolean isReleasable() {
                holder[0] = bufferPool.poll();
                return holder[0] != null;
            }
        });
        ByteBuffer buf = holder[0];
        buf.clear();
        buf.order(ByteOrder.BIG_ENDIAN);
        return buf;
    }

    /**
     * Submits an async positional write for the record in {@code r.data} at {@code r.fileOffset}.
     * Returns immediately; the write completes in the background and the buffer is
     * automatically returned to the pool via the completion handler.
     */
    void submit(WriteResult r) throws IOException {
        checkError();
        ByteBuffer buf = r.data;
        // buf.position()==0, buf.limit()==recordSize as set by writeInlineNodeRecord
        pendingWrites.incrementAndGet();
        new WriteRequest(buf, r.fileOffset).start();
    }

    /**
     * Blocks until all previously submitted writes have completed (or one has failed).
     *
     * @throws IOException          if any async write failed
     * @throws InterruptedException if the calling thread is interrupted while waiting
     */
    void awaitCompletion() throws IOException, InterruptedException {
        synchronized (this) {
            while (pendingWrites.get() > 0) {
                checkError();
                wait(50);
            }
        }
        checkError();
    }

    @Override
    public void close() throws IOException {
        channel.close();
    }

    private void returnBuffer(ByteBuffer buf) {
        buf.clear();
        bufferPool.offer(buf);
        int remaining = pendingWrites.decrementAndGet();
        if (remaining == 0) {
            synchronized (AsyncBaseLayerWriter.this) {
                notifyAll();
            }
        }
    }

    private void checkError() throws IOException {
        Throwable err = writeError.get();
        if (err != null) {
            throw new IOException("Async base-layer write failed", err);
        }
    }

    /**
     * Handles a single (possibly partial) async write, retrying automatically on partial
     * completion to satisfy the full {@code recordSize} contract.
     */
    private final class WriteRequest implements CompletionHandler<Integer, Void> {
        private final ByteBuffer buf;
        private long pos;

        WriteRequest(ByteBuffer buf, long startPos) {
            this.buf = buf;
            this.pos = startPos;
        }

        void start() {
            channel.write(buf, pos, null, this);
        }

        @Override
        public void completed(Integer bytesWritten, Void attachment) {
            pos += bytesWritten;
            if (buf.hasRemaining()) {
                // Rare partial write — retry for the remaining bytes
                channel.write(buf, pos, null, this);
            } else {
                returnBuffer(buf);
            }
        }

        @Override
        public void failed(Throwable exc, Void attachment) {
            writeError.compareAndSet(null, exc);
            returnBuffer(buf);
        }
    }
}
