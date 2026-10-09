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

package io.github.jbellis.jvector.disk;

import io.github.jbellis.jvector.graph.BatchedVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.VectorCursor;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.EOFException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.ArrayList;
import java.util.ArrayDeque;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Supplier;

/**
 * A read-only, fixed-dimension fvecs source with bounded asynchronous range/selection
 * reads. The file must remain unchanged while open. Copies share the channel, executor
 * and reusable buffer budget; each copy and cursor has its own borrowed vector.
 *
 * <p>Close all cursors and stop consumers before closing the source. Closing the root
 * releases all resources, including resources used by copies. No memory mapping or
 * OS cache eviction is performed. Retained training vectors are caller-owned memory.
 */
public final class FvecFileVectorValues implements BatchedVectorValues, AutoCloseable {
    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final long DEFAULT_MAX_BYTES = 64L << 20;
    private final Storage storage;
    private final boolean ownsStorage;
    private final VectorFloat<?> scratch;
    private int previousOrdinal = -2;
    private Read currentRead;
    private boolean closed;

    /**
     * Open with at most min(1% of file bytes, 64 MiB) of reusable I/O buffers,
     * forty-eight I/O workers and three batches of read-ahead. One record is the minimum budget.
     */
    public static FvecFileVectorValues open(Path path) throws IOException {
        return open(path, Options.defaults());
    }

    /** Open a file with named limits. The returned root owns the shared I/O resources. */
    public static FvecFileVectorValues open(Path path, Options options) throws IOException {
        Objects.requireNonNull(options, "options");
        return new FvecFileVectorValues(new Storage(path, options.maxBufferBytes, options.ioThreads,
                options.batchVectors, options.readAhead), true);
    }

    /**
     * Immutable source options. Start with {@link #defaults()} and change only the
     * limits needed. Buffer capacity is shared across all copies and cursors;
     * retained vectors and the OS page cache are outside this payload budget.
     */
    public static final class Options {
        private static final Options DEFAULTS = new Options(DEFAULT_MAX_BYTES, 48, 64, 3);
        private final long maxBufferBytes;
        private final int ioThreads, batchVectors, readAhead;

        private Options(long maxBufferBytes, int ioThreads, int batchVectors, int readAhead) {
            if (maxBufferBytes <= 0 || ioThreads <= 0 || batchVectors <= 0 || readAhead < 0)
                throw new IllegalArgumentException("Invalid buffered source limits");
            this.maxBufferBytes = maxBufferBytes;
            this.ioThreads = ioThreads;
            this.batchVectors = batchVectors;
            this.readAhead = readAhead;
        }

        /** Defaults: 64 MiB maximum, 48 I/O workers, 64 vectors/batch, 3 batches ahead. */
        public static Options defaults() { return DEFAULTS; }

        /** Maximum payload bytes; also capped at 1% of file bytes, with one record minimum. */
        public long maxBufferBytes() { return maxBufferBytes; }
        /** Number of asynchronous I/O workers. */
        public int ioThreads() { return ioThreads; }
        /** Maximum vectors per batch; reduced when necessary to fit the shared budget. */
        public int batchVectors() { return batchVectors; }
        /** Number of speculative batches ahead of demand; zero disables read-ahead. */
        public int readAhead() { return readAhead; }

        /** Return options with a different positive maximum payload budget. */
        public Options withMaxBufferBytes(long bytes) {
            return new Options(bytes, ioThreads, batchVectors, readAhead);
        }
        /** Return options with a different positive I/O worker count. */
        public Options withIoThreads(int threads) {
            return new Options(maxBufferBytes, threads, batchVectors, readAhead);
        }
        /** Return options with a different positive maximum batch size. */
        public Options withBatchVectors(int vectors) {
            return new Options(maxBufferBytes, ioThreads, vectors, readAhead);
        }
        /** Return options with a different nonnegative number of batches ahead. */
        public Options withReadAhead(int batches) {
            return new Options(maxBufferBytes, ioThreads, batchVectors, batches);
        }
    }

    private FvecFileVectorValues(Storage storage, boolean ownsStorage) {
        this.storage = storage;
        this.ownsStorage = ownsStorage;
        scratch = VTS.createFloatVector(storage.dimension);
    }

    @Override public int size() { return storage.size; }
    @Override public int dimension() { return storage.dimension; }
    @Override public boolean isValueShared() { return true; }

    private void ensureOpen() {
        if (closed || storage.closed) throw new IllegalStateException("Vector source is closed");
    }

    /**
     * Supply one point-access view per thread. Views share the root's I/O resources
     * and must stop being used before the root closes.
     */
    @Override public Supplier<RandomAccessVectorValues> threadLocalSupplier() {
        var local = ThreadLocal.withInitial(this::copy);
        return local::get;
    }

    /**
     * Create a view with its own scratch vector. Closing this view leaves other views
     * usable; closing the root invalidates every view. Point access is single-consumer.
     */
    @Override public FvecFileVectorValues copy() {
        ensureOpen();
        return new FvecFileVectorValues(storage, false);
    }

    @Override public VectorFloat<?> getVector(int ordinal) {
        getVectorInto(ordinal, scratch, 0);
        return scratch;
    }

    @Override public void getVectorInto(int ordinal, VectorFloat<?> destination, int offset) {
        ensureOpen();
        Objects.checkIndex(ordinal, size());
        Objects.checkFromIndexSize(offset, dimension(), destination.length());
        boolean sequential = ordinal == previousOrdinal + 1;
        boolean withinCurrent = currentRead != null && ordinal >= currentRead.request.start
                && ordinal < currentRead.request.start + currentRead.request.count;
        if (!withinCurrent) {
            if (currentRead != null) currentRead.consumed = true;
            int first = sequential ? ordinal / storage.batchVectors * storage.batchVectors : ordinal;
            int count = sequential ? Math.min(storage.batchVectors, size() - first) : 1;
            currentRead = storage.decode(new Request(first, count, null), ordinal - first, destination, offset);
            storage.prefetchRange(first + count, size(), sequential ? storage.readAhead : 0);
        } else if (!storage.tryDecode(currentRead, ordinal - currentRead.request.start, destination, offset)) {
            currentRead = storage.decode(currentRead.request, ordinal - currentRead.request.start, destination, offset);
        }
        previousOrdinal = ordinal;
    }

    @Override public VectorCursor openRange(int startInclusive, int endExclusive) {
        ensureOpen();
        Objects.checkFromToIndex(startInclusive, endExclusive, size());
        return new Cursor(startInclusive, endExclusive - startInclusive, null);
    }

    @Override public VectorCursor openSelection(int[] ordinals, int offset, int count) {
        ensureOpen();
        Objects.checkFromIndexSize(offset, count, ordinals.length);
        for (int i = offset; i < offset + count; i++) Objects.checkIndex(ordinals[i], size());
        return new Cursor(offset, count, ordinals);
    }

    /** Cumulative source I/O counters; bytes are logical input reads, not device traffic. */
    public Statistics statistics() {
        synchronized (storage) {
            return new Statistics(storage.reads.sum(), storage.bytes.sum(), storage.allocatedBytes, storage.budget);
        }
    }

    public static final class Statistics {
        public final long reads, bytesRead, bufferBytes, bufferLimitBytes;
        private Statistics(long reads, long bytes, long allocated, long limit) {
            this.reads = reads; bytesRead = bytes; bufferBytes = allocated; bufferLimitBytes = limit;
        }
    }

    /**
     * Close this view. For the root, first close all cursors and stop all consumers;
     * root closure shuts down shared I/O and invalidates every copy.
     */
    @Override public void close() throws IOException {
        if (closed) return;
        closed = true;
        currentRead = null;
        if (ownsStorage) storage.close();
    }

    private final class Cursor implements VectorCursor {
        private final int start, count;
        private final int[] ordinals;
        private final VectorFloat<?> value = VTS.createFloatVector(dimension());
        private int position = -1;
        private boolean cursorClosed;
        private Read current;

        Cursor(int start, int count, int[] ordinals) {
            this.start = start; this.count = count; this.ordinals = ordinals;
            prefetch(0);
        }

        private Request request(int position) {
            int first = position / storage.batchVectors * storage.batchVectors;
            return new Request(start + first, Math.min(storage.batchVectors, count - first), ordinals);
        }

        private void prefetch(int position) {
            for (int i = 0; i <= storage.readAhead; i++) {
                long next = (long) position + (long) i * storage.batchVectors;
                if (next >= count) break;
                storage.schedule(request((int) next));
            }
        }

        @Override public boolean next() {
            ensureOpen();
            if (cursorClosed) throw new IllegalStateException("Cursor is closed");
            if (position == count) return false;
            if (++position == count) { release(); return false; }
            int row = position % storage.batchVectors;
            if (row == 0) {
                if (current != null) current.consumed = true;
                prefetch(position);
                current = storage.decode(request(position), row, value, 0);
            } else if (!storage.tryDecode(current, row, value, 0)) {
                current = storage.decode(current.request, row, value, 0);
            }
            return true;
        }

        private void checkPosition() {
            ensureOpen();
            if (cursorClosed || position < 0 || position >= count) throw new IllegalStateException("No current vector");
        }
        @Override public int ordinal() { checkPosition(); return ordinals == null ? start + position : ordinals[start + position]; }
        @Override public VectorFloat<?> vector() { checkPosition(); return value; }
        @Override public boolean isValueShared() { return true; }
        private void release() { current = null; storage.discard(ordinals, start, count); }
        @Override public void close() { if (!cursorClosed) { cursorClosed = true; release(); } }
    }

    private static final class Request {
        final int start, count;
        final int[] ordinals;
        Request(int start, int count, int[] ordinals) { this.start = start; this.count = count; this.ordinals = ordinals; }
        int ordinal(int i) { return ordinals == null ? start + i : ordinals[start + i]; }
        @Override public int hashCode() { return 31 * (31 * System.identityHashCode(ordinals) + start) + count; }
        @Override public boolean equals(Object other) {
            if (!(other instanceof Request)) return false;
            Request r = (Request) other;
            return r.start == start && r.count == count && r.ordinals == ordinals;
        }
    }

    private static final class Buffer extends ByteBufferReader {
        final FloatBuffer floats;
        Buffer(int bytes) {
            // Read directly into the pooled payload; heap buffers require a temporary native copy.
            super(ByteBuffer.allocateDirect(bytes).order(ByteOrder.LITTLE_ENDIAN));
            floats = bb.asFloatBuffer();
        }
        @Override public void read(float[] destination, int offset, int count) {
            Objects.checkFromIndexSize(offset, count, destination.length);
            Objects.checkFromIndexSize(bb.position(), Math.multiplyExact(count, 4), bb.limit());
            floats.position(bb.position() / 4);
            floats.get(destination, offset, count);
            bb.position(bb.position() + count * 4);
        }
        @Override public void readFully(float[] destination) { read(destination, 0, destination.length); }
    }

    private static final class Read {
        final Request request;
        final Buffer buffer;
        final FutureTask<Buffer> task;
        int users;
        volatile boolean consumed, invalidated;
        Read(Request request, Buffer buffer, Storage storage) {
            this.request = request; this.buffer = buffer;
            task = new FutureTask<>(() -> { storage.read(request, buffer); return buffer; }) {
                @Override protected void done() { synchronized (storage) { storage.notifyAll(); } }
            };
        }
    }

    private static final class Storage {
        final FileChannel channel;
        final int dimension, size, rowBytes, batchVectors, readAhead, slots;
        final long budget;
        final ThreadPoolExecutor executor;
        final LinkedHashMap<Request, Read> cache = new LinkedHashMap<>(16, .75f, true);
        final LongAdder reads = new LongAdder(), bytes = new LongAdder();
        final ArrayDeque<Buffer> free = new ArrayDeque<>();
        long allocatedBytes;
        int allocated;
        volatile boolean closed;

        Storage(Path path, long maxBytes, int threads, int maxBatch, int ahead) throws IOException {
            channel = FileChannel.open(path, StandardOpenOption.READ);
            try {
                ByteBuffer header = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
                readFully(channel, header, 0); header.flip();
                dimension = header.getInt();
                if (dimension <= 0 || dimension > Integer.MAX_VALUE / 4 - 1)
                    throw new IOException("Invalid fvec dimension: " + dimension);
                rowBytes = (dimension + 1) * 4;
                long length = channel.size();
                if (length % rowBytes != 0 || length / rowBytes > Integer.MAX_VALUE)
                    throw new IOException("Invalid fvec file length: " + length);
                size = (int) (length / rowBytes);
                budget = Math.max(rowBytes, Math.min(maxBytes, length / 100));
                // Leave room for concurrent consumers and their three-batch look-ahead.
                long concurrentRows = budget / rowBytes / threads / (ahead + 1L);
                batchVectors = (int) Math.max(1, Math.min(Math.min(maxBatch, Integer.MAX_VALUE / rowBytes), concurrentRows));
                readAhead = ahead;
                slots = (int) Math.min(Integer.MAX_VALUE, budget / (batchVectors * (long) rowBytes));
                executor = new ThreadPoolExecutor(threads, threads, 0, TimeUnit.MILLISECONDS,
                        new ArrayBlockingQueue<>(slots), runnable -> {
                            Thread t = new Thread(runnable, "jvector-input-io"); t.setDaemon(true); return t;
                        });
            } catch (Throwable failure) { channel.close(); throw failure; }
        }

        private void read(Request request, Buffer buffer) throws IOException {
            synchronized (buffer) {
                buffer.bb.clear();
                boolean contiguous = true;
                for (int i = 1; i < request.count && contiguous; i++)
                    contiguous = request.ordinal(i) == request.ordinal(0) + i;
                if (contiguous) {
                    buffer.bb.limit(request.count * rowBytes);
                    readFully(channel, buffer.bb, (long) request.ordinal(0) * rowBytes);
                    reads.increment(); bytes.add((long) request.count * rowBytes);
                } else {
                    for (int i = 0; i < request.count; i++) {
                        buffer.bb.limit((i + 1) * rowBytes);
                        readFully(channel, buffer.bb, (long) request.ordinal(i) * rowBytes);
                        reads.increment(); bytes.add(rowBytes);
                    }
                }
                buffer.bb.flip();
            }
        }

        synchronized Read schedule(Request request) {
            if (closed) throw new IllegalStateException("Vector source is closed");
            Read present = cache.get(request);
            if (present != null) return present;
            Buffer buffer = free.pollFirst();
            if (buffer == null && allocated < slots) {
                buffer = new Buffer(batchVectors * rowBytes);
                allocated++; allocatedBytes += batchVectors * (long) rowBytes;
            } else if (buffer == null) {
                Read victim = null;
                for (Read r : cache.values()) {
                    // Never recycle an in-flight or actively decoded payload.
                    if (r.users == 0 && r.task.isDone() && (victim == null || r.consumed)) {
                        victim = r;
                        if (r.consumed) break;
                    }
                }
                if (victim != null) { victim.invalidated = true; cache.remove(victim.request); buffer = victim.buffer; }
            }
            if (buffer == null) return null;
            Read read = new Read(request, buffer, this);
            cache.put(request, read);
            try { executor.execute(read.task); }
            catch (RejectedExecutionException e) {
                // There can be at most slots reads, including queued and in-flight reads.
                cache.remove(request); allocated--; allocatedBytes -= buffer.bb.capacity();
                throw e;
            }
            return read;
        }

        Read decode(Request request, int row, VectorFloat<?> destination, int offset) {
            Read read;
            synchronized (this) {
                while ((read = schedule(request)) == null) {
                    try { wait(); }
                    catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new IllegalStateException("Input read interrupted", e); }
                }
                // Demand owns a lease before waiting, so ready data cannot be recycled
                // between I/O completion and decoding. Speculation alone never pins data.
                read.users++;
            }
            try {
                read.task.get();
                synchronized (read.buffer) {
                    read.buffer.seek((long) row * rowBytes);
                    int dim = read.buffer.readInt();
                    if (dim != dimension) throw new IOException("Unexpected fvec dimension " + dim + " at " + request.ordinal(row));
                    VTS.readFloatVector(read.buffer, dimension, destination, offset);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("Input read interrupted", e);
            } catch (ExecutionException e) {
                if (e.getCause() instanceof IOException) throw new UncheckedIOException((IOException) e.getCause());
                throw new IllegalStateException("Input read failed", e.getCause());
            } catch (IOException e) { throw new UncheckedIOException(e); }
            finally {
                synchronized (this) {
                    read.users--;
                    if (row == request.count - 1) read.consumed = true;
                    notifyAll();
                }
            }
            return read;
        }

        boolean tryDecode(Read read, int row, VectorFloat<?> destination, int offset) {
            if (read == null || read.invalidated || !read.task.isDone()) return false;
            synchronized (read.buffer) {
                // Reuse is synchronized on the payload, independently of the scheduler.
                // Within a batch, no future or shared scheduler lookup is needed.
                if (read.invalidated) return false;
                try {
                    read.buffer.seek((long) row * rowBytes);
                    int dim = read.buffer.readInt();
                    if (dim != dimension) throw new IOException("Unexpected fvec dimension " + dim + " at " + read.request.ordinal(row));
                    VTS.readFloatVector(read.buffer, dimension, destination, offset);
                    return true;
                } catch (IOException e) { throw new UncheckedIOException(e); }
            }
        }

        void prefetchRange(int start, int end, int ahead) {
            for (int i = 0; i < ahead; i++) {
                long first = (long) start + (long) i * batchVectors;
                if (first >= end) break;
                schedule(new Request((int) first, Math.min(batchVectors, end - (int) first), null));
            }
        }

        void discard(int[] ordinals, int start, int count) {
            var pending = new ArrayList<Read>();
            synchronized (this) {
                for (Read r : cache.values()) {
                    if (r.request.ordinals == ordinals && r.request.start >= start
                            && (long) r.request.start < (long) start + count) {
                        r.consumed = true;
                        pending.add(r);
                    }
                }
            }
            // Drain every read before allowing the caller to reuse its ordinal array,
            // including when an earlier read failed or this thread was interrupted.
            boolean interrupted = false;
            RuntimeException failure = null;
            for (Read r : pending) {
                boolean done = false;
                while (!done) {
                    try { r.task.get(); done = true; }
                    catch (InterruptedException e) { interrupted = true; }
                    catch (ExecutionException e) {
                        done = true;
                        RuntimeException error = e.getCause() instanceof IOException
                                ? new UncheckedIOException((IOException) e.getCause())
                                : new IllegalStateException("Input read failed", e.getCause());
                        if (failure == null) failure = error; else failure.addSuppressed(error);
                    }
                }
            }
            synchronized (this) {
                for (Read r : pending) {
                    if (cache.get(r.request) == r && r.users == 0) {
                        r.invalidated = true;
                        cache.remove(r.request);
                        free.addLast(r.buffer);
                    }
                }
                notifyAll();
            }
            if (interrupted) Thread.currentThread().interrupt();
            if (failure != null) throw failure;
        }

        void close() throws IOException {
            synchronized (this) { if (closed) return; closed = true; notifyAll(); }
            executor.shutdown();
            boolean interrupted = false;
            while (!executor.isTerminated()) {
                try { executor.awaitTermination(1, TimeUnit.SECONDS); }
                catch (InterruptedException e) { interrupted = true; }
            }
            try { channel.close(); }
            finally { synchronized (this) { cache.clear(); free.clear(); } if (interrupted) Thread.currentThread().interrupt(); }
        }
    }

    private static void readFully(FileChannel channel, ByteBuffer buffer, long offset) throws IOException {
        int start = buffer.position();
        while (buffer.hasRemaining()) {
            int n = channel.read(buffer, offset + buffer.position() - start);
            if (n < 0) throw new EOFException("Truncated fvec record at " + offset);
            if (n == 0) Thread.onSpinWait();
        }
    }
}
