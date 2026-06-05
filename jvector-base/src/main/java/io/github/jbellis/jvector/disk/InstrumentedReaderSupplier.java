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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.atomic.LongAdder;

/**
 * A {@link ReaderSupplier} decorator that counts I/O operations and bytes read across
 * all readers it vends.  Intended for benchmarking and profiling use; it adds
 * lightweight atomic accounting to every call but is not designed for production
 * throughput measurement where the overhead would distort the results being measured.
 *
 * <p><strong>Usage:</strong>
 * <pre>{@code
 * var instrumented = new InstrumentedReaderSupplier(ReaderSupplierFactory.open(path));
 * // ... run benchmark against instrumented ...
 * IOMetrics metrics = instrumented.getMetrics();
 * System.out.println(metrics);
 * }</pre>
 *
 * <p>All per-thread {@link RandomAccessReader} instances vended by this supplier share
 * the same {@link InstrumentedReaderSupplier.IOMetrics} counters, so the metrics represent
 * the aggregate across all threads that used this supplier.
 *
 * <p>Call {@link InstrumentedReaderSupplier.IOMetrics#reset()} between test phases to
 * isolate build-time I/O from search-time I/O.
 */
public class InstrumentedReaderSupplier implements ReaderSupplier {

    private final ReaderSupplier delegate;
    private final IOMetrics metrics = new IOMetrics();

    public InstrumentedReaderSupplier(ReaderSupplier delegate) {
        this.delegate = delegate;
    }

    @Override
    public RandomAccessReader get() throws IOException {
        return new InstrumentedReader(delegate.get(), metrics);
    }

    @Override
    public void close() throws IOException {
        delegate.close();
    }

    /** Returns the shared metrics object accumulated across all readers from this supplier. */
    public IOMetrics getMetrics() {
        return metrics;
    }

    // -----------------------------------------------------------------------

    /**
     * Accumulated I/O counters for a single {@link InstrumentedReaderSupplier}.
     *
     * <p>Counters use {@code LongAdder} for low-contention concurrent updates from
     * multiple reader threads.
     *
     * <p>The seek:read ratio ({@code seekCount / readCount}) is the most diagnostic
     * single number for storage analysis: a ratio near 1.0 means every read is preceded
     * by a seek (maximally random I/O); a lower ratio means reads are being issued
     * sequentially after fewer seeks (more sequential, more OS-friendly).
     */
    public static class IOMetrics {

        private final LongAdder readCount  = new LongAdder();
        private final LongAdder bytesRead  = new LongAdder();
        private final LongAdder seekCount  = new LongAdder();

        void recordRead(long bytes) {
            readCount.increment();
            bytesRead.add(bytes);
        }

        void recordSeek() {
            seekCount.increment();
        }

        public long getReadCount()  { return readCount.sum(); }
        public long getBytesRead()  { return bytesRead.sum(); }
        public long getSeekCount()  { return seekCount.sum(); }

        /** Average bytes per read call, or 0 if no reads have been recorded. */
        public double getAvgReadBytes() {
            long rc = readCount.sum();
            return rc == 0 ? 0.0 : (double) bytesRead.sum() / rc;
        }

        /**
         * Seek-to-read ratio: the number of seeks per read call.
         * 1.0 = fully random (one seek per read); &lt;1.0 = sequential runs present.
         */
        public double getSeekReadRatio() {
            long rc = readCount.sum();
            return rc == 0 ? 0.0 : (double) seekCount.sum() / rc;
        }

        /** Resets all counters to zero. Call between benchmark phases to isolate measurements. */
        public void reset() {
            readCount.reset();
            bytesRead.reset();
            seekCount.reset();
        }

        /**
         * Returns a point-in-time snapshot of the current counters as a new {@link IOMetrics}
         * instance whose values will not change further.  Useful for sampling over time.
         */
        public IOMetrics snapshot() {
            IOMetrics snap = new IOMetrics();
            snap.readCount.add(this.readCount.sum());
            snap.bytesRead.add(this.bytesRead.sum());
            snap.seekCount.add(this.seekCount.sum());
            return snap;
        }

        @Override
        public String toString() {
            long rc = readCount.sum();
            long br = bytesRead.sum();
            long sc = seekCount.sum();
            return String.format(
                "IOMetrics[reads=%,d  bytes=%,d (avg %.0f bytes/read)  seeks=%,d  seek:read=%.2f]",
                rc, br, rc == 0 ? 0.0 : (double) br / rc, sc, rc == 0 ? 0.0 : (double) sc / rc);
        }
    }

    // -----------------------------------------------------------------------

    private static final class InstrumentedReader implements RandomAccessReader {

        private final RandomAccessReader delegate;
        private final IOMetrics metrics;

        InstrumentedReader(RandomAccessReader delegate, IOMetrics metrics) {
            this.delegate = delegate;
            this.metrics  = metrics;
        }

        @Override
        public void seek(long offset) throws IOException {
            metrics.recordSeek();
            delegate.seek(offset);
        }

        @Override
        public long getPosition() throws IOException {
            return delegate.getPosition();
        }

        @Override
        public int readInt() throws IOException {
            metrics.recordRead(Integer.BYTES);
            return delegate.readInt();
        }

        @Override
        public float readFloat() throws IOException {
            metrics.recordRead(Float.BYTES);
            return delegate.readFloat();
        }

        @Override
        public long readLong() throws IOException {
            metrics.recordRead(Long.BYTES);
            return delegate.readLong();
        }

        @Override
        public void readFully(byte[] bytes) throws IOException {
            metrics.recordRead(bytes.length);
            delegate.readFully(bytes);
        }

        @Override
        public void readFully(ByteBuffer buffer) throws IOException {
            metrics.recordRead(buffer.remaining());
            delegate.readFully(buffer);
        }

        @Override
        public void readFully(float[] floats) throws IOException {
            metrics.recordRead((long) floats.length * Float.BYTES);
            delegate.readFully(floats);
        }

        @Override
        public void readFully(long[] vector) throws IOException {
            metrics.recordRead((long) vector.length * Long.BYTES);
            delegate.readFully(vector);
        }

        @Override
        public void read(int[] ints, int offset, int count) throws IOException {
            metrics.recordRead((long) count * Integer.BYTES);
            delegate.read(ints, offset, count);
        }

        @Override
        public void read(float[] floats, int offset, int count) throws IOException {
            metrics.recordRead((long) count * Float.BYTES);
            delegate.read(floats, offset, count);
        }

        @Override
        public long length() throws IOException {
            return delegate.length();
        }

        @Override
        public void close() throws IOException {
            delegate.close();
        }
    }
}
