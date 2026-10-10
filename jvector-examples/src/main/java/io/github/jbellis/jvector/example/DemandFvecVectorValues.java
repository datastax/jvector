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

import io.github.jbellis.jvector.disk.ByteBufferReader;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
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
import java.util.Objects;
import java.util.concurrent.atomic.LongAdder;

/** Benchmark control: synchronous point reads, with no executor, cache or read-ahead. */
final class DemandFvecVectorValues implements RandomAccessVectorValues, AutoCloseable {
    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final FileChannel channel;
    private final int dimension, size, recordBytes;
    private final boolean owner;
    private final Buffer buffer;
    private final VectorFloat<?> scratch;
    private final LongAdder reads, bytes;

    DemandFvecVectorValues(Path file) throws IOException {
        channel = FileChannel.open(file, StandardOpenOption.READ);
        try {
            var header = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            readFully(channel, header, 0);
            dimension = header.flip().getInt();
            if (dimension <= 0 || dimension > 100_000) throw new IOException("Invalid fvecs dimension: " + dimension);
            recordBytes = Math.multiplyExact(dimension + 1, 4);
            long length = channel.size();
            if (length % recordBytes != 0 || length / recordBytes > Integer.MAX_VALUE)
                throw new IOException("Invalid fvecs file length: " + length);
            size = (int) (length / recordBytes);
            owner = true;
            buffer = new Buffer(recordBytes);
            scratch = VTS.createFloatVector(dimension);
            reads = new LongAdder();
            bytes = new LongAdder();
        } catch (Throwable e) {
            channel.close();
            throw e;
        }
    }

    private DemandFvecVectorValues(DemandFvecVectorValues root) {
        channel = root.channel; dimension = root.dimension; size = root.size;
        recordBytes = root.recordBytes; owner = false;
        buffer = new Buffer(recordBytes); scratch = VTS.createFloatVector(dimension);
        reads = root.reads; bytes = root.bytes;
    }

    @Override public int size() { return size; }
    @Override public int dimension() { return dimension; }
    @Override public boolean isValueShared() { return true; }
    @Override public DemandFvecVectorValues copy() { return new DemandFvecVectorValues(this); }
    @Override public java.util.function.Supplier<RandomAccessVectorValues> threadLocalSupplier() {
        // Views share the owner's channel; closing the owner releases it after all workers finish.
        var local = ThreadLocal.withInitial(this::copy);
        return local::get;
    }
    @Override public VectorFloat<?> getVector(int ordinal) {
        Objects.checkIndex(ordinal, size);
        try {
            buffer.payload.clear();
            readFully(channel, buffer.payload, (long) ordinal * recordBytes);
            buffer.payload.flip();
            if (buffer.readInt() != dimension) throw new IOException("Dimension mismatch at ordinal " + ordinal);
            VTS.readFloatVector(buffer, dimension, scratch, 0);
            reads.increment(); bytes.add(recordBytes);
            return scratch;
        } catch (IOException e) { throw new UncheckedIOException(e); }
    }
    long reads() { return reads.sum(); }
    long bytes() { return bytes.sum(); }
    @Override public void close() throws IOException { if (owner) channel.close(); }

    private static void readFully(FileChannel channel, ByteBuffer buffer, long offset) throws IOException {
        while (buffer.hasRemaining()) {
            int n = channel.read(buffer, offset);
            if (n < 0) throw new EOFException("Incomplete fvecs record");
            offset += n;
        }
    }

    // Match the PR source's float decoding so the control differs in input scheduling.
    private static final class Buffer extends ByteBufferReader {
        final ByteBuffer payload;
        final FloatBuffer floats;
        Buffer(int bytes) {
            super(ByteBuffer.allocateDirect(bytes).order(ByteOrder.LITTLE_ENDIAN));
            payload = bb; floats = bb.asFloatBuffer();
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
}
