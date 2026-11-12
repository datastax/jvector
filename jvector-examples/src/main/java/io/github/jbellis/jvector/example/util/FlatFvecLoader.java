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

package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.FlatVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.EOFException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * Loads an fvecs file into an appendable {@link FlatVectorValues} the way a database ingests vectors: each row is
 * decoded into one reused vector and handed to {@link FlatVectorValues#add}, which copies it into the store. Rows are
 * read in parallel by contiguous ranges with positional reads, so no per-vector objects are created and the file is
 * read once.
 */
public final class FlatFvecLoader {
    /** Bytes each thread reads per positional read. */
    private static final int READ_BYTES = 1 << 20;

    private FlatFvecLoader() {}

    /** Loads every row of {@code file}, using the common pool. */
    public static FlatVectorValues load(Path file) {
        return load(file, ForkJoinPool.commonPool());
    }

    /** Loads every row of {@code file} into a new appendable store, reading in parallel on {@code pool}. */
    public static FlatVectorValues load(Path file, ForkJoinPool pool) {
        try (var channel = FileChannel.open(file, StandardOpenOption.READ)) {
            long size = channel.size();
            if (size < Integer.BYTES) throw new IOException("Empty fvecs file: " + file);
            var header = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            readFully(channel, header, 0);
            int dimension = header.getInt(0);
            if (dimension <= 0 || dimension > 100_000) {
                throw new IOException("Unreasonable dimension " + dimension + " in fvecs file " + file);
            }
            long rowBytes = Integer.BYTES + (long) dimension * Float.BYTES;
            if (size % rowBytes != 0) {
                throw new IOException("fvecs file " + file + " is not a whole number of " + dimension + "-dimension rows");
            }
            if (size / rowBytes > Integer.MAX_VALUE) throw new IOException("Too many rows in " + file);
            int rows = (int) (size / rowBytes);
            var flat = FlatVectorValues.appendable(dimension);
            int parts = Math.max(1, Math.min(rows, pool.getParallelism()));
            pool.submit(() -> IntStream.range(0, parts).parallel().forEach(part -> {
                int start = (int) ((long) part * rows / parts);
                int end = (int) ((long) (part + 1) * rows / parts);
                loadRows(channel, flat, dimension, rowBytes, start, end);
            })).join();
            if (flat.size() != rows) throw new IllegalStateException("Loaded " + flat.size() + " of " + rows + " rows");
            return flat;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** Adds rows {@code [start, end)}, decoding each into one reused vector. */
    private static void loadRows(FileChannel channel, FlatVectorValues flat, int dimension, long rowBytes,
                                 int start, int end) {
        int rowsPerRead = (int) Math.max(1, READ_BYTES / rowBytes);
        var buffer = ByteBuffer.allocate((int) (rowsPerRead * rowBytes)).order(ByteOrder.LITTLE_ENDIAN);
        float[] values = new float[dimension];
        VectorFloat<?> vector = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(values);
        try {
            for (int first = start; first < end; first += rowsPerRead) {
                int count = Math.min(rowsPerRead, end - first);
                buffer.clear().limit((int) (count * rowBytes));
                readFully(channel, buffer, first * rowBytes);
                buffer.flip();
                for (int row = first; row < first + count; row++) {
                    int rowDimension = buffer.getInt();
                    if (rowDimension != dimension) {
                        throw new IOException("Row " + row + " has dimension " + rowDimension + "; expected " + dimension);
                    }
                    buffer.asFloatBuffer().get(values); // the float view starts at the buffer's position
                    buffer.position(buffer.position() + dimension * Float.BYTES);
                    flat.add(row, vector);
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** Fills {@code buffer} from {@code position}; a channel that stops making progress fails instead of spinning. */
    private static void readFully(FileChannel channel, ByteBuffer buffer, long position) throws IOException {
        int zeroReads = 0;
        while (buffer.hasRemaining()) {
            int read = channel.read(buffer, position + buffer.position());
            if (read < 0) throw new EOFException("Truncated fvecs file at byte " + (position + buffer.position()));
            if (read == 0 && ++zeroReads >= 16) {
                throw new IOException("No progress reading fvecs data at byte " + (position + buffer.position()));
            }
            if (read > 0) zeroReads = 0;
        }
    }
}
