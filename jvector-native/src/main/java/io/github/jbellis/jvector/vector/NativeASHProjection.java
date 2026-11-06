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

package io.github.jbellis.jvector.vector;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.Objects;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * Immutable native FP32 matrix for single-vector and bulk ASH projection.
 * The automatically managed arena lives with the model; it is not thread confined.
 * Native query/output staging is reused per calling thread, so projections may run concurrently.
 * Large projections use ordinary FFM and copy only the small query and result, allowing
 * safepoints during the dense multiply. Matrix storage is never copied per query.
 */
final class NativeASHProjection implements VectorUtilSupport.AshProjection {
    private static final String SYMBOL = "ash_project_f32";
    private static final boolean TRACE = Boolean.getBoolean("jextract.trace.downcalls");
    // Conservative work bound for short, synchronous critical calls; not a timing claim.
    private static final int MAX_CRITICAL_CELLS = 1 << 16;
    private static final MemorySegment ADDRESS_OF_KERNEL = SymbolLookup.loaderLookup().find(SYMBOL).orElse(null);
    private static final FunctionDescriptor DESCRIPTOR =
            FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS);
    private static final MethodHandle CRITICAL = ADDRESS_OF_KERNEL == null ? null
            : Linker.nativeLinker().downcallHandle(ADDRESS_OF_KERNEL, DESCRIPTOR, Linker.Option.critical(true));
    private static final MethodHandle ORDINARY = ADDRESS_OF_KERNEL == null ? null
            : Linker.nativeLinker().downcallHandle(ADDRESS_OF_KERNEL, DESCRIPTOR);
    // Packing width is an ABI layout constant, not a limit on the batch size.
    private static final int PACKED_ROWS = 32;
    private static final MethodHandle BATCH = SymbolLookup.loaderLookup().find("ash_project_batch_f32")
            .map(symbol -> Linker.nativeLinker().downcallHandle(symbol,
                    FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS)))
            .orElse(null);

    // Calling-thread storage is shared across models and FP32/FP16/INT8 views. It
    // holds no model reference, grows only when necessary and dies with the thread.
    private static final ThreadLocal<Buffers> BUFFERS = ThreadLocal.withInitial(Buffers::new);

    static final class Buffers {
        MemorySegment input = MemorySegment.NULL, output = MemorySegment.NULL;
        // Allocated only by the FP16/INT8 views; carries errors without native thread affinity.
        MemorySegment status = MemorySegment.NULL;

        void ensure(long inputBytes, long outputBytes) {
            if (input.byteSize() >= inputBytes && output.byteSize() >= outputBytes) return;
            long inCapacity = Math.addExact(Math.max(inputBytes, input.byteSize()), 63) & ~63L;
            long outCapacity = Math.addExact(Math.max(outputBytes, output.byteSize()), 63) & ~63L;
            MemorySegment storage = Arena.ofAuto().allocate(Math.addExact(inCapacity, outCapacity), 64);
            input = storage.asSlice(0, inCapacity);
            output = storage.asSlice(inCapacity, outCapacity);
        }
    }

    static Buffers buffers(long inputBytes, long outputBytes) {
        Buffers buffers = BUFFERS.get();
        buffers.ensure(inputBytes, outputBytes);
        return buffers;
    }

    private final MemorySegment matrix;
    // Allocated only for bulk encoding, shared by all encoding workers. Queries
    // continue to use the original matrix and do not pay for this extra backing.
    private volatile MemorySegment batchMatrix;
    private final int rows;
    private final int columns;

    static boolean isAvailable() { return ADDRESS_OF_KERNEL != null; }

    NativeASHProjection(float[][] source) {
        Objects.requireNonNull(source, "matrix");
        if (!isAvailable()) throw new UnsupportedOperationException("Native library does not provide " + SYMBOL);
        rows = source.length;
        if (rows == 0 || Objects.requireNonNull(source[0], "matrix row").length == 0)
            throw new IllegalArgumentException("ASH projection dimensions must be positive");
        columns = source[0].length;
        for (float[] row : source) {
            if (Objects.requireNonNull(row, "matrix row").length != columns) throw new IllegalArgumentException("Ragged ASH projection");
        }
        long rowBytes = (long) columns * Float.BYTES;
        matrix = Arena.ofAuto().allocate(Math.multiplyExact((long) rows, rowBytes), 64);
        for (int row = 0; row < rows; row++)
            MemorySegment.copy(MemorySegment.ofArray(source[row]), 0, matrix, row * rowBytes, rowBytes);
    }

    @Override
    public long ramBytesUsed() {
        MemorySegment packed = batchMatrix;
        return matrix.byteSize() + (packed == null ? 0 : packed.byteSize());
    }

    int rows() { return rows; }
    int columns() { return columns; }
    MemorySegment matrix() { return matrix; }

    private MemorySegment batchMatrix() {
        MemorySegment ready = batchMatrix;
        if (ready == null) {
            synchronized (this) {
                ready = batchMatrix;
                if (ready == null) {
                    ready = Arena.ofAuto().allocate(matrix.byteSize(), 64);
                    float[] tile = new float[Math.multiplyExact(columns, Math.min(rows, PACKED_ROWS))];
                    for (long firstRow = 0; firstRow < rows; firstRow += PACKED_ROWS) {
                        int width = (int) Math.min(PACKED_ROWS, rows - firstRow);
                        for (int col = 0; col < columns; col++) {
                            for (int lane = 0; lane < width; lane++) {
                                long row = firstRow + lane;
                                tile[col * width + lane] = matrix.getAtIndex(JAVA_FLOAT, row * columns + col);
                            }
                        }
                        MemorySegment.copy(MemorySegment.ofArray(tile), 0, ready,
                                firstRow * columns * Float.BYTES, (long) width * columns * Float.BYTES);
                    }
                    batchMatrix = ready;
                }
            }
        }
        return ready;
    }

    @Override
    public void projectBatch(float[][] inputs, float[][] outputs, int count) {
        VectorUtilSupport.AshProjection.validateBatch(inputs, outputs, count);
        if (count == 0) return;
        if (inputs[0].length != columns || outputs[0].length != rows)
            throw new IllegalArgumentException("ASH projection batch shape mismatch");
        if (BATCH == null) {
            // Compatibility with an older native library; Panama providers use
            // the same default batch contract with their own FP32 projection.
            VectorUtilSupport.AshProjection.super.projectBatch(inputs, outputs, count);
            return;
        }
        long inputRowBytes = (long) columns * Float.BYTES;
        long outputRowBytes = (long) rows * Float.BYTES;
        Buffers buffers = buffers(Math.multiplyExact(count, inputRowBytes),
                Math.multiplyExact(count, outputRowBytes));
        for (int i = 0; i < count; i++)
            MemorySegment.copy(MemorySegment.ofArray(inputs[i]), 0, buffers.input,
                    i * inputRowBytes, inputRowBytes);
        // Bulk work uses ordinary FFM so the JVM can reach safepoints. The
        // matrix is packed once; input/output staging grows only when needed.
        try {
            BATCH.invokeExact(batchMatrix(), buffers.input, rows, columns, count, buffers.output);
        } catch (RuntimeException | Error error) {
            throw error;
        } catch (Throwable error) {
            throw new AssertionError("Unexpected checked exception from ash_project_batch_f32", error);
        }
        for (int i = 0; i < count; i++)
            MemorySegment.copy(buffers.output, i * outputRowBytes,
                    MemorySegment.ofArray(outputs[i]), 0, outputRowBytes);
    }

    @Override
    public void project(float[] query, float[] out) {
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(out, "out");
        if (query.length != columns || out.length != rows || query == out)
            throw new IllegalArgumentException("ASH projection shape or aliasing mismatch");
        if ((long) rows * columns <= MAX_CRITICAL_CELLS) {
            invoke(CRITICAL, MemorySegment.ofArray(query), MemorySegment.ofArray(out));
        } else {
            long queryBytes = (long) columns * Float.BYTES;
            long outputBytes = (long) rows * Float.BYTES;
            Buffers buffers = buffers(queryBytes, outputBytes);
            MemorySegment.copy(MemorySegment.ofArray(query), 0, buffers.input, 0, queryBytes);
            invoke(ORDINARY, buffers.input, buffers.output);
            MemorySegment.copy(buffers.output, 0, MemorySegment.ofArray(out), 0, outputBytes);
        }
    }

    @Override
    public boolean supportsBatchProjection() { return BATCH != null; }

    /** Query-only selection; the original FP32 matrix and encoding path are retained. */
    @Override
    public VectorUtilSupport.AshProjection forQueryScoring() {
        String mode = System.getProperty("jvector.ash.queryProjection", "auto");
        switch (mode) {
            case "fp32": return this;
            case "int8": return NativeASHInt8Projection.prepare(matrix, rows, columns);
            case "auto": return NativeASHFP16Projection.prepare(matrix, rows, columns, this, false);
            case "fp16": return NativeASHFP16Projection.prepare(matrix, rows, columns, this, true);
            default: throw new IllegalArgumentException(
                    "jvector.ash.queryProjection must be auto, fp32, fp16 or int8: " + mode);
        }
    }

    private void invoke(MethodHandle handle, MemorySegment query, MemorySegment out) {
        if (TRACE) System.out.printf("%s(%d, %d)%n", SYMBOL, rows, columns);
        try {
            handle.invokeExact(matrix, query, rows, columns, out);
        } catch (RuntimeException | Error error) {
            throw error;
        } catch (Throwable error) {
            throw new AssertionError("Unexpected checked exception from " + SYMBOL, error);
        }
    }
}
