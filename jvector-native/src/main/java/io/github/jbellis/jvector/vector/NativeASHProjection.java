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

/**
 * Immutable native row-major matrix for one complete ASH query projection per downcall.
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

    // Calling-thread storage is shared across models and FP32/FP16 views. It
    // holds no model reference, grows only when necessary and dies with the thread.
    private static final ThreadLocal<Buffers> BUFFERS = ThreadLocal.withInitial(Buffers::new);

    static final class Buffers {
        MemorySegment input = MemorySegment.NULL, output = MemorySegment.NULL;
        // Allocated only by the FP16 view; carries errors without native thread affinity.
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
    public long ramBytesUsed() { return matrix.byteSize(); }

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

    /** Query-only selection; the original FP32 matrix and encoding path are retained. */
    @Override
    public VectorUtilSupport.AshProjection forQueryScoring() {
        String mode = System.getProperty("jvector.ash.queryProjection", "auto");
        switch (mode) {
            case "fp32": return this;
            case "auto": return NativeASHFP16Projection.prepare(matrix, rows, columns, this, false);
            case "fp16": return NativeASHFP16Projection.prepare(matrix, rows, columns, this, true);
            default: throw new IllegalArgumentException(
                    "jvector.ash.queryProjection must be auto, fp32 or fp16: " + mode);
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
