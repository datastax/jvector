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

import io.github.jbellis.jvector.vector.cnative.LibraryLoader;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.Objects;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Immutable, model-owned oneDNN INT8 query view. Training/encoding retain FP32.
 * Reuses the existing projection contract, caller-owned batching and calling-thread
 * staging. Native scratch is thread-local and retains no model or packed weights.
 * Explicit INT8 selection never silently substitutes another precision.
 */
final class NativeASHInt8Projection implements VectorUtilSupport.AshProjection {
    private static final int OK = 0, UNSUPPORTED = 1, INVALID_ARGUMENT = 2,
            RANGE = 3, NO_MEMORY = 4;
    // The optional add-on shares the existing ABI-1 caller-owned result structure.
    private static final MemoryLayout RESULT_LAYOUT = MemoryLayout.structLayout(
            JAVA_INT, MemoryLayout.sequenceLayout(256, JAVA_BYTE));

    private static final class Bindings {
        static final Bindings INSTANCE = load();
        final MethodHandle create, destroy, batch, bytes;

        private Bindings(SymbolLookup lookup) throws Throwable {
            MethodHandle abi = bind(lookup, "jvector_ash_fp16_abi", FunctionDescriptor.of(JAVA_INT));
            if ((int) abi.invokeExact() != 1)
                throw new UnsatisfiedLinkError("Unsupported ASH projection add-on ABI");
            create = bind(lookup, "jvector_ash_int8_create",
                    FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS));
            destroy = bind(lookup, "jvector_ash_int8_destroy", FunctionDescriptor.ofVoid(ADDRESS));
            batch = bind(lookup, "jvector_ash_int8_project_batch",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));
            bytes = bind(lookup, "jvector_ash_int8_bytes", FunctionDescriptor.of(JAVA_LONG, ADDRESS));
        }

        private static MethodHandle bind(SymbolLookup lookup, String name, FunctionDescriptor descriptor) {
            MemorySegment symbol = lookup.find(name)
                    .orElseThrow(() -> new UnsatisfiedLinkError("Missing ASH INT8 symbol: " + name));
            return Linker.nativeLinker().downcallHandle(symbol, descriptor);
        }

        private static Bindings load() {
            if (!LibraryLoader.loadAshFp16()) return null;
            try {
                return new Bindings(SymbolLookup.loaderLookup());
            } catch (UnsatisfiedLinkError | IllegalCallerException | SecurityException
                     | UnsupportedOperationException unavailable) {
                return null;
            } catch (RuntimeException | Error failure) {
                throw failure;
            } catch (Throwable failure) {
                throw new AssertionError("Unexpected ASH INT8 binding failure", failure);
            }
        }
    }

    private final MemorySegment context;
    private final int rows, columns;

    private NativeASHInt8Projection(MemorySegment context, int rows, int columns) {
        this.context = context;
        this.rows = rows;
        this.columns = columns;
    }

    static VectorUtilSupport.AshProjection prepare(MemorySegment matrix, int rows, int columns) {
        Bindings bindings = Bindings.INSTANCE;
        if (bindings == null)
            throw new UnsupportedOperationException("ASH INT8 query projection add-on is unavailable; "
                    + "build with -Dnative.onednn=true or select fp32/auto");
        try (Arena setup = Arena.ofConfined()) {
            MemorySegment status = setup.allocate(RESULT_LAYOUT);
            MemorySegment pointer = (MemorySegment) bindings.create.invokeExact(matrix, rows, columns, status);
            int code = status.get(JAVA_INT, 0);
            if (pointer.address() == 0) throw failure(code, status);
            if (code != OK) {
                bindings.destroy.invokeExact(pointer);
                throw failure(code, status);
            }
            MemorySegment owned;
            try {
                owned = pointer.reinterpret(1, Arena.ofAuto(), NativeASHInt8Projection::destroy);
            } catch (Throwable failure) {
                bindings.destroy.invokeExact(pointer);
                throw failure;
            }
            return new NativeASHInt8Projection(owned, rows, columns);
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new AssertionError("Unexpected ASH INT8 initialization failure", failure);
        }
    }

    private static void destroy(MemorySegment pointer) {
        try {
            Bindings.INSTANCE.destroy.invokeExact(pointer);
        } catch (Throwable failure) {
            throw new AssertionError("Unexpected ASH INT8 cleanup failure", failure);
        }
    }

    private static RuntimeException failure(int code, MemorySegment status) {
        String message = "ASH INT8 projection: " + status.getString(JAVA_INT.byteSize());
        if (code == NO_MEMORY) throw new OutOfMemoryError(message);
        if (code == UNSUPPORTED) return new UnsupportedOperationException(message);
        if (code == INVALID_ARGUMENT || code == RANGE) return new IllegalArgumentException(message);
        return new IllegalStateException(message + " (status " + code + ")");
    }

    /** Retained native model storage, including lazily packed batch weights. */
    @Override
    public long ramBytesUsed() {
        try {
            return (long) Bindings.INSTANCE.bytes.invokeExact(context);
        } catch (Throwable failure) {
            throw new AssertionError("Unexpected ASH INT8 accounting failure", failure);
        }
    }

    @Override
    public String operandPrecision() { return "int8"; }

    @Override
    public boolean supportsBatchProjection() { return true; }

    @Override
    public void project(float[] query, float[] output) {
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(output, "output");
        if (query.length != columns || output.length != rows || query == output)
            throw new IllegalArgumentException("ASH projection shape or aliasing mismatch");
        long inputBytes = (long) columns * Float.BYTES;
        long outputBytes = (long) rows * Float.BYTES;
        var buffers = NativeASHProjection.buffers(inputBytes, outputBytes);
        MemorySegment.copy(MemorySegment.ofArray(query), 0, buffers.input, 0, inputBytes);
        execute(buffers, 1);
        MemorySegment.copy(buffers.output, 0, MemorySegment.ofArray(output), 0, outputBytes);
    }

    private void execute(NativeASHProjection.Buffers buffers, int count) {
        if (buffers.status.byteSize() == 0) buffers.status = Arena.ofAuto().allocate(RESULT_LAYOUT);
        try {
            int status = (int) Bindings.INSTANCE.batch.invokeExact(
                    context, buffers.input, count, buffers.output, buffers.status);
            if (status != OK) throw failure(status, buffers.status);
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new AssertionError("Unexpected ASH INT8 execution failure", failure);
        }
    }

    @Override
    public void projectBatch(float[][] queries, float[][] outputs, int count) {
        VectorUtilSupport.AshProjection.validateBatch(queries, outputs, count);
        if (count == 0) return;
        if (queries[0].length != columns || outputs[0].length != rows)
            throw new IllegalArgumentException("ASH INT8 projection batch shape mismatch");
        long inputRowBytes = (long) columns * Float.BYTES;
        long outputRowBytes = (long) rows * Float.BYTES;
        var buffers = NativeASHProjection.buffers(
                Math.multiplyExact(count, inputRowBytes), Math.multiplyExact(count, outputRowBytes));
        for (int i = 0; i < count; i++)
            MemorySegment.copy(MemorySegment.ofArray(queries[i]), 0, buffers.input,
                    i * inputRowBytes, inputRowBytes);
        execute(buffers, count);
        for (int i = 0; i < count; i++)
            MemorySegment.copy(buffers.output, i * outputRowBytes,
                    MemorySegment.ofArray(outputs[i]), 0, outputRowBytes);
    }
}
