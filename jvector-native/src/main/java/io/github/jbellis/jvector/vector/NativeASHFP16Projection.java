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
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Immutable, model-owned FP16 query matrix with strict FP32 accumulation/output.
 * The optional library uses sequential oneDNN; parallelism belongs to search
 * callers. A projection shares the Highway adapter's calling-thread staging,
 * while native conversion storage and scratch are reused by the calling thread.
 * Neither thread-local cache retains the model or its matrix.
 *
 * <p>Only query-scoring views use this representation. Encoding retains Highway
 * FP32. Singleton and batch calls execute on the calling thread.
 */
final class NativeASHFP16Projection implements VectorUtilSupport.AshProjection {
    private static final int OK = 0, UNSUPPORTED = 1, INVALID_ARGUMENT = 2,
            RANGE = 3, NO_MEMORY = 4;

    private static final MemoryLayout RESULT_LAYOUT = MemoryLayout.structLayout(
            JAVA_INT, MemoryLayout.sequenceLayout(256, JAVA_BYTE));

    /** Loaded only if a query actually asks for an FP16 view. */
    private static final class Bindings {
        static final Bindings INSTANCE = load();
        final MethodHandle create, destroy, project, bytes, batch;

        private Bindings(SymbolLookup lookup) throws Throwable {
            MethodHandle abi = bind(lookup, "jvector_ash_fp16_abi", FunctionDescriptor.of(JAVA_INT));
            if ((int) abi.invokeExact() != 1)
                throw new UnsatisfiedLinkError("Unsupported ASH FP16 library ABI");
            create = bind(lookup, "jvector_ash_fp16_create",
                    FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS));
            destroy = bind(lookup, "jvector_ash_fp16_destroy", FunctionDescriptor.ofVoid(ADDRESS));
            project = bind(lookup, "jvector_ash_fp16_project",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
            bytes = bind(lookup, "jvector_ash_fp16_bytes", FunctionDescriptor.of(JAVA_LONG, ADDRESS));
            // Older optional libraries retain correct singleton FP16 behavior.
            batch = lookup.find("jvector_ash_fp16_project_batch")
                    .map(symbol -> Linker.nativeLinker().downcallHandle(symbol,
                            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS)))
                    .orElse(null);
        }

        private static MethodHandle bind(SymbolLookup lookup, String name, FunctionDescriptor descriptor) {
            MemorySegment symbol = lookup.find(name)
                    .orElseThrow(() -> new UnsatisfiedLinkError("Missing ASH FP16 symbol: " + name));
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
                throw new AssertionError("Unexpected ASH FP16 binding failure", failure);
            }
        }
    }

    private final MemorySegment context;
    private final int rows, columns;
    private volatile long retainedBytes;
    private volatile boolean batchAvailable;
    // Set only on the existing automatic FP32 range-fallback branch.
    private volatile boolean fp32FallbackUsed;
    private static final java.lang.invoke.VarHandle RETAINED_BYTES;
    static {
        try {
            RETAINED_BYTES = java.lang.invoke.MethodHandles.lookup().findVarHandle(
                    NativeASHFP16Projection.class, "retainedBytes", long.class);
        } catch (ReflectiveOperationException failure) {
            throw new ExceptionInInitializerError(failure);
        }
    }
    // Present only in automatic mode, for queries outside the finite FP16 range.
    private final VectorUtilSupport.AshProjection fallback;

    private NativeASHFP16Projection(MemorySegment context, int rows, int columns,
                                    long weightBytes, VectorUtilSupport.AshProjection fallback) {
        this.context = context;
        this.rows = rows;
        this.columns = columns;
        this.retainedBytes = weightBytes;
        this.batchAvailable = Bindings.INSTANCE.batch != null;
        this.fallback = fallback;
    }

    /** Model initialization only; returns fallback for genuinely unsupported automatic selection. */
    static VectorUtilSupport.AshProjection prepare(MemorySegment matrix, int rows, int columns,
                                                   VectorUtilSupport.AshProjection fallback,
                                                   boolean required) {
        Bindings bindings = Bindings.INSTANCE;
        if (bindings == null) {
            if (!required) return fallback;
            throw new UnsupportedOperationException("ASH FP16 query projection library is unavailable; "
                    + "build with -Dnative.onednn=true or select fp32/auto");
        }
        try (Arena setup = Arena.ofConfined()) {
            MemorySegment status = setup.allocate(RESULT_LAYOUT);
            MemorySegment pointer = (MemorySegment) bindings.create.invokeExact(matrix, rows, columns, status);
            int code = status.get(JAVA_INT, 0);
            if (pointer.address() == 0) {
                if (!required && (code == UNSUPPORTED || code == RANGE)) return fallback;
                throw failure(code, status);
            }
            if (code != OK) {
                bindings.destroy.invokeExact(pointer);
                throw failure(code, status);
            }
            long bytes;
            MemorySegment owned;
            try {
                bytes = (long) bindings.bytes.invokeExact(pointer);
                owned = pointer.reinterpret(1, Arena.ofAuto(), NativeASHFP16Projection::destroy);
            } catch (Throwable failure) {
                bindings.destroy.invokeExact(pointer);
                throw failure;
            }
            return new NativeASHFP16Projection(owned, rows, columns, bytes, required ? null : fallback);
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new AssertionError("Unexpected ASH FP16 initialization failure", failure);
        }
    }

    private static void destroy(MemorySegment pointer) {
        try {
            Bindings.INSTANCE.destroy.invokeExact(pointer);
        } catch (Throwable failure) {
            throw new AssertionError("Unexpected ASH FP16 cleanup failure", failure);
        }
    }

    private static RuntimeException failure(int code, MemorySegment result) {
        String message = "ASH FP16 projection: " + result.getString(JAVA_INT.byteSize());
        if (code == NO_MEMORY) throw new OutOfMemoryError(message);
        if (code == UNSUPPORTED) return new UnsupportedOperationException(message);
        if (code == INVALID_ARGUMENT || code == RANGE) return new IllegalArgumentException(message);
        return new IllegalStateException(message + " (status " + code + ")");
    }

    /** Native owned storage; per-thread staging and opaque oneDNN scratch are excluded. */
    @Override
    public long ramBytesUsed() { return retainedBytes; }

    @Override
    public String operandPrecision() { return fp32FallbackUsed ? "fp16+fp32" : "fp16"; }

    @Override
    public boolean supportsBatchProjection() { return batchAvailable; }

    /** Shares prepared FP16 weights; bounded native tiles handle arbitrary batch lengths. */
    @Override
    public void projectBatch(float[][] inputs, float[][] outputs, int count) {
        VectorUtilSupport.AshProjection.validateBatch(inputs, outputs, count);
        if (count == 0) return;
        if (inputs[0].length != columns || outputs[0].length != rows)
            throw new IllegalArgumentException("ASH FP16 batch shape mismatch");
        if (count == 1 || !batchAvailable) {
            for (int i = 0; i < count; i++) project(inputs[i], outputs[i]);
            return;
        }
        long inputRowBytes = (long) columns * Float.BYTES;
        long outputRowBytes = (long) rows * Float.BYTES;
        NativeASHProjection.Buffers buffers = NativeASHProjection.buffers(
                Math.multiplyExact(count, inputRowBytes), Math.multiplyExact(count, outputRowBytes));
        for (int i = 0; i < count; i++)
            MemorySegment.copy(MemorySegment.ofArray(inputs[i]), 0, buffers.input,
                    i * inputRowBytes, inputRowBytes);
        if (buffers.status.byteSize() == 0) buffers.status = Arena.ofAuto().allocate(RESULT_LAYOUT);
        try {
            int status = (int) Bindings.INSTANCE.batch.invokeExact(
                    context, buffers.input, count, buffers.output, buffers.status);
            // Publish lazy packing even on a partially successful/failed native
            // call. Concurrent callers cannot overwrite a newer, larger total.
            long current = (long) Bindings.INSTANCE.bytes.invokeExact(context);
            for (long previous; (previous = retainedBytes) < current;)
                if (RETAINED_BYTES.compareAndSet(this, previous, current)) break;
            if (status == UNSUPPORTED) batchAvailable = false;
            if (status == UNSUPPORTED || (status == RANGE && fallback != null)) {
                // Preserve each query's FP16/automatic-range behavior, including
                // compatibility with libraries lacking an optimized batch shape.
                for (int i = 0; i < count; i++) project(inputs[i], outputs[i]);
                return;
            }
            if (status != OK) throw failure(status, buffers.status);
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new AssertionError("Unexpected ASH FP16 batch execution failure", failure);
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
        long inputBytes = (long) columns * Float.BYTES;
        long outputBytes = (long) rows * Float.BYTES;
        NativeASHProjection.Buffers buffers = NativeASHProjection.buffers(inputBytes, outputBytes);
        MemorySegment.copy(MemorySegment.ofArray(query), 0, buffers.input, 0, inputBytes);
        if (buffers.status.byteSize() == 0) buffers.status = Arena.ofAuto().allocate(RESULT_LAYOUT);
        try {
            int status = (int) Bindings.INSTANCE.project.invokeExact(
                    context, buffers.input, buffers.output, buffers.status);
            if (status == RANGE && fallback != null) {
                fallback.project(query, out);
                fp32FallbackUsed = true;
                return;
            }
            if (status != OK) throw failure(status, buffers.status);
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new AssertionError("Unexpected ASH FP16 execution failure", failure);
        }
        MemorySegment.copy(buffers.output, 0, MemorySegment.ofArray(out), 0, outputBytes);
    }
}
