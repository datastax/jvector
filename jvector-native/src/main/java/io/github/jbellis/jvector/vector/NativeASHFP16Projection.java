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

import io.github.jbellis.jvector.vector.cnative.jvector_ash_fp16_result;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

import static io.github.jbellis.jvector.vector.cnative.NativeAshOneDnn.*;

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
    private static final int OK = JV_ASH_FP16_OK(), UNSUPPORTED = JV_ASH_FP16_UNSUPPORTED(),
            INVALID_ARGUMENT = JV_ASH_FP16_INVALID_ARGUMENT(), RANGE = JV_ASH_FP16_RANGE(),
            NO_MEMORY = JV_ASH_FP16_NO_MEMORY();

    /** Checked only if a query actually asks for an FP16 view. */
    private static final class Bindings {
        /** Null when usable, otherwise why the add-on cannot serve FP16 queries. */
        static final String UNAVAILABLE = NativeASHAddOn.unavailable("jvector_ash_fp16_create",
                "jvector_ash_fp16_destroy", "jvector_ash_fp16_project", "jvector_ash_fp16_bytes");
        // Older optional libraries retain correct singleton FP16 behavior.
        static final boolean BATCH = UNAVAILABLE == null
                && NativeASHAddOn.unavailable("jvector_ash_fp16_project_batch") == null;
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
        this.batchAvailable = Bindings.BATCH;
        this.fallback = fallback;
    }

    /** Probes the add-on once, without a model: null when FP16 query projection works here, else why not. */
    static String unsupportedReason() {
        return Probe.REASON;
    }

    private static final class Probe {
        static final String REASON = probe();

        private static String probe() {
            if (Bindings.UNAVAILABLE != null) return NativeASHAddOn.missing("FP16", Bindings.UNAVAILABLE);
            int rows = 64, columns = 128;
            try (Arena setup = Arena.ofConfined()) {
                MemorySegment matrix = setup.allocate(java.lang.foreign.ValueLayout.JAVA_FLOAT, (long) rows * columns);
                for (int r = 0; r < rows; r++)
                    matrix.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, (long) r * columns + r, 1f);
                MemorySegment status = jvector_ash_fp16_result.allocate(setup);
                MemorySegment pointer = jvector_ash_fp16_create(matrix, rows, columns, status);
                if (pointer.address() != 0) jvector_ash_fp16_destroy(pointer);
                int code = jvector_ash_fp16_result.status(status);
                if (code == OK) return null;
                if (code == UNSUPPORTED)
                    return "FP16 query projection needs AVX-512 FP16; this CPU's best tier is "
                            + NativeASHAddOn.activeIsa() + ". Use auto or fp32.";
                return "FP16 query projection failed to initialize: "
                        + NativeASHAddOn.trim(jvector_ash_fp16_result.message(status).getString(0)) + ". Use auto or fp32.";
            }
        }
    }

    /** Model initialization only; returns fallback for genuinely unsupported automatic selection. */
    static VectorUtilSupport.AshProjection prepare(MemorySegment matrix, int rows, int columns,
                                                   VectorUtilSupport.AshProjection fallback,
                                                   boolean required) {
        if (Bindings.UNAVAILABLE != null) {
            if (!required) return fallback;
            throw new UnsupportedOperationException(unsupportedReason());
        }
        try (Arena setup = Arena.ofConfined()) {
            MemorySegment status = jvector_ash_fp16_result.allocate(setup);
            MemorySegment pointer = jvector_ash_fp16_create(matrix, rows, columns, status);
            int code = jvector_ash_fp16_result.status(status);
            if (pointer.address() == 0) {
                if (!required && (code == UNSUPPORTED || code == RANGE)) return fallback;
                if (code == UNSUPPORTED) {
                    String reason = unsupportedReason();
                    if (reason != null) throw new UnsupportedOperationException(reason);
                }
                throw failure(code, status);
            }
            if (code != OK) {
                jvector_ash_fp16_destroy(pointer);
                throw failure(code, status);
            }
            long bytes;
            MemorySegment owned;
            try {
                bytes = jvector_ash_fp16_bytes(pointer);
                owned = pointer.reinterpret(1, Arena.ofAuto(), NativeASHFP16Projection::destroy);
            } catch (Throwable failure) {
                jvector_ash_fp16_destroy(pointer);
                throw failure;
            }
            return new NativeASHFP16Projection(owned, rows, columns, bytes, required ? null : fallback);
        }
    }

    private static void destroy(MemorySegment pointer) {
        jvector_ash_fp16_destroy(pointer);
    }

    private static RuntimeException failure(int code, MemorySegment result) {
        String message = "ASH FP16 projection: " + jvector_ash_fp16_result.message(result).getString(0);
        if (code == NO_MEMORY) throw new OutOfMemoryError(message);
        if (code == UNSUPPORTED) return new UnsupportedOperationException(message);
        if (code == INVALID_ARGUMENT || code == RANGE) return new IllegalArgumentException(message);
        return new IllegalStateException(message + " (status " + code + ")");
    }

    /** Native owned storage; per-thread staging and opaque oneDNN scratch are excluded. */
    @Override
    public long ramBytesUsed() { return retainedBytes; }

    @Override
    public long nativeBytesUsed() { return retainedBytes; }

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
        if (buffers.status.byteSize() == 0) buffers.status = jvector_ash_fp16_result.allocate(Arena.ofAuto());
        int status = jvector_ash_fp16_project_batch(
                context, buffers.input, count, buffers.output, buffers.status);
        // Publish lazy packing even on a partially successful/failed native
        // call. Concurrent callers cannot overwrite a newer, larger total.
        long current = jvector_ash_fp16_bytes(context);
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
        if (buffers.status.byteSize() == 0) buffers.status = jvector_ash_fp16_result.allocate(Arena.ofAuto());
        int status = jvector_ash_fp16_project(
                context, buffers.input, buffers.output, buffers.status);
        if (status == RANGE && fallback != null) {
            fallback.project(query, out);
            fp32FallbackUsed = true;
            return;
        }
        if (status != OK) throw failure(status, buffers.status);
        MemorySegment.copy(buffers.output, 0, MemorySegment.ofArray(out), 0, outputBytes);
    }
}
