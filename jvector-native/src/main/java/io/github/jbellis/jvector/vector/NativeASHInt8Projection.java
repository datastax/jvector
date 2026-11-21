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

import io.github.jbellis.jvector.vector.cnative.NativeSimdOps;
import io.github.jbellis.jvector.vector.cnative.jvector_ash_fp16_result;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

import static io.github.jbellis.jvector.vector.cnative.NativeAshOneDnn.*;

/**
 * Immutable, model-owned oneDNN INT8 query view. Training/encoding retain FP32.
 * Reuses the existing projection contract, caller-owned batching and calling-thread
 * staging. Native scratch is thread-local and retains no model or packed weights.
 * Explicit INT8 selection never silently substitutes another precision.
 */
final class NativeASHInt8Projection implements VectorUtilSupport.AshProjection {
    private static final int OK = JV_ASH_FP16_OK(), UNSUPPORTED = JV_ASH_FP16_UNSUPPORTED(),
            INVALID_ARGUMENT = JV_ASH_FP16_INVALID_ARGUMENT(), RANGE = JV_ASH_FP16_RANGE(),
            NO_MEMORY = JV_ASH_FP16_NO_MEMORY();

    // The optional add-on shares the existing ABI-1 caller-owned result structure.
    static final String[] SYMBOLS = {"jvector_ash_int8_create", "jvector_ash_int8_destroy",
            "jvector_ash_int8_project_batch", "jvector_ash_int8_bytes"};

    private static final class Bindings {
        /** Null when usable, otherwise why the add-on cannot serve INT8 queries. */
        static final String UNAVAILABLE = NativeASHAddOn.unavailable(SYMBOLS);
    }

    private final MemorySegment context;
    private final int rows, columns;

    private NativeASHInt8Projection(MemorySegment context, int rows, int columns) {
        this.context = context;
        this.rows = rows;
        this.columns = columns;
    }

    /** Probes the add-on once, without a model: null when it projects INT8 on this processor, else why not. */
    static String unsupportedReason() {
        return Probe.REASON;
    }

    private static final class Probe {
        static final String REASON = probe();

        private static String probe() {
            if (Bindings.UNAVAILABLE != null) return "the oneDNN add-on is unavailable (" + Bindings.UNAVAILABLE + ")";
            int rows = 64, columns = 128;
            try (Arena setup = Arena.ofConfined()) {
                MemorySegment matrix = setup.allocate(java.lang.foreign.ValueLayout.JAVA_FLOAT, (long) rows * columns);
                for (int r = 0; r < rows; r++)
                    matrix.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, (long) r * columns + r, 1f);
                MemorySegment status = jvector_ash_fp16_result.allocate(setup);
                MemorySegment pointer = jvector_ash_int8_create(matrix, rows, columns, status);
                if (pointer.address() != 0) jvector_ash_int8_destroy(pointer);
                int code = jvector_ash_fp16_result.status(status);
                if (code == OK) return null;
                if (code == UNSUPPORTED) return "oneDNN has no optimized INT8 kernel for it";
                return "oneDNN reports " + jvector_ash_fp16_result.message(status).getString(0).replaceAll("[.\\s]+$", "");
            }
        }
    }

    /** The full explanation for an INT8 failure on this processor. */
    static String explain(String reason) {
        String isa = NativeSimdOps.jvector_simd_get_active_isa().reinterpret(Long.MAX_VALUE).getString(0);
        return "INT8 ASH query projection is not available on this processor (Highway tier " + isa + "): "
                + reason + ". It needs AVX-512 VNNI, or AVX-VNNI with the oneDNN add-on (-Dnative.onednn=true)."
                + " Use auto or fp32.";
    }

    static VectorUtilSupport.AshProjection prepare(MemorySegment matrix, int rows, int columns) {
        if (Bindings.UNAVAILABLE != null)
            throw new UnsupportedOperationException(explain("the oneDNN add-on is unavailable ("
                    + Bindings.UNAVAILABLE + ")"));
        try (Arena setup = Arena.ofConfined()) {
            MemorySegment status = jvector_ash_fp16_result.allocate(setup);
            MemorySegment pointer = jvector_ash_int8_create(matrix, rows, columns, status);
            int code = jvector_ash_fp16_result.status(status);
            if (pointer.address() == 0) throw failure(code, status);
            if (code != OK) {
                jvector_ash_int8_destroy(pointer);
                throw failure(code, status);
            }
            MemorySegment owned;
            try {
                owned = pointer.reinterpret(1, Arena.ofAuto(), NativeASHInt8Projection::destroy);
            } catch (Throwable failure) {
                jvector_ash_int8_destroy(pointer);
                throw failure;
            }
            return new NativeASHInt8Projection(owned, rows, columns);
        }
    }

    private static void destroy(MemorySegment pointer) {
        jvector_ash_int8_destroy(pointer);
    }

    private static RuntimeException failure(int code, MemorySegment status) {
        String message = "ASH INT8 projection: " + jvector_ash_fp16_result.message(status).getString(0);
        if (code == NO_MEMORY) throw new OutOfMemoryError(message);
        if (code == UNSUPPORTED) return new UnsupportedOperationException(explain("oneDNN has no optimized INT8 kernel for it"));
        if (code == INVALID_ARGUMENT || code == RANGE) return new IllegalArgumentException(message);
        return new IllegalStateException(message + " (status " + code + ")");
    }

    /** Retained native model storage, including lazily packed batch weights. */
    @Override
    public long ramBytesUsed() {
        return jvector_ash_int8_bytes(context);
    }

    /** All of this view's storage is native. */
    @Override
    public long nativeBytesUsed() { return ramBytesUsed(); }

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
        if (buffers.status.byteSize() == 0) buffers.status = jvector_ash_fp16_result.allocate(Arena.ofAuto());
        int status = jvector_ash_int8_project_batch(
                context, buffers.input, count, buffers.output, buffers.status);
        if (status != OK) throw failure(status, buffers.status);
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
