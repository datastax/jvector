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

import io.github.jbellis.jvector.quantization.ASHQueryProjection;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import java.lang.foreign.MemorySegment;

import static io.github.jbellis.jvector.vector.cnative.NativeAshOneDnn.*;

/**
 * Preferred oneDNN FP32 GEMM for bulk base-vector projection, with Highway fallback.
 * Owns no second matrix: the immutable row-major Highway matrix is shared.
 * Single-vector projection and query precision selection stay with the delegate.
 * Library scratch and JIT preparation are included in the caller's first use.
 */
final class NativeASHFP32BatchProjection implements VectorUtilSupport.AshProjection {
    // The optional package already embeds sequential oneDNN for FP16
    // queries. Older packages simply lack the new FP32 entry point.
    private static final class Bindings {
        static final boolean BATCH = NativeASHAddOn.unavailable("jvector_ash_fp32_project_batch") == null;
    }

    // The model charges this adapter; its retained delegate is separate.
    private static final long FALLBACK_HEAP_BYTES =
            RamUsageEstimator.shallowSizeOfInstance(NativeASHProjection.class);

    private final NativeASHProjection fallback;
    static final int HIGHWAY_MIN_ROWS = 4, HIGHWAY_MAX_ROWS = 32;

    static VectorUtilSupport.AshProjection prepare(NativeASHProjection fallback) {
        return Bindings.BATCH ? new NativeASHFP32BatchProjection(fallback) : fallback;
    }

    private NativeASHFP32BatchProjection(NativeASHProjection fallback) {
        this.fallback = fallback;
    }

    @Override
    public long ramBytesUsed() { return FALLBACK_HEAP_BYTES + fallback.ramBytesUsed(); }

    @Override
    public long nativeBytesUsed() { return fallback.nativeBytesUsed(); }

    @Override
    public void project(float[] input, float[] output) { fallback.project(input, output); }

    @Override
    public VectorUtilSupport.AshProjection forQueryScoring(ASHQueryProjection precision) {
        var query = fallback.forQueryScoring(precision);
        // No new representation means the same backing. Preserve the owning view's
        // identity so model accounting does not charge the FP32 matrix twice.
        return query == fallback ? this : query;
    }

    /** projectBatch always succeeds: oneDNN SGEMM, or the Highway batch kernel when oneDNN declines. */
    @Override
    public boolean supportsBatchProjection() { return true; }

    @Override
    public void projectBatch(float[][] inputs, float[][] outputs, int count) {
        VectorUtilSupport.AshProjection.validateBatch(inputs, outputs, count);
        if (count == 0) return;
        // Highway's register-tiled kernel beats SGEMM for query-sized batches
        // (about 2x at 8 rows on Ice Lake and Granite Rapids); oneDNN wins for
        // pairs and for encoding-sized batches. Both accumulate in FP32.
        if (count >= HIGHWAY_MIN_ROWS && count <= HIGHWAY_MAX_ROWS && fallback.supportsBatchProjection()) {
            fallback.projectBatch(inputs, outputs, count);
            return;
        }
        final int rows = fallback.rows(), columns = fallback.columns();
        if (inputs[0].length != columns || outputs[0].length != rows)
            throw new IllegalArgumentException("ASH projection batch shape mismatch");
        long inputRowBytes = (long) columns * Float.BYTES;
        long outputRowBytes = (long) rows * Float.BYTES;
        NativeASHProjection.Buffers buffers = NativeASHProjection.buffers(
                Math.multiplyExact(count, inputRowBytes), Math.multiplyExact(count, outputRowBytes));
        for (int i = 0; i < count; i++)
            MemorySegment.copy(MemorySegment.ofArray(inputs[i]), 0, buffers.input,
                    i * inputRowBytes, inputRowBytes);
        final int status = jvector_ash_fp32_project_batch(fallback.matrix(), buffers.input,
                rows, columns, count, buffers.output);
        // oneDNN status values are translated by the C wrapper. Unsupported
        // kernels retain Highway; allocation/runtime failures are never hidden.
        if (status == 3) {
            fallback.projectBatch(inputs, outputs, count);
            return;
        }
        if (status == 1) throw new OutOfMemoryError("oneDNN FP32 projection allocation failed");
        if (status == 2) throw new IllegalArgumentException("Invalid oneDNN FP32 projection arguments");
        if (status != 0) throw new IllegalStateException("oneDNN FP32 projection failed: " + status);
        for (int i = 0; i < count; i++)
            MemorySegment.copy(buffers.output, i * outputRowBytes,
                    MemorySegment.ofArray(outputs[i]), 0, outputRowBytes);
    }
}
