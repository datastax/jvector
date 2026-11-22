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
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * Immutable INT8 query view projected by the Highway kernel on CPUs with AVX-512 VNNI.
 * The model owns its INT8 rows, one FP32 scale and one int32 sum per row, packed once for
 * the kernel; the kernel itself is stateless. Each query is two INT8 digits with one scale
 * and every dot product is exact, so results do not depend on the CPU tier or batch size and
 * equal the oneDNN view's. Training and encoding keep the FP32 projection.
 */
final class NativeASHInt8HighwayProjection implements VectorUtilSupport.AshProjection {
    private static final MethodHandle AVAILABLE = downcall("ash_project_i8_available",
            FunctionDescriptor.of(JAVA_INT));
    private static final MethodHandle BATCH = downcall("ash_project_batch_i8",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT,
                    ADDRESS, ADDRESS));
    private static final int OK = 0, INVALID_SHAPE = 2, SCALE_UNDERFLOW = 3;
    /** Query rows per kernel tile; the scratch holds one tile. */
    private static final int TILE = 8;
    /** int32 accumulators stay exact up to this many columns. */
    static final int MAX_COLUMNS = 65535;

    private static MethodHandle downcall(String symbol, FunctionDescriptor descriptor) {
        return SymbolLookup.loaderLookup().find(symbol)
                .map(address -> Linker.nativeLinker().downcallHandle(address, descriptor))
                .orElse(null);
    }

    /** Whether the loaded native library projects INT8 on this CPU (a VNNI tier). */
    static boolean isAvailable() {
        if (AVAILABLE == null || BATCH == null) return false;
        try {
            return (int) AVAILABLE.invokeExact() == 1;
        } catch (Throwable error) {
            throw new AssertionError("Unexpected failure from ash_project_i8_available", error);
        }
    }

    private final MemorySegment packed, scales, sums;
    private final int rows, columns, paddedRows, paddedColumns;
    private final long handleBytes;

    private NativeASHInt8HighwayProjection(MemorySegment packed, MemorySegment scales, MemorySegment sums,
                                           int rows, int columns) {
        this.packed = packed;
        this.scales = scales;
        this.sums = sums;
        this.rows = rows;
        this.columns = columns;
        this.paddedRows = (rows + 15) & ~15;
        this.paddedColumns = (columns + 3) & ~3;
        // The three segments share one arena, so its scope is counted once.
        this.handleBytes = NativeSegmentHeap.of(packed) + io.github.jbellis.jvector.util.RamUsageEstimator.shallowSizeOf(scales)
                + io.github.jbellis.jvector.util.RamUsageEstimator.shallowSizeOf(sums);
    }

    /**
     * Quantizes each row of the row-major FP32 {@code matrix} to INT8 with its own scale and
     * packs it for the kernel: [row block of 16][column group of 4][16 rows][4 columns], zero padded.
     */
    static VectorUtilSupport.AshProjection prepare(MemorySegment matrix, int rows, int columns) {
        if (!isAvailable())
            throw new UnsupportedOperationException("ASH INT8 query projection requires a CPU with AVX-512 VNNI");
        if (rows <= 0 || columns <= 0) throw new IllegalArgumentException("Invalid INT8 matrix dimensions");
        if (columns > MAX_COLUMNS) throw new IllegalArgumentException("INT8 accumulator dimension exceeds 65535");
        int paddedRows = (rows + 15) & ~15, paddedColumns = (columns + 3) & ~3;
        Arena arena = Arena.ofAuto();
        MemorySegment packed = arena.allocate(Math.multiplyExact((long) paddedRows, paddedColumns), 64);
        MemorySegment scales = arena.allocate((long) rows * Float.BYTES, 64);
        MemorySegment sums = arena.allocate((long) rows * Integer.BYTES, 64);
        float[] values = new float[columns];
        byte[] codes = new byte[columns];
        long groupBytes = 64, blockBytes = (long) (paddedColumns / 4) * groupBytes;
        for (int row = 0; row < rows; row++) {
            MemorySegment.copy(matrix, JAVA_FLOAT, (long) row * columns * Float.BYTES, values, 0, columns);
            int sum = 0;
            float scale = scale(values);
            for (int column = 0; column < columns; column++) {
                int q = Math.max(-127, Math.min(127, (int) Math.rint(values[column] / scale)));
                codes[column] = (byte) q;
                sum += q;
            }
            scales.setAtIndex(JAVA_FLOAT, row, scale);
            sums.setAtIndex(JAVA_INT, row, sum);
            long blockStart = (row / 16) * blockBytes + (row % 16) * 4L;
            for (int column = 0; column < columns; column++)
                packed.set(JAVA_BYTE, blockStart + (column / 4) * groupBytes + column % 4, codes[column]);
        }
        return new NativeASHInt8HighwayProjection(packed, scales, sums, rows, columns);
    }

    /** The row's scale, maxabs / 127, exactly as the kernel scales a query. */
    private static float scale(float[] values) {
        float maxabs = 0;
        for (float value : values) {
            if (!Float.isFinite(value)) throw new IllegalArgumentException("Nonfinite INT8 matrix operand");
            maxabs = Math.max(maxabs, Math.abs(value));
        }
        float scale = maxabs == 0 ? 1.0f : maxabs / 127.0f;
        if (scale == 0) throw new IllegalArgumentException("INT8 operand scale underflows FP32; rescale input");
        return scale;
    }

    /** Packed rows, scales and sums, all native. */
    @Override
    public long ramBytesUsed() {
        return nativeBytesUsed() + handleBytes;
    }

    @Override
    public long nativeBytesUsed() { return packed.byteSize() + scales.byteSize() + sums.byteSize(); }

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

    private void execute(NativeASHProjection.Buffers buffers, int count) {
        // Two digit rows per query and two int32 dot products per output, for one tile.
        MemorySegment scratch = buffers.scratch(2L * TILE * paddedColumns + 2L * TILE * paddedRows * Integer.BYTES);
        int status;
        try {
            status = (int) BATCH.invokeExact(packed, scales, sums, rows, columns, buffers.input, count,
                    scratch, buffers.output);
        } catch (RuntimeException | Error error) {
            throw error;
        } catch (Throwable error) {
            throw new AssertionError("Unexpected checked exception from ash_project_batch_i8", error);
        }
        if (status == OK) return;
        if (status == SCALE_UNDERFLOW)
            throw new IllegalArgumentException("ASH INT8 projection: INT8 operand scale underflows FP32; rescale input");
        if (status == INVALID_SHAPE)
            throw new IllegalArgumentException("ASH INT8 projection: invalid shape or buffers");
        throw new IllegalStateException("ASH INT8 projection failed (status " + status + ")");
    }
}
