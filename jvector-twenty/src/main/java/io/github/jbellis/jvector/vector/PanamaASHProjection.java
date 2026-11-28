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
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;

import java.util.Objects;

/**
 * ASH projection on the Panama provider, with the native provider's views: FP32 for encoding and queries, with one
 * call for a whole batch of queries, and INT8 and FP16 query views. Every body takes plain arrays and is compiled
 * on its own.
 *
 * <p>FP32: a single query keeps {@code ashDotRow}'s arithmetic (one accumulator, then a scalar tail); a batch uses
 * the native batch kernel and layout (a packed copy of the matrix, one output per lane, no lane reduction). The two
 * round differently in the last bits, as the native single and batch kernels do. INT8: the native kernel's quantization and exact integer dot products, so
 * its outputs equal the native INT8 view's. FP16: weights rounded to half precision, FP32 accumulation.
 *
 * <p>Automatic query precision follows the native rule: INT8 where the CPU has an integer dot-product instruction,
 * else FP16 where it has FP16 arithmetic, else FP32. The Vector API reaches neither instruction, so automatic
 * selection here is FP32; INT8 and FP16 are available on request.
 */
final class PanamaASHProjection implements VectorUtilSupport.AshProjection {
    private static final VectorSpecies<Float> F = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> I = VectorSpecies.of(int.class, F.vectorShape());
    // One byte (INT8 weights) or short (FP16 weights) per int lane, widened in registers
    private static final VectorSpecies<Byte> B = F.length() <= 8 ? ByteVector.SPECIES_64 : ByteVector.SPECIES_128;
    private static final VectorSpecies<Short> S = F.length() <= 4 ? ShortVector.SPECIES_64
            : F.length() == 8 ? ShortVector.SPECIES_128 : ShortVector.SPECIES_256;

    // Bytes and shorts widened to int lanes without convertShape, whose shared type profile can leave the result's
    // exact type unknown to C2: element i of a full-width vector is spread to int lane i, then masked or sign-extended.
    private static final VectorSpecies<Byte> FB = VectorSpecies.of(byte.class, F.vectorShape());
    private static final VectorSpecies<Short> FS = VectorSpecies.of(short.class, F.vectorShape());
    private static final VectorShuffle<Byte> SPREAD4;
    private static final VectorShuffle<Short> SPREAD2;
    private static final VectorMask<Byte> FB_LANES = FB.indexInRange(0, F.length());
    private static final VectorMask<Short> FS_LANES = FS.indexInRange(0, F.length());
    static {
        int[] bytes = new int[FB.length()], shorts = new int[FS.length()];
        for (int i = 0; i < bytes.length; i++) bytes[i] = i / 4;
        for (int i = 0; i < shorts.length; i++) shorts[i] = i / 2;
        SPREAD4 = VectorShuffle.fromArray(FB, bytes, 0);
        SPREAD2 = VectorShuffle.fromArray(FS, shorts, 0);
    }

    /** Per-thread tile of query handles and scales, so a projection allocates nothing. */
    private static final class Tile {
        final float[] scales = new float[Int8.TILE];
        final float[][] inputs = new float[Int8.TILE][], outputs = new float[Int8.TILE][];
        int[] coarse = new int[0], fine = new int[0];
        float[] halfQuery = new float[0];
    }
    private static final ThreadLocal<Tile> TILES = ThreadLocal.withInitial(Tile::new);

    /** Whether the vector shape suits these bodies (4, 8 or 16 float lanes). */
    static boolean supported() { return F.length() == 4 || F.length() == 8 || F.length() == 16; }

    private final float[][] matrix;
    private final int rows, columns;
    // The native batch layout, made on the first batch: tiles of 32 output rows, each holding its dimensions as
    // contiguous runs of up to 32 weights (one per row of the tile). Queries alone never pay for it.
    private volatile float[] packed;

    PanamaASHProjection(float[][] matrix) {
        Objects.requireNonNull(matrix, "matrix");
        if (matrix.length == 0 || Objects.requireNonNull(matrix[0], "matrix row").length == 0)
            throw new IllegalArgumentException("ASH projection dimensions must be positive");
        int columns = matrix[0].length;
        for (float[] row : matrix) {
            if (Objects.requireNonNull(row, "matrix row").length != columns)
                throw new IllegalArgumentException("Ragged ASH projection");
        }
        this.matrix = matrix;
        this.rows = matrix.length;
        this.columns = columns;
    }

    @Override
    public void project(float[] query, float[] out) {
        if (query.length != columns || out.length != rows || query == out)
            throw new IllegalArgumentException("ASH projection shape or aliasing mismatch");
        projectRows(matrix, query, out);
    }

    @Override
    public void projectBatch(float[][] inputs, float[][] outputs, int count) {
        VectorUtilSupport.AshProjection.validateBatch(inputs, outputs, count);
        if (count == 0) return;
        if (inputs[0].length != columns || outputs[0].length != rows)
            throw new IllegalArgumentException("ASH projection batch shape mismatch");
        if (count == 1) {
            // A query batched alone projects exactly as project() does, so its scores match a directly projected query
            if (inputs[0] == outputs[0]) throw new IllegalArgumentException("ASH projection shape or aliasing mismatch");
            projectRows(matrix, inputs[0], outputs[0]);
            return;
        }
        if (count > QUERY_BATCH) {
            // Encoding sends large batches in one short burst on every core. A body called once per batch can finish
            // the burst in C1, which leaves every vector on the heap; calls of up to QUERY_BATCH vectors reach C2 in
            // time.
            float[] ready = packed();
            Tile tile = TILES.get();
            try {
                for (int first = 0; first < count; first += QUERY_BATCH) {
                    int n = Math.min(QUERY_BATCH, count - first);
                    System.arraycopy(inputs, first, tile.inputs, 0, n);
                    System.arraycopy(outputs, first, tile.outputs, 0, n);
                    packedBatch(ready, rows, columns, tile.inputs, tile.outputs, n);
                }
            } finally {
                java.util.Arrays.fill(tile.inputs, null);
                java.util.Arrays.fill(tile.outputs, null);
            }
            return;
        }
        packedBatch(packed(), rows, columns, inputs, outputs, count);
    }

    /** Batches up to this size, as queued queries form them, use the native batch kernel. */
    static final int QUERY_BATCH = 8;

    private float[] packed() {
        float[] ready = packed;
        if (ready == null) {
            synchronized (this) {
                ready = packed;
                if (ready == null) {
                    ready = new float[Math.multiplyExact(rows, columns)];
                    for (int first = 0; first < rows; first += 32) {
                        int width = Math.min(32, rows - first), base = first * columns;
                        for (int c = 0; c < columns; c++)
                            for (int lane = 0; lane < width; lane++) ready[base + c * width + lane] = matrix[first + lane][c];
                    }
                    packed = ready;
                }
            }
        }
        return ready;
    }

    /** The packed batch copy, once made; the caller owns the row-major matrix. */
    @Override
    public long ramBytesUsed() {
        float[] ready = packed;
        return RamUsageEstimator.shallowSizeOf(this) + (ready == null ? 0 : RamUsageEstimator.sizeOf(ready));
    }

    @Override
    public boolean supportsBatchProjection() { return true; }

    /** Query-only views; the FP32 matrix still serves encoding and training. */
    @Override
    public VectorUtilSupport.AshProjection forQueryScoring(ASHQueryProjection precision) {
        switch (precision) {
            case AUTO:
            case FP32: return this;
            case INT8: return new Int8(matrix);
            case FP16: return new Fp16(matrix);
            default: throw new AssertionError(precision);
        }
    }

    /**
     * {@code out[r] = <matrix[r], x>} for every row. Eight rows per pass share each load of {@code x}; every row keeps
     * {@code ashDotRow}'s single accumulator and scalar tail.
     */
    static void projectRows(float[][] matrix, float[] x, float[] out) {
        final int len = x.length;
        final int upper = F.loopBound(len);
        int r = 0;
        for (; r + 8 <= matrix.length; r += 8) {
            final float[] m0 = matrix[r], m1 = matrix[r + 1], m2 = matrix[r + 2], m3 = matrix[r + 3];
            final float[] m4 = matrix[r + 4], m5 = matrix[r + 5], m6 = matrix[r + 6], m7 = matrix[r + 7];
            FloatVector a0 = FloatVector.zero(F), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
            int i = 0;
            for (; i < upper; i += F.length()) {
                FloatVector vx = FloatVector.fromArray(F, x, i);
                a0 = FloatVector.fromArray(F, m0, i).lanewise(VectorOperators.FMA, vx, a0);
                a1 = FloatVector.fromArray(F, m1, i).lanewise(VectorOperators.FMA, vx, a1);
                a2 = FloatVector.fromArray(F, m2, i).lanewise(VectorOperators.FMA, vx, a2);
                a3 = FloatVector.fromArray(F, m3, i).lanewise(VectorOperators.FMA, vx, a3);
                a4 = FloatVector.fromArray(F, m4, i).lanewise(VectorOperators.FMA, vx, a4);
                a5 = FloatVector.fromArray(F, m5, i).lanewise(VectorOperators.FMA, vx, a5);
                a6 = FloatVector.fromArray(F, m6, i).lanewise(VectorOperators.FMA, vx, a6);
                a7 = FloatVector.fromArray(F, m7, i).lanewise(VectorOperators.FMA, vx, a7);
            }
            float s0 = a0.reduceLanes(VectorOperators.ADD), s1 = a1.reduceLanes(VectorOperators.ADD);
            float s2 = a2.reduceLanes(VectorOperators.ADD), s3 = a3.reduceLanes(VectorOperators.ADD);
            float s4 = a4.reduceLanes(VectorOperators.ADD), s5 = a5.reduceLanes(VectorOperators.ADD);
            float s6 = a6.reduceLanes(VectorOperators.ADD), s7 = a7.reduceLanes(VectorOperators.ADD);
            for (; i < len; i++) {
                float v = x[i];
                s0 += m0[i] * v; s1 += m1[i] * v; s2 += m2[i] * v; s3 += m3[i] * v;
                s4 += m4[i] * v; s5 += m5[i] * v; s6 += m6[i] * v; s7 += m7[i] * v;
            }
            out[r] = s0; out[r + 1] = s1; out[r + 2] = s2; out[r + 3] = s3;
            out[r + 4] = s4; out[r + 5] = s5; out[r + 6] = s6; out[r + 7] = s7;
        }
        for (; r < matrix.length; r++) {
            final float[] m = matrix[r];
            FloatVector sum = FloatVector.zero(F);
            int i = 0;
            for (; i < upper; i += F.length()) {
                sum = FloatVector.fromArray(F, m, i).lanewise(VectorOperators.FMA, FloatVector.fromArray(F, x, i), sum);
            }
            float acc = sum.reduceLanes(VectorOperators.ADD);
            for (; i < len; i++) acc += m[i] * x[i];
            out[r] = acc;
        }
    }

    /**
     * The native batch kernel: for each tile of 32 output rows and each vector of its rows, four queries at a time,
     * every dimension's weights for those rows load once and multiply-add into one accumulator per query against
     * that query's broadcast value. Each lane is one output, summed dimension by dimension with no lane reduction,
     * so a batch's outputs do not depend on how the JIT compiles a reduction. All arithmetic is FP32.
     */
    static void packedBatch(float[] packed, int rows, int columns, float[][] xs, float[][] outs, int count) {
        final int lanes = F.length();
        for (int first = 0; first < rows; first += 32) {
            final int width = Math.min(32, rows - first), base = first * columns;
            for (int offset = 0; offset < width; offset += 2 * lanes) {
                final int n0 = Math.min(lanes, width - offset);
                final int n1 = width - offset > lanes ? Math.min(lanes, width - offset - lanes) : 0;
                final var m0 = F.indexInRange(0, n0);
                final var m1 = F.indexInRange(0, n1);
                final int row = first + offset;
                int b = 0;
                for (; b + 4 <= count; b += 4) {
                    final float[] x0 = xs[b], x1 = xs[b + 1], x2 = xs[b + 2], x3 = xs[b + 3];
                    // Accumulators start from the first dimension's products, never from zero(): a vector that
                    // reaches intoArray through a loop phi with zero() can lose its exact type and box.
                    final int w0 = base + offset;
                    var a00 = FloatVector.fromArray(F, packed, w0, m0);
                    var a10 = FloatVector.fromArray(F, packed, w0 + lanes, m1);
                    var p0 = FloatVector.broadcast(F, x0[0]);
                    var p1 = FloatVector.broadcast(F, x1[0]);
                    var p2 = FloatVector.broadcast(F, x2[0]);
                    var p3 = FloatVector.broadcast(F, x3[0]);
                    var s0 = a00.lanewise(VectorOperators.MUL, p0);
                    var s1 = a00.lanewise(VectorOperators.MUL, p1);
                    var s2 = a00.lanewise(VectorOperators.MUL, p2);
                    var s3 = a00.lanewise(VectorOperators.MUL, p3);
                    var t0 = a10.lanewise(VectorOperators.MUL, p0);
                    var t1 = a10.lanewise(VectorOperators.MUL, p1);
                    var t2 = a10.lanewise(VectorOperators.MUL, p2);
                    var t3 = a10.lanewise(VectorOperators.MUL, p3);
                    for (int c = 1, w = w0 + width; c < columns; c++, w += width) {
                        var a0 = FloatVector.fromArray(F, packed, w, m0);
                        var a1 = FloatVector.fromArray(F, packed, w + lanes, m1);
                        var q0 = FloatVector.broadcast(F, x0[c]);
                        var q1 = FloatVector.broadcast(F, x1[c]);
                        var q2 = FloatVector.broadcast(F, x2[c]);
                        var q3 = FloatVector.broadcast(F, x3[c]);
                        s0 = a0.lanewise(VectorOperators.FMA, q0, s0);
                        s1 = a0.lanewise(VectorOperators.FMA, q1, s1);
                        s2 = a0.lanewise(VectorOperators.FMA, q2, s2);
                        s3 = a0.lanewise(VectorOperators.FMA, q3, s3);
                        t0 = a1.lanewise(VectorOperators.FMA, q0, t0);
                        t1 = a1.lanewise(VectorOperators.FMA, q1, t1);
                        t2 = a1.lanewise(VectorOperators.FMA, q2, t2);
                        t3 = a1.lanewise(VectorOperators.FMA, q3, t3);
                    }
                    s0.intoArray(outs[b], row, m0);
                    s1.intoArray(outs[b + 1], row, m0);
                    s2.intoArray(outs[b + 2], row, m0);
                    s3.intoArray(outs[b + 3], row, m0);
                    if (n1 != 0) {
                        t0.intoArray(outs[b], row + lanes, m1);
                        t1.intoArray(outs[b + 1], row + lanes, m1);
                        t2.intoArray(outs[b + 2], row + lanes, m1);
                        t3.intoArray(outs[b + 3], row + lanes, m1);
                    }
                }
                for (; b < count; b++) {
                    final float[] x = xs[b];
                    final int w0 = base + offset;
                    var p = FloatVector.broadcast(F, x[0]);
                    var s0 = FloatVector.fromArray(F, packed, w0, m0).lanewise(VectorOperators.MUL, p);
                    var t0 = FloatVector.fromArray(F, packed, w0 + lanes, m1).lanewise(VectorOperators.MUL, p);
                    for (int c = 1, w = w0 + width; c < columns; c++, w += width) {
                        var q = FloatVector.broadcast(F, x[c]);
                        s0 = FloatVector.fromArray(F, packed, w, m0).lanewise(VectorOperators.FMA, q, s0);
                        t0 = FloatVector.fromArray(F, packed, w + lanes, m1).lanewise(VectorOperators.FMA, q, t0);
                    }
                    s0.intoArray(outs[b], row, m0);
                    if (n1 != 0) t0.intoArray(outs[b], row + lanes, m1);
                }
            }
        }
    }

    /**
     * INT8 query view with the native kernel's numbers: each row quantized to INT8 with scale {@code maxabs / 127};
     * each query as two INT8 digits with one scale, {@code q / scale ~ coarse + fine / 254}; both dot products exact
     * in int32 (up to 65535 columns); outputs combined as
     * {@code float(254 * coarseDot + fineDot) * (1 / 254f) * queryScale * rowScale}. A query with a non-finite value
     * projects to NaN.
     */
    static final class Int8 implements VectorUtilSupport.AshProjection {
        static final int MAX_COLUMNS = 65535;
        private static final float INV254 = 1.0f / 254.0f;
        static final int TILE = 8;

        private final byte[] weights; // rows * padded, row-major, zero padded, plus slack for the last load
        private final float[] scales;
        private final int rows, columns, padded;

        Int8(float[][] matrix) {
            rows = matrix.length;
            columns = matrix[0].length;
            if (columns > MAX_COLUMNS) throw new IllegalArgumentException("INT8 accumulator dimension exceeds 65535");
            padded = (columns + I.length() - 1) / I.length() * I.length();
            weights = new byte[Math.addExact(Math.multiplyExact(rows, padded), B.length())];
            scales = new float[rows];
            for (int r = 0; r < rows; r++) {
                float[] values = matrix[r];
                float maxabs = 0;
                for (float v : values) {
                    if (!Float.isFinite(v)) throw new IllegalArgumentException("Nonfinite INT8 matrix operand");
                    maxabs = Math.max(maxabs, Math.abs(v));
                }
                float scale = maxabs == 0 ? 1.0f : maxabs / 127.0f;
                if (scale == 0) throw new IllegalArgumentException("INT8 operand scale underflows FP32; rescale input");
                scales[r] = scale;
                for (int c = 0; c < columns; c++)
                    weights[r * padded + c] = (byte) Math.max(-127, Math.min(127, (int) Math.rint(values[c] / scale)));
            }
        }

        @Override
        public void project(float[] query, float[] out) {
            Objects.requireNonNull(query, "query");
            Objects.requireNonNull(out, "out");
            if (query.length != columns || out.length != rows || query == out)
                throw new IllegalArgumentException("ASH projection shape or aliasing mismatch");
            Tile tile = tile();
            tile.inputs[0] = query;
            tile.outputs[0] = out;
            try {
                run(tile, 1);
            } finally {
                tile.inputs[0] = null;
                tile.outputs[0] = null;
            }
        }

        @Override
        public void projectBatch(float[][] queries, float[][] outputs, int count) {
            VectorUtilSupport.AshProjection.validateBatch(queries, outputs, count);
            if (count == 0) return;
            if (queries[0].length != columns || outputs[0].length != rows)
                throw new IllegalArgumentException("ASH INT8 projection batch shape mismatch");
            Tile tile = tile();
            try {
                for (int first = 0; first < count; first += TILE) {
                    int size = Math.min(TILE, count - first);
                    for (int i = 0; i < size; i++) {
                        tile.inputs[i] = queries[first + i];
                        tile.outputs[i] = outputs[first + i];
                    }
                    run(tile, size);
                }
            } finally {
                java.util.Arrays.fill(tile.inputs, null);
                java.util.Arrays.fill(tile.outputs, null);
            }
        }

        private void run(Tile tile, int count) {
            if (!projectTile(weights, scales, rows, columns, padded, tile.inputs, tile.outputs, count, tile.coarse, tile.fine, tile.scales))
                throw new IllegalArgumentException("ASH INT8 projection: INT8 operand scale underflows FP32; rescale input");
        }

        private Tile tile() {
            Tile tile = TILES.get();
            if (tile.coarse.length < TILE * padded) {
                tile.coarse = new int[TILE * padded];
                tile.fine = new int[TILE * padded];
            }
            return tile;
        }

        /**
         * One tile of up to {@link #TILE} queries in one call, as the native kernel does it: each query quantized to
         * two INT8 digits with one scale (NaN for a non-finite query), both exact integer dot products with every
         * row, then the combine. Returns false, before any output is written, when a nonzero query's scale underflows.
         */
        private static boolean projectTile(byte[] w, float[] rowScales, int rows, int columns, int padded, float[][] inputs,
                                       float[][] outs, int count, int[] coarse, int[] fine,
                                       float[] queryScales) {
            for (int j = 0; j < count; j++) {
                final float[] input = inputs[j];
                final int at = j * padded;
                float maxabs = 0;
                boolean finite = true;
                for (int i = 0; i < columns; i++) {
                    if (!Float.isFinite(input[i])) { finite = false; break; }
                    maxabs = Math.max(maxabs, Math.abs(input[i]));
                }
                float scale = !finite ? Float.NaN : maxabs == 0 ? 1.0f : maxabs / 127.0f;
                if (scale == 0) return false;
                if (finite) {
                    for (int i = 0; i < columns; i++) {
                        int q = Math.max(-127, Math.min(127, (int) Math.rint(input[i] / scale)));
                        float remainder = input[i] / scale - (float) q;
                        coarse[at + i] = q;
                        fine[at + i] = Math.max(-127, Math.min(127, (int) Math.rint(remainder * 254.0f)));
                    }
                    java.util.Arrays.fill(coarse, at + columns, at + padded, 0);
                    java.util.Arrays.fill(fine, at + columns, at + padded, 0);
                }
                queryScales[j] = scale;
            }
            // One query at a time, with 32-bit multiplies: the Vector API has no 8-bit dot-product operation, so this
            // cannot use VNNI (vpdpbusd) as native does. That would need the proposed Vector API dot-product
            // operation (JDK-8377386).
            for (int j = 0; j < count; j++) {
                final int q = j * padded;
                final float queryScale = queryScales[j];
                final float[] out = outs[j];
                for (int r = 0; r < rows; r++) {
                    final int row = r * padded;
                    IntVector c = IntVector.zero(I), f = IntVector.zero(I);
                    for (int i = 0; i < padded; i += I.length()) {
                        var wv = ByteVector.fromArray(FB, w, row + i, FB_LANES).rearrange(SPREAD4).reinterpretAsInts()
                                .lanewise(VectorOperators.LSHL, 24).lanewise(VectorOperators.ASHR, 24);
                        c = c.lanewise(VectorOperators.ADD, wv.lanewise(VectorOperators.MUL, IntVector.fromArray(I, coarse, q + i)));
                        f = f.lanewise(VectorOperators.ADD, wv.lanewise(VectorOperators.MUL, IntVector.fromArray(I, fine, q + i)));
                    }
                    int cs = c.reduceLanes(VectorOperators.ADD), fs = f.reduceLanes(VectorOperators.ADD);
                    out[r] = (float) ((long) cs * 254 + fs) * INV254 * queryScale * rowScales[r];
                }
            }
            return true;
        }

        @Override
        public boolean supportsBatchProjection() { return true; }

        @Override
        public String operandPrecision() { return "int8"; }

        @Override
        public long ramBytesUsed() {
            return RamUsageEstimator.shallowSizeOf(this) + RamUsageEstimator.sizeOf(weights) + RamUsageEstimator.sizeOf(scales);
        }
    }

    /**
     * FP16 query view, with the native view's numbers: weights and each query rounded to half precision (round to
     * nearest even), products accumulated in FP32. A query value outside the FP16 range is rejected.
     */
    static final class Fp16 implements VectorUtilSupport.AshProjection {
        private final short[] weights; // rows * columns, row-major
        private final int rows, columns;

        Fp16(float[][] matrix) {
            rows = matrix.length;
            columns = matrix[0].length;
            weights = new short[Math.multiplyExact(rows, columns)];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    float v = matrix[r][c];
                    short h = Float.floatToFloat16(v);
                    if (!Float.isFinite(v) || !Float.isFinite(Float.float16ToFloat(h)))
                        throw new IllegalArgumentException("ASH FP16 projection: a matrix value is outside the FP16 range");
                    weights[r * columns + c] = h;
                }
            }
        }

        @Override
        public void project(float[] query, float[] out) {
            Objects.requireNonNull(query, "query");
            Objects.requireNonNull(out, "out");
            if (query.length != columns || out.length != rows || query == out)
                throw new IllegalArgumentException("ASH projection shape or aliasing mismatch");
            Tile tile = tile();
            tile.inputs[0] = query;
            tile.outputs[0] = out;
            try {
                run(tile.inputs, tile.outputs, 1, tile.halfQuery);
            } finally {
                tile.inputs[0] = null;
                tile.outputs[0] = null;
            }
        }

        @Override
        public void projectBatch(float[][] queries, float[][] outputs, int count) {
            VectorUtilSupport.AshProjection.validateBatch(queries, outputs, count);
            if (count == 0) return;
            if (queries[0].length != columns || outputs[0].length != rows)
                throw new IllegalArgumentException("ASH FP16 projection batch shape mismatch");
            run(queries, outputs, count, tile().halfQuery);
        }

        private Tile tile() {
            Tile tile = TILES.get();
            if (tile.halfQuery.length < 4 * columns) tile.halfQuery = new float[4 * columns];
            return tile;
        }

        private void run(float[][] queries, float[][] outputs, int count, float[] halfQuery) {
            if (!halfRows(weights, rows, columns, queries, outputs, count, halfQuery))
                throw new IllegalArgumentException("ASH FP16 projection: a query value is outside the FP16 range");
        }

        /**
         * Every query, rounded to half precision, against every half-precision row: weights decoded in registers
         * (sign, exponent and mantissa moved into FP32 position; subnormals scaled from the mantissa; infinities
         * kept), each decoded vector shared by four queries, one accumulator per output and a scalar tail. Returns false, before writing a group's outputs, when one of its
         * query values overflows FP16. {@code x} holds four rounded queries.
         */
        private static boolean halfRows(short[] w, int rows, int columns, float[][] xs, float[][] outs, int count,
                                        float[] x) {
            final int upper = F.loopBound(columns);
            for (int first = 0; first < count; first += 4) {
                final int n = Math.min(4, count - first);
                for (int k = 0; k < n; k++) {
                    final float[] input = xs[first + k];
                    for (int i = 0; i < columns; i++) {
                        float v = Float.float16ToFloat(Float.floatToFloat16(input[i]));
                        if (Float.isInfinite(v) && !Float.isInfinite(input[i])) return false;
                        x[k * columns + i] = v;
                    }
                }
                // A group of fewer than four reuses query 0's values for the missing ones and stores only its own outputs
                final int x1 = n > 1 ? columns : 0, x2 = n > 2 ? 2 * columns : 0, x3 = n > 3 ? 3 * columns : 0;
                for (int r = 0; r < rows; r++) {
                    final int row = r * columns;
                    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    int i = 0;
                    if (upper > 0) {
                        // Accumulators start from the first chunk's products, never from zero()
                        FloatVector a0, a1, a2, a3;
                        {
                            var h = ShortVector.fromArray(FS, w, row + 0, FS_LANES).rearrange(SPREAD2).reinterpretAsInts().lanewise(VectorOperators.AND, 0xFFFF);
                            var sign = h.lanewise(VectorOperators.AND, 0x8000).lanewise(VectorOperators.LSHL, 16);
                            var exponent = h.lanewise(VectorOperators.LSHR, 10).lanewise(VectorOperators.AND, 31);
                            var mantissa = h.lanewise(VectorOperators.AND, 0x3FF);
                            var bits = exponent.lanewise(VectorOperators.ADD, 112).lanewise(VectorOperators.LSHL, 23)
                                    .lanewise(VectorOperators.OR, mantissa.lanewise(VectorOperators.LSHL, 13)).lanewise(VectorOperators.OR, sign);
                            bits = bits.blend(mantissa.lanewise(VectorOperators.LSHL, 13).lanewise(VectorOperators.OR, 0x7F800000).lanewise(VectorOperators.OR, sign),
                                    exponent.compare(VectorOperators.EQ, 31));
                            var subnormal = mantissa.lanewise(VectorOperators.OR, 0x4B000000).reinterpretAsFloats()
                                    .lanewise(VectorOperators.SUB, 0x1p23f).lanewise(VectorOperators.MUL, 0x1p-24f)
                                    .reinterpretAsInts().lanewise(VectorOperators.OR, sign);
                            var wv = bits.blend(subnormal, exponent.compare(VectorOperators.EQ, 0)).reinterpretAsFloats();
                            a0 = wv.lanewise(VectorOperators.MUL, FloatVector.fromArray(F, x, 0));
                            a1 = wv.lanewise(VectorOperators.MUL, FloatVector.fromArray(F, x, x1));
                            a2 = wv.lanewise(VectorOperators.MUL, FloatVector.fromArray(F, x, x2));
                            a3 = wv.lanewise(VectorOperators.MUL, FloatVector.fromArray(F, x, x3));
                        }
                        for (i = F.length(); i < upper; i += F.length()) {
                            var h = ShortVector.fromArray(FS, w, row + i, FS_LANES).rearrange(SPREAD2).reinterpretAsInts().lanewise(VectorOperators.AND, 0xFFFF);
                            var sign = h.lanewise(VectorOperators.AND, 0x8000).lanewise(VectorOperators.LSHL, 16);
                            var exponent = h.lanewise(VectorOperators.LSHR, 10).lanewise(VectorOperators.AND, 31);
                            var mantissa = h.lanewise(VectorOperators.AND, 0x3FF);
                            var bits = exponent.lanewise(VectorOperators.ADD, 112).lanewise(VectorOperators.LSHL, 23)
                                    .lanewise(VectorOperators.OR, mantissa.lanewise(VectorOperators.LSHL, 13)).lanewise(VectorOperators.OR, sign);
                            bits = bits.blend(mantissa.lanewise(VectorOperators.LSHL, 13).lanewise(VectorOperators.OR, 0x7F800000).lanewise(VectorOperators.OR, sign),
                                    exponent.compare(VectorOperators.EQ, 31));
                            var subnormal = mantissa.lanewise(VectorOperators.OR, 0x4B000000).reinterpretAsFloats()
                                    .lanewise(VectorOperators.SUB, 0x1p23f).lanewise(VectorOperators.MUL, 0x1p-24f)
                                    .reinterpretAsInts().lanewise(VectorOperators.OR, sign);
                            var wv = bits.blend(subnormal, exponent.compare(VectorOperators.EQ, 0)).reinterpretAsFloats();
                            a0 = wv.lanewise(VectorOperators.FMA, FloatVector.fromArray(F, x, i), a0);
                            a1 = wv.lanewise(VectorOperators.FMA, FloatVector.fromArray(F, x, x1 + i), a1);
                            a2 = wv.lanewise(VectorOperators.FMA, FloatVector.fromArray(F, x, x2 + i), a2);
                            a3 = wv.lanewise(VectorOperators.FMA, FloatVector.fromArray(F, x, x3 + i), a3);
                        }
                        s0 = a0.reduceLanes(VectorOperators.ADD);
                        s1 = a1.reduceLanes(VectorOperators.ADD);
                        s2 = a2.reduceLanes(VectorOperators.ADD);
                        s3 = a3.reduceLanes(VectorOperators.ADD);
                    }
                    for (; i < columns; i++) {
                        float wi = Float.float16ToFloat(w[row + i]);
                        s0 += wi * x[i];
                        s1 += wi * x[x1 + i];
                        s2 += wi * x[x2 + i];
                        s3 += wi * x[x3 + i];
                    }
                    outs[first][r] = s0;
                    if (n > 1) outs[first + 1][r] = s1;
                    if (n > 2) outs[first + 2][r] = s2;
                    if (n > 3) outs[first + 3][r] = s3;
                }
            }
            return true;
        }

        @Override
        public boolean supportsBatchProjection() { return true; }

        @Override
        public String operandPrecision() { return "fp16"; }

        @Override
        public long ramBytesUsed() { return RamUsageEstimator.shallowSizeOf(this) + RamUsageEstimator.sizeOf(weights); }
    }
}
