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

package io.github.jbellis.jvector.gpu;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.stream.IntStream;

import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Vectors as packed FP16 for {@link PruneKernels#gramMmaPacked}, half the size of FP32 (10M x 1024 is 20.5 GB).
 * <ul>
 * <li>Each row is padded with zeros to a multiple of 16 dimensions (the tensor-core K step), which changes no dot
 * product or distance.</li>
 * <li>A row is {@code paddedDim / 2} words of two halves, the low half holding the even dimension.</li>
 * <li>Rows are split into at most {@link PruneKernels#SHARDS} {@link IntArray}s, since a TornadoVM array holds at
 * most {@code Integer.MAX_VALUE} elements.</li>
 * </ul>
 * For EUCLIDEAN, the squared norms of the FP16 rows are kept too, so that distances derived from the Gram matrix
 * are consistent with it.
 */
final class PackedVectors {

    final IntArray[] shards;
    final int rowsPerShard;
    final int paddedDim;
    /** Squared norms of the FP16 rows, or null. */
    final float[] squaredNorms;

    private PackedVectors(IntArray[] shards, int rowsPerShard, int paddedDim, float[] squaredNorms) {
        this.shards = shards;
        this.rowsPerShard = rowsPerShard;
        this.paddedDim = paddedDim;
        this.squaredNorms = squaredNorms;
    }

    static int paddedDimension(int dim) {
        return (dim + 15) / 16 * 16;
    }

    /** Device bytes of {@code n} packed rows. */
    static long bytes(long n, int dim) {
        return n * paddedDimension(dim) * 2L;
    }

    /** Whether {@code n} rows fit in the shards. */
    static boolean fits(long n, int dim) {
        return n * (paddedDimension(dim) / 2) <= (long) PruneKernels.SHARDS * (Integer.MAX_VALUE - 15);
    }

    /**
     * Packs {@code n x dim} row-major float32 vectors held in host memory.
     *
     * @param normalise    scale every row to unit length while packing
     * @param squaredNorms keep the squared norm of every packed row
     */
    static PackedVectors of(MemorySegment rows, int n, int dim, boolean normalise, boolean squaredNorms) {
        if (!fits(n, dim)) {
            throw new IllegalArgumentException(n + " x " + dim + " does not fit in " + PruneKernels.SHARDS + " shards");
        }
        int paddedDim = paddedDimension(dim);
        int words = paddedDim / 2;
        long maxRows = (Integer.MAX_VALUE - 15) / words;
        int shardsNeeded = (int) ((n + maxRows - 1) / maxRows);
        int rowsPerShard = (n + shardsNeeded - 1) / Math.max(shardsNeeded, 1);
        float[] norms = squaredNorms ? new float[n] : null;
        IntArray[] shards = new IntArray[PruneKernels.SHARDS];
        for (int s = 0; s < shards.length; s++) {
            int first = s * rowsPerShard;
            int count = Math.max(0, Math.min(rowsPerShard, n - first));
            // unused shards still need a buffer for the kernel signature
            IntArray shard = new IntArray(Math.max(1, count * words));
            MemorySegment out = shard.getSegment();
            IntStream.range(0, count).parallel().forEach(r -> {
                long from = (long) (first + r) * dim * Float.BYTES;
                float scale = 1.0f;
                if (normalise) {
                    double sq = 0;
                    for (int d = 0; d < dim; d++) {
                        float x = rows.get(ValueLayout.JAVA_FLOAT_UNALIGNED, from + (long) d * Float.BYTES);
                        sq += x * x;
                    }
                    scale = (float) (1.0 / Math.sqrt(sq));
                }
                float norm = 0;
                long to = (long) r * words * Integer.BYTES;
                for (int p = 0; p < words; p++) {
                    short lo = half(rows, from, 2 * p, dim, scale);
                    short hi = half(rows, from, 2 * p + 1, dim, scale);
                    if (norms != null) {
                        float a = Float.float16ToFloat(lo);
                        float b = Float.float16ToFloat(hi);
                        norm += a * a + b * b;
                    }
                    out.set(ValueLayout.JAVA_INT_UNALIGNED, to + (long) p * Integer.BYTES, (lo & 0xFFFF) | (hi << 16));
                }
                if (norms != null) {
                    norms[first + r] = norm;
                }
            });
            shards[s] = shard;
        }
        return new PackedVectors(shards, rowsPerShard, paddedDim, norms);
    }

    /** Dimension {@code d} of the row at {@code from} as FP16, zero beyond {@code dim}. */
    private static short half(MemorySegment rows, long from, int d, int dim, float scale) {
        if (d >= dim) {
            return 0;
        }
        return Float.floatToFloat16(rows.get(ValueLayout.JAVA_FLOAT_UNALIGNED, from + (long) d * Float.BYTES) * scale);
    }

    /** Dimension {@code d} of row {@code row}, as packed. */
    float get(int row, int d) {
        int s = row / rowsPerShard;
        int word = shards[s].get((row - s * rowsPerShard) * (paddedDim / 2) + d / 2);
        return Float.float16ToFloat((short) (d % 2 == 0 ? word : word >>> 16));
    }
}
