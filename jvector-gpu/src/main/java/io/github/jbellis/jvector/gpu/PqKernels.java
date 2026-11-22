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

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * GPU kernels for product quantization, all subspaces at once: k-means++ seeding and Lloyd iterations over a
 * training set (as {@code KMeansPlusPlusClusterer} does per subspace), and encoding. Codebooks are laid out as one
 * {@code K x dim} array: subspace {@code m}, centroid {@code c} starts at {@code K * offset[m] + c * size[m]}.
 * <p>
 * One task graph runs every step; {@code params[P_STEP]} says which kernel does work and the others return at once.
 */
final class PqKernels {

    /** Indices into {@code params}. */
    static final int P_ROWS = 0, P_DIM = 1, P_M = 2, P_K = 3, P_STEP = 4, P_PICK = 5, P_SEED = 6, P_MAX_SIZE = 7;
    /** Steps. */
    static final int SEED_DISTANCES = 0, SEED_PICK = 1, SEED_SET = 2, ASSIGN = 3, UPDATE = 4, NONE = 5;
    /** Largest subspace the kernels hold in registers (JVector's usual M = dim / 8 gives 8). */
    static final int MAX_SUBSPACE = 32;
    /** Threads per subspace in {@link #pick}. */
    static final int PICK_THREADS = 256;

    private PqKernels() {
    }

    /**
     * k-means++ seeding: the squared distance of every training row to its nearest chosen centroid, per subspace,
     * after centroid {@code pick} was chosen. Before the first pick every row weighs 1 (a uniform choice).
     */
    static void seedDistances(KernelContext ctx, FloatArray rows, FloatArray codebooks, IntArray sizes, IntArray offsets, IntArray params, FloatArray d2) {
        if (params.get(P_STEP) != SEED_DISTANCES) {
            return;
        }
        int t = params.get(P_ROWS);
        int gid = ctx.globalIdx;
        if (gid >= params.get(P_M) * t) {
            return;
        }
        int m = gid / t;
        int i = gid - m * t;
        int pick = params.get(P_PICK);
        int dim = params.get(P_DIM);
        int k = params.get(P_K);
        int off = offsets.get(m);
        int size = sizes.get(m);
        float dd = 0;
        for (int j = 0; j < size; j++) {
            float x = rows.get(i * dim + off + j) - codebooks.get(k * off + pick * size + j);
            dd += x * x;
        }
        d2.set(gid, pick == 0 ? dd : TornadoMath.min(d2.get(gid), dd));
    }

    /**
     * k-means++ seeding: the next centroid of every subspace, a row drawn with probability proportional to
     * {@code d2}. Weighted sampling as the row with the largest {@code log(u) / weight} (Efraimidis-Spirakis), with
     * {@code u} from a hash of (seed, pick, subspace, row). One work-group of {@link #PICK_THREADS} per subspace.
     */
    static void pick(KernelContext ctx, FloatArray d2, IntArray params, IntArray picked) {
        if (params.get(P_STEP) != SEED_PICK) {
            return;
        }
        int m = ctx.groupIdx;
        int lid = ctx.localIdx;
        int t = params.get(P_ROWS);
        int pick = params.get(P_PICK);
        int seed = params.get(P_SEED);
        float[] keys = ctx.allocateFloatLocalArray(PICK_THREADS);
        int[] rows = ctx.allocateIntLocalArray(PICK_THREADS);
        float bestKey = -Float.MAX_VALUE;
        int bestRow = 0;
        for (int i = lid; i < t; i += PICK_THREADS) {
            float w = pick == 0 ? 1.0f : d2.get(m * t + i);
            if (w > 0.0f) {
                int x = seed ^ (pick * 0x9E3779B1) ^ (m * 0x85EBCA77) ^ (i * 0xC2B2AE3D);
                x ^= x >>> 16;
                x *= 0x7FEB352D;
                x ^= x >>> 15;
                x *= 0x846CA68B;
                x ^= x >>> 16;
                float u = ((x >>> 8) + 0.5f) / 16777216.0f;
                float key = TornadoMath.log(u) / w;
                if (key > bestKey) {
                    bestKey = key;
                    bestRow = i;
                }
            }
        }
        keys[lid] = bestKey;
        rows[lid] = bestRow;
        ctx.localBarrier();
        for (int stride = PICK_THREADS / 2; stride > 0; stride /= 2) {
            if (lid < stride && keys[lid + stride] > keys[lid]) {
                keys[lid] = keys[lid + stride];
                rows[lid] = rows[lid + stride];
            }
            ctx.localBarrier();
        }
        if (lid == 0) {
            picked.set(m, rows[0]);
        }
    }

    /** k-means++ seeding: copies each subspace's picked row into centroid {@code pick}. */
    static void setCentroid(KernelContext ctx, FloatArray rows, FloatArray codebooks, IntArray sizes, IntArray offsets, IntArray params, IntArray picked) {
        if (params.get(P_STEP) != SEED_SET) {
            return;
        }
        int maxSize = params.get(P_MAX_SIZE);
        int gid = ctx.globalIdx;
        if (gid >= params.get(P_M) * maxSize) {
            return;
        }
        int m = gid / maxSize;
        int j = gid - m * maxSize;
        int size = sizes.get(m);
        if (j < size) {
            int off = offsets.get(m);
            int k = params.get(P_K);
            int dim = params.get(P_DIM);
            codebooks.set(k * off + params.get(P_PICK) * size + j, rows.get(picked.get(m) * dim + off + j));
        }
    }

    /** Lloyd: the nearest centroid of every training row, per subspace. */
    static void assign(KernelContext ctx, FloatArray rows, FloatArray codebooks, IntArray sizes, IntArray offsets, IntArray params, IntArray assignment) {
        if (params.get(P_STEP) != ASSIGN) {
            return;
        }
        int t = params.get(P_ROWS);
        int gid = ctx.globalIdx;
        if (gid >= params.get(P_M) * t) {
            return;
        }
        int m = gid / t;
        int i = gid - m * t;
        int dim = params.get(P_DIM);
        int k = params.get(P_K);
        int off = offsets.get(m);
        int size = sizes.get(m);
        float[] x = new float[MAX_SUBSPACE];
        for (int j = 0; j < size; j++) {
            x[j] = rows.get(i * dim + off + j);
        }
        int best = 0;
        float bestD = Float.MAX_VALUE;
        int cb = k * off;
        for (int c = 0; c < k; c++) {
            float d = 0;
            for (int j = 0; j < size; j++) {
                float e = x[j] - codebooks.get(cb + j);
                d += e * e;
            }
            cb += size;
            if (d < bestD) {
                bestD = d;
                best = c;
            }
        }
        assignment.set(gid, best);
    }

    /** Lloyd: every centroid becomes the mean of its rows (unchanged if it has none). One thread per centroid. */
    static void update(KernelContext ctx, FloatArray rows, FloatArray codebooks, IntArray sizes, IntArray offsets, IntArray params, IntArray assignment) {
        if (params.get(P_STEP) != UPDATE) {
            return;
        }
        int k = params.get(P_K);
        int gid = ctx.globalIdx;
        if (gid >= params.get(P_M) * k) {
            return;
        }
        int m = gid / k;
        int c = gid - m * k;
        int t = params.get(P_ROWS);
        int dim = params.get(P_DIM);
        int off = offsets.get(m);
        int size = sizes.get(m);
        int base = k * off + c * size;
        int count = 0;
        for (int i = 0; i < t; i++) {
            if (assignment.get(m * t + i) == c) {
                if (count == 0) {
                    for (int j = 0; j < size; j++) {
                        codebooks.set(base + j, 0.0f);
                    }
                }
                for (int j = 0; j < size; j++) {
                    codebooks.set(base + j, codebooks.get(base + j) + rows.get(i * dim + off + j));
                }
                count++;
            }
        }
        if (count > 0) {
            for (int j = 0; j < size; j++) {
                codebooks.set(base + j, codebooks.get(base + j) / count);
            }
        }
    }

    /**
     * Encoding: the nearest centroid of every row in every subspace, after subtracting {@code centroid}. One thread
     * per (subspace, row), subspace-major; codes are written row-major ({@code row * M + subspace}).
     */
    static void encode(KernelContext ctx, FloatArray rows, FloatArray centroid, FloatArray codebooks, IntArray sizes, IntArray offsets, IntArray params, ByteArray codes) {
        int n = params.get(P_ROWS);
        int mCount = params.get(P_M);
        int gid = ctx.globalIdx;
        if (gid >= n * mCount) {
            return;
        }
        // subspace-major: the threads of a warp read the same codebook entry (one broadcast load)
        int m = gid / n;
        int i = gid - m * n;
        int dim = params.get(P_DIM);
        int k = params.get(P_K);
        int off = offsets.get(m);
        int size = sizes.get(m);
        float[] x = new float[MAX_SUBSPACE];
        for (int j = 0; j < size; j++) {
            x[j] = rows.get(i * dim + off + j) - centroid.get(off + j);
        }
        int best = 0;
        float bestD = Float.MAX_VALUE;
        int cb = k * off;
        for (int c = 0; c < k; c++) {
            float d = 0;
            for (int j = 0; j < size; j++) {
                float e = x[j] - codebooks.get(cb + j);
                d += e * e;
            }
            cb += size;
            if (d < bestD) {
                bestD = d;
                best = c;
            }
        }
        codes.set(i * mCount + m, (byte) best);
    }
}
