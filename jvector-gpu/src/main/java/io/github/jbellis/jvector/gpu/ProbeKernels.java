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
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * GPU kernels of the compaction's probe step ({@link TornadoCompactionAccelerator#probe}): every query of a block
 * against every center (level-1 node), then each query's best {@code P} centers, sorted.
 */
final class ProbeKernels {

    /** Rows and columns of a tile of {@link #centerScores}; 256 threads compute 4 x 4 outputs each. */
    static final int TILE = 64;
    static final int TILE_K = 16;
    static final int GROUP = 256;
    /** Largest number of probes per query (a power of two: the bitonic sort's size). */
    static final int MAX_PROBES = 1024;

    // params
    static final int P_DIM = 0;
    static final int P_QUERIES = 1;
    static final int P_CENTERS = 2;
    static final int P_SIMILARITY = 3;
    static final int P_PROBES = 4;
    /** Columns of the score matrix: the center count rounded up to a whole tile. */
    static final int P_STRIDE = 5;
    /** Two-level probes: coarse lists per query, and candidates kept per query. */
    static final int P_COARSE_PROBES = 6;
    static final int P_LIST_CAP = 7;
    static final int MAX_COARSE_PROBES = 128;

    private ProbeKernels() {
    }

    /**
     * Scores of a block of queries against all centers, {@code scores[q * stride + c]}, ordered like JVector's
     * similarity: EUCLIDEAN {@code 2 q.c - |c|^2} (the squared distance up to the query's own norm), COSINE
     * {@code q.c / |c|}, DOT_PRODUCT {@code q.c}. One work-group per 64 x 64 tile, staged through local memory in
     * steps of 16 dimensions; padded rows and columns score {@code -inf}.
     */
    public static void centerScores(KernelContext ctx, FloatArray queries, FloatArray centers, FloatArray centerNorms,
                                    IntArray params, FloatArray scores) {
        int dim = params.get(P_DIM);
        int nq = params.get(P_QUERIES);
        int nc = params.get(P_CENTERS);
        int similarity = params.get(P_SIMILARITY);
        int stride = params.get(P_STRIDE);
        int tilesC = stride / TILE;
        int group = ctx.groupIdx;
        int tq = group / tilesC;
        int tc = group - tq * tilesC;
        int lid = ctx.localIdx;
        int ty = lid / 16;
        int tx = lid - ty * 16;
        float[] a = ctx.allocateFloatLocalArray(TILE * TILE_K);
        float[] b = ctx.allocateFloatLocalArray(TILE * TILE_K);
        float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
        float c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;
        float c20 = 0.0f, c21 = 0.0f, c22 = 0.0f, c23 = 0.0f;
        float c30 = 0.0f, c31 = 0.0f, c32 = 0.0f, c33 = 0.0f;
        for (int k0 = 0; k0 < dim; k0 += TILE_K) {
            // 64 x 16 of each side, 4 values per thread
            for (int e = lid; e < TILE * TILE_K; e += GROUP) {
                int row = e / TILE_K;
                int kk = e - row * TILE_K;
                int q = tq * TILE + row;
                int c = tc * TILE + row;
                int d = k0 + kk;
                a[e] = q < nq && d < dim ? queries.get(q * dim + d) : 0.0f;
                b[e] = c < nc && d < dim ? centers.get(c * dim + d) : 0.0f;
            }
            ctx.localBarrier();
            for (int kk = 0; kk < TILE_K; kk++) {
                float a0 = a[(ty * 4) * TILE_K + kk];
                float a1 = a[(ty * 4 + 1) * TILE_K + kk];
                float a2 = a[(ty * 4 + 2) * TILE_K + kk];
                float a3 = a[(ty * 4 + 3) * TILE_K + kk];
                float b0 = b[(tx * 4) * TILE_K + kk];
                float b1 = b[(tx * 4 + 1) * TILE_K + kk];
                float b2 = b[(tx * 4 + 2) * TILE_K + kk];
                float b3 = b[(tx * 4 + 3) * TILE_K + kk];
                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
            }
            ctx.localBarrier();
        }
        int q0 = tq * TILE + ty * 4;
        int cb = tc * TILE + tx * 4;
        store(scores, centerNorms, similarity, nq, nc, stride, q0, cb, c00, c01, c02, c03);
        store(scores, centerNorms, similarity, nq, nc, stride, q0 + 1, cb, c10, c11, c12, c13);
        store(scores, centerNorms, similarity, nq, nc, stride, q0 + 2, cb, c20, c21, c22, c23);
        store(scores, centerNorms, similarity, nq, nc, stride, q0 + 3, cb, c30, c31, c32, c33);
    }

    private static void store(FloatArray scores, FloatArray centerNorms, int similarity, int nq, int nc, int stride, int q,
                              int c0, float d0, float d1, float d2, float d3) {
        if (q < nq) {
            scores.set(q * stride + c0, rank(centerNorms, similarity, nc, c0, d0));
            scores.set(q * stride + c0 + 1, rank(centerNorms, similarity, nc, c0 + 1, d1));
            scores.set(q * stride + c0 + 2, rank(centerNorms, similarity, nc, c0 + 2, d2));
            scores.set(q * stride + c0 + 3, rank(centerNorms, similarity, nc, c0 + 3, d3));
        }
    }

    private static float rank(FloatArray centerNorms, int similarity, int nc, int c, float dot) {
        if (c >= nc) {
            return -3.0e38f;
        }
        float norm2 = centerNorms.get(c);
        if (similarity == CellWalkKernels.SIM_EUCLIDEAN) {
            return 2.0f * dot - norm2;
        } else if (similarity == CellWalkKernels.SIM_COSINE) {
            return dot / TornadoMath.max(1e-12f, TornadoMath.sqrt(norm2));
        }
        return dot;
    }

    /**
     * Each query's best {@code P} centers, best first (ties to the lower position). The P-th best score is bracketed
     * by bisection between the smallest and largest score (40 counting passes: the bracket shrinks to the float
     * rounding of the scores); the centers above the bracket and, in position order, those inside it are gathered,
     * then sorted by a bitonic sort of {@value #MAX_PROBES} entries in local memory.
     */
    public static void selectTop(KernelContext ctx, FloatArray scores, IntArray params, IntArray probes) {
        int q = ctx.groupIdx;
        int lid = ctx.localIdx;
        int nc = params.get(P_CENTERS);
        int p = params.get(P_PROBES);
        int stride = params.get(P_STRIDE);
        float[] red = ctx.allocateFloatLocalArray(GROUP);
        float[] red2 = ctx.allocateFloatLocalArray(GROUP);
        int[] counts = ctx.allocateIntLocalArray(GROUP);
        float[] keys = ctx.allocateFloatLocalArray(MAX_PROBES);
        int[] pos = ctx.allocateIntLocalArray(MAX_PROBES);
        int base = q * stride;
        int want = TornadoMath.min(p, nc);
        // smallest and largest score
        float mn = 3.0e38f;
        float mx = -3.0e38f;
        for (int e = lid; e < nc; e += GROUP) {
            float v = scores.get(base + e);
            mn = TornadoMath.min(mn, v);
            mx = TornadoMath.max(mx, v);
        }
        red[lid] = mn;
        red2[lid] = mx;
        ctx.localBarrier();
        for (int s = GROUP / 2; s > 0; s /= 2) {
            if (lid < s) {
                red[lid] = TornadoMath.min(red[lid], red[lid + s]);
                red2[lid] = TornadoMath.max(red2[lid], red2[lid + s]);
            }
            ctx.localBarrier();
        }
        // invariant: count(score >= lo) >= want > count(score >= hi)
        float lo = red[0];
        float hi = red2[0] + TornadoMath.max(1.0f, TornadoMath.abs(red2[0]));
        ctx.localBarrier();
        for (int it = 0; it < 40; it++) {
            float mid = lo + (hi - lo) * 0.5f;
            int c = 0;
            for (int e = lid; e < nc; e += GROUP) {
                if (scores.get(base + e) >= mid) {
                    c++;
                }
            }
            counts[lid] = c;
            ctx.localBarrier();
            for (int s = GROUP / 2; s > 0; s /= 2) {
                if (lid < s) {
                    counts[lid] += counts[lid + s];
                }
                ctx.localBarrier();
            }
            int total = counts[0];
            ctx.localBarrier();
            if (total >= want) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // gather: scores >= hi (fewer than want), then those in [lo, hi) in position order; each thread takes a
        // contiguous range of positions
        int per = (nc + GROUP - 1) / GROUP;
        int from = lid * per;
        int to = TornadoMath.min(nc, from + per);
        int above = 0;
        int inside = 0;
        for (int e = from; e < to; e++) {
            float v = scores.get(base + e);
            if (v >= hi) {
                above++;
            } else if (v >= lo) {
                inside++;
            }
        }
        counts[lid] = above;
        ctx.localBarrier();
        if (lid == 0) {
            int run = 0;
            for (int i = 0; i < GROUP; i++) {
                int v = counts[i];
                counts[i] = run;
                run += v;
            }
            pos[0] = run;
        }
        ctx.localBarrier();
        int aboveAt = counts[lid];
        int aboveTotal = pos[0];
        ctx.localBarrier();
        counts[lid] = inside;
        ctx.localBarrier();
        if (lid == 0) {
            int run = 0;
            for (int i = 0; i < GROUP; i++) {
                int v = counts[i];
                counts[i] = run;
                run += v;
            }
        }
        ctx.localBarrier();
        int insideAt = aboveTotal + counts[lid];
        for (int i = lid; i < MAX_PROBES; i += GROUP) {
            keys[i] = -3.0e38f;
            pos[i] = 2147483647;
        }
        ctx.localBarrier();
        for (int e = from; e < to; e++) {
            float v = scores.get(base + e);
            if (v >= hi) {
                keys[aboveAt] = v;
                pos[aboveAt] = e;
                aboveAt++;
            } else if (v >= lo) {
                if (insideAt < want) {
                    keys[insideAt] = v;
                    pos[insideAt] = e;
                }
                insideAt++;
            }
        }
        ctx.localBarrier();
        // bitonic sort, best first: larger score, then smaller position
        for (int size = 2; size <= MAX_PROBES; size *= 2) {
            for (int step = size / 2; step > 0; step /= 2) {
                for (int i = lid; i < MAX_PROBES; i += GROUP) {
                    int j = i ^ step;
                    if (j > i) {
                        boolean descending = (i & size) == 0;
                        float ki = keys[i];
                        float kj = keys[j];
                        int pi = pos[i];
                        int pj = pos[j];
                        boolean iFirst = ki > kj || (ki == kj && pi < pj);
                        if (descending != iFirst) {
                            keys[i] = kj;
                            keys[j] = ki;
                            pos[i] = pj;
                            pos[j] = pi;
                        }
                    }
                }
                ctx.localBarrier();
            }
        }
        for (int i = lid; i < want; i += GROUP) {
            probes.set(q * p + i, pos[i]);
        }
    }

    /** Row {@code r}'s best column (ties to the lower one): each center's nearest coarse center. */
    public static void argmaxRows(KernelContext ctx, FloatArray scores, IntArray params, IntArray best) {
        int r = ctx.groupIdx;
        int lid = ctx.localIdx;
        int nc = params.get(P_CENTERS);
        int stride = params.get(P_STRIDE);
        float[] bs = ctx.allocateFloatLocalArray(GROUP);
        int[] bi = ctx.allocateIntLocalArray(GROUP);
        float b = -3.0e38f;
        int at = 2147483647;
        for (int e = lid; e < nc; e += GROUP) {
            float v = scores.get(r * stride + e);
            if (v > b) {
                b = v;
                at = e;
            }
        }
        bs[lid] = b;
        bi[lid] = at;
        ctx.localBarrier();
        for (int s = GROUP / 2; s > 0; s /= 2) {
            if (lid < s) {
                float o = bs[lid + s];
                int oi = bi[lid + s];
                if (o > bs[lid] || (o == bs[lid] && oi < bi[lid])) {
                    bs[lid] = o;
                    bi[lid] = oi;
                }
            }
            ctx.localBarrier();
        }
        if (lid == 0) {
            best.set(r, bi[0]);
        }
    }

    /**
     * Each query's candidates from its {@code M} best coarse lists: every member center scored exactly (one warp per
     * center, lanes over the dimensions), at most {@code listCap} of them, written at {@code q * listCap}.
     */
    public static void gatherScores(KernelContext ctx, FloatArray queries, FloatArray centers, FloatArray centerNorms,
                                    IntArray coarseProbes, IntArray listStart, IntArray listMembers, IntArray params,
                                    FloatArray candScore, IntArray candPos, IntArray candCount) {
        int q = ctx.groupIdx;
        int lid = ctx.localIdx;
        int dim = params.get(P_DIM);
        int similarity = params.get(P_SIMILARITY);
        int m = params.get(P_COARSE_PROBES);
        int cap = params.get(P_LIST_CAP);
        float[] qv = ctx.allocateFloatLocalArray(CellWalkKernels.MAX_DIM);
        int[] starts = ctx.allocateIntLocalArray(MAX_COARSE_PROBES + 1);
        int[] lists = ctx.allocateIntLocalArray(MAX_COARSE_PROBES);
        float[] part = ctx.allocateFloatLocalArray(GROUP);
        for (int d = lid; d < dim; d += GROUP) {
            qv[d] = queries.get(q * dim + d);
        }
        if (lid == 0) {
            int running = 0;
            for (int j = 0; j < m; j++) {
                int c = coarseProbes.get(q * m + j);
                lists[j] = c;
                starts[j] = running;
                running += listStart.get(c + 1) - listStart.get(c);
            }
            starts[m] = running;
        }
        ctx.localBarrier();
        int total = TornadoMath.min(starts[m], cap);
        int warp = lid / 32;
        int lane = lid - warp * 32;
        for (int base = 0; base < total; base += GROUP / 32) {
            int e = base + warp;
            float dot = 0.0f;
            int center = 0;
            if (e < total) {
                int j = 0;
                while (j + 1 < m && starts[j + 1] <= e) {
                    j++;
                }
                center = listMembers.get(listStart.get(lists[j]) + e - starts[j]);
                int row = center * dim;
                for (int d = lane; d < dim; d += 32) {
                    dot += centers.get(row + d) * qv[d];
                }
            }
            part[lid] = dot;
            ctx.localBarrier();
            if (lane == 0 && e < total) {
                float sum = 0.0f;
                for (int l = 0; l < 32; l++) {
                    sum += part[lid + l];
                }
                candScore.set(q * cap + e, rank(centerNorms, similarity, 2147483647, center, sum));
                candPos.set(q * cap + e, center);
            }
            ctx.localBarrier();
        }
        if (lid == 0) {
            candCount.set(q, total);
        }
    }

    /**
     * As {@link #selectTop}, over each query's own candidate list ({@code candCount[q]} scores from
     * {@code q * listCap}, their center positions in {@code candPos}); writes the best {@code P} positions and how many.
     */
    public static void selectListed(KernelContext ctx, FloatArray scores, IntArray candPos, IntArray candCount, IntArray params,
                                    IntArray probes, IntArray probeCount) {
        int q = ctx.groupIdx;
        int lid = ctx.localIdx;
        int nc = candCount.get(q);
        int p = params.get(P_PROBES);
        int stride = params.get(P_LIST_CAP);
        float[] red = ctx.allocateFloatLocalArray(GROUP);
        float[] red2 = ctx.allocateFloatLocalArray(GROUP);
        int[] counts = ctx.allocateIntLocalArray(GROUP);
        float[] keys = ctx.allocateFloatLocalArray(MAX_PROBES);
        int[] pos = ctx.allocateIntLocalArray(MAX_PROBES);
        int base = q * stride;
        int want = TornadoMath.min(p, nc);
        // smallest and largest score
        float mn = 3.0e38f;
        float mx = -3.0e38f;
        for (int e = lid; e < nc; e += GROUP) {
            float v = scores.get(base + e);
            mn = TornadoMath.min(mn, v);
            mx = TornadoMath.max(mx, v);
        }
        red[lid] = mn;
        red2[lid] = mx;
        ctx.localBarrier();
        for (int s = GROUP / 2; s > 0; s /= 2) {
            if (lid < s) {
                red[lid] = TornadoMath.min(red[lid], red[lid + s]);
                red2[lid] = TornadoMath.max(red2[lid], red2[lid + s]);
            }
            ctx.localBarrier();
        }
        // invariant: count(score >= lo) >= want > count(score >= hi)
        float lo = red[0];
        float hi = red2[0] + TornadoMath.max(1.0f, TornadoMath.abs(red2[0]));
        ctx.localBarrier();
        for (int it = 0; it < 40; it++) {
            float mid = lo + (hi - lo) * 0.5f;
            int c = 0;
            for (int e = lid; e < nc; e += GROUP) {
                if (scores.get(base + e) >= mid) {
                    c++;
                }
            }
            counts[lid] = c;
            ctx.localBarrier();
            for (int s = GROUP / 2; s > 0; s /= 2) {
                if (lid < s) {
                    counts[lid] += counts[lid + s];
                }
                ctx.localBarrier();
            }
            int total = counts[0];
            ctx.localBarrier();
            if (total >= want) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // gather: scores >= hi (fewer than want), then those in [lo, hi) in position order; each thread takes a
        // contiguous range of positions
        int per = (nc + GROUP - 1) / GROUP;
        int from = lid * per;
        int to = TornadoMath.min(nc, from + per);
        int above = 0;
        int inside = 0;
        for (int e = from; e < to; e++) {
            float v = scores.get(base + e);
            if (v >= hi) {
                above++;
            } else if (v >= lo) {
                inside++;
            }
        }
        counts[lid] = above;
        ctx.localBarrier();
        if (lid == 0) {
            int run = 0;
            for (int i = 0; i < GROUP; i++) {
                int v = counts[i];
                counts[i] = run;
                run += v;
            }
            pos[0] = run;
        }
        ctx.localBarrier();
        int aboveAt = counts[lid];
        int aboveTotal = pos[0];
        ctx.localBarrier();
        counts[lid] = inside;
        ctx.localBarrier();
        if (lid == 0) {
            int run = 0;
            for (int i = 0; i < GROUP; i++) {
                int v = counts[i];
                counts[i] = run;
                run += v;
            }
        }
        ctx.localBarrier();
        int insideAt = aboveTotal + counts[lid];
        for (int i = lid; i < MAX_PROBES; i += GROUP) {
            keys[i] = -3.0e38f;
            pos[i] = 2147483647;
        }
        ctx.localBarrier();
        for (int e = from; e < to; e++) {
            float v = scores.get(base + e);
            if (v >= hi) {
                keys[aboveAt] = v;
                pos[aboveAt] = e;
                aboveAt++;
            } else if (v >= lo) {
                if (insideAt < want) {
                    keys[insideAt] = v;
                    pos[insideAt] = e;
                }
                insideAt++;
            }
        }
        ctx.localBarrier();
        // bitonic sort, best first: larger score, then smaller position
        for (int size = 2; size <= MAX_PROBES; size *= 2) {
            for (int step = size / 2; step > 0; step /= 2) {
                for (int i = lid; i < MAX_PROBES; i += GROUP) {
                    int j = i ^ step;
                    if (j > i) {
                        boolean descending = (i & size) == 0;
                        float ki = keys[i];
                        float kj = keys[j];
                        int pi = pos[i];
                        int pj = pos[j];
                        boolean iFirst = ki > kj || (ki == kj && pi < pj);
                        if (descending != iFirst) {
                            keys[i] = kj;
                            keys[j] = ki;
                            pos[i] = pj;
                            pos[j] = pi;
                        }
                    }
                }
                ctx.localBarrier();
            }
        }
        for (int i = lid; i < want; i += GROUP) {
            probes.set(q * p + i, candPos.get(base + pos[i]));
        }
        if (lid == 0) {
            probeCount.set(q, want);
        }
    }
}
