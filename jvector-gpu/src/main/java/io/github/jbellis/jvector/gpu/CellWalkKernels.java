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
 * GPU kernels of the compaction's cell walk ({@link TornadoCompactionAccelerator}). One launch covers a chunk of a
 * target's rows and the (cell, query) pairs whose cells lie in it: {@link #score} computes every pair's member scores
 * into a candidate buffer at offsets the host assigned, and {@link #topk} merges each query's candidates into its
 * running top k.
 */
final class CellWalkKernels {

    /** Threads per work-group: eight warps. */
    static final int GROUP = 256;
    static final int WARPS = GROUP / 32;
    /** Largest dimension: the query is staged in local memory. */
    static final int MAX_DIM = 4096;
    /** Largest top k. */
    static final int MAX_K = 64;

    // params
    static final int P_DIM = 0;
    static final int P_ROW0 = 1;
    static final int P_SIMILARITY = 2;
    static final int P_K = 3;
    static final int SIM_DOT = 0;
    static final int SIM_COSINE = 1;
    static final int SIM_EUCLIDEAN = 2;

    private CellWalkKernels() {
    }

    /** One work-group per query: its squared norm, a tree sum over the group. */
    public static void norms(KernelContext ctx, FloatArray queries, IntArray params, FloatArray queryNorms) {
        int query = ctx.groupIdx;
        int lid = ctx.localIdx;
        int dim = params.get(P_DIM);
        float[] part = ctx.allocateFloatLocalArray(GROUP);
        float sum = 0.0f;
        for (int d = lid; d < dim; d += GROUP) {
            float v = queries.get(query * dim + d);
            sum += v * v;
        }
        part[lid] = sum;
        ctx.localBarrier();
        for (int stride = GROUP / 2; stride > 0; stride /= 2) {
            if (lid < stride) {
                part[lid] += part[lid + stride];
            }
            ctx.localBarrier();
        }
        if (lid == 0) {
            queryNorms.set(query, part[0]);
        }
    }

    /**
     * One work-group per pair: the query in local memory, one warp per member row (lanes stride the dimensions, so
     * reads are coalesced), JVector's score written at {@code offsets[pair] + member - first}.
     */
    public static void score(KernelContext ctx, FloatArray queries, FloatArray queryNorms, FloatArray rows,
                             IntArray pairFirst, IntArray pairEnd, IntArray pairQuery, IntArray pairOffset, IntArray params,
                             FloatArray candScore, IntArray candOrdinal) {
        int p = ctx.groupIdx;
        int lid = ctx.localIdx;
        int dim = params.get(P_DIM);
        int row0 = params.get(P_ROW0);
        int similarity = params.get(P_SIMILARITY);
        float[] q = ctx.allocateFloatLocalArray(MAX_DIM);
        float[] dots = ctx.allocateFloatLocalArray(GROUP);
        float[] norms = ctx.allocateFloatLocalArray(GROUP);
        int query = pairQuery.get(p);
        for (int d = lid; d < dim; d += GROUP) {
            q[d] = queries.get(query * dim + d);
        }
        ctx.localBarrier();
        int first = pairFirst.get(p);
        int end = pairEnd.get(p);
        int offset = pairOffset.get(p);
        float queryNorm = queryNorms.get(query);
        int warp = lid / 32;
        int lane = lid - warp * 32;
        for (int base = first; base < end; base += WARPS) {
            int member = base + warp;
            float dot = 0.0f;
            float norm = 0.0f;
            if (member < end) {
                int row = (member - row0) * dim;
                for (int d = lane; d < dim; d += 32) {
                    float v = rows.get(row + d);
                    dot += v * q[d];
                    norm += v * v;
                }
            }
            dots[lid] = dot;
            norms[lid] = norm;
            ctx.localBarrier();
            if (lane == 0 && member < end) {
                float sumDot = 0.0f;
                float sumNorm = 0.0f;
                for (int l = 0; l < 32; l++) {
                    sumDot += dots[lid + l];
                    sumNorm += norms[lid + l];
                }
                float s;
                if (similarity == SIM_DOT) {
                    s = (1.0f + sumDot) / 2.0f;
                } else if (similarity == SIM_EUCLIDEAN) {
                    s = 1.0f / (1.0f + TornadoMath.max(0.0f, queryNorm + sumNorm - 2.0f * sumDot));
                } else {
                    s = (1.0f + sumDot / TornadoMath.sqrt(TornadoMath.max(1e-12f, queryNorm * sumNorm))) / 2.0f;
                }
                candScore.set(offset + member - first, s);
                candOrdinal.set(offset + member - first, member);
            }
            ctx.localBarrier();
        }
    }

    /**
     * One work-group per query: its running top k ({@code topCount} entries) plus this launch's candidates
     * ({@code queryCount} from {@code queryOffset}) go through k rounds of a parallel arg-max, ties broken by the larger
     * ordinal (the order of JVector's heap keys), and the best k become the new running top k.
     */
    public static void topk(KernelContext ctx, FloatArray candScore, IntArray candOrdinal, IntArray queryOffset,
                            IntArray queryCount, IntArray params, FloatArray topScore, IntArray topOrdinal, IntArray topCount) {
        int query = ctx.groupIdx;
        int lid = ctx.localIdx;
        int k = params.get(P_K);
        float[] oldScore = ctx.allocateFloatLocalArray(MAX_K);
        int[] oldOrdinal = ctx.allocateIntLocalArray(MAX_K);
        float[] newScore = ctx.allocateFloatLocalArray(MAX_K);
        int[] newOrdinal = ctx.allocateIntLocalArray(MAX_K);
        float[] bestScore = ctx.allocateFloatLocalArray(GROUP);
        int[] bestOrdinal = ctx.allocateIntLocalArray(GROUP);
        int[] bestAt = ctx.allocateIntLocalArray(GROUP);
        int count = queryCount.get(query);
        int offset = queryOffset.get(query);
        int old = topCount.get(query);
        if (lid < old) {
            oldScore[lid] = topScore.get(query * k + lid);
            oldOrdinal[lid] = topOrdinal.get(query * k + lid);
        }
        ctx.localBarrier();
        int total = count + old;
        int rounds = TornadoMath.min(k, total);
        if (count > 0) {
            for (int t = 0; t < rounds; t++) {
                float bs = -2.0f;
                int bo = -1;
                int ba = -1;
                for (int e = lid; e < total; e += GROUP) {
                    float s;
                    int o;
                    if (e < count) {
                        s = candScore.get(offset + e);
                        o = candOrdinal.get(offset + e);
                    } else {
                        s = oldScore[e - count];
                        o = oldOrdinal[e - count];
                    }
                    if (s > bs || (s == bs && o > bo)) {
                        bs = s;
                        bo = o;
                        ba = e;
                    }
                }
                bestScore[lid] = bs;
                bestOrdinal[lid] = bo;
                bestAt[lid] = ba;
                ctx.localBarrier();
                for (int stride = GROUP / 2; stride > 0; stride /= 2) {
                    if (lid < stride) {
                        float os = bestScore[lid + stride];
                        int oo = bestOrdinal[lid + stride];
                        if (os > bestScore[lid] || (os == bestScore[lid] && oo > bestOrdinal[lid])) {
                            bestScore[lid] = os;
                            bestOrdinal[lid] = oo;
                            bestAt[lid] = bestAt[lid + stride];
                        }
                    }
                    ctx.localBarrier();
                }
                if (lid == 0) {
                    newScore[t] = bestScore[0];
                    newOrdinal[t] = bestOrdinal[0];
                    int at = bestAt[0];
                    if (at < count) {
                        candScore.set(offset + at, -2.0f);
                    } else {
                        oldScore[at - count] = -2.0f;
                    }
                }
                ctx.localBarrier();
            }
            if (lid < rounds) {
                topScore.set(query * k + lid, newScore[lid]);
                topOrdinal.set(query * k + lid, newOrdinal[lid]);
            }
            if (lid == 0) {
                topCount.set(query, rounds);
            }
        }
    }
}
