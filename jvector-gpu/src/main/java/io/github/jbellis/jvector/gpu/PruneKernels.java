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
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * GPU kernels for JVector's diversity pruning of candidate lists, in batches of nodes: the Gram matrix of every
 * node's candidates ({@link #gramMmaPacked}), then the alpha rule over it ({@link #prune}).
 */
final class PruneKernels {

    /** Candidate slots per node: lists are padded to this. The Gram kernel's shape depends on it. */
    static final int SLOTS = 96;
    /** Warps per node in {@link #gramMmaPacked}: one per 16-row block of the Gram matrix. */
    static final int MMA_WARPS = SLOTS / 16;
    /** Row shards of {@link PackedVectors}: one TornadoVM array holds at most {@code Integer.MAX_VALUE} elements. */
    static final int SHARDS = 4;

    /** Indices into {@code params}. */
    static final int P_NODES = 0, P_DIM = 1, P_MAX_DEGREE = 2, P_ROWS_PER_SHARD = 3, P_EUCLIDEAN = 4;

    private PruneKernels() {
    }

    /**
     * JVector's diversity rule ({@code VamanaDiversityProvider}) for one node per work-group of {@link #SLOTS}
     * threads. Candidates are sorted by decreasing score; for alpha = 1.0, 1.2, ... up to {@code alpha}, a candidate
     * is kept unless some kept candidate is more similar to it than {@code score * alpha}, until
     * {@code maxDegree} are kept. Similarities come from the Gram matrix in JVector's units: {@code (1 + g) / 2} for
     * COSINE (unit vectors) and DOT_PRODUCT, {@code 1 / (1 + |a|^2 + |b|^2 - 2g)} for EUCLIDEAN.
     */
    public static void prune(KernelContext ctx, FloatArray gram, FloatArray candScores, FloatArray candNorms, IntArray counts, IntArray params, IntArray keep, float alpha) {
        int b = ctx.groupIdx;
        int lid = ctx.localIdx;
        int maxDegree = params.get(P_MAX_DEGREE);
        int euclidean = params.get(P_EUCLIDEAN);
        int size = counts.get(b);
        int[] kept = ctx.allocateIntLocalArray(SLOTS);
        int[] violated = ctx.allocateIntLocalArray(1);
        kept[lid] = 0;
        int selected = 0;
        float currentAlpha = 1.0f;
        int row = b * SLOTS * SLOTS;
        while (currentAlpha <= alpha + 1E-6f && selected < maxDegree) {
            for (int i = 0; i < size && selected < maxDegree; i++) {
                if (lid == 0) {
                    violated[0] = 0;
                }
                ctx.localBarrier();
                if (kept[i] == 0) {
                    float threshold = candScores.get(b * SLOTS + i) * currentAlpha;
                    if (lid < size && kept[lid] == 1) {
                        float g = gram.get(row + i * SLOTS + lid);
                        float sim;
                        if (euclidean == 1) {
                            float d2 = candNorms.get(b * SLOTS + i) + candNorms.get(b * SLOTS + lid) - 2.0f * g;
                            sim = 1.0f / (1.0f + TornadoMath.max(d2, 0.0f));
                        } else {
                            sim = (1.0f + g) / 2.0f;
                        }
                        if (sim > threshold) {
                            violated[0] = 1;
                        }
                    }
                }
                ctx.localBarrier();
                if (kept[i] == 0 && violated[0] == 0) {
                    selected++;
                    ctx.localBarrier();
                    if (lid == 0) {
                        kept[i] = 1;
                    }
                }
                ctx.localBarrier();
            }
            currentAlpha += 0.2f;
        }
        keep.set(b * SLOTS + lid, kept[lid]);
    }

    /**
     * The 96 x 96 Gram matrix of every node's candidates, with tensor cores ({@code mma.sync m16n8k16},
     * FP16 inputs, FP32 accumulation). Work-group {@code b} (6 warps) handles node {@code b}: each 16-dimension
     * slice of its candidates is staged once in shared memory as both the A operand (rows) and the B operand (the
     * same rows, transposed), and warp {@code w} accumulates rows {@code 16w..16w+15} against all 12 column
     * blocks.
     * <p>
     * Vectors are packed FP16 ({@link PackedVectors}): row {@code id} lives in shard {@code id / rowsPerShard} as
     * {@code dim / 2} words of two halves (low = even dimension), so the A operand is staged with one plain load per
     * word. Generated by {@code codegen/gen_gram_mma_packed.py}.
     */
    public static void gramMmaPacked(KernelContext ctx, IntArray s0, IntArray s1, IntArray s2, IntArray s3, IntArray candIds, IntArray params, FloatArray gram) {
        int b = ctx.groupIdx;
        int lid = ctx.localIdx;
        int warp = lid / 32;
        int dim = params.get(P_DIM);
        int words = dim / 2;
        int rowsPerShard = params.get(P_ROWS_PER_SHARD);
        int[] aForm = ctx.allocateIntLocalArray(768);
        int[] bForm = ctx.allocateIntLocalArray(768);
        int[] ids = ctx.allocateIntLocalArray(96);
        int[] shard = ctx.allocateIntLocalArray(96);
        int[] base = ctx.allocateIntLocalArray(96);
        if (lid < 96) {
            int id = candIds.get(b * 96 + lid);
            ids[lid] = id;
            int s = 0;
            int local = 0;
            if (id >= 0) {
                s = id / rowsPerShard;
                local = (id - s * rowsPerShard) * words;
            }
            shard[lid] = s;
            base[lid] = local;
        }
        ctx.localBarrier();
        float[] f0 = ctx.mmaFragment(0.0f);
        float[] f1 = ctx.mmaFragment(0.0f);
        float[] f2 = ctx.mmaFragment(0.0f);
        float[] f3 = ctx.mmaFragment(0.0f);
        float[] f4 = ctx.mmaFragment(0.0f);
        float[] f5 = ctx.mmaFragment(0.0f);
        float[] f6 = ctx.mmaFragment(0.0f);
        float[] f7 = ctx.mmaFragment(0.0f);
        float[] f8 = ctx.mmaFragment(0.0f);
        float[] f9 = ctx.mmaFragment(0.0f);
        float[] f10 = ctx.mmaFragment(0.0f);
        float[] f11 = ctx.mmaFragment(0.0f);
        int aOffset = warp * 16 * 8 * 4;
        for (int k0 = 0; k0 < dim; k0 += 16) {
            int w0 = k0 / 2;
            for (int e = lid; e < 96 * 8; e += 192) {
                int r = e / 8;
                int p = e % 8;
                int v = 0;
                if (ids[r] >= 0) {
                    int at = base[r] + w0 + p;
                    if (shard[r] == 0) {
                        v = s0.get(at);
                    } else if (shard[r] == 1) {
                        v = s1.get(at);
                    } else if (shard[r] == 2) {
                        v = s2.get(at);
                    } else {
                        v = s3.get(at);
                    }
                }
                aForm[r * 8 + p] = v;
            }
            for (int e = lid; e < 12 * 64; e += 192) {
                int c = e / 64;
                int kr = (e % 64) / 4;
                int jp = e % 4;
                int j0 = c * 8 + 2 * jp;
                int j1 = j0 + 1;
                int word = w0 + kr / 2;
                int shift = (kr % 2) * 16;
                int v0 = 0;
                if (ids[j0] >= 0) {
                    int at = base[j0] + word;
                    if (shard[j0] == 0) {
                        v0 = s0.get(at);
                    } else if (shard[j0] == 1) {
                        v0 = s1.get(at);
                    } else if (shard[j0] == 2) {
                        v0 = s2.get(at);
                    } else {
                        v0 = s3.get(at);
                    }
                }
                int v1 = 0;
                if (ids[j1] >= 0) {
                    int at = base[j1] + word;
                    if (shard[j1] == 0) {
                        v1 = s0.get(at);
                    } else if (shard[j1] == 1) {
                        v1 = s1.get(at);
                    } else if (shard[j1] == 2) {
                        v1 = s2.get(at);
                    } else {
                        v1 = s3.get(at);
                    }
                }
                bForm[e] = ((v0 >>> shift) & 0xFFFF) | (((v1 >>> shift) & 0xFFFF) << 16);
            }
            ctx.localBarrier();
            HalfFloat[] fa = ctx.mmaLoadA(aForm, 16, aOffset);
            f0 = ctx.mma(fa, ctx.mmaLoadB(bForm, 16, 0), f0, MMAShape.M16N8K16);
            f1 = ctx.mma(fa, ctx.mmaLoadB(bForm, 16, 256), f1, MMAShape.M16N8K16);
            f2 = ctx.mma(fa, ctx.mmaLoadB(bForm, 16, 512), f2, MMAShape.M16N8K16);
            f3 = ctx.mma(fa, ctx.mmaLoadB(bForm, 16, 768), f3, MMAShape.M16N8K16);
            f4 = ctx.mma(fa, ctx.mmaLoadB(bForm, 16, 1024), f4, MMAShape.M16N8K16);
            f5 = ctx.mma(fa, ctx.mmaLoadB(bForm, 16, 1280), f5, MMAShape.M16N8K16);
            f6 = ctx.mma(fa, ctx.mmaLoadB(bForm, 16, 1536), f6, MMAShape.M16N8K16);
            f7 = ctx.mma(fa, ctx.mmaLoadB(bForm, 16, 1792), f7, MMAShape.M16N8K16);
            f8 = ctx.mma(fa, ctx.mmaLoadB(bForm, 16, 2048), f8, MMAShape.M16N8K16);
            f9 = ctx.mma(fa, ctx.mmaLoadB(bForm, 16, 2304), f9, MMAShape.M16N8K16);
            f10 = ctx.mma(fa, ctx.mmaLoadB(bForm, 16, 2560), f10, MMAShape.M16N8K16);
            f11 = ctx.mma(fa, ctx.mmaLoadB(bForm, 16, 2816), f11, MMAShape.M16N8K16);
            ctx.localBarrier();
        }
        int row = b * 96 + warp * 16;
        ctx.mmaStore(f0, gram, row, 0, 96);
        ctx.mmaStore(f1, gram, row, 8, 96);
        ctx.mmaStore(f2, gram, row, 16, 96);
        ctx.mmaStore(f3, gram, row, 24, 96);
        ctx.mmaStore(f4, gram, row, 32, 96);
        ctx.mmaStore(f5, gram, row, 40, 96);
        ctx.mmaStore(f6, gram, row, 48, 96);
        ctx.mmaStore(f7, gram, row, 56, 96);
        ctx.mmaStore(f8, gram, row, 64, 96);
        ctx.mmaStore(f9, gram, row, 72, 96);
        ctx.mmaStore(f10, gram, row, 80, 96);
        ctx.mmaStore(f11, gram, row, 88, 96);
    }
}
