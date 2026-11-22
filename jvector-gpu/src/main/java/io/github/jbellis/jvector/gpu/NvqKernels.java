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
 * NVQ encoding (8 bits a dimension), one work-group per (row, sub-vector): JVector's
 * {@code NVQuantization.QuantizedSubVector.quantizeTo} with the scalar formulas of {@code DefaultVectorUtilSupport}
 * ({@code nvqUniformLoss}, {@code nvqLoss}, {@code nvqQuantize8bit}). The growth-rate search walks the same candidate
 * values in the same order; only the order of each loss sum differs (a tree reduction), so a near-tie can pick a
 * neighbouring growth rate.
 */
final class NvqKernels {
    /** Indices into {@code params}. */
    static final int P_ROWS = 0, P_DIM = 1, P_SUBS = 2, P_LEARN = 3;
    /** Threads per (row, sub-vector). */
    static final int THREADS = 256;

    private NvqKernels() {
    }

    /**
     * Encodes rows (with the global mean subtracted here): per (row, sub-vector), min, max and growth rate into
     * {@code stats} (three floats), and the 8-bit codes into {@code codes} at the row's dimensions.
     */
    static void encode(KernelContext ctx, FloatArray rows, FloatArray mean, IntArray sizes, IntArray offsets, IntArray params,
                       FloatArray stats, ByteArray codes) {
        int subs = params.get(P_SUBS);
        int group = ctx.groupIdx;
        int row = group / subs;
        if (row >= params.get(P_ROWS)) {
            return;
        }
        int sub = group - row * subs;
        int lid = ctx.localIdx;
        int dim = params.get(P_DIM);
        int size = sizes.get(sub);
        int off = offsets.get(sub);
        int base = row * dim + off;
        float[] a = ctx.allocateFloatLocalArray(THREADS);
        float[] b = ctx.allocateFloatLocalArray(THREADS);

        // min and max
        float mn = Float.MAX_VALUE;
        float mx = -Float.MAX_VALUE;
        for (int d = lid; d < size; d += THREADS) {
            float v = rows.get(base + d) - mean.get(off + d);
            mn = TornadoMath.min(mn, v);
            mx = TornadoMath.max(mx, v);
        }
        a[lid] = mn;
        b[lid] = mx;
        ctx.localBarrier();
        for (int stride = THREADS / 2; stride > 0; stride /= 2) {
            if (lid < stride) {
                a[lid] = TornadoMath.min(a[lid], a[lid + stride]);
                b[lid] = TornadoMath.max(b[lid], b[lid + stride]);
            }
            ctx.localBarrier();
        }
        float minValue = a[0];
        float maxValue = b[0];
        ctx.localBarrier();

        // the uniform quantizer's loss, the baseline of the search
        float delta = maxValue - minValue;
        float partial = 0.0f;
        for (int d = lid; d < size; d += THREADS) {
            float v = rows.get(base + d) - mean.get(off + d);
            float r = (v - minValue) / delta;
            r = round(255.0f * r) / 255.0f;
            r = r * delta + minValue;
            float diff = v - r;
            partial += diff * diff;
        }
        a[lid] = partial;
        ctx.localBarrier();
        for (int stride = THREADS / 2; stride > 0; stride /= 2) {
            if (lid < stride) {
                a[lid] += a[lid + stride];
            }
            ctx.localBarrier();
        }
        float baseline = a[0];
        ctx.localBarrier();

        // the growth-rate search: coarse steps of 1 from 1e-6 below 20, then steps of 0.1 around the best
        float growthRate = 1e-2f;
        if (params.get(P_LEARN) != 0) {
            float best = Float.MIN_VALUE;
            float coarse = 1e-2f;
            float fine = 1e-2f;
            for (int phase = 0; phase < 2; phase++) {
                float gr = phase == 0 ? 1e-6f : coarse - 1.0f;
                float end = phase == 0 ? 20.0f : coarse + 1.0f;
                float step = phase == 0 ? 1.0f : 0.1f;
                if (phase == 1) {
                    fine = coarse;
                }
                while (gr < end) {
                    float scaledGrowth = gr / delta;
                    float bias = logistic(minValue, scaledGrowth);
                    float scale = (logistic(maxValue, scaledGrowth) - bias) / 255.0f;
                    float inverseGrowth = 1.0f / scaledGrowth;
                    float sum = 0.0f;
                    for (int d = lid; d < size; d += THREADS) {
                        float v = rows.get(base + d) - mean.get(off + d);
                        float q = round((logistic(v, scaledGrowth) - bias) * (1.0f / scale));
                        float r = logit(q * scale + bias, inverseGrowth);
                        float diff = v - r;
                        sum += diff * diff;
                    }
                    a[lid] = sum;
                    ctx.localBarrier();
                    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
                        if (lid < stride) {
                            a[lid] += a[lid + stride];
                        }
                        ctx.localBarrier();
                    }
                    float loss = baseline / a[0];
                    ctx.localBarrier();
                    if (loss > best) {
                        best = loss;
                        if (phase == 0) {
                            coarse = gr;
                        } else {
                            fine = gr;
                        }
                    }
                    gr += step;
                }
            }
            growthRate = fine;
        }

        // the codes
        float scaledGrowth = growthRate / delta;
        float bias = logistic(minValue, scaledGrowth);
        float scale = (logistic(maxValue, scaledGrowth) - bias) / 255.0f;
        for (int d = lid; d < size; d += THREADS) {
            float v = rows.get(base + d) - mean.get(off + d);
            int q = (int) round((logistic(v, scaledGrowth) - bias) * (1.0f / scale));
            codes.set(base + d, (byte) q);
        }
        if (lid == 0) {
            int s = 3 * group;
            stats.set(s, minValue);
            stats.set(s + 1, maxValue);
            stats.set(s + 2, growthRate);
        }
    }

    /** Java's {@code Math.round(float)}: half up. */
    private static float round(float x) {
        return TornadoMath.floor(x + 0.5f);
    }

    /** {@code logisticFunctionNQT(value, alpha, 0)}: 2^temp approximated as (1 + (temp - p) / 2) * 2^p. */
    private static float logistic(float value, float alpha) {
        float temp = value * alpha;
        int p = (int) round(temp + 0.5f);
        float y = ((temp - p) * 0.5f + 1.0f) * powerOfTwo(p);
        return y / (y + 1.0f);
    }

    /** {@code logitNQT(value, inverseAlpha, 0)}: log2 z approximated as mantissa in [1, 2) plus exponent minus one. */
    private static float logit(float value, float inverseAlpha) {
        float z = TornadoMath.abs(value / (1.0f - value));
        float m = z;
        int e = 0;
        if (z == 0.0f) {
            m = 1.0f;
            e = -127;
        } else {
            while (m >= 2.0f) {
                m *= 0.5f;
                e++;
            }
            while (m < 1.0f) {
                m *= 2.0f;
                e--;
            }
        }
        return (m + (e - 1)) * inverseAlpha;
    }

    /** 2^p exactly, for the small integer powers the logistic needs. */
    private static float powerOfTwo(int p) {
        // one loop with a signed step rather than an if/else with a loop in each arm, which TornadoVM 7.2's PTX
        // backend (gone since 8.0) compiled wrongly
        float factor = p >= 0 ? 2.0f : 0.5f;
        int steps = p >= 0 ? p : -p;
        float r = 1.0f;
        for (int i = 0; i < steps; i++) {
            r *= factor;
        }
        return r;
    }
}
