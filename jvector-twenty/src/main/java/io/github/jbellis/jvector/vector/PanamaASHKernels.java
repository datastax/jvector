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

import jdk.incubator.vector.*;
import java.util.Objects;

/**
 * Tunable ASH SIMD kernels. Settings are JVM-lifetime constants so HotSpot can remove
 * unused registers and unrolled updates. Accumulators are named locals, never an array.
 *
 * <p>Unroll is the number of SIMD code chunks grouped in an outer-loop iteration.
 * Accumulators rotate independently of that grouping, so any supported combination
 * uses all requested registers. A chunk is a nibble pair for LUT scoring or one
 * SIMD-width of code bytes for projection scoring. HotSpot controls final machine
 * unrolling; keeping a single vector-helper call site avoids heap allocation caused
 * by exhausting the compiler's inlining budget.</p>
 *
 * <p>Optional VM settings (defaults shown):
 * {@code jvector.ash.lut.accumulators=2}, {@code jvector.ash.lut.unroll=1},
 * {@code jvector.ash.projection.accumulators=1}, {@code jvector.ash.projection.unroll=1}.
 * Accumulators accept 1/2/4/8; unroll accepts 1/2/4. Invalid values fail explicitly.
 * These settings are independent of the one-bit {@code jvector.ash.512.accumulators}.
 * Use a fresh JVM for each tuning trial.</p>
 */
final class PanamaASHKernels {
    private static final VectorSpecies<Float> F = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> I = VectorSpecies.of(int.class, F.vectorShape());
    private static final VectorSpecies<Byte> B = F.length() <= 8 ? ByteVector.SPECIES_64 : ByteVector.SPECIES_128;
    private static final int[] TWO_BIT_INDICES = queryIndices(4);
    private static final int[] FOUR_BIT_INDICES = queryIndices(2);

    // Conservative defaults from JDK 23/AVX-512 sweeps and end-to-end validation.
    // The original compact projection loop is retained for the default 1/1 setting.
    private static final int LUT_ACC = setting("lut.accumulators", 2, 8);
    private static final int LUT_UNROLL = setting("lut.unroll", 1, 4);
    private static final int PROJECTION_ACC = setting("projection.accumulators", 1, 8);
    private static final int PROJECTION_UNROLL = setting("projection.unroll", 1, 4);

    private PanamaASHKernels() {}

    private static int setting(String suffix, int fallback, int maximum) {
        String name = "jvector.ash." + suffix;
        String value = System.getProperty(name, Integer.toString(fallback)).trim();
        final int parsed;
        try { parsed = Integer.parseInt(value); }
        catch (NumberFormatException e) { throw new IllegalArgumentException(name + " must be a power of two in [1," + maximum + "]: " + value, e); }
        if (parsed < 1 || parsed > maximum || (parsed & (parsed - 1)) != 0) {
            throw new IllegalArgumentException(name + " must be a power of two in [1," + maximum + "]: " + value);
        }
        return parsed;
    }

    static String description() {
        return "LUT accumulators=" + LUT_ACC + ", unroll=" + LUT_UNROLL
                + "; projection accumulators=" + PROJECTION_ACC + ", unroll=" + PROJECTION_UNROLL;
    }

    static boolean supported() { return F.length() == 4 || F.length() == 8 || F.length() == 16; }

    static boolean projectionTuningEnabled() { return PROJECTION_ACC != 1 || PROJECTION_UNROLL != 1; }

    private static int[] queryIndices(int stride) {
        int[] indices = new int[F.length()];
        for (int i = 0; i < indices.length; i++) indices[i] = i * stride;
        return indices;
    }

    static void lutScore(byte[] codes, int offset, int groups, int stride,
                         int lane, int count, float[] lut, float[] out, int outOffset) {
        ASHLutScoring.checkBounds(codes, offset, groups, stride, lane, count, lut, out, outOffset);
        int pairs = groups / 2;
        for (int i = 0; i < count; i += F.length()) {
            int active = Math.min(F.length(), count - i);
            var byteMask = B.indexInRange(0, active);
            FloatVector a0 = FloatVector.zero(F);
            FloatVector a1 = FloatVector.zero(F);
            FloatVector a2 = FloatVector.zero(F);
            FloatVector a3 = FloatVector.zero(F);
            FloatVector a4 = FloatVector.zero(F);
            FloatVector a5 = FloatVector.zero(F);
            FloatVector a6 = FloatVector.zero(F);
            FloatVector a7 = FloatVector.zero(F);
            int pair = 0;
            while (pair < pairs) {
                int end = Math.min(pairs, pair + LUT_UNROLL);
                // One lookup call site avoids HotSpot's vector inlining budget limit.
                // A constant-trip inner loop supplies the requested unroll grouping.
                do {
                    FloatVector value = lutPair(codes, offset, pair, stride, lane + i, lut, byteMask);
                    FloatVector next = a0.add(value);
                    if (LUT_ACC == 1) {
                        a0 = next;
                    }
                    else if (LUT_ACC == 2) {
                        a0 = a1;
                        a1 = next;
                    }
                    else if (LUT_ACC == 4) {
                        a0 = a1;
                        a1 = a2;
                        a2 = a3;
                        a3 = next;
                    }
                    else if (LUT_ACC == 8) {
                        a0 = a1;
                        a1 = a2;
                        a2 = a3;
                        a3 = a4;
                        a4 = a5;
                        a5 = a6;
                        a6 = a7;
                        a7 = next;
                    }
                } while (++pair < end);
            }
            if ((groups & 1) != 0) {
                var packed = loadCodes(codes, offset + pairs * stride + lane + i, byteMask);
                a0 = a0.add(lookup(lut, (groups - 1) * 16, packed.and(15)));
            }
            if (LUT_ACC > 1) a0 = a0.add(a1);
            if (LUT_ACC > 2) a0 = a0.add(a2);
            if (LUT_ACC > 3) a0 = a0.add(a3);
            if (LUT_ACC > 4) a0 = a0.add(a4);
            if (LUT_ACC > 5) a0 = a0.add(a5);
            if (LUT_ACC > 6) a0 = a0.add(a6);
            if (LUT_ACC > 7) a0 = a0.add(a7);
            a0.intoArray(out, outOffset + i, F.indexInRange(0, active));
        }
    }

    private static IntVector loadCodes(byte[] codes, int offset, VectorMask<Byte> mask) {
        return (IntVector) ByteVector.fromArray(B, codes, offset, mask).convertShape(VectorOperators.B2I, I, 0);
    }

    private static FloatVector lutPair(byte[] codes, int offset, int pair, int stride,
                                       int lane, float[] lut, VectorMask<Byte> mask) {
        var packed = loadCodes(codes, offset + pair * stride + lane, mask);
        return lookup(lut, pair * 32, packed.and(15))
                .add(lookup(lut, pair * 32 + 16, packed.lanewise(VectorOperators.LSHR, 4).and(15)));
    }

    private static FloatVector lookup(float[] lut, int offset, IntVector indices) {
        int width = F.length();
        // Keep the shuffle and data in the same lane type. Casting an Int shuffle
        // to a Float shuffle can materialize heap objects on JDK 23. Reinterpreting
        // the LUT bits instead preserves register-only permutation.
        var shuffle = indices.and(width - 1).toShuffle();
        IntVector result = FloatVector.fromArray(F, lut, offset).reinterpretAsInts().rearrange(shuffle);
        for (int table = width; table < 16; table += width) {
            var values = FloatVector.fromArray(F, lut, offset + table).reinterpretAsInts().rearrange(shuffle);
            result = result.blend(values, indices.compare(VectorOperators.GE, table));
        }
        return result.reinterpretAsFloats();
    }

    static float projectionDot(float[] query, byte[] code, int dimensions, int bits) {
        if (bits != 2 && bits != 4) throw new IllegalArgumentException("Projection scoring requires 2 or 4 bits per dimension");
        Objects.checkFromIndexSize(0, dimensions, query.length);
        int perByte = 8 / bits;
        int bytes = dimensions / perByte + (dimensions % perByte == 0 ? 0 : 1);
        Objects.checkFromIndexSize(0, bytes, code.length);
        int sign = 1 << (bits - 1);
        int[] indices = bits == 2 ? TWO_BIT_INDICES : FOUR_BIT_INDICES;
        FloatVector a0 = FloatVector.zero(F);
        FloatVector a1 = FloatVector.zero(F);
        FloatVector a2 = FloatVector.zero(F);
        FloatVector a3 = FloatVector.zero(F);
        FloatVector a4 = FloatVector.zero(F);
        FloatVector a5 = FloatVector.zero(F);
        FloatVector a6 = FloatVector.zero(F);
        FloatVector a7 = FloatVector.zero(F);
        int base = 0;
        while (base < bytes) {
            int remainingChunks = (bytes - base - 1) / F.length() + 1;
            int chunks = Math.min(PROJECTION_UNROLL, remainingChunks);
            for (int u = 0; u < chunks; u++, base += F.length()) {
                var packed = loadCodes(code, base, B.indexInRange(0, Math.min(F.length(), bytes - base)));
                // Reuse each widened code byte across its 2/4 projected dimensions.
                // Keep the decode, query gather and accumulation in one compilation unit.
                for (int slot = 0; slot < perByte; slot++) {
                    int remaining = dimensions - base * perByte - slot;
                    if (remaining <= 0) break;
                    int active = Math.min(F.length(), (remaining - 1) / perByte + 1);
                    var fields = packed.lanewise(VectorOperators.LSHR, slot * bits);
                    var magnitude = ((FloatVector) fields.and(sign - 1).convertShape(VectorOperators.I2F, F, 0)).add(0.5f);
                    var values = magnitude.blend(magnitude.neg(), fields.and(sign).compare(VectorOperators.EQ, 0).cast(F));
                    var q = FloatVector.fromArray(F, query, base * perByte + slot, indices, 0, F.indexInRange(0, active));
                    FloatVector next = a0.add(q.mul(values));
                    if (PROJECTION_ACC == 1) {
                        a0 = next;
                    }
                    else if (PROJECTION_ACC == 2) {
                        a0 = a1;
                        a1 = next;
                    }
                    else if (PROJECTION_ACC == 4) {
                        a0 = a1;
                        a1 = a2;
                        a2 = a3;
                        a3 = next;
                    }
                    else if (PROJECTION_ACC == 8) {
                        a0 = a1;
                        a1 = a2;
                        a2 = a3;
                        a3 = a4;
                        a4 = a5;
                        a5 = a6;
                        a6 = a7;
                        a7 = next;
                    }
                }
            }
        }
        if (PROJECTION_ACC > 1) a0 = a0.add(a1);
        if (PROJECTION_ACC > 2) a0 = a0.add(a2);
        if (PROJECTION_ACC > 3) a0 = a0.add(a3);
        if (PROJECTION_ACC > 4) a0 = a0.add(a4);
        if (PROJECTION_ACC > 5) a0 = a0.add(a5);
        if (PROJECTION_ACC > 6) a0 = a0.add(a6);
        if (PROJECTION_ACC > 7) a0 = a0.add(a7);
        return a0.reduceLanes(VectorOperators.ADD);
    }
}
