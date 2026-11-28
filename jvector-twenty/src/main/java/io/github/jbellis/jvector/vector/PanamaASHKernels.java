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
 * unrolling; keeping a single copy of the lookup code avoids heap allocation caused
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
    // Code bytes widened to int lanes without convertShape, whose shared type profile can leave the result's exact
    // type unknown to C2: byte i of a full-width byte vector is spread to int lane i, then masked.
    private static final VectorSpecies<Byte> FB = VectorSpecies.of(byte.class, F.vectorShape());
    private static final VectorShuffle<Byte> SPREAD4;
    static {
        int[] index = new int[FB.length()];
        for (int i = 0; i < index.length; i++) index[i] = i / 4;
        SPREAD4 = VectorShuffle.fromArray(FB, index, 0);
    }
    private static final VectorMask<Byte> FB_LANES = FB.indexInRange(0, F.length());
    private static final int[] TWO_BIT_INDICES = queryIndices(4);
    private static final int[] FOUR_BIT_INDICES = queryIndices(2);
    private static final int[][] TWO_BIT_TAIL_INDICES = tailQueryIndices(4);
    private static final int[][] FOUR_BIT_TAIL_INDICES = tailQueryIndices(2);

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

    /**
     * Gather indices for a partial final block of {@code active} lanes. Inactive lanes point at
     * index 0: a masked gather may still bounds-check them, and a short query has no element
     * at their full-block positions.
     */
    private static int[][] tailQueryIndices(int stride) {
        int[][] tails = new int[F.length()][];
        for (int active = 1; active < tails.length; active++) {
            int[] indices = new int[F.length()];
            for (int lane = 0; lane < active; lane++) indices[lane] = lane * stride;
            tails[active] = indices;
        }
        return tails;
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
                // One copy of the lookup avoids HotSpot's vector inlining budget limit.
                // A constant-trip inner loop supplies the requested unroll grouping.
                do {
                    // Two nibbles per code byte; each indexes a 16-entry table (see the note after lutScore).
                    var packed = (IntVector) ByteVector.fromArray(B, codes, offset + pair * stride + lane + i, byteMask)
                            .convertShape(VectorOperators.B2I, I, 0);
                    var low = packed.lanewise(VectorOperators.AND, 15);
                    var high = packed.lanewise(VectorOperators.LSHR, 4).lanewise(VectorOperators.AND, 15);
                    var lowShuffle = low.lanewise(VectorOperators.AND, F.length() - 1).toShuffle();
                    var highShuffle = high.lanewise(VectorOperators.AND, F.length() - 1).toShuffle();
                    IntVector lowValues = FloatVector.fromArray(F, lut, pair * 32).reinterpretAsInts().rearrange(lowShuffle);
                    IntVector highValues = FloatVector.fromArray(F, lut, pair * 32 + 16).reinterpretAsInts().rearrange(highShuffle);
                    for (int table = F.length(); table < 16; table += F.length()) {
                        lowValues = lowValues.blend(FloatVector.fromArray(F, lut, pair * 32 + table)
                                .reinterpretAsInts().rearrange(lowShuffle), low.compare(VectorOperators.GE, table));
                        highValues = highValues.blend(FloatVector.fromArray(F, lut, pair * 32 + 16 + table)
                                .reinterpretAsInts().rearrange(highShuffle), high.compare(VectorOperators.GE, table));
                    }
                    FloatVector value = lowValues.reinterpretAsFloats().lanewise(VectorOperators.ADD, highValues.reinterpretAsFloats());
                    FloatVector next = a0.lanewise(VectorOperators.ADD, value);
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
                var low = ((IntVector) ByteVector.fromArray(B, codes, offset + pairs * stride + lane + i, byteMask)
                        .convertShape(VectorOperators.B2I, I, 0)).lanewise(VectorOperators.AND, 15);
                var lowShuffle = low.lanewise(VectorOperators.AND, F.length() - 1).toShuffle();
                IntVector lowValues = FloatVector.fromArray(F, lut, (groups - 1) * 16).reinterpretAsInts().rearrange(lowShuffle);
                for (int table = F.length(); table < 16; table += F.length()) {
                    lowValues = lowValues.blend(FloatVector.fromArray(F, lut, (groups - 1) * 16 + table)
                            .reinterpretAsInts().rearrange(lowShuffle), low.compare(VectorOperators.GE, table));
                }
                a0 = a0.lanewise(VectorOperators.ADD, lowValues.reinterpretAsFloats());
            }
            if (LUT_ACC > 1) a0 = a0.lanewise(VectorOperators.ADD, a1);
            if (LUT_ACC > 2) a0 = a0.lanewise(VectorOperators.ADD, a2);
            if (LUT_ACC > 3) a0 = a0.lanewise(VectorOperators.ADD, a3);
            if (LUT_ACC > 4) a0 = a0.lanewise(VectorOperators.ADD, a4);
            if (LUT_ACC > 5) a0 = a0.lanewise(VectorOperators.ADD, a5);
            if (LUT_ACC > 6) a0 = a0.lanewise(VectorOperators.ADD, a6);
            if (LUT_ACC > 7) a0 = a0.lanewise(VectorOperators.ADD, a7);
            a0.intoArray(out, outOffset + i, F.indexInRange(0, active));
        }
    }

    /**
     * Per-lane estimates {@code scale * dot + landmarkDot + offset} of a whole C=1 FusedASH neighborhood of nibble
     * codes (1, 2 and 4 bits), with the native kernel's arithmetic: per lane, the even groups' table values sum in one
     * accumulator and the odd groups' in another, {@code dot = even + odd}, then {@code fma(scale, dot, landmarkDot)
     * + offset}. Lookups are written out as in {@link #lutScore}. One body for every block, compiled on its own.
     *
     * @throws IllegalStateException if a scored lane names a landmark other than 0
     */
    static void lutNeighborhoodEstimates(byte[] codes, int groups, int blockSize, int capacity, float[] lut,
                                         float landmarkDot, float[] out) {
        if (groups <= 0 || capacity <= 0 || (blockSize != 8 && blockSize != 16 && blockSize != 32))
            throw new IllegalArgumentException("Invalid ASH neighborhood shape");
        if (out == lut) throw new IllegalArgumentException("ASH neighborhood output must not alias the LUT");
        // Sizes in long, so an impossible shape reports the array that is too small
        long bodyBytes = ((long) groups + 1) / 2 * blockSize;
        int blocks = (capacity - 1) / blockSize + 1;
        Objects.checkFromIndexSize(0L, blocks * (bodyBytes + 5L * blockSize), codes.length);
        Objects.checkFromIndexSize(0L, 16L * groups, lut.length);
        Objects.checkFromIndexSize(0, capacity, out.length);
        int body = (int) bodyBytes, blockBytes = body + 5 * blockSize;
        int pairs = groups / 2;
        for (int block = 0; block < blocks; block++) {
            int blockOffset = block * blockBytes, first = block * blockSize;
            int count = Math.min(blockSize, capacity - first);
            if (F.length() == 16 && blockSize == 32 && count == 32) {
                // The native full-B32 path: both 16-lane halves in one pass, sharing each group's table loads
                // Accumulators start from the first pair's values, never from zero(); one group alone is its own sum
                if (pairs == 0) {
                    {
                        int at = blockOffset + pairs * 32;
                        var c0 = ByteVector.fromArray(FB, codes, at, FB_LANES).rearrange(SPREAD4).reinterpretAsInts().lanewise(VectorOperators.AND, 0xFF);
                        var c1 = ByteVector.fromArray(FB, codes, at + 16, FB_LANES).rearrange(SPREAD4).reinterpretAsInts().lanewise(VectorOperators.AND, 0xFF);
                        var lowTable = FloatVector.fromArray(F, lut, pairs * 32).reinterpretAsInts();
                        var l0 = lowTable.rearrange(c0.lanewise(VectorOperators.AND, 15).toShuffle()).reinterpretAsFloats();
                        var l1 = lowTable.rearrange(c1.lanewise(VectorOperators.AND, 15).toShuffle()).reinterpretAsFloats();
                        l0.intoArray(out, first);
                        l1.intoArray(out, first + 16);
                    }
                } else {
                    FloatVector lo0, hi0, lo1, hi1;
                    {
                        int at = blockOffset + 0 * 32;
                        var c0 = ByteVector.fromArray(FB, codes, at, FB_LANES).rearrange(SPREAD4).reinterpretAsInts().lanewise(VectorOperators.AND, 0xFF);
                        var c1 = ByteVector.fromArray(FB, codes, at + 16, FB_LANES).rearrange(SPREAD4).reinterpretAsInts().lanewise(VectorOperators.AND, 0xFF);
                        var lowTable = FloatVector.fromArray(F, lut, 0 * 32).reinterpretAsInts();
                        var highTable = FloatVector.fromArray(F, lut, 0 * 32 + 16).reinterpretAsInts();
                        var l0 = lowTable.rearrange(c0.lanewise(VectorOperators.AND, 15).toShuffle()).reinterpretAsFloats();
                        var h0 = highTable.rearrange(c0.lanewise(VectorOperators.LSHR, 4).lanewise(VectorOperators.AND, 15).toShuffle()).reinterpretAsFloats();
                        var l1 = lowTable.rearrange(c1.lanewise(VectorOperators.AND, 15).toShuffle()).reinterpretAsFloats();
                        var h1 = highTable.rearrange(c1.lanewise(VectorOperators.LSHR, 4).lanewise(VectorOperators.AND, 15).toShuffle()).reinterpretAsFloats();
                        lo0 = l0; hi0 = h0; lo1 = l1; hi1 = h1;
                    }
                    for (int pair = 1; pair < pairs; pair++) {
                        int at = blockOffset + pair * 32;
                        var c0 = ByteVector.fromArray(FB, codes, at, FB_LANES).rearrange(SPREAD4).reinterpretAsInts().lanewise(VectorOperators.AND, 0xFF);
                        var c1 = ByteVector.fromArray(FB, codes, at + 16, FB_LANES).rearrange(SPREAD4).reinterpretAsInts().lanewise(VectorOperators.AND, 0xFF);
                        var lowTable = FloatVector.fromArray(F, lut, pair * 32).reinterpretAsInts();
                        var highTable = FloatVector.fromArray(F, lut, pair * 32 + 16).reinterpretAsInts();
                        var l0 = lowTable.rearrange(c0.lanewise(VectorOperators.AND, 15).toShuffle()).reinterpretAsFloats();
                        var h0 = highTable.rearrange(c0.lanewise(VectorOperators.LSHR, 4).lanewise(VectorOperators.AND, 15).toShuffle()).reinterpretAsFloats();
                        var l1 = lowTable.rearrange(c1.lanewise(VectorOperators.AND, 15).toShuffle()).reinterpretAsFloats();
                        var h1 = highTable.rearrange(c1.lanewise(VectorOperators.LSHR, 4).lanewise(VectorOperators.AND, 15).toShuffle()).reinterpretAsFloats();
                        lo0 = lo0.lanewise(VectorOperators.ADD, l0);
                        hi0 = hi0.lanewise(VectorOperators.ADD, h0);
                        lo1 = lo1.lanewise(VectorOperators.ADD, l1);
                        hi1 = hi1.lanewise(VectorOperators.ADD, h1);
                    }
                    if ((groups & 1) != 0) {
                        int at = blockOffset + pairs * 32;
                        var c0 = ByteVector.fromArray(FB, codes, at, FB_LANES).rearrange(SPREAD4).reinterpretAsInts().lanewise(VectorOperators.AND, 0xFF);
                        var c1 = ByteVector.fromArray(FB, codes, at + 16, FB_LANES).rearrange(SPREAD4).reinterpretAsInts().lanewise(VectorOperators.AND, 0xFF);
                        var lowTable = FloatVector.fromArray(F, lut, pairs * 32).reinterpretAsInts();
                        var l0 = lowTable.rearrange(c0.lanewise(VectorOperators.AND, 15).toShuffle()).reinterpretAsFloats();
                        var l1 = lowTable.rearrange(c1.lanewise(VectorOperators.AND, 15).toShuffle()).reinterpretAsFloats();
                        lo0 = lo0.lanewise(VectorOperators.ADD, l0);
                        lo1 = lo1.lanewise(VectorOperators.ADD, l1);
                    }
                    lo0.lanewise(VectorOperators.ADD, hi0).intoArray(out, first);
                    lo1.lanewise(VectorOperators.ADD, hi1).intoArray(out, first + 16);
                }
            } else
            for (int i = 0; i < count; i += F.length()) {
                int active = Math.min(F.length(), count - i);
                var byteMask = FB.indexInRange(0, active);
                // Accumulators start from the first pair's values, never from zero(); one group alone is its own sum
                if (pairs == 0) {
                    var low = ByteVector.fromArray(FB, codes, blockOffset + pairs * blockSize + i, byteMask).rearrange(SPREAD4).reinterpretAsInts().lanewise(VectorOperators.AND, 0xFF).lanewise(VectorOperators.AND, 15);
                    var lowShuffle = low.lanewise(VectorOperators.AND, F.length() - 1).toShuffle();
                    IntVector lowValues = FloatVector.fromArray(F, lut, pairs * 32).reinterpretAsInts().rearrange(lowShuffle);
                    for (int table = F.length(); table < 16; table += F.length()) {
                        lowValues = lowValues.blend(FloatVector.fromArray(F, lut, pairs * 32 + table)
                                .reinterpretAsInts().rearrange(lowShuffle), low.compare(VectorOperators.GE, table));
                    }
                    lowValues.reinterpretAsFloats().intoArray(out, first + i, F.indexInRange(0, active));
                    continue;
                }
                FloatVector even, odd;
                {
                    var packed = ByteVector.fromArray(FB, codes, blockOffset + 0 * blockSize + i, byteMask).rearrange(SPREAD4).reinterpretAsInts().lanewise(VectorOperators.AND, 0xFF);
                    var low = packed.lanewise(VectorOperators.AND, 15);
                    var high = packed.lanewise(VectorOperators.LSHR, 4).lanewise(VectorOperators.AND, 15);
                    var lowShuffle = low.lanewise(VectorOperators.AND, F.length() - 1).toShuffle();
                    var highShuffle = high.lanewise(VectorOperators.AND, F.length() - 1).toShuffle();
                    IntVector lowValues = FloatVector.fromArray(F, lut, 0 * 32).reinterpretAsInts().rearrange(lowShuffle);
                    IntVector highValues = FloatVector.fromArray(F, lut, 0 * 32 + 16).reinterpretAsInts().rearrange(highShuffle);
                    for (int table = F.length(); table < 16; table += F.length()) {
                        lowValues = lowValues.blend(FloatVector.fromArray(F, lut, 0 * 32 + table)
                                .reinterpretAsInts().rearrange(lowShuffle), low.compare(VectorOperators.GE, table));
                        highValues = highValues.blend(FloatVector.fromArray(F, lut, 0 * 32 + 16 + table)
                                .reinterpretAsInts().rearrange(highShuffle), high.compare(VectorOperators.GE, table));
                    }
                    even = lowValues.reinterpretAsFloats();
                    odd = highValues.reinterpretAsFloats();
                }
                for (int pair = 1; pair < pairs; pair++) {
                    var packed = ByteVector.fromArray(FB, codes, blockOffset + pair * blockSize + i, byteMask).rearrange(SPREAD4).reinterpretAsInts().lanewise(VectorOperators.AND, 0xFF);
                    var low = packed.lanewise(VectorOperators.AND, 15);
                    var high = packed.lanewise(VectorOperators.LSHR, 4).lanewise(VectorOperators.AND, 15);
                    var lowShuffle = low.lanewise(VectorOperators.AND, F.length() - 1).toShuffle();
                    var highShuffle = high.lanewise(VectorOperators.AND, F.length() - 1).toShuffle();
                    IntVector lowValues = FloatVector.fromArray(F, lut, pair * 32).reinterpretAsInts().rearrange(lowShuffle);
                    IntVector highValues = FloatVector.fromArray(F, lut, pair * 32 + 16).reinterpretAsInts().rearrange(highShuffle);
                    for (int table = F.length(); table < 16; table += F.length()) {
                        lowValues = lowValues.blend(FloatVector.fromArray(F, lut, pair * 32 + table)
                                .reinterpretAsInts().rearrange(lowShuffle), low.compare(VectorOperators.GE, table));
                        highValues = highValues.blend(FloatVector.fromArray(F, lut, pair * 32 + 16 + table)
                                .reinterpretAsInts().rearrange(highShuffle), high.compare(VectorOperators.GE, table));
                    }
                    even = even.lanewise(VectorOperators.ADD, lowValues.reinterpretAsFloats());
                    odd = odd.lanewise(VectorOperators.ADD, highValues.reinterpretAsFloats());
                }
                if ((groups & 1) != 0) {
                    var low = ByteVector.fromArray(FB, codes, blockOffset + pairs * blockSize + i, byteMask).rearrange(SPREAD4).reinterpretAsInts().lanewise(VectorOperators.AND, 0xFF).lanewise(VectorOperators.AND, 15);
                    var lowShuffle = low.lanewise(VectorOperators.AND, F.length() - 1).toShuffle();
                    IntVector lowValues = FloatVector.fromArray(F, lut, pairs * 32).reinterpretAsInts().rearrange(lowShuffle);
                    for (int table = F.length(); table < 16; table += F.length()) {
                        lowValues = lowValues.blend(FloatVector.fromArray(F, lut, pairs * 32 + table)
                                .reinterpretAsInts().rearrange(lowShuffle), low.compare(VectorOperators.GE, table));
                    }
                    even = even.lanewise(VectorOperators.ADD, lowValues.reinterpretAsFloats());
                }
                even.lanewise(VectorOperators.ADD, odd).intoArray(out, first + i, F.indexInRange(0, active));
            }
            // Headers per lane: big-endian half-float scale and offset, then the landmark byte
            int scales = blockOffset + body, offsets = scales + 2 * blockSize, landmarks = offsets + 2 * blockSize;
            for (int lane = 0; lane < count; lane++) {
                if (codes[landmarks + lane] != 0)
                    throw new IllegalStateException("Invalid FusedASH landmark: " + (codes[landmarks + lane] & 0xFF));
            }
            for (int lane = 0; lane < count; lane++) {
                int s = scales + 2 * lane, o = offsets + 2 * lane;
                float scale = Float.float16ToFloat((short) (((codes[s] & 0xFF) << 8) | (codes[s + 1] & 0xFF)));
                float offset = Float.float16ToFloat((short) (((codes[o] & 0xFF) << 8) | (codes[o + 1] & 0xFF)));
                out[first + lane] = Math.fma(scale, out[first + lane], landmarkDot) + offset;
            }
        }
    }

    // The canonical kernel's gather offsets: lane k reads the table of pair (i + k), 32 floats apart
    private static final int[] CANONICAL_TABLE_OFFSETS = queryIndices(32);
    // Per-thread gather maps for the canonical kernel's data-dependent indices
    private static final ThreadLocal<int[]> CANONICAL_MAPS = ThreadLocal.withInitial(() -> new int[2 * F.length()]);

    /**
     * The native canonical LUT dot: code byte {@code p} holds groups {@code 2p} (low nibble) and {@code 2p + 1} (high
     * nibble); one gather per nibble reads {@code lut[32p + nibble]} and {@code lut[32p + 16 + nibble]} for
     * {@code F.length()} pairs at once, the low and high sums add lane by lane, then the remaining pairs and an odd
     * final group add in order.
     */
    static float canonicalLutDot(byte[] code, int codeOffset, int groups, float[] lut) {
        if (groups <= 0) throw new IllegalArgumentException("groups must be positive");
        Objects.checkFromIndexSize(codeOffset, (groups + 1) / 2, code.length);
        Objects.checkFromIndexSize(0, Math.multiplyExact(groups, 16), lut.length);
        int pairs = groups / 2, lanes = F.length();
        int[] maps = CANONICAL_MAPS.get();
        var offsets = IntVector.fromArray(I, CANONICAL_TABLE_OFFSETS, 0);
        FloatVector low = FloatVector.zero(F), high = FloatVector.zero(F);
        int i = 0;
        for (; i + lanes <= pairs; i += lanes) {
            var packed = ByteVector.fromArray(FB, code, codeOffset + i, FB_LANES).rearrange(SPREAD4).reinterpretAsInts().lanewise(VectorOperators.AND, 0xFF);
            offsets.lanewise(VectorOperators.ADD, packed.lanewise(VectorOperators.AND, 15)).intoArray(maps, 0);
            offsets.lanewise(VectorOperators.ADD, packed.lanewise(VectorOperators.LSHR, 4).lanewise(VectorOperators.AND, 15)).lanewise(VectorOperators.ADD, 16).intoArray(maps, lanes);
            low = low.lanewise(VectorOperators.ADD, FloatVector.fromArray(F, lut, i * 32, maps, 0));
            high = high.lanewise(VectorOperators.ADD, FloatVector.fromArray(F, lut, i * 32, maps, lanes));
        }
        float sum = low.lanewise(VectorOperators.ADD, high).reduceLanes(VectorOperators.ADD);
        for (; i < pairs; i++) {
            sum += lut[i * 32 + (code[codeOffset + i] & 15)];
            sum += lut[i * 32 + 16 + ((code[codeOffset + i] >>> 4) & 15)];
        }
        if ((groups & 1) != 0) sum += lut[pairs * 32 + (code[codeOffset + pairs] & 15)];
        return sum;
    }

    // Table lookups are written out in lutScore: each 16-entry table is one, two or four registers, permuted by the
    // low bits of each lane's index and blended by its high bits. The shuffle and the data share the int lane type:
    // casting an int shuffle to a float shuffle can put vectors on the heap on JDK 23, so the table bits are
    // reinterpreted as ints instead.

    /** Projection dot with the code starting at {@code codeOffset}. */
    static float projectionDot(float[] query, byte[] code, int codeOffset, int dimensions, int bits) {
        if (bits != 2 && bits != 4) throw new IllegalArgumentException("Projection scoring requires 2 or 4 bits per dimension");
        Objects.checkFromIndexSize(0, dimensions, query.length);
        int perByte = 8 / bits;
        int bytes = dimensions / perByte + (dimensions % perByte == 0 ? 0 : 1);
        Objects.checkFromIndexSize(codeOffset, bytes, code.length);
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
                var packed = (IntVector) ByteVector.fromArray(B, code, codeOffset + base,
                        B.indexInRange(0, Math.min(F.length(), bytes - base))).convertShape(VectorOperators.B2I, I, 0);
                // Reuse each widened code byte across its 2/4 projected dimensions.
                // Keep the decode, query gather and accumulation in one compilation unit.
                for (int slot = 0; slot < perByte; slot++) {
                    int remaining = dimensions - base * perByte - slot;
                    if (remaining <= 0) break;
                    int active = Math.min(F.length(), (remaining - 1) / perByte + 1);
                    var fields = packed.lanewise(VectorOperators.LSHR, slot * bits);
                    var magnitude = ((FloatVector) fields.lanewise(VectorOperators.AND, sign - 1).convertShape(VectorOperators.I2F, F, 0)).lanewise(VectorOperators.ADD, 0.5f);
                    var values = magnitude.blend(magnitude.lanewise(VectorOperators.NEG), fields.lanewise(VectorOperators.AND, sign).compare(VectorOperators.EQ, 0).cast(F));
                    int[] gather = active == F.length() ? indices
                            : (bits == 2 ? TWO_BIT_TAIL_INDICES : FOUR_BIT_TAIL_INDICES)[active];
                    var q = FloatVector.fromArray(F, query, base * perByte + slot, gather, 0, F.indexInRange(0, active));
                    FloatVector next = a0.lanewise(VectorOperators.ADD, q.lanewise(VectorOperators.MUL, values));
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
        if (PROJECTION_ACC > 1) a0 = a0.lanewise(VectorOperators.ADD, a1);
        if (PROJECTION_ACC > 2) a0 = a0.lanewise(VectorOperators.ADD, a2);
        if (PROJECTION_ACC > 3) a0 = a0.lanewise(VectorOperators.ADD, a3);
        if (PROJECTION_ACC > 4) a0 = a0.lanewise(VectorOperators.ADD, a4);
        if (PROJECTION_ACC > 5) a0 = a0.lanewise(VectorOperators.ADD, a5);
        if (PROJECTION_ACC > 6) a0 = a0.lanewise(VectorOperators.ADD, a6);
        if (PROJECTION_ACC > 7) a0 = a0.lanewise(VectorOperators.ADD, a7);
        return a0.reduceLanes(VectorOperators.ADD);
    }
}
