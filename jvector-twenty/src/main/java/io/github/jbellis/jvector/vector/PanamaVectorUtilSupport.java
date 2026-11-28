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
import io.github.jbellis.jvector.util.MathUtil;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.FloatArray;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.util.Objects;
import jdk.incubator.vector.VectorShape;

import java.util.List;

class PanamaVectorUtilSupport implements VectorUtilSupport {
    // JDK 20's C2 miscompiles the masked AVX-512 loads and stores in the ASH kernels,
    // which crashes or corrupts the heap. JDK 20 is past end of life, so ASH uses the
    // scalar kernels there. The constant folds away on later JDKs.
    private static final boolean ASH_SIMD = Runtime.version().feature() != 20;
    // The prepared 2/4-bit projection dot, faster on JDK 25+. JDK 22's C2 loses its vectors to the heap once it is
    // inlined into a scorer, so earlier JDKs use the register-decode kernel (ashProjectionDot).
    private static final boolean ASH_PREPARED_PROJECTION = Runtime.version().feature() >= 25;
    private static final VectorSpecies<Float> ASH_FLOATS = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> ASH_INTS = VectorSpecies.of(
            int.class, ASH_FLOATS.vectorShape());
    // Byte loads need only one byte per float lane; conversion part zero widens those lanes.
    private static final VectorSpecies<Byte> ASH_BYTES = ASH_FLOATS.length() <= 8
            ? ByteVector.SPECIES_64 : ByteVector.SPECIES_128;
    private static final VectorSpecies<Long> ASH_SYM_LONGS = LongVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Byte> ASH_SYM_WORD_BYTES = VectorSpecies.of(byte.class, ASH_SYM_LONGS.vectorShape());
    private static final VectorSpecies<Byte> ASH_SYM_BYTES = ByteVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Short> ASH_SYM_SHORTS = VectorSpecies.of(short.class, ASH_SYM_BYTES.vectorShape());
    private static final VectorSpecies<Integer> ASH_SYM_INTS = VectorSpecies.of(int.class, ASH_SYM_BYTES.vectorShape());
    private static final VectorSpecies<Float> ASH_SYM_FLOATS = VectorSpecies.of(float.class, ASH_SYM_BYTES.vectorShape());
    // Symmetric kernels need at least 128-bit vectors; narrower shapes score in Java.
    private static final boolean ASH_SYM_SIMD = ASH_SYM_SHORTS.vectorBitSize() >= 128;
    private static final VectorSpecies<Byte> ASH_SYM_BLOCK_BYTES = VectorSpecies.of(byte.class,
            jdk.incubator.vector.VectorShape.forBitSize(Math.max(64, ASH_SYM_SHORTS.vectorBitSize() / 2)));

    @Override
    public void ashSymmetricLutScore(byte[] codes, int groups, int stride, int lane, int count,
                                      short[] lut, float[] out, int outOffset) {
        var shorts = ASH_SYM_SHORTS;
        if (shorts.length() < 16 || shorts.length() > 32) {
            VectorUtilSupport.super.ashSymmetricLutScore(codes, groups, stride, lane, count, lut, out, outOffset);
            return;
        }
        var bytes = ASH_SYM_BLOCK_BYTES;
        var ints = ASH_SYM_INTS;
        var floats = ASH_SYM_FLOATS;
        for (int i = 0; i < count; i += shorts.length()) {
            int active = Math.min(shorts.length(), count - i);
            var mask = bytes.indexInRange(0, active);
            var sum0 = FloatVector.zero(floats);
            var sum1 = FloatVector.zero(floats);
            for (int first = 0; first < groups; first += 64) {
                var subtotal = jdk.incubator.vector.ShortVector.zero(shorts);
                int end = Math.min(groups, first + 64);
                for (int g = first; g < end; g += 2) {
                    var packed = ByteVector.fromArray(bytes, codes, (g / 2) * stride + lane + i, mask);
                    var fields = (jdk.incubator.vector.ShortVector) packed.convertShape(VectorOperators.B2S, shorts, 0);
                    var low = jdk.incubator.vector.ShortVector.fromArray(shorts, lut, g * 32)
                            .rearrange(fields.lanewise(VectorOperators.AND, (short)15).toShuffle());
                    if (g + 1 < end) {
                        var high = jdk.incubator.vector.ShortVector.fromArray(shorts, lut, (g + 1) * 32)
                                .rearrange(fields.lanewise(VectorOperators.LSHR, 4).lanewise(VectorOperators.AND, (short)15).toShuffle());
                        low = low.lanewise(VectorOperators.ADD, high);
                    }
                    subtotal = subtotal.lanewise(VectorOperators.ADD, low);
                }
                var lo = (IntVector) subtotal.convertShape(VectorOperators.S2I, ints, 0);
                var hi = (IntVector) subtotal.convertShape(VectorOperators.S2I, ints, 1);
                sum0 = sum0.lanewise(VectorOperators.ADD, (FloatVector)lo.convertShape(VectorOperators.I2F, floats, 0));
                sum1 = sum1.lanewise(VectorOperators.ADD, (FloatVector)hi.convertShape(VectorOperators.I2F, floats, 0));
            }
            sum0.lanewise(VectorOperators.MUL, .25f).intoArray(out, outOffset + i, floats.indexInRange(0, active));
            if (active > floats.length()) sum1.lanewise(VectorOperators.MUL, .25f).intoArray(out, outOffset + i + floats.length(), floats.indexInRange(floats.length(), active));
        }
    }

    @Override
    public boolean supportsAshSymmetricScoring() { return ASH_SIMD && ASH_SYM_SIMD; }

    @Override
    public float ashSymmetricDot(long[] aBits, byte[] aCodes, long[] bBits, byte[] bCodes,
                                  int dimensions, int bits) {
        // Keep the Vector API compilation units small and species compile-time constant.
        // A shared, large dispatch body can leave boxed vectors in graph-pruning callers.
        return switch (bits) {
            case 1 -> ashSymmetricBinary(aBits, 0, bBits, 0, dimensions);
            case 2 -> ashSymmetricTwoBit(aCodes, 0, bCodes, 0, dimensions);
            case 4 -> ashSymmetricFourBit(aCodes, 0, bCodes, 0, dimensions);
            default -> throw new IllegalArgumentException("Symmetric SIMD supports 1, 2, and 4 bits");
        };
    }

    @Override
    public boolean supportsAshSymmetricDotAt() { return ASH_SIMD && ASH_SYM_SIMD; }

    @Override
    public float ashSymmetricDotAt(byte[] aCodes, int aOffset, byte[] bCodes, int bOffset,
                                   int dimensions, int bits) {
        // Other widths go to the offset-binary kernel, which also rejects invalid arguments
        if (dimensions < 0 || (bits != 2 && bits != 4))
            return ashSymmetricOffsetBinary(aCodes, aOffset, bCodes, bOffset, dimensions, bits);
        long bytes = ((long) dimensions * bits + 7) / 8;
        if (aOffset < 0 || bOffset < 0 || aOffset + bytes > aCodes.length || bOffset + bytes > bCodes.length)
            throw new IndexOutOfBoundsException("ASH code array ranges");
        return bits == 2 ? ashSymmetricTwoBit(aCodes, aOffset, bCodes, bOffset, dimensions)
                : ashSymmetricFourBit(aCodes, aOffset, bCodes, bOffset, dimensions);
    }

    @Override
    public float ashSymmetricBinaryDotAt(long[] aBits, int aOffset, long[] bBits, int bOffset, int dimensions) {
        if (dimensions < 0) throw new IllegalArgumentException("dimensions must be nonnegative");
        int words = (int) ((dimensions + 63L) >>> 6);
        Objects.checkFromIndexSize(aOffset, words, aBits.length);
        Objects.checkFromIndexSize(bOffset, words, bBits.length);
        return ashSymmetricBinary(aBits, aOffset, bBits, bOffset, dimensions);
    }

    @Override
    public boolean supportsAshSymmetricDotRows() { return supportsAshSymmetricDotAt(); }

    /**
     * All rows in one call, so the vector work runs in a body compiled on its own rather than inlined, one call per
     * row, deep in a graph-build caller. Each row's arithmetic is that of {@link #ashSymmetricDotAt}.
     */
    @Override
    public void ashSymmetricDotRows(byte[] query, int queryOffset, byte[] rows, int stride, int count,
                                    int dimensions, int bits, float[] out) {
        if (count <= 0) return;
        if (dimensions < 0 || bits < 2 || bits > 9)
            throw new IllegalArgumentException("Symmetric ASH rows require nonnegative dimensions and 2 to 9 bits");
        long bytes = ((long) dimensions * bits + 7) / 8;
        if (stride < bytes) throw new IllegalArgumentException("ASH row stride is shorter than a code");
        if (queryOffset < 0 || queryOffset + bytes > query.length || (long) (count - 1) * stride + bytes > rows.length)
            throw new IndexOutOfBoundsException("ASH code array ranges");
        Objects.checkFromIndexSize(0, count, out.length);
        if (bits == 2) ashSymmetricTwoBitRows(query, queryOffset, rows, stride, count, dimensions, out);
        else if (bits == 4) ashSymmetricFourBitRows(query, queryOffset, rows, stride, count, dimensions, out);
        else for (int i = 0; i < count; i++) out[i] = ashSymmetricOffsetBinary(query, queryOffset, rows, i * stride, dimensions, bits);
    }

    /** {@link #ashSymmetricTwoBit} of one code against every row, written out in the row loop. */
    private static void ashSymmetricTwoBitRows(byte[] aCodes, int aOff, byte[] rows, int stride, int count,
                                               int dimensions, float[] out) {
        final long slots = 0x5555555555555555L;
        int words = dimensions / 32;
        int tail = dimensions % 32;
        for (int row = 0; row < count; row++) {
            int bOff = row * stride;
            long sum = (long) words * 32;
            for (int word = 0; word < words; word += ASH_SYM_LONGS.length()) {
                var byteMask = ASH_SYM_WORD_BYTES.indexInRange(word * 8, words * 8);
                var a = ByteVector.fromArray(ASH_SYM_WORD_BYTES, aCodes, aOff + word * 8, byteMask).reinterpretAsLongs();
                var b = ByteVector.fromArray(ASH_SYM_WORD_BYTES, rows, bOff + word * 8, byteMask).reinterpretAsLongs();
                var negative = a.lanewise(VectorOperators.XOR, b).lanewise(VectorOperators.LSHR, 1).lanewise(VectorOperators.AND, slots);
                var ma = a.lanewise(VectorOperators.AND, slots);
                var mb = b.lanewise(VectorOperators.AND, slots);
                var both = ma.lanewise(VectorOperators.AND, mb);
                var delta = negative.lanewise(VectorOperators.BIT_COUNT).lanewise(VectorOperators.MUL, -2)
                        .add(ma.lanewise(VectorOperators.BIT_COUNT).lanewise(VectorOperators.MUL, 2))
                        .add(mb.lanewise(VectorOperators.BIT_COUNT).lanewise(VectorOperators.MUL, 2))
                        .add(ma.lanewise(VectorOperators.AND, negative).lanewise(VectorOperators.BIT_COUNT).lanewise(VectorOperators.MUL, -4))
                        .add(mb.lanewise(VectorOperators.AND, negative).lanewise(VectorOperators.BIT_COUNT).lanewise(VectorOperators.MUL, -4))
                        .add(both.lanewise(VectorOperators.BIT_COUNT).lanewise(VectorOperators.MUL, 4))
                        .add(both.lanewise(VectorOperators.AND, negative).lanewise(VectorOperators.BIT_COUNT).lanewise(VectorOperators.MUL, -8));
                sum += delta.reduceLanes(VectorOperators.ADD);
            }
            if (tail != 0) {
                long a = 0, b = 0;
                for (int j = 0; j < (tail + 3) / 4; j++) {
                    a |= (long) (aCodes[aOff + words * 8 + j] & 255) << (8 * j);
                    b |= (long) (rows[bOff + words * 8 + j] & 255) << (8 * j);
                }
                long valid = slots & ((1L << (2 * tail)) - 1);
                long negative = ((a ^ b) >>> 1) & valid;
                long ma = a & valid, mb = b & valid, both = ma & mb;
                sum += tail - 2 * Long.bitCount(negative)
                        + 2 * (Long.bitCount(ma) + Long.bitCount(mb))
                        - 4 * (Long.bitCount(ma & negative) + Long.bitCount(mb & negative))
                        + 4 * Long.bitCount(both) - 8 * Long.bitCount(both & negative);
            }
            out[row] = sum * 0.25f;
        }
    }

    /** {@link #ashSymmetricFourBit} of one code against every row, written out in the row loop. */
    private static void ashSymmetricFourBitRows(byte[] aCodes, int aOff, byte[] rows, int stride, int count,
                                                int dimensions, float[] out) {
        int half = dimensions / 2;
        for (int row = 0; row < count; row++) {
            int bOff = row * stride;
            long sum = 0;
            for (int offset = 0; offset < half; offset += ASH_SYM_BYTES.length()) {
                int active = Math.min(ASH_SYM_BYTES.length(), half - offset);
                var mask = ASH_SYM_BYTES.indexInRange(0, active);
                var a = ByteVector.fromArray(ASH_SYM_BYTES, aCodes, aOff + offset, mask);
                var b = ByteVector.fromArray(ASH_SYM_BYTES, rows, bOff + offset, mask);
                var total = jdk.incubator.vector.ShortVector.zero(ASH_SYM_SHORTS);
                for (int shift = 0; shift < 8; shift += 4) {
                    var ax = a.lanewise(VectorOperators.LSHR, shift);
                    var bx = b.lanewise(VectorOperators.LSHR, shift);
                    var am = ax.lanewise(VectorOperators.AND, (byte) 7).lanewise(VectorOperators.MUL, (byte) 2).lanewise(VectorOperators.ADD, (byte) 1);
                    var bm = bx.lanewise(VectorOperators.AND, (byte) 7).lanewise(VectorOperators.MUL, (byte) 2).lanewise(VectorOperators.ADD, (byte) 1);
                    am = am.blend(am.lanewise(VectorOperators.NEG), ax.lanewise(VectorOperators.AND, (byte) 8).compare(VectorOperators.EQ, (byte) 0));
                    bm = bm.blend(bm.lanewise(VectorOperators.NEG), bx.lanewise(VectorOperators.AND, (byte) 8).compare(VectorOperators.EQ, (byte) 0));
                    for (int part = 0; part < 2; part++) {
                        var av = (jdk.incubator.vector.ShortVector) am.convertShape(VectorOperators.B2S, ASH_SYM_SHORTS, part);
                        var bv = (jdk.incubator.vector.ShortVector) bm.convertShape(VectorOperators.B2S, ASH_SYM_SHORTS, part);
                        total = total.lanewise(VectorOperators.ADD, av.lanewise(VectorOperators.MUL, bv));
                    }
                }
                for (int part = 0; part < 2; part++) {
                    sum += ((IntVector) total.convertShape(VectorOperators.S2I, ASH_SYM_INTS, part))
                            .reduceLanes(VectorOperators.ADD);
                }
                sum -= 2L * (ASH_SYM_BYTES.length() - active); // zero padding decodes to -1 in each nibble
            }
            if ((dimensions & 1) != 0) {
                int a = aCodes[aOff + half] & 15, b = rows[bOff + half] & 15;
                int av = (2 * (a & 7) + 1) * ((a & 8) == 0 ? -1 : 1);
                int bv = (2 * (b & 7) + 1) * ((b & 8) == 0 ? -1 : 1);
                sum += av * bv;
            }
            out[row] = sum * 0.25f;
        }
    }

    private static float ashSymmetricBinary(long[] aBits, int aOff, long[] bBits, int bOff, int dimensions) {
        var longs = ASH_SYM_LONGS;
            long mismatches = 0;
            int words = dimensions >>> 6;
            for (int w = 0; w < words; w += longs.length()) {
                var mask = longs.indexInRange(w, words);
                var a = LongVector.fromArray(longs, aBits, aOff + w, mask);
                var b = LongVector.fromArray(longs, bBits, bOff + w, mask);
                mismatches += a.lanewise(VectorOperators.XOR, b)
                        .lanewise(VectorOperators.BIT_COUNT).reduceLanes(VectorOperators.ADD);
            }
            int tail = dimensions & 63;
            if (tail != 0) mismatches += Long.bitCount((aBits[aOff + words] ^ bBits[bOff + words]) & ((1L << tail) - 1));
            return dimensions - 2f * mismatches;
    }

    private static float ashSymmetricTwoBit(byte[] aCodes, int aOff, byte[] bCodes, int bOff, int dimensions) {
        var longs = ASH_SYM_LONGS;
            // The doubled signed code is sign * (1 + 2*magnitudeBit).
            // Weighted popcounts evaluate its product without unpacking components.
            final long slots = 0x5555555555555555L;
            int words = dimensions / 32;
            long sum = (long) words * 32;
            for (int word = 0; word < words; word += longs.length()) {
                var byteSpecies = ASH_SYM_WORD_BYTES;
                var byteMask = byteSpecies.indexInRange(word * 8, words * 8);
                var a = ByteVector.fromArray(byteSpecies, aCodes, aOff + word * 8, byteMask).reinterpretAsLongs();
                var b = ByteVector.fromArray(byteSpecies, bCodes, bOff + word * 8, byteMask).reinterpretAsLongs();
                var negative = a.lanewise(VectorOperators.XOR, b).lanewise(VectorOperators.LSHR, 1).lanewise(VectorOperators.AND, slots);
                var ma = a.lanewise(VectorOperators.AND, slots);
                var mb = b.lanewise(VectorOperators.AND, slots);
                var both = ma.lanewise(VectorOperators.AND, mb);
                var delta = negative.lanewise(VectorOperators.BIT_COUNT).lanewise(VectorOperators.MUL, -2)
                        .add(ma.lanewise(VectorOperators.BIT_COUNT).lanewise(VectorOperators.MUL, 2))
                        .add(mb.lanewise(VectorOperators.BIT_COUNT).lanewise(VectorOperators.MUL, 2))
                        .add(ma.lanewise(VectorOperators.AND, negative).lanewise(VectorOperators.BIT_COUNT).lanewise(VectorOperators.MUL, -4))
                        .add(mb.lanewise(VectorOperators.AND, negative).lanewise(VectorOperators.BIT_COUNT).lanewise(VectorOperators.MUL, -4))
                        .add(both.lanewise(VectorOperators.BIT_COUNT).lanewise(VectorOperators.MUL, 4))
                        .add(both.lanewise(VectorOperators.AND, negative).lanewise(VectorOperators.BIT_COUNT).lanewise(VectorOperators.MUL, -8));
                sum += delta.reduceLanes(VectorOperators.ADD);
            }
            int tail = dimensions % 32;
            if (tail != 0) {
                long a = 0, b = 0;
                for (int j = 0; j < (tail + 3) / 4; j++) {
                    a |= (long) (aCodes[aOff + words * 8 + j] & 255) << (8 * j);
                    b |= (long) (bCodes[bOff + words * 8 + j] & 255) << (8 * j);
                }
                long valid = slots & ((1L << (2 * tail)) - 1);
                long negative = ((a ^ b) >>> 1) & valid;
                long ma = a & valid, mb = b & valid, both = ma & mb;
                sum += tail - 2 * Long.bitCount(negative)
                        + 2 * (Long.bitCount(ma) + Long.bitCount(mb))
                        - 4 * (Long.bitCount(ma & negative) + Long.bitCount(mb & negative))
                        + 4 * Long.bitCount(both) - 8 * Long.bitCount(both & negative);
            }
            return sum * 0.25f;
    }

    private static float ashSymmetricFourBit(byte[] aCodes, int aOff, byte[] bCodes, int bOff, int dimensions) {
        var bytes = ASH_SYM_BYTES;
        var shorts = ASH_SYM_SHORTS;
        int count = dimensions / 2;
        long sum = 0;
        for (int offset = 0; offset < count; offset += bytes.length()) {
            int active = Math.min(bytes.length(), count - offset);
            var mask = bytes.indexInRange(0, active);
            var a = ByteVector.fromArray(bytes, aCodes, aOff + offset, mask);
            var b = ByteVector.fromArray(bytes, bCodes, bOff + offset, mask);
            var total = jdk.incubator.vector.ShortVector.zero(shorts);
            for (int shift = 0; shift < 8; shift += 4) {
                var ax = a.lanewise(VectorOperators.LSHR, shift);
                var bx = b.lanewise(VectorOperators.LSHR, shift);
                var am = ax.lanewise(VectorOperators.AND, (byte) 7).lanewise(VectorOperators.MUL, (byte) 2).lanewise(VectorOperators.ADD, (byte) 1);
                var bm = bx.lanewise(VectorOperators.AND, (byte) 7).lanewise(VectorOperators.MUL, (byte) 2).lanewise(VectorOperators.ADD, (byte) 1);
                am = am.blend(am.lanewise(VectorOperators.NEG), ax.lanewise(VectorOperators.AND, (byte) 8).compare(VectorOperators.EQ, (byte) 0));
                bm = bm.blend(bm.lanewise(VectorOperators.NEG), bx.lanewise(VectorOperators.AND, (byte) 8).compare(VectorOperators.EQ, (byte) 0));
                for (int part = 0; part < 2; part++) {
                    var av = (jdk.incubator.vector.ShortVector) am.convertShape(VectorOperators.B2S, shorts, part);
                    var bv = (jdk.incubator.vector.ShortVector) bm.convertShape(VectorOperators.B2S, shorts, part);
                    total = total.lanewise(VectorOperators.ADD, av.lanewise(VectorOperators.MUL, bv));
                }
            }
            // Widen before reduction: no short-sum overflow, including wider future species.
            for (int part = 0; part < 2; part++) {
                sum += ((IntVector) total.convertShape(VectorOperators.S2I, ASH_SYM_INTS, part))
                        .reduceLanes(VectorOperators.ADD);
            }
            sum -= 2L * (bytes.length() - active); // zero padding decodes to -1 in each nibble
        }
        if ((dimensions & 1) != 0) {
            int a = aCodes[aOff + count] & 15, b = bCodes[bOff + count] & 15;
            int av = (2 * (a & 7) + 1) * ((a & 8) == 0 ? -1 : 1);
            int bv = (2 * (b & 7) + 1) * ((b & 8) == 0 ? -1 : 1);
            sum += av * bv;
        }
        return sum * 0.25f;
    }

    /**
     * Offset-binary codes (3 and 5 to 9 bits): the exact sum of {@code (2x - m)(2y - m)} over fields x and y, with
     * {@code m = 2^b - 1}, times 1/4, so any summation order gives the same score. Fields are decoded in registers as
     * in {@link #ashOffsetBinaryDot}, from {@code sum(xy)} and {@code sum(x + y)}.
     */
    private static float ashSymmetricOffsetBinary(byte[] a, int aOff, byte[] b, int bOff, int dimensions, int bits) {
        if (dimensions < 0 || bits < 2 || bits > 9)
            throw new IllegalArgumentException("Offset symmetric ASH requires nonnegative dimensions and 2 to 9 bits");
        int codeBytes = (int) (((long) dimensions * bits + 7) / 8);
        if (aOff < 0 || bOff < 0 || aOff + codeBytes > a.length || bOff + codeBytes > b.length)
            throw new IndexOutOfBoundsException("ASH code array ranges");
        int lanes = ASH_FLOATS.length(), m = (1 << bits) - 1;
        // Four-lane and other untabled shapes decode in Java; 32-byte loads serve every width.
        int steps = ASH_OB_SIMD && lanes >= 8 ? dimensions / lanes : 0, stepBytes = lanes * bits / 8, load = ASH_OB_LOAD_WIDE.length();
        long products = 0, fieldSums = 0;
        if (steps > 0) {
            int last = (steps - 1) * stepBytes + load;
            if (aOff + last > a.length || bOff + last > b.length) {
                // One padded copy holds both codes, so every step loads in place.
                byte[] both = ashPaddedCopy(a, aOff, codeBytes, codeBytes + 2 * load);
                System.arraycopy(b, bOff, both, codeBytes + load, codeBytes);
                a = b = both;
                aOff = 0;
                bOff = codeBytes + load;
            }
            var windows = ASH_OB_SHUFFLES[bits];
            var shifts = IntVector.fromArray(ASH_INTS, ASH_OB_SHIFTS[bits], 0);
            // Each chunk's int sums stay below 2^31: at most 8192 fields of at most 511 * 511.
            for (int from = 0, chunk = 8192 / lanes; from < steps; from += chunk) {
                IntVector xy = IntVector.zero(ASH_INTS), sums = IntVector.zero(ASH_INTS);
                for (int step = from, end = Math.min(steps, from + chunk); step < end; step++) {
                    var x = ((IntVector) ((ByteVector) ByteVector.fromArray(ASH_OB_LOAD_WIDE, a, aOff + step * stepBytes)
                            .reinterpretShape(ASH_OB_BYTES, 0)).rearrange(windows).reinterpretAsInts())
                            .lanewise(VectorOperators.LSHR, shifts).lanewise(VectorOperators.AND, m);
                    var y = ((IntVector) ((ByteVector) ByteVector.fromArray(ASH_OB_LOAD_WIDE, b, bOff + step * stepBytes)
                            .reinterpretShape(ASH_OB_BYTES, 0)).rearrange(windows).reinterpretAsInts())
                            .lanewise(VectorOperators.LSHR, shifts).lanewise(VectorOperators.AND, m);
                    xy = xy.lanewise(VectorOperators.ADD, x.lanewise(VectorOperators.MUL, y));
                    sums = sums.lanewise(VectorOperators.ADD, x.lanewise(VectorOperators.ADD, y));
                }
                products += xy.reduceLanes(VectorOperators.ADD);
                fieldSums += sums.reduceLanes(VectorOperators.ADD);
            }
        }
        int from = steps * lanes;
        long sum = 4 * products - 2L * m * fieldSums + (long) from * m * m;
        for (int j = from; j < dimensions; j++) {
            sum += (long) (2 * ashOffsetBinaryField(a, aOff, j, bits) - m) * (2 * ashOffsetBinaryField(b, bOff, j, bits) - m);
        }
        return sum * 0.25f;
    }

    /** Field {@code j} of an offset-binary code: bits {@code [j*b, (j+1)*b)}, little-endian. */
    private static int ashOffsetBinaryField(byte[] code, int codeOffset, int j, int bits) {
        int bit = j * bits, at = codeOffset + (bit >>> 3), shift = bit & 7;
        int window = code[at] & 0xFF;
        if (shift + bits > 8) window |= (code[at + 1] & 0xFF) << 8;
        if (shift + bits > 16) window |= (code[at + 2] & 0xFF) << 16;
        return (window >>> shift) & ((1 << bits) - 1);
    }

    @Override
    public boolean supportsAshProjectionScoring() { return ASH_SIMD && PanamaASHKernels.supported(); }

    @Override
    public boolean supportsAshLutScoring() { return ASH_SIMD && PanamaASHKernels.supported(); }

    // Codes decoded in registers: offset-binary (3 and 5 to 9 bits) and sign-magnitude (2 and 4 bits) fields, field j
    // at bits [j*b, (j+1)*b). A step of ASH_OB_STEP fields starts on a byte boundary (8 fields fill b bytes). Each
    // float lane takes the little-endian 32-bit window starting at the byte that holds its field's first bit, shifts the
    // field down and masks it; a field ends at most 16 bits into its window.
    private static final VectorSpecies<Byte> ASH_OB_BYTES = VectorSpecies.of(byte.class, ASH_FLOATS.vectorShape());
    // The decode tables cover 4, 8 and 16 float lanes; other vector shapes score these codes in Java.
    private static final boolean ASH_OB_SIMD =
            ASH_FLOATS.length() == 4 || ASH_FLOATS.length() == 8 || ASH_FLOATS.length() == 16;
    private static final int ASH_OB_STEP = Math.max(8, ASH_FLOATS.length());
    private static final int ASH_OB_VECTORS = ASH_OB_STEP / ASH_FLOATS.length();
    // With 8 or more lanes a step loads only the bytes its fields span, widened in registers: 16 bytes, or 32 for 8 and
    // 9 bits with 16 lanes. A code in an exactly sized array then copies only its last steps.
    private static final VectorSpecies<Byte> ASH_OB_LOAD =
            ASH_FLOATS.length() >= 8 ? ByteVector.SPECIES_128 : ASH_OB_BYTES;
    private static final VectorSpecies<Byte> ASH_OB_LOAD_WIDE =
            ASH_FLOATS.length() >= 16 ? ByteVector.SPECIES_256 : ASH_OB_LOAD;
    // Per width: each lane's window bytes as shuffles, built once (JDK 22 does not reliably fold per-call shuffles),
    // and each lane's shift
    @SuppressWarnings("unchecked")
    private static final VectorShuffle<Byte>[] ASH_OB_SHUFFLES = new VectorShuffle[10], ASH_OB_SHUFFLES_HI = new VectorShuffle[10];
    private static final int[][] ASH_OB_SHIFTS = new int[10][];
    // Codes copied with load padding when they end too close to the end of their array
    private static final ThreadLocal<byte[][]> ASH_OB_PADDED =
            ThreadLocal.withInitial(() -> new byte[][] {new byte[0]});

    static {
        int lanes = ASH_FLOATS.length();
        for (int bits = 2; ASH_OB_SIMD && bits <= 9; bits++) {
            int[] windows = new int[ASH_OB_STEP * 4], shifts = new int[ASH_OB_STEP];
            for (int field = 0; field < ASH_OB_STEP; field++) {
                int bit = field * bits;
                int last = ashObLoad(bits).length() - 1;
                for (int k = 0; k < 4; k++) windows[field * 4 + k] = Math.min(bit / 8 + k, last);
                shifts[field] = bit & 7;
            }
            assert (ASH_OB_STEP - 1) * bits / 8 + 1 < ashObLoad(bits).length() && lanes * ASH_OB_VECTORS == ASH_OB_STEP;
            ASH_OB_SHUFFLES[bits] = VectorShuffle.fromArray(ASH_OB_BYTES, windows, 0);
            if (ASH_OB_VECTORS == 2) ASH_OB_SHUFFLES_HI[bits] = VectorShuffle.fromArray(ASH_OB_BYTES, windows, 4 * lanes);
            ASH_OB_SHIFTS[bits] = shifts;
        }
    }

    // INT8 field-block lane dot: one 8-field unit per 256-bit integer vector, windows as above.
    private static final VectorSpecies<Byte> ASH_UNIT_BYTES = ByteVector.SPECIES_256;
    private static final VectorSpecies<Integer> ASH_UNIT_INTS = IntVector.SPECIES_256;
    // Exact class for the INT8 field kernels: each value states it rather than rely on a type profile, which a
    // deep caller (the per-node dot is inlined into it) or another compile order can lose.
    private static final Class<?> ASH_UNIT_INTS_CLASS = ASH_UNIT_INTS.vectorType();
    @SuppressWarnings("unchecked")
    private static final VectorShuffle<Byte>[] ASH_UNIT_SHUFFLES = new VectorShuffle[10];
    private static final int[][] ASH_UNIT_SHIFTS = new int[10][];
    // A unit whose array ends before a whole 32-byte load, copied
    private static final ThreadLocal<byte[]> ASH_UNIT_TAIL = ThreadLocal.withInitial(() -> new byte[32]);

    static {
        for (int bits = 3; bits <= 9; bits++) {
            int[] windows = new int[32], shifts = new int[8];
            for (int field = 0; field < 8; field++) {
                for (int k = 0; k < 4; k++) windows[field * 4 + k] = Math.min(field * bits / 8 + k, 31);
                shifts[field] = (field * bits) & 7;
            }
            ASH_UNIT_SHUFFLES[bits] = VectorShuffle.fromArray(ASH_UNIT_BYTES, windows, 0);
            ASH_UNIT_SHIFTS[bits] = shifts;
        }
    }

    // 256-bit integer vectors are native on AVX2 and AVX-512; narrower CPUs keep the Java loop.
    @Override
    public boolean supportsAshOffsetBinaryInt8Dot() { return ASH_SIMD && ASH_FLOATS.length() >= 8; }

    @Override
    public long ashOffsetBinaryInt8Dot(byte[] codes, int offset, int unitStride, int units, int bits, int[] digits) {
        if (bits < 3 || bits > 9 || bits == 4) {
            throw new IllegalArgumentException("Offset-binary codes have 3 or 5 to 9 bits per dimension, got " + bits);
        }
        Objects.checkFromIndexSize(0, units * 8, digits.length);
        if (units > 0) Objects.checkFromIndexSize(offset, (long) (units - 1) * unitStride + bits, codes.length);
        var windows = ASH_UNIT_SHUFFLES[bits];
        var shifts = (IntVector) ASH_UNIT_INTS_CLASS.cast(IntVector.fromArray(ASH_UNIT_INTS, ASH_UNIT_SHIFTS[bits], 0));
        int mask = (1 << bits) - 1;
        // Exact: |digit * field| < 2^16, so 2^15 units fit an int lane, in any order. Keep one copy of the vector
        // operations: unrolled copies were sometimes compiled as heap-allocating calls.
        IntVector acc = (IntVector) ASH_UNIT_INTS_CLASS.cast(IntVector.zero(ASH_UNIT_INTS));
        // Units whose 32-byte load stays inside the array
        int inPlace = Math.max(0, Math.min(units, Math.floorDiv(codes.length - offset - 32, unitStride) + 1));
        for (int unit = 0, at = offset; unit < units; unit++, at += unitStride) {
            byte[] source = codes;
            int from = at;
            if (unit >= inPlace) {
                source = ASH_UNIT_TAIL.get();
                System.arraycopy(codes, at, source, 0, bits);
                from = 0;
            }
            var fields = (IntVector) ASH_UNIT_INTS_CLASS.cast(ByteVector.fromArray(ASH_UNIT_BYTES, source, from).rearrange(windows)
                    .reinterpretAsInts().lanewise(VectorOperators.LSHR, shifts).lanewise(VectorOperators.AND, mask));
            var weights = (IntVector) ASH_UNIT_INTS_CLASS.cast(IntVector.fromArray(ASH_UNIT_INTS, digits, unit * 8));
            acc = (IntVector) ASH_UNIT_INTS_CLASS.cast(acc.lanewise(VectorOperators.ADD, fields.lanewise(VectorOperators.MUL, weights)));
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    @Override
    public boolean supportsAshInt8NeighborhoodScoring() { return supportsAshOffsetBinaryInt8Dot(); }

    // Both neighborhood kernels, nibble LUT and INT8 field blocks, have their EUCLIDEAN forms.
    @Override
    public boolean supportsAshEuclideanNeighborhoodScoring() { return supportsAshNeighborhoodScoring(); }

    @Override
    public boolean supportsAshNeighborhoodScoring() { return ASH_SIMD && PanamaASHKernels.supported(); }

    @Override
    public void ashScoreNeighborhood(byte[] codes, int groups, int blockSize, int capacity,
                                     float[] lut, float landmarkDot, float[] out) {
        PanamaASHKernels.lutNeighborhoodEstimates(codes, groups, blockSize, capacity, lut, landmarkDot, out);
        for (int i = 0; i < capacity; i++) out[i] = Math.fma(out[i], 0.5f, 0.5f);
    }

    @Override
    public void ashScoreNeighborhoodEuclidean(byte[] codes, int groups, int blockSize, int capacity, float[] lut,
                                              float landmarkDot, float squaredDistancePerEstimate, float[] out) {
        if (!(squaredDistancePerEstimate < 0f) || Float.isInfinite(squaredDistancePerEstimate))
            throw new IllegalArgumentException("A EUCLIDEAN factor must be negative and finite");
        PanamaASHKernels.lutNeighborhoodEstimates(codes, groups, blockSize, capacity, lut, landmarkDot, out);
        for (int i = 0; i < capacity; i++) out[i] = 1f / (1f + Math.max(squaredDistancePerEstimate * out[i], 0f));
    }

    // Direct (non-fused) node scoring as native chooses it: 2-bit codes through the register-decode projection dot,
    // the canonical LUT where a caller asks for it. Exception: on JDK 25+ the prepared projection dot scores 2- and
    // 4-bit nodes (see ASH_PREPARED_PROJECTION).
    @Override
    public boolean supportsNativeAshTwoBitProjectionDot() {
        return ASH_SIMD && PanamaASHKernels.supported() && !ASH_PREPARED_PROJECTION;
    }

    @Override
    public boolean supportsAshCanonicalLutScoring() {
        return ASH_SIMD && PanamaASHKernels.supported() && !ASH_PREPARED_PROJECTION;
    }

    @Override
    public boolean supportsNativeAshCanonicalLutScoring() { return supportsAshCanonicalLutScoring(); }

    @Override
    public float ashCanonicalLutDot(byte[] code, int codeOffset, int groups, float[] lut) {
        if (!ASH_SIMD || !PanamaASHKernels.supported()) return VectorUtilSupport.super.ashCanonicalLutDot(code, codeOffset, groups, lut);
        return PanamaASHKernels.canonicalLutDot(code, codeOffset, groups, lut);
    }

    /** FP32, INT8 and FP16 query projection, as on the native provider. */
    @Override
    public String ashQueryProjectionUnsupportedReason(ASHQueryProjection precision) {
        return ASH_SIMD && PanamaASHProjection.supported() ? null : VectorUtilSupport.super.ashQueryProjectionUnsupportedReason(precision);
    }

    // The query widened to int, per thread, for the neighborhood kernel's integer multiplies
    private static final ThreadLocal<int[][]> ASH_WIDE_QUERY = ThreadLocal.withInitial(() -> new int[][] {new int[0]});

    @Override
    public void ashScoreNeighborhoodInt8(byte[] codes, int bits, int units, int blockSize, int capacity,
                                         byte[] q, float invScale, float center, float landmarkDot, float[] out) {
        ashInt8NeighborhoodEstimates(codes, bits, units, blockSize, capacity, ashWideQuery(q, units), invScale, center,
                landmarkDot, out);
        for (int i = 0; i < capacity; i++) out[i] = Math.fma(out[i], 0.5f, 0.5f);
    }

    @Override
    public void ashScoreNeighborhoodInt8Euclidean(byte[] codes, int bits, int units, int blockSize, int capacity,
                                                  byte[] q, float invScale, float center, float landmarkDot,
                                                  float squaredDistancePerEstimate, float[] out) {
        if (!(squaredDistancePerEstimate < 0f) || Float.isInfinite(squaredDistancePerEstimate))
            throw new IllegalArgumentException("A EUCLIDEAN factor must be negative and finite");
        ashInt8NeighborhoodEstimates(codes, bits, units, blockSize, capacity, ashWideQuery(q, units), invScale, center,
                landmarkDot, out);
        for (int i = 0; i < capacity; i++) out[i] = 1f / (1f + Math.max(squaredDistancePerEstimate * out[i], 0f));
    }

    private static int[] ashWideQuery(byte[] q, int units) {
        if (units <= 0) throw new IllegalArgumentException("units must be positive");
        Objects.checkFromIndexSize(0L, 8L * units, q.length);
        int n = units * 8;
        int[][] holder = ASH_WIDE_QUERY.get();
        int[] wide = holder[0];
        if (wide.length < n) holder[0] = wide = new int[n];
        for (int i = 0; i < n; i++) wide[i] = q[i];
        return wide;
    }

    // The native INT8 neighborhood layout: one 64-bit word per node, a vector of words for consecutive nodes. A node's
    // unit (eight fields in `bits` bytes) is gathered into its word, each field is shifted out and multiplied by its
    // broadcast query digit, and products add into the word's low 32-bit lane.
    private static final VectorSpecies<Long> ASH_WORD_LONGS = LongVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Byte> ASH_WORD_BYTES = VectorSpecies.of(byte.class, ASH_WORD_LONGS.vectorShape());
    private static final VectorSpecies<Integer> ASH_WORD_INTS = VectorSpecies.of(int.class, ASH_WORD_LONGS.vectorShape());
    private static final int ASH_WORD_LANES = ASH_WORD_LONGS.length();
    @SuppressWarnings("unchecked")
    private static final VectorShuffle<Byte>[] ASH_WORD_GATHER = new VectorShuffle[9];
    private static final ThreadLocal<int[]> ASH_WORD_SUMS = ThreadLocal.withInitial(() -> new int[ASH_WORD_INTS.length()]);
    static {
        for (int bits = 3; bits <= 8; bits++) {
            int[] gather = new int[ASH_WORD_BYTES.length()];
            for (int i = 0; i < gather.length; i++) gather[i] = (i / 8) * bits + Math.min(i % 8, bits - 1);
            ASH_WORD_GATHER[bits] = VectorShuffle.fromArray(ASH_WORD_BYTES, gather, 0);
        }
    }

    /** {@link #ashInt8NeighborhoodEstimates} for 3 and 5 to 8 bits, in the native word layout; same exact sums. */
    private static void ashInt8NeighborhoodWords(byte[] codes, int bits, int units, int blockSize, int capacity,
                                                 int[] digits, float invScale, float center, float landmarkDot,
                                                 float[] out, int body, int blockBytes, int blocks) {
        final int lanes = ASH_WORD_LANES, rowBytes = blockSize * bits, loadBytes = ASH_WORD_BYTES.length();
        final var gather = ASH_WORD_GATHER[bits];
        final long mask = (1L << bits) - 1;
        final int[] sums = ASH_WORD_SUMS.get();
        for (int block = 0; block < blocks; block++) {
            int blockOffset = block * blockBytes, first = block * blockSize;
            int count = Math.min(blockSize, capacity - first);
            int scales = blockOffset + body, offsets = scales + 2 * blockSize, landmarks = offsets + 2 * blockSize;
            for (int lane = 0; lane < count; lane++) {
                if (codes[landmarks + lane] != 0) throw new IllegalStateException("Invalid FusedASH landmark: "
                        + (codes[landmarks + lane] & 0xFF));
            }
            for (int group = 0; group < count; group += lanes) {
                // Unit 0 seeds the accumulator from its first product, never from zero()
                int src = blockOffset + group * bits;
                var bytes = src + loadBytes <= codes.length ? ByteVector.fromArray(ASH_WORD_BYTES, codes, src)
                        : ByteVector.fromArray(ASH_WORD_BYTES, codes, src, ASH_WORD_BYTES.indexInRange(src, codes.length));
                var words = bytes.rearrange(gather).reinterpretAsLongs();
                var acc = words.lanewise(VectorOperators.AND, mask).reinterpretAsInts()
                        .lanewise(VectorOperators.MUL, IntVector.broadcast(ASH_WORD_INTS, digits[0]));
                for (int k = 1; k < 8; k++) {
                    var field = words.lanewise(VectorOperators.LSHR, k * bits).lanewise(VectorOperators.AND, mask).reinterpretAsInts();
                    acc = acc.lanewise(VectorOperators.ADD, field.lanewise(VectorOperators.MUL, IntVector.broadcast(ASH_WORD_INTS, digits[k])));
                }
                for (int unit = 1; unit < units; unit++) {
                    src += rowBytes;
                    bytes = src + loadBytes <= codes.length ? ByteVector.fromArray(ASH_WORD_BYTES, codes, src)
                            : ByteVector.fromArray(ASH_WORD_BYTES, codes, src, ASH_WORD_BYTES.indexInRange(src, codes.length));
                    words = bytes.rearrange(gather).reinterpretAsLongs();
                    for (int k = 0; k < 8; k++) {
                        var field = words.lanewise(VectorOperators.LSHR, k * bits).lanewise(VectorOperators.AND, mask).reinterpretAsInts();
                        acc = acc.lanewise(VectorOperators.ADD, field.lanewise(VectorOperators.MUL,
                                IntVector.broadcast(ASH_WORD_INTS, digits[unit * 8 + k])));
                    }
                }
                acc.intoArray(sums, 0);
                for (int l = 0, lane = group; l < lanes && lane < count; l++, lane++) {
                    float dot = ((float) sums[2 * l] - center) * invScale;
                    int s = scales + 2 * lane, o = offsets + 2 * lane;
                    float scale = Float.float16ToFloat((short) (((codes[s] & 0xFF) << 8) | (codes[s + 1] & 0xFF)));
                    float shift = Float.float16ToFloat((short) (((codes[o] & 0xFF) << 8) | (codes[o + 1] & 0xFF)));
                    out[first + lane] = Math.fma(scale, dot, landmarkDot) + shift;
                }
            }
        }
    }

    /**
     * Per-lane estimates {@code fma(scale, dot, landmarkDot) + offset} of a whole field-block neighborhood, with
     * {@code dot = (<q, fields> - center) * invScale}: the native kernel's arithmetic, in one body compiled on its
     * own. Each lane's integer sum is exact, as in {@link #ashOffsetBinaryInt8Dot}.
     */
    private static void ashInt8NeighborhoodEstimates(byte[] codes, int bits, int units, int blockSize, int capacity,
                                                     int[] digits, float invScale, float center, float landmarkDot,
                                                     float[] out) {
        if (bits < 3 || bits > 9 || bits == 4)
            throw new IllegalArgumentException("Offset-binary codes have 3 or 5 to 9 bits per dimension, got " + bits);
        if (blockSize != 8 && blockSize != 16 && blockSize != 32)
            throw new IllegalArgumentException("blockSize must be 8, 16 or 32, got " + blockSize);
        if (capacity <= 0) throw new IllegalArgumentException("capacity must be positive");
        int unitStride = blockSize * bits;
        // Headers per lane: a big-endian half-float scale and offset, then a landmark byte. Sizes in long, so an
        // impossible shape reports the array that is too small.
        long bodyBytes = (long) units * unitStride;
        int blocks = (capacity - 1) / blockSize + 1;
        Objects.checkFromIndexSize(0L, blocks * (bodyBytes + 5L * blockSize), codes.length);
        int body = (int) bodyBytes, blockBytes = body + 5 * blockSize;
        Objects.checkFromIndexSize(0, capacity, out.length);
        if (bits <= 8 && ASH_WORD_LANES >= 2) {
            ashInt8NeighborhoodWords(codes, bits, units, blockSize, capacity, digits, invScale, center, landmarkDot, out,
                    body, blockBytes, blocks);
            return;
        }
        // Nine-bit nodes span nine bytes, more than a 64-bit word holds; they keep one node per pass.
        var windows = ASH_UNIT_SHUFFLES[bits];
        var shifts = (IntVector) ASH_UNIT_INTS_CLASS.cast(IntVector.fromArray(ASH_UNIT_INTS, ASH_UNIT_SHIFTS[bits], 0));
        int mask = (1 << bits) - 1;
        for (int block = 0; block < blocks; block++) {
            int blockOffset = block * blockBytes, first = block * blockSize;
            int lanes = Math.min(blockSize, capacity - first);
            int scales = blockOffset + body, offsets = scales + 2 * blockSize, landmarks = offsets + 2 * blockSize;
            for (int lane = 0; lane < lanes; lane++) {
                if (codes[landmarks + lane] != 0) throw new IllegalStateException("Invalid FusedASH landmark: "
                        + (codes[landmarks + lane] & 0xFF));
                int offset = blockOffset + lane * bits;
                IntVector acc = (IntVector) ASH_UNIT_INTS_CLASS.cast(IntVector.zero(ASH_UNIT_INTS));
                int inPlace = Math.max(0, Math.min(units, Math.floorDiv(codes.length - offset - 32, unitStride) + 1));
                for (int unit = 0, at = offset; unit < units; unit++, at += unitStride) {
                    byte[] source = codes;
                    int from = at;
                    if (unit >= inPlace) {
                        source = ASH_UNIT_TAIL.get();
                        System.arraycopy(codes, at, source, 0, bits);
                        from = 0;
                    }
                    var fields = (IntVector) ASH_UNIT_INTS_CLASS.cast(ByteVector.fromArray(ASH_UNIT_BYTES, source, from).rearrange(windows)
                            .reinterpretAsInts().lanewise(VectorOperators.LSHR, shifts).lanewise(VectorOperators.AND, mask));
                    var weights = (IntVector) ASH_UNIT_INTS_CLASS.cast(IntVector.fromArray(ASH_UNIT_INTS, digits, unit * 8));
                    acc = (IntVector) ASH_UNIT_INTS_CLASS.cast(acc.lanewise(VectorOperators.ADD, fields.lanewise(VectorOperators.MUL, weights)));
                }
                float dot = ((float) acc.reduceLanes(VectorOperators.ADD) - center) * invScale;
                int s = scales + 2 * lane, o = offsets + 2 * lane;
                float scale = Float.float16ToFloat((short) (((codes[s] & 0xFF) << 8) | (codes[s + 1] & 0xFF)));
                float shift = Float.float16ToFloat((short) (((codes[o] & 0xFF) << 8) | (codes[o + 1] & 0xFF)));
                out[first + lane] = Math.fma(scale, dot, landmarkDot) + shift;
            }
        }
    }

    /** {@code code[offset, offset + bytes)} at the start of this thread's buffer, followed by {@code load} or more bytes. */
    private static byte[] ashPaddedCopy(byte[] code, int offset, int bytes, int load) {
        byte[][] holder = ASH_OB_PADDED.get();
        byte[] buffer = holder[0];
        if (buffer.length < bytes + load) holder[0] = buffer = new byte[bytes + load];
        System.arraycopy(code, offset, buffer, 0, bytes);
        return buffer;
    }

    private static VectorSpecies<Byte> ashObLoad(int bits) {
        return (ASH_FLOATS.length() - 1) * bits / 8 + 1 < ASH_OB_LOAD.length() ? ASH_OB_LOAD : ASH_OB_LOAD_WIDE;
    }

    @Override
    public boolean supportsAshOffsetBinaryDot() { return ASH_SIMD && ASH_OB_SIMD && PanamaASHKernels.supported(); }

    @Override
    public float ashOffsetBinaryDot(float[] query, byte[] code, int codeOffset, int dimensions, int bits) {
        if (bits < 3 || bits > 9 || bits == 4) {
            throw new IllegalArgumentException("Offset-binary codes have 3 or 5 to 9 bits per dimension, got " + bits);
        }
        Objects.checkFromIndexSize(0, dimensions, query.length);
        int codeBytes = (int) (((long) dimensions * bits + 7) / 8);
        Objects.checkFromIndexSize(codeOffset, codeBytes, code.length);
        if (ASH_OB_VECTORS != 1) return ashOffsetBinaryDot2(query, code, codeOffset, dimensions, bits, codeBytes);
        return ashObLoad(bits) == ASH_OB_LOAD
                ? ashOffsetBinaryDot1(query, code, codeOffset, dimensions, bits, codeBytes)
                : ashOffsetBinaryDot1Wide(query, code, codeOffset, dimensions, bits, codeBytes);
    }

    /** One vector of fields per step: 8 lanes (AVX2) or 16 (AVX-512). */
    private static float ashOffsetBinaryDot1(float[] query, byte[] code, int codeOffset, int dimensions, int bits,
                                             int codeBytes) {
        int lanes = ASH_FLOATS.length(), load = ASH_OB_LOAD.length(), stepBytes = lanes * bits / 8;
        var windows = ASH_OB_SHUFFLES[bits];
        var shifts = IntVector.fromArray(ASH_INTS, ASH_OB_SHIFTS[bits], 0);
        int mask = (1 << bits) - 1;
        float center = mask * 0.5f;
        int steps = dimensions / lanes;
        // A code whose last load would pass the end of its array is copied once into a padded buffer, so every step
        // loads in place and the loop is the same wherever the code is.
        if (steps > 0 && codeOffset + (steps - 1) * stepBytes + load > code.length) {
            code = ashPaddedCopy(code, codeOffset, codeBytes, load);
            codeOffset = 0;
        }
        FloatVector a0 = FloatVector.zero(ASH_FLOATS), a1 = FloatVector.zero(ASH_FLOATS);
        int at = codeOffset;
        for (int step = 0; step < steps; step++, at += stepBytes) {
            var w0 = (ByteVector) ByteVector.fromArray(ASH_OB_LOAD, code, at).reinterpretShape(ASH_OB_BYTES, 0);
            var f0 = ((IntVector) w0.rearrange(windows).reinterpretAsInts())
                    .lanewise(VectorOperators.LSHR, shifts).lanewise(VectorOperators.AND, mask);
            var v0 = ((FloatVector) f0.convert(VectorOperators.I2F, 0)).lanewise(VectorOperators.SUB, center);
            // Even and odd steps keep separate sums in a fixed order: the accumulators trade places every step.
            // One copy of the decode, as in ashOffsetBinaryInt8Dot.
            var next = FloatVector.fromArray(ASH_FLOATS, query, step * lanes).lanewise(VectorOperators.FMA, v0, a0);
            a0 = a1;
            a1 = next;
        }
        // Fixed-order lane sum: the same score whether or not C2 has compiled this thread's copy yet.
        // Fixed-order lane sum, as in the note above ROTATE_32.
        var laneSum = a0.lanewise(VectorOperators.ADD, a1);
        if (ROTATE_32 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_32));
        if (ROTATE_16 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_16));
        if (ROTATE_8 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_8));
        if (ROTATE_4 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_4));
        if (ROTATE_2 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_2));
        laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_1));
        float sum = laneSum.lane(0);
        return sum + ashOffsetBinaryTail(query, code, codeOffset, steps * lanes, dimensions, bits);
    }

    /** As {@link #ashOffsetBinaryDot1} with 32-byte loads: 8 and 9 bits with 16 lanes. One method per load shape. */
    private static float ashOffsetBinaryDot1Wide(float[] query, byte[] code, int codeOffset, int dimensions, int bits,
                                             int codeBytes) {
        int lanes = ASH_FLOATS.length(), load = ASH_OB_LOAD_WIDE.length(), stepBytes = lanes * bits / 8;
        var windows = ASH_OB_SHUFFLES[bits];
        var shifts = IntVector.fromArray(ASH_INTS, ASH_OB_SHIFTS[bits], 0);
        int mask = (1 << bits) - 1;
        float center = mask * 0.5f;
        int steps = dimensions / lanes;
        // Padded copy, as in ashOffsetBinaryDot1
        if (steps > 0 && codeOffset + (steps - 1) * stepBytes + load > code.length) {
            code = ashPaddedCopy(code, codeOffset, codeBytes, load);
            codeOffset = 0;
        }
        FloatVector a0 = FloatVector.zero(ASH_FLOATS), a1 = FloatVector.zero(ASH_FLOATS);
        int at = codeOffset;
        for (int step = 0; step < steps; step++, at += stepBytes) {
            var w0 = (ByteVector) ByteVector.fromArray(ASH_OB_LOAD_WIDE, code, at).reinterpretShape(ASH_OB_BYTES, 0);
            var f0 = ((IntVector) w0.rearrange(windows).reinterpretAsInts())
                    .lanewise(VectorOperators.LSHR, shifts).lanewise(VectorOperators.AND, mask);
            var v0 = ((FloatVector) f0.convert(VectorOperators.I2F, 0)).lanewise(VectorOperators.SUB, center);
            // Separate even and odd sums, as in ashOffsetBinaryDot1
            var next = FloatVector.fromArray(ASH_FLOATS, query, step * lanes).lanewise(VectorOperators.FMA, v0, a0);
            a0 = a1;
            a1 = next;
        }
        // Fixed-order lane sum: the same score whether or not C2 has compiled this thread's copy yet.
        // Fixed-order lane sum, as in the note above ROTATE_32.
        var laneSum = a0.lanewise(VectorOperators.ADD, a1);
        if (ROTATE_32 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_32));
        if (ROTATE_16 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_16));
        if (ROTATE_8 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_8));
        if (ROTATE_4 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_4));
        if (ROTATE_2 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_2));
        laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_1));
        float sum = laneSum.lane(0);
        return sum + ashOffsetBinaryTail(query, code, codeOffset, steps * lanes, dimensions, bits);
    }

    /** Two vectors of fields per eight-field step: 4 lanes (NEON). */
    private static float ashOffsetBinaryDot2(float[] query, byte[] code, int codeOffset, int dimensions, int bits,
                                             int codeBytes) {
        int lanes = ASH_FLOATS.length(), load = ASH_OB_BYTES.length(), stepBytes = bits;
        var windows0 = ASH_OB_SHUFFLES[bits];
        var windows1 = ASH_OB_SHUFFLES_HI[bits];
        var shifts0 = IntVector.fromArray(ASH_INTS, ASH_OB_SHIFTS[bits], 0);
        var shifts1 = IntVector.fromArray(ASH_INTS, ASH_OB_SHIFTS[bits], lanes);
        int mask = (1 << bits) - 1;
        float center = mask * 0.5f;
        int steps = dimensions / 8;
        // Padded copy, as in ashOffsetBinaryDot1
        if (steps > 0 && codeOffset + (steps - 1) * stepBytes + load > code.length) {
            code = ashPaddedCopy(code, codeOffset, codeBytes, load);
            codeOffset = 0;
        }
        FloatVector a0 = FloatVector.zero(ASH_FLOATS), a1 = FloatVector.zero(ASH_FLOATS);
        int at = codeOffset;
        for (int step = 0; step < steps; step++, at += stepBytes) {
            var window = ByteVector.fromArray(ASH_OB_BYTES, code, at);
            var f0 = ((IntVector) window.rearrange(windows0).reinterpretAsInts())
                    .lanewise(VectorOperators.LSHR, shifts0).lanewise(VectorOperators.AND, mask);
            var f1 = ((IntVector) window.rearrange(windows1).reinterpretAsInts())
                    .lanewise(VectorOperators.LSHR, shifts1).lanewise(VectorOperators.AND, mask);
            var v0 = ((FloatVector) f0.convert(VectorOperators.I2F, 0)).lanewise(VectorOperators.SUB, center);
            var v1 = ((FloatVector) f1.convert(VectorOperators.I2F, 0)).lanewise(VectorOperators.SUB, center);
            a0 = FloatVector.fromArray(ASH_FLOATS, query, step * 8).lanewise(VectorOperators.FMA, v0, a0);
            a1 = FloatVector.fromArray(ASH_FLOATS, query, step * 8 + lanes).lanewise(VectorOperators.FMA, v1, a1);
        }
        // Fixed-order lane sum, as in the note above ROTATE_32.
        var laneSum = a0.lanewise(VectorOperators.ADD, a1);
        if (ROTATE_32 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_32));
        if (ROTATE_16 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_16));
        if (ROTATE_8 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_8));
        if (ROTATE_4 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_4));
        if (ROTATE_2 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_2));
        laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_1));
        float sum = laneSum.lane(0);
        return sum + ashOffsetBinaryTail(query, code, codeOffset, steps * 8, dimensions, bits);
    }

    /**
     * Sign-magnitude 2 and 4-bit codes in registers, the query in its own order: no gathers and no prepared query.
     * Field {@code f} is {@code (2 * (f & m) + 1) / 2} with {@code m = 2^(b-1) - 1}, negative unless its top bit is set;
     * lanes accumulate the odd integers {@code 2v} and the sum is halved once.
     */
    private static float ashSignMagnitudeDot(float[] query, byte[] code, int codeOffset, int dimensions, int bits) {
        Objects.checkFromIndexSize(0, dimensions, query.length);
        int codeBytes = (int) (((long) dimensions * bits + 7) / 8);
        Objects.checkFromIndexSize(codeOffset, codeBytes, code.length);
        return ASH_OB_VECTORS == 1 ? ashSignMagnitudeDot1(query, code, codeOffset, dimensions, bits, codeBytes)
                : ashSignMagnitudeDot2(query, code, codeOffset, dimensions, bits, codeBytes);
    }

    /** One vector of fields per step: 8 lanes (AVX2) or 16 (AVX-512). */
    private static float ashSignMagnitudeDot1(float[] query, byte[] code, int codeOffset, int dimensions, int bits,
                                              int codeBytes) {
        int lanes = ASH_FLOATS.length(), load = ASH_OB_LOAD.length(), stepBytes = lanes * bits / 8;
        var windows = ASH_OB_SHUFFLES[bits];
        var shifts = IntVector.fromArray(ASH_INTS, ASH_OB_SHIFTS[bits], 0);
        int magnitude = (1 << (bits - 1)) - 1, sign = bits - 1;
        int steps = dimensions / lanes;
        // Padded copy, as in ashOffsetBinaryDot1
        if (steps > 0 && codeOffset + (steps - 1) * stepBytes + load > code.length) {
            code = ashPaddedCopy(code, codeOffset, codeBytes, load);
            codeOffset = 0;
        }
        FloatVector a0 = FloatVector.zero(ASH_FLOATS), a1 = FloatVector.zero(ASH_FLOATS);
        int at = codeOffset;
        for (int step = 0; step < steps; step++, at += stepBytes) {
            var w0 = (ByteVector) ByteVector.fromArray(ASH_OB_LOAD, code, at).reinterpretShape(ASH_OB_BYTES, 0);
            var f0 = ((IntVector) w0.rearrange(windows).reinterpretAsInts())
                    .lanewise(VectorOperators.LSHR, shifts);
            // t = 0 for a positive field, -1 for a negative one; (m ^ t) - t negates m when t = -1.
            var t0 = f0.lanewise(VectorOperators.LSHR, sign).lanewise(VectorOperators.AND, 1).lanewise(VectorOperators.SUB, 1);
            var m0 = f0.lanewise(VectorOperators.AND, magnitude).lanewise(VectorOperators.LSHL, 1).lanewise(VectorOperators.ADD, 1);
            var v0 = (FloatVector) m0.lanewise(VectorOperators.XOR, t0).lanewise(VectorOperators.SUB, t0).convert(VectorOperators.I2F, 0);
            // Separate even and odd sums, as in ashOffsetBinaryDot1
            var next = FloatVector.fromArray(ASH_FLOATS, query, step * lanes).lanewise(VectorOperators.FMA, v0, a0);
            a0 = a1;
            a1 = next;
        }
        // Fixed-order lane sum: the same score whether or not C2 has compiled this thread's copy yet.
        // Fixed-order lane sum, as in the note above ROTATE_32.
        var laneSum = a0.lanewise(VectorOperators.ADD, a1);
        if (ROTATE_32 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_32));
        if (ROTATE_16 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_16));
        if (ROTATE_8 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_8));
        if (ROTATE_4 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_4));
        if (ROTATE_2 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_2));
        laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_1));
        float sum = laneSum.lane(0);
        return (sum + ashSignMagnitudeTail(query, code, codeOffset, steps * lanes, dimensions, bits)) * 0.5f;
    }

    /** Two vectors of fields per eight-field step: 4 lanes (NEON). */
    private static float ashSignMagnitudeDot2(float[] query, byte[] code, int codeOffset, int dimensions, int bits,
                                              int codeBytes) {
        int lanes = ASH_FLOATS.length(), load = ASH_OB_BYTES.length(), stepBytes = bits;
        var windows0 = ASH_OB_SHUFFLES[bits];
        var windows1 = ASH_OB_SHUFFLES_HI[bits];
        var shifts0 = IntVector.fromArray(ASH_INTS, ASH_OB_SHIFTS[bits], 0);
        var shifts1 = IntVector.fromArray(ASH_INTS, ASH_OB_SHIFTS[bits], lanes);
        int magnitude = (1 << (bits - 1)) - 1, sign = bits - 1;
        int steps = dimensions / 8;
        // Padded copy, as in ashOffsetBinaryDot1
        if (steps > 0 && codeOffset + (steps - 1) * stepBytes + load > code.length) {
            code = ashPaddedCopy(code, codeOffset, codeBytes, load);
            codeOffset = 0;
        }
        FloatVector a0 = FloatVector.zero(ASH_FLOATS), a1 = FloatVector.zero(ASH_FLOATS);
        int at = codeOffset;
        for (int step = 0; step < steps; step++, at += stepBytes) {
            var window = ByteVector.fromArray(ASH_OB_BYTES, code, at);
            var f0 = ((IntVector) window.rearrange(windows0).reinterpretAsInts())
                    .lanewise(VectorOperators.LSHR, shifts0);
            var f1 = ((IntVector) window.rearrange(windows1).reinterpretAsInts())
                    .lanewise(VectorOperators.LSHR, shifts1);
            var t0 = f0.lanewise(VectorOperators.LSHR, sign).lanewise(VectorOperators.AND, 1).lanewise(VectorOperators.SUB, 1);
            var t1 = f1.lanewise(VectorOperators.LSHR, sign).lanewise(VectorOperators.AND, 1).lanewise(VectorOperators.SUB, 1);
            var m0 = f0.lanewise(VectorOperators.AND, magnitude).lanewise(VectorOperators.LSHL, 1).lanewise(VectorOperators.ADD, 1);
            var m1 = f1.lanewise(VectorOperators.AND, magnitude).lanewise(VectorOperators.LSHL, 1).lanewise(VectorOperators.ADD, 1);
            var v0 = (FloatVector) m0.lanewise(VectorOperators.XOR, t0).lanewise(VectorOperators.SUB, t0).convert(VectorOperators.I2F, 0);
            var v1 = (FloatVector) m1.lanewise(VectorOperators.XOR, t1).lanewise(VectorOperators.SUB, t1).convert(VectorOperators.I2F, 0);
            a0 = FloatVector.fromArray(ASH_FLOATS, query, step * 8).lanewise(VectorOperators.FMA, v0, a0);
            a1 = FloatVector.fromArray(ASH_FLOATS, query, step * 8 + lanes).lanewise(VectorOperators.FMA, v1, a1);
        }
        // Fixed-order lane sum, as in the note above ROTATE_32.
        var laneSum = a0.lanewise(VectorOperators.ADD, a1);
        if (ROTATE_32 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_32));
        if (ROTATE_16 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_16));
        if (ROTATE_8 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_8));
        if (ROTATE_4 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_4));
        if (ROTATE_2 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_2));
        laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_1));
        float sum = laneSum.lane(0);
        return (sum + ashSignMagnitudeTail(query, code, codeOffset, steps * 8, dimensions, bits)) * 0.5f;
    }

    /** Fields {@code [from, dimensions)} as odd integers {@code 2v}, in order. */
    private static float ashSignMagnitudeTail(float[] query, byte[] code, int codeOffset, int from, int dimensions,
                                              int bits) {
        int magnitude = (1 << (bits - 1)) - 1;
        float sum = 0f;
        for (int j = from; j < dimensions; j++) {
            int bit = j * bits;
            int field = ((code[codeOffset + (bit >>> 3)] & 0xFF) >>> (bit & 7)) & ((1 << bits) - 1);
            int v = 2 * (field & magnitude) + 1;
            sum = Math.fma(query[j], (field >>> (bits - 1)) == 0 ? -v : v, sum);
        }
        return sum;
    }

    /** Fields {@code [from, dimensions)}, in order. */
    private static float ashOffsetBinaryTail(float[] query, byte[] code, int codeOffset, int from, int dimensions,
                                             int bits) {
        int mask = (1 << bits) - 1;
        float center = mask * 0.5f, sum = 0f;
        for (int j = from; j < dimensions; j++) {
            int bit = j * bits, at = codeOffset + (bit >>> 3), shift = bit & 7;
            int window = code[at] & 0xFF;
            if (shift + bits > 8) window |= (code[at + 1] & 0xFF) << 8;
            if (shift + bits > 16) window |= (code[at + 2] & 0xFF) << 16;
            sum = Math.fma(query[j], ((window >>> shift) & mask) - center, sum);
        }
        return sum;
    }

    @Override
    public String ashKernelDescription() { return PanamaASHKernels.description(); }

    @Override
    public boolean usesAshProjectionTuning() { return PanamaASHKernels.projectionTuningEnabled(); }

    @Override
    public float ashProjectionDotTuned(float[] query, byte[] code, int codeOffset, int dimensions, int bitsPerDimension) {
        if (!supportsAshProjectionScoring()) {
            return VectorUtilSupport.super.ashProjectionDot(query, code, codeOffset, dimensions, bitsPerDimension);
        }
        return PanamaASHKernels.projectionDot(query, code, codeOffset, dimensions, bitsPerDimension);
    }

    @Override
    public float ashProjectionDot(float[] query, byte[] code, int codeOffset, int dimensions, int bitsPerDimension) {
        if (!ASH_OB_SIMD || !supportsAshProjectionScoring()) {
            return VectorUtilSupport.super.ashProjectionDot(query, code, codeOffset, dimensions, bitsPerDimension);
        }
        if (bitsPerDimension != 2 && bitsPerDimension != 4) {
            throw new IllegalArgumentException("Projection scoring requires 2 or 4 bits per dimension");
        }
        // Decoded in registers: no gathers, which are slow on some CPUs (Ice Lake), and no prepared query.
        // Exception to the native design: native decodes each field slot into its own lane and de-interleaves the
        // query with an interleaved load, which the Vector API lacks. Fields decode in dimension order against the
        // query as stored instead.
        return ashSignMagnitudeDot(query, code, codeOffset, dimensions, bitsPerDimension);
    }

    /**
     * The query in slot-major blocks: for each block of {@code lanes} code bytes, the query values of slot 0 of every
     * byte, then slot 1, and so on, zero-padded to whole blocks. {@link #ashProjectionDotPrepared} then loads each
     * slot's query values contiguously instead of gathering them, which is slow on some CPUs (Ice Lake).
     */
    @Override
    public float[] ashPrepareProjectionQuery(float[] query, int dimensions, int bitsPerDimension) {
        if (!ASH_PREPARED_PROJECTION || !supportsAshProjectionScoring()
                || (bitsPerDimension != 2 && bitsPerDimension != 4)) return null;
        Objects.checkFromIndexSize(0, dimensions, query.length);
        int perByte = 8 / bitsPerDimension, lanes = ASH_FLOATS.length(), blockDims = lanes * perByte;
        float[] prepared = new float[Math.multiplyExact((dimensions + blockDims - 1) / blockDims, blockDims)];
        for (int i = 0; i < dimensions; i++) {
            int block = i / blockDims, r = i - block * blockDims;
            prepared[block * blockDims + (r % perByte) * lanes + r / perByte] = query[i];
        }
        return prepared;
    }

    @Override
    public float ashProjectionDotPrepared(float[] prepared, byte[] code, int codeOffset, int dimensions,
                                          int bitsPerDimension) {
        if (bitsPerDimension != 2 && bitsPerDimension != 4) {
            throw new IllegalArgumentException("Projection scoring requires 2 or 4 bits per dimension");
        }
        int perByte = 8 / bitsPerDimension, blockDims = ASH_FLOATS.length() * perByte;
        Objects.checkFromIndexSize(codeOffset, (dimensions + perByte - 1) / perByte, code.length);
        int blocks = (dimensions + blockDims - 1) / blockDims;
        Objects.checkFromIndexSize(0, blocks * blockDims, prepared.length);
        return bitsPerDimension == 2 ? ashPreparedDot2(prepared, code, codeOffset, dimensions, blocks)
                : ashPreparedDot4(prepared, code, codeOffset, dimensions, blocks);
    }

    // One method per width with no vector-valued helpers, every slot decoded before any is accumulated.
    private static float ashPreparedDot2(float[] prepared, byte[] code, int codeOffset, int dimensions, int blocks) {
        int lanes = ASH_FLOATS.length(), loadBytes = ASH_BYTES.length();
        // A code whose last load would pass the end of its array is copied once into a padded buffer, so every block
        // loads in place and the score does not depend on where the code sits.
        if (blocks > 0 && codeOffset + (blocks - 1) * lanes + loadBytes > code.length) {
            code = ashPaddedCopy(code, codeOffset, (dimensions + 3) / 4, loadBytes);
            codeOffset = 0;
        }
        FloatVector a0 = FloatVector.zero(ASH_FLOATS), a1 = FloatVector.zero(ASH_FLOATS);
        FloatVector a2 = FloatVector.zero(ASH_FLOATS), a3 = FloatVector.zero(ASH_FLOATS);
        for (int block = 0, q = 0; block < blocks; block++, q += 4 * lanes) {
            var packed = (IntVector) ByteVector.fromArray(ASH_BYTES, code, codeOffset + block * lanes)
                    .convertShape(VectorOperators.B2I, ASH_INTS, 0);
            var f0 = packed;
            var f2 = packed.lanewise(VectorOperators.LSHR, 2);
            var f4 = packed.lanewise(VectorOperators.LSHR, 4);
            var f6 = packed.lanewise(VectorOperators.LSHR, 6);
            var m0 = ((FloatVector) f0.lanewise(VectorOperators.AND, 1).convertShape(VectorOperators.I2F, ASH_FLOATS, 0)).lanewise(VectorOperators.ADD, 0.5f);
            var m2 = ((FloatVector) f2.lanewise(VectorOperators.AND, 1).convertShape(VectorOperators.I2F, ASH_FLOATS, 0)).lanewise(VectorOperators.ADD, 0.5f);
            var m4 = ((FloatVector) f4.lanewise(VectorOperators.AND, 1).convertShape(VectorOperators.I2F, ASH_FLOATS, 0)).lanewise(VectorOperators.ADD, 0.5f);
            var m6 = ((FloatVector) f6.lanewise(VectorOperators.AND, 1).convertShape(VectorOperators.I2F, ASH_FLOATS, 0)).lanewise(VectorOperators.ADD, 0.5f);
            a0 = a0.lanewise(VectorOperators.ADD, FloatVector.fromArray(ASH_FLOATS, prepared, q)
                    .mul(m0.blend(m0.lanewise(VectorOperators.NEG), f0.lanewise(VectorOperators.AND, 2).compare(VectorOperators.EQ, 0).cast(ASH_FLOATS))));
            a1 = a1.lanewise(VectorOperators.ADD, FloatVector.fromArray(ASH_FLOATS, prepared, q + lanes)
                    .mul(m2.blend(m2.lanewise(VectorOperators.NEG), f2.lanewise(VectorOperators.AND, 2).compare(VectorOperators.EQ, 0).cast(ASH_FLOATS))));
            a2 = a2.lanewise(VectorOperators.ADD, FloatVector.fromArray(ASH_FLOATS, prepared, q + 2 * lanes)
                    .mul(m4.blend(m4.lanewise(VectorOperators.NEG), f4.lanewise(VectorOperators.AND, 2).compare(VectorOperators.EQ, 0).cast(ASH_FLOATS))));
            a3 = a3.lanewise(VectorOperators.ADD, FloatVector.fromArray(ASH_FLOATS, prepared, q + 3 * lanes)
                    .mul(m6.blend(m6.lanewise(VectorOperators.NEG), f6.lanewise(VectorOperators.AND, 2).compare(VectorOperators.EQ, 0).cast(ASH_FLOATS))));
        }
        // Fixed-order lane sum: the same score whether or not C2 has compiled this thread's copy yet.
        // Fixed-order lane sum, as in the note above ROTATE_32.
        var laneSum = a0.lanewise(VectorOperators.ADD, a1).lanewise(VectorOperators.ADD, a2.lanewise(VectorOperators.ADD, a3));
        if (ROTATE_32 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_32));
        if (ROTATE_16 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_16));
        if (ROTATE_8 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_8));
        if (ROTATE_4 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_4));
        if (ROTATE_2 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_2));
        laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_1));
        float sum = laneSum.lane(0);
        return sum;
    }

    // One method per width with no vector-valued helpers, every slot decoded before any is accumulated.
    private static float ashPreparedDot4(float[] prepared, byte[] code, int codeOffset, int dimensions, int blocks) {
        int lanes = ASH_FLOATS.length(), loadBytes = ASH_BYTES.length();
        // A code whose last load would pass the end of its array is copied once into a padded buffer, so every block
        // loads in place and the score does not depend on where the code sits.
        if (blocks > 0 && codeOffset + (blocks - 1) * lanes + loadBytes > code.length) {
            code = ashPaddedCopy(code, codeOffset, (dimensions + 1) / 2, loadBytes);
            codeOffset = 0;
        }
        FloatVector a0 = FloatVector.zero(ASH_FLOATS), a1 = FloatVector.zero(ASH_FLOATS);
        for (int block = 0, q = 0; block < blocks; block++, q += 2 * lanes) {
            var packed = (IntVector) ByteVector.fromArray(ASH_BYTES, code, codeOffset + block * lanes)
                    .convertShape(VectorOperators.B2I, ASH_INTS, 0);
            var f0 = packed;
            var f4 = packed.lanewise(VectorOperators.LSHR, 4);
            var m0 = ((FloatVector) f0.lanewise(VectorOperators.AND, 7).convertShape(VectorOperators.I2F, ASH_FLOATS, 0)).lanewise(VectorOperators.ADD, 0.5f);
            var m4 = ((FloatVector) f4.lanewise(VectorOperators.AND, 7).convertShape(VectorOperators.I2F, ASH_FLOATS, 0)).lanewise(VectorOperators.ADD, 0.5f);
            a0 = a0.lanewise(VectorOperators.ADD, FloatVector.fromArray(ASH_FLOATS, prepared, q)
                    .mul(m0.blend(m0.lanewise(VectorOperators.NEG), f0.lanewise(VectorOperators.AND, 8).compare(VectorOperators.EQ, 0).cast(ASH_FLOATS))));
            a1 = a1.lanewise(VectorOperators.ADD, FloatVector.fromArray(ASH_FLOATS, prepared, q + lanes)
                    .mul(m4.blend(m4.lanewise(VectorOperators.NEG), f4.lanewise(VectorOperators.AND, 8).compare(VectorOperators.EQ, 0).cast(ASH_FLOATS))));
        }
        // Fixed-order lane sum: the same score whether or not C2 has compiled this thread's copy yet.
        // Fixed-order lane sum, as in the note above ROTATE_32.
        var laneSum = a0.lanewise(VectorOperators.ADD, a1);
        if (ROTATE_32 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_32));
        if (ROTATE_16 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_16));
        if (ROTATE_8 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_8));
        if (ROTATE_4 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_4));
        if (ROTATE_2 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_2));
        laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_1));
        float sum = laneSum.lane(0);
        return sum;
    }

    @Override
    public void ashLutScore(byte[] codes, int offset, int groups, int stride,
                            int lane, int count, float[] lut, float[] out, int outOffset) {
        if (!supportsAshLutScoring()) {
            VectorUtilSupport.super.ashLutScore(codes, offset, groups, stride, lane, count, lut, out, outOffset);
            return;
        }
        PanamaASHKernels.lutScore(codes, offset, groups, stride, lane, count, lut, out, outOffset);
    }

    static final int PREFERRED_BIT_SIZE = FloatVector.SPECIES_PREFERRED.vectorBitSize();
    static {
        // Publish the value to System Properties for other packages (other language levels...)
        System.setProperty("io.github.jbellis.jvector.vector_bit_size", String.valueOf(PREFERRED_BIT_SIZE));
    }
    static final IntVector BYTE_TO_INT_MASK_512 = IntVector.broadcast(IntVector.SPECIES_512, 0xff);
    static final IntVector BYTE_TO_INT_MASK_256 = IntVector.broadcast(IntVector.SPECIES_256, 0xff);

    static final ThreadLocal<int[]> scratchInt512 = ThreadLocal.withInitial(() -> new int[IntVector.SPECIES_512.length()]);
    static final ThreadLocal<int[]> scratchInt256 = ThreadLocal.withInitial(() -> new int[IntVector.SPECIES_256.length()]);

    protected FloatVector fromVectorFloat(VectorSpecies<Float> SPEC, VectorFloat<?> vector, int offset) {
        return FloatVector.fromArray(SPEC, ((ArrayVectorFloat) vector).get(), offset);
    }

    protected FloatVector fromVectorFloat(VectorSpecies<Float> SPEC, VectorFloat<?> vector, int offset, int[] indices, int indicesOffset) {
        return FloatVector.fromArray(SPEC, ((ArrayVectorFloat) vector).get(), offset, indices, indicesOffset);
    }

    protected void intoVectorFloat(FloatVector vector, VectorFloat<?> v, int offset) {
        vector.intoArray(((ArrayVectorFloat) v).get(), offset);
    }

    /// Loads from the backing array of an array-backed sequence (plain or sliced).
    /// `offset` is an index into the backing array, so callers working with a slice
    /// must include `vector.offset()`.
    protected ByteVector fromByteSequence(VectorSpecies<Byte> SPEC, ByteSequence<?> vector, int offset) {
        return ByteVector.fromArray(SPEC, (byte[]) vector.get(), offset);
    }

    protected void intoByteSequence(ByteVector vector, ByteSequence<?> v, int offset) {
        vector.intoArray(((ArrayByteSequence) v).get(), offset);
    }

    protected void intoByteSequence(ByteVector vector, ByteSequence<?> v, int offset, VectorMask<Byte> mask) {
        vector.intoArray(((ArrayByteSequence) v).get(), offset, mask);
    }

    /** The backing array of an array-backed vector (heap or the native provider's heap-backed segments). */
    private static float[] arr(VectorFloat<?> v) {
        return ((FloatArray) v).array();
    }

    @Override
    public float sum(VectorFloat<?> vector) {
        var sum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        // Process the remainder
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(vector), i);
            sum = sum.lanewise(VectorOperators.ADD, a);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < vector.length(); i++) {
            res += vector.get(i);
        }

        return res;
    }

    @Override
    public VectorFloat<?> sum(List<VectorFloat<?>> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            throw new IllegalArgumentException("Input list cannot be null or empty");
        }

        int dimension = vectors.get(0).length();
        VectorFloat<?> sum = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(dimension);

        // Process each vector from the list
        for (VectorFloat<?> vector : vectors) {
            addInPlace(sum, vector);
        }

        return sum;
    }

    @Override
    public void scale(VectorFloat<?> vector, float multiplier) {
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(vector), i);
            var divResult = a.lanewise(VectorOperators.MUL, multiplier);
            divResult.intoArray(arr(vector), i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < vector.length(); i++) {
            vector.set(i, vector.get(i) * multiplier);
        }
    }

    @Override
    public float dotProduct(VectorFloat<?> v1, VectorFloat<?> v2) {
        return dotProduct(v1, 0, v2, 0, v1.length());
    }

    // Whether a short dot product's leftover can fill a 128-bit vector (never on 128-bit SIMD; the check folds away).
    private static final boolean NARROW_TAILS =
            FloatVector.SPECIES_PREFERRED.length() > FloatVector.SPECIES_128.length();

    /**
     * Every length's path is written out in this one method (dotProductPreferred, for four or more vectors, is its
     * own): too large to inline, so always compiled on its own. As separate small methods inlined deep in a caller,
     * the vector operations were sometimes left as calls that put vectors on the heap. The arithmetic is unchanged.
     */
    @Override
    public float dotProduct(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, final int length) {
        // Exactly one vector wide (the PQ subvector size on 128-bit hardware): handle it here.
        if (length == FloatVector.SPECIES_PREFERRED.length())
            return FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v1), v1offset)
                .mul(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v2), v2offset)).reduceLanes(VectorOperators.ADD);

        //Common case first
        if (length >= 4 * FloatVector.SPECIES_PREFERRED.length())
            return dotProductPreferred(v1, v1offset, v2, v2offset, length);

        if (length > FloatVector.SPECIES_PREFERRED.length()) {
            // More than one and fewer than four preferred vectors. The first two vectors use separate sums, as in
            // dotProductPreferred, so their multiply-adds do not wait on each other.
            int vectorLength = FloatVector.SPECIES_PREFERRED.length();
            assert length > vectorLength && length < 4 * vectorLength : length;
            FloatVector sum0 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
            int i = 0;
            if (length >= 2 * vectorLength) {
                FloatVector sum1 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v1), v1offset + vectorLength)
                        .fma(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v2), v2offset + vectorLength), sum0);
                sum0 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v1), v1offset)
                        .fma(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v2), v2offset), sum0);
                sum0 = sum0.lanewise(VectorOperators.ADD, sum1);
                i = 2 * vectorLength;
            }
            if (length - i >= vectorLength) {
                sum0 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v1), v1offset + i)
                        .fma(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v2), v2offset + i), sum0);
                i += vectorLength;
            }

            float resVec = sum0.reduceLanes(VectorOperators.ADD);
            // A partial vector of four or more floats (8 of 24 on AVX-512, say) runs on narrower vectors. Shorter ones
            // stay scalar: 64-bit float vectors are slower than scalar code on some CPUs, such as 128-bit ARM.
            if (NARROW_TAILS && length - i >= FloatVector.SPECIES_128.length())
                return resVec + dotProduct(v1, v1offset + i, v2, v2offset + i, length - i);
            float resTail = 0;
            for (; i < length; i++)
                resTail += v1.get(v1offset + i) * v2.get(v2offset + i);
            return resVec + resTail;
        }

        // Fewer floats than one preferred vector, on the widest narrower shape that fits.
        if (length < FloatVector.SPECIES_128.length()) {
            if (length == FloatVector.SPECIES_64.length())
                return FloatVector.fromArray(FloatVector.SPECIES_64, arr(v1), v1offset)
                    .mul(FloatVector.fromArray(FloatVector.SPECIES_64, arr(v2), v2offset)).reduceLanes(VectorOperators.ADD);

            final int vectorizedLength = FloatVector.SPECIES_64.loopBound(length);
            FloatVector sum = FloatVector.zero(FloatVector.SPECIES_64);
            int i = 0;
            // Process the vectorized part
            for (; i < vectorizedLength; i += FloatVector.SPECIES_64.length()) {
                FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_64, arr(v1), v1offset + i);
                FloatVector b = FloatVector.fromArray(FloatVector.SPECIES_64, arr(v2), v2offset + i);
                sum = a.lanewise(VectorOperators.FMA, b, sum);
            }

            float res = sum.reduceLanes(VectorOperators.ADD);

            // Process the tail
            for (; i < length; ++i)
                res += v1.get(v1offset + i) * v2.get(v2offset + i);

            return res;
        } else if (length < FloatVector.SPECIES_256.length()) {
            if (length == FloatVector.SPECIES_128.length())
                return FloatVector.fromArray(FloatVector.SPECIES_128, arr(v1), v1offset)
                    .mul(FloatVector.fromArray(FloatVector.SPECIES_128, arr(v2), v2offset)).reduceLanes(VectorOperators.ADD);

            final int vectorizedLength = FloatVector.SPECIES_128.loopBound(length);
            FloatVector sum = FloatVector.zero(FloatVector.SPECIES_128);

            int i = 0;
            // Process the vectorized part
            for (; i < vectorizedLength; i += FloatVector.SPECIES_128.length()) {
                FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_128, arr(v1), v1offset + i);
                FloatVector b = FloatVector.fromArray(FloatVector.SPECIES_128, arr(v2), v2offset + i);
                sum = a.lanewise(VectorOperators.FMA, b, sum);
            }

            float res = sum.reduceLanes(VectorOperators.ADD);

            // Process the tail
            for (; i < length; ++i)
                res += v1.get(v1offset + i) * v2.get(v2offset + i);

            return res;
        } else {
            if (length == FloatVector.SPECIES_256.length())
                return FloatVector.fromArray(FloatVector.SPECIES_256, arr(v1), v1offset)
                    .mul(FloatVector.fromArray(FloatVector.SPECIES_256, arr(v2), v2offset)).reduceLanes(VectorOperators.ADD);

            final int vectorizedLength = FloatVector.SPECIES_256.loopBound(length);
            FloatVector sum = FloatVector.zero(FloatVector.SPECIES_256);

            int i = 0;
            // Process the vectorized part
            for (; i < vectorizedLength; i += FloatVector.SPECIES_256.length()) {
                FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_256, arr(v1), v1offset + i);
                FloatVector b = FloatVector.fromArray(FloatVector.SPECIES_256, arr(v2), v2offset + i);
                sum = a.lanewise(VectorOperators.FMA, b, sum);
            }

            float res = sum.reduceLanes(VectorOperators.ADD);

            // Process the tail
            for (; i < length; ++i)
                res += v1.get(v1offset + i) * v2.get(v2offset + i);

            return res;
        }
    }

    /** Four or more preferred vectors; {@link #dotProduct} sends shorter slices elsewhere. */
    float dotProductPreferred(VectorFloat<?> va, int vaoffset, VectorFloat<?> vb, int vboffset, int length) {
        FloatVector sum0 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        FloatVector sum1 = sum0;
        FloatVector a0, a1, b0, b1;

        int vectorLength = FloatVector.SPECIES_PREFERRED.length();

        // Unrolled vector loop; for dot product from L1 cache, an unroll factor of 2 generally suffices.
        // If we are going to be getting data that's further down the hierarchy but not fetched off disk/network,
        // we might want to unroll further, e.g. to 8 (4 sets of a,b,sum with 3-ahead reads seems to work best).
        length -= vectorLength * 2;
        a0 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(va), vaoffset + vectorLength * 0);
        b0 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(vb), vboffset + vectorLength * 0);
        a1 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(va), vaoffset + vectorLength * 1);
        b1 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(vb), vboffset + vectorLength * 1);
        vaoffset += vectorLength * 2;
        vboffset += vectorLength * 2;
        while (length >= vectorLength * 2) {
            // All instructions in the main loop have no dependencies between them and can be executed in parallel.
            length -= vectorLength * 2;
            sum0 = a0.lanewise(VectorOperators.FMA, b0, sum0);
            a0 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(va), vaoffset + vectorLength * 0);
            b0 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(vb), vboffset + vectorLength * 0);
            sum1 = a1.lanewise(VectorOperators.FMA, b1, sum1);
            a1 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(va), vaoffset + vectorLength * 1);
            b1 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(vb), vboffset + vectorLength * 1);
            vaoffset += vectorLength * 2;
            vboffset += vectorLength * 2;
        }
        sum0 = a0.lanewise(VectorOperators.FMA, b0, sum0);
        sum1 = a1.lanewise(VectorOperators.FMA, b1, sum1);
        sum0 = sum0.lanewise(VectorOperators.ADD, sum1);

        // Process the remaining few vectors
        while (length >= vectorLength) {
            length -= vectorLength;
            FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(va), vaoffset);
            FloatVector b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(vb), vboffset);
            vaoffset += vectorLength;
            vboffset += vectorLength;
            sum0 = a.lanewise(VectorOperators.FMA, b, sum0);
        }

        float resVec = sum0.reduceLanes(VectorOperators.ADD);
        float resTail = 0;

        // Process the tail
        for (; length > 0; --length)
            resTail += va.get(vaoffset++) * vb.get(vboffset++);

        return resVec + resTail;
    }

    @Override
    public float cosine(VectorFloat<?> v1, VectorFloat<?> v2) {
        if (v1.length() != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        var vsum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vaMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vbMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());
        // Process the vectorized part, convert from 8 bytes to 8 ints
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v1), i);
            var b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v2), i);
            vsum = a.lanewise(VectorOperators.FMA, b, vsum);
            vaMagnitude = a.lanewise(VectorOperators.FMA, a, vaMagnitude);
            vbMagnitude = b.lanewise(VectorOperators.FMA, b, vbMagnitude);
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);
        float aMagnitude = vaMagnitude.reduceLanes(VectorOperators.ADD);
        float bMagnitude = vbMagnitude.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            sum += v1.get(i) * v2.get(i);
            aMagnitude += v1.get(i) * v1.get(i);
            bMagnitude += v2.get(i) * v2.get(i);
        }

        return (float) (sum / Math.sqrt(aMagnitude * bMagnitude));
    }

    @Override
    public float cosine(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {
        var vsum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vaMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vbMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v1), v1offset + i);
            var b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v2), v2offset + i);
            vsum = a.lanewise(VectorOperators.FMA, b, vsum);
            vaMagnitude = a.lanewise(VectorOperators.FMA, a, vaMagnitude);
            vbMagnitude = b.lanewise(VectorOperators.FMA, b, vbMagnitude);
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);
        float aMagnitude = vaMagnitude.reduceLanes(VectorOperators.ADD);
        float bMagnitude = vbMagnitude.reduceLanes(VectorOperators.ADD);

        for (int i = vectorizedLength; i < length; i++) {
            sum += v1.get(v1offset + i) * v2.get(v2offset + i);
            aMagnitude += v1.get(v1offset + i) * v1.get(v1offset + i);
            bMagnitude += v2.get(v2offset + i) * v2.get(v2offset + i);
        }

        return (float) (sum / Math.sqrt(aMagnitude * bMagnitude));
    }

    // Bulk kernels: one query against four targets per pass. A random target is usually not in cache, and the four
    // independent load streams keep more of them in flight than four sequential calls do. Each target accumulates
    // exactly as in the single-pair method it stands for, so the results are the same; lengths that method handles
    // on a different path are scored one pair at a time.

    @Override
    public void cosine(VectorFloat<?> q, VectorFloat<?>[] targets, int count, float[] results) {
        int i = 0;
        for (; i + 4 <= count; i += 4) {
            cosine4(arr(q), arr(targets[i]), arr(targets[i + 1]), arr(targets[i + 2]), arr(targets[i + 3]), q.length(), results, i);
        }
        for (; i < count; i++) {
            results[i] = cosine(q, targets[i]);
        }
    }

    /** Four cosine(q, t) at once, accumulated as in {@link #cosine(VectorFloat, VectorFloat)}. */
    private static void cosine4(float[] q, float[] t0, float[] t1, float[] t2, float[] t3, int length, float[] results, int at) {
        var qq = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var s0 = qq; var s1 = qq; var s2 = qq; var s3 = qq;
        var n0 = qq; var n1 = qq; var n2 = qq; var n3 = qq;
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, q, i);
            var b0 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t0, i);
            var b1 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t1, i);
            var b2 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t2, i);
            var b3 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t3, i);
            qq = a.lanewise(VectorOperators.FMA, a, qq);
            s0 = a.lanewise(VectorOperators.FMA, b0, s0); n0 = b0.lanewise(VectorOperators.FMA, b0, n0);
            s1 = a.lanewise(VectorOperators.FMA, b1, s1); n1 = b1.lanewise(VectorOperators.FMA, b1, n1);
            s2 = a.lanewise(VectorOperators.FMA, b2, s2); n2 = b2.lanewise(VectorOperators.FMA, b2, n2);
            s3 = a.lanewise(VectorOperators.FMA, b3, s3); n3 = b3.lanewise(VectorOperators.FMA, b3, n3);
        }
        float aMagnitude = qq.reduceLanes(VectorOperators.ADD);
        float sum0 = s0.reduceLanes(VectorOperators.ADD), b0Magnitude = n0.reduceLanes(VectorOperators.ADD);
        float sum1 = s1.reduceLanes(VectorOperators.ADD), b1Magnitude = n1.reduceLanes(VectorOperators.ADD);
        float sum2 = s2.reduceLanes(VectorOperators.ADD), b2Magnitude = n2.reduceLanes(VectorOperators.ADD);
        float sum3 = s3.reduceLanes(VectorOperators.ADD), b3Magnitude = n3.reduceLanes(VectorOperators.ADD);
        for (int i = vectorizedLength; i < length; i++) {
            float a = q[i];
            aMagnitude += a * a;
            sum0 += a * t0[i]; b0Magnitude += t0[i] * t0[i];
            sum1 += a * t1[i]; b1Magnitude += t1[i] * t1[i];
            sum2 += a * t2[i]; b2Magnitude += t2[i] * t2[i];
            sum3 += a * t3[i]; b3Magnitude += t3[i] * t3[i];
        }
        results[at] = (float) (sum0 / Math.sqrt(aMagnitude * b0Magnitude));
        results[at + 1] = (float) (sum1 / Math.sqrt(aMagnitude * b1Magnitude));
        results[at + 2] = (float) (sum2 / Math.sqrt(aMagnitude * b2Magnitude));
        results[at + 3] = (float) (sum3 / Math.sqrt(aMagnitude * b3Magnitude));
    }

    @Override
    public void dotProduct(VectorFloat<?> q, VectorFloat<?>[] targets, int count, float[] results) {
        int i = 0;
        // shorter vectors take the other paths of dotProduct
        if (q.length() >= 2 * FloatVector.SPECIES_PREFERRED.length()) {
            for (; i + 4 <= count; i += 4) {
                dotProduct4(arr(q), arr(targets[i]), arr(targets[i + 1]), arr(targets[i + 2]), arr(targets[i + 3]), q.length(), results, i);
            }
        }
        for (; i < count; i++) {
            results[i] = dotProduct(q, targets[i]);
        }
    }

    /**
     * Four dotProduct(q, t) at once, accumulated as in {@link #dotProductPreferred}: even vector-width chunks into one
     * sum and odd ones into another, then a last odd chunk, the lane reduction, and the scalar tail.
     */
    private static void dotProduct4(float[] q, float[] t0, float[] t1, float[] t2, float[] t3, int length, float[] results, int at) {
        int vectorLength = FloatVector.SPECIES_PREFERRED.length();
        var e0 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var e1 = e0; var e2 = e0; var e3 = e0;
        var o0 = e0; var o1 = e0; var o2 = e0; var o3 = e0;
        int i = 0;
        for (; i + 2 * vectorLength <= length; i += 2 * vectorLength) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, q, i);
            e0 = a.lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t0, i), e0);
            e1 = a.lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t1, i), e1);
            e2 = a.lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t2, i), e2);
            e3 = a.lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t3, i), e3);
            a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, q, i + vectorLength);
            o0 = a.lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t0, i + vectorLength), o0);
            o1 = a.lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t1, i + vectorLength), o1);
            o2 = a.lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t2, i + vectorLength), o2);
            o3 = a.lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t3, i + vectorLength), o3);
        }
        e0 = e0.lanewise(VectorOperators.ADD, o0); e1 = e1.lanewise(VectorOperators.ADD, o1); e2 = e2.lanewise(VectorOperators.ADD, o2); e3 = e3.lanewise(VectorOperators.ADD, o3);
        if (i + vectorLength <= length) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, q, i);
            e0 = a.lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t0, i), e0);
            e1 = a.lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t1, i), e1);
            e2 = a.lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t2, i), e2);
            e3 = a.lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t3, i), e3);
            i += vectorLength;
        }
        float tail0 = 0, tail1 = 0, tail2 = 0, tail3 = 0;
        for (; i < length; i++) {
            float a = q[i];
            tail0 += a * t0[i];
            tail1 += a * t1[i];
            tail2 += a * t2[i];
            tail3 += a * t3[i];
        }
        results[at] = e0.reduceLanes(VectorOperators.ADD) + tail0;
        results[at + 1] = e1.reduceLanes(VectorOperators.ADD) + tail1;
        results[at + 2] = e2.reduceLanes(VectorOperators.ADD) + tail2;
        results[at + 3] = e3.reduceLanes(VectorOperators.ADD) + tail3;
    }

    @Override
    public void squareDistance(VectorFloat<?> q, VectorFloat<?>[] targets, int count, float[] results) {
        int i = 0;
        // shorter vectors, and those exactly one vector wide, take the other paths of squareDistance
        if (q.length() > FloatVector.SPECIES_PREFERRED.length()) {
            for (; i + 4 <= count; i += 4) {
                squareDistance4(arr(q), arr(targets[i]), arr(targets[i + 1]), arr(targets[i + 2]), arr(targets[i + 3]), q.length(), results, i);
            }
        }
        for (; i < count; i++) {
            results[i] = squareDistance(q, targets[i]);
        }
    }

    /** Four squareDistance(q, t) at once, accumulated as in {@link #squareDistance(VectorFloat, int, VectorFloat, int, int)}. */
    private static void squareDistance4(float[] q, float[] t0, float[] t1, float[] t2, float[] t3, int length, float[] results, int at) {
        var s0 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var s1 = s0; var s2 = s0; var s3 = s0;
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        int i = 0;
        for (; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, q, i);
            var d0 = a.lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t0, i));
            var d1 = a.lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t1, i));
            var d2 = a.lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t2, i));
            var d3 = a.lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t3, i));
            s0 = d0.lanewise(VectorOperators.FMA, d0, s0);
            s1 = d1.lanewise(VectorOperators.FMA, d1, s1);
            s2 = d2.lanewise(VectorOperators.FMA, d2, s2);
            s3 = d3.lanewise(VectorOperators.FMA, d3, s3);
        }
        float r0 = s0.reduceLanes(VectorOperators.ADD);
        float r1 = s1.reduceLanes(VectorOperators.ADD);
        float r2 = s2.reduceLanes(VectorOperators.ADD);
        float r3 = s3.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            float a = q[i];
            var d0 = a - t0[i]; r0 += d0 * d0;
            var d1 = a - t1[i]; r1 += d1 * d1;
            var d2 = a - t2[i]; r2 += d2 * d2;
            var d3 = a - t3[i]; r3 += d3 * d3;
        }
        results[at] = r0;
        results[at + 1] = r1;
        results[at + 2] = r2;
        results[at + 3] = r3;
    }

    @Override
    public float squareDistance(VectorFloat<?> v1, VectorFloat<?> v2) {
        return squareDistance(v1, 0, v2, 0, v1.length());
    }

    @Override
    public void dotProductMulti(VectorFloat<?> vector, VectorFloat<?>[] queries, int count, float[] out) {
        if (!(vector instanceof FloatArray)) {
            VectorUtilSupport.super.dotProductMulti(vector, queries, count, out);
            return;
        }
        final float[] v = ((FloatArray) vector).array();
        final int dim = vector.length();
        final int bound = FloatVector.SPECIES_PREFERRED.loopBound(dim);
        int i = 0;
        // eight queries per pass: one vector chunk load feeds eight FMAs into eight accumulators
        for (; i + 8 <= count; i += 8) {
            boolean heap = true;
            for (int j = 0; j < 8; j++) heap &= queries[i + j] instanceof FloatArray;
            if (!heap) break;
            final float[] q0 = ((FloatArray) queries[i]).array(), q1 = ((FloatArray) queries[i + 1]).array();
            final float[] q2 = ((FloatArray) queries[i + 2]).array(), q3 = ((FloatArray) queries[i + 3]).array();
            final float[] q4 = ((FloatArray) queries[i + 4]).array(), q5 = ((FloatArray) queries[i + 5]).array();
            final float[] q6 = ((FloatArray) queries[i + 6]).array(), q7 = ((FloatArray) queries[i + 7]).array();
            FloatVector a0 = FloatVector.zero(FloatVector.SPECIES_PREFERRED), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
            int k = 0;
            for (; k < bound; k += FloatVector.SPECIES_PREFERRED.length()) {
                FloatVector vv = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v, k);
                a0 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, q0, k).lanewise(VectorOperators.FMA, vv, a0);
                a1 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, q1, k).lanewise(VectorOperators.FMA, vv, a1);
                a2 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, q2, k).lanewise(VectorOperators.FMA, vv, a2);
                a3 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, q3, k).lanewise(VectorOperators.FMA, vv, a3);
                a4 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, q4, k).lanewise(VectorOperators.FMA, vv, a4);
                a5 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, q5, k).lanewise(VectorOperators.FMA, vv, a5);
                a6 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, q6, k).lanewise(VectorOperators.FMA, vv, a6);
                a7 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, q7, k).lanewise(VectorOperators.FMA, vv, a7);
            }
            float s0 = a0.reduceLanes(VectorOperators.ADD), s1 = a1.reduceLanes(VectorOperators.ADD);
            float s2 = a2.reduceLanes(VectorOperators.ADD), s3 = a3.reduceLanes(VectorOperators.ADD);
            float s4 = a4.reduceLanes(VectorOperators.ADD), s5 = a5.reduceLanes(VectorOperators.ADD);
            float s6 = a6.reduceLanes(VectorOperators.ADD), s7 = a7.reduceLanes(VectorOperators.ADD);
            for (; k < dim; k++) {
                float x = v[k];
                s0 += x * q0[k]; s1 += x * q1[k]; s2 += x * q2[k]; s3 += x * q3[k];
                s4 += x * q4[k]; s5 += x * q5[k]; s6 += x * q6[k]; s7 += x * q7[k];
            }
            out[i] = s0; out[i + 1] = s1; out[i + 2] = s2; out[i + 3] = s3;
            out[i + 4] = s4; out[i + 5] = s5; out[i + 6] = s6; out[i + 7] = s7;
        }
        for (; i < count; i++) {
            out[i] = dotProduct(vector, queries[i]);
        }
    }

    /**
     * Every length's path is written out in this one method: too large to inline, so always compiled on its own. As
     * separate small methods inlined deep in a caller, the vector operations were sometimes left as calls that put
     * vectors on the heap. The arithmetic is unchanged.
     */
    @Override
    public float squareDistance(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, final int length)
    {
        //Common case first
        if (length >= FloatVector.SPECIES_PREFERRED.length()) {
            if (length == FloatVector.SPECIES_PREFERRED.length())
                {
                var diff = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v1), v1offset)
                        .sub(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v2), v2offset));
                return diff.lanewise(VectorOperators.MUL, diff).reduceLanes(VectorOperators.ADD);
            }

            final int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
            FloatVector sum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

            int i = 0;
            // Process the vectorized part
            for (; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
                FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v1), v1offset + i);
                FloatVector b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v2), v2offset + i);
                var diff = a.lanewise(VectorOperators.SUB, b);
                sum = diff.lanewise(VectorOperators.FMA, diff, sum);
            }

            float res = sum.reduceLanes(VectorOperators.ADD);

            // Process the tail
            for (; i < length; ++i) {
                var diff = v1.get(v1offset + i) - v2.get(v2offset + i);
                res += diff * diff;
            }

            return res;
        }

        if (length < FloatVector.SPECIES_128.length()) {
            if (length == FloatVector.SPECIES_64.length())
                {
                var diff = FloatVector.fromArray(FloatVector.SPECIES_64, arr(v1), v1offset)
                        .sub(FloatVector.fromArray(FloatVector.SPECIES_64, arr(v2), v2offset));
                return diff.lanewise(VectorOperators.MUL, diff).reduceLanes(VectorOperators.ADD);
            }

            final int vectorizedLength = FloatVector.SPECIES_64.loopBound(length);
            FloatVector sum = FloatVector.zero(FloatVector.SPECIES_64);

            int i = 0;
            // Process the vectorized part
            for (; i < vectorizedLength; i += FloatVector.SPECIES_64.length()) {
                FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_64, arr(v1), v1offset + i);
                FloatVector b = FloatVector.fromArray(FloatVector.SPECIES_64, arr(v2), v2offset + i);
                var diff = a.lanewise(VectorOperators.SUB, b);
                sum = diff.lanewise(VectorOperators.FMA, diff, sum);
            }

            float res = sum.reduceLanes(VectorOperators.ADD);

            // Process the tail
            for (; i < length; ++i) {
                var diff = v1.get(v1offset + i) - v2.get(v2offset + i);
                res += diff * diff;
            }

            return res;
        } else if (length < FloatVector.SPECIES_256.length()) {
            if (length == FloatVector.SPECIES_128.length())
                {
                var diff = FloatVector.fromArray(FloatVector.SPECIES_128, arr(v1), v1offset)
                        .sub(FloatVector.fromArray(FloatVector.SPECIES_128, arr(v2), v2offset));
                return diff.lanewise(VectorOperators.MUL, diff).reduceLanes(VectorOperators.ADD);
            }

            final int vectorizedLength = FloatVector.SPECIES_128.loopBound(length);
            FloatVector sum = FloatVector.zero(FloatVector.SPECIES_128);

            int i = 0;
            // Process the vectorized part
            for (; i < vectorizedLength; i += FloatVector.SPECIES_128.length()) {
                FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_128, arr(v1), v1offset + i);
                FloatVector b = FloatVector.fromArray(FloatVector.SPECIES_128, arr(v2), v2offset + i);
                var diff = a.lanewise(VectorOperators.SUB, b);
                sum = diff.lanewise(VectorOperators.FMA, diff, sum);
            }

            float res = sum.reduceLanes(VectorOperators.ADD);

            // Process the tail
            for (; i < length; ++i) {
                var diff = v1.get(v1offset + i) - v2.get(v2offset + i);
                res += diff * diff;
            }

            return res;
        } else {
            if (length == FloatVector.SPECIES_256.length())
                {
                var diff = FloatVector.fromArray(FloatVector.SPECIES_256, arr(v1), v1offset)
                        .sub(FloatVector.fromArray(FloatVector.SPECIES_256, arr(v2), v2offset));
                return diff.lanewise(VectorOperators.MUL, diff).reduceLanes(VectorOperators.ADD);
            }

            final int vectorizedLength = FloatVector.SPECIES_256.loopBound(length);
            FloatVector sum = FloatVector.zero(FloatVector.SPECIES_256);

            int i = 0;
            // Process the vectorized part
            for (; i < vectorizedLength; i += FloatVector.SPECIES_256.length()) {
                FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_256, arr(v1), v1offset + i);
                FloatVector b = FloatVector.fromArray(FloatVector.SPECIES_256, arr(v2), v2offset + i);
                var diff = a.lanewise(VectorOperators.SUB, b);
                sum = diff.lanewise(VectorOperators.FMA, diff, sum);
            }

            float res = sum.reduceLanes(VectorOperators.ADD);

            // Process the tail
            for (; i < length; ++i) {
                var diff = v1.get(v1offset + i) - v2.get(v2offset + i);
                res += diff * diff;
            }

            return res;
        }
    }

    /**
     * Runs the whole scan in this one method so C2 compiles it on its own. Compiled inside a large caller, the
     * per-centroid loop sometimes failed to intrinsify the small-vector subtraction on JDK 22 and allocated on every
     * distance, making PQ encoding and training up to 15x slower. Keep this method above C2's inline limit
     * (FreqInlineSize, 325 bytecodes) and each branch's species constant: a shared helper would be inlined back
     * into the callers. Ties keep the first centroid.
     */
    @Override
    public int tryClosestCentroid(VectorFloat<?> point, int pointOffset,
                                  VectorFloat<?> centroids, int centroidCount, int dimension) {
        if (centroidCount <= 0) {
            throw new IllegalArgumentException("centroidCount must be positive");
        }
        Objects.checkFromIndexSize(pointOffset, dimension, point.length());
        if ((long) centroidCount * dimension > centroids.length()) {
            throw new IndexOutOfBoundsException("centroids do not contain centroidCount vectors");
        }
        int preferred = FloatVector.SPECIES_PREFERRED.length();
        int nearest = 0;
        float min = Float.MAX_VALUE;
        if (dimension == FloatVector.SPECIES_128.length() && dimension < preferred) {
            var x = FloatVector.fromArray(FloatVector.SPECIES_128, arr(point), pointOffset);
            for (int i = 0, at = 0; i < centroidCount; i++, at += dimension) {
                var diff = x.lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_128, arr(centroids), at));
                float d = diff.lanewise(VectorOperators.MUL, diff).reduceLanes(VectorOperators.ADD);
                if (d < min) {
                    min = d;
                    nearest = i;
                }
            }
        } else if (dimension == FloatVector.SPECIES_256.length() && dimension < preferred) {
            var x = FloatVector.fromArray(FloatVector.SPECIES_256, arr(point), pointOffset);
            for (int i = 0, at = 0; i < centroidCount; i++, at += dimension) {
                var diff = x.lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_256, arr(centroids), at));
                float d = diff.lanewise(VectorOperators.MUL, diff).reduceLanes(VectorOperators.ADD);
                if (d < min) {
                    min = d;
                    nearest = i;
                }
            }
        } else if (dimension == FloatVector.SPECIES_64.length() && dimension < preferred) {
            var x = FloatVector.fromArray(FloatVector.SPECIES_64, arr(point), pointOffset);
            for (int i = 0, at = 0; i < centroidCount; i++, at += dimension) {
                var diff = x.lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_64, arr(centroids), at));
                float d = diff.lanewise(VectorOperators.MUL, diff).reduceLanes(VectorOperators.ADD);
                if (d < min) {
                    min = d;
                    nearest = i;
                }
            }
        } else {
            for (int i = 0, at = 0; i < centroidCount; i++, at += dimension) {
                float d = squareDistance(point, pointOffset, centroids, at, dimension);
                if (d < min) {
                    min = d;
                    nearest = i;
                }
            }
        }
        return nearest;
    }

    /**
     * The k-means++ counterpart of {@link #tryClosestCentroid}, written the same way for the same reason: left in
     * its caller, the per-point loop made PQ training about 3x slower in some JDK 22 JVMs.
     */
    @Override
    public boolean tryUpdateKMeansDistances(VectorFloat<?>[] points, VectorFloat<?> centroid,
                                            VectorFloat<?> minima, VectorFloat<?> scratch) {
        int count = points.length;
        if (minima.length() != count || scratch.length() != count) {
            throw new IllegalArgumentException("Distance vector lengths");
        }
        int dimension = centroid.length();
        int preferred = FloatVector.SPECIES_PREFERRED.length();
        if (dimension == FloatVector.SPECIES_128.length() && dimension < preferred) {
            var c = FloatVector.fromArray(FloatVector.SPECIES_128, arr(centroid), 0);
            for (int i = 0; i < count; i++) {
                var point = points[i];
                if (point.length() != dimension) {
                    throw new IllegalArgumentException("Point dimensions differ from centroid");
                }
                var diff = FloatVector.fromArray(FloatVector.SPECIES_128, arr(point), 0).lanewise(VectorOperators.SUB, c);
                scratch.set(i, diff.lanewise(VectorOperators.MUL, diff).reduceLanes(VectorOperators.ADD));
            }
        } else if (dimension == FloatVector.SPECIES_256.length() && dimension < preferred) {
            var c = FloatVector.fromArray(FloatVector.SPECIES_256, arr(centroid), 0);
            for (int i = 0; i < count; i++) {
                var point = points[i];
                if (point.length() != dimension) {
                    throw new IllegalArgumentException("Point dimensions differ from centroid");
                }
                var diff = FloatVector.fromArray(FloatVector.SPECIES_256, arr(point), 0).lanewise(VectorOperators.SUB, c);
                scratch.set(i, diff.lanewise(VectorOperators.MUL, diff).reduceLanes(VectorOperators.ADD));
            }
        } else if (dimension == FloatVector.SPECIES_64.length() && dimension < preferred) {
            var c = FloatVector.fromArray(FloatVector.SPECIES_64, arr(centroid), 0);
            for (int i = 0; i < count; i++) {
                var point = points[i];
                if (point.length() != dimension) {
                    throw new IllegalArgumentException("Point dimensions differ from centroid");
                }
                var diff = FloatVector.fromArray(FloatVector.SPECIES_64, arr(point), 0).lanewise(VectorOperators.SUB, c);
                scratch.set(i, diff.lanewise(VectorOperators.MUL, diff).reduceLanes(VectorOperators.ADD));
            }
        } else {
            for (int i = 0; i < count; i++) {
                var point = points[i];
                if (point.length() != dimension) {
                    throw new IllegalArgumentException("Point dimensions differ from centroid");
                }
                scratch.set(i, squareDistance(point, 0, centroid, 0, dimension));
            }
        }
        minInPlace(minima, scratch);
        return true;
    }

    void addInPlace64(VectorFloat<?> v1, VectorFloat<?> v2) {
        var a = FloatVector.fromArray(FloatVector.SPECIES_64, arr(v1), 0);
        var b = FloatVector.fromArray(FloatVector.SPECIES_64, arr(v2), 0);
        a.lanewise(VectorOperators.ADD, b).intoArray(arr(v1), 0);
    }

    void addInPlace64(VectorFloat<?> v1, float value) {
        var a = FloatVector.fromArray(FloatVector.SPECIES_64, arr(v1), 0);
        a.lanewise(VectorOperators.ADD, value).intoArray(arr(v1), 0);
    }

    @Override
    public void addInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        if (v1.length() != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        if (v1.length() == 2) {
            addInPlace64(v1, v2);
            return;
        }

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v1), i);
            var b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v2), i);
            a.lanewise(VectorOperators.ADD, b).intoArray(arr(v1), i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            v1.set(i, v1.get(i) + v2.get(i));
        }
    }

    @Override
    public void addInPlace(VectorFloat<?> v1, float value) {
        if (v1.length() == 2) {
            addInPlace64(v1, value);
            return;
        }

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v1), i);
            a.lanewise(VectorOperators.ADD, value).intoArray(arr(v1), i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            v1.set(i, v1.get(i) + value);
        }
    }

    @Override
    public void subInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        if (v1.length() != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v1), i);
            var b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v2), i);
            a.lanewise(VectorOperators.SUB, b).intoArray(arr(v1), i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            v1.set(i, v1.get(i) - v2.get(i));
        }
    }

    @Override
    public void subInPlace(VectorFloat<?> vector, float value) {
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(vector), i);
            a.lanewise(VectorOperators.SUB, value).intoArray(arr(vector), i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < vector.length(); i++) {
            vector.set(i, vector.get(i) - value);
        }
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, float value) {
        return sub(a, 0, value, a.length());
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, VectorFloat<?> b) {
        return sub(a, 0, b, 0, a.length());
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length) {
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        VectorFloat<?> res = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(length);

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var lhs = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(a), aOffset + i);
            var rhs = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(b), bOffset + i);
            var subResult = lhs.lanewise(VectorOperators.SUB, rhs);
            subResult.intoArray(arr(res), i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < length; i++) {
            res.set(i, a.get(aOffset + i) - b.get(bOffset + i));
        }

        return res;
    }

    public VectorFloat<?> sub(VectorFloat<?> a, int aOffset, float value, int length) {
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        VectorFloat<?> res = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(length);

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var lhs = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(a), aOffset + i);
            var subResult = lhs.lanewise(VectorOperators.SUB, value);
            subResult.intoArray(arr(res), i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < length; i++) {
            res.set(i, a.get(aOffset + i) - value);
        }

        return res;
    }

    @Override
    public void minInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        if (v1.length() != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v1), i);
            var b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v2), i);
            a.lanewise(VectorOperators.MIN, b).intoArray(arr(v1), i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            v1.set(i, Math.min(v1.get(i), v2.get(i)));
        }
    }

    @Override
    public float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets) {
        return assembleAndSum(data, dataBase, baseOffsets, 0, baseOffsets.length());
    }

    /**
     * Search calls this for every node it scores. The register size is chosen here once and every size's loop is
     * written out in this one method, which is then too large to inline (above C2's FreqInlineSize, 325 bytecodes)
     * and so always compiled on its own: as separate small methods inlined deep in a scorer, the gathers were
     * sometimes left as calls that put vectors on the heap. The loops are unchanged.
     */
    @Override
    public float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
        if (PREFERRED_BIT_SIZE == 512) {
            int[] convOffsets = scratchInt512.get();
            FloatVector sum = FloatVector.zero(FloatVector.SPECIES_512);
            int i = 0;
            int limit = ByteVector.SPECIES_128.loopBound(baseOffsetsLength);
            var scale = IntVector.zero(IntVector.SPECIES_512).addIndex(dataBase);

            for (; i < limit; i += ByteVector.SPECIES_128.length()) {
                ByteVector.fromArray(ByteVector.SPECIES_128, (byte[]) baseOffsets.get(), i + baseOffsets.offset() + baseOffsetsOffset)
                        .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                        .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_512)
                        .reinterpretAsInts()
                        .add(scale)
                        .intoArray(convOffsets, 0);

                var offset = i * dataBase;
                sum = sum.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_512, arr(data), offset, convOffsets, 0));
            }

            float res = sum.reduceLanes(VectorOperators.ADD);

            //Process tail
            for (; i < baseOffsetsLength; i++)
                res += data.get(dataBase * i + Byte.toUnsignedInt(baseOffsets.get(i + baseOffsetsOffset)));

            return res;
        }
        if (PREFERRED_BIT_SIZE == 256) {
            int[] convOffsets = scratchInt256.get();
            FloatVector sum = FloatVector.zero(FloatVector.SPECIES_256);
            int i = 0;
            int limit = ByteVector.SPECIES_64.loopBound(baseOffsetsLength);
            var scale = IntVector.zero(IntVector.SPECIES_256).addIndex(dataBase);

            for (; i < limit; i += ByteVector.SPECIES_64.length()) {

                ByteVector.fromArray(ByteVector.SPECIES_64, (byte[]) baseOffsets.get(), i + baseOffsets.offset() + baseOffsetsOffset)
                        .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                        .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_256)
                        .reinterpretAsInts()
                        .add(scale)
                        .intoArray(convOffsets, 0);

                var offset = i * dataBase;
                sum = sum.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_256, arr(data), offset, convOffsets, 0));
            }

            float res = sum.reduceLanes(VectorOperators.ADD);

            // Process tail
            for (; i < baseOffsetsLength; i++)
                res += data.get(dataBase * i + Byte.toUnsignedInt(baseOffsets.get(i + baseOffsetsOffset)));

            return res;
        }
        if (PREFERRED_BIT_SIZE == 128) {
            // benchmarking a 128-bit SIMD implementation showed it performed worse than scalar
            float sum = 0f;
            for (int i = 0; i < baseOffsetsLength; i++) {
                sum += data.get(dataBase * i + Byte.toUnsignedInt(baseOffsets.get(i + baseOffsetsOffset)));
            }
            return sum;
        }
        throw new IllegalStateException("Unsupported vector width: " + PREFERRED_BIT_SIZE);
    }

    // Package-private: the native provider's 1-float pair body uses the same registers.
    static final VectorSpecies<Float> PAIR_FLOATS = FloatVector.SPECIES_PREFERRED;
    static final VectorSpecies<Integer> PAIR_INTS = VectorSpecies.of(int.class, PAIR_FLOATS.vectorShape());
    static final int PAIR_LANES = PAIR_FLOATS.length();
    // One code byte per float lane, for 1-float subspaces. Byte vectors start at 64 bits, so narrower
    // registers leave this null and use the plain loop.
    static final VectorSpecies<Byte> PAIR_BYTES = PAIR_LANES >= 8
            ? VectorSpecies.of(byte.class, VectorShape.forBitSize(PAIR_LANES * 8)) : null;
    static final ThreadLocal<int[]> PAIR_INDICES = ThreadLocal.withInitial(() -> new int[2 * PAIR_LANES]);


    private static float pairScalar(boolean dot, float[] cb, int clusterCount, int[] subspaceOffsets, int[] subspaceWidths,
                                    ByteSequence<?> code1, int offset1, ByteSequence<?> code2, int offset2) {
        float sum = 0;
        for (int m = 0; m < subspaceOffsets.length; m++) {
            int width = subspaceWidths[m], base = clusterCount * subspaceOffsets[m];
            int b1 = base + Byte.toUnsignedInt(code1.get(offset1 + m)) * width;
            int b2 = base + Byte.toUnsignedInt(code2.get(offset2 + m)) * width;
            for (int t = 0; t < width; t++) {
                float x = cb[b1 + t], y = cb[b2 + t];
                sum += dot ? x * y : (x - y) * (x - y);
            }
        }
        return sum;
    }

    // Four code bytes at a time, so a group of four subspaces reads each code with one bounds check.
    private static final VarHandle CODES4 = MethodHandles.byteArrayViewVarHandle(int[].class, ByteOrder.LITTLE_ENDIAN);

    // The common padded case: every centroid is one register (2, 4 or 8 floats). No inner loop, the codes are read
    // four bytes at a time, and dot product and squared distance are separate methods, so the loop has no branch.
    // Vectors stay inside each method, which takes arrays and offsets and returns a float.
    private static float pairDot8(float[] cb, int clusterCount, int subspaces, byte[] c1, int o1, byte[] c2, int o2) {
        final int stride = clusterCount * 8;
        FloatVector a0 = FloatVector.zero(FloatVector.SPECIES_256), a1 = a0, a2 = a0, a3 = a0;
        int m = 0, b = 0;
        for (; m + 4 <= subspaces; m += 4, b += 4 * stride) {
            int p = (int) CODES4.get(c1, o1 + m), q = (int) CODES4.get(c2, o2 + m);
            a0 = FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + (p & 0xff) * 8).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + (q & 0xff) * 8), a0);
            a1 = FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + stride + ((p >>> 8) & 0xff) * 8).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + stride + ((q >>> 8) & 0xff) * 8), a1);
            a2 = FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + 2 * stride + ((p >>> 16) & 0xff) * 8).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + 2 * stride + ((q >>> 16) & 0xff) * 8), a2);
            a3 = FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + 3 * stride + (p >>> 24) * 8).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + 3 * stride + (q >>> 24) * 8), a3);
        }
        for (; m < subspaces; m++, b += stride) {
            a0 = FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + Byte.toUnsignedInt(c1[o1 + m]) * 8).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + Byte.toUnsignedInt(c2[o2 + m]) * 8), a0);
        }
        return a0.lanewise(VectorOperators.ADD, a1).lanewise(VectorOperators.ADD, a2.lanewise(VectorOperators.ADD, a3)).reduceLanes(VectorOperators.ADD);
    }

    private static float pairDistance8(float[] cb, int clusterCount, int subspaces, byte[] c1, int o1, byte[] c2, int o2) {
        final int stride = clusterCount * 8;
        FloatVector a0 = FloatVector.zero(FloatVector.SPECIES_256), a1 = a0, a2 = a0, a3 = a0, d;
        int m = 0, b = 0;
        for (; m + 4 <= subspaces; m += 4, b += 4 * stride) {
            int p = (int) CODES4.get(c1, o1 + m), q = (int) CODES4.get(c2, o2 + m);
            d = FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + (p & 0xff) * 8).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + (q & 0xff) * 8));
            a0 = d.lanewise(VectorOperators.FMA, d, a0);
            d = FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + stride + ((p >>> 8) & 0xff) * 8).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + stride + ((q >>> 8) & 0xff) * 8));
            a1 = d.lanewise(VectorOperators.FMA, d, a1);
            d = FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + 2 * stride + ((p >>> 16) & 0xff) * 8).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + 2 * stride + ((q >>> 16) & 0xff) * 8));
            a2 = d.lanewise(VectorOperators.FMA, d, a2);
            d = FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + 3 * stride + (p >>> 24) * 8).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + 3 * stride + (q >>> 24) * 8));
            a3 = d.lanewise(VectorOperators.FMA, d, a3);
        }
        for (; m < subspaces; m++, b += stride) {
            d = FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + Byte.toUnsignedInt(c1[o1 + m]) * 8).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_256, cb, b + Byte.toUnsignedInt(c2[o2 + m]) * 8));
            a0 = d.lanewise(VectorOperators.FMA, d, a0);
        }
        return a0.lanewise(VectorOperators.ADD, a1).lanewise(VectorOperators.ADD, a2.lanewise(VectorOperators.ADD, a3)).reduceLanes(VectorOperators.ADD);
    }

    private static float pairDot4(float[] cb, int clusterCount, int subspaces, byte[] c1, int o1, byte[] c2, int o2) {
        final int stride = clusterCount * 4;
        FloatVector a0 = FloatVector.zero(FloatVector.SPECIES_128), a1 = a0, a2 = a0, a3 = a0;
        int m = 0, b = 0;
        for (; m + 4 <= subspaces; m += 4, b += 4 * stride) {
            int p = (int) CODES4.get(c1, o1 + m), q = (int) CODES4.get(c2, o2 + m);
            a0 = FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + (p & 0xff) * 4).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + (q & 0xff) * 4), a0);
            a1 = FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + stride + ((p >>> 8) & 0xff) * 4).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + stride + ((q >>> 8) & 0xff) * 4), a1);
            a2 = FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + 2 * stride + ((p >>> 16) & 0xff) * 4).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + 2 * stride + ((q >>> 16) & 0xff) * 4), a2);
            a3 = FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + 3 * stride + (p >>> 24) * 4).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + 3 * stride + (q >>> 24) * 4), a3);
        }
        for (; m < subspaces; m++, b += stride) {
            a0 = FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + Byte.toUnsignedInt(c1[o1 + m]) * 4).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + Byte.toUnsignedInt(c2[o2 + m]) * 4), a0);
        }
        return a0.lanewise(VectorOperators.ADD, a1).lanewise(VectorOperators.ADD, a2.lanewise(VectorOperators.ADD, a3)).reduceLanes(VectorOperators.ADD);
    }

    private static float pairDistance4(float[] cb, int clusterCount, int subspaces, byte[] c1, int o1, byte[] c2, int o2) {
        final int stride = clusterCount * 4;
        FloatVector a0 = FloatVector.zero(FloatVector.SPECIES_128), a1 = a0, a2 = a0, a3 = a0, d;
        int m = 0, b = 0;
        for (; m + 4 <= subspaces; m += 4, b += 4 * stride) {
            int p = (int) CODES4.get(c1, o1 + m), q = (int) CODES4.get(c2, o2 + m);
            d = FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + (p & 0xff) * 4).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + (q & 0xff) * 4));
            a0 = d.lanewise(VectorOperators.FMA, d, a0);
            d = FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + stride + ((p >>> 8) & 0xff) * 4).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + stride + ((q >>> 8) & 0xff) * 4));
            a1 = d.lanewise(VectorOperators.FMA, d, a1);
            d = FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + 2 * stride + ((p >>> 16) & 0xff) * 4).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + 2 * stride + ((q >>> 16) & 0xff) * 4));
            a2 = d.lanewise(VectorOperators.FMA, d, a2);
            d = FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + 3 * stride + (p >>> 24) * 4).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + 3 * stride + (q >>> 24) * 4));
            a3 = d.lanewise(VectorOperators.FMA, d, a3);
        }
        for (; m < subspaces; m++, b += stride) {
            d = FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + Byte.toUnsignedInt(c1[o1 + m]) * 4).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_128, cb, b + Byte.toUnsignedInt(c2[o2 + m]) * 4));
            a0 = d.lanewise(VectorOperators.FMA, d, a0);
        }
        return a0.lanewise(VectorOperators.ADD, a1).lanewise(VectorOperators.ADD, a2.lanewise(VectorOperators.ADD, a3)).reduceLanes(VectorOperators.ADD);
    }

    private static float pairDot2(float[] cb, int clusterCount, int subspaces, byte[] c1, int o1, byte[] c2, int o2) {
        final int stride = clusterCount * 2;
        FloatVector a0 = FloatVector.zero(FloatVector.SPECIES_64), a1 = a0, a2 = a0, a3 = a0;
        int m = 0, b = 0;
        for (; m + 4 <= subspaces; m += 4, b += 4 * stride) {
            int p = (int) CODES4.get(c1, o1 + m), q = (int) CODES4.get(c2, o2 + m);
            a0 = FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + (p & 0xff) * 2).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + (q & 0xff) * 2), a0);
            a1 = FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + stride + ((p >>> 8) & 0xff) * 2).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + stride + ((q >>> 8) & 0xff) * 2), a1);
            a2 = FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + 2 * stride + ((p >>> 16) & 0xff) * 2).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + 2 * stride + ((q >>> 16) & 0xff) * 2), a2);
            a3 = FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + 3 * stride + (p >>> 24) * 2).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + 3 * stride + (q >>> 24) * 2), a3);
        }
        for (; m < subspaces; m++, b += stride) {
            a0 = FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + Byte.toUnsignedInt(c1[o1 + m]) * 2).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + Byte.toUnsignedInt(c2[o2 + m]) * 2), a0);
        }
        return a0.lanewise(VectorOperators.ADD, a1).lanewise(VectorOperators.ADD, a2.lanewise(VectorOperators.ADD, a3)).reduceLanes(VectorOperators.ADD);
    }

    private static float pairDistance2(float[] cb, int clusterCount, int subspaces, byte[] c1, int o1, byte[] c2, int o2) {
        final int stride = clusterCount * 2;
        FloatVector a0 = FloatVector.zero(FloatVector.SPECIES_64), a1 = a0, a2 = a0, a3 = a0, d;
        int m = 0, b = 0;
        for (; m + 4 <= subspaces; m += 4, b += 4 * stride) {
            int p = (int) CODES4.get(c1, o1 + m), q = (int) CODES4.get(c2, o2 + m);
            d = FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + (p & 0xff) * 2).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + (q & 0xff) * 2));
            a0 = d.lanewise(VectorOperators.FMA, d, a0);
            d = FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + stride + ((p >>> 8) & 0xff) * 2).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + stride + ((q >>> 8) & 0xff) * 2));
            a1 = d.lanewise(VectorOperators.FMA, d, a1);
            d = FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + 2 * stride + ((p >>> 16) & 0xff) * 2).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + 2 * stride + ((q >>> 16) & 0xff) * 2));
            a2 = d.lanewise(VectorOperators.FMA, d, a2);
            d = FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + 3 * stride + (p >>> 24) * 2).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + 3 * stride + (q >>> 24) * 2));
            a3 = d.lanewise(VectorOperators.FMA, d, a3);
        }
        for (; m < subspaces; m++, b += stride) {
            d = FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + Byte.toUnsignedInt(c1[o1 + m]) * 2).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_64, cb, b + Byte.toUnsignedInt(c2[o2 + m]) * 2));
            a0 = d.lanewise(VectorOperators.FMA, d, a0);
        }
        return a0.lanewise(VectorOperators.ADD, a1).lanewise(VectorOperators.ADD, a2.lanewise(VectorOperators.ADD, a3)).reduceLanes(VectorOperators.ADD);
    }

    // Wider padded widths (several registers per centroid): the same four-subspace loop with an inner loop over
    // registers. Vectors stay inside each method, which takes arrays and offsets and returns a float; one method per
    // register size keeps the species a constant.
    private static float pairPadded512(boolean dot, float[] cb, int clusterCount, int width, int subspaces,
                                         byte[] c1, int o1, byte[] c2, int o2) {
        final int lanes = FloatVector.SPECIES_512.length(), stride = clusterCount * width;
        FloatVector a0 = FloatVector.zero(FloatVector.SPECIES_512), a1 = a0, a2 = a0, a3 = a0, d;
        int m = 0, b = 0;
        for (; m + 4 <= subspaces; m += 4, b += 4 * stride) {
            int x0 = b + Byte.toUnsignedInt(c1[o1 + m]) * width, y0 = b + Byte.toUnsignedInt(c2[o2 + m]) * width;
            int x1 = b + stride + Byte.toUnsignedInt(c1[o1 + m + 1]) * width;
            int y1 = b + stride + Byte.toUnsignedInt(c2[o2 + m + 1]) * width;
            int x2 = b + 2 * stride + Byte.toUnsignedInt(c1[o1 + m + 2]) * width;
            int y2 = b + 2 * stride + Byte.toUnsignedInt(c2[o2 + m + 2]) * width;
            int x3 = b + 3 * stride + Byte.toUnsignedInt(c1[o1 + m + 3]) * width;
            int y3 = b + 3 * stride + Byte.toUnsignedInt(c2[o2 + m + 3]) * width;
            for (int t = 0; t < width; t += lanes) {
                if (dot) {
                    a0 = FloatVector.fromArray(FloatVector.SPECIES_512, cb, x0 + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_512, cb, y0 + t), a0);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_512, cb, x0 + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_512, cb, y0 + t));
                    a0 = d.lanewise(VectorOperators.FMA, d, a0);
                }
                if (dot) {
                    a1 = FloatVector.fromArray(FloatVector.SPECIES_512, cb, x1 + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_512, cb, y1 + t), a1);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_512, cb, x1 + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_512, cb, y1 + t));
                    a1 = d.lanewise(VectorOperators.FMA, d, a1);
                }
                if (dot) {
                    a2 = FloatVector.fromArray(FloatVector.SPECIES_512, cb, x2 + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_512, cb, y2 + t), a2);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_512, cb, x2 + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_512, cb, y2 + t));
                    a2 = d.lanewise(VectorOperators.FMA, d, a2);
                }
                if (dot) {
                    a3 = FloatVector.fromArray(FloatVector.SPECIES_512, cb, x3 + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_512, cb, y3 + t), a3);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_512, cb, x3 + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_512, cb, y3 + t));
                    a3 = d.lanewise(VectorOperators.FMA, d, a3);
                }
            }
        }
        // The last 1-3 subspaces add into a0.
        for (; m < subspaces; m++, b += stride) {
            int x = b + Byte.toUnsignedInt(c1[o1 + m]) * width, y = b + Byte.toUnsignedInt(c2[o2 + m]) * width;
            for (int t = 0; t < width; t += lanes) {
                if (dot) {
                    a0 = FloatVector.fromArray(FloatVector.SPECIES_512, cb, x + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_512, cb, y + t), a0);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_512, cb, x + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_512, cb, y + t));
                    a0 = d.lanewise(VectorOperators.FMA, d, a0);
                }
            }
        }
        return a0.lanewise(VectorOperators.ADD, a1).lanewise(VectorOperators.ADD, a2.lanewise(VectorOperators.ADD, a3)).reduceLanes(VectorOperators.ADD);
    }

    private static float pairPadded256(boolean dot, float[] cb, int clusterCount, int width, int subspaces,
                                         byte[] c1, int o1, byte[] c2, int o2) {
        final int lanes = FloatVector.SPECIES_256.length(), stride = clusterCount * width;
        FloatVector a0 = FloatVector.zero(FloatVector.SPECIES_256), a1 = a0, a2 = a0, a3 = a0, d;
        int m = 0, b = 0;
        for (; m + 4 <= subspaces; m += 4, b += 4 * stride) {
            int x0 = b + Byte.toUnsignedInt(c1[o1 + m]) * width, y0 = b + Byte.toUnsignedInt(c2[o2 + m]) * width;
            int x1 = b + stride + Byte.toUnsignedInt(c1[o1 + m + 1]) * width;
            int y1 = b + stride + Byte.toUnsignedInt(c2[o2 + m + 1]) * width;
            int x2 = b + 2 * stride + Byte.toUnsignedInt(c1[o1 + m + 2]) * width;
            int y2 = b + 2 * stride + Byte.toUnsignedInt(c2[o2 + m + 2]) * width;
            int x3 = b + 3 * stride + Byte.toUnsignedInt(c1[o1 + m + 3]) * width;
            int y3 = b + 3 * stride + Byte.toUnsignedInt(c2[o2 + m + 3]) * width;
            for (int t = 0; t < width; t += lanes) {
                if (dot) {
                    a0 = FloatVector.fromArray(FloatVector.SPECIES_256, cb, x0 + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_256, cb, y0 + t), a0);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_256, cb, x0 + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_256, cb, y0 + t));
                    a0 = d.lanewise(VectorOperators.FMA, d, a0);
                }
                if (dot) {
                    a1 = FloatVector.fromArray(FloatVector.SPECIES_256, cb, x1 + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_256, cb, y1 + t), a1);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_256, cb, x1 + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_256, cb, y1 + t));
                    a1 = d.lanewise(VectorOperators.FMA, d, a1);
                }
                if (dot) {
                    a2 = FloatVector.fromArray(FloatVector.SPECIES_256, cb, x2 + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_256, cb, y2 + t), a2);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_256, cb, x2 + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_256, cb, y2 + t));
                    a2 = d.lanewise(VectorOperators.FMA, d, a2);
                }
                if (dot) {
                    a3 = FloatVector.fromArray(FloatVector.SPECIES_256, cb, x3 + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_256, cb, y3 + t), a3);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_256, cb, x3 + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_256, cb, y3 + t));
                    a3 = d.lanewise(VectorOperators.FMA, d, a3);
                }
            }
        }
        // The last 1-3 subspaces add into a0.
        for (; m < subspaces; m++, b += stride) {
            int x = b + Byte.toUnsignedInt(c1[o1 + m]) * width, y = b + Byte.toUnsignedInt(c2[o2 + m]) * width;
            for (int t = 0; t < width; t += lanes) {
                if (dot) {
                    a0 = FloatVector.fromArray(FloatVector.SPECIES_256, cb, x + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_256, cb, y + t), a0);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_256, cb, x + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_256, cb, y + t));
                    a0 = d.lanewise(VectorOperators.FMA, d, a0);
                }
            }
        }
        return a0.lanewise(VectorOperators.ADD, a1).lanewise(VectorOperators.ADD, a2.lanewise(VectorOperators.ADD, a3)).reduceLanes(VectorOperators.ADD);
    }

    private static float pairPadded128(boolean dot, float[] cb, int clusterCount, int width, int subspaces,
                                         byte[] c1, int o1, byte[] c2, int o2) {
        final int lanes = FloatVector.SPECIES_128.length(), stride = clusterCount * width;
        FloatVector a0 = FloatVector.zero(FloatVector.SPECIES_128), a1 = a0, a2 = a0, a3 = a0, d;
        int m = 0, b = 0;
        for (; m + 4 <= subspaces; m += 4, b += 4 * stride) {
            int x0 = b + Byte.toUnsignedInt(c1[o1 + m]) * width, y0 = b + Byte.toUnsignedInt(c2[o2 + m]) * width;
            int x1 = b + stride + Byte.toUnsignedInt(c1[o1 + m + 1]) * width;
            int y1 = b + stride + Byte.toUnsignedInt(c2[o2 + m + 1]) * width;
            int x2 = b + 2 * stride + Byte.toUnsignedInt(c1[o1 + m + 2]) * width;
            int y2 = b + 2 * stride + Byte.toUnsignedInt(c2[o2 + m + 2]) * width;
            int x3 = b + 3 * stride + Byte.toUnsignedInt(c1[o1 + m + 3]) * width;
            int y3 = b + 3 * stride + Byte.toUnsignedInt(c2[o2 + m + 3]) * width;
            for (int t = 0; t < width; t += lanes) {
                if (dot) {
                    a0 = FloatVector.fromArray(FloatVector.SPECIES_128, cb, x0 + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_128, cb, y0 + t), a0);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_128, cb, x0 + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_128, cb, y0 + t));
                    a0 = d.lanewise(VectorOperators.FMA, d, a0);
                }
                if (dot) {
                    a1 = FloatVector.fromArray(FloatVector.SPECIES_128, cb, x1 + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_128, cb, y1 + t), a1);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_128, cb, x1 + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_128, cb, y1 + t));
                    a1 = d.lanewise(VectorOperators.FMA, d, a1);
                }
                if (dot) {
                    a2 = FloatVector.fromArray(FloatVector.SPECIES_128, cb, x2 + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_128, cb, y2 + t), a2);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_128, cb, x2 + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_128, cb, y2 + t));
                    a2 = d.lanewise(VectorOperators.FMA, d, a2);
                }
                if (dot) {
                    a3 = FloatVector.fromArray(FloatVector.SPECIES_128, cb, x3 + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_128, cb, y3 + t), a3);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_128, cb, x3 + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_128, cb, y3 + t));
                    a3 = d.lanewise(VectorOperators.FMA, d, a3);
                }
            }
        }
        // The last 1-3 subspaces add into a0.
        for (; m < subspaces; m++, b += stride) {
            int x = b + Byte.toUnsignedInt(c1[o1 + m]) * width, y = b + Byte.toUnsignedInt(c2[o2 + m]) * width;
            for (int t = 0; t < width; t += lanes) {
                if (dot) {
                    a0 = FloatVector.fromArray(FloatVector.SPECIES_128, cb, x + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_128, cb, y + t), a0);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_128, cb, x + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_128, cb, y + t));
                    a0 = d.lanewise(VectorOperators.FMA, d, a0);
                }
            }
        }
        return a0.lanewise(VectorOperators.ADD, a1).lanewise(VectorOperators.ADD, a2.lanewise(VectorOperators.ADD, a3)).reduceLanes(VectorOperators.ADD);
    }

    private static float pairPadded64(boolean dot, float[] cb, int clusterCount, int width, int subspaces,
                                         byte[] c1, int o1, byte[] c2, int o2) {
        final int lanes = FloatVector.SPECIES_64.length(), stride = clusterCount * width;
        FloatVector a0 = FloatVector.zero(FloatVector.SPECIES_64), a1 = a0, a2 = a0, a3 = a0, d;
        int m = 0, b = 0;
        for (; m + 4 <= subspaces; m += 4, b += 4 * stride) {
            int x0 = b + Byte.toUnsignedInt(c1[o1 + m]) * width, y0 = b + Byte.toUnsignedInt(c2[o2 + m]) * width;
            int x1 = b + stride + Byte.toUnsignedInt(c1[o1 + m + 1]) * width;
            int y1 = b + stride + Byte.toUnsignedInt(c2[o2 + m + 1]) * width;
            int x2 = b + 2 * stride + Byte.toUnsignedInt(c1[o1 + m + 2]) * width;
            int y2 = b + 2 * stride + Byte.toUnsignedInt(c2[o2 + m + 2]) * width;
            int x3 = b + 3 * stride + Byte.toUnsignedInt(c1[o1 + m + 3]) * width;
            int y3 = b + 3 * stride + Byte.toUnsignedInt(c2[o2 + m + 3]) * width;
            for (int t = 0; t < width; t += lanes) {
                if (dot) {
                    a0 = FloatVector.fromArray(FloatVector.SPECIES_64, cb, x0 + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_64, cb, y0 + t), a0);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_64, cb, x0 + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_64, cb, y0 + t));
                    a0 = d.lanewise(VectorOperators.FMA, d, a0);
                }
                if (dot) {
                    a1 = FloatVector.fromArray(FloatVector.SPECIES_64, cb, x1 + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_64, cb, y1 + t), a1);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_64, cb, x1 + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_64, cb, y1 + t));
                    a1 = d.lanewise(VectorOperators.FMA, d, a1);
                }
                if (dot) {
                    a2 = FloatVector.fromArray(FloatVector.SPECIES_64, cb, x2 + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_64, cb, y2 + t), a2);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_64, cb, x2 + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_64, cb, y2 + t));
                    a2 = d.lanewise(VectorOperators.FMA, d, a2);
                }
                if (dot) {
                    a3 = FloatVector.fromArray(FloatVector.SPECIES_64, cb, x3 + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_64, cb, y3 + t), a3);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_64, cb, x3 + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_64, cb, y3 + t));
                    a3 = d.lanewise(VectorOperators.FMA, d, a3);
                }
            }
        }
        // The last 1-3 subspaces add into a0.
        for (; m < subspaces; m++, b += stride) {
            int x = b + Byte.toUnsignedInt(c1[o1 + m]) * width, y = b + Byte.toUnsignedInt(c2[o2 + m]) * width;
            for (int t = 0; t < width; t += lanes) {
                if (dot) {
                    a0 = FloatVector.fromArray(FloatVector.SPECIES_64, cb, x + t).lanewise(VectorOperators.FMA, FloatVector.fromArray(FloatVector.SPECIES_64, cb, y + t), a0);
                } else {
                    d = FloatVector.fromArray(FloatVector.SPECIES_64, cb, x + t).lanewise(VectorOperators.SUB, FloatVector.fromArray(FloatVector.SPECIES_64, cb, y + t));
                    a0 = d.lanewise(VectorOperators.FMA, d, a0);
                }
            }
        }
        return a0.lanewise(VectorOperators.ADD, a1).lanewise(VectorOperators.ADD, a2.lanewise(VectorOperators.ADD, a3)).reduceLanes(VectorOperators.ADD);
    }

    @Override
    public int pqCodePairStride(int minWidth, int maxWidth) {
        if (maxWidth <= 1) return 0;
        if (minWidth == maxWidth && (maxWidth & 1) == 0) return maxWidth;
        return maxWidth <= 2 ? 2 : maxWidth <= 4 ? 4 : (maxWidth + 7) & ~7;
    }

    /**
     * A table costs one lookup per subspace, while scoring from the codebooks grows with the width: the table wins
     * from 16 floats (measured with AVX-512 and AVX2).
     */
    @Override
    public boolean pqCodePairUsesTable(int minWidth) {
        return minWidth >= 16;
    }

    @Override
    public float pqCodePairSimilarity(boolean dot, VectorFloat<?> codebooks, int clusterCount,
                                      int[] subspaceOffsets, int[] subspaceWidths, int minWidth, int maxWidth, int[] dimBase,
                                      ByteSequence<?> code1, int offset1, ByteSequence<?> code2, int offset2) {
        // A build calls this tens of millions of times, so it only dispatches: each path is its own small
        // method, keeping the whole call chain small enough to inline.
        float[] cb = ((FloatArray) codebooks).array();
        if (minWidth == maxWidth && (minWidth & 1) == 0) {
            // Padded layout: read the codes straight from their arrays.
            int m = subspaceOffsets.length;
            byte[] c1 = (byte[]) code1.get(), c2 = (byte[]) code2.get();
            int o1 = code1.offset() + offset1, o2 = code2.offset() + offset2;
            if (minWidth <= 8 && minWidth <= PAIR_LANES && (minWidth & (minWidth - 1)) == 0) {
                // One register per centroid.
                if (minWidth == 8) return dot ? pairDot8(cb, clusterCount, m, c1, o1, c2, o2) : pairDistance8(cb, clusterCount, m, c1, o1, c2, o2);
                if (minWidth == 4) return dot ? pairDot4(cb, clusterCount, m, c1, o1, c2, o2) : pairDistance4(cb, clusterCount, m, c1, o1, c2, o2);
                return dot ? pairDot2(cb, clusterCount, m, c1, o1, c2, o2) : pairDistance2(cb, clusterCount, m, c1, o1, c2, o2);
            }
            // Wider padded widths: registers of the largest power of two dividing the width, up to the hardware width.
            int lanes = Math.min(PAIR_LANES, minWidth & -minWidth);
            if (lanes >= 16) return pairPadded512(dot, cb, clusterCount, minWidth, m, c1, o1, c2, o2);
            if (lanes >= 8) return pairPadded256(dot, cb, clusterCount, minWidth, m, c1, o1, c2, o2);
            if (lanes >= 4) return pairPadded128(dot, cb, clusterCount, minWidth, m, c1, o1, c2, o2);
            return pairPadded64(dot, cb, clusterCount, minWidth, m, c1, o1, c2, o2);
        }
        if (maxWidth == 1 && dimBase.length >= PAIR_LANES && PAIR_BYTES != null) {
            return pairOneFloat(dot, cb, dimBase, code1, offset1, code2, offset2);
        }
        // Unpadded mixed widths, or 1-float subspaces too few to fill a register: a plain loop.
        return pairScalar(dot, cb, clusterCount, subspaceOffsets, subspaceWidths, code1, offset1, code2, offset2);
    }

    /**
     * 1-float subspaces, at least a register of them. Only unwraps the codes; the native provider overrides it for its
     * memory-segment codes. Either way the vector work stays in one static body.
     */
    protected float pairOneFloat(boolean dot, float[] cb, int[] dimBase,
                                 ByteSequence<?> code1, int offset1, ByteSequence<?> code2, int offset2) {
        return pairGatherOneFloat(dot, cb, dimBase, PAIR_INDICES.get(), (byte[]) code1.get(), code1.offset() + offset1,
                                  (byte[]) code2.get(), code2.offset() + offset2);
    }

    /** Full registers of 1-float subspaces: each lane's code is one byte of a vector load. */
    private static float pairGatherOneFloat(boolean dot, float[] cb, int[] dimBase, int[] indices,
                                            byte[] c1, int o1, byte[] c2, int o2) {
        int dim = dimBase.length;
        FloatVector acc = FloatVector.zero(PAIR_FLOATS), d;
        int j = 0;
        for (; j + PAIR_LANES <= dim; j += PAIR_LANES) {
            IntVector lanesBase = IntVector.fromArray(PAIR_INTS, dimBase, j);
            ((IntVector) ByteVector.fromArray(PAIR_BYTES, c1, o1 + j).convertShape(VectorOperators.B2I, PAIR_INTS, 0))
                    .lanewise(VectorOperators.AND, 0xff).lanewise(VectorOperators.ADD, lanesBase).intoArray(indices, 0);
            ((IntVector) ByteVector.fromArray(PAIR_BYTES, c2, o2 + j).convertShape(VectorOperators.B2I, PAIR_INTS, 0))
                    .lanewise(VectorOperators.AND, 0xff).lanewise(VectorOperators.ADD, lanesBase).intoArray(indices, PAIR_LANES);
            FloatVector x = FloatVector.fromArray(PAIR_FLOATS, cb, 0, indices, 0);
            FloatVector y = FloatVector.fromArray(PAIR_FLOATS, cb, 0, indices, PAIR_LANES);
            if (dot) {
                acc = x.lanewise(VectorOperators.FMA, y, acc);
            } else {
                d = x.lanewise(VectorOperators.SUB, y);
                acc = d.lanewise(VectorOperators.FMA, d, acc);
            }
        }
        float tail = 0;
        for (; j < dim; j++) {
            float x = cb[dimBase[j] + Byte.toUnsignedInt(c1[o1 + j])];
            float y = cb[dimBase[j] + Byte.toUnsignedInt(c2[o2 + j])];
            tail += dot ? x * y : (x - y) * (x - y);
        }
        return acc.reduceLanes(VectorOperators.ADD) + tail;
    }

    /**
     * Sums the centroid-pair table entries two codes pick: one gather per register of subspaces. The register size is
     * chosen once here; each body holds a single gather call site and keeps all its vectors to itself.
     */
    @Override
    public float assembleAndSumPQ(
            VectorFloat<?> codebookPartialSums,
            int subspaceCount,                  // = M
            ByteSequence<?> vector1Ordinals,
            int vector1OrdinalOffset,
            ByteSequence<?> vector2Ordinals,
            int vector2OrdinalOffset,
            int clusterCount                    // = k
    ) {
        float[] table = ((FloatArray) codebookPartialSums).array();
        byte[] c1 = (byte[]) vector1Ordinals.get(), c2 = (byte[]) vector2Ordinals.get();
        int o1 = vector1Ordinals.offset() + vector1OrdinalOffset, o2 = vector2Ordinals.offset() + vector2OrdinalOffset;
        if (PREFERRED_BIT_SIZE >= 512) {
            return pairTable512(table, subspaceCount, c1, o1, c2, o2, clusterCount, scratchInt512.get());
        }
        if (PREFERRED_BIT_SIZE >= 256) {
            return pairTable256(table, subspaceCount, c1, o1, c2, o2, clusterCount, scratchInt256.get());
        }
        // Narrower registers have no gather instruction: plain lookups.
        float sum = 0;
        for (int m = 0, b = 0, block = clusterCount * (clusterCount + 1) / 2; m < subspaceCount; m++, b += block) {
            sum += table[b + pairEntry(c1[o1 + m], c2[o2 + m], clusterCount)];
        }
        return sum;
    }

    private static float pairTable512(float[] table, int subspaces, byte[] c1, int o1, byte[] c2, int o2, int k,
                                        int[] idx) {
        final int block = k * (k + 1) / 2;
        // Lane j reads from block j of the 16 that start at the gather's base offset.
        final IntVector blocks = IntVector.zero(IntVector.SPECIES_512).addIndex(block);
        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_512);
        int m = 0;
        for (; m + 16 <= subspaces; m += 16) {
            IntVector a = ((IntVector) ByteVector.fromArray(ByteVector.SPECIES_128, c1, o1 + m)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)).lanewise(VectorOperators.AND, 0xff);
            IntVector b = ((IntVector) ByteVector.fromArray(ByteVector.SPECIES_128, c2, o2 + m)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)).lanewise(VectorOperators.AND, 0xff);
            // Entry r * k - r * (r - 1) / 2 + (c - r) of the block, for row r = min and column c = max.
            IntVector r = a.lanewise(VectorOperators.MIN, b), c = a.lanewise(VectorOperators.MAX, b);
            r.lanewise(VectorOperators.MUL, k).lanewise(VectorOperators.SUB, r.lanewise(VectorOperators.MUL, r.lanewise(VectorOperators.SUB, 1)).lanewise(VectorOperators.ASHR, 1)).lanewise(VectorOperators.ADD, c.lanewise(VectorOperators.SUB, r)).lanewise(VectorOperators.ADD, blocks).intoArray(idx, 0);
            acc = acc.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_512, table, m * block, idx, 0));
        }
        // The last few subspaces, with no vectors.
        float s0 = 0, s1 = 0;
        int b = m * block;
        for (; m + 2 <= subspaces; m += 2, b += 2 * block) {
            s0 += table[b + pairEntry(c1[o1 + m], c2[o2 + m], k)];
            s1 += table[b + block + pairEntry(c1[o1 + m + 1], c2[o2 + m + 1], k)];
        }
        if (m < subspaces) s0 += table[b + pairEntry(c1[o1 + m], c2[o2 + m], k)];
        return acc.reduceLanes(VectorOperators.ADD) + (s0 + s1);
    }

    private static float pairTable256(float[] table, int subspaces, byte[] c1, int o1, byte[] c2, int o2, int k,
                                        int[] idx) {
        final int block = k * (k + 1) / 2;
        // Lane j reads from block j of the 8 that start at the gather's base offset.
        final IntVector blocks = IntVector.zero(IntVector.SPECIES_256).addIndex(block);
        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_256);
        int m = 0;
        for (; m + 8 <= subspaces; m += 8) {
            IntVector a = ((IntVector) ByteVector.fromArray(ByteVector.SPECIES_64, c1, o1 + m)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)).lanewise(VectorOperators.AND, 0xff);
            IntVector b = ((IntVector) ByteVector.fromArray(ByteVector.SPECIES_64, c2, o2 + m)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)).lanewise(VectorOperators.AND, 0xff);
            // Entry r * k - r * (r - 1) / 2 + (c - r) of the block, for row r = min and column c = max.
            IntVector r = a.lanewise(VectorOperators.MIN, b), c = a.lanewise(VectorOperators.MAX, b);
            r.lanewise(VectorOperators.MUL, k).lanewise(VectorOperators.SUB, r.lanewise(VectorOperators.MUL, r.lanewise(VectorOperators.SUB, 1)).lanewise(VectorOperators.ASHR, 1)).lanewise(VectorOperators.ADD, c.lanewise(VectorOperators.SUB, r)).lanewise(VectorOperators.ADD, blocks).intoArray(idx, 0);
            acc = acc.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_256, table, m * block, idx, 0));
        }
        // The last few subspaces, with no vectors.
        float s0 = 0, s1 = 0;
        int b = m * block;
        for (; m + 2 <= subspaces; m += 2, b += 2 * block) {
            s0 += table[b + pairEntry(c1[o1 + m], c2[o2 + m], k)];
            s1 += table[b + block + pairEntry(c1[o1 + m + 1], c2[o2 + m + 1], k)];
        }
        if (m < subspaces) s0 += table[b + pairEntry(c1[o1 + m], c2[o2 + m], k)];
        return acc.reduceLanes(VectorOperators.ADD) + (s0 + s1);
    }

    /** The index of the pair (x, y) in one subspace's upper-triangular block of k * (k + 1) / 2 entries. */
    private static int pairEntry(byte x, byte y, int k) {
        int a = Byte.toUnsignedInt(x), b = Byte.toUnsignedInt(y);
        int r = Math.min(a, b), c = Math.max(a, b);
        return r * k - (r * (r - 1) >> 1) + (c - r);
    }

                // -----------------------------------------------------------------------
    // ByteSequence similarity metrics – Panama SIMD implementations
    //
    // Strategy: widen signed bytes to int32 via B2I (no AND-mask needed for
    // signed arithmetic), accumulate products in IntVector lanes, then reduce.
    // The byte-vector species is 1/4 the width of the int species:
    //   512-bit int (16 lanes) <- SPECIES_128 bytes
    //   256-bit int  (8 lanes) <- SPECIES_64  bytes
    //   128-bit preferred      <- scalar (ByteVector.SPECIES_32 does not exist;
    //                             128-bit SIMD shows no benefit for this workload)
    // -----------------------------------------------------------------------

    /**
     * Vectorized dot product of two signed int8 byte vectors.
     */
    @Override
    public float dotProduct(ByteSequence<?> a, ByteSequence<?> b) {
        // Each register size's loop written out in this one method, compiled on its own (too large to inline);
        // the loops are unchanged.
        if (PREFERRED_BIT_SIZE == 512) {
            final int length = a.length();
            final int step   = ByteVector.SPECIES_128.length(); // 16
            final int limit  = ByteVector.SPECIES_128.loopBound(length);
            IntVector acc    = IntVector.zero(IntVector.SPECIES_512);

            for (int i = 0; i < limit; i += step) {
                IntVector va = ByteVector.fromArray(ByteVector.SPECIES_128, (byte[]) a.get(), a.offset() + i)
                        .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                        .reinterpretAsInts();
                IntVector vb = ByteVector.fromArray(ByteVector.SPECIES_128, (byte[]) b.get(), b.offset() + i)
                        .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                        .reinterpretAsInts();
                acc = acc.lanewise(VectorOperators.ADD, va.lanewise(VectorOperators.MUL, vb));
            }

            int result = acc.reduceLanes(VectorOperators.ADD);
            for (int i = limit; i < length; i++) {
                result += a.get(i) * b.get(i);
            }
            return result;
        } else if (PREFERRED_BIT_SIZE == 256) {
            final int length = a.length();
            final int step   = ByteVector.SPECIES_64.length(); // 8
            final int limit  = ByteVector.SPECIES_64.loopBound(length);
            IntVector acc    = IntVector.zero(IntVector.SPECIES_256);

            for (int i = 0; i < limit; i += step) {
                IntVector va = ByteVector.fromArray(ByteVector.SPECIES_64, (byte[]) a.get(), a.offset() + i)
                        .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                        .reinterpretAsInts();
                IntVector vb = ByteVector.fromArray(ByteVector.SPECIES_64, (byte[]) b.get(), b.offset() + i)
                        .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                        .reinterpretAsInts();
                acc = acc.lanewise(VectorOperators.ADD, va.lanewise(VectorOperators.MUL, vb));
            }

            int result = acc.reduceLanes(VectorOperators.ADD);
            for (int i = limit; i < length; i++) {
                result += a.get(i) * b.get(i);
            }
            return result;
        } else {
            // ByteVector.SPECIES_32 does not exist; scalar is fastest at 128-bit width
            final int length = a.length();
            int result = 0;
            for (int i = 0; i < length; i++) {
                result += a.get(i) * b.get(i);
            }
            return result;
        }
    }

    /**
     * Vectorized sum of squared differences between two signed int8 byte vectors.
     */
    @Override
    public float squareDistance(ByteSequence<?> a, ByteSequence<?> b) {
        // Each register size's loop written out in this one method, compiled on its own (too large to inline);
        // the loops are unchanged.
        if (PREFERRED_BIT_SIZE == 512) {
            final int length = a.length();
            final int step   = ByteVector.SPECIES_128.length();
            final int limit  = ByteVector.SPECIES_128.loopBound(length);
            IntVector acc    = IntVector.zero(IntVector.SPECIES_512);

            for (int i = 0; i < limit; i += step) {
                IntVector va   = ByteVector.fromArray(ByteVector.SPECIES_128, (byte[]) a.get(), a.offset() + i)
                        .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                        .reinterpretAsInts();
                IntVector vb   = ByteVector.fromArray(ByteVector.SPECIES_128, (byte[]) b.get(), b.offset() + i)
                        .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                        .reinterpretAsInts();
                IntVector diff = va.lanewise(VectorOperators.SUB, vb);
                acc = acc.lanewise(VectorOperators.ADD, diff.lanewise(VectorOperators.MUL, diff));
            }

            int result = acc.reduceLanes(VectorOperators.ADD);
            for (int i = limit; i < length; i++) {
                int diff = a.get(i) - b.get(i);
                result += diff * diff;
            }
            return result;
        } else if (PREFERRED_BIT_SIZE == 256) {
            final int length = a.length();
            final int step   = ByteVector.SPECIES_64.length();
            final int limit  = ByteVector.SPECIES_64.loopBound(length);
            IntVector acc    = IntVector.zero(IntVector.SPECIES_256);

            for (int i = 0; i < limit; i += step) {
                IntVector va   = ByteVector.fromArray(ByteVector.SPECIES_64, (byte[]) a.get(), a.offset() + i)
                        .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                        .reinterpretAsInts();
                IntVector vb   = ByteVector.fromArray(ByteVector.SPECIES_64, (byte[]) b.get(), b.offset() + i)
                        .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                        .reinterpretAsInts();
                IntVector diff = va.lanewise(VectorOperators.SUB, vb);
                acc = acc.lanewise(VectorOperators.ADD, diff.lanewise(VectorOperators.MUL, diff));
            }

            int result = acc.reduceLanes(VectorOperators.ADD);
            for (int i = limit; i < length; i++) {
                int diff = a.get(i) - b.get(i);
                result += diff * diff;
            }
            return result;
        } else {
            // ByteVector.SPECIES_32 does not exist; scalar is fastest at 128-bit width
            final int length = a.length();
            int result = 0;
            for (int i = 0; i < length; i++) {
                int diff = a.get(i) - b.get(i);
                result += diff * diff;
            }
            return result;
        }
    }

    /**
     * Vectorized cosine similarity between two signed int8 byte vectors.
     */
    @Override
    public float cosine(ByteSequence<?> a, ByteSequence<?> b) {
        // Each register size's loop written out in this one method, compiled on its own (too large to inline);
        // the loops are unchanged.
        if (PREFERRED_BIT_SIZE == 512) {
            final int length = a.length();
            final int step   = ByteVector.SPECIES_128.length();
            final int limit  = ByteVector.SPECIES_128.loopBound(length);
            IntVector dot    = IntVector.zero(IntVector.SPECIES_512);
            IntVector normA  = IntVector.zero(IntVector.SPECIES_512);
            IntVector normB  = IntVector.zero(IntVector.SPECIES_512);

            for (int i = 0; i < limit; i += step) {
                IntVector va = ByteVector.fromArray(ByteVector.SPECIES_128, (byte[]) a.get(), a.offset() + i)
                        .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                        .reinterpretAsInts();
                IntVector vb = ByteVector.fromArray(ByteVector.SPECIES_128, (byte[]) b.get(), b.offset() + i)
                        .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                        .reinterpretAsInts();
                dot   = dot.lanewise(VectorOperators.ADD, va.lanewise(VectorOperators.MUL, vb));
                normA = normA.lanewise(VectorOperators.ADD, va.lanewise(VectorOperators.MUL, va));
                normB = normB.lanewise(VectorOperators.ADD, vb.lanewise(VectorOperators.MUL, vb));
            }

            long dotResult   = dot.reduceLanes(VectorOperators.ADD);
            long normAResult = normA.reduceLanes(VectorOperators.ADD);
            long normBResult = normB.reduceLanes(VectorOperators.ADD);

            for (int i = limit; i < length; i++) {
                int ai = a.get(i), bi = b.get(i);
                dotResult   += ai * bi;
                normAResult += ai * ai;
                normBResult += bi * bi;
            }
            return (float) (dotResult / Math.sqrt((double) normAResult * normBResult));
        } else if (PREFERRED_BIT_SIZE == 256) {
            final int length = a.length();
            final int step   = ByteVector.SPECIES_64.length();
            final int limit  = ByteVector.SPECIES_64.loopBound(length);
            IntVector dot    = IntVector.zero(IntVector.SPECIES_256);
            IntVector normA  = IntVector.zero(IntVector.SPECIES_256);
            IntVector normB  = IntVector.zero(IntVector.SPECIES_256);

            for (int i = 0; i < limit; i += step) {
                IntVector va = ByteVector.fromArray(ByteVector.SPECIES_64, (byte[]) a.get(), a.offset() + i)
                        .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                        .reinterpretAsInts();
                IntVector vb = ByteVector.fromArray(ByteVector.SPECIES_64, (byte[]) b.get(), b.offset() + i)
                        .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                        .reinterpretAsInts();
                dot   = dot.lanewise(VectorOperators.ADD, va.lanewise(VectorOperators.MUL, vb));
                normA = normA.lanewise(VectorOperators.ADD, va.lanewise(VectorOperators.MUL, va));
                normB = normB.lanewise(VectorOperators.ADD, vb.lanewise(VectorOperators.MUL, vb));
            }

            long dotResult   = dot.reduceLanes(VectorOperators.ADD);
            long normAResult = normA.reduceLanes(VectorOperators.ADD);
            long normBResult = normB.reduceLanes(VectorOperators.ADD);

            for (int i = limit; i < length; i++) {
                int ai = a.get(i), bi = b.get(i);
                dotResult   += ai * bi;
                normAResult += ai * ai;
                normBResult += bi * bi;
            }
            return (float) (dotResult / Math.sqrt((double) normAResult * normBResult));
        } else {
            // ByteVector.SPECIES_32 does not exist; scalar is fastest at 128-bit width
            final int length = a.length();
            long dotResult = 0, normAResult = 0, normBResult = 0;
            for (int i = 0; i < length; i++) {
                int ai = a.get(i), bi = b.get(i);
                dotResult   += ai * bi;
                normAResult += ai * ai;
                normBResult += bi * bi;
            }
            return (float) (dotResult / Math.sqrt((double) normAResult * normBResult));
        }
    }

    @Override
    public int hammingDistance(long[] a, long[] b) {
        var sum = LongVector.zero(LongVector.SPECIES_PREFERRED);
        int vectorizedLength = LongVector.SPECIES_PREFERRED.loopBound(a.length);

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += LongVector.SPECIES_PREFERRED.length()) {
            var va = LongVector.fromArray(LongVector.SPECIES_PREFERRED, a, i);
            var vb = LongVector.fromArray(LongVector.SPECIES_PREFERRED, b, i);

            var xorResult = va.lanewise(VectorOperators.XOR, vb);
            sum = sum.lanewise(VectorOperators.ADD, xorResult.lanewise(VectorOperators.BIT_COUNT));
        }

        int res = (int) sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < a.length; i++) {
            res += Long.bitCount(a[i] ^ b[i]);
        }

        return res;
    }

    @Override
    public float max(VectorFloat<?> v) {
        var accum = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, -Float.MAX_VALUE);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v.length());
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v), i);
            accum = accum.lanewise(VectorOperators.MAX, a);
        }
        float max = accum.reduceLanes(VectorOperators.MAX);
        for (int i = vectorizedLength; i < v.length(); i++) {
            max = Math.max(max, v.get(i));
        }
        return max;
    }

    @Override
    public float min(VectorFloat<?> v) {
        var accum = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, Float.MAX_VALUE);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v.length());
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(v), i);
            accum = accum.lanewise(VectorOperators.MIN, a);
        }
        float min = accum.reduceLanes(VectorOperators.MIN);
        for (int i = vectorizedLength; i < v.length(); i++) {
            min = Math.min(min, v.get(i));
        }
        return min;
    }

    private static int combineBytes(int i, int shuffle, ByteSequence<?> quantizedPartials) {
        var lowByte = quantizedPartials.get(i * 512 + shuffle);
        var highByte = quantizedPartials.get((i * 512) + 256 + shuffle);
        return ((Byte.toUnsignedInt(highByte) << 8) | Byte.toUnsignedInt(lowByte));
    }

    private static float combineBytes(int i, int shuffle, VectorFloat<?> partials) {
        return partials.get(i * 256 + shuffle);
    }

    private static int computeSingleShuffle(int codebookPosition, int neighborPosition, ByteSequence<?> shuffles, int codebookCount) {
        int blockSize = ByteVector.SPECIES_PREFERRED.length();

        int blockIndex = neighborPosition / blockSize;
        int positionWithinBlock = neighborPosition % blockSize;
        int offset = blockIndex * blockSize * codebookCount;
        return Byte.toUnsignedInt(shuffles.get(offset + blockSize * codebookPosition + positionWithinBlock));
    }

    /** As {@link #assembleAndSum}: every register size's loop in this one method, compiled on its own; loops unchanged. */
    @Override
    public float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int encodedOffset, int encodedLength, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
        if (PREFERRED_BIT_SIZE == 512) {
            var sum = FloatVector.zero(FloatVector.SPECIES_512);
            var vaMagnitude = FloatVector.zero(FloatVector.SPECIES_512);

            int[] convOffsets = scratchInt512.get();
            int i = 0;
            int limit = i + ByteVector.SPECIES_128.loopBound(encodedLength);

            var scale = IntVector.zero(IntVector.SPECIES_512).addIndex(clusterCount);

            for (; i < limit; i += ByteVector.SPECIES_128.length()) {

                ByteVector.fromArray(ByteVector.SPECIES_128, (byte[]) encoded.get(), i + encoded.offset() + encodedOffset)
                        .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                        .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_512)
                        .reinterpretAsInts()
                        .add(scale)
                        .intoArray(convOffsets, 0);

                var offset = i * clusterCount;
                sum = sum.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_512, arr(partialSums), offset, convOffsets, 0));
                vaMagnitude = vaMagnitude.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_512, arr(aMagnitude), offset, convOffsets, 0));
            }

            float sumResult = sum.reduceLanes(VectorOperators.ADD);
            float aMagnitudeResult = vaMagnitude.reduceLanes(VectorOperators.ADD);

            for (; i < encodedLength; i++) {
                int offset = clusterCount * i + Byte.toUnsignedInt(encoded.get(i + encodedOffset));
                sumResult += partialSums.get(offset);
                aMagnitudeResult += aMagnitude.get(offset);
            }

            return (float) (sumResult / Math.sqrt(aMagnitudeResult * bMagnitude));
        }
        if (PREFERRED_BIT_SIZE == 256) {
            var sum = FloatVector.zero(FloatVector.SPECIES_256);
            var vaMagnitude = FloatVector.zero(FloatVector.SPECIES_256);

            int[] convOffsets = scratchInt256.get();
            int i = 0;
            int limit = ByteVector.SPECIES_64.loopBound(encodedLength);

            var scale = IntVector.zero(IntVector.SPECIES_256).addIndex(clusterCount);

            for (; i < limit; i += ByteVector.SPECIES_64.length()) {

                ByteVector.fromArray(ByteVector.SPECIES_64, (byte[]) encoded.get(), i + encoded.offset() + encodedOffset)
                        .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                        .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_256)
                        .reinterpretAsInts()
                        .add(scale)
                        .intoArray(convOffsets, 0);

                var offset = i * clusterCount;
                sum = sum.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_256, arr(partialSums), offset, convOffsets, 0));
                vaMagnitude = vaMagnitude.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_256, arr(aMagnitude), offset, convOffsets, 0));
            }

            float sumResult = sum.reduceLanes(VectorOperators.ADD);
            float aMagnitudeResult = vaMagnitude.reduceLanes(VectorOperators.ADD);

            for (; i < encodedLength; i++) {
                int offset = clusterCount * i + Byte.toUnsignedInt(encoded.get(i + encodedOffset));
                sumResult += partialSums.get(offset);
                aMagnitudeResult += aMagnitude.get(offset);
            }

            return (float) (sumResult / Math.sqrt(aMagnitudeResult * bMagnitude));
        }
        if (PREFERRED_BIT_SIZE == 128) {
            // benchmarking showed that a 128-bit SIMD implementation performed worse than scalar
            float sum = 0.0f;
            float aMag = 0.0f;

            for (int m = 0; m < encodedLength; ++m) {
                int centroidIndex = Byte.toUnsignedInt(encoded.get(m + encodedOffset));
                var index = m * clusterCount + centroidIndex;
                sum += partialSums.get(index);
                aMag += aMagnitude.get(index);
            }

            return (float) (sum / Math.sqrt(aMag * bMagnitude));
        }
        throw new IllegalStateException("Unsupported vector width: " + PREFERRED_BIT_SIZE);
    }

    //---------------------------------------------
    // NVQ quantization instructions start here
    //---------------------------------------------

    static final FloatVector const1f = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 1.f);
    static final FloatVector const05f = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 0.5f);

    float logisticNQT(float value, float alpha, float x0) {
        float temp = Math.fma(value, alpha, -alpha * x0);
        int p = (int) Math.floor(temp + 1);
        int m = Float.floatToIntBits(Math.fma(temp - p, 0.5f, 1));

        temp = Float.intBitsToFloat(m + (p << 23));  // temp = m * 2^p
        return temp / (temp + 1);
    }

    float logitNQT(float value, float inverseAlpha, float x0) {
        float z = value / (1 - value);

        int temp = Float.floatToIntBits(z);
        int e = temp & 0x7f800000;
        float p = (float) ((e >> 23) - 128);
        float m = Float.intBitsToFloat((temp & 0x007fffff) + 0x3f800000);

        return Math.fma(m + p, inverseAlpha, x0);
    }

    // The 8-bit codes are unpacked FastLanes-style (https://www.vldb.org/pvldb/vol16/p2132-afroozeh.pdf): a byte
    // vector is read as int lanes, and part j of a block holds byte j of every lane (shifted right by 8 * j, masked
    // to 0xff). nvqShuffleQueryInPlace8bit puts the query in the same order. The logistic and logit steps are written
    // out in each kernel rather than called, so no vector is passed to or returned from a method that might not be
    // inlined; the arithmetic is the same as in the scalar logisticNQT and logitNQT.
    @Override
    public void nvqQuantize8bit(VectorFloat<?> vector, float alpha, float x0, float minValue, float maxValue, ByteSequence<?> destination) {
        final int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());
        final var mask = ByteVector.SPECIES_PREFERRED.indexInRange(0, FloatVector.SPECIES_PREFERRED.length());

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var scaledX0 = x0 * delta;
        var logisticBias = logisticNQT(minValue, scaledAlpha, scaledX0);
        var invLogisticScale = 255 / (logisticNQT(maxValue, scaledAlpha, scaledX0) - logisticBias);

        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            // logisticNQT, written out in place (see the scalar version)
            var x = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(vector), i);
            var lt = x.lanewise(VectorOperators.FMA, scaledAlpha, -scaledAlpha * scaledX0);
            var lp = lt.lanewise(VectorOperators.ADD, 1, lt.test(VectorOperators.IS_NEGATIVE).not())
                    .convert(VectorOperators.F2I, 0).reinterpretAsInts();
            var lm = lt.lanewise(VectorOperators.SUB, lp.convert(VectorOperators.I2F, 0).reinterpretAsFloats()).lanewise(VectorOperators.FMA, 0.5f, 1).reinterpretAsInts();
            lt = lm.lanewise(VectorOperators.ADD, lp.lanewise(VectorOperators.LSHL, 23)).reinterpretAsFloats();
            var arr = lt.lanewise(VectorOperators.DIV, lt.lanewise(VectorOperators.ADD, 1));
            arr = arr.lanewise(VectorOperators.SUB, logisticBias).lanewise(VectorOperators.MUL, invLogisticScale);
            var bytes = arr.lanewise(VectorOperators.ADD, const05f)
                    .convertShape(VectorOperators.F2B, ByteVector.SPECIES_PREFERRED, 0)
                    .reinterpretAsBytes();

            bytes.intoArray((byte[]) destination.get(), i, mask);
        }

        // Process the tail
        for (int d = vectorizedLength; d < vector.length(); d++) {
            // Ensure the quantized value is within the 0 to constant range
            float value = vector.get(d);
            value = logisticNQT(value, scaledAlpha, scaledX0);
            value = (value - logisticBias) * invLogisticScale;
            int quantizedValue = Math.round(value);
            destination.set(d, (byte) quantizedValue);
        }
    }

    @Override
    public float nvqLoss(VectorFloat<?> vector, float alpha, float x0, float minValue, float maxValue, int nBits) {
        int constant = (1 << nBits) - 1;
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        FloatVector squaredSumVec = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var invScaledAlpha = 1 / scaledAlpha;
        var scaledX0 = x0 * delta;
        var logisticBias = logisticNQT(minValue, scaledAlpha, scaledX0);
        var logisticScale = (logisticNQT(maxValue, scaledAlpha, scaledX0) - logisticBias) / constant;
        var invLogisticScale = 1 / logisticScale;

        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            // logisticNQT, written out in place (see the scalar version)
            var arr = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(vector), i);
            var lt = arr.lanewise(VectorOperators.FMA, scaledAlpha, -scaledAlpha * scaledX0);
            var lp = lt.lanewise(VectorOperators.ADD, 1, lt.test(VectorOperators.IS_NEGATIVE).not())
                    .convert(VectorOperators.F2I, 0).reinterpretAsInts();
            var lm = lt.lanewise(VectorOperators.SUB, lp.convert(VectorOperators.I2F, 0).reinterpretAsFloats()).lanewise(VectorOperators.FMA, 0.5f, 1).reinterpretAsInts();
            lt = lm.lanewise(VectorOperators.ADD, lp.lanewise(VectorOperators.LSHL, 23)).reinterpretAsFloats();
            var lr = lt.lanewise(VectorOperators.DIV, lt.lanewise(VectorOperators.ADD, 1));
            var rq = lr.lanewise(VectorOperators.SUB, logisticBias).lanewise(VectorOperators.MUL, invLogisticScale).lanewise(VectorOperators.ADD, const05f)
                    .convert(VectorOperators.F2I, 0)
                    .reinterpretAsInts()
                    .convert(VectorOperators.I2F, 0)
                    .reinterpretAsFloats()
                    .fma(logisticScale, logisticBias);
            // logitNQT, written out in place (see the scalar version)
            var zrecArr = rq.lanewise(VectorOperators.DIV, const1f.lanewise(VectorOperators.SUB, rq)).reinterpretAsInts();
            var precArr = zrecArr.lanewise(VectorOperators.AND, 0x7f800000).lanewise(VectorOperators.LSHR, 23).lanewise(VectorOperators.SUB, 128)
                    .convert(VectorOperators.I2F, 0).reinterpretAsFloats();
            var recArr = zrecArr.lanewise(VectorOperators.AND, 0x007fffff).lanewise(VectorOperators.ADD, 0x3f800000).reinterpretAsFloats()
                    .add(precArr).lanewise(VectorOperators.FMA, invScaledAlpha, scaledX0);

            var diff = arr.lanewise(VectorOperators.SUB, recArr);
            squaredSumVec = diff.lanewise(VectorOperators.FMA, diff, squaredSumVec);
        }

        float squaredSum = squaredSumVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value, recValue;
        for (int i = vectorizedLength; i < vector.length(); i++) {
            value = vector.get(i);

            recValue = logisticNQT(value, scaledAlpha, scaledX0);
            recValue = (recValue - logisticBias) * invLogisticScale;
            recValue = Math.round(recValue);
            recValue = Math.fma(logisticScale, recValue, logisticBias);
            recValue = logitNQT(recValue, invScaledAlpha, scaledX0);

            squaredSum += MathUtil.square(value - recValue);
        }

        return squaredSum;
    }

    @Override
    public float nvqUniformLoss(VectorFloat<?> vector, float minValue, float maxValue, int nBits) {
        float constant = (1 << nBits) - 1;
        float delta = maxValue - minValue;

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        FloatVector squaredSumVec = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var arr = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(vector), i);
            var recArr = arr.lanewise(VectorOperators.SUB, minValue).lanewise(VectorOperators.MUL, constant / delta);
            recArr = recArr.lanewise(VectorOperators.ADD, const05f)
                    .convert(VectorOperators.F2I, 0)
                    .reinterpretAsInts()
                    .convert(VectorOperators.I2F, 0)
                    .reinterpretAsFloats();
            recArr = recArr.lanewise(VectorOperators.FMA, delta / constant, minValue);

            var diff = arr.lanewise(VectorOperators.SUB, recArr);
            squaredSumVec = diff.lanewise(VectorOperators.FMA, diff, squaredSumVec);
        }

        float squaredSum = squaredSumVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value, recValue;
        for (int i = vectorizedLength; i < vector.length(); i++) {
            value = vector.get(i);

            recValue = (value - minValue) / delta;
            recValue = Math.round(constant * recValue) / constant;
            recValue = recValue * delta + minValue;

            squaredSum += MathUtil.square(value - recValue);
        }

        return squaredSum;
    }

    @Override
    public float nvqSquareL2Distance8bit(VectorFloat<?> vector, ByteSequence<?> quantizedVector,
                                         float alpha, float x0, float minValue, float maxValue) {
        FloatVector squaredSumVec = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = ByteVector.SPECIES_PREFERRED.loopBound(quantizedVector.length());
        int floatStep = FloatVector.SPECIES_PREFERRED.length();

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var invScaledAlpha = 1 / scaledAlpha;
        var scaledX0 = x0 * delta;
        var logisticBias = logisticNQT(minValue, scaledAlpha, scaledX0);
        var logisticScale = (logisticNQT(maxValue, scaledAlpha, scaledX0) - logisticBias) / 255;

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_PREFERRED.length()) {
            var byteArr = ByteVector.fromArray(ByteVector.SPECIES_PREFERRED, (byte[]) quantizedVector.get(), i);

            for (int j = 0; j < 4; j++) {
                var v1 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(vector), i + floatStep * j);
                // Byte j of each int lane (see the note above nvqQuantize8bit), scaled back from 0..255
                var qv2 = byteArr.reinterpretAsInts().lanewise(VectorOperators.LSHR, 8 * j).lanewise(VectorOperators.AND, 0xff)
                        .convert(VectorOperators.I2F, 0).reinterpretAsFloats().lanewise(VectorOperators.FMA, logisticScale, logisticBias);
                // logitNQT, written out in place (see the scalar version)
                var zv2 = qv2.lanewise(VectorOperators.DIV, const1f.lanewise(VectorOperators.SUB, qv2)).reinterpretAsInts();
                var pv2 = zv2.lanewise(VectorOperators.AND, 0x7f800000).lanewise(VectorOperators.LSHR, 23).lanewise(VectorOperators.SUB, 128)
                        .convert(VectorOperators.I2F, 0).reinterpretAsFloats();
                var v2 = zv2.lanewise(VectorOperators.AND, 0x007fffff).lanewise(VectorOperators.ADD, 0x3f800000).reinterpretAsFloats()
                        .add(pv2).lanewise(VectorOperators.FMA, invScaledAlpha, scaledX0);

                var diff = v1.lanewise(VectorOperators.SUB, v2);
                squaredSumVec = diff.lanewise(VectorOperators.FMA, diff, squaredSumVec);
            }
        }

        float squaredSum = squaredSumVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value2, diff;
        for (int i = vectorizedLength; i < quantizedVector.length(); i++) {
            value2 = Byte.toUnsignedInt(quantizedVector.get(i));
            value2 = Math.fma(logisticScale, value2, logisticBias);
            value2 = logitNQT(value2, invScaledAlpha, scaledX0);
            diff = vector.get(i) - value2;
            squaredSum += MathUtil.square(diff);
        }

        return squaredSum;
    }

    @Override
    public float nvqDotProduct8bit(VectorFloat<?> vector, ByteSequence<?> quantizedVector,
                                   float alpha, float x0, float minValue, float maxValue) {
        FloatVector dotProdVec = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = ByteVector.SPECIES_PREFERRED.loopBound(quantizedVector.length());
        int floatStep = FloatVector.SPECIES_PREFERRED.length();

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var invScaledAlpha = 1 / scaledAlpha;
        var scaledX0 = x0 * delta;
        var logisticBias = logisticNQT(minValue, scaledAlpha, scaledX0);
        var logisticScale = (logisticNQT(maxValue, scaledAlpha, scaledX0) - logisticBias) / 255;

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_PREFERRED.length()) {
            var byteArr = ByteVector.fromArray(ByteVector.SPECIES_PREFERRED, (byte[]) quantizedVector.get(), i);

            for (int j = 0; j < 4; j++) {
                var v1 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(vector), i + floatStep * j);
                // Byte j of each int lane (see the note above nvqQuantize8bit), scaled back from 0..255
                var qv2 = byteArr.reinterpretAsInts().lanewise(VectorOperators.LSHR, 8 * j).lanewise(VectorOperators.AND, 0xff)
                        .convert(VectorOperators.I2F, 0).reinterpretAsFloats().lanewise(VectorOperators.FMA, logisticScale, logisticBias);
                // logitNQT, written out in place (see the scalar version)
                var zv2 = qv2.lanewise(VectorOperators.DIV, const1f.lanewise(VectorOperators.SUB, qv2)).reinterpretAsInts();
                var pv2 = zv2.lanewise(VectorOperators.AND, 0x7f800000).lanewise(VectorOperators.LSHR, 23).lanewise(VectorOperators.SUB, 128)
                        .convert(VectorOperators.I2F, 0).reinterpretAsFloats();
                var v2 = zv2.lanewise(VectorOperators.AND, 0x007fffff).lanewise(VectorOperators.ADD, 0x3f800000).reinterpretAsFloats()
                        .add(pv2).lanewise(VectorOperators.FMA, invScaledAlpha, scaledX0);
                dotProdVec = v1.lanewise(VectorOperators.FMA, v2, dotProdVec);
            }
        }

        float dotProd = dotProdVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value2;
        for (int i = vectorizedLength; i < quantizedVector.length(); i++) {
            value2 = Byte.toUnsignedInt(quantizedVector.get(i));
            value2 = Math.fma(logisticScale, value2, logisticBias);
            value2 = logitNQT(value2, invScaledAlpha, scaledX0);
            dotProd = Math.fma(vector.get(i), value2, dotProd);
        }

        return dotProd;
    }

    @Override
    public float[] nvqCosine8bit(VectorFloat<?> vector,
                                 ByteSequence<?> quantizedVector, float alpha, float x0, float minValue, float maxValue,
                                 VectorFloat<?> centroid) {
        if (vector.length() != centroid.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var invScaledAlpha = 1 / scaledAlpha;
        var scaledX0 = x0 * delta;
        var logisticBias = logisticNQT(minValue, scaledAlpha, scaledX0);
        var logisticScale = (logisticNQT(maxValue, scaledAlpha, scaledX0) - logisticBias) / 255;

        var vsum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vbMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = ByteVector.SPECIES_PREFERRED.loopBound(vector.length());
        int floatStep = FloatVector.SPECIES_PREFERRED.length();

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_PREFERRED.length()) {
            var byteArr = ByteVector.fromArray(ByteVector.SPECIES_PREFERRED, (byte[]) quantizedVector.get(), i);

            for (int j = 0; j < 4; j++) {
                var va = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(vector), i + floatStep * j);
                // Byte j of each int lane (see the note above nvqQuantize8bit), scaled back from 0..255
                var qvd = byteArr.reinterpretAsInts().lanewise(VectorOperators.LSHR, 8 * j).lanewise(VectorOperators.AND, 0xff)
                        .convert(VectorOperators.I2F, 0).reinterpretAsFloats().lanewise(VectorOperators.FMA, logisticScale, logisticBias);
                // logitNQT, written out in place (see the scalar version)
                var zvd = qvd.lanewise(VectorOperators.DIV, const1f.lanewise(VectorOperators.SUB, qvd)).reinterpretAsInts();
                var pvd = zvd.lanewise(VectorOperators.AND, 0x7f800000).lanewise(VectorOperators.LSHR, 23).lanewise(VectorOperators.SUB, 128)
                        .convert(VectorOperators.I2F, 0).reinterpretAsFloats();
                var vd = zvd.lanewise(VectorOperators.AND, 0x007fffff).lanewise(VectorOperators.ADD, 0x3f800000).reinterpretAsFloats()
                        .add(pvd).lanewise(VectorOperators.FMA, invScaledAlpha, scaledX0);

                var vb = vd.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, arr(centroid), i + floatStep * j));

                vsum = va.lanewise(VectorOperators.FMA, vb, vsum);
                vbMagnitude = vb.lanewise(VectorOperators.FMA, vb, vbMagnitude);
            }
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);
        float bMagnitude = vbMagnitude.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value2;
        for (int i = vectorizedLength; i < vector.length(); i++) {
            value2 = Byte.toUnsignedInt(quantizedVector.get(i));
            value2 = Math.fma(logisticScale, value2, logisticBias);
            value2 = logitNQT(value2, invScaledAlpha, scaledX0) + centroid.get(i);
            sum = Math.fma(vector.get(i), value2, sum);
            bMagnitude = Math.fma(value2, value2, bMagnitude);
        }

        // TODO can we avoid returning a new array?
        return new float[]{sum, bMagnitude};
    }

    void transpose(VectorFloat<?> arr, int first, int last, int nRows) {
        final int mn1 = (last - first - 1);
        final int n = (last - first) / nRows;
        boolean[] visited = new boolean[last - first];
        float temp;
        int cycle = first;
        while (++cycle != last) {
            if (visited[cycle - first])
                continue;
            int a = cycle - first;
            do {
                a = a == mn1 ? mn1 : (n * a) % mn1;
                temp = arr.get(first + a);
                arr.set(first + a, arr.get(cycle));
                arr.set(cycle, temp);
                visited[a] = true;
            } while ((first + a) != cycle);
        }
    }

    @Override
    public void nvqShuffleQueryInPlace8bit(VectorFloat<?> vector) {
        // To understand this shuffle, see nvqDequantize8bit

        final int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());
        final int step = FloatVector.SPECIES_PREFERRED.length() * 4;

        for (int i = 0; i + step <= vectorizedLength; i += step) {
            transpose(vector, i, i + step, 4);
        }
    }

    // PQ query tables. Every search builds one: for each subspace, the query's dot product with (or squared distance
    // to) each of its centroids. From codebooks transposed so that one vector holds the same dimension of consecutive
    // centroids, every score builds up lane by lane with multiply-adds against the broadcast query value: no
    // reductions, no per-centroid calls. Scoring centroid by centroid through dotProduct put the vector operations
    // deep below the decoder, where C2 can leave them as calls that put every centroid's vectors on the heap.

    @Override
    public boolean supportsTransposedPartialSums() {
        return true;
    }

    @Override
    public void calculatePartialSumsTransposed(VectorFloat<?> transposed, int[] subspaceSizes, int[] subspaceOffsets,
                                               int clusterCount, VectorFloat<?> query, VectorFloat<?> center,
                                               VectorSimilarityFunction vsf, VectorFloat<?> partialSums) {
        boolean dot;
        switch (vsf) {
            case DOT_PRODUCT: dot = true; break;
            case EUCLIDEAN: dot = false; break;
            default: throw new UnsupportedOperationException("Unsupported similarity function " + vsf);
        }
        float[] t = arr(transposed), q = arr(query), mu = arr(center), out = arr(partialSums);
        int subspaces = subspaceSizes.length, dimension = 0;
        if (subspaceOffsets.length != subspaces) throw new IllegalArgumentException("subspace sizes and offsets differ in length");
        for (int m = 0; m < subspaces; m++) dimension = Math.max(dimension, subspaceSizes[m] + subspaceOffsets[m]);
        Objects.checkFromIndexSize(0, dimension, q.length);
        Objects.checkFromIndexSize(0, dimension, mu.length);
        Objects.checkFromIndexSize(0, Math.multiplyExact(clusterCount, dimension), t.length);
        Objects.checkFromIndexSize(0, Math.multiplyExact(clusterCount, subspaces), out.length);
        if (dot) transposedDotSums(t, subspaceSizes, subspaceOffsets, clusterCount, q, mu, out);
        else transposedSquareSums(t, subspaceSizes, subspaceOffsets, clusterCount, q, mu, out);
    }

    // The exact vector class, so a store states its vector's shape instead of relying on C2's type profile, which some
    // compiles lose for a loop-carried accumulator (the store then puts the vector on the heap).
    private static final Class<?> PREFERRED_FLOAT_CLASS = FloatVector.SPECIES_PREFERRED.vectorType();

    /**
     * The whole DOT_PRODUCT query table from transposed codebooks. Four vectors of centroids are in flight at once; the last centroids use a masked
     * vector. The query is centered here, {@code q[i] - mu[i]}, with the arithmetic of subtracting the vectors first.
     * One operation per body and the species named directly, never through a local, so every compile of the body
     * (including one entered mid-loop) knows each vector's shape; compiled on its own (above C2's inline limit).
     */
    private static void transposedDotSums(float[] t, int[] subspaceSizes, int[] subspaceOffsets, int k, float[] q, float[] mu,
                                  float[] out) {
        final int lanes = FloatVector.SPECIES_PREFERRED.length();
        for (int m = 0; m < subspaceSizes.length; m++) {
            int size = subspaceSizes[m], offset = subspaceOffsets[m];
            int base = k * offset, outBase = m * k;
            int c = 0;
            for (; c + 4 * lanes <= k; c += 4 * lanes) {
                int row = base + c;
                var x = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, q[offset] - mu[offset]);
                var acc0 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row).lanewise(VectorOperators.MUL, x);
                var acc1 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row + lanes).lanewise(VectorOperators.MUL, x);
                var acc2 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row + 2 * lanes).lanewise(VectorOperators.MUL, x);
                var acc3 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row + 3 * lanes).lanewise(VectorOperators.MUL, x);
                for (int d = 1; d < size; d++) {
                    row += k;
                    x = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, q[offset + d] - mu[offset + d]);
                    acc0 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row).lanewise(VectorOperators.FMA, x, acc0);
                    acc1 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row + lanes).lanewise(VectorOperators.FMA, x, acc1);
                    acc2 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row + 2 * lanes).lanewise(VectorOperators.FMA, x, acc2);
                    acc3 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row + 3 * lanes).lanewise(VectorOperators.FMA, x, acc3);
                }
                ((FloatVector) PREFERRED_FLOAT_CLASS.cast(acc0)).intoArray(out, outBase + c);
                ((FloatVector) PREFERRED_FLOAT_CLASS.cast(acc1)).intoArray(out, outBase + c + lanes);
                ((FloatVector) PREFERRED_FLOAT_CLASS.cast(acc2)).intoArray(out, outBase + c + 2 * lanes);
                ((FloatVector) PREFERRED_FLOAT_CLASS.cast(acc3)).intoArray(out, outBase + c + 3 * lanes);
            }
            for (; c < k; c += lanes) {
                var mask = FloatVector.SPECIES_PREFERRED.indexInRange(c, k);
                int row = base + c;
                var acc = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row, mask)
                        .lanewise(VectorOperators.MUL, FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, q[offset] - mu[offset]));
                for (int d = 1; d < size; d++) {
                    row += k;
                    acc = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row, mask)
                            .lanewise(VectorOperators.FMA, FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, q[offset + d] - mu[offset + d]), acc);
                }
                ((FloatVector) PREFERRED_FLOAT_CLASS.cast(acc)).intoArray(out, outBase + c, mask);
            }
        }
    }

    /**
     * The whole EUCLIDEAN query table from transposed codebooks. Four vectors of centroids are in flight at once; the last centroids use a masked
     * vector. The query is centered here, {@code q[i] - mu[i]}, with the arithmetic of subtracting the vectors first.
     * One operation per body and the species named directly, never through a local, so every compile of the body
     * (including one entered mid-loop) knows each vector's shape; compiled on its own (above C2's inline limit).
     */
    private static void transposedSquareSums(float[] t, int[] subspaceSizes, int[] subspaceOffsets, int k, float[] q, float[] mu,
                                  float[] out) {
        final int lanes = FloatVector.SPECIES_PREFERRED.length();
        for (int m = 0; m < subspaceSizes.length; m++) {
            int size = subspaceSizes[m], offset = subspaceOffsets[m];
            int base = k * offset, outBase = m * k;
            int c = 0;
            for (; c + 4 * lanes <= k; c += 4 * lanes) {
                int row = base + c;
                var x = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, q[offset] - mu[offset]);
                var d0 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row).lanewise(VectorOperators.SUB, x);
                var d1 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row + lanes).lanewise(VectorOperators.SUB, x);
                var d2 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row + 2 * lanes).lanewise(VectorOperators.SUB, x);
                var d3 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row + 3 * lanes).lanewise(VectorOperators.SUB, x);
                var acc0 = d0.lanewise(VectorOperators.MUL, d0);
                var acc1 = d1.lanewise(VectorOperators.MUL, d1);
                var acc2 = d2.lanewise(VectorOperators.MUL, d2);
                var acc3 = d3.lanewise(VectorOperators.MUL, d3);
                for (int d = 1; d < size; d++) {
                    row += k;
                    x = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, q[offset + d] - mu[offset + d]);
                    d0 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row).lanewise(VectorOperators.SUB, x);
                    d1 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row + lanes).lanewise(VectorOperators.SUB, x);
                    d2 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row + 2 * lanes).lanewise(VectorOperators.SUB, x);
                    d3 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row + 3 * lanes).lanewise(VectorOperators.SUB, x);
                    acc0 = d0.lanewise(VectorOperators.FMA, d0, acc0);
                    acc1 = d1.lanewise(VectorOperators.FMA, d1, acc1);
                    acc2 = d2.lanewise(VectorOperators.FMA, d2, acc2);
                    acc3 = d3.lanewise(VectorOperators.FMA, d3, acc3);
                }
                ((FloatVector) PREFERRED_FLOAT_CLASS.cast(acc0)).intoArray(out, outBase + c);
                ((FloatVector) PREFERRED_FLOAT_CLASS.cast(acc1)).intoArray(out, outBase + c + lanes);
                ((FloatVector) PREFERRED_FLOAT_CLASS.cast(acc2)).intoArray(out, outBase + c + 2 * lanes);
                ((FloatVector) PREFERRED_FLOAT_CLASS.cast(acc3)).intoArray(out, outBase + c + 3 * lanes);
            }
            for (; c < k; c += lanes) {
                var mask = FloatVector.SPECIES_PREFERRED.indexInRange(c, k);
                int row = base + c;
                var diff = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row, mask)
                        .lanewise(VectorOperators.SUB, FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, q[offset] - mu[offset]));
                var acc = diff.lanewise(VectorOperators.MUL, diff);
                for (int d = 1; d < size; d++) {
                    row += k;
                    diff = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, t, row, mask)
                            .lanewise(VectorOperators.SUB, FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, q[offset + d] - mu[offset + d]));
                    acc = diff.lanewise(VectorOperators.FMA, diff, acc);
                }
                ((FloatVector) PREFERRED_FLOAT_CLASS.cast(acc)).intoArray(out, outBase + c, mask);
            }
        }
    }

    @Override
    public void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int queryOffset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums) {
        int codebookBase = codebookIndex * clusterCount;
        for (int i = 0; i < clusterCount; i++) {
            switch (vsf) {
                case DOT_PRODUCT:
                    partialSums.set(codebookBase + i, dotProduct(codebook, i * size, query, queryOffset, size));
                    break;
                case EUCLIDEAN:
                    partialSums.set(codebookBase + i, squareDistance(codebook, i * size, query, queryOffset, size));
                    break;
                default:
                    throw new UnsupportedOperationException("Unsupported similarity function " + vsf);
            }
        }
    }

    @Override
    public float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
        return pqDecodedCosineSimilarity(encoded, 0, encoded.length(), clusterCount, partialSums, aMagnitude, bMagnitude);
    }

    @Override
    public float ashDotRow(float[] Arow, float[] x) {
        assert Arow.length == x.length : "Arow.length != x.length";

        final int len = x.length;
        final int upper = FloatVector.SPECIES_PREFERRED.loopBound(len);

        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int i = 0;
        for (; i < upper; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector va = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, Arow, i);
            FloatVector vx = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, x, i);
            sum = va.lanewise(VectorOperators.FMA, vx, sum);
        }

        float acc = sum.reduceLanes(VectorOperators.ADD);

        // scalar tail
        for (; i < len; i++) {
            acc += Arow[i] * x[i];
        }

        return acc;
    }

    /** Whole-matrix, batch and lower-precision query projections: {@link PanamaASHProjection}. */
    @Override
    public AshProjection prepareAshProjection(float[][] matrix) {
        return PanamaASHProjection.supported() ? new PanamaASHProjection(matrix) : VectorUtilSupport.super.prepareAshProjection(matrix);
    }

    @Override
    public boolean supportsAshMaskedLoad() {
        return ASH_SIMD;
    }

    private static final int ASH_512_ACCUMULATORS =
            Integer.getInteger("jvector.ash.512.accumulators", 8);

    /** Species-width-independent masked add; the fallback for machines without 16 float lanes. */
    private float ashMaskedAddFlat(float[] tildeQ,
                                  int qOffset,
                                  long[] allPackedVectors,
                                  int packedBase,
                                  int d,
                                  int words) {
        final int LANES = FloatVector.SPECIES_PREFERRED.length();

        // 4 Accumulators to hide instruction latency (Pipeline depth ~4)
        FloatVector acc0 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        FloatVector acc1 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        FloatVector acc2 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        FloatVector acc3 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        // Pre-calculate mask extraction constants based on LANES
        long laneMask;
        if (LANES == 16)      laneMask = 0xFFFFL;       // AVX-512
        else if (LANES == 8)  laneMask = 0xFFL;         // AVX2
        else if (LANES == 4)  laneMask = 0xFL;          // NEON / SSE
        else                  laneMask = (1L << LANES) - 1;

        // Safe limit for Fast Path
        int safeWords = d / 64;

        for (int w = 0; w < safeWords; w++) {
            long wordBits = allPackedVectors[packedBase + w];
            if (wordBits == 0) continue; // Branch prediction handles this well

            int baseDim = w * 64;

            // Iterate 64 bits in steps of LANES (e.g., 0, 8, 16, 24... for AVX2)
            // We unroll manually inside the loop using modulo to pick accumulators
            // The JIT is smart enough to unroll this loop completely since LANES is constant.
            int accumulatorIdx = 0;

            for (int off = 0; off < 64; off += LANES) {
                // Shift and mask
                long m = (wordBits >>> off) & laneMask;

                if (m != 0) {
                    VectorMask<Float> bitMask = VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, m);
                    FloatVector loaded = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qOffset + baseDim + off);

                    // Round-robin distribution to accumulators to break dependency chains
                    switch (accumulatorIdx & 3) { // (idx % 4)
                        case 0 -> acc0 = acc0.lanewise(VectorOperators.ADD, loaded, bitMask);
                        case 1 -> acc1 = acc1.lanewise(VectorOperators.ADD, loaded, bitMask);
                        case 2 -> acc2 = acc2.lanewise(VectorOperators.ADD, loaded, bitMask);
                        case 3 -> acc3 = acc3.lanewise(VectorOperators.ADD, loaded, bitMask);
                    }
                }
                accumulatorIdx++;
            }
        }

        for (int w = safeWords; w < words; w++) {
            long wordBits = allPackedVectors[packedBase + w];
            if (wordBits == 0) continue;

            int baseDim = w * 64;

            for (int off = 0; off < 64; off += LANES) {
                int localIdx = baseDim + off;
                if (localIdx >= d) break;

                long m = (wordBits >>> off) & laneMask;
                if (m == 0) continue;

                VectorMask<Float> bitMask = VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, m);
                int absIdx = qOffset + localIdx;

                if (localIdx + LANES <= d) {
                    // Safe unmasked load
                    acc0 = acc0.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, absIdx), bitMask);
                } else {
                    // Boundary masked load
                    VectorMask<Float> rangeMask = FloatVector.SPECIES_PREFERRED.indexInRange(absIdx, qOffset + d);
                    acc0 = acc0.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, absIdx, rangeMask),
                            bitMask.and(rangeMask));
                }
            }
        }

        // Reduce all accumulators
        // Fixed-order lane sum, as in the note above ROTATE_32.
        var laneSum = acc0.lanewise(VectorOperators.ADD, acc1).lanewise(VectorOperators.ADD, acc2).lanewise(VectorOperators.ADD, acc3);
        if (ROTATE_32 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_32));
        if (ROTATE_16 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_16));
        if (ROTATE_8 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_8));
        if (ROTATE_4 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_4));
        if (ROTATE_2 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_2));
        laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_1));
        return laneSum.lane(0);
    }

    public float ashMaskedAdd_512(
            float[] tildeQ,
            int qOffset,
            long[] allPackedVectors,
            int packedBase,
            int d,
            int words) {
        return ashMaskedAdd_512(
                tildeQ,
                qOffset,
                allPackedVectors,
                packedBase,
                d,
                words,
                ASH_512_ACCUMULATORS);
    }

    /**
     * Package-private overload for correctness tests and microbenchmarks.
     */
    float ashMaskedAdd_512(
            float[] tildeQ,
            int qOffset,
            long[] allPackedVectors,
            int packedBase,
            int d,
            int words,
            int accumulatorCount) {
        if (d < 0) {
            throw new IllegalArgumentException("d must be nonnegative: " + d);
        }

        int requiredWords = d / 64 + (d % 64 == 0 ? 0 : 1);
        if (words < requiredWords) {
            throw new IllegalArgumentException(
                    "Not enough packed words: required="
                            + requiredWords
                            + ", provided="
                            + words);
        }

        Objects.checkFromIndexSize(qOffset, d, tildeQ.length);
        Objects.checkFromIndexSize(packedBase, requiredWords, allPackedVectors.length);
        // The unrolled kernels below address sixteen float lanes per vector.
        // Other machines must use the species-width-independent implementation.
        if (FloatVector.SPECIES_PREFERRED.length() != 16) {
            if (accumulatorCount != 4 && accumulatorCount != 8) {
                throw new IllegalArgumentException("accumulatorCount must be 4 or 8");
            }
            return ashMaskedAddFlat(tildeQ, qOffset, allPackedVectors, packedBase, d, requiredWords);
        }

        if (accumulatorCount == 4) {
            return ashMaskedAdd4(
                    tildeQ,
                    qOffset,
                    allPackedVectors,
                    packedBase,
                    d);
        }

        if (accumulatorCount == 8) {
            return ashMaskedAdd8(
                    tildeQ,
                    qOffset,
                    allPackedVectors,
                    packedBase,
                    d);
        }

        throw new IllegalArgumentException(
                "ASH 512 accumulator count must be 4 or 8: "
                        + accumulatorCount);
    }

    private float ashMaskedAdd4(
            float[] tildeQ,
            int qOffset,
            long[] packedVectors,
            int packedBase,
            int d) {

        FloatVector acc0 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        FloatVector acc1 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        FloatVector acc2 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        FloatVector acc3 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int fullWords = d >>> 6;

        for (int w = 0; w < fullWords; w++) {
            int qBase = qOffset + (w << 6);
            long packed = packedVectors[packedBase + w];

            acc0 = acc0.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase),
                    VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed >>> 0) & 0xFFFFL));
            acc1 = acc1.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase + 16),
                    VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed >>> 16) & 0xFFFFL));
            acc2 = acc2.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase + 32),
                    VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed >>> 32) & 0xFFFFL));
            acc3 = acc3.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase + 48),
                    VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed >>> 48) & 0xFFFFL));
        }

        int remainder = d & 63;
        if (remainder != 0) {
            int qBase = qOffset + (fullWords << 6);
            long packed = packedVectors[packedBase + fullWords];

            if (remainder > 0) {
                VectorMask<Float> valid = FloatVector.SPECIES_PREFERRED.indexInRange(0, Math.min(remainder, 16));
                acc0 = acc0.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase, valid),
                        VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed >>> 0) & 0xFFFFL).and(valid));
            }
            if (remainder > 16) {
                VectorMask<Float> valid = FloatVector.SPECIES_PREFERRED.indexInRange(0, Math.min(remainder - 16, 16));
                acc1 = acc1.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase + 16, valid),
                        VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed >>> 16) & 0xFFFFL).and(valid));
            }
            if (remainder > 32) {
                VectorMask<Float> valid = FloatVector.SPECIES_PREFERRED.indexInRange(0, Math.min(remainder - 32, 16));
                acc2 = acc2.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase + 32, valid),
                        VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed >>> 32) & 0xFFFFL).and(valid));
            }
            if (remainder > 48) {
                VectorMask<Float> valid = FloatVector.SPECIES_PREFERRED.indexInRange(0, remainder - 48);
                acc3 = acc3.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase + 48, valid),
                        VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed >>> 48) & 0xFFFFL).and(valid));
            }
        }

        // Fixed-order lane sum, as in the note above ROTATE_32.
        var laneSum = acc0.lanewise(VectorOperators.ADD, acc1).lanewise(VectorOperators.ADD, acc2.lanewise(VectorOperators.ADD, acc3));
        if (ROTATE_32 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_32));
        if (ROTATE_16 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_16));
        if (ROTATE_8 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_8));
        if (ROTATE_4 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_4));
        if (ROTATE_2 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_2));
        laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_1));
        return laneSum.lane(0);
    }

    private float ashMaskedAdd8(
            float[] tildeQ,
            int qOffset,
            long[] packedVectors,
            int packedBase,
            int d) {

        FloatVector acc0 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        FloatVector acc1 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        FloatVector acc2 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        FloatVector acc3 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        FloatVector acc4 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        FloatVector acc5 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        FloatVector acc6 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        FloatVector acc7 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int fullWords = d >>> 6;
        int w = 0;

        // Two packed words produce eight independent SIMD additions.
        for (; w + 1 < fullWords; w += 2) {
            int qBase = qOffset + (w << 6);

            long packed0 = packedVectors[packedBase + w];
            acc0 = acc0.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase),
                    VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed0 >>> 0) & 0xFFFFL));
            acc1 = acc1.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase + 16),
                    VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed0 >>> 16) & 0xFFFFL));
            acc2 = acc2.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase + 32),
                    VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed0 >>> 32) & 0xFFFFL));
            acc3 = acc3.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase + 48),
                    VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed0 >>> 48) & 0xFFFFL));

            long packed1 = packedVectors[packedBase + w + 1];
            int qBase1 = qBase + 64;

            acc4 = acc4.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase1),
                    VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed1 >>> 0) & 0xFFFFL));
            acc5 = acc5.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase1 + 16),
                    VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed1 >>> 16) & 0xFFFFL));
            acc6 = acc6.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase1 + 32),
                    VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed1 >>> 32) & 0xFFFFL));
            acc7 = acc7.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase1 + 48),
                    VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed1 >>> 48) & 0xFFFFL));
        }

        // One remaining complete word.
        if (w < fullWords) {
            int qBase = qOffset + (w << 6);
            long packed = packedVectors[packedBase + w];

            acc0 = acc0.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase),
                    VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed >>> 0) & 0xFFFFL));
            acc1 = acc1.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase + 16),
                    VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed >>> 16) & 0xFFFFL));
            acc2 = acc2.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase + 32),
                    VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed >>> 32) & 0xFFFFL));
            acc3 = acc3.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase + 48),
                    VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed >>> 48) & 0xFFFFL));
        }

        int remainder = d & 63;
        if (remainder != 0) {
            int qBase = qOffset + (fullWords << 6);
            long packed = packedVectors[packedBase + fullWords];

            if (remainder > 0) {
                VectorMask<Float> valid = FloatVector.SPECIES_PREFERRED.indexInRange(0, Math.min(remainder, 16));
                acc4 = acc4.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase, valid),
                        VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed >>> 0) & 0xFFFFL).and(valid));
            }
            if (remainder > 16) {
                VectorMask<Float> valid = FloatVector.SPECIES_PREFERRED.indexInRange(0, Math.min(remainder - 16, 16));
                acc5 = acc5.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase + 16, valid),
                        VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed >>> 16) & 0xFFFFL).and(valid));
            }
            if (remainder > 32) {
                VectorMask<Float> valid = FloatVector.SPECIES_PREFERRED.indexInRange(0, Math.min(remainder - 32, 16));
                acc6 = acc6.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase + 32, valid),
                        VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed >>> 32) & 0xFFFFL).and(valid));
            }
            if (remainder > 48) {
                VectorMask<Float> valid = FloatVector.SPECIES_PREFERRED.indexInRange(0, remainder - 48);
                acc7 = acc7.lanewise(VectorOperators.ADD, FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase + 48, valid),
                        VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, (packed >>> 48) & 0xFFFFL).and(valid));
            }
        }

        // Fixed-order lane sum, as in the note above ROTATE_32.
        var laneSum = acc0.lanewise(VectorOperators.ADD, acc1).lanewise(VectorOperators.ADD, acc2.lanewise(VectorOperators.ADD, acc3)).lanewise(VectorOperators.ADD, acc4.lanewise(VectorOperators.ADD, acc5).lanewise(VectorOperators.ADD, acc6.lanewise(VectorOperators.ADD, acc7)));
        if (ROTATE_32 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_32));
        if (ROTATE_16 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_16));
        if (ROTATE_8 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_8));
        if (ROTATE_4 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_4));
        if (ROTATE_2 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_2));
        laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_1));
        return laneSum.lane(0);
    }

    // Fixed-order lane sums. Kernels that must give the same score compiled or interpreted add a vector's lanes by a
    // fixed tree of lane-wise adds instead of reduceLanes, whose order can change when the JIT compiles the caller:
    // each lane plus the lane half the width away, then a quarter, and so on, written out in each kernel (a helper
    // taking the vector can be left uninlined, which puts the vector on the heap). These rotate SPECIES_PREFERRED by
    // 32, 16, 8, 4, 2 and 1 lanes; a rotation as wide as the vector is null and its step folds away. Six steps cover
    // the widest vectors the Vector API allows (2048 bits, 64 lanes).
    private static final int ROTATE_LANES = FloatVector.SPECIES_PREFERRED.length();
    private static final VectorShuffle<Float>
            ROTATE_32 = 32 < ROTATE_LANES ? VectorShuffle.iota(FloatVector.SPECIES_PREFERRED, 32, 1, true) : null,
            ROTATE_16 = 16 < ROTATE_LANES ? VectorShuffle.iota(FloatVector.SPECIES_PREFERRED, 16, 1, true) : null,
            ROTATE_8 = 8 < ROTATE_LANES ? VectorShuffle.iota(FloatVector.SPECIES_PREFERRED, 8, 1, true) : null,
            ROTATE_4 = 4 < ROTATE_LANES ? VectorShuffle.iota(FloatVector.SPECIES_PREFERRED, 4, 1, true) : null,
            ROTATE_2 = 2 < ROTATE_LANES ? VectorShuffle.iota(FloatVector.SPECIES_PREFERRED, 2, 1, true) : null,
            ROTATE_1 = VectorShuffle.iota(FloatVector.SPECIES_PREFERRED, 1, 1, true);

    @Override
    public void ashMaskedAddBlockAllWords(float[] tildeQ,
                                          int d,
                                          long[] packedBits,
                                          int blockWordBase,
                                          int words,
                                          int blockSize,
                                          int laneStart,
                                          int blockLen,
                                          float[] outMaskedAdd) {

        // Like in the ASH paper, SIMD lanes are dimensions (SPECIES_PREFERRED),
        // masked load from tildeQ and horizontal sum per neighbor.
        //
        // Optimization: amortize horizontal reductions.
        // Instead of reducing each masked chunk immediately, we accumulate masked
        // vectors into a vector register accumulator and reduce once per lane.
        final int LANES = FloatVector.SPECIES_PREFERRED.length();

        for (int lane = 0; lane < blockLen; lane++) {
            // Vector accumulator: stays in registers (no array accumulation).
            FloatVector accVec = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

            // Scalar accumulator for unusual widths fallback only.
            float scalarAcc = 0f;

            int laneIndex = laneStart + lane;

            for (int w = 0; w < words; w++) {
                long wordBits = packedBits[blockWordBase + (w * blockSize) + laneIndex];
                int baseDim = w * 64;

                // process this 64-bit word in LANES-bit chunks
                for (int off = 0; off < 64 && (baseDim + off) < d; off += LANES) {
                    int qBase = baseDim + off;

                    // Range mask for tail dims beyond d
                    VectorMask<Float> inRange = FloatVector.SPECIES_PREFERRED.indexInRange(qBase, d);

                    long maskBits;
                    if (LANES == 16) {
                        maskBits = (wordBits >>> off) & 0xFFFFL;
                    } else if (LANES == 8) {
                        maskBits = (wordBits >>> off) & 0xFFL;
                    } else if (LANES == 4) {
                        maskBits = (wordBits >>> off) & 0xFL;
                    } else {
                        // Unusual width; fall back to scalar for this chunk
                        long m = wordBits >>> off;
                        float s = 0f;
                        for (int b = 0; b < LANES; b++) {
                            if (((m >>> b) & 1L) != 0L) {
                                int idx = qBase + b;
                                if (idx < d) s += tildeQ[idx];
                            }
                        }
                        scalarAcc += s;
                        continue;
                    }

                    if (maskBits == 0L) continue;

                    VectorMask<Float> bitMask =
                            VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, maskBits).and(inRange);

                    // masked load (paper)
                    FloatVector masked = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, tildeQ, qBase, bitMask);

                    // amortized accumulation (register)
                    accVec = accVec.lanewise(VectorOperators.ADD, masked);
                }
            }

            // One horizontal sum per lane (instead of per chunk)
            // Fixed-order lane sum, as in the note above ROTATE_32.
            var laneSum = accVec;
            if (ROTATE_32 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_32));
            if (ROTATE_16 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_16));
            if (ROTATE_8 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_8));
            if (ROTATE_4 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_4));
            if (ROTATE_2 != null) laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_2));
            laneSum = laneSum.lanewise(VectorOperators.ADD, laneSum.rearrange(ROTATE_1));
            float maskedAdd = laneSum.lane(0) + scalarAcc;

            outMaskedAdd[lane] = maskedAdd;
        }
    }
}
