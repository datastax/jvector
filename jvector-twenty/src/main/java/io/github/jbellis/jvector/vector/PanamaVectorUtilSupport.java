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
                            .rearrange(fields.and((short)15).toShuffle());
                    if (g + 1 < end) {
                        var high = jdk.incubator.vector.ShortVector.fromArray(shorts, lut, (g + 1) * 32)
                                .rearrange(fields.lanewise(VectorOperators.LSHR, 4).and((short)15).toShuffle());
                        low = low.add(high);
                    }
                    subtotal = subtotal.add(low);
                }
                var lo = (IntVector) subtotal.convertShape(VectorOperators.S2I, ints, 0);
                var hi = (IntVector) subtotal.convertShape(VectorOperators.S2I, ints, 1);
                sum0 = sum0.add((FloatVector)lo.convertShape(VectorOperators.I2F, floats, 0));
                sum1 = sum1.add((FloatVector)hi.convertShape(VectorOperators.I2F, floats, 0));
            }
            sum0.mul(.25f).intoArray(out, outOffset + i, floats.indexInRange(0, active));
            if (active > floats.length()) sum1.mul(.25f).intoArray(out, outOffset + i + floats.length(), floats.indexInRange(floats.length(), active));
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
                var negative = a.lanewise(VectorOperators.XOR, b).lanewise(VectorOperators.LSHR, 1).and(slots);
                var ma = a.and(slots);
                var mb = b.and(slots);
                var both = ma.and(mb);
                var delta = negative.lanewise(VectorOperators.BIT_COUNT).mul(-2)
                        .add(ma.lanewise(VectorOperators.BIT_COUNT).mul(2))
                        .add(mb.lanewise(VectorOperators.BIT_COUNT).mul(2))
                        .add(ma.and(negative).lanewise(VectorOperators.BIT_COUNT).mul(-4))
                        .add(mb.and(negative).lanewise(VectorOperators.BIT_COUNT).mul(-4))
                        .add(both.lanewise(VectorOperators.BIT_COUNT).mul(4))
                        .add(both.and(negative).lanewise(VectorOperators.BIT_COUNT).mul(-8));
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
                var am = ax.and((byte) 7).mul((byte) 2).add((byte) 1);
                var bm = bx.and((byte) 7).mul((byte) 2).add((byte) 1);
                am = am.blend(am.neg(), ax.and((byte) 8).compare(VectorOperators.EQ, (byte) 0));
                bm = bm.blend(bm.neg(), bx.and((byte) 8).compare(VectorOperators.EQ, (byte) 0));
                for (int part = 0; part < 2; part++) {
                    var av = (jdk.incubator.vector.ShortVector) am.convertShape(VectorOperators.B2S, shorts, part);
                    var bv = (jdk.incubator.vector.ShortVector) bm.convertShape(VectorOperators.B2S, shorts, part);
                    total = total.add(av.mul(bv));
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
                            .lanewise(VectorOperators.LSHR, shifts).and(m);
                    var y = ((IntVector) ((ByteVector) ByteVector.fromArray(ASH_OB_LOAD_WIDE, b, bOff + step * stepBytes)
                            .reinterpretShape(ASH_OB_BYTES, 0)).rearrange(windows).reinterpretAsInts())
                            .lanewise(VectorOperators.LSHR, shifts).and(m);
                    xy = xy.add(x.mul(y));
                    sums = sums.add(x.add(y));
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
        var shifts = IntVector.fromArray(ASH_UNIT_INTS, ASH_UNIT_SHIFTS[bits], 0);
        int mask = (1 << bits) - 1;
        // Exact: |digit * field| < 2^16, so 2^15 units fit an int lane, in any order. Keep one copy of the vector
        // operations: unrolled copies were sometimes compiled as heap-allocating calls.
        IntVector acc = IntVector.zero(ASH_UNIT_INTS);
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
            var fields = ByteVector.fromArray(ASH_UNIT_BYTES, source, from).rearrange(windows).reinterpretAsInts()
                    .lanewise(VectorOperators.LSHR, shifts).and(mask);
            acc = acc.add(fields.mul(IntVector.fromArray(ASH_UNIT_INTS, digits, unit * 8)));
        }
        return acc.reduceLanes(VectorOperators.ADD);
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
                    .lanewise(VectorOperators.LSHR, shifts).and(mask);
            var v0 = ((FloatVector) f0.convert(VectorOperators.I2F, 0)).sub(center);
            // Even and odd steps keep separate sums in a fixed order: the accumulators trade places every step.
            // One copy of the decode, as in ashOffsetBinaryInt8Dot.
            var next = FloatVector.fromArray(ASH_FLOATS, query, step * lanes).fma(v0, a0);
            a0 = a1;
            a1 = next;
        }
        // Fixed-order lane sum: the same score whether or not C2 has compiled this thread's copy yet.
        float sum = addLanesPairwise(a0.add(a1));
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
                    .lanewise(VectorOperators.LSHR, shifts).and(mask);
            var v0 = ((FloatVector) f0.convert(VectorOperators.I2F, 0)).sub(center);
            // Separate even and odd sums, as in ashOffsetBinaryDot1
            var next = FloatVector.fromArray(ASH_FLOATS, query, step * lanes).fma(v0, a0);
            a0 = a1;
            a1 = next;
        }
        // Fixed-order lane sum: the same score whether or not C2 has compiled this thread's copy yet.
        float sum = addLanesPairwise(a0.add(a1));
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
                    .lanewise(VectorOperators.LSHR, shifts0).and(mask);
            var f1 = ((IntVector) window.rearrange(windows1).reinterpretAsInts())
                    .lanewise(VectorOperators.LSHR, shifts1).and(mask);
            var v0 = ((FloatVector) f0.convert(VectorOperators.I2F, 0)).sub(center);
            var v1 = ((FloatVector) f1.convert(VectorOperators.I2F, 0)).sub(center);
            a0 = FloatVector.fromArray(ASH_FLOATS, query, step * 8).fma(v0, a0);
            a1 = FloatVector.fromArray(ASH_FLOATS, query, step * 8 + lanes).fma(v1, a1);
        }
        float sum = addLanesPairwise(a0.add(a1));
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
            var t0 = f0.lanewise(VectorOperators.LSHR, sign).and(1).sub(1);
            var m0 = f0.and(magnitude).lanewise(VectorOperators.LSHL, 1).add(1);
            var v0 = (FloatVector) m0.lanewise(VectorOperators.XOR, t0).sub(t0).convert(VectorOperators.I2F, 0);
            // Separate even and odd sums, as in ashOffsetBinaryDot1
            var next = FloatVector.fromArray(ASH_FLOATS, query, step * lanes).fma(v0, a0);
            a0 = a1;
            a1 = next;
        }
        // Fixed-order lane sum: the same score whether or not C2 has compiled this thread's copy yet.
        float sum = addLanesPairwise(a0.add(a1));
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
            var t0 = f0.lanewise(VectorOperators.LSHR, sign).and(1).sub(1);
            var t1 = f1.lanewise(VectorOperators.LSHR, sign).and(1).sub(1);
            var m0 = f0.and(magnitude).lanewise(VectorOperators.LSHL, 1).add(1);
            var m1 = f1.and(magnitude).lanewise(VectorOperators.LSHL, 1).add(1);
            var v0 = (FloatVector) m0.lanewise(VectorOperators.XOR, t0).sub(t0).convert(VectorOperators.I2F, 0);
            var v1 = (FloatVector) m1.lanewise(VectorOperators.XOR, t1).sub(t1).convert(VectorOperators.I2F, 0);
            a0 = FloatVector.fromArray(ASH_FLOATS, query, step * 8).fma(v0, a0);
            a1 = FloatVector.fromArray(ASH_FLOATS, query, step * 8 + lanes).fma(v1, a1);
        }
        float sum = addLanesPairwise(a0.add(a1));
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
            var m0 = ((FloatVector) f0.and(1).convertShape(VectorOperators.I2F, ASH_FLOATS, 0)).add(0.5f);
            var m2 = ((FloatVector) f2.and(1).convertShape(VectorOperators.I2F, ASH_FLOATS, 0)).add(0.5f);
            var m4 = ((FloatVector) f4.and(1).convertShape(VectorOperators.I2F, ASH_FLOATS, 0)).add(0.5f);
            var m6 = ((FloatVector) f6.and(1).convertShape(VectorOperators.I2F, ASH_FLOATS, 0)).add(0.5f);
            a0 = a0.add(FloatVector.fromArray(ASH_FLOATS, prepared, q)
                    .mul(m0.blend(m0.neg(), f0.and(2).compare(VectorOperators.EQ, 0).cast(ASH_FLOATS))));
            a1 = a1.add(FloatVector.fromArray(ASH_FLOATS, prepared, q + lanes)
                    .mul(m2.blend(m2.neg(), f2.and(2).compare(VectorOperators.EQ, 0).cast(ASH_FLOATS))));
            a2 = a2.add(FloatVector.fromArray(ASH_FLOATS, prepared, q + 2 * lanes)
                    .mul(m4.blend(m4.neg(), f4.and(2).compare(VectorOperators.EQ, 0).cast(ASH_FLOATS))));
            a3 = a3.add(FloatVector.fromArray(ASH_FLOATS, prepared, q + 3 * lanes)
                    .mul(m6.blend(m6.neg(), f6.and(2).compare(VectorOperators.EQ, 0).cast(ASH_FLOATS))));
        }
        // Fixed-order lane sum: the same score whether or not C2 has compiled this thread's copy yet.
        float sum = addLanesPairwise(a0.add(a1).add(a2.add(a3)));
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
            var m0 = ((FloatVector) f0.and(7).convertShape(VectorOperators.I2F, ASH_FLOATS, 0)).add(0.5f);
            var m4 = ((FloatVector) f4.and(7).convertShape(VectorOperators.I2F, ASH_FLOATS, 0)).add(0.5f);
            a0 = a0.add(FloatVector.fromArray(ASH_FLOATS, prepared, q)
                    .mul(m0.blend(m0.neg(), f0.and(8).compare(VectorOperators.EQ, 0).cast(ASH_FLOATS))));
            a1 = a1.add(FloatVector.fromArray(ASH_FLOATS, prepared, q + lanes)
                    .mul(m4.blend(m4.neg(), f4.and(8).compare(VectorOperators.EQ, 0).cast(ASH_FLOATS))));
        }
        // Fixed-order lane sum: the same score whether or not C2 has compiled this thread's copy yet.
        float sum = addLanesPairwise(a0.add(a1));
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

    @Override
    public float sum(VectorFloat<?> vector) {
        var sum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        // Process the remainder
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i);
            sum = sum.add(a);
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
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i);
            var divResult = a.mul(multiplier);
            intoVectorFloat(divResult, vector, i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < vector.length(); i++) {
            vector.set(i, vector.get(i) * multiplier);
        }
    }

    float dot64(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        var a = fromVectorFloat(FloatVector.SPECIES_64, v1, offset1);
        var b = fromVectorFloat(FloatVector.SPECIES_64, v2, offset2);
        return a.mul(b).reduceLanes(VectorOperators.ADD);
    }

    float dot128(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        var a = fromVectorFloat(FloatVector.SPECIES_128, v1, offset1);
        var b = fromVectorFloat(FloatVector.SPECIES_128, v2, offset2);
        return a.mul(b).reduceLanes(VectorOperators.ADD);
    }

    float dot256(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        var a = fromVectorFloat(FloatVector.SPECIES_256, v1, offset1);
        var b = fromVectorFloat(FloatVector.SPECIES_256, v2, offset2);
        return a.mul(b).reduceLanes(VectorOperators.ADD);
    }

    float dotPreferred(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v1, offset1);
        var b = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v2, offset2);
        return a.mul(b).reduceLanes(VectorOperators.ADD);
    }

    @Override
    public float dotProduct(VectorFloat<?> v1, VectorFloat<?> v2) {
        return dotProduct(v1, 0, v2, 0, v1.length());
    }

    @Override
    public float dotProduct(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, final int length) {
        //Common case first
        if (length >= 4 * FloatVector.SPECIES_PREFERRED.length())
            return dotProductPreferred(v1, v1offset, v2, v2offset, length);

        // One to three vectors, such as PQ subvectors. dotProductPreferred is too large for the JIT to inline, so
        // these short slices take a small path of their own instead of paying for a call each time.
        if (length >= FloatVector.SPECIES_PREFERRED.length())
            return dotProductShort(v1, v1offset, v2, v2offset, length);

        if (length < FloatVector.SPECIES_128.length())
            return dotProduct64(v1, v1offset, v2, v2offset, length);
        else if (length < FloatVector.SPECIES_256.length())
            return dotProduct128(v1, v1offset, v2, v2offset, length);
        else
            return dotProduct256(v1, v1offset, v2, v2offset, length);

    }

    float dotProduct64(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {

        if (length == FloatVector.SPECIES_64.length())
            return dot64(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_64.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_64);
        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_64.length()) {
            FloatVector a = fromVectorFloat(FloatVector.SPECIES_64, v1, v1offset + i);
            FloatVector b = fromVectorFloat(FloatVector.SPECIES_64, v2, v2offset + i);
            sum = a.fma(b, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i)
            res += v1.get(v1offset + i) * v2.get(v2offset + i);

        return res;
    }

    float dotProduct128(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {

        if (length == FloatVector.SPECIES_128.length())
            return dot128(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_128.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_128);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_128.length()) {
            FloatVector a = fromVectorFloat(FloatVector.SPECIES_128, v1, v1offset + i);
            FloatVector b = fromVectorFloat(FloatVector.SPECIES_128, v2, v2offset + i);
            sum = a.fma(b, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i)
            res += v1.get(v1offset + i) * v2.get(v2offset + i);

        return res;
    }

    float dotProduct256(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {

        if (length == FloatVector.SPECIES_256.length())
            return dot256(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_256.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_256);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_256.length()) {
            FloatVector a = fromVectorFloat(FloatVector.SPECIES_256, v1, v1offset + i);
            FloatVector b = fromVectorFloat(FloatVector.SPECIES_256, v2, v2offset + i);
            sum = a.fma(b, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i)
            res += v1.get(v1offset + i) * v2.get(v2offset + i);

        return res;
    }

    /**
     * One to three preferred vectors and a tail, small enough for the JIT to inline. The first two vectors use separate
     * sums, as in {@link #dotProductPreferred}, so their multiply-adds do not wait on each other.
     */
    float dotProductShort(VectorFloat<?> va, int vaoffset, VectorFloat<?> vb, int vboffset, int length) {
        int vectorLength = FloatVector.SPECIES_PREFERRED.length();
        assert length >= vectorLength && length < 4 * vectorLength : length;
        if (length == vectorLength)
            return dotPreferred(va, vaoffset, vb, vboffset);

        FloatVector sum0 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        int i = 0;
        if (length >= 2 * vectorLength) {
            FloatVector sum1 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, va, vaoffset + vectorLength)
                    .fma(fromVectorFloat(FloatVector.SPECIES_PREFERRED, vb, vboffset + vectorLength), sum0);
            sum0 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, va, vaoffset)
                    .fma(fromVectorFloat(FloatVector.SPECIES_PREFERRED, vb, vboffset), sum0);
            sum0 = sum0.add(sum1);
            i = 2 * vectorLength;
        }
        if (length - i >= vectorLength) {
            sum0 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, va, vaoffset + i)
                    .fma(fromVectorFloat(FloatVector.SPECIES_PREFERRED, vb, vboffset + i), sum0);
            i += vectorLength;
        }

        float resVec = sum0.reduceLanes(VectorOperators.ADD);
        float resTail = 0;
        for (; i < length; i++)
            resTail += va.get(vaoffset + i) * vb.get(vboffset + i);

        return resVec + resTail;
    }

    float dotProductPreferred(VectorFloat<?> va, int vaoffset, VectorFloat<?> vb, int vboffset, int length) {
        if (length == FloatVector.SPECIES_PREFERRED.length())
            return dotPreferred(va, vaoffset, vb, vboffset);

        FloatVector sum0 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        FloatVector sum1 = sum0;
        FloatVector a0, a1, b0, b1;

        int vectorLength = FloatVector.SPECIES_PREFERRED.length();

        // Unrolled vector loop; for dot product from L1 cache, an unroll factor of 2 generally suffices.
        // If we are going to be getting data that's further down the hierarchy but not fetched off disk/network,
        // we might want to unroll further, e.g. to 8 (4 sets of a,b,sum with 3-ahead reads seems to work best).
        if (length >= vectorLength * 2) {
            length -= vectorLength * 2;
            a0 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, va, vaoffset + vectorLength * 0);
            b0 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vb, vboffset + vectorLength * 0);
            a1 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, va, vaoffset + vectorLength * 1);
            b1 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vb, vboffset + vectorLength * 1);
            vaoffset += vectorLength * 2;
            vboffset += vectorLength * 2;
            while (length >= vectorLength * 2) {
                // All instructions in the main loop have no dependencies between them and can be executed in parallel.
                length -= vectorLength * 2;
                sum0 = a0.fma(b0, sum0);
                a0 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, va, vaoffset + vectorLength * 0);
                b0 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vb, vboffset + vectorLength * 0);
                sum1 = a1.fma(b1, sum1);
                a1 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, va, vaoffset + vectorLength * 1);
                b1 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vb, vboffset + vectorLength * 1);
                vaoffset += vectorLength * 2;
                vboffset += vectorLength * 2;
            }
            sum0 = a0.fma(b0, sum0);
            sum1 = a1.fma(b1, sum1);
        }
        sum0 = sum0.add(sum1);

        // Process the remaining few vectors
        while (length >= vectorLength) {
            length -= vectorLength;
            FloatVector a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, va, vaoffset);
            FloatVector b = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vb, vboffset);
            vaoffset += vectorLength;
            vboffset += vectorLength;
            sum0 = a.fma(b, sum0);
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
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v1, i);
            var b = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v2, i);
            vsum = a.fma(b, vsum);
            vaMagnitude = a.fma(a, vaMagnitude);
            vbMagnitude = b.fma(b, vbMagnitude);
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
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v1, v1offset + i);
            var b = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v2, v2offset + i);
            vsum = a.fma(b, vsum);
            vaMagnitude = a.fma(a, vaMagnitude);
            vbMagnitude = b.fma(b, vbMagnitude);
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

    float squareDistance64(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        var a = fromVectorFloat(FloatVector.SPECIES_64, v1, offset1);
        var b = fromVectorFloat(FloatVector.SPECIES_64, v2, offset2);
        var diff = a.sub(b);
        return diff.mul(diff).reduceLanes(VectorOperators.ADD);
    }

    float squareDistance128(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        var a = fromVectorFloat(FloatVector.SPECIES_128, v1, offset1);
        var b = fromVectorFloat(FloatVector.SPECIES_128, v2, offset2);
        var diff = a.sub(b);
        return diff.mul(diff).reduceLanes(VectorOperators.ADD);
    }

    float squareDistance256(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        var a = fromVectorFloat(FloatVector.SPECIES_256, v1, offset1);
        var b = fromVectorFloat(FloatVector.SPECIES_256, v2, offset2);
        var diff = a.sub(b);
        return diff.mul(diff).reduceLanes(VectorOperators.ADD);
    }

    float squareDistancePreferred(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v1, offset1);
        var b = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v2, offset2);
        var diff = a.sub(b);
        return diff.mul(diff).reduceLanes(VectorOperators.ADD);
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
        final VectorSpecies<Float> FS = FloatVector.SPECIES_PREFERRED;
        final float[] v = ((FloatArray) vector).array();
        final int dim = vector.length();
        final int bound = FS.loopBound(dim);
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
            FloatVector a0 = FloatVector.zero(FS), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
            int k = 0;
            for (; k < bound; k += FS.length()) {
                FloatVector vv = FloatVector.fromArray(FS, v, k);
                a0 = FloatVector.fromArray(FS, q0, k).fma(vv, a0);
                a1 = FloatVector.fromArray(FS, q1, k).fma(vv, a1);
                a2 = FloatVector.fromArray(FS, q2, k).fma(vv, a2);
                a3 = FloatVector.fromArray(FS, q3, k).fma(vv, a3);
                a4 = FloatVector.fromArray(FS, q4, k).fma(vv, a4);
                a5 = FloatVector.fromArray(FS, q5, k).fma(vv, a5);
                a6 = FloatVector.fromArray(FS, q6, k).fma(vv, a6);
                a7 = FloatVector.fromArray(FS, q7, k).fma(vv, a7);
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

    @Override
    public float squareDistance(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, final int length)
    {
        //Common case first
        if (length >= FloatVector.SPECIES_PREFERRED.length())
            return squareDistancePreferred(v1, v1offset, v2, v2offset, length);

        if (length < FloatVector.SPECIES_128.length())
            return squareDistance64(v1, v1offset, v2, v2offset, length);
        else if (length < FloatVector.SPECIES_256.length())
            return squareDistance128(v1, v1offset, v2, v2offset, length);
        else
            return squareDistance256(v1, v1offset, v2, v2offset, length);
    }

    float squareDistance64(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {
        if (length == FloatVector.SPECIES_64.length())
            return squareDistance64(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_64.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_64);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_64.length()) {
            FloatVector a = fromVectorFloat(FloatVector.SPECIES_64, v1, v1offset + i);
            FloatVector b = fromVectorFloat(FloatVector.SPECIES_64, v2, v2offset + i);
            var diff = a.sub(b);
            sum = diff.fma(diff, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i) {
            var diff = v1.get(v1offset + i) - v2.get(v2offset + i);
            res += diff * diff;
        }

        return res;
    }

    float squareDistance128(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {
        if (length == FloatVector.SPECIES_128.length())
            return squareDistance128(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_128.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_128);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_128.length()) {
            FloatVector a = fromVectorFloat(FloatVector.SPECIES_128, v1, v1offset + i);
            FloatVector b = fromVectorFloat(FloatVector.SPECIES_128, v2, v2offset + i);
            var diff = a.sub(b);
            sum = diff.fma(diff, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i) {
            var diff = v1.get(v1offset + i) - v2.get(v2offset + i);
            res += diff * diff;
        }

        return res;
    }

    float squareDistance256(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {
        if (length == FloatVector.SPECIES_256.length())
            return squareDistance256(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_256.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_256);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_256.length()) {
            FloatVector a = fromVectorFloat(FloatVector.SPECIES_256, v1, v1offset + i);
            FloatVector b = fromVectorFloat(FloatVector.SPECIES_256, v2, v2offset + i);
            var diff = a.sub(b);
            sum = diff.fma(diff, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i) {
            var diff = v1.get(v1offset + i) - v2.get(v2offset + i);
            res += diff * diff;
        }

        return res;
    }

    float squareDistancePreferred(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {

        if (length == FloatVector.SPECIES_PREFERRED.length())
            return squareDistancePreferred(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v1, v1offset + i);
            FloatVector b = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v2, v2offset + i);
            var diff = a.sub(b);
            sum = diff.fma(diff, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i) {
            var diff = v1.get(v1offset + i) - v2.get(v2offset + i);
            res += diff * diff;
        }

        return res;
    }

    void addInPlace64(VectorFloat<?> v1, VectorFloat<?> v2) {
        var a = fromVectorFloat(FloatVector.SPECIES_64, v1, 0);
        var b = fromVectorFloat(FloatVector.SPECIES_64, v2, 0);
        intoVectorFloat(a.add(b), v1, 0);
    }

    void addInPlace64(VectorFloat<?> v1, float value) {
        var a = fromVectorFloat(FloatVector.SPECIES_64, v1, 0);
        intoVectorFloat(a.add(value), v1, 0);
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
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v1, i);
            var b = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v2, i);
            intoVectorFloat(a.add(b), v1, i);
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
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v1, i);
            intoVectorFloat(a.add(value), v1, i);
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
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v1, i);
            var b = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v2, i);
            intoVectorFloat(a.sub(b), v1, i);
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
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i);
            intoVectorFloat(a.sub(value), vector, i);
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
            var lhs = fromVectorFloat(FloatVector.SPECIES_PREFERRED, a, aOffset + i);
            var rhs = fromVectorFloat(FloatVector.SPECIES_PREFERRED, b, bOffset + i);
            var subResult = lhs.sub(rhs);
            intoVectorFloat(subResult, res, i);
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
            var lhs = fromVectorFloat(FloatVector.SPECIES_PREFERRED, a, aOffset + i);
            var subResult = lhs.sub(value);
            intoVectorFloat(subResult, res, i);
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
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v1, i);
            var b = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v2, i);
            intoVectorFloat(a.min(b), v1, i);
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

    @Override
    public float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
        return switch (PREFERRED_BIT_SIZE) {
            case 512 -> assembleAndSum512(data, dataBase, baseOffsets, baseOffsetsOffset, baseOffsetsLength);
            case 256 -> assembleAndSum256(data, dataBase, baseOffsets, baseOffsetsOffset, baseOffsetsLength);
            case 128 -> assembleAndSum128(data, dataBase, baseOffsets, baseOffsetsOffset, baseOffsetsLength);
            default -> throw new IllegalStateException("Unsupported vector width: " + PREFERRED_BIT_SIZE);
        };
    }

    float assembleAndSum512(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
        int[] convOffsets = scratchInt512.get();
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_512);
        int i = 0;
        int limit = ByteVector.SPECIES_128.loopBound(baseOffsetsLength);
        var scale = IntVector.zero(IntVector.SPECIES_512).addIndex(dataBase);

        for (; i < limit; i += ByteVector.SPECIES_128.length()) {
            fromByteSequence(ByteVector.SPECIES_128, baseOffsets, i + baseOffsets.offset() + baseOffsetsOffset)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_512)
                    .reinterpretAsInts()
                    .add(scale)
                    .intoArray(convOffsets, 0);

            var offset = i * dataBase;
            sum = sum.add(fromVectorFloat(FloatVector.SPECIES_512, data, offset, convOffsets, 0));
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        //Process tail
        for (; i < baseOffsetsLength; i++)
            res += data.get(dataBase * i + Byte.toUnsignedInt(baseOffsets.get(i + baseOffsetsOffset)));

        return res;
    }

    float assembleAndSum256(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
        int[] convOffsets = scratchInt256.get();
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_256);
        int i = 0;
        int limit = ByteVector.SPECIES_64.loopBound(baseOffsetsLength);
        var scale = IntVector.zero(IntVector.SPECIES_256).addIndex(dataBase);

        for (; i < limit; i += ByteVector.SPECIES_64.length()) {

            fromByteSequence(ByteVector.SPECIES_64, baseOffsets, i + baseOffsets.offset() + baseOffsetsOffset)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_256)
                    .reinterpretAsInts()
                    .add(scale)
                    .intoArray(convOffsets, 0);

            var offset = i * dataBase;
            sum = sum.add(fromVectorFloat(FloatVector.SPECIES_256, data, offset, convOffsets, 0));
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process tail
        for (; i < baseOffsetsLength; i++)
            res += data.get(dataBase * i + Byte.toUnsignedInt(baseOffsets.get(i + baseOffsetsOffset)));

        return res;
    }

    float assembleAndSum128(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
        // benchmarking a 128-bit SIMD implementation showed it performed worse than scalar
        float sum = 0f;
        for (int i = 0; i < baseOffsetsLength; i++) {
            sum += data.get(dataBase * i + Byte.toUnsignedInt(baseOffsets.get(i + baseOffsetsOffset)));
        }
        return sum;
    }

    private static final VectorSpecies<Float> PAIR_FLOATS = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> PAIR_INTS = VectorSpecies.of(int.class, PAIR_FLOATS.vectorShape());
    private static final int PAIR_LANES = PAIR_FLOATS.length();
    // One code byte per float lane, for 1-float subspaces. Byte vectors start at 64 bits, so narrower
    // registers leave this null and use the plain loop.
    private static final VectorSpecies<Byte> PAIR_BYTES = PAIR_LANES >= 8
            ? VectorSpecies.of(byte.class, VectorShape.forBitSize(PAIR_LANES * 8)) : null;
    private static final ThreadLocal<int[]> PAIR_INDICES = ThreadLocal.withInitial(() -> new int[2 * PAIR_LANES]);


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

    private static float pairContiguous512(boolean dot, float[] cb, int clusterCount, int[] subspaceOffsets, int[] subspaceWidths,
                                          ByteSequence<?> code1, int offset1, ByteSequence<?> code2, int offset2) {
        final VectorSpecies<Float> species = FloatVector.SPECIES_512;
        final int lanes = species.length();
        FloatVector acc = FloatVector.zero(species);
        float tail = 0;
        for (int m = 0; m < subspaceOffsets.length; m++) {
            int width = subspaceWidths[m], base = clusterCount * subspaceOffsets[m];
            int b1 = base + Byte.toUnsignedInt(code1.get(offset1 + m)) * width;
            int b2 = base + Byte.toUnsignedInt(code2.get(offset2 + m)) * width;
            int t = 0;
            for (; t + lanes <= width; t += lanes) {
                var x = FloatVector.fromArray(species, cb, b1 + t);
                var y = FloatVector.fromArray(species, cb, b2 + t);
                acc = dot ? x.fma(y, acc) : x.sub(y).fma(x.sub(y), acc);
            }
            for (; t < width; t++) {
                float x = cb[b1 + t], y = cb[b2 + t];
                tail += dot ? x * y : (x - y) * (x - y);
            }
        }
        return acc.reduceLanes(VectorOperators.ADD) + tail;
    }

    private static float pairContiguous256(boolean dot, float[] cb, int clusterCount, int[] subspaceOffsets, int[] subspaceWidths,
                                          ByteSequence<?> code1, int offset1, ByteSequence<?> code2, int offset2) {
        final VectorSpecies<Float> species = FloatVector.SPECIES_256;
        final int lanes = species.length();
        FloatVector acc = FloatVector.zero(species);
        float tail = 0;
        for (int m = 0; m < subspaceOffsets.length; m++) {
            int width = subspaceWidths[m], base = clusterCount * subspaceOffsets[m];
            int b1 = base + Byte.toUnsignedInt(code1.get(offset1 + m)) * width;
            int b2 = base + Byte.toUnsignedInt(code2.get(offset2 + m)) * width;
            int t = 0;
            for (; t + lanes <= width; t += lanes) {
                var x = FloatVector.fromArray(species, cb, b1 + t);
                var y = FloatVector.fromArray(species, cb, b2 + t);
                acc = dot ? x.fma(y, acc) : x.sub(y).fma(x.sub(y), acc);
            }
            for (; t < width; t++) {
                float x = cb[b1 + t], y = cb[b2 + t];
                tail += dot ? x * y : (x - y) * (x - y);
            }
        }
        return acc.reduceLanes(VectorOperators.ADD) + tail;
    }

    private static float pairContiguous128(boolean dot, float[] cb, int clusterCount, int[] subspaceOffsets, int[] subspaceWidths,
                                          ByteSequence<?> code1, int offset1, ByteSequence<?> code2, int offset2) {
        final VectorSpecies<Float> species = FloatVector.SPECIES_128;
        final int lanes = species.length();
        FloatVector acc = FloatVector.zero(species);
        float tail = 0;
        for (int m = 0; m < subspaceOffsets.length; m++) {
            int width = subspaceWidths[m], base = clusterCount * subspaceOffsets[m];
            int b1 = base + Byte.toUnsignedInt(code1.get(offset1 + m)) * width;
            int b2 = base + Byte.toUnsignedInt(code2.get(offset2 + m)) * width;
            int t = 0;
            for (; t + lanes <= width; t += lanes) {
                var x = FloatVector.fromArray(species, cb, b1 + t);
                var y = FloatVector.fromArray(species, cb, b2 + t);
                acc = dot ? x.fma(y, acc) : x.sub(y).fma(x.sub(y), acc);
            }
            for (; t < width; t++) {
                float x = cb[b1 + t], y = cb[b2 + t];
                tail += dot ? x * y : (x - y) * (x - y);
            }
        }
        return acc.reduceLanes(VectorOperators.ADD) + tail;
    }

    private static float pairContiguous64(boolean dot, float[] cb, int clusterCount, int[] subspaceOffsets, int[] subspaceWidths,
                                          ByteSequence<?> code1, int offset1, ByteSequence<?> code2, int offset2) {
        final VectorSpecies<Float> species = FloatVector.SPECIES_64;
        final int lanes = species.length();
        FloatVector acc = FloatVector.zero(species);
        float tail = 0;
        for (int m = 0; m < subspaceOffsets.length; m++) {
            int width = subspaceWidths[m], base = clusterCount * subspaceOffsets[m];
            int b1 = base + Byte.toUnsignedInt(code1.get(offset1 + m)) * width;
            int b2 = base + Byte.toUnsignedInt(code2.get(offset2 + m)) * width;
            int t = 0;
            for (; t + lanes <= width; t += lanes) {
                var x = FloatVector.fromArray(species, cb, b1 + t);
                var y = FloatVector.fromArray(species, cb, b2 + t);
                acc = dot ? x.fma(y, acc) : x.sub(y).fma(x.sub(y), acc);
            }
            for (; t < width; t++) {
                float x = cb[b1 + t], y = cb[b2 + t];
                tail += dot ? x * y : (x - y) * (x - y);
            }
        }
        return acc.reduceLanes(VectorOperators.ADD) + tail;
    }

    @Override
    public float pqCodePairSimilarity(boolean dot, VectorFloat<?> codebooks, int clusterCount,
                                      int[] subspaceOffsets, int[] subspaceWidths, int minWidth, int maxWidth, int[] dimBase,
                                      ByteSequence<?> code1, int offset1, ByteSequence<?> code2, int offset2) {
        // A build calls this tens of millions of times, so it only dispatches: each path is its own small
        // method, keeping the whole call chain small enough to inline.
        float[] cb = ((FloatArray) codebooks).array();
        if (minWidth >= 4) {
            // Every centroid fills at least a 128-bit register: load both contiguously. Each width gets
            // its own method so the species is a constant the JIT can compile to SIMD instructions.
            if (minWidth >= 16 && PREFERRED_BIT_SIZE >= 512) {
                return pairContiguous512(dot, cb, clusterCount, subspaceOffsets, subspaceWidths, code1, offset1, code2, offset2);
            }
            if (minWidth >= 8 && PREFERRED_BIT_SIZE >= 256) {
                return pairContiguous256(dot, cb, clusterCount, subspaceOffsets, subspaceWidths, code1, offset1, code2, offset2);
            }
            return pairContiguous128(dot, cb, clusterCount, subspaceOffsets, subspaceWidths, code1, offset1, code2, offset2);
        }
        if (minWidth >= 2) {
            // 2- and 3-float subspaces: one 64-bit load per centroid.
            return pairContiguous64(dot, cb, clusterCount, subspaceOffsets, subspaceWidths, code1, offset1, code2, offset2);
        }
        if (maxWidth == 1 && dimBase.length >= PAIR_LANES && PAIR_BYTES != null) {
            return pairGatherOneFloat(dot, cb, dimBase, code1, offset1, code2, offset2);
        }
        // Mixed 1-float subspaces, or too few to fill a register: a plain loop.
        return pairScalar(dot, cb, clusterCount, subspaceOffsets, subspaceWidths, code1, offset1, code2, offset2);
    }

    /** Full registers of 1-float subspaces: each lane's code is one byte of a vector load. */
    private float pairGatherOneFloat(boolean dot, float[] cb, int[] dimBase,
                                     ByteSequence<?> code1, int offset1, ByteSequence<?> code2, int offset2) {
        int dim = dimBase.length;
        int[] indices = PAIR_INDICES.get();
        int base1 = code1.offset() + offset1, base2 = code2.offset() + offset2;
        FloatVector acc = FloatVector.zero(PAIR_FLOATS);
        int j = 0;
        for (; j + PAIR_LANES <= dim; j += PAIR_LANES) {
            var lanesBase = IntVector.fromArray(PAIR_INTS, dimBase, j);
            ((IntVector) fromByteSequence(PAIR_BYTES, code1, base1 + j).convertShape(VectorOperators.B2I, PAIR_INTS, 0))
                    .and(0xff).add(lanesBase).intoArray(indices, 0);
            ((IntVector) fromByteSequence(PAIR_BYTES, code2, base2 + j).convertShape(VectorOperators.B2I, PAIR_INTS, 0))
                    .and(0xff).add(lanesBase).intoArray(indices, PAIR_LANES);
            var x = FloatVector.fromArray(PAIR_FLOATS, cb, 0, indices, 0);
            var y = FloatVector.fromArray(PAIR_FLOATS, cb, 0, indices, PAIR_LANES);
            acc = dot ? x.fma(y, acc) : x.sub(y).fma(x.sub(y), acc);
        }
        float tail = 0;
        for (; j < dim; j++) {
            float x = cb[dimBase[j] + Byte.toUnsignedInt(code1.get(offset1 + j))];
            float y = cb[dimBase[j] + Byte.toUnsignedInt(code2.get(offset2 + j))];
            tail += dot ? x * y : (x - y) * (x - y);
        }
        return acc.reduceLanes(VectorOperators.ADD) + tail;
    }

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
        //compute the size of the subvector
        return switch (PREFERRED_BIT_SIZE) {
            case 512 ->
                    assembleAndSumPQ_512(codebookPartialSums, subspaceCount, vector1Ordinals, vector1OrdinalOffset, vector2Ordinals, vector2OrdinalOffset, clusterCount);
            case 256 ->
                    assembleAndSumPQ_256(codebookPartialSums, subspaceCount, vector1Ordinals, vector1OrdinalOffset, vector2Ordinals, vector2OrdinalOffset, clusterCount);
            case 128 ->
                    assembleAndSumPQ_128(codebookPartialSums, subspaceCount, vector1Ordinals, vector1OrdinalOffset, vector2Ordinals, vector2OrdinalOffset, clusterCount);
            default -> throw new IllegalStateException("Unsupported vector width: " + PREFERRED_BIT_SIZE);
        };
    }

    float assembleAndSumPQ_128(
            VectorFloat<?> data,
            int subspaceCount,                  // = M
            ByteSequence<?> baseOffsets1,
            int baseOffsetsOffset1,
            ByteSequence<?> baseOffsets2,
            int baseOffsetsOffset2,
            int clusterCount                    // = k
    ) {
        final int k = clusterCount;
        final int blockSize = k * (k + 1) / 2;
        float res = 0f;

        for (int i = 0; i < subspaceCount; i++) {
            int c1 = Byte.toUnsignedInt(baseOffsets1.get(i + baseOffsetsOffset1));
            int c2 = Byte.toUnsignedInt(baseOffsets2.get(i + baseOffsetsOffset2));
            int r = Math.min(c1, c2);
            int c = Math.max(c1, c2);

            int offsetRow = r * k - (r * (r - 1) / 2);
            int idxInBlock = offsetRow + (c - r);
            int base = i * blockSize;

            res += data.get(base + idxInBlock);
        }

        return res;
    }

    float assembleAndSumPQ_256(
            VectorFloat<?> data,
            int subspaceCount,                  // = M
            ByteSequence<?> baseOffsets1,
            int baseOffsetsOffset1,
            ByteSequence<?> baseOffsets2,
            int baseOffsetsOffset2,
            int clusterCount                    // = k
    ) {
        final VectorSpecies<Float> FSPECIES = FloatVector.SPECIES_256;
        final int LANES = FSPECIES.length();
        final int k = clusterCount;
        final int blockSize = k * (k + 1) / 2;
        final int M = subspaceCount;

        int[] convOffsets = scratchInt256.get();
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_256);
        FloatVector scale = FloatVector.zero(FloatVector.SPECIES_256).addIndex(blockSize);
        FloatVector kvec = FloatVector.broadcast(FloatVector.SPECIES_256, k);
        FloatVector onevec = FloatVector.broadcast(FloatVector.SPECIES_256, 1);
        FloatVector twovec = FloatVector.broadcast(FloatVector.SPECIES_256, 0.5f);

        for (int i = 0; i + LANES <= M; i += LANES) {

            FloatVector c1v = fromByteSequence(ByteVector.SPECIES_64, baseOffsets1, i + baseOffsets1.offset() + baseOffsetsOffset1)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_256)
                    .convertShape(VectorOperators.I2F, FSPECIES, 0)
                    .reinterpretAsFloats();

            FloatVector c2v = fromByteSequence(ByteVector.SPECIES_64, baseOffsets2, i + baseOffsets2.offset() + baseOffsetsOffset2)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_256)
                    .convertShape(VectorOperators.I2F, FSPECIES, 0)
                    .reinterpretAsFloats();

            // b) r = min(c1,c2), c = max(c1,c2)
            var r = c1v.min(c2v);
            var c = c1v.max(c2v);

            // c) offsetRow = r*k - (r*(r-1))/2
            var rk = r.mul(kvec);
            var triangular = r.mul(r.sub(onevec)).mul(twovec);
            var offsetRow = rk.sub(triangular);

            // d) idxInBlock = offsetRow + (c - r) + (i * blockSize)
            offsetRow.add(c.sub(r)).add(scale)
                    .convertShape(VectorOperators.F2I, IntVector.SPECIES_256, 0)
                    .reinterpretAsInts()
                    .intoArray(convOffsets, 0);

            // e) gather LANES floats from `partials` at those indices
            FloatVector chunk = fromVectorFloat(FSPECIES, data, i * blockSize, convOffsets, 0);

            // f) horizontal sum the chunk and add to our scalar accumulator
            sum = sum.add(chunk);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        //
        // 3) Remainder: fall back to your scalar code for i % LANES != 0
        //
        for (int i = (M / LANES) * LANES; i < M; i++) {
            int c1 = Byte.toUnsignedInt(baseOffsets1.get(i + baseOffsetsOffset1));
            int c2 = Byte.toUnsignedInt(baseOffsets2.get(i + baseOffsetsOffset2));
            int r = Math.min(c1, c2);
            int c = Math.max(c1, c2);

            int offsetRow = r * k - (r * (r - 1) / 2);
            int idxInBlock = offsetRow + (c - r);
            int base = i * blockSize;

            res += data.get(base + idxInBlock);
        }

        return res;
    }

    float assembleAndSumPQ_512(
            VectorFloat<?> data,
            int subspaceCount,                  // = M
            ByteSequence<?> baseOffsets1,
            int baseOffsetsOffset1,
            ByteSequence<?> baseOffsets2,
            int baseOffsetsOffset2,
            int clusterCount                    // = k
    ) {
        final VectorSpecies<Float> FSPECIES = FloatVector.SPECIES_512;
        final int LANES = FSPECIES.length();
        final int k = clusterCount;
        final int blockSize = k * (k + 1) / 2;
        final int M = subspaceCount;

        int[] convOffsets = scratchInt512.get();
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_512);
        FloatVector scale = FloatVector.zero(FloatVector.SPECIES_512).addIndex(blockSize);
        FloatVector kvec = FloatVector.broadcast(FloatVector.SPECIES_512, k);
        FloatVector onevec = FloatVector.broadcast(FloatVector.SPECIES_512, 1);
        FloatVector twovec = FloatVector.broadcast(FloatVector.SPECIES_512, 0.5f);

        for (int i = 0; i + LANES <= M; i += LANES) {
            FloatVector c1v = fromByteSequence(ByteVector.SPECIES_128, baseOffsets1, i + baseOffsets1.offset() + baseOffsetsOffset1)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_512)
                    .convertShape(VectorOperators.I2F, FSPECIES, 0)
                    .reinterpretAsFloats();

            FloatVector c2v = fromByteSequence(ByteVector.SPECIES_128, baseOffsets2, i + baseOffsets2.offset() + baseOffsetsOffset2)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_512)
                    .convertShape(VectorOperators.I2F, FSPECIES, 0)
                    .reinterpretAsFloats();

            // b) r = min(c1,c2), c = max(c1,c2)
            var r = c1v.min(c2v);
            var c = c1v.max(c2v);

            // c) offsetRow = r*k - (r*(r-1))/2
            var rk = r.mul(kvec);
            var triangular = r.mul(r.sub(onevec)).mul(twovec);
            var offsetRow = rk.sub(triangular);

            // d) idxInBlock = offsetRow + (c - r) + (i * blockSize)
            offsetRow.add(c.sub(r)).add(scale)
                    .convertShape(VectorOperators.F2I, IntVector.SPECIES_512, 0)
                    .reinterpretAsInts()
                    .intoArray(convOffsets, 0);

            // e) gather LANES floats from `partials` at those indices
            FloatVector chunk = fromVectorFloat(FSPECIES, data, i * blockSize, convOffsets, 0);

            // f) horizontal sum the chunk and add to our scalar accumulator
            sum = sum.add(chunk);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        //
        // 3) Remainder: fall back to your scalar code for i % LANES != 0
        //
        for (int i = (M / LANES) * LANES; i < M; i++) {
            int c1 = Byte.toUnsignedInt(baseOffsets1.get(i + baseOffsetsOffset1));
            int c2 = Byte.toUnsignedInt(baseOffsets2.get(i + baseOffsetsOffset2));
            int r = Math.min(c1, c2);
            int c = Math.max(c1, c2);

            int offsetRow = r * k - (r * (r - 1) / 2);
            int idxInBlock = offsetRow + (c - r);
            int base = i * blockSize;

            res += data.get(base + idxInBlock);
        }

        return res;
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
        return switch (PREFERRED_BIT_SIZE) {
            case 512 -> dotProductBytes512(a, b);
            case 256 -> dotProductBytes256(a, b);
            default  -> dotProductBytes128(a, b);
        };
    }

    float dotProductBytes512(ByteSequence<?> a, ByteSequence<?> b) {
        final int length = a.length();
        final int step   = ByteVector.SPECIES_128.length(); // 16
        final int limit  = ByteVector.SPECIES_128.loopBound(length);
        IntVector acc    = IntVector.zero(IntVector.SPECIES_512);

        for (int i = 0; i < limit; i += step) {
            IntVector va = fromByteSequence(ByteVector.SPECIES_128, a, a.offset() + i)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                    .reinterpretAsInts();
            IntVector vb = fromByteSequence(ByteVector.SPECIES_128, b, b.offset() + i)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                    .reinterpretAsInts();
            acc = acc.add(va.mul(vb));
        }

        int result = acc.reduceLanes(VectorOperators.ADD);
        for (int i = limit; i < length; i++) {
            result += a.get(i) * b.get(i);
        }
        return result;
    }

    float dotProductBytes256(ByteSequence<?> a, ByteSequence<?> b) {
        final int length = a.length();
        final int step   = ByteVector.SPECIES_64.length(); // 8
        final int limit  = ByteVector.SPECIES_64.loopBound(length);
        IntVector acc    = IntVector.zero(IntVector.SPECIES_256);

        for (int i = 0; i < limit; i += step) {
            IntVector va = fromByteSequence(ByteVector.SPECIES_64, a, a.offset() + i)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                    .reinterpretAsInts();
            IntVector vb = fromByteSequence(ByteVector.SPECIES_64, b, b.offset() + i)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                    .reinterpretAsInts();
            acc = acc.add(va.mul(vb));
        }

        int result = acc.reduceLanes(VectorOperators.ADD);
        for (int i = limit; i < length; i++) {
            result += a.get(i) * b.get(i);
        }
        return result;
    }

    float dotProductBytes128(ByteSequence<?> a, ByteSequence<?> b) {
        // ByteVector.SPECIES_32 does not exist; scalar is fastest at 128-bit width
        final int length = a.length();
        int result = 0;
        for (int i = 0; i < length; i++) {
            result += a.get(i) * b.get(i);
        }
        return result;
    }

    /**
     * Vectorized sum of squared differences between two signed int8 byte vectors.
     */
    @Override
    public float squareDistance(ByteSequence<?> a, ByteSequence<?> b) {
        return switch (PREFERRED_BIT_SIZE) {
            case 512 -> squareDistanceBytes512(a, b);
            case 256 -> squareDistanceBytes256(a, b);
            default  -> squareDistanceBytes128(a, b);
        };
    }

    float squareDistanceBytes512(ByteSequence<?> a, ByteSequence<?> b) {
        final int length = a.length();
        final int step   = ByteVector.SPECIES_128.length();
        final int limit  = ByteVector.SPECIES_128.loopBound(length);
        IntVector acc    = IntVector.zero(IntVector.SPECIES_512);

        for (int i = 0; i < limit; i += step) {
            IntVector va   = fromByteSequence(ByteVector.SPECIES_128, a, a.offset() + i)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                    .reinterpretAsInts();
            IntVector vb   = fromByteSequence(ByteVector.SPECIES_128, b, b.offset() + i)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                    .reinterpretAsInts();
            IntVector diff = va.sub(vb);
            acc = acc.add(diff.mul(diff));
        }

        int result = acc.reduceLanes(VectorOperators.ADD);
        for (int i = limit; i < length; i++) {
            int diff = a.get(i) - b.get(i);
            result += diff * diff;
        }
        return result;
    }

    float squareDistanceBytes256(ByteSequence<?> a, ByteSequence<?> b) {
        final int length = a.length();
        final int step   = ByteVector.SPECIES_64.length();
        final int limit  = ByteVector.SPECIES_64.loopBound(length);
        IntVector acc    = IntVector.zero(IntVector.SPECIES_256);

        for (int i = 0; i < limit; i += step) {
            IntVector va   = fromByteSequence(ByteVector.SPECIES_64, a, a.offset() + i)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                    .reinterpretAsInts();
            IntVector vb   = fromByteSequence(ByteVector.SPECIES_64, b, b.offset() + i)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                    .reinterpretAsInts();
            IntVector diff = va.sub(vb);
            acc = acc.add(diff.mul(diff));
        }

        int result = acc.reduceLanes(VectorOperators.ADD);
        for (int i = limit; i < length; i++) {
            int diff = a.get(i) - b.get(i);
            result += diff * diff;
        }
        return result;
    }

    float squareDistanceBytes128(ByteSequence<?> a, ByteSequence<?> b) {
        // ByteVector.SPECIES_32 does not exist; scalar is fastest at 128-bit width
        final int length = a.length();
        int result = 0;
        for (int i = 0; i < length; i++) {
            int diff = a.get(i) - b.get(i);
            result += diff * diff;
        }
        return result;
    }

    /**
     * Vectorized cosine similarity between two signed int8 byte vectors.
     */
    @Override
    public float cosine(ByteSequence<?> a, ByteSequence<?> b) {
        return switch (PREFERRED_BIT_SIZE) {
            case 512 -> cosineBytes512(a, b);
            case 256 -> cosineBytes256(a, b);
            default  -> cosineBytes128(a, b);
        };
    }

    float cosineBytes512(ByteSequence<?> a, ByteSequence<?> b) {
        final int length = a.length();
        final int step   = ByteVector.SPECIES_128.length();
        final int limit  = ByteVector.SPECIES_128.loopBound(length);
        IntVector dot    = IntVector.zero(IntVector.SPECIES_512);
        IntVector normA  = IntVector.zero(IntVector.SPECIES_512);
        IntVector normB  = IntVector.zero(IntVector.SPECIES_512);

        for (int i = 0; i < limit; i += step) {
            IntVector va = fromByteSequence(ByteVector.SPECIES_128, a, a.offset() + i)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                    .reinterpretAsInts();
            IntVector vb = fromByteSequence(ByteVector.SPECIES_128, b, b.offset() + i)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                    .reinterpretAsInts();
            dot   = dot.add(va.mul(vb));
            normA = normA.add(va.mul(va));
            normB = normB.add(vb.mul(vb));
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
    }

    float cosineBytes256(ByteSequence<?> a, ByteSequence<?> b) {
        final int length = a.length();
        final int step   = ByteVector.SPECIES_64.length();
        final int limit  = ByteVector.SPECIES_64.loopBound(length);
        IntVector dot    = IntVector.zero(IntVector.SPECIES_256);
        IntVector normA  = IntVector.zero(IntVector.SPECIES_256);
        IntVector normB  = IntVector.zero(IntVector.SPECIES_256);

        for (int i = 0; i < limit; i += step) {
            IntVector va = fromByteSequence(ByteVector.SPECIES_64, a, a.offset() + i)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                    .reinterpretAsInts();
            IntVector vb = fromByteSequence(ByteVector.SPECIES_64, b, b.offset() + i)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                    .reinterpretAsInts();
            dot   = dot.add(va.mul(vb));
            normA = normA.add(va.mul(va));
            normB = normB.add(vb.mul(vb));
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
    }

    float cosineBytes128(ByteSequence<?> a, ByteSequence<?> b) {
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

    @Override
    public int hammingDistance(long[] a, long[] b) {
        var sum = LongVector.zero(LongVector.SPECIES_PREFERRED);
        int vectorizedLength = LongVector.SPECIES_PREFERRED.loopBound(a.length);

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += LongVector.SPECIES_PREFERRED.length()) {
            var va = LongVector.fromArray(LongVector.SPECIES_PREFERRED, a, i);
            var vb = LongVector.fromArray(LongVector.SPECIES_PREFERRED, b, i);

            var xorResult = va.lanewise(VectorOperators.XOR, vb);
            sum = sum.add(xorResult.lanewise(VectorOperators.BIT_COUNT));
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
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v, i);
            accum = accum.max(a);
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
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v, i);
            accum = accum.min(a);
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

    @Override
    public float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int encodedOffset, int encodedLength, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
        return switch (PREFERRED_BIT_SIZE) {
            case 512 ->
                    pqDecodedCosineSimilarity512(encoded, encodedOffset, encodedLength, clusterCount, partialSums, aMagnitude, bMagnitude);
            case 256 ->
                    pqDecodedCosineSimilarity256(encoded, encodedOffset, encodedLength, clusterCount, partialSums, aMagnitude, bMagnitude);
            case 128 ->
                    pqDecodedCosineSimilarity128(encoded, encodedOffset, encodedLength, clusterCount, partialSums, aMagnitude, bMagnitude);
            default -> throw new IllegalStateException("Unsupported vector width: " + PREFERRED_BIT_SIZE);
        };
    }

    float pqDecodedCosineSimilarity512(ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
        var sum = FloatVector.zero(FloatVector.SPECIES_512);
        var vaMagnitude = FloatVector.zero(FloatVector.SPECIES_512);

        int[] convOffsets = scratchInt512.get();
        int i = 0;
        int limit = i + ByteVector.SPECIES_128.loopBound(baseOffsetsLength);

        var scale = IntVector.zero(IntVector.SPECIES_512).addIndex(clusterCount);

        for (; i < limit; i += ByteVector.SPECIES_128.length()) {

            fromByteSequence(ByteVector.SPECIES_128, baseOffsets, i + baseOffsets.offset() + baseOffsetsOffset)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_512)
                    .reinterpretAsInts()
                    .add(scale)
                    .intoArray(convOffsets, 0);

            var offset = i * clusterCount;
            sum = sum.add(fromVectorFloat(FloatVector.SPECIES_512, partialSums, offset, convOffsets, 0));
            vaMagnitude = vaMagnitude.add(fromVectorFloat(FloatVector.SPECIES_512, aMagnitude, offset, convOffsets, 0));
        }

        float sumResult = sum.reduceLanes(VectorOperators.ADD);
        float aMagnitudeResult = vaMagnitude.reduceLanes(VectorOperators.ADD);

        for (; i < baseOffsetsLength; i++) {
            int offset = clusterCount * i + Byte.toUnsignedInt(baseOffsets.get(i + baseOffsetsOffset));
            sumResult += partialSums.get(offset);
            aMagnitudeResult += aMagnitude.get(offset);
        }

        return (float) (sumResult / Math.sqrt(aMagnitudeResult * bMagnitude));
    }

    float pqDecodedCosineSimilarity256(ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
        var sum = FloatVector.zero(FloatVector.SPECIES_256);
        var vaMagnitude = FloatVector.zero(FloatVector.SPECIES_256);

        int[] convOffsets = scratchInt256.get();
        int i = 0;
        int limit = ByteVector.SPECIES_64.loopBound(baseOffsetsLength);

        var scale = IntVector.zero(IntVector.SPECIES_256).addIndex(clusterCount);

        for (; i < limit; i += ByteVector.SPECIES_64.length()) {

            fromByteSequence(ByteVector.SPECIES_64, baseOffsets, i + baseOffsets.offset() + baseOffsetsOffset)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_256)
                    .reinterpretAsInts()
                    .add(scale)
                    .intoArray(convOffsets, 0);

            var offset = i * clusterCount;
            sum = sum.add(fromVectorFloat(FloatVector.SPECIES_256, partialSums, offset, convOffsets, 0));
            vaMagnitude = vaMagnitude.add(fromVectorFloat(FloatVector.SPECIES_256, aMagnitude, offset, convOffsets, 0));
        }

        float sumResult = sum.reduceLanes(VectorOperators.ADD);
        float aMagnitudeResult = vaMagnitude.reduceLanes(VectorOperators.ADD);

        for (; i < baseOffsetsLength; i++) {
            int offset = clusterCount * i + Byte.toUnsignedInt(baseOffsets.get(i + baseOffsetsOffset));
            sumResult += partialSums.get(offset);
            aMagnitudeResult += aMagnitude.get(offset);
        }

        return (float) (sumResult / Math.sqrt(aMagnitudeResult * bMagnitude));
    }

    float pqDecodedCosineSimilarity128(ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
        // benchmarking showed that a 128-bit SIMD implementation performed worse than scalar
        float sum = 0.0f;
        float aMag = 0.0f;

        for (int m = 0; m < baseOffsetsLength; ++m) {
            int centroidIndex = Byte.toUnsignedInt(baseOffsets.get(m + baseOffsetsOffset));
            var index = m * clusterCount + centroidIndex;
            sum += partialSums.get(index);
            aMag += aMagnitude.get(index);
        }

        return (float) (sum / Math.sqrt(aMag * bMagnitude));
    }

    //---------------------------------------------
    // NVQ quantization instructions start here
    //---------------------------------------------

    static final FloatVector const1f = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 1.f);
    static final FloatVector const05f = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 0.5f);

    FloatVector logisticNQT(FloatVector vector, float alpha, float x0) {
        FloatVector temp = vector.fma(alpha, -alpha * x0);
        VectorMask<Float> isPositive = temp.test(VectorOperators.IS_NEGATIVE).not();
        IntVector p = temp.add(1, isPositive)
                .convert(VectorOperators.F2I, 0)
                .reinterpretAsInts();
        FloatVector e = p.convert(VectorOperators.I2F, 0).reinterpretAsFloats();
        IntVector m = temp.sub(e).fma(0.5f, 1).reinterpretAsInts();

        temp = m.add(p.lanewise(VectorOperators.LSHL, 23)).reinterpretAsFloats();  // temp = m * 2^p
        return temp.div(temp.add(1));
    }

    float logisticNQT(float value, float alpha, float x0) {
        float temp = Math.fma(value, alpha, -alpha * x0);
        int p = (int) Math.floor(temp + 1);
        int m = Float.floatToIntBits(Math.fma(temp - p, 0.5f, 1));

        temp = Float.intBitsToFloat(m + (p << 23));  // temp = m * 2^p
        return temp / (temp + 1);
    }

    FloatVector logitNQT(FloatVector vector, float inverseAlpha, float x0) {
        FloatVector z = vector.div(const1f.sub(vector));

        IntVector temp = z.reinterpretAsInts();
        FloatVector p = temp.and(0x7f800000)
                .lanewise(VectorOperators.LSHR, 23).sub(128)
                .convert(VectorOperators.I2F, 0)
                .reinterpretAsFloats();
        FloatVector m = temp.lanewise(VectorOperators.AND, 0x007fffff).add(0x3f800000).reinterpretAsFloats();

        return m.add(p).fma(inverseAlpha, x0);
    }

    float logitNQT(float value, float inverseAlpha, float x0) {
        float z = value / (1 - value);

        int temp = Float.floatToIntBits(z);
        int e = temp & 0x7f800000;
        float p = (float) ((e >> 23) - 128);
        float m = Float.intBitsToFloat((temp & 0x007fffff) + 0x3f800000);

        return Math.fma(m + p, inverseAlpha, x0);
    }

    FloatVector nvqDequantize8bit(ByteVector bytes, float inverseAlpha, float x0, float logisticScale, float logisticBias, int part) {
        /*
         * We unpack the vector using the FastLanes strategy:
         * https://www.vldb.org/pvldb/vol16/p2132-afroozeh.pdf?ref=blog.lancedb.com
         *
         * We treat the ByteVector bytes as a vector of integers.
         * | Int0                    | Int1                    | ...
         * | Byte3 Byte2 Byte1 Byte0 | Byte3 Byte2 Byte1 Byte0 | ...
         *
         * The argument part indicates which byte we want to extract from each integer.
         * With part=0, we extract
         *      Int0\Byte0, Int1\Byte0, etc.
         * With part=1, we shift by 8 bits and then extract
         *      Int0\Byte1, Int1\Byte1, etc.
         * With part=2, we shift by 16 bits and then extract
         *      Int0\Byte2, Int1\Byte2, etc.
         * With part=3, we shift by 24 bits and then extract
         *      Int0\Byte3, Int1\Byte3, etc.
         */
        var arr = bytes.reinterpretAsInts()
                .lanewise(VectorOperators.LSHR, 8 * part)
                .lanewise(VectorOperators.AND, 0xff)
                .convert(VectorOperators.I2F, 0)
                .reinterpretAsFloats();

        arr = arr.fma(logisticScale, logisticBias);
        return logitNQT(arr, inverseAlpha, x0);
    }

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
            var arr = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i);
            arr = logisticNQT(arr, scaledAlpha, scaledX0);
            arr = arr.sub(logisticBias).mul(invLogisticScale);
            var bytes = arr.add(const05f)
                    .convertShape(VectorOperators.F2B, ByteVector.SPECIES_PREFERRED, 0)
                    .reinterpretAsBytes();

            intoByteSequence(bytes, destination, i, mask);
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
            var arr = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i);
            var recArr = logisticNQT(arr, scaledAlpha, scaledX0);
            recArr = recArr.sub(logisticBias).mul(invLogisticScale);
            recArr = recArr.add(const05f)
                    .convert(VectorOperators.F2I, 0)
                    .reinterpretAsInts()
                    .convert(VectorOperators.I2F, 0)
                    .reinterpretAsFloats();
            recArr = recArr.fma(logisticScale, logisticBias);
            recArr = logitNQT(recArr, invScaledAlpha, scaledX0);

            var diff = arr.sub(recArr);
            squaredSumVec = diff.fma(diff, squaredSumVec);
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
            var arr = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i);
            var recArr = arr.sub(minValue).mul(constant / delta);
            recArr = recArr.add(const05f)
                    .convert(VectorOperators.F2I, 0)
                    .reinterpretAsInts()
                    .convert(VectorOperators.I2F, 0)
                    .reinterpretAsFloats();
            recArr = recArr.fma(delta / constant, minValue);

            var diff = arr.sub(recArr);
            squaredSumVec = diff.fma(diff, squaredSumVec);
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
            var byteArr = fromByteSequence(ByteVector.SPECIES_PREFERRED, quantizedVector, i);

            for (int j = 0; j < 4; j++) {
                var v1 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i + floatStep * j);
                var v2 = nvqDequantize8bit(byteArr, invScaledAlpha, scaledX0, logisticScale, logisticBias, j);

                var diff = v1.sub(v2);
                squaredSumVec = diff.fma(diff, squaredSumVec);
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
            var byteArr = fromByteSequence(ByteVector.SPECIES_PREFERRED, quantizedVector, i);

            for (int j = 0; j < 4; j++) {
                var v1 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i + floatStep * j);
                var v2 = nvqDequantize8bit(byteArr, invScaledAlpha, scaledX0, logisticScale, logisticBias, j);
                dotProdVec = v1.fma(v2, dotProdVec);
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
            var byteArr = fromByteSequence(ByteVector.SPECIES_PREFERRED, quantizedVector, i);

            for (int j = 0; j < 4; j++) {
                var va = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i + floatStep * j);
                var vb = nvqDequantize8bit(byteArr, invScaledAlpha, scaledX0, logisticScale, logisticBias, j);

                var vCentroid = fromVectorFloat(FloatVector.SPECIES_PREFERRED, centroid, i + floatStep * j);
                vb = vb.add(vCentroid);

                vsum = va.fma(vb, vsum);
                vbMagnitude = vb.fma(vb, vbMagnitude);
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

        final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED;
        final int len = x.length;
        final int upper = SPEC.loopBound(len);

        FloatVector sum = FloatVector.zero(SPEC);

        int i = 0;
        for (; i < upper; i += SPEC.length()) {
            FloatVector va = FloatVector.fromArray(SPEC, Arow, i);
            FloatVector vx = FloatVector.fromArray(SPEC, x, i);
            sum = va.fma(vx, sum);
        }

        float acc = sum.reduceLanes(VectorOperators.ADD);

        // scalar tail
        for (; i < len; i++) {
            acc += Arow[i] * x[i];
        }

        return acc;
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
        final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED;
        final int LANES = SPEC.length();

        // 4 Accumulators to hide instruction latency (Pipeline depth ~4)
        FloatVector acc0 = FloatVector.zero(SPEC);
        FloatVector acc1 = FloatVector.zero(SPEC);
        FloatVector acc2 = FloatVector.zero(SPEC);
        FloatVector acc3 = FloatVector.zero(SPEC);

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
                    VectorMask<Float> bitMask = VectorMask.fromLong(SPEC, m);
                    FloatVector loaded = FloatVector.fromArray(SPEC, tildeQ, qOffset + baseDim + off);

                    // Round-robin distribution to accumulators to break dependency chains
                    switch (accumulatorIdx & 3) { // (idx % 4)
                        case 0 -> acc0 = acc0.add(loaded, bitMask);
                        case 1 -> acc1 = acc1.add(loaded, bitMask);
                        case 2 -> acc2 = acc2.add(loaded, bitMask);
                        case 3 -> acc3 = acc3.add(loaded, bitMask);
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

                VectorMask<Float> bitMask = VectorMask.fromLong(SPEC, m);
                int absIdx = qOffset + localIdx;

                if (localIdx + LANES <= d) {
                    // Safe unmasked load
                    acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, absIdx), bitMask);
                } else {
                    // Boundary masked load
                    VectorMask<Float> rangeMask = SPEC.indexInRange(absIdx, qOffset + d);
                    acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, absIdx, rangeMask),
                            bitMask.and(rangeMask));
                }
            }
        }

        // Reduce all accumulators
        return addLanesPairwise(acc0.add(acc1).add(acc2).add(acc3));
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
        final VectorSpecies<Float> spec = FloatVector.SPECIES_PREFERRED;

        FloatVector acc0 = FloatVector.zero(spec);
        FloatVector acc1 = FloatVector.zero(spec);
        FloatVector acc2 = FloatVector.zero(spec);
        FloatVector acc3 = FloatVector.zero(spec);

        int fullWords = d >>> 6;

        for (int w = 0; w < fullWords; w++) {
            int qBase = qOffset + (w << 6);
            long packed = packedVectors[packedBase + w];

            acc0 = addMasked16(acc0, tildeQ, qBase, packed, 0, spec);
            acc1 = addMasked16(acc1, tildeQ, qBase + 16, packed, 16, spec);
            acc2 = addMasked16(acc2, tildeQ, qBase + 32, packed, 32, spec);
            acc3 = addMasked16(acc3, tildeQ, qBase + 48, packed, 48, spec);
        }

        int remainder = d & 63;
        if (remainder != 0) {
            int qBase = qOffset + (fullWords << 6);
            long packed = packedVectors[packedBase + fullWords];

            if (remainder > 0) {
                acc0 = addMaskedTail(
                        acc0, tildeQ, qBase, packed, 0,
                        Math.min(remainder, 16), spec);
            }
            if (remainder > 16) {
                acc1 = addMaskedTail(
                        acc1, tildeQ, qBase + 16, packed, 16,
                        Math.min(remainder - 16, 16), spec);
            }
            if (remainder > 32) {
                acc2 = addMaskedTail(
                        acc2, tildeQ, qBase + 32, packed, 32,
                        Math.min(remainder - 32, 16), spec);
            }
            if (remainder > 48) {
                acc3 = addMaskedTail(
                        acc3, tildeQ, qBase + 48, packed, 48,
                        remainder - 48, spec);
            }
        }

        return reduce4(acc0, acc1, acc2, acc3);
    }

    private float ashMaskedAdd8(
            float[] tildeQ,
            int qOffset,
            long[] packedVectors,
            int packedBase,
            int d) {
        final VectorSpecies<Float> spec = FloatVector.SPECIES_PREFERRED;

        FloatVector acc0 = FloatVector.zero(spec);
        FloatVector acc1 = FloatVector.zero(spec);
        FloatVector acc2 = FloatVector.zero(spec);
        FloatVector acc3 = FloatVector.zero(spec);
        FloatVector acc4 = FloatVector.zero(spec);
        FloatVector acc5 = FloatVector.zero(spec);
        FloatVector acc6 = FloatVector.zero(spec);
        FloatVector acc7 = FloatVector.zero(spec);

        int fullWords = d >>> 6;
        int w = 0;

        // Two packed words produce eight independent SIMD additions.
        for (; w + 1 < fullWords; w += 2) {
            int qBase = qOffset + (w << 6);

            long packed0 = packedVectors[packedBase + w];
            acc0 = addMasked16(acc0, tildeQ, qBase, packed0, 0, spec);
            acc1 = addMasked16(acc1, tildeQ, qBase + 16, packed0, 16, spec);
            acc2 = addMasked16(acc2, tildeQ, qBase + 32, packed0, 32, spec);
            acc3 = addMasked16(acc3, tildeQ, qBase + 48, packed0, 48, spec);

            long packed1 = packedVectors[packedBase + w + 1];
            int qBase1 = qBase + 64;

            acc4 = addMasked16(acc4, tildeQ, qBase1, packed1, 0, spec);
            acc5 = addMasked16(acc5, tildeQ, qBase1 + 16, packed1, 16, spec);
            acc6 = addMasked16(acc6, tildeQ, qBase1 + 32, packed1, 32, spec);
            acc7 = addMasked16(acc7, tildeQ, qBase1 + 48, packed1, 48, spec);
        }

        // One remaining complete word.
        if (w < fullWords) {
            int qBase = qOffset + (w << 6);
            long packed = packedVectors[packedBase + w];

            acc0 = addMasked16(acc0, tildeQ, qBase, packed, 0, spec);
            acc1 = addMasked16(acc1, tildeQ, qBase + 16, packed, 16, spec);
            acc2 = addMasked16(acc2, tildeQ, qBase + 32, packed, 32, spec);
            acc3 = addMasked16(acc3, tildeQ, qBase + 48, packed, 48, spec);
        }

        int remainder = d & 63;
        if (remainder != 0) {
            int qBase = qOffset + (fullWords << 6);
            long packed = packedVectors[packedBase + fullWords];

            if (remainder > 0) {
                acc4 = addMaskedTail(
                        acc4, tildeQ, qBase, packed, 0,
                        Math.min(remainder, 16), spec);
            }
            if (remainder > 16) {
                acc5 = addMaskedTail(
                        acc5, tildeQ, qBase + 16, packed, 16,
                        Math.min(remainder - 16, 16), spec);
            }
            if (remainder > 32) {
                acc6 = addMaskedTail(
                        acc6, tildeQ, qBase + 32, packed, 32,
                        Math.min(remainder - 32, 16), spec);
            }
            if (remainder > 48) {
                acc7 = addMaskedTail(
                        acc7, tildeQ, qBase + 48, packed, 48,
                        remainder - 48, spec);
            }
        }

        return reduce8(
                acc0, acc1, acc2, acc3,
                acc4, acc5, acc6, acc7);
    }

    private static FloatVector addMasked16(
            FloatVector accumulator,
            float[] values,
            int valueOffset,
            long packed,
            int bitShift,
            VectorSpecies<Float> spec) {
        long bits = (packed >>> bitShift) & 0xFFFFL;
        VectorMask<Float> selected = VectorMask.fromLong(spec, bits);

        return accumulator.add(
                FloatVector.fromArray(spec, values, valueOffset),
                selected);
    }

    private static FloatVector addMaskedTail(
            FloatVector accumulator,
            float[] values,
            int valueOffset,
            long packed,
            int bitShift,
            int laneCount,
            VectorSpecies<Float> spec) {
        VectorMask<Float> validLanes =
                spec.indexInRange(0, laneCount);

        long bits = (packed >>> bitShift) & 0xFFFFL;
        VectorMask<Float> selected =
                VectorMask.fromLong(spec, bits).and(validLanes);

        FloatVector vector =
                FloatVector.fromArray(
                        spec,
                        values,
                        valueOffset,
                        validLanes);

        return accumulator.add(vector, selected);
    }

    private static float reduce4(
            FloatVector acc0,
            FloatVector acc1,
            FloatVector acc2,
            FloatVector acc3) {
        FloatVector sum01 = acc0.add(acc1);
        FloatVector sum23 = acc2.add(acc3);

        return addLanesPairwise(sum01.add(sum23));
    }

    private static float reduce8(
            FloatVector acc0,
            FloatVector acc1,
            FloatVector acc2,
            FloatVector acc3,
            FloatVector acc4,
            FloatVector acc5,
            FloatVector acc6,
            FloatVector acc7) {
        FloatVector sum01 = acc0.add(acc1);
        FloatVector sum23 = acc2.add(acc3);
        FloatVector sum45 = acc4.add(acc5);
        FloatVector sum67 = acc6.add(acc7);

        return addLanesPairwise(sum01.add(sum23).add(sum45.add(sum67)));
    }

    // Shuffles that rotate SPECIES_PREFERRED by half, a quarter, ... of its lanes, for addLanesPairwise.
    private static final VectorShuffle<Float>[] PAIRWISE_ROTATIONS = pairwiseRotations();
    private static final int PAIRWISE_LANES = FloatVector.SPECIES_PREFERRED.length();
    private static final VectorShuffle<Float> ROTATE_8 = pairwiseRotation(8), ROTATE_4 = pairwiseRotation(4),
            ROTATE_2 = pairwiseRotation(2), ROTATE_1 = pairwiseRotation(1);

    private static VectorShuffle<Float> pairwiseRotation(int lanes) {
        return lanes < PAIRWISE_LANES ? VectorShuffle.iota(FloatVector.SPECIES_PREFERRED, lanes, 1, true) : null;
    }

    @SuppressWarnings("unchecked")
    private static VectorShuffle<Float>[] pairwiseRotations() {
        int lanes = FloatVector.SPECIES_PREFERRED.length();
        int steps = Integer.numberOfTrailingZeros(lanes);
        VectorShuffle<Float>[] rotations = (VectorShuffle<Float>[]) new VectorShuffle<?>[steps];
        for (int s = 0, half = lanes / 2; s < steps; s++, half /= 2)
            rotations[s] = VectorShuffle.iota(FloatVector.SPECIES_PREFERRED, half, 1, true);
        return rotations;
    }

    /**
     * The sum of {@code v}'s lanes (a SPECIES_PREFERRED vector) by a fixed tree of lane-wise adds: each lane plus
     * the lane half the width away, then a quarter, and so on. Unlike reduceLanes, whose order can change when the
     * JIT compiles its caller, every step here is exact lane-wise arithmetic, so the result
     * is the same compiled or not; and unlike a lane-by-lane loop it stays in vector registers.
     */
    private static float addLanesPairwise(FloatVector v) {
        // Straight-line steps for up to 16 lanes: carried around a loop, the vector was sometimes compiled as
        // heap-allocating calls.
        if (PAIRWISE_LANES > 16) {
            for (VectorShuffle<Float> rotation : PAIRWISE_ROTATIONS) v = v.add(v.rearrange(rotation));
            return v.lane(0);
        }
        if (PAIRWISE_LANES > 8) v = v.add(v.rearrange(ROTATE_8));
        if (PAIRWISE_LANES > 4) v = v.add(v.rearrange(ROTATE_4));
        if (PAIRWISE_LANES > 2) v = v.add(v.rearrange(ROTATE_2));
        return v.add(v.rearrange(ROTATE_1)).lane(0);
    }

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
        final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED;
        final int LANES = SPEC.length();

        for (int lane = 0; lane < blockLen; lane++) {
            // Vector accumulator: stays in registers (no array accumulation).
            FloatVector accVec = FloatVector.zero(SPEC);

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
                    VectorMask<Float> inRange = SPEC.indexInRange(qBase, d);

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
                            VectorMask.fromLong(SPEC, maskBits).and(inRange);

                    // masked load (paper)
                    FloatVector masked = FloatVector.fromArray(SPEC, tildeQ, qBase, bitMask);

                    // amortized accumulation (register)
                    accVec = accVec.add(masked);
                }
            }

            // One horizontal sum per lane (instead of per chunk)
            float maskedAdd = addLanesPairwise(accVec) + scalarAcc;

            outMaskedAdd[lane] = maskedAdd;
        }
    }
}
