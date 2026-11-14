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

package io.github.jbellis.jvector.quantization;

/**
 * Layout and scalar reference helpers for FusedASH neighborhood blocks.
 *
 * <p>The canonical ASH vector format remains owned by {@link AsymmetricHashing}.
 * FusedASH stores each origin node's neighbor codes in a neighborhood-local block
 * layout optimized for scoring the whole adjacency list at once. The projection
 * code itself follows the C++ fast-scan nibble model for bits-per-dimension
 * values {@code 1}, {@code 2}, and {@code 4}:</p>
 *
 * <ul>
 *   <li>1 bit/dim: four projected dimensions per nibble</li>
 *   <li>2 bits/dim: two projected dimensions per nibble</li>
 *   <li>4 bits/dim: one projected dimension per nibble</li>
 * </ul>
 *
 * <p>Within a fused block, bytes are stored by code-group pair first and lane
 * second:</p>
 *
 * <pre>
 *   byteIndex = blockOffset + (group / 2) * blockSize + lane
 * </pre>
 *
 * <p>The low nibble stores even groups and the high nibble stores odd groups.
 * This is deliberately graph-neighborhood-oriented: it supports block sizes
 * {@code 8}, {@code 16}, and {@code 32} without copying the IVF-specific B32
 * addressing scheme.</p>
 *
 * <p>Offset-binary codes (3 and 5 to 9 bits per dimension) use field blocks instead, scored with an
 * INT8 query: each unit of eight dimensions holds every lane's {@code b} code bytes contiguously,</p>
 *
 * <pre>
 *   byteIndex = blockOffset + (unit * blockSize + lane) * b + k,   k in [0, b)
 * </pre>
 *
 * <p>so one 64-bit word per lane holds that unit's eight fields. Both layouts are followed by the
 * same per-lane headers.</p>
 *
 * <p>Block and neighborhood byte-size calculations use checked arithmetic. Shapes whose
 * serialized storage cannot fit a Java array fail with {@link ArithmeticException}.</p>
 */
public final class FusedASHLayout {
    private FusedASHLayout() {}

    public static final int SCALE_BYTES = Short.BYTES;
    public static final int OFFSET_BYTES = Short.BYTES;
    public static final int LANDMARK_BYTES = Byte.BYTES;
    public static final int HEADER_BYTES_PER_LANE = AsymmetricHashing.HEADER_BYTES;

    /**
     * Every ASH width is supported. Each lane's canonical code is a bit stream scanned in 4-bit
     * groups: 1-bit signs and 2/4-bit sign-magnitude fields fill whole groups, and the offset-binary
     * fields of the other widths are linear in their bits, so any group of 4 bits has a 16-entry
     * query table.
     */
    public static boolean supportsBitsPerDimension(int bitsPerDimension) {
        return bitsPerDimension >= 1 && bitsPerDimension <= 9;
    }

    public static void validateBitsPerDimension(int bitsPerDimension) {
        if (!supportsBitsPerDimension(bitsPerDimension)) {
            throw new IllegalArgumentException(
                    "FusedASH supports bitsPerDimension 1 through 9; got " + bitsPerDimension);
        }
    }

    public static void validateBlockSize(int blockSize) {
        if (blockSize != 8 && blockSize != 16 && blockSize != 32) {
            throw new IllegalArgumentException(
                    "FusedASH blockSize must be one of {8,16,32}; got " + blockSize);
        }
    }

    public static int chooseBlockSize(int maxDegree) {
        if (maxDegree <= 0) {
            throw new IllegalArgumentException("maxDegree must be > 0");
        }
        if (maxDegree <= 8) return 8;
        if (maxDegree <= 16) return 16;
        return 32;
    }

    /** Dimensions per 4-bit group for the widths whose fields fill whole groups: 1, 2 and 4 bits. */
    public static int projectionDimsPerNibble(int bitsPerDimension) {
        if (bitsPerDimension != 1 && bitsPerDimension != 2 && bitsPerDimension != 4) {
            throw new IllegalArgumentException("Whole-nibble fields require 1, 2 or 4 bits; got " + bitsPerDimension);
        }
        return 4 / bitsPerDimension;
    }

    /** 4-bit groups in one canonical code: {@code ceil(quantizedDim * bits / 4)}. */
    public static int codeGroups(int quantizedDim, int bitsPerDimension) {
        if (quantizedDim <= 0) {
            throw new IllegalArgumentException("quantizedDim must be > 0");
        }
        validateBitsPerDimension(bitsPerDimension);
        return Math.toIntExact(((long) quantizedDim * bitsPerDimension + 3) / 4);
    }

    public static int canonicalCodeBytes(int quantizedDim, int bitsPerDimension) {
        return (codeGroups(quantizedDim, bitsPerDimension) - 1) / 2 + 1;
    }

    /**
     * Whether the graph neighborhood body uses field blocks: offset-binary codes (3 and 5 to 9 bits) are stored
     * as {@code [unit][lane][bits bytes]}, where a unit is eight dimensions, so each lane's eight fields are
     * contiguous for INT8 dot products. Other widths store byte {@code p} of every lane's code in row {@code p}.
     */
    public static boolean usesFieldBlocks(int bitsPerDimension) {
        return AsymmetricHashing.usesOffsetBinaryCode(bitsPerDimension);
    }

    /** Eight-dimension units in a field-block code. */
    public static int fieldUnits(int quantizedDim) {
        if (quantizedDim <= 0) {
            throw new IllegalArgumentException("quantizedDim must be > 0");
        }
        return (quantizedDim - 1) / 8 + 1;
    }

    /** Body bytes of one graph neighborhood block, in this width's layout. */
    public static int blockBodyBytes(int quantizedDim, int bitsPerDimension, int blockSize) {
        validateBlockSize(blockSize);
        if (usesFieldBlocks(bitsPerDimension)) {
            return Math.multiplyExact(Math.multiplyExact(fieldUnits(quantizedDim), bitsPerDimension), blockSize);
        }
        return nibbleBlockBodyBytes(quantizedDim, bitsPerDimension, blockSize);
    }

    /** Body bytes of one block whose rows hold byte {@code p} of every lane's code (the LUT layout). */
    public static int nibbleBlockBodyBytes(int quantizedDim, int bitsPerDimension, int blockSize) {
        validateBlockSize(blockSize);
        return Math.multiplyExact(canonicalCodeBytes(quantizedDim, bitsPerDimension), blockSize);
    }

    /**
     * Copies one lane's canonical code out of a graph neighborhood block into {@code dest}, which must hold
     * {@link #canonicalCodeBytes} bytes. For 1-bit codes these are the packed sign nibbles.
     */
    public static void readLaneCode(byte[] block, int blockOffset, int lane, int quantizedDim, int bitsPerDimension,
                                    int blockSize, byte[] dest) {
        validateLane(lane, blockSize);
        int bytes = canonicalCodeBytes(quantizedDim, bitsPerDimension);
        if (usesFieldBlocks(bitsPerDimension)) {
            int b = bitsPerDimension;
            for (int i = 0; i < bytes; i++) {
                int unit = i / b;
                dest[i] = block[blockOffset + (unit * blockSize + lane) * b + i % b];
            }
            return;
        }
        for (int i = 0; i < bytes; i++) dest[i] = block[blockOffset + i * blockSize + lane];
    }

    public static int blockHeaderBytes(int blockSize) {
        validateBlockSize(blockSize);
        return HEADER_BYTES_PER_LANE * blockSize;
    }

    public static int blockBytes(int quantizedDim, int bitsPerDimension, int blockSize) {
        return Math.addExact(blockBodyBytes(quantizedDim, bitsPerDimension, blockSize), blockHeaderBytes(blockSize));
    }

    public static int blocksPerNode(int maxDegree, int blockSize) {
        if (maxDegree <= 0) {
            throw new IllegalArgumentException("maxDegree must be > 0");
        }
        validateBlockSize(blockSize);
        return (maxDegree - 1) / blockSize + 1;
    }

    public static int featureSize(int maxDegree, int quantizedDim, int bitsPerDimension, int blockSize) {
        return Math.multiplyExact(blocksPerNode(maxDegree, blockSize), blockBytes(quantizedDim, bitsPerDimension, blockSize));
    }

    public static int blockOffset(int blockIndex, int quantizedDim, int bitsPerDimension, int blockSize) {
        if (blockIndex < 0) {
            throw new IllegalArgumentException("blockIndex must be nonnegative");
        }
        return Math.multiplyExact(blockIndex, blockBytes(quantizedDim, bitsPerDimension, blockSize));
    }

    public static int scaleOffset(int blockOffset, int quantizedDim, int bitsPerDimension, int blockSize) {
        return blockOffset + blockBodyBytes(quantizedDim, bitsPerDimension, blockSize);
    }

    public static int offsetOffset(int blockOffset, int quantizedDim, int bitsPerDimension, int blockSize) {
        return scaleOffset(blockOffset, quantizedDim, bitsPerDimension, blockSize) + SCALE_BYTES * blockSize;
    }

    public static int landmarkOffset(int blockOffset, int quantizedDim, int bitsPerDimension, int blockSize) {
        return offsetOffset(blockOffset, quantizedDim, bitsPerDimension, blockSize) + OFFSET_BYTES * blockSize;
    }

    public static int getPackedNibble(byte[] block, int blockOffset, int lane, int group, int blockSize) {
        validateLane(lane, blockSize);
        int byteIndex = blockOffset + (group >>> 1) * blockSize + lane;
        int value = block[byteIndex] & 0xFF;
        return ((group & 1) == 0) ? (value & 0x0F) : ((value >>> 4) & 0x0F);
    }

    public static void setPackedNibble(byte[] block, int blockOffset, int lane, int group, int blockSize, int nibble) {
        validateLane(lane, blockSize);
        int byteIndex = blockOffset + (group >>> 1) * blockSize + lane;
        int old = block[byteIndex] & 0xFF;
        int n = nibble & 0x0F;
        block[byteIndex] = (byte) (((group & 1) == 0)
                ? ((old & 0xF0) | n)
                : ((old & 0x0F) | (n << 4)));
    }

    public static int flatNibble(byte[] code, int group) {
        int b = code[group >>> 1] & 0xFF;
        return ((group & 1) == 0) ? (b & 0x0F) : ((b >>> 4) & 0x0F);
    }

    public static void setFlatNibble(byte[] code, int group, int nibble) {
        int byteIndex = group >>> 1;
        int old = code[byteIndex] & 0xFF;
        int n = nibble & 0x0F;
        code[byteIndex] = (byte) (((group & 1) == 0)
                ? ((old & 0xF0) | n)
                : ((old & 0x0F) | (n << 4)));
    }

    /** Value of a 1-bit sign or a 2/4-bit sign-magnitude field. */
    public static float decodeProjectionComponent(int field, int bitsPerDimension) {
        projectionDimsPerNibble(bitsPerDimension);
        if (bitsPerDimension == 1) {
            return field != 0 ? 1.0f : -1.0f;
        }

        int exBits = bitsPerDimension - 1;
        int magMask = (1 << exBits) - 1;
        boolean positive = ((field >>> exBits) & 1) != 0;
        float mag = (field & magMask) + 0.5f;
        return positive ? mag : -mag;
    }

    public static int signNibbleFromWords(long[] signWords, int group, int quantizedDim) {
        int baseDim = group << 2;
        int nibble = 0;
        for (int slot = 0; slot < 4; slot++) {
            int dim = baseDim + slot;
            if (dim >= quantizedDim) break;
            if (((signWords[dim >>> 6] >>> (dim & 63)) & 1L) != 0L) {
                nibble |= 1 << slot;
            }
        }
        return nibble;
    }

    public static void packQuantizedVector(
            byte[] block,
            int blockOffset,
            int lane,
            AsymmetricHashing.QuantizedVector vector,
            int quantizedDim,
            int bitsPerDimension,
            int blockSize) {
        validateBitsPerDimension(bitsPerDimension);
        validateLane(lane, blockSize);

        int groups = codeGroups(quantizedDim, bitsPerDimension);
        if (bitsPerDimension == 1) {
            for (int group = 0; group < groups; group++) {
                setPackedNibble(block, blockOffset, lane, group, blockSize,
                        signNibbleFromWords(vector.binaryVector, group, quantizedDim));
            }
        } else if (usesFieldBlocks(bitsPerDimension)) {
            byte[] code = vector.extraBits;
            int bytes = canonicalCodeBytes(quantizedDim, bitsPerDimension);
            if (code == null || code.length < bytes) {
                throw new IllegalArgumentException("ASH projection code is missing or too short");
            }
            // Bytes past the canonical code, in the last unit, are zero padding.
            int b = bitsPerDimension, units = fieldUnits(quantizedDim);
            for (int unit = 0; unit < units; unit++) {
                int destination = blockOffset + (unit * blockSize + lane) * b;
                for (int k = 0; k < b; k++) {
                    int i = unit * b + k;
                    block[destination + k] = i < bytes ? code[i] : 0;
                }
            }
        } else {
            byte[] code = vector.extraBits;
            int bytes = canonicalCodeBytes(quantizedDim, bitsPerDimension);
            if (code == null || code.length < bytes) {
                throw new IllegalArgumentException("ASH projection code is missing or too short");
            }
            int fullBytes = groups >>> 1;
            int destination = blockOffset + lane;
            for (int i = 0; i < fullBytes; i++, destination += blockSize) {
                block[destination] = code[i];
            }
            if ((groups & 1) != 0) {
                block[destination] = (byte) ((block[destination] & 0xF0) | (code[fullBytes] & 0x0F));
            }
        }

        writeLaneHeader(block, blockOffset, lane, quantizedDim, bitsPerDimension, blockSize,
                vector.scale, vector.offset, vector.landmark);
    }

    /**
     * Packs one canonical ASH vector into the graph's fused neighborhood representation.
     *
     * <p>Fused scoring uses {@code Aq} for every bit width. Canonical 1/2/4-bit offsets already
     * contain the landmark adjustment {@code -scale * dot(A(mu), code)}, so the header is copied as
     * stored. Benchmark fixtures call this to use the graph writer's exact bytes.</p>
     *
     * @param block destination containing a complete serialized block
     * @param blockOffset byte offset of that block
     * @param lane destination lane within the block
     * @param vector canonical quantized vector, whose landmark belongs to {@code ash}
     * @param ash compressor that owns the vector's projection and landmarks
     * @param blockSize serialized block width: 8, 16, or 32
     * @throws IllegalArgumentException if the landmark or block shape is invalid
     */
    public static void packForFused(byte[] block, int blockOffset, int lane,
                                    AsymmetricHashing.QuantizedVector vector,
                                    AsymmetricHashing ash, int blockSize) {
        int landmark = vector.landmark & 0xFF;
        if (landmark >= ash.landmarkCount) {
            throw new IllegalArgumentException("Invalid ASH landmark: " + landmark);
        }
        packQuantizedVector(block, blockOffset, lane, vector,
                ash.quantizedDim, ash.bitsPerDimension, blockSize);
    }

    public static void writeLaneHeader(
            byte[] block,
            int blockOffset,
            int lane,
            int quantizedDim,
            int bitsPerDimension,
            int blockSize,
            float scale,
            float offset,
            byte landmark) {
        validateLane(lane, blockSize);
        writeFloat16(block, scaleOffset(blockOffset, quantizedDim, bitsPerDimension, blockSize) + lane * SCALE_BYTES, scale);
        writeFloat16(block, offsetOffset(blockOffset, quantizedDim, bitsPerDimension, blockSize) + lane * OFFSET_BYTES, offset);
        block[landmarkOffset(blockOffset, quantizedDim, bitsPerDimension, blockSize) + lane] = landmark;
    }

    public static float readScale(byte[] block, int blockOffset, int lane, int quantizedDim, int bitsPerDimension, int blockSize) {
        validateLane(lane, blockSize);
        return readFloat16(block, scaleOffset(blockOffset, quantizedDim, bitsPerDimension, blockSize) + lane * SCALE_BYTES);
    }

    public static float readOffset(byte[] block, int blockOffset, int lane, int quantizedDim, int bitsPerDimension, int blockSize) {
        validateLane(lane, blockSize);
        return readFloat16(block, offsetOffset(blockOffset, quantizedDim, bitsPerDimension, blockSize) + lane * OFFSET_BYTES);
    }

    public static int readLandmark(byte[] block, int blockOffset, int lane, int quantizedDim, int bitsPerDimension, int blockSize) {
        validateLane(lane, blockSize);
        return block[landmarkOffset(blockOffset, quantizedDim, bitsPerDimension, blockSize) + lane] & 0xFF;
    }

    public static void buildQueryLut(float[] qProj, int quantizedDim, int bitsPerDimension, float[] lut) {
        if (AsymmetricHashing.usesOffsetBinaryCode(bitsPerDimension)) {
            buildOffsetBinaryLut(qProj, quantizedDim, bitsPerDimension, lut);
            return;
        }
        validateBitsPerDimension(bitsPerDimension);
        int groupDims = projectionDimsPerNibble(bitsPerDimension);
        int groups = codeGroups(quantizedDim, bitsPerDimension);
        if (qProj.length < quantizedDim) {
            throw new IllegalArgumentException("qProj length is smaller than quantizedDim");
        }
        if (lut.length < groups * 16) {
            throw new IllegalArgumentException("lut length is smaller than groups * 16");
        }

        // Every entry is still 0 + q[d0]*c0 + q[d1]*c1 + ..., added in slot order,
        // so the table is bit-identical to the per-entry definition. Each product
        // is formed once per dimension instead of once per entry, and the field
        // decode is a constant table rather than a branch per term.
        int complete = quantizedDim / groupDims;
        if (bitsPerDimension == 2) {
            for (int group = 0; group < complete; group++) lut2(qProj, group << 1, lut, group << 4);
        } else if (bitsPerDimension == 1) {
            for (int group = 0; group < complete; group++) lut1(qProj, group << 2, lut, group << 4);
        } else {
            for (int group = 0; group < complete; group++) {
                float q = qProj[group];
                int base = group << 4;
                for (int field = 0; field < 16; field++) lut[base + field] = 0f + q * DECODE_4BIT[field];
            }
        }
        // A partial final group adds only the dimensions that exist.
        for (int group = complete; group < groups; group++) {
            int dimBase = group * groupDims;
            int lutBase = group << 4;
            for (int nibble = 0; nibble < 16; nibble++) {
                float sum = 0f;
                for (int slot = 0; slot < groupDims; slot++) {
                    int dim = dimBase + slot;
                    if (dim >= quantizedDim) break;
                    int mask = (1 << bitsPerDimension) - 1;
                    int field = (nibble >>> (slot * bitsPerDimension)) & mask;
                    sum += qProj[dim] * decodeProjectionComponent(field, bitsPerDimension);
                }
                lut[lutBase + nibble] = sum;
            }
        }
    }

    /**
     * Table for offset-binary codes. Group {@code k} covers code bits {@code 4k..4k+3}; bit {@code i} is
     * bit {@code i % b} of dimension {@code i / b} and adds {@code 2^(i % b) * q[i / b]}. Each dimension's
     * centering term {@code -(2^b - 1)/2 * q[j]} is added to every entry of the group holding its bit 0,
     * so each lookup sum includes it exactly once.
     */
    private static void buildOffsetBinaryLut(float[] qProj, int quantizedDim, int bits, float[] lut) {
        int groups = codeGroups(quantizedDim, bits);
        if (qProj.length < quantizedDim) {
            throw new IllegalArgumentException("qProj length is smaller than quantizedDim");
        }
        if (lut.length < groups * 16) {
            throw new IllegalArgumentException("lut length is smaller than groups * 16");
        }
        float center = ((1 << bits) - 1) * 0.5f;
        float[] weight = new float[4];
        for (int group = 0; group < groups; group++) {
            float base = 0f;
            for (int slot = 0; slot < 4; slot++) {
                int bit = 4 * group + slot, dim = bit / bits, power = bit % bits;
                weight[slot] = dim < quantizedDim ? qProj[dim] * (float) (1 << power) : 0f;
                if (dim < quantizedDim && power == 0) base -= center * qProj[dim];
            }
            int at = group << 4;
            lut[at] = base;
            for (int n = 1; n < 16; n++) lut[at + n] = lut[at + (n & (n - 1))] + weight[Integer.numberOfTrailingZeros(n)];
        }
    }

    private static final float[] DECODE_4BIT = new float[16];
    static {
        for (int field = 0; field < 16; field++) DECODE_4BIT[field] = decodeProjectionComponent(field, 4);
    }

    // Two 2-bit fields per nibble: entry = (0 + a[n & 3]) + b[n >>> 2].
    // Field values -1.5, -0.5, 0.5, 1.5 (sign bit high, magnitude low).
    private static void lut2(float[] qProj, int dim, float[] lut, int base) {
        float q0 = qProj[dim], q1 = qProj[dim + 1];
        float a0 = 0f + q0 * -0.5f, a1 = 0f + q0 * -1.5f, a2 = 0f + q0 * 0.5f, a3 = 0f + q0 * 1.5f;
        float b0 = q1 * -0.5f, b1 = q1 * -1.5f, b2 = q1 * 0.5f, b3 = q1 * 1.5f;
        lut[base] = a0 + b0; lut[base + 1] = a1 + b0; lut[base + 2] = a2 + b0; lut[base + 3] = a3 + b0;
        lut[base + 4] = a0 + b1; lut[base + 5] = a1 + b1; lut[base + 6] = a2 + b1; lut[base + 7] = a3 + b1;
        lut[base + 8] = a0 + b2; lut[base + 9] = a1 + b2; lut[base + 10] = a2 + b2; lut[base + 11] = a3 + b2;
        lut[base + 12] = a0 + b3; lut[base + 13] = a1 + b3; lut[base + 14] = a2 + b3; lut[base + 15] = a3 + b3;
    }

    // Four sign bits per nibble: entry = (((0 + ±q0) + ±q1) + ±q2) + ±q3.
    private static void lut1(float[] qProj, int dim, float[] lut, int base) {
        float p0 = qProj[dim] * 1.0f, n0 = qProj[dim] * -1.0f;
        float p1 = qProj[dim + 1] * 1.0f, n1 = qProj[dim + 1] * -1.0f;
        float p2 = qProj[dim + 2] * 1.0f, n2 = qProj[dim + 2] * -1.0f;
        float p3 = qProj[dim + 3] * 1.0f, n3 = qProj[dim + 3] * -1.0f;
        for (int nibble = 0; nibble < 16; nibble++) {
            float sum = 0f + ((nibble & 1) != 0 ? p0 : n0);
            sum += (nibble & 2) != 0 ? p1 : n1;
            sum += (nibble & 4) != 0 ? p2 : n2;
            sum += (nibble & 8) != 0 ? p3 : n3;
            lut[base + nibble] = sum;
        }
    }

    public static float scoreLane(
            byte[] block,
            int blockOffset,
            int lane,
            int quantizedDim,
            int bitsPerDimension,
            int blockSize,
            float[] lut,
            float[] dotQMuByLandmark) {
        validateLane(lane, blockSize);
        int groups = codeGroups(quantizedDim, bitsPerDimension);
        float ip = 0f;
        for (int group = 0; group < groups; group++) {
            int nibble = getPackedNibble(block, blockOffset, lane, group, blockSize);
            ip += lut[(group << 4) + nibble];
        }

        int landmark = readLandmark(block, blockOffset, lane, quantizedDim, bitsPerDimension, blockSize);
        float dotQMu = landmark < dotQMuByLandmark.length ? dotQMuByLandmark[landmark] : 0f;
        return readScale(block, blockOffset, lane, quantizedDim, bitsPerDimension, blockSize) * ip
                + dotQMu
                + readOffset(block, blockOffset, lane, quantizedDim, bitsPerDimension, blockSize);
    }

    private static void validateLane(int lane, int blockSize) {
        validateBlockSize(blockSize);
        if (lane < 0 || lane >= blockSize) {
            throw new IllegalArgumentException("lane out of range: " + lane + " for blockSize=" + blockSize);
        }
    }

    private static void writeFloat16(byte[] out, int offset, float value) {
        int bits = floatToHalfBits(value);
        out[offset] = (byte) ((bits >>> 8) & 0xFF);
        out[offset + 1] = (byte) (bits & 0xFF);
    }

    private static float readFloat16(byte[] in, int offset) {
        int hi = in[offset] & 0xFF;
        int lo = in[offset + 1] & 0xFF;
        return halfBitsToFloat((hi << 8) | lo);
    }

    /** IEEE-754 binary16 round-to-nearest-even conversion. */
    private static int floatToHalfBits(float value) {
        int bits = Float.floatToRawIntBits(value);
        int sign = (bits >>> 16) & 0x8000;
        int abs = bits & 0x7FFF_FFFF;

        if (abs >= 0x7F80_0000) {
            if ((abs & 0x007F_FFFF) == 0) {
                return sign | 0x7C00;
            }
            return sign | 0x7E00;
        }

        int exp = ((abs >>> 23) & 0xFF) - 127 + 15;
        int mant = abs & 0x007F_FFFF;

        if (exp >= 31) {
            return sign | 0x7C00;
        }

        if (exp <= 0) {
            if (exp < -10) {
                return sign;
            }

            mant |= 0x0080_0000;
            int shift = 14 - exp;
            int halfMant = mant >>> shift;

            int roundBit = 1 << (shift - 1);
            int remainder = mant & (roundBit - 1);

            if ((mant & roundBit) != 0 && (remainder != 0 || (halfMant & 1) != 0)) {
                halfMant++;
            }

            return sign | halfMant;
        }

        int halfMant = mant >>> 13;
        int roundBit = 0x0000_1000;
        int remainder = mant & (roundBit - 1);

        if ((mant & roundBit) != 0 && (remainder != 0 || (halfMant & 1) != 0)) {
            halfMant++;
            if (halfMant == 0x0400) {
                halfMant = 0;
                exp++;
                if (exp >= 31) {
                    return sign | 0x7C00;
                }
            }
        }

        return sign | (exp << 10) | halfMant;
    }

    private static float halfBitsToFloat(int half) {
        int h = half & 0xFFFF;
        int sign = (h & 0x8000) << 16;
        int exp = (h >>> 10) & 0x1F;
        int mant = h & 0x03FF;

        if (exp == 0) {
            if (mant == 0) {
                return Float.intBitsToFloat(sign);
            }

            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= 0x03FF;
        } else if (exp == 31) {
            return Float.intBitsToFloat(sign | 0x7F80_0000 | (mant << 13));
        }

        int floatExp = exp + (127 - 15);
        return Float.intBitsToFloat(sign | (floatExp << 23) | (mant << 13));
    }
}
