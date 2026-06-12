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
 */
public final class FusedASHLayout {
    private FusedASHLayout() {}

    public static final int SCALE_BYTES = Short.BYTES;
    public static final int OFFSET_BYTES = Short.BYTES;
    public static final int LANDMARK_BYTES = Byte.BYTES;
    public static final int HEADER_BYTES_PER_LANE = SCALE_BYTES + OFFSET_BYTES + LANDMARK_BYTES;

    public static boolean supportsBitsPerDimension(int bitsPerDimension) {
        return bitsPerDimension == 1 || bitsPerDimension == 2 || bitsPerDimension == 4;
    }

    public static void validateBitsPerDimension(int bitsPerDimension) {
        if (!supportsBitsPerDimension(bitsPerDimension)) {
            throw new IllegalArgumentException(
                    "FusedASH supports bitsPerDimension in {1,2,4}; got " + bitsPerDimension);
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

    public static int projectionDimsPerNibble(int bitsPerDimension) {
        validateBitsPerDimension(bitsPerDimension);
        return 4 / bitsPerDimension;
    }

    public static int codeGroups(int quantizedDim, int bitsPerDimension) {
        if (quantizedDim <= 0) {
            throw new IllegalArgumentException("quantizedDim must be > 0");
        }
        int groupDims = projectionDimsPerNibble(bitsPerDimension);
        return (quantizedDim + groupDims - 1) / groupDims;
    }

    public static int canonicalCodeBytes(int quantizedDim, int bitsPerDimension) {
        return (codeGroups(quantizedDim, bitsPerDimension) + 1) >>> 1;
    }

    public static int blockBodyBytes(int quantizedDim, int bitsPerDimension, int blockSize) {
        validateBlockSize(blockSize);
        return canonicalCodeBytes(quantizedDim, bitsPerDimension) * blockSize;
    }

    public static int blockHeaderBytes(int blockSize) {
        validateBlockSize(blockSize);
        return HEADER_BYTES_PER_LANE * blockSize;
    }

    public static int blockBytes(int quantizedDim, int bitsPerDimension, int blockSize) {
        return blockBodyBytes(quantizedDim, bitsPerDimension, blockSize) + blockHeaderBytes(blockSize);
    }

    public static int blocksPerNode(int maxDegree, int blockSize) {
        if (maxDegree <= 0) {
            throw new IllegalArgumentException("maxDegree must be > 0");
        }
        validateBlockSize(blockSize);
        return (maxDegree + blockSize - 1) / blockSize;
    }

    public static int featureSize(int maxDegree, int quantizedDim, int bitsPerDimension, int blockSize) {
        return blocksPerNode(maxDegree, blockSize) * blockBytes(quantizedDim, bitsPerDimension, blockSize);
    }

    public static int blockOffset(int blockIndex, int quantizedDim, int bitsPerDimension, int blockSize) {
        return blockIndex * blockBytes(quantizedDim, bitsPerDimension, blockSize);
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

    public static float decodeProjectionComponent(int field, int bitsPerDimension) {
        validateBitsPerDimension(bitsPerDimension);
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
        } else {
            byte[] code = vector.extraBits;
            int bytes = canonicalCodeBytes(quantizedDim, bitsPerDimension);
            if (code == null || code.length < bytes) {
                throw new IllegalArgumentException("ASH projection code is missing or too short");
            }
            for (int group = 0; group < groups; group++) {
                setPackedNibble(block, blockOffset, lane, group, blockSize, flatNibble(code, group));
            }
        }

        writeLaneHeader(block, blockOffset, lane, quantizedDim, bitsPerDimension, blockSize,
                vector.scale, vector.offset, vector.landmark);
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
        validateBitsPerDimension(bitsPerDimension);
        int groupDims = projectionDimsPerNibble(bitsPerDimension);
        int groups = codeGroups(quantizedDim, bitsPerDimension);
        if (qProj.length < quantizedDim) {
            throw new IllegalArgumentException("qProj length is smaller than quantizedDim");
        }
        if (lut.length < groups * 16) {
            throw new IllegalArgumentException("lut length is smaller than groups * 16");
        }

        for (int group = 0; group < groups; group++) {
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
