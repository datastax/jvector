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

import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.VectorUtil;

/**
 * Symmetric ASH scoring from already encoded operands. Graph construction remains C=1.
 * C=1 uses Appendix B directly. Multiple landmarks use the same header-calibrated
 * decoded dot product, including cross-landmark terms, without projecting raw vectors.
 */
public final class ASHSymmetricScorer {
    public enum Kernel { AUTO, SCALAR, SIMD }
    private static final java.lang.invoke.VarHandle SYMMETRIC_WORD = java.lang.invoke.MethodHandles.byteArrayViewVarHandle(long[].class, java.nio.ByteOrder.LITTLE_ENDIAN);
    private static final short[] DOT_4 = nibbleProducts(4);
    private final ASHVectors vectors;
    private final AsymmetricHashing ash;
    private final int dimensions;
    private final int bits;
    private final int landmarks;
    private final float[][] landmarkDots;
    private final float[] centeredOffsets;
    private final boolean simd;
    private final io.github.jbellis.jvector.vector.VectorUtilSupport backend =
            io.github.jbellis.jvector.vector.VectorizationProvider.getInstance().getVectorUtilSupport();

    public ASHSymmetricScorer(ASHVectors vectors) {
        this(vectors, Kernel.valueOf(System.getProperty("jvector.ash.singleKernel", "auto")
                .trim().toUpperCase(java.util.Locale.ROOT)));
    }

    public ASHSymmetricScorer(ASHVectors vectors, Kernel kernel) {
        this.vectors = vectors;
        ash = vectors.getCompressor();
        dimensions = ash.quantizedDim;
        bits = ash.bitsPerDimension;
        landmarks = ash.landmarkCount;
        if (landmarks < 1 || landmarks > 256) throw new IllegalArgumentException("ASH requires 1..256 landmarks");
        simd = selectSimd(kernel);
        landmarkDots = new float[landmarks][landmarks];
        for (int a = 0; a < landmarks; a++) {
            for (int b = 0; b <= a; b++) {
                landmarkDots[a][b] = landmarkDots[b][a] = VectorUtil.dotProduct(ash.landmarks[a], ash.landmarks[b]);
            }
        }
        if (AsymmetricHashing.usesFastScanProjectionCode(bits)) {
            centeredOffsets = new float[vectors.count()];
            for (int i = 0; i < centeredOffsets.length; i++) centeredOffsets[i] = centeredOffset(vectors.get(i));
        } else {
            centeredOffsets = null;
        }
    }

    private boolean selectSimd(Kernel kernel) {
        boolean supported = (bits == 1 || bits == 2 || bits == 4) && backend.supportsAshSymmetricScoring();
        if (landmarks != 1) supported &= bits == 1 ? backend.supportsAshMaskedLoad() : backend.supportsAshProjectionScoring();
        if (kernel == Kernel.SIMD && !supported) throw new UnsupportedOperationException("Symmetric SIMD requires a vector backend and bits 1, 2, or 4");
        return kernel != Kernel.SCALAR && supported;
    }

    public String description() { return "symmetric " + (simd ? "SIMD" : "scalar") + ", bits=" + bits + ", C=" + landmarks; }

    private float centeredOffset(AsymmetricHashing.QuantizedVector vector) {
        int c = vector.landmark & 255;
        if (c >= landmarks) throw new IllegalArgumentException("Invalid landmark " + c);
        if (!AsymmetricHashing.usesFastScanProjectionCode(bits)) return vector.offset;
        return vector.offset + vector.scale * AsymmetricHashing.dotProjectionCode(
                ash.landmarkProj[c], vector.extraBits, dimensions, bits);
    }

    private float offset(int node) { return centeredOffsets == null ? vectors.get(node).offset : centeredOffsets[node]; }

    private float packedDot(AsymmetricHashing.QuantizedVector a, AsymmetricHashing.QuantizedVector b) {
        return simd ? backend.ashSymmetricDot(a.binaryVector, a.extraBits, b.binaryVector, b.extraBits, dimensions, bits)
                : codeDot(a, b, dimensions, bits);
    }

    /** Raw symmetric dot product. No encoding or original-dimensional projection. */
    public float dotProduct(int a, int b) {
        var x = vectors.get(a); var y = vectors.get(b);
        int ca = x.landmark & 255, cb = y.landmark & 255;
        float crossA = 0, crossB = 0;
        if (ca != cb) {
            for (int j = 0; j < dimensions; j++) {
                float delta = ash.landmarkProj[cb][j] - ash.landmarkProj[ca][j];
                crossA += component(x, j, bits) * delta;
                crossB += component(y, j, bits) * -delta;
            }
        }
        return (x.scale * y.scale) * packedDot(x, y)
                + ((x.scale * crossA + y.scale * crossB) + (offset(a) + offset(b))) + landmarkDots[ca][cb];
    }

    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(int node) {
        if (landmarks != 1) return scoreFunctionFor(vectors.get(node));
        return c1ScoreFunction(vectors.get(node), offset(node));
    }

    /** The operand must already be encoded with this instance's compressor. */
    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(AsymmetricHashing.QuantizedVector query) {
        if (landmarks != 1) return vectors.scoreFunctionFor(preparedState(query), simd);
        return c1ScoreFunction(query, centeredOffset(query));
    }

    private ScoreFunction.ApproximateScoreFunction c1ScoreFunction(
            AsymmetricHashing.QuantizedVector query, float queryOffset) {
        return node -> {
            var target = vectors.get(node);
            float dot = (query.scale * target.scale) * packedDot(query, target)
                    + (queryOffset + offset(node)) + landmarkDots[0][0];
            return ASHScorer.toSimilarity(dot);
        };
    }

    /** Score encoded pairs in ranges; C=1 uses integer kernels, C>1 reuses projected-state kernels. */
    public ASHBlockScorer blockScorerFor(AsymmetricHashing.QuantizedVector query, int blockSize) {
        Kernel kernel = Kernel.valueOf(System.getProperty("jvector.ash.blockKernel", "auto")
                .trim().toUpperCase(java.util.Locale.ROOT));
        return blockScorerFor(query, blockSize, kernel);
    }

    public ASHBlockScorer blockScorerFor(AsymmetricHashing.QuantizedVector query, int blockSize, Kernel kernel) {
        if (blockSize <= 0) throw new IllegalArgumentException("blockSize must be positive");
        if (landmarks == 1 && (bits == 2 || bits == 4)) {
            if (kernel == Kernel.SIMD && !backend.supportsAshSymmetricScoring()) throw new UnsupportedOperationException("Symmetric SIMD is unavailable");
            final boolean useVector = kernel != Kernel.SCALAR && backend.supportsAshSymmetricScoring();
            final byte[][] blocks = vectors.symmetricProjectionBlocks(blockSize);
            final int components = 4 / bits;
            final int groups = (dimensions + components - 1) / components;
            final short[] lut = new short[Math.multiplyExact(groups, 32)];
            for (int g = 0; g < groups; g++) {
                int q0 = (int)(2*component(query, g*components, bits));
                if (bits == 4) {
                    for (int magnitude = 0; magnitude < 8; magnitude++) {
                        int product = q0*(2*magnitude+1);
                        lut[g*32+magnitude] = (short)-product;
                        lut[g*32+magnitude+8] = (short)product;
                    }
                } else {
                    int q1 = g*components+1 < dimensions ? (int)(2*component(query,g*components+1,bits)) : 0;
                    for (int value = 0; value < 16; value++) {
                        int a = 2*(value&1)+1, b = 2*((value>>>2)&1)+1;
                        if ((value&2)==0) a = -a;
                        if ((value&8)==0) b = -b;
                        lut[g*32+value] = (short)(q0*a+q1*b);
                    }
                }
                System.arraycopy(lut,g*32,lut,g*32+16,16);
            }
            final float queryOffset = centeredOffset(query);
            return new ASHBlockScorer() {
                public String description() { return "symmetric exact integer LUT " + (useVector ? "SIMD" : "scalar"); }
                public void scoreRange(int start, int count, float[] out) {
                    java.util.Objects.checkFromIndexSize(start, count, vectors.count());
                    java.util.Objects.checkFromIndexSize(0, count, out.length);
                    for (int done = 0; done < count;) {
                        int ordinal = start + done, lane = ordinal % blockSize;
                        int lanes = Math.min(count - done, blockSize - lane);
                        byte[] body = blocks[ordinal / blockSize];
                        if (useVector) backend.ashSymmetricLutScore(body, groups, blockSize, lane, lanes, lut, out, done);
                        else for (int i = 0; i < lanes; i++) {
                            long sum = 0;
                            for (int g = 0; g < groups; g++) sum += lut[g * 32 + ((body[(g / 2)*blockSize+lane+i] >>> (4*(g%2))) & 15)];
                            out[done+i] = sum*.25f;
                        }
                        for (int i = 0; i < lanes; i++) {
                            var target = vectors.get(ordinal+i);
                            out[done+i] = ASHScorer.toSimilarity((query.scale*target.scale)*out[done+i]
                                    +(queryOffset+offset(ordinal+i))+landmarkDots[0][0]);
                        }
                        done += lanes;
                    }
                }
            };
        }
        if (bits == 1 && landmarks == 1) {
            boolean supported = backend.supportsAshSymmetricScoring();
            if (kernel == Kernel.SIMD && !supported) throw new UnsupportedOperationException("Symmetric SIMD is unavailable");
            boolean vector = kernel != Kernel.SCALAR && supported;
            float queryOffset = centeredOffset(query);
            return new ASHBlockScorer() {
                public String description() { return "symmetric binary packed-pair range " + (vector ? "SIMD" : "scalar"); }
                public void scoreRange(int start, int count, float[] out) {
                    java.util.Objects.checkFromIndexSize(start, count, vectors.count());
                    java.util.Objects.checkFromIndexSize(0, count, out.length);
                    for (int i = 0; i < count; i++) {
                        int node = start + i;
                        var target = vectors.get(node);
                        float dot = vector ? backend.ashSymmetricDot(query.binaryVector, query.extraBits,
                                target.binaryVector, target.extraBits, dimensions, bits)
                                : codeDot(query, target, dimensions, bits);
                        out[i] = ASHScorer.toSimilarity((query.scale * target.scale) * dot
                                + (queryOffset + offset(node)) + landmarkDots[0][0]);
                    }
                }
            };
        }
        boolean supported = bits == 1 ? backend.supportsAshMaskedLoad()
                : (bits == 2 || bits == 4) && backend.supportsAshLutScoring();
        if (kernel == Kernel.SIMD && !supported) throw new UnsupportedOperationException("Symmetric block SIMD is unavailable for this backend/bit width");
        return vectors.blockScorerFor(preparedState(query), blockSize, kernel != Kernel.SCALAR && supported);
    }

    // qProj is derived from encoded coordinates, NOT from W times a raw D-dimensional query.
    // Header calibration adds the query's stored reconstruction correction to each centroid dot.
    private ASHScorer.QueryPrecompute preparedState(AsymmetricHashing.QuantizedVector query) {
        int cq = query.landmark & 255;
        if (cq >= landmarks) throw new IllegalArgumentException("Invalid query landmark " + cq);
        float[] code = new float[dimensions];
        float[] qProj = new float[dimensions];
        float ownProjection = 0;
        for (int j = 0; j < dimensions; j++) {
            code[j] = component(query, j, bits);
            qProj[j] = ash.landmarkProj[cq][j] + query.scale * code[j];
            ownProjection += code[j] * ash.landmarkProj[cq][j];
        }
        float correction = AsymmetricHashing.usesFastScanProjectionCode(bits)
                ? query.offset : query.offset - query.scale * ownProjection;
        float[] tilde = new float[Math.multiplyExact(landmarks, dimensions)];
        float[] sums = new float[landmarks];
        float[] dots = new float[landmarks];
        for (int c = 0; c < landmarks; c++) {
            float dot = 0, sum = 0;
            for (int j = 0; j < dimensions; j++) {
                dot += code[j] * ash.landmarkProj[c][j];
                float value = qProj[j] - ash.landmarkProj[c][j];
                tilde[c * dimensions + j] = value;
                sum += value;
            }
            sums[c] = sum;
            dots[c] = landmarkDots[cq][c] + query.scale * dot + correction;
        }
        return new ASHScorer.QueryPrecompute(dimensions, landmarks, qProj, tilde, sums, dots);
    }

    private static float component(AsymmetricHashing.QuantizedVector v, int j, int bits) {
        if (bits == 1) return AsymmetricHashing.QuantizedVector.getBit(v.binaryVector, j) ? 1f : -1f;
        if (bits == 2 || bits == 4) return AsymmetricHashing.projectionComponent(v.extraBits, j, bits);
        int ex = bits - 1;
        int code = ((AsymmetricHashing.QuantizedVector.getBit(v.binaryVector, j) ? 1 : 0) << ex)
                + AsymmetricHashing.QuantizedVector.readExtraCode(v.extraBits, j, ex);
        return code - ((1 << ex) - 0.5f);
    }

    private static float symmetric2ScalarWords(byte[] ac, byte[] bc, int d) {
        long sum = 0;
        for (int start = 0; start < d; start += 32) {
            int active = Math.min(32, d - start);
            long a = 0, b = 0;
            if (active == 32) {
                a = (long) SYMMETRIC_WORD.get(ac, start / 4);
                b = (long) SYMMETRIC_WORD.get(bc, start / 4);
            } else {
                for (int j = 0; j < (active + 3) / 4; j++) {
                    a |= (long) (ac[start / 4 + j] & 255) << (8*j);
                    b |= (long) (bc[start / 4 + j] & 255) << (8*j);
                }
            }
            long valid = 0x5555555555555555L;
            if (active < 32) valid &= (1L << (2*active)) - 1;
            long neg = ((a ^ b) >>> 1) & valid, ma = a & valid, mb = b & valid, both = ma & mb;
            sum += active - 2*Long.bitCount(neg) + 2*(Long.bitCount(ma)+Long.bitCount(mb))
                    -4*(Long.bitCount(ma & neg)+Long.bitCount(mb & neg))
                    +4*Long.bitCount(both)-8*Long.bitCount(both & neg);
        }
        return sum * .25f;
    }

    static float codeDot(AsymmetricHashing.QuantizedVector a,
                         AsymmetricHashing.QuantizedVector b, int d, int bits) {
        if (bits == 1) {
            long mismatches = 0;
            int words = d >>> 6;
            for (int w = 0; w < words; w++) {
                mismatches += Long.bitCount(a.binaryVector[w] ^ b.binaryVector[w]);
            }
            int tail = d & 63;
            if (tail != 0) {
                mismatches += Long.bitCount((a.binaryVector[words] ^ b.binaryVector[words])
                        & ((1L << tail) - 1));
            }
            return d - 2f * mismatches;
        }
        if (bits == 2) return symmetric2ScalarWords(a.extraBits, b.extraBits, d);
        if (bits == 4) {
            // Each table entry sums products of doubled (odd integer) code values.
            // 256 entries = 512 bytes; shared across all nodes and threads.
            short[] table = DOT_4;
            int dimsPerByte = 8 / bits;
            int bytes = d / dimsPerByte;
            long sum = 0;
            for (int i = 0; i < bytes; i++) {
                int x = a.extraBits[i] & 255;
                int y = b.extraBits[i] & 255;
                sum += table[((x & 15) << 4) | (y & 15)]
                        + table[(x & 240) | (y >>> 4)];
            }
            for (int j = bytes * dimsPerByte; j < d; j++) {
                int x = (int) (2 * AsymmetricHashing.projectionComponent(a.extraBits, j, bits));
                int y = (int) (2 * AsymmetricHashing.projectionComponent(b.extraBits, j, bits));
                sum += x * y;
            }
            return sum * 0.25f;
        }
        // Generic sign+extra representation supports the remaining widths through 9 bits.
        int extra = bits - 1;
        int bias = (1 << bits) - 1;
        long sum = 0;
        for (int j = 0; j < d; j++) {
            int x = ((AsymmetricHashing.QuantizedVector.getBit(a.binaryVector, j) ? 1 : 0) << extra)
                    + AsymmetricHashing.QuantizedVector.readExtraCode(a.extraBits, j, extra);
            int y = ((AsymmetricHashing.QuantizedVector.getBit(b.binaryVector, j) ? 1 : 0) << extra)
                    + AsymmetricHashing.QuantizedVector.readExtraCode(b.extraBits, j, extra);
            sum += (long) (2 * x - bias) * (2 * y - bias);
        }
        return sum * 0.25f;
    }

    private static short[] nibbleProducts(int bits) {
        short[] table = new short[256];
        int mask = (1 << bits) - 1;
        for (int a = 0; a < 16; a++) {
            for (int b = 0; b < 16; b++) {
                int sum = 0;
                for (int shift = 0; shift < 4; shift += bits) {
                    int x = (int) (2 * AsymmetricHashing.decodeProjectionComponent((a >>> shift) & mask, bits));
                    int y = (int) (2 * AsymmetricHashing.decodeProjectionComponent((b >>> shift) & mask, bits));
                    sum += x * y;
                }
                table[(a << 4) | b] = (short) sum;
            }
        }
        return table;
    }
}
