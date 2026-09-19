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
 * Compressed-to-compressed ASH dot products for C=1 (paper Appendix B, B.2-B.4).
 * The codes remain packed; no original vectors or per-pair projections are needed.
 * Instances require immutable encoded vectors, as does ASHVectors itself.
 */
public final class ASHSymmetricScorer {
    private static final short[] DOT_2 = nibbleProducts(2);
    private static final short[] DOT_4 = nibbleProducts(4);
    private final ASHVectors vectors;
    private final int dimensions;
    private final int bits;
    private final float muNormSquared;
    // Recover <x,mu>-||mu||^2 once from the projection-mode query offset.
    // Four bytes per vector, only for 2/4-bit construction; never serialized.
    private final float[] centeredOffsets;

    public ASHSymmetricScorer(ASHVectors vectors) {
        var ash = vectors.getCompressor();
        if (ash.landmarkCount != 1) {
            throw new IllegalArgumentException("Symmetric ASH scoring requires landmarkCount=1");
        }
        this.vectors = vectors;
        dimensions = ash.quantizedDim;
        bits = ash.bitsPerDimension;
        muNormSquared = VectorUtil.dotProduct(ash.landmarks[0], ash.landmarks[0]);
        if (AsymmetricHashing.usesFastScanProjectionCode(bits)) {
            centeredOffsets = new float[vectors.count()];
            for (int i = 0; i < centeredOffsets.length; i++) {
                var v = vectors.get(i);
                centeredOffsets[i] = v.offset + v.scale * AsymmetricHashing.dotProjectionCode(
                        ash.landmarkProj[0], v.extraBits, dimensions, bits);
            }
        } else {
            centeredOffsets = null;
        }
    }

    private float offset(int node) {
        return centeredOffsets == null ? vectors.get(node).offset : centeredOffsets[node];
    }

    /** Raw dot product, for diagnostics and independent comparison with input-space ground truth. */
    public float dotProduct(int a, int b) {
        var x = vectors.get(a);
        var y = vectors.get(b);
        // Sum offsets first to preserve exactly the same float result in both directions.
        return (x.scale * y.scale) * codeDot(x, y, dimensions, bits)
                + (offset(a) + offset(b)) + muNormSquared;
    }

    /** Graph-facing similarity, using the same normalization as asymmetric ASH. */
    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(int node) {
        return other -> ASHScorer.toSimilarity(dotProduct(node, other));
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
        if (bits == 2 || bits == 4) {
            // Each table entry sums products of doubled (odd integer) code values.
            // 256 entries = 512 bytes; shared across all nodes and threads.
            short[] table = bits == 2 ? DOT_2 : DOT_4;
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
