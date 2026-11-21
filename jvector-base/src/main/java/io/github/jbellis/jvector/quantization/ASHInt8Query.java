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
 * A projected query {@code Aq} rounded to INT8 for scoring offset-binary codes (3 and 5 to 9 bits) in graph
 * neighborhoods. With {@code s = 127 / max|Aq|}, the dot product with a code's unsigned fields {@code u} is
 * exact integer arithmetic followed by {@code (<q, u> - center) * invScale}, where
 * {@code center = (2^b - 1)/2 * sum(q)} removes the offset-binary bias. Rounding the query changes scores by
 * well under a code step; exhaustive recall@10 on 1M datasets moved by at most 0.0003.
 *
 * <p>Instances are query-owned scratch: {@link #prepare} overwrites them for the next query.</p>
 */
final class ASHInt8Query {
    final int quantizedDim;
    final int bits;
    final int units;
    /** The rounded query, {@code units * 8} entries; zero past {@code quantizedDim}. */
    final byte[] digits;
    float invScale;
    float center;

    ASHInt8Query(int quantizedDim, int bits) {
        if (!AsymmetricHashing.usesOffsetBinaryCode(bits)) {
            throw new IllegalArgumentException("INT8 neighborhood queries require offset-binary codes; got " + bits);
        }
        this.quantizedDim = quantizedDim;
        this.bits = bits;
        this.units = FusedASHLayout.fieldUnits(quantizedDim);
        this.digits = new byte[units * 8];
    }

    /** Quantizes {@code qProj[0, quantizedDim)}. A non-finite component makes every score NaN. */
    void prepare(float[] qProj) {
        float max = 0f;
        for (int j = 0; j < quantizedDim; j++) max = Math.max(max, Math.abs(qProj[j]));
        java.util.Arrays.fill(digits, (byte) 0);
        if (!Float.isFinite(max)) {
            invScale = Float.NaN;
            center = 0f;
            return;
        }
        if (max == 0f) {
            invScale = 0f;
            center = 0f;
            return;
        }
        float scale = 127f / max;
        long sum = 0;
        for (int j = 0; j < quantizedDim; j++) {
            int q = Math.max(-127, Math.min(127, Math.round(qProj[j] * scale)));
            digits[j] = (byte) q;
            sum += q;
        }
        invScale = max / 127f;
        center = (float) (((1 << bits) - 1) * 0.5 * sum);
    }

    /**
     * The dot product of the query with one lane of a field-block neighborhood block, with the native
     * kernel's arithmetic: an exact integer sum, then {@code (sum - center) * invScale}.
     */
    float laneDot(byte[] block, int blockOffset, int lane, int blockSize) {
        int b = bits;
        long dot = 0;
        for (int unit = 0; unit < units; unit++) {
            int base = blockOffset + (unit * blockSize + lane) * b;
            for (int k = 0; k < 8; k++) {
                dot += digits[unit * 8 + k] * AsymmetricHashing.offsetBinaryField(block, base, k, b);
            }
        }
        return ((float) dot - center) * invScale;
    }
}
