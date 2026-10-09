/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.vector;

import java.util.Objects;

/** Portable reference and bounds contract for interleaved ASH nibble blocks. */
public final class ASHLutScoring {
    private ASHLutScoring() {}

    public static void checkBounds(byte[] codes, int offset, int groups, int stride,
                                   int lane, int count, float[] lut, float[] out, int outOffset) {
        if (groups < 0 || stride <= 0) {
            throw new IllegalArgumentException("groups must be nonnegative and stride must be positive");
        }
        Objects.checkFromIndexSize(lane, count, stride);
        Objects.checkFromIndexSize(outOffset, count, out.length);
        Objects.checkFromIndexSize(0, Math.multiplyExact(groups, 16), lut.length);
        int bytes = Math.multiplyExact(groups / 2 + groups % 2, stride);
        Objects.checkFromIndexSize(offset, bytes, codes.length);
    }

    /** Returns projection dot products only; callers apply scale, offset and landmark terms. */
    public static void score(byte[] codes, int offset, int groups, int stride,
                             int lane, int count, float[] lut, float[] out, int outOffset) {
        checkBounds(codes, offset, groups, stride, lane, count, lut, out, outOffset);
        for (int i = 0; i < count; i++) {
            float sum = 0;
            for (int group = 0; group < groups; group++) {
                int packed = codes[offset + (group >>> 1) * stride + lane + i] & 255;
                int nibble = (packed >>> ((group & 1) * 4)) & 15;
                sum += lut[group * 16 + nibble];
            }
            out[outOffset + i] = sum;
        }
    }
}
