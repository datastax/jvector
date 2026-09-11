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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.vector.ASHLutScoring;
import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import java.util.Locale;

/** Query-lifetime dispatch, shared by standalone and fused nibble-block scoring. */
final class ASHLutKernel {
    private final VectorUtilSupport backend = VectorizationProvider.getInstance().getVectorUtilSupport();
    private final boolean simd;

    ASHLutKernel() {
        String mode = System.getProperty("jvector.ash.blockKernel", "auto").trim().toLowerCase(Locale.ROOT);
        switch (mode) {
            case "scalar": simd = false; break;
            case "auto": simd = backend.supportsAshLutScoring(); break;
            case "simd":
                if (!backend.supportsAshLutScoring()) {
                    throw new IllegalStateException("ASH SIMD LUT scoring requested, but backend "
                            + backend.getClass().getName() + " does not support it");
                }
                simd = true;
                break;
            default: throw new IllegalArgumentException("jvector.ash.blockKernel must be auto, scalar or simd: " + mode);
        }
    }

    void score(byte[] codes, int offset, int groups, int stride, int lane, int count,
               float[] lut, float[] out, int outOffset) {
        if (simd) backend.ashLutScore(codes, offset, groups, stride, lane, count, lut, out, outOffset);
        else ASHLutScoring.score(codes, offset, groups, stride, lane, count, lut, out, outOffset);
    }

    @Override
    public String toString() { return simd ? "SIMD ASH LUT (" + backend.getClass().getSimpleName() + ")" : "scalar ASH LUT"; }
}
