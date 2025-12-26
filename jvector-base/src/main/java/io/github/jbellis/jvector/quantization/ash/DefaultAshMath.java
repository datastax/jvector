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

package io.github.jbellis.jvector.quantization.ash;

/**
 * Scalar reference implementation of ASH math kernels.
 */
public final class DefaultAshMath implements AshMath {

    @Override
    public void project(float[][] A, float[] x, float[] y) {
        final int rows = A.length;
        final int cols = (rows == 0 ? 0 : A[0].length);

        assert y.length == rows :
                "ASH project shape mismatch: y.length=" + y.length +
                        ", A.rows=" + rows +
                        ", A.cols=" + cols +
                        ", x.length=" + x.length;

        assert cols == x.length :
                "ASH project shape mismatch: A.cols=" + cols +
                        " != x.length=" + x.length +
                        " (A.rows=" + rows +
                        ", y.length=" + y.length + ")";

        for (int i = 0; i < rows; i++) {
            float acc = 0.0f;
            final float[] Arow = A[i];
            for (int j = 0; j < cols; j++) {
                acc += Arow[j] * x[j];
            }
            y[i] = acc;
        }
    }

    @Override
    public float maskedAdd(float[] tildeQ, long[] bits, int d) {
        assert d <= tildeQ.length : "d must be <= tildeQ.length";

        float sum = 0.0f;
        int base = 0;

        for (int w = 0; w < bits.length && base < d; w++, base += 64) {
            long word = bits[w];
            while (word != 0L) {
                int bit = Long.numberOfTrailingZeros(word);
                int idx = base + bit;
                if (idx < d) sum += tildeQ[idx];
                word &= (word - 1);
            }
        }

        return sum;
    }
}
