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
 * ASH math kernels.
 *
 * This interface forms the SIMD seam for ASH-specific operations.
 * Implementations may be scalar, Panama Vector API, or native.
 */
public interface AshMath {

    /**
     * Compute y = A · x, where:
     *  - A is [d][D]
     *  - x is [D]
     *  - y is [d]
     */
    void project(float[][] A, float[] x, float[] y);

    /**
     * Compute masked-add ⟨tildeQ, b⟩ where b ∈ {0,1}^d stored as bitpacked longs.
     * Implementations must interpret bits with the same convention as existing ASH code:
     * bit j corresponds to dimension j (little-endian within each long).
     *
     * @param tildeQ projected query (length d)
     * @param bits bitpacked database code
     * @param d number of valid dimensions in tildeQ
     * @return sum_{j: b_j=1} tildeQ[j]
     */
    float maskedAdd(float[] tildeQ, long[] bits, int d);
}
