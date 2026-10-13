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
 * Block-oriented scorer for ASH vectors.
 * A block scorer computes scores for a contiguous range of vector ordinals
 * into a caller-provided output buffer.
 * This interface exists to support:
 *  - Blocked SIMD scoring
 *  - Query reuse across multiple neighbors
 *  - Graph-local scoring (e.g., FusedASH)
 */
public interface ASHBlockScorer {

    /** Human-readable implementation and dispatch details for benchmark diagnostics. */
    default String description() { return getClass().getSimpleName(); }

    /**
     * Score {@code count} vectors starting at {@code start}.
     * Results are written to {@code out[0..count-1]}.
     * @param start starting ordinal
     * @param count number of vectors to score
     * @param out output buffer (must have length ≥ count)
     */
    void scoreRange(int start, int count, float[] out);
}
