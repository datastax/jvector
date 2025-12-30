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

    /**
     * Score {@code count} vectors starting at {@code start}.
     * Results are written to {@code out[0..count-1]}.
     * @param start starting ordinal
     * @param count number of vectors to score
     * @param out output buffer (must have length ≥ count)
     */
    void scoreRange(int start, int count, float[] out);
}

