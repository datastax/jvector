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

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;

/**
 * Symmetric distance computation (SDC) over PQ codes: approximates the similarity between two
 * <em>encoded</em> vectors using only per-subspace codebook-entry-to-codebook-entry lookup tables,
 * with no access to the original float vectors. One table lookup per subspace per pair.
 * <p>
 * This is less accurate than asymmetric (query-to-code) scoring because both operands carry
 * quantization error, so it is intended for candidate <em>generation</em> where the survivors are
 * exactly rescored afterwards — e.g. bucketed candidate acquisition during graph compaction.
 * <p>
 * PQ codes live in centered space when the quantization was computed with a global centroid
 * {@code g}. The tables account for this per metric:
 * <ul>
 *   <li>EUCLIDEAN: distances are shift-invariant, so centered pair terms are used directly.</li>
 *   <li>DOT_PRODUCT: {@code dot(u,v) = dot(u_c,v_c) + dot(g,u_c) + dot(g,v_c) + |g|^2}; the
 *       per-code correction {@code dot(g,x_c)} is exposed via {@link #centroidDotTerm} so callers
 *       can compute it once per code rather than once per pair.</li>
 *   <li>COSINE: as DOT_PRODUCT for the numerator, with per-code approximate squared norms
 *       ({@link #normSquaredTerm}) reconstructed the same way for the denominator.</li>
 * </ul>
 * Scores are returned in the same [0,1]-oriented space as
 * {@link VectorSimilarityFunction#compare}, so they are directly comparable (approximately) with
 * exact similarities. Thread-safe after construction; all scoring state is immutable.
 */
@Experimental
public final class PQSymmetricDistanceTables {
    private final VectorSimilarityFunction vsf;
    private final int subspaceCount;
    private final int clusterCount;
    /** [m][k1 * clusterCount + k2]: squared L2 (EUCLIDEAN) or dot product (DOT_PRODUCT/COSINE) between entries. */
    private final float[][] pairTables;
    /** [m][k]: dot(g_m, entry) for centered DOT_PRODUCT/COSINE; null when not applicable. */
    private final float[][] centroidDotTables;
    /** [m][k]: |entry|^2, COSINE only; null otherwise. */
    private final float[][] normTables;
    /** |g|^2, or 0 when there is no global centroid. */
    private final float centroidNormSquared;

    public PQSymmetricDistanceTables(ProductQuantization pq, VectorSimilarityFunction vsf) {
        this.vsf = vsf;
        this.subspaceCount = pq.getSubspaceCount();
        this.clusterCount = pq.getClusterCount();
        boolean euclidean = vsf == VectorSimilarityFunction.EUCLIDEAN;
        boolean centered = pq.globalCentroid != null;

        pairTables = new float[subspaceCount][];
        centroidDotTables = (!euclidean && centered) ? new float[subspaceCount][] : null;
        normTables = (vsf == VectorSimilarityFunction.COSINE) ? new float[subspaceCount][] : null;

        for (int m = 0; m < subspaceCount; m++) {
            int subSize = pq.subvectorSizesAndOffsets[m][0];
            int subOffset = pq.subvectorSizesAndOffsets[m][1];
            var codebook = pq.codebooks[m];

            var pairs = new float[clusterCount * clusterCount];
            for (int k1 = 0; k1 < clusterCount; k1++) {
                for (int k2 = k1; k2 < clusterCount; k2++) {
                    float term = euclidean
                            ? VectorUtil.squareL2Distance(codebook, k1 * subSize, codebook, k2 * subSize, subSize)
                            : VectorUtil.dotProduct(codebook, k1 * subSize, codebook, k2 * subSize, subSize);
                    pairs[k1 * clusterCount + k2] = term;
                    pairs[k2 * clusterCount + k1] = term;
                }
            }
            pairTables[m] = pairs;

            if (centroidDotTables != null) {
                var dots = new float[clusterCount];
                for (int k = 0; k < clusterCount; k++) {
                    dots[k] = VectorUtil.dotProduct(pq.globalCentroid, subOffset, codebook, k * subSize, subSize);
                }
                centroidDotTables[m] = dots;
            }
            if (normTables != null) {
                var norms = new float[clusterCount];
                for (int k = 0; k < clusterCount; k++) {
                    norms[k] = VectorUtil.dotProduct(codebook, k * subSize, codebook, k * subSize, subSize);
                }
                normTables[m] = norms;
            }
        }
        centroidNormSquared = (!euclidean && centered)
                ? VectorUtil.dotProduct(pq.globalCentroid, pq.globalCentroid)
                : 0f;
    }

    /** Whether {@link #centroidDotTerm} contributes to scoring (centered DOT_PRODUCT/COSINE). */
    public boolean needsCentroidDot() {
        return centroidDotTables != null;
    }

    /** Whether {@link #normSquaredTerm} is required for scoring (COSINE). */
    public boolean needsNorm() {
        return normTables != null;
    }

    /**
     * Per-code correction {@code dot(g, x_c)}; 0 when not applicable. Compute once per code and
     * pass to {@link #approximateSimilarity} for every pair the code participates in.
     */
    public float centroidDotTerm(byte[] codes, int off) {
        if (centroidDotTables == null) {
            return 0f;
        }
        float sum = 0f;
        for (int m = 0; m < subspaceCount; m++) {
            sum += centroidDotTables[m][codes[off + m] & 0xFF];
        }
        return sum;
    }

    /**
     * Approximate squared norm {@code |x|^2} of the decoded vector including the global centroid,
     * reconstructed as {@code |x_c|^2 + 2*dot(g,x_c) + |g|^2}. COSINE only.
     *
     * @param centroidDot the value previously returned by {@link #centroidDotTerm} for this code
     */
    public float normSquaredTerm(byte[] codes, int off, float centroidDot) {
        float sum = 0f;
        for (int m = 0; m < subspaceCount; m++) {
            sum += normTables[m][codes[off + m] & 0xFF];
        }
        return sum + 2 * centroidDot + centroidNormSquared;
    }

    /**
     * Approximate similarity between two PQ codes, oriented like
     * {@link VectorSimilarityFunction#compare} (higher = closer).
     *
     * @param gA    {@link #centroidDotTerm} of code A (ignored for EUCLIDEAN; pass 0)
     * @param gB    {@link #centroidDotTerm} of code B
     * @param normA {@link #normSquaredTerm} of code A (ignored except for COSINE; pass 0)
     * @param normB {@link #normSquaredTerm} of code B
     */
    public float approximateSimilarity(byte[] codesA, int offA, byte[] codesB, int offB,
                                       float gA, float gB, float normA, float normB) {
        float pairSum = 0f;
        for (int m = 0; m < subspaceCount; m++) {
            pairSum += pairTables[m][(codesA[offA + m] & 0xFF) * clusterCount + (codesB[offB + m] & 0xFF)];
        }
        switch (vsf) {
            case EUCLIDEAN:
                return 1f / (1f + pairSum);
            case DOT_PRODUCT:
                return (1f + pairSum + gA + gB + centroidNormSquared) / 2;
            case COSINE:
                float raw = pairSum + gA + gB + centroidNormSquared;
                double denom = Math.sqrt((double) normA * normB);
                return denom <= 0 ? 0f : (float) ((1 + raw / denom) / 2);
            default:
                throw new IllegalStateException("Unsupported similarity function " + vsf);
        }
    }

    /** Approximate table memory footprint in bytes; useful for logging by callers. */
    public long tableBytes() {
        long entries = (long) subspaceCount * clusterCount * clusterCount;
        if (centroidDotTables != null) {
            entries += (long) subspaceCount * clusterCount;
        }
        if (normTables != null) {
            entries += (long) subspaceCount * clusterCount;
        }
        return entries * Float.BYTES;
    }
}
