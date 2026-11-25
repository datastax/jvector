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
import io.github.jbellis.jvector.vector.VectorizationProvider;

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Query-time raw dot-product scorer for ASH vectors, with scalar and SIMD paths.
 * Graph-facing scorers convert its estimates through the model's {@code ASHScoreScale}; {@link #toSimilarity(float)}
 * is the DOT_PRODUCT and COSINE scale.
 *
 * <p>
 * The 1-bit path preserves the original ASH decomposition:
 * </p>
 *
 * <pre>
 *   score = scale * (2 * &lt;A(q - mu), b&gt; - &lt;A(q - mu), 1&gt;)
 *         + &lt;q, mu&gt;
 *         + offset
 * </pre>
 *
 * <p>
 * Fast-scan projection codes, currently bitsPerDimension in {2,4}, use the
 * C++ projection-mode decomposition.  During encoding, {@code offset} already
 * includes {@code -scale * &lt;A(mu), code&gt;}, so scoring uses only {@code A(q)}:
 * </p>
 *
 * <pre>
 *   score = scale * &lt;A(q), code&gt; + &lt;q, mu&gt; + offset
 * </pre>
 *
 * <p>
 * Other multibit widths keep the generic sign+extra-bit reference scorer used
 * for standalone ASH and future ASH reranking experiments.
 * </p>
 *
 * <p>
 * Every similarity uses this decomposition. A EUCLIDEAN model stores {@code -||x - mu||^2 / 2} in the offset's
 * place and uses {@code -||q - mu||^2 / 2} for {@code <q, mu>}, so the same kernels estimate
 * {@code -||q - x||^2 / 2}; see {@code ASHEuclidean}.
 * </p>
 */
@Experimental
public final class ASHScorer {
    /**
     * Converts a raw ASH dot-product estimate to the graph's DOT_PRODUCT similarity scale, (1 + dot) / 2, the same
     * scale exact DOT_PRODUCT scoring uses. The estimate is not clamped: like an exact dot product of vectors that are
     * not unit length, it can fall below -1 or above 1, and ranking needs those values. Graph search starts hierarchy
     * descent from negative infinity, so negative similarities do not lose the entry point. Kernels and this class's
     * ASHScoreFunction retain raw dot products; node and block scorer adapters must apply this conversion exactly once.
     */
    public static float toSimilarity(float dotProduct) {
        return (1f + dotProduct) / 2f;
    }

    private final AsymmetricHashing ash;

    public ASHScorer(AsymmetricHashing ash) {
        this.ash = ash;

        final int d = ash.quantizedDim;
        final int C = ash.landmarkCount;

        assert ash.landmarkProj != null;
        assert ash.landmarkProjSum != null;
        assert ash.landmarkProj.length == C :
                "landmarkProj length mismatch";
        assert ash.landmarkProjSum.length == C :
                "landmarkProjSum length mismatch";

        for (int c = 0; c < C; c++) {
            assert ash.landmarkProj[c].length == d :
                    "landmarkProj[" + c + "] length != quantizedDim";
        }

        assert ash.stiefelTransform.rows == ash.quantizedDim
                : "Stiefel rows must equal quantizedDim";
    }

    /**
     * The raw kernel estimate for {@code query}: the dot product for a DOT_PRODUCT model, and
     * {@code -||q - x||^2 / 2} in model units for a EUCLIDEAN one. {@link ASHVectors} scores similarities.
     *
     * @throws IllegalArgumentException unless {@code similarityFunction} is the model's
     */
    public ASHScoreFunction scoreFunctionFor(
            VectorFloat<?> query,
            VectorSimilarityFunction similarityFunction) {
        ash.checkSimilarity(similarityFunction);
        return scoreFunctionFor(precomputeQuery(query));
    }

    /**
     * Scorer for dot product approximation.
     */
    ASHScoreFunction scoreFunctionFor(QueryPrecompute qp) {
        return forVectors(codeScorerFor(qp));
    }

    /** As {@link #scoreFunctionFor(QueryPrecompute)}, scoring codes in place. */
    CodeScorer codeScorerFor(QueryPrecompute qp) {
        if (ash.bitsPerDimension == 1) {
            return oneBitDotProductScorer(qp);
        }

        if (AsymmetricHashing.usesFastScanProjectionCode(ash.bitsPerDimension)) {
            return fastScanProjectionDotProductScorer(qp);
        }
        var mode = ASHVectors.AshSingleKernel.fromProperty();
        var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        boolean simd = backend.supportsAshCanonicalLutScoring() || backend.supportsAshOffsetBinaryDot();
        if (mode == ASHVectors.AshSingleKernel.SIMD && !simd) {
            throw new IllegalStateException("ASH single-vector SIMD requested, but backend "
                    + backend.getClass().getName() + " does not support offset-binary scoring");
        }
        return genericMultibitDotProductScorer(qp, mode != ASHVectors.AshSingleKernel.SCALAR && simd);
    }

    /** Reuses a fused scorer's table for direct nodes when the backend supports it. */
    ASHScoreFunction scoreFunctionFor(QueryPrecompute qp, float[] lut) {
        var mode = ASHVectors.AshSingleKernel.fromProperty();
        var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        // One-bit codes need no table: the masked-add kernel non-fused search uses.
        if (ash.bitsPerDimension == 1
                && mode != ASHVectors.AshSingleKernel.SCALAR
                && backend.supportsAshMaskedLoad()) {
            final int d = qp.d;
            final int words = AsymmetricHashing.QuantizedVector.wordsForDims(d);
            return v -> v.scale * (2f * backend.ashMaskedAdd_512(qp.qProj, 0, v.binaryVector, 0, d, words)
                    - qp.sumQProj) + qp.dotQMuByLandmark[v.landmark & 255] + v.offset;
        }
        if (ash.bitsPerDimension == 2
                && mode != ASHVectors.AshSingleKernel.SCALAR
                && backend.supportsNativeAshTwoBitProjectionDot()) {
            final int d = qp.d;
            return v -> v.scale * backend.ashProjectionDot(qp.qProj, v.extraBits, 0, d, 2)
                    + qp.dotQMuByLandmark[v.landmark & 255] + v.offset;
        }
        // Two coordinates share each 2-bit LUT entry. Four-bit codes instead
        // use the backend's contiguous projection dot, avoiding a 16x larger table.
        // Offset-binary codes decode faster in registers than through the LUT, where the backend can.
        if (directScoringReadsLut(ash.bitsPerDimension, mode, backend)) {
            int groups = FusedASHLayout.codeGroups(qp.d, ash.bitsPerDimension);
            return v -> v.scale * backend.ashCanonicalLutDot(v.extraBits, groups, lut)
                    + qp.dotQMuByLandmark[v.landmark & 255] + v.offset;
        }
        return scoreFunctionFor(qp);
    }

    /** Whether {@link #scoreFunctionFor(QueryPrecompute, float[])} scores direct nodes through the query table. */
    private static boolean directScoringReadsLut(int bits, ASHVectors.AshSingleKernel mode, VectorUtilSupport backend) {
        return (bits == 2 || (AsymmetricHashing.usesOffsetBinaryCode(bits) && !backend.supportsAshOffsetBinaryDot()))
                && mode != ASHVectors.AshSingleKernel.SCALAR && backend.supportsAshCanonicalLutScoring();
    }

    /**
     * Whether anything reads a query's nibble table at this width: fused neighborhoods at 1, 2 and 4 bits, and
     * direct-node scoring of offset-binary codes (3 and 5 to 9 bits) on a backend that has no other SIMD kernel.
     */
    static boolean readsQueryLut(int bitsPerDimension) {
        return !AsymmetricHashing.usesOffsetBinaryCode(bitsPerDimension)
                || directScoringReadsLut(bitsPerDimension, ASHVectors.AshSingleKernel.fromProperty(),
                        VectorizationProvider.getInstance().getVectorUtilSupport());
    }

    /** Prepared encoded-query state; uses the same arithmetic as the raw-query scorer. */
    ASHScoreFunction scoreFunctionFor(QueryPrecompute qp, boolean simd) {
        return forVectors(codeScorerFor(qp, simd));
    }

    /** As {@link #scoreFunctionFor(QueryPrecompute, boolean)}, scoring codes in place. */
    CodeScorer codeScorerFor(QueryPrecompute qp, boolean simd) {
        if (ash.bitsPerDimension == 1) return oneBitDotProductScorer(qp);
        if (AsymmetricHashing.usesFastScanProjectionCode(ash.bitsPerDimension)) {
            return fastScanProjectionDotProductScorer(qp, simd);
        }
        return genericMultibitDotProductScorer(qp, simd);
    }

    /** Adapts a code scorer to standalone vectors, whose arrays start at offset zero. */
    static ASHScoreFunction forVectors(CodeScorer scorer) {
        return v -> scorer.dot(v.binaryVector, 0, v.extraBits, 0, v.scale, v.offset, v.landmark & 0xFF);
    }

    private CodeScorer oneBitDotProductScorer(QueryPrecompute qp) {
        final int d = qp.d;
        final int C = qp.C;
        final int words = AsymmetricHashing.QuantizedVector.wordsForDims(d);

        return (bits, bitsOffset, body, bodyOffset, scale, offset, c) -> {
            assert c < C : "Invalid landmark id " + c + " for landmarkCount=" + C;

            // maskedAdd = <Aq, b>, where b ∈ {0,1}^d is stored as packed longs.
            float maskedAdd = 0f;

            int dimBase = 0;
            for (int w = 0; w < words && dimBase < d; w++, dimBase += 64) {
                long word = bits[bitsOffset + w];
                while (word != 0L) {
                    int bit = Long.numberOfTrailingZeros(word);
                    int idx = dimBase + bit;
                    if (idx < d) {
                        maskedAdd += qp.qProj[idx];
                    }
                    word &= (word - 1);
                }
            }

            // 1-bit ASH dot-product approximation, with <Aμ_c, s> folded into the offset:
            //   scale * <Aq, s> + <q,μ_c> + offset, where <Aq, s> = 2<Aq,b> - <Aq,1>
            return scale * (2f * maskedAdd - qp.sumQProj)
                    + qp.dotQMuByLandmark[c]
                    + offset;
        };
    }

    /** Independent scalar oracle for the reference block-scoring API. */
    ASHScoreFunction scalarScoreFunctionFor(VectorFloat<?> query) {
        return forVectors(scalarCodeScorerFor(query));
    }

    /** As {@link #scalarScoreFunctionFor}, scoring codes in place. */
    CodeScorer scalarCodeScorerFor(VectorFloat<?> query) {
        QueryPrecompute qp = precomputeQuery(query);
        if (ash.bitsPerDimension == 1) return oneBitDotProductScorer(qp);
        if (AsymmetricHashing.usesFastScanProjectionCode(ash.bitsPerDimension)) {
            return fastScanProjectionDotProductScorer(qp, false);
        }
        return genericMultibitDotProductScorer(qp, false);
    }

    private CodeScorer fastScanProjectionDotProductScorer(QueryPrecompute qp) {
        final var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        final var mode = ASHVectors.AshSingleKernel.fromProperty();
        if (mode == ASHVectors.AshSingleKernel.SIMD && !backend.supportsAshProjectionScoring()) {
            throw new IllegalStateException("ASH single-vector SIMD requested, but backend "
                    + backend.getClass().getName() + " does not support projection scoring");
        }
        final boolean simd = mode != ASHVectors.AshSingleKernel.SCALAR && backend.supportsAshProjectionScoring();
        return fastScanProjectionDotProductScorer(qp, simd);
    }

    private CodeScorer fastScanProjectionDotProductScorer(QueryPrecompute qp, boolean simd) {
        final int d = qp.d;
        final int C = qp.C;
        final int bitsPerDimension = ash.bitsPerDimension;
        final var backend = VectorizationProvider.getInstance().getVectorUtilSupport();

        if (simd && "native".equalsIgnoreCase(
                System.getProperty("jvector.vectorization_provider", "").trim())) {
            backend.requireNativeAshSingleScoring(bitsPerDimension);
        }
        // A native 2-bit dot decodes codes in registers instead of gathering from a query LUT.
        if (simd && bitsPerDimension == 2 && backend.supportsNativeAshTwoBitProjectionDot()) {
            return (bits, bitsOffset, code, codeOffset, scale, offset, c) ->
                    scale * backend.ashProjectionDot(qp.qProj, code, codeOffset, d, bitsPerDimension)
                    + qp.dotQMuByLandmark[c] + offset;
        }
        // The native two-bit LUT kernel consumes a nibble LUT, prepared once per scorer.
        // Keep it query-owned: prepared scorers may outlive this thread or run concurrently.
        if (simd && bitsPerDimension == 2 && backend.supportsNativeAshCanonicalLutScoring()) {
            final int groups = FusedASHLayout.codeGroups(d, bitsPerDimension);
            final float[] lut = new float[Math.multiplyExact(groups, 16)];
            FusedASHLayout.buildQueryLut(qp.qProj, d, bitsPerDimension, lut);
            return (bits, bitsOffset, code, codeOffset, scale, offset, c) ->
                    scale * backend.ashCanonicalLutDot(code, codeOffset, groups, lut)
                    + qp.dotQMuByLandmark[c] + offset;
        }

        // Keep tuning dispatch out of the per-vector default kernel. Its compact
        // compilation shape matters for end-to-end scoring throughput on HotSpot.
        if (simd && backend.usesAshProjectionTuning()) {
            return (bits, bitsOffset, code, codeOffset, scale, offset, c) ->
                    scale * backend.ashProjectionDotTuned(qp.qProj, code, codeOffset, d, bitsPerDimension)
                    + qp.dotQMuByLandmark[c] + offset;
        }

        // A backend may rearrange the query once so each code slot's query values load contiguously.
        final float[] prepared = simd ? backend.ashPrepareProjectionQuery(qp.qProj, d, bitsPerDimension) : null;
        if (prepared != null) {
            return (bits, bitsOffset, code, codeOffset, scale, offset, c) ->
                    scale * backend.ashProjectionDotPrepared(prepared, code, codeOffset, d, bitsPerDimension)
                    + qp.dotQMuByLandmark[c] + offset;
        }

        return (bits, bitsOffset, code, codeOffset, scale, offset, c) -> {
            assert c < C : "Invalid landmark id " + c + " for landmarkCount=" + C;

            // C++ projection-mode scoring:
            //   score = scale * <Aq, code> + <q, μ_c> + stored_offset
            // stored_offset already includes -scale * <Aμ_c, code>.
            float ip = simd ? backend.ashProjectionDot(qp.qProj, code, codeOffset, d, bitsPerDimension)
                    : AsymmetricHashing.dotProjectionCode(qp.qProj, code, codeOffset, d, bitsPerDimension);

            return scale * ip + qp.dotQMuByLandmark[c] + offset;
        };
    }

    /**
     * Offset-binary codes (3 and 5 to 9 bits): {@code scale * <Aq, code> + <q, mu_c> + offset}. The SIMD path
     * decodes fields in registers where the backend can, or else sums the code's 4-bit groups through a query LUT
     * prepared once per scorer; the scalar path decodes each field and is the independent reference.
     */
    private CodeScorer genericMultibitDotProductScorer(QueryPrecompute qp, boolean simd) {
        final int d = qp.d;
        final int C = qp.C;
        final int bitsPerDimension = ash.bitsPerDimension;
        final var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        if (simd && backend.supportsAshOffsetBinaryDot()) {
            return (signBits, signOffset, code, codeOffset, scale, offset, c) ->
                    scale * backend.ashOffsetBinaryDot(qp.qProj, code, codeOffset, d, bitsPerDimension)
                    + qp.dotQMuByLandmark[c] + offset;
        }
        if (simd) {
            final int groups = FusedASHLayout.codeGroups(d, bitsPerDimension);
            // Query-owned: prepared scorers may outlive this thread or run concurrently.
            final float[] lut = new float[Math.multiplyExact(groups, 16)];
            FusedASHLayout.buildQueryLut(qp.qProj, d, bitsPerDimension, lut);
            return (signBits, signOffset, code, codeOffset, scale, offset, c) ->
                    scale * backend.ashCanonicalLutDot(code, codeOffset, groups, lut) + qp.dotQMuByLandmark[c] + offset;
        }
        return (signBits, signOffset, code, codeOffset, scale, offset, c) -> {
            assert c < C : "Invalid landmark id " + c + " for landmarkCount=" + C;
            return scale * AsymmetricHashing.dotOffsetBinaryCode(qp.qProj, code, codeOffset, d, bitsPerDimension)
                    + qp.dotQMuByLandmark[c] + offset;
        };
    }

    /** Allocates the preparation terms of one query: {@code Aq} and the landmark dots. */
    QueryPrecompute newQueryPrecompute() {
        return new QueryPrecompute(ash.quantizedDim, ash.landmarkCount, new float[ash.quantizedDim],
                new float[ash.landmarkCount]);
    }

    /** Independently owned preparation for retained standalone scorers. */
    QueryPrecompute precomputeQuery(VectorFloat<?> query) {
        validateQuery(query);
        QueryPrecompute prepared = newQueryPrecompute();
        prepareQuery(query, new float[ash.originalDimension], prepared);
        return prepared;
    }

    void validateQuery(VectorFloat<?> query) {
        java.util.Objects.requireNonNull(query, "query");
        if (query.length() != ash.originalDimension) {
            throw new IllegalArgumentException("ASH query dimension must be " + ash.originalDimension);
        }
    }

    /** Refills model-owned scratch; callers must enforce its query lifetime. */
    void prepareQuery(VectorFloat<?> query, float[] input, QueryPrecompute prepared) {
        validateQuery(query);
        // Scratch is constructed with this model and never resized on the hot path.
        VectorFloat<?> scoring = ash.loadQuery(query, input, null);
        ash.stiefelTransform.projectQuery(input, prepared.qProj);
        finishQuery(scoring, prepared);
    }

    /** Completes the non-projection terms after direct or batched query projection; {@code query} is the scoring query. */
    void finishQuery(VectorFloat<?> query, QueryPrecompute prepared) {
        for (int c = 0; c < ash.landmarkCount; c++) {
            prepared.dotQMuByLandmark[c] = ash.queryLandmarkTerm(query, c);
        }
        // Stored offsets include the landmark term, so every scorer needs only Aq and, for 1-bit
        // codes, its sum.
        prepared.sumQProj = sum(prepared.qProj);
        prepared.queryNorm = (float) Math.sqrt(VectorUtil.dotProduct(query, query));
    }

    /**
     * An upper bound on the dot products ASH estimates for {@code prepared}'s query: its length times the largest
     * encoded vector length. A query prepared from a stored code, whose length is not kept, is bounded by the
     * largest length. Estimates above it raise adaptive stopping's bound, which only loosens the margin.
     */
    float dotProductBound(QueryPrecompute prepared) {
        float m = ash.maxEncodedNorm();
        return (Float.isNaN(prepared.queryNorm) ? m : prepared.queryNorm) * m;
    }

    /** Left-to-right FP32 sum, the order every 1-bit scorer uses for {@code <Aq, 1>}. */
    static float sum(float[] values) {
        float total = 0f;
        for (float value : values) total += value;
        return total;
    }

    static final class QueryPrecompute {
        final int d;
        final int C;
        final float[] qProj;                   // [d], Aq
        final float[] dotQMuByLandmark;        // [C]
        float sumQProj;                        // <Aq, 1>, set with qProj
        float queryNorm = Float.NaN;           // ||q||, when prepared from a raw query

        QueryPrecompute(int d, int C, float[] qProj, float[] dotQMuByLandmark) {
            this.d = d;
            this.C = C;
            this.qProj = qProj;
            this.dotQMuByLandmark = dotQMuByLandmark;
        }
    }

    @FunctionalInterface
    public interface ASHScoreFunction {
        float similarityTo(AsymmetricHashing.QuantizedVector vector2);
    }

    /**
     * Raw dot product of a prepared query with one code read in place: sign words from {@code bits}
     * at {@code bitsOffset} and the byte body from {@code body} at {@code bodyOffset} (either is
     * unused when the bit width stores none), with the code's decoded header and unsigned landmark.
     * Lets {@link ASHVectors} score its compact pages and standalone vectors with one formula.
     */
    @FunctionalInterface
    interface CodeScorer {
        float dot(long[] bits, int bitsOffset, byte[] body, int bodyOffset, float scale, float offset, int landmark);
    }
}
