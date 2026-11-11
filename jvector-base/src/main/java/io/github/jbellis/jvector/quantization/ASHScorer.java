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

import io.github.jbellis.jvector.vector.VectorizationProvider;

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Query-time raw dot-product scorer for ASH vectors, with scalar and SIMD paths.
 * Graph-facing scorers convert its estimates using {@link #toSimilarity(float)}.
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
 * NOTE: Only DOT_PRODUCT is supported for now.
 */
public final class ASHScorer {
    /**
     * Converts a raw ASH dot-product estimate to the graph's DOT_PRODUCT similarity
     * scale. Graph search uses zero as its minimum similarity, including during
     * hierarchy descent; passing a negative raw dot product can lose the entry point.
     * Clamp the lower tail because quantization can estimate a dot product below -1.
     * Kernels and this class's ASHScoreFunction retain raw dot products; node and
     * block scorer adapters must apply this conversion exactly once.
     */
    public static float toSimilarity(float dotProduct) {
        return Math.max(0f, (1f + dotProduct) / 2f);
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

    public ASHScoreFunction scoreFunctionFor(
            VectorFloat<?> query,
            VectorSimilarityFunction similarityFunction) {
        if (similarityFunction != VectorSimilarityFunction.DOT_PRODUCT) {
            throw new UnsupportedOperationException(
                    "ASH scorer supports DOT_PRODUCT only (requested " + similarityFunction + ")"
            );
        }
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
        if (mode == ASHVectors.AshSingleKernel.SIMD && !backend.supportsAshCanonicalLutScoring()) {
            throw new IllegalStateException("ASH single-vector SIMD requested, but backend "
                    + backend.getClass().getName() + " does not support canonical LUT scoring");
        }
        return genericMultibitDotProductScorer(qp,
                mode != ASHVectors.AshSingleKernel.SCALAR && backend.supportsAshCanonicalLutScoring());
    }

    /** Reuses a fused scorer's table for direct nodes when the backend supports it. */
    ASHScoreFunction scoreFunctionFor(QueryPrecompute qp, float[] lut) {
        var mode = ASHVectors.AshSingleKernel.fromProperty();
        var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        // Two coordinates share each 2-bit LUT entry. Four-bit codes instead
        // use the backend's contiguous projection dot, avoiding a 16x larger table.
        if ((ash.bitsPerDimension == 2 || AsymmetricHashing.usesOffsetBinaryCode(ash.bitsPerDimension))
                && mode != ASHVectors.AshSingleKernel.SCALAR
                && backend.supportsAshCanonicalLutScoring()) {
            int groups = FusedASHLayout.codeGroups(qp.d, ash.bitsPerDimension);
            return v -> v.scale * backend.ashCanonicalLutDot(v.extraBits, groups, lut)
                    + qp.dotQMuByLandmark[v.landmark & 255] + v.offset;
        }
        return scoreFunctionFor(qp);
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
        // The native two-bit kernel consumes a nibble LUT, prepared once per scorer.
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
     * Offset-binary codes (3 and 5 to 9 bits): {@code scale * <Aq, code> + <q, mu_c> + offset}. The SIMD
     * path sums the code's 4-bit groups through a query LUT prepared once per scorer; the scalar path
     * decodes each field and is the independent reference.
     */
    private CodeScorer genericMultibitDotProductScorer(QueryPrecompute qp, boolean simd) {
        final int d = qp.d;
        final int C = qp.C;
        final int bitsPerDimension = ash.bitsPerDimension;
        if (simd) {
            final var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
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
        for (int k = 0; k < input.length; k++) input[k] = query.get(k);
        ash.stiefelTransform.projectQuery(input, prepared.qProj);
        finishQuery(query, prepared);
    }

    /** Completes the non-projection terms after direct or batched query projection. */
    void finishQuery(VectorFloat<?> query, QueryPrecompute prepared) {
        for (int c = 0; c < ash.landmarkCount; c++) {
            prepared.dotQMuByLandmark[c] = VectorUtil.dotProduct(query, ash.landmarks[c]);
        }
        // Stored offsets include the landmark term, so every scorer needs only Aq and, for 1-bit
        // codes, its sum.
        prepared.sumQProj = sum(prepared.qProj);
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
