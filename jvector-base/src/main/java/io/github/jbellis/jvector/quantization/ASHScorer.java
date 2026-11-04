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
        if (ash.bitsPerDimension == 1) {
            return oneBitDotProductScoreFunction(qp);
        }

        if (AsymmetricHashing.usesFastScanProjectionCode(ash.bitsPerDimension)) {
            return fastScanProjectionDotProductScoreFunction(qp);
        }

        if (ASHVectors.AshSingleKernel.fromProperty() == ASHVectors.AshSingleKernel.SIMD) {
            throw new UnsupportedOperationException("SIMD ASH projection scoring supports 2 and 4 bits per dimension");
        }
        return genericMultibitDotProductScoreFunction(qp);
    }

    /** Reuses a fused scorer's table for direct nodes when the backend supports it. */
    ASHScoreFunction scoreFunctionFor(QueryPrecompute qp, float[] lut) {
        var mode = ASHVectors.AshSingleKernel.fromProperty();
        var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        // Two coordinates share each 2-bit LUT entry. Four-bit codes instead
        // use the backend's contiguous projection dot, avoiding a 16x larger table.
        if (ash.bitsPerDimension == 2
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
        if (ash.bitsPerDimension == 1) return oneBitDotProductScoreFunction(qp);
        if (AsymmetricHashing.usesFastScanProjectionCode(ash.bitsPerDimension)) {
            return fastScanProjectionDotProductScoreFunction(qp, simd);
        }
        if (simd) throw new UnsupportedOperationException("SIMD ASH requires bits 1, 2, or 4");
        return genericMultibitDotProductScoreFunction(qp);
    }

    private ASHScoreFunction oneBitDotProductScoreFunction(QueryPrecompute qp) {
        final int d = qp.d;
        final int C = qp.C;
        final int words = AsymmetricHashing.QuantizedVector.wordsForDims(d);

        return (AsymmetricHashing.QuantizedVector v) -> {
            final int c = v.landmark & 0xFF; // unsigned [0,C)
            assert c < C : "Invalid landmark id " + c + " for landmarkCount=" + C;

            // maskedAdd = <q̃_c, b>, where b ∈ {0,1}^d is stored as packed longs.
            float maskedAdd = 0f;

            final int tildeBase = c * d;
            final long[] bits = v.binaryVector;
            int dimBase = 0;
            for (int w = 0; w < words && dimBase < d; w++, dimBase += 64) {
                long word = bits[w];
                while (word != 0L) {
                    int bit = Long.numberOfTrailingZeros(word);
                    int idx = dimBase + bit;
                    if (idx < d) {
                        maskedAdd += qp.tildeQPool[tildeBase + idx];
                    }
                    word &= (word - 1);
                }
            }

            // 1-bit ASH dot-product approximation:
            //   scale * (2<q̃_c,b> - <q̃_c,1>) + <q,μ_c> + offset
            return v.scale * (2f * maskedAdd - qp.sumTildeQByLandmark[c])
                    + qp.dotQMuByLandmark[c]
                    + v.offset;
        };
    }

    /** Independent scalar oracle for the reference block-scoring API. */
    ASHScoreFunction scalarScoreFunctionFor(VectorFloat<?> query) {
        QueryPrecompute qp = precomputeQuery(query);
        if (ash.bitsPerDimension == 1) return oneBitDotProductScoreFunction(qp);
        if (AsymmetricHashing.usesFastScanProjectionCode(ash.bitsPerDimension)) {
            return fastScanProjectionDotProductScoreFunction(qp, false);
        }
        return genericMultibitDotProductScoreFunction(qp);
    }

    private ASHScoreFunction fastScanProjectionDotProductScoreFunction(QueryPrecompute qp) {
        final var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        final var mode = ASHVectors.AshSingleKernel.fromProperty();
        if (mode == ASHVectors.AshSingleKernel.SIMD && !backend.supportsAshProjectionScoring()) {
            throw new IllegalStateException("ASH single-vector SIMD requested, but backend "
                    + backend.getClass().getName() + " does not support projection scoring");
        }
        final boolean simd = mode != ASHVectors.AshSingleKernel.SCALAR && backend.supportsAshProjectionScoring();
        return fastScanProjectionDotProductScoreFunction(qp, simd);
    }

    private ASHScoreFunction fastScanProjectionDotProductScoreFunction(QueryPrecompute qp, boolean simd) {
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
            return v -> v.scale * backend.ashCanonicalLutDot(v.extraBits, groups, lut)
                    + qp.dotQMuByLandmark[v.landmark & 255] + v.offset;
        }

        // Keep tuning dispatch out of the per-vector default kernel. Its compact
        // compilation shape matters for end-to-end scoring throughput on HotSpot.
        if (simd && backend.usesAshProjectionTuning()) {
            return v -> v.scale * backend.ashProjectionDotTuned(qp.qProj, v.extraBits, d, bitsPerDimension)
                    + qp.dotQMuByLandmark[v.landmark & 255] + v.offset;
        }

        return (AsymmetricHashing.QuantizedVector v) -> {
            final int c = v.landmark & 0xFF; // unsigned [0,C)
            assert c < C : "Invalid landmark id " + c + " for landmarkCount=" + C;

            // C++ projection-mode scoring:
            //   score = scale * <Aq, code> + <q, μ_c> + stored_offset
            // stored_offset already includes -scale * <Aμ_c, code>.
            float ip = simd ? backend.ashProjectionDot(qp.qProj, v.extraBits, d, bitsPerDimension)
                    : AsymmetricHashing.dotProjectionCode(
                    qp.qProj,
                    v.extraBits,
                    d,
                    bitsPerDimension);

            return v.scale * ip + qp.dotQMuByLandmark[c] + v.offset;
        };
    }

    private ASHScoreFunction genericMultibitDotProductScoreFunction(QueryPrecompute qp) {
        final int d = qp.d;
        final int C = qp.C;
        final int bitsPerDimension = ash.bitsPerDimension;
        final int exBits = bitsPerDimension - 1;
        final int signMagnitudeOffset = 1 << exBits;
        final float centerBias = -(signMagnitudeOffset - 0.5f);

        return (AsymmetricHashing.QuantizedVector v) -> {
            final int c = v.landmark & 0xFF; // unsigned [0,C)
            assert c < C : "Invalid landmark id " + c + " for landmarkCount=" + C;

            final int tildeBase = c * d;
            final long[] signBits = v.binaryVector;
            final byte[] extraBits = v.extraBits;

            float ip = 0f;
            for (int j = 0; j < d; j++) {
                int sign = AsymmetricHashing.QuantizedVector.getBit(signBits, j) ? 1 : 0;
                int extra = AsymmetricHashing.QuantizedVector.readExtraCode(extraBits, j, exBits);

                // Matches the C++ generic scalar reconstruction formula:
                //   (sign << exBits) + extra - (2^exBits - 0.5)
                float code = (float) ((sign << exBits) + extra) + centerBias;
                ip += qp.tildeQPool[tildeBase + j] * code;
            }

            return v.scale * ip + qp.dotQMuByLandmark[c] + v.offset;
        };
    }

    private static final float[] EMPTY = new float[0];

    /** Allocates only the preparation terms used by this model's scoring formula. */
    QueryPrecompute newQueryPrecompute() {
        int d = ash.quantizedDim;
        int c = ash.landmarkCount;
        boolean generic = !AsymmetricHashing.usesFastScanProjectionCode(ash.bitsPerDimension);
        return new QueryPrecompute(d, c, new float[d],
                generic ? new float[Math.multiplyExact(c, d)] : EMPTY,
                generic ? new float[c] : EMPTY, new float[c]);
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
        // Projection codes already include landmark adjustment in their stored offset.
        // Neither their direct scorer nor their LUT needs the generic centered vectors.
        if (AsymmetricHashing.usesFastScanProjectionCode(ash.bitsPerDimension)) return;
        float sumQProj = 0f;
        for (float value : prepared.qProj) sumQProj += value;
        for (int c = 0; c < ash.landmarkCount; c++) {
            prepared.sumTildeQByLandmark[c] = sumQProj - ash.landmarkProjSum[c];
            int base = c * ash.quantizedDim;
            for (int j = 0; j < ash.quantizedDim; j++) {
                prepared.tildeQPool[base + j] = prepared.qProj[j] - ash.landmarkProj[c][j];
            }
        }
    }

    static final class QueryPrecompute {
        final int d;
        final int C;
        final float[] qProj;                   // [d], Aq
        final float[] tildeQPool;              // [C * d], Aq - Aμ_c
        final float[] sumTildeQByLandmark;     // [C]
        final float[] dotQMuByLandmark;        // [C]

        QueryPrecompute(
                int d,
                int C,
                float[] qProj,
                float[] tildeQPool,
                float[] sumTildeQByLandmark,
                float[] dotQMuByLandmark) {
            this.d = d;
            this.C = C;
            this.qProj = qProj;
            this.tildeQPool = tildeQPool;
            this.sumTildeQByLandmark = sumTildeQByLandmark;
            this.dotQMuByLandmark = dotQMuByLandmark;
        }
    }

    @FunctionalInterface
    public interface ASHScoreFunction {
        float similarityTo(AsymmetricHashing.QuantizedVector vector2);
    }
}
