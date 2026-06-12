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

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Query-time scalar scorer for ASH vectors.
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
        return dotProductScoreFunctionFor(query);
    }

    /**
     * Scorer for dot product approximation.
     */
    private ASHScoreFunction dotProductScoreFunctionFor(VectorFloat<?> query) {
        QueryPrecompute qp = precomputeQuery(query);

        if (ash.bitsPerDimension == 1) {
            return oneBitDotProductScoreFunction(qp);
        }

        if (AsymmetricHashing.usesFastScanProjectionCode(ash.bitsPerDimension)) {
            return fastScanProjectionDotProductScoreFunction(qp);
        }

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

    private ASHScoreFunction fastScanProjectionDotProductScoreFunction(QueryPrecompute qp) {
        final int d = qp.d;
        final int C = qp.C;
        final int bitsPerDimension = ash.bitsPerDimension;

        return (AsymmetricHashing.QuantizedVector v) -> {
            final int c = v.landmark & 0xFF; // unsigned [0,C)
            assert c < C : "Invalid landmark id " + c + " for landmarkCount=" + C;

            // C++ projection-mode scoring:
            //   score = scale * <Aq, code> + <q, μ_c> + stored_offset
            // stored_offset already includes -scale * <Aμ_c, code>.
            float ip = AsymmetricHashing.dotProjectionCode(
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

    private QueryPrecompute precomputeQuery(VectorFloat<?> query) {
        final int d = ash.quantizedDim;
        final int D = ash.originalDimension;
        final int C = ash.landmarkCount;

        final float[][] A = ash.stiefelTransform.AFloat; // [d][D]

        final var vecUtil =
                io.github.jbellis.jvector.vector.VectorizationProvider
                        .getInstance()
                        .getVectorUtilSupport();

        // Materialize query once to avoid VectorFloat.get in inner loops.
        final float[] qArr = new float[D];
        for (int k = 0; k < D; k++) {
            qArr[k] = query.get(k);
        }

        // Compute Aq once per query:
        //   qProj[j] = <A[j], q>
        final float[] qProj = new float[d];
        float sumQProj = 0f;
        for (int j = 0; j < d; j++) {
            float v = vecUtil.ashDotRow(A[j], qArr);
            qProj[j] = v;
            sumQProj += v;
        }

        // For each landmark c:
        //   tildeQ_c   = Aq - Aμ_c       (1-bit and generic multibit paths)
        //   sumTildeQc = sum(Aq) - sum(Aμ_c)
        //   dotQMu_c   = <q, μ_c>        (all paths)
        final float[] tildeQPool = new float[C * d];
        final float[] sumTildeQByLandmark = new float[C];
        final float[] dotQMuByLandmark = new float[C];

        for (int c = 0; c < C; c++) {
            dotQMuByLandmark[c] = VectorUtil.dotProduct(query, ash.landmarks[c]);
            sumTildeQByLandmark[c] = sumQProj - ash.landmarkProjSum[c];

            final int base = c * d;
            final float[] muProj = ash.landmarkProj[c]; // [d]
            for (int j = 0; j < d; j++) {
                tildeQPool[base + j] = qProj[j] - muProj[j];
            }
        }

        return new QueryPrecompute(d, C, qProj, tildeQPool, sumTildeQByLandmark, dotQMuByLandmark);
    }

    private static final class QueryPrecompute {
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
