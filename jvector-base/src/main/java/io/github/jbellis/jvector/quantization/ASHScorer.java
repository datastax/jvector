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
 * Multi-landmark scoring follows the same decomposition used by the 1-bit ASH path:
 * for each landmark c we form q̃_c = A(q - μ_c) = Aq - Aμ_c once per query, then
 * score each vector using its stored landmark id.
 * </p>
 *
 * <p>
 * For 1-bit ASH, the packed sign bits are interpreted as values in {-1,+1}. For
 * multibit ASH, sign bits plus packed extra bits reconstruct the centered scalar
 * code used by the encoder:
 * </p>
 *
 * <pre>
 *   code_j = ((sign_j &lt;&lt; exBits) + extra_j) - (2^exBits - 0.5)
 * </pre>
 *
 * <p>
 * This exactly matches the C++ scalar reconstruction formula and keeps SIMD/block
 * paths optional correctness optimizations outside this reference scorer.
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

        return multibitDotProductScoreFunction(qp);
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
            //   scale * (2<q̃_c,b> - <q̃_c,1>) + <q,μ_c> + (<x,μ_c> - ||μ_c||²)
            return v.scale * (2f * maskedAdd - qp.sumTildeQByLandmark[c])
                    + qp.dotQMuByLandmark[c]
                    + v.offset;
        };
    }

    private ASHScoreFunction multibitDotProductScoreFunction(QueryPrecompute qp) {
        final int d = qp.d;
        final int C = qp.C;
        final int exBits = ash.bitsPerDimension - 1;
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

                // Matches C++:
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
        //   tildeQ_c   = Aq - Aμ_c
        //   sumTildeQc = sum(Aq) - sum(Aμ_c)
        //   dotQMu_c   = <q, μ_c>
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

        return new QueryPrecompute(d, C, tildeQPool, sumTildeQByLandmark, dotQMuByLandmark);
    }

    private static final class QueryPrecompute {
        final int d;
        final int C;
        final float[] tildeQPool;             // [C * d]
        final float[] sumTildeQByLandmark;    // [C]
        final float[] dotQMuByLandmark;       // [C]

        QueryPrecompute(
                int d,
                int C,
                float[] tildeQPool,
                float[] sumTildeQByLandmark,
                float[] dotQMuByLandmark) {
            this.d = d;
            this.C = C;
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

