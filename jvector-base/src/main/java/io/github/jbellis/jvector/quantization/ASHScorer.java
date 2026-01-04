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
 * Query-time scorer for ASH vectors.
 *
 * Multi-landmark scoring (C >= 1) following the paper’s Eq. 11 decomposition:
 * for each landmark c we form q̃_c = A (q - μ_c) = (Aq) - (Aμ_c) once per query,
 * then score each vector using its stored landmark id.
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

    public ASHScoreFunction scoreFunctionFor(VectorFloat<?> query, VectorSimilarityFunction similarityFunction) {
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
        final int d = ash.quantizedDim;
        final int D = ash.originalDimension;
        final int C = ash.landmarkCount;

        final float[][] A = ash.stiefelTransform.AFloat; // [d][D]

        final var vecUtil =
                io.github.jbellis.jvector.vector.VectorizationProvider
                        .getInstance()
                        .getVectorUtilSupport();

        // Materialize query once (avoid VectorFloat.get in inner loops)
        final float[] qArr = new float[D];
        for (int k = 0; k < D; k++) qArr[k] = query.get(k);

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
        // Pooled storage for q̃_c vectors:
        // tildeQPool[c*d + j] == q̃_c[j]
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

        return (AsymmetricHashing.QuantizedVector v) -> {
            final int c = v.landmark & 0xFF; // unsigned [0,C)
            assert c < C : "Invalid landmark id " + c + " for landmarkCount=" + C;

            // scale = ||x − μ_c|| / √d  (precomputed at encode-time)
            final float scale = v.scale;

            // offset = <x, μ_c> − ||μ_c||²  (precomputed at encode-time)
            final float offset = v.offset;

            // maskedAdd = <q̃_c, b>
            // b ∈ {0,1}^d stored as packed longs
            float maskedAdd = 0f;

            final int tildeBase = c * d;
            final float sumTildeQ = sumTildeQByLandmark[c];
            final float dotQMu = dotQMuByLandmark[c];

            final long[] bits = v.binaryVector;
            int dimBase = 0;
            for (int w = 0; w < bits.length && dimBase < d; w++, dimBase += 64) {
                long word = bits[w];
                while (word != 0L) {
                    int bit = Long.numberOfTrailingZeros(word);
                    int idx = dimBase + bit;
                    if (idx < d) {
                        maskedAdd += tildeQPool[tildeBase + idx];
                    }
                    word &= (word - 1);
                }
            }

            // ASH dot-product approximation (multi-landmark, {0,1} encoding):
            //
            //   scale * (2⟨q̃_c, b⟩ − ⟨q̃_c, 1⟩) + ⟨q, μ_c⟩ + (⟨x, μ_c⟩ − ||μ_c||²)
            //
            // where q̃_c = A(q − μ_c) is precomputed once per query and per landmark c.
            return scale * (2f * maskedAdd - sumTildeQ) + dotQMu + offset;
        };
    }

    @FunctionalInterface
    public interface ASHScoreFunction {
        float similarityTo(AsymmetricHashing.QuantizedVector vector2);
    }
}
