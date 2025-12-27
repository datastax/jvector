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
 * Single-landmark case (C = 1).
 * NOTE: Only DOT_PRODUCT is supported for now.
 */
public final class ASHScorer {
    private final AsymmetricHashing ash;

    public ASHScorer(AsymmetricHashing ash) {
        this.ash = ash;
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

        // Compute q_centered = q - μ
        // (avoid allocation by doing subtraction on the fly during projection)
        final VectorFloat<?> mu = ash.globalMean;

        final float[][] A = ash.stiefelTransform.AFloat;  // shape: d × originalDim
        final int originalDim = ash.stiefelTransform.cols;
        final float[] muArr = new float[originalDim];

        final var vecUtil =
                io.github.jbellis.jvector.vector.VectorizationProvider
                        .getInstance()
                        .getVectorUtilSupport();

        for (int k = 0; k < originalDim; k++) muArr[k] = mu.get(k);

        final float[] qArr = new float[originalDim];
        for (int k = 0; k < originalDim; k++) qArr[k] = query.get(k);

        // Project query: tildeQ[j] = (A (q - μ))[j]
        final float[] tildeQ = new float[d];
        for (int j = 0; j < d; j++) {
            final float[] Aj = A[j];
            float acc = 0.0f;
            for (int k = 0; k < originalDim; k++) {
                acc += Aj[k] * (qArr[k] - muArr[k]);
            }
            tildeQ[j] = acc;
        }

        // <tildeQ, 1> = sum_j tildeQ[j]
        float tmpSum = 0f;
        for (int j = 0; j < d; j++) {
            tmpSum += tildeQ[j];
        }
        final float sumTildeQ = tmpSum;

        // <q, μ>
        final float dotQMu = VectorUtil.dotProduct(query, mu);

        // ||μ||^2
        final float muNormSq = VectorUtil.dotProduct(mu, mu);

        // Precompute invSqrtD once
        final float invSqrtD = (float) (1.0 / Math.sqrt(d));

        return (AsymmetricHashing.QuantizedVector v) -> {
            // Single landmark baseline: ignore v.landmark for now (assumed 0)
            final float residualNorm = v.residualNorm;       // ||x - μ||₂
            final float dotXMu = v.dotWithLandmark;          // <x, μ>

            // scale = d^{-1/2} * ||x - μ||
            final float scale = invSqrtD * residualNorm;

            // masked-add term: <tildeQ, bin(xhat)> where bin ∈ {0,1}^d
            // This is sum of tildeQ[j] over set bits in v.binaryVector.
            float maskedAdd = 0f;

            // v.binaryVector stores bits packed in longs, little-endian per word (bit j corresponds to dimension base+j)
            final long[] bits = v.binaryVector;
            int base = 0;
            for (int w = 0; w < bits.length && base < d; w++, base += 64) {
                long word = bits[w];
                while (word != 0L) {
                    int bit = Long.numberOfTrailingZeros(word);
                    int idx = base + bit;
                    if (idx < d) maskedAdd += tildeQ[idx];
                    word &= (word - 1);
                }
            }

            // ASH dot-product approximation (single landmark, {0,1} encoding):
            //
            //   scale * (2⟨q̃, b⟩ − ⟨q̃, 1⟩) + ⟨q, μ⟩ + (⟨x, μ⟩ − ||μ||²)
            //
            // Notes:
            // - b ∈ {0,1}^d is the binary code (converted from the paper’s {−1,+1} form).
            // - scale = ||x − μ|| / √d  (pure Eq. 11 baseline; no bias correction).
            // - ⟨q, μ⟩ is a query-only offset and MUST NOT be scaled.
            // - Scaling ⟨q, μ⟩ leads to large systematic error (see paper Eq. 11 regrouping).
            return scale * (2f * maskedAdd - sumTildeQ) + dotQMu + (dotXMu - muNormSq);

        };
    }

    @FunctionalInterface
    public interface ASHScoreFunction {
        float similarityTo(AsymmetricHashing.QuantizedVector vector2);
    }
}

