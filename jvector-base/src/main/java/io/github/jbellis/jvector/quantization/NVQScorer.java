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

import io.github.jbellis.jvector.vector.VectorSimilarityType;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;

public class NVQScorer {
    final NVQuantization nvq;

    /**
     * Initialize the NVQScorer with an instance of NVQuantization.
     */
    public NVQScorer(NVQuantization nvq) {
        this.nvq = nvq;
    }

    public NVQScoreFunction scoreFunctionFor(VectorFloat<?> query, VectorSimilarityType similarityFunction) {
        switch (similarityFunction) {
            case DOT_PRODUCT:
                return dotProductScoreFunctionFor(query);
            case EUCLIDEAN:
                return euclideanScoreFunctionFor(query);
            case COSINE:
                return cosineScoreFunctionFor(query);
            default:
                throw new IllegalArgumentException("Unsupported similarity function " + similarityFunction);
        }
    }

    private NVQScoreFunction dotProductScoreFunctionFor(VectorFloat<?> query) {
        /* Each sub-vector of query vector (full resolution) will be compared to NVQ quantized sub-vectors that were
         * first de-meaned by subtracting the global mean.
         */
        var queryGlobalBias = VectorUtil.dotProduct(query, this.nvq.globalMean);
        var querySubVectors = this.nvq.getSubVectors(query);

        switch (this.nvq.bitsPerDimension) {
            case EIGHT:
                for (VectorFloat<?> querySubVector : querySubVectors) {
                    VectorUtil.nvqShuffleQueryInPlace8bit(querySubVector);
                }

                return vector2 -> {
                    float nvqDot = 0;
                    for (int i = 0; i < querySubVectors.length; i++) {
                        var svDB = vector2.subVectors[i];
                        nvqDot += VectorUtil.nvqDotProduct8bit(querySubVectors[i],
                                svDB.bytes, svDB.growthRate, svDB.midpoint,
                                svDB.minValue, svDB.maxValue
                        );
                    }
                    return (1 + nvqDot + queryGlobalBias) / 2;
                };
            default:
                throw new IllegalArgumentException("Unsupported bits per dimension " + this.nvq.bitsPerDimension);
        }
    }

    private NVQScoreFunction euclideanScoreFunctionFor(VectorFloat<?> query) {
        /* Each sub-vector of query vector (full resolution) will be compared to NVQ quantized sub-vectors that were
         * first de-meaned by subtracting the global mean.
         */
        var shiftedQuery = VectorUtil.sub(query, this.nvq.globalMean);
        var querySubVectors = this.nvq.getSubVectors(shiftedQuery);

        switch (this.nvq.bitsPerDimension) {
            case EIGHT:
                for (VectorFloat<?> querySubVector : querySubVectors) {
                    VectorUtil.nvqShuffleQueryInPlace8bit(querySubVector);
                }

                return vector2 -> {
                    float dist = 0;
                    for (int i = 0; i < querySubVectors.length; i++) {
                        var svDB = vector2.subVectors[i];
                        dist += VectorUtil.nvqSquareL2Distance8bit(
                                querySubVectors[i],
                                svDB.bytes, svDB.growthRate, svDB.midpoint,
                                svDB.minValue, svDB.maxValue
                        );
                    }

                    return 1 / (1 + dist);
                };
            default:
                throw new IllegalArgumentException("Unsupported bits per dimension " + this.nvq.bitsPerDimension);
        }
    }

    private NVQScoreFunction cosineScoreFunctionFor(VectorFloat<?> query) {
        float queryNorm = (float) Math.sqrt(VectorUtil.dotProduct(query, query));
        var querySubVectors = this.nvq.getSubVectors(query);
        var meanSubVectors = this.nvq.getSubVectors(this.nvq.globalMean);

        switch (this.nvq.bitsPerDimension) {
            case EIGHT:
                for (var i = 0; i < querySubVectors.length; i++) {
                    VectorUtil.nvqShuffleQueryInPlace8bit(querySubVectors[i]);
                    VectorUtil.nvqShuffleQueryInPlace8bit(meanSubVectors[i]);
                }

                return vector2 -> {
                    float cos = 0;
                    float squaredNormalization = 0;
                    for (int i = 0; i < querySubVectors.length; i++) {
                        var svDB = vector2.subVectors[i];
                        var partialCosSim = VectorUtil.nvqCosine8bit(querySubVectors[i],
                                svDB.bytes, svDB.growthRate, svDB.midpoint,
                                svDB.minValue, svDB.maxValue,
                                meanSubVectors[i]);
                        cos += partialCosSim[0];
                        squaredNormalization += partialCosSim[1];
                    }
                    float cosine = (cos / queryNorm) / (float) Math.sqrt(squaredNormalization);

                    return (1 + cosine) / 2;
                };
            default:
                throw new IllegalArgumentException("Unsupported bits per dimension " + this.nvq.bitsPerDimension);
        }
    }

    public interface NVQScoreFunction {
        /**
         * @return the similarity to another vector
         */
        float similarityTo(NVQuantization.QuantizedVector vector2);
    }
}
