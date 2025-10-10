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
 * Provides scoring functions for comparing NVQ-quantized vectors with query vectors.
 * Supports dot product, Euclidean, and cosine similarity functions.
 */
public class NVQScorer {
    final NVQuantization nvq;

    /**
     * Constructs an NVQScorer with the given NVQuantization instance.
     *
     * @param nvq the NVQuantization instance to use for scoring
     */
    public NVQScorer(NVQuantization nvq) {
        this.nvq = nvq;
    }

    /**
     * Creates a score function for comparing the query vector against NVQ-quantized vectors.
     *
     * @param query the query vector
     * @param similarityFunction the similarity function to use
     * @return a score function for the given query and similarity function
     * @throws IllegalArgumentException if the similarity function is not supported
     */
    public NVQScoreFunction scoreFunctionFor(VectorFloat<?> query, VectorSimilarityFunction similarityFunction) {
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

    /**
     * Creates a dot product score function for the given query vector.
     *
     * @param query the query vector
     * @return a dot product score function
     * @throws IllegalArgumentException if the bits per dimension is not supported
     */
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

    /**
     * Creates a Euclidean distance score function for the given query vector.
     * The score is converted to a similarity using 1 / (1 + distance).
     *
     * @param query the query vector
     * @return a Euclidean similarity score function
     * @throws IllegalArgumentException if the bits per dimension is not supported
     */
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

    /**
     * Creates a cosine similarity score function for the given query vector.
     *
     * @param query the query vector
     * @return a cosine similarity score function
     * @throws IllegalArgumentException if the bits per dimension is not supported
     */
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

    /**
     * A functional interface for computing similarity between a query and an NVQ-quantized vector.
     */
    public interface NVQScoreFunction {
        /**
         * Computes the similarity score to another quantized vector.
         *
         * @param vector2 the quantized vector to compare against
         * @return the similarity score
         */
        float similarityTo(NVQuantization.QuantizedVector vector2);
    }
}
