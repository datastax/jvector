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

/**
 * How an ASH model's kernel estimate becomes the graph's similarity, chosen once per model by its similarity.
 *
 * <p>Every ASH kernel computes {@code scale * <Aq, code> + queryTerm[landmark] + offset}. For DOT_PRODUCT and
 * COSINE that estimates the dot product (of normalized vectors, for COSINE), and the similarity is
 * {@code (1 + e) / 2}. A EUCLIDEAN model stores its headers and query terms so the same kernels estimate
 * {@code -||q - x||^2 / 2} in model units (paper, Appendix A), and the similarity is {@code 1 / (1 + ||q - x||^2)}.
 * Scorers call {@link #similarity} once per score on their model's instance; a JVM that uses one of the two
 * scales compiles the call to that scale's arithmetic alone.</p>
 */
abstract class ASHScoreScale {
    /** DOT_PRODUCT and COSINE. */
    static final ASHScoreScale DOT_PRODUCT = new DotProduct();

    /** The similarity for kernel estimate {@code estimate}. */
    abstract float similarity(float estimate);

    /**
     * Converts {@code scores[from, from + count)} in place from the DOT_PRODUCT scale, {@code (1 + e) / 2}, which
     * native whole-neighborhood kernels write, to this scale.
     */
    abstract void fromDotProductScale(float[] scores, int from, int count);

    /**
     * For a EUCLIDEAN scale, the factor {@code k} with {@code ||q - x||^2 = k * e}, which native neighborhood kernels
     * take to write this scale directly; zero for the DOT_PRODUCT scale.
     */
    abstract float squaredDistancePerEstimate();

    static ASHScoreScale euclidean(float unit) {
        return new Euclidean(unit);
    }

    private static final class DotProduct extends ASHScoreScale {
        @Override
        float similarity(float estimate) {
            return (1f + estimate) / 2f;
        }

        @Override
        void fromDotProductScale(float[] scores, int from, int count) {
            // Already on this scale.
        }

        @Override
        float squaredDistancePerEstimate() {
            return 0f;
        }
    }

    /** Squared distances below zero, which the estimate can give, are clamped to zero. */
    private static final class Euclidean extends ASHScoreScale {
        /** {@code d^2 = perEstimate * e}; and {@code d^2 = base - slope * s} for a DOT_PRODUCT-scale {@code s}. */
        private final float perEstimate;
        private final float base;
        private final float slope;

        Euclidean(float unit) {
            float inverseSquare = 1f / (unit * unit); // a power of two, so these are exact
            this.perEstimate = -2f * inverseSquare;
            this.base = 2f * inverseSquare;
            this.slope = 4f * inverseSquare;
        }

        @Override
        float similarity(float estimate) {
            return 1f / (1f + Math.max(0f, perEstimate * estimate));
        }

        @Override
        void fromDotProductScale(float[] scores, int from, int count) {
            for (int i = from, end = from + count; i < end; i++) {
                scores[i] = 1f / (1f + Math.max(0f, base - slope * scores[i]));
            }
        }

        @Override
        float squaredDistancePerEstimate() {
            return perEstimate;
        }
    }
}
