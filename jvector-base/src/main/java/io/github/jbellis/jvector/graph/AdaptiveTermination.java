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
package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

/**
 * Distance-adaptive termination: a top-K search stops when the best unexpanded candidate is farther from the query
 * than the k-th best result found, by a relative margin:
 *
 * <pre>
 *   d(q, candidate) > (1 + gamma) * d(q, k-th best)
 * </pre>
 *
 * <p>{@code d} is a true metric distance, so {@code gamma} means the same thing for every similarity function and
 * whatever the lengths of the vectors. Scores are read back as distances as follows; the rule compares squared
 * distances against {@code (1 + gamma)^2}, and any positive scale factor cancels:
 *
 * <ul>
 * <li>EUCLIDEAN: the score is {@code 1 / (1 + d^2)}, so {@code d^2 = 1 / score - 1}.</li>
 * <li>COSINE: the score is {@code (1 + cos) / 2}, and {@code d^2 = 2 - 2 cos} is proportional to {@code 1 - score}.</li>
 * <li>DOT_PRODUCT: the score is {@code (1 + q.x) / 2}. With {@code B} an upper bound on {@code q.x}, such as the
 * query's length times the largest stored vector length, {@code B - q.x} is proportional to the squared Euclidean
 * distance between the query and the vector after the standard reduction of inner-product search to nearest-neighbor
 * search (each vector scaled into the unit ball and lifted onto the unit sphere by one extra coordinate, the query
 * scaled to unit length). In score units that is {@code top - score} with {@code top = (1 + B) / 2}. An approximate
 * score above {@code top} raises {@code top} to it, which only loosens the margin. On unit-length vectors
 * {@code B = 1} and the rule reads {@code 1 - score}, as cosine does.</li>
 * </ul>
 *
 * <p>The search never stops earlier than the standard rule would, and never later than a standard search with
 * {@value #CAP_FACTOR} times the beam would: the searcher keeps the {@value #CAP_FACTOR}{@code k}-th best score found
 * and passes it to {@link #raiseCap}, and candidates scoring below it stop the search whatever the margin. The cap
 * bounds the work on data whose distances are concentrated, where a relative margin admits many candidates.
 *
 * <p>A non-finite score, or a dot-product search whose bound is unknown, leaves the standard rule to decide.
 */
final class AdaptiveTermination {
    /** The default margin on the metric distance: 1% on squared distances. */
    static final float DEFAULT_GAMMA = (float) (Math.sqrt(1.01d) - 1.0d);

    /** Adaptive stopping searches no further than a standard search with this many times the beam. */
    static final int CAP_FACTOR = 2;

    // Room for float rounding in the scores, relative to the scale of the distances compared.
    private static final double EPSILON = 1e-6d;

    enum Decision {
        CONTINUE,
        TERMINATE,
        FALL_BACK_TO_STANDARD
    }

    private int k;
    /** (1 + gamma)^2: the margin on squared distances. */
    private double factor;
    /** Euclidean scores are read as 1 / score - 1; the others as top - score. */
    private boolean euclidean;
    /** In score units, the score of a distance of zero (linear forms only). Rises with any higher score observed. */
    private float top;
    /** The cap: the CAP_FACTOR * k-th best score found, once known. Only rises. */
    private float capScore;
    /**
     * The highest bar {@link #lowestContinuableScore} has returned. Callers skip nodes scored below a bar, so the rule
     * never continues a candidate below it; otherwise a rising {@code top} would lower the bar after such a skip.
     */
    private float floor;
    private boolean unsupportedScoreObserved;

    AdaptiveTermination() {
        reset(1, DEFAULT_GAMMA, VectorSimilarityFunction.COSINE, Float.NaN);
    }

    /**
     * Prepares for a search of the {@code k} best results scored by {@code similarity}. {@code dotProductBound} is an
     * upper bound on the query's dot product with any stored vector, needed for DOT_PRODUCT and ignored otherwise.
     *
     * @return whether this rule can decide the search; if not, the standard rule must be used
     */
    boolean reset(int k, float gamma, VectorSimilarityFunction similarity, float dotProductBound) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be > 0");
        }
        validateGamma(gamma);
        this.k = k;
        this.factor = (1.0d + gamma) * (1.0d + gamma);
        this.unsupportedScoreObserved = false;
        this.capScore = Float.NEGATIVE_INFINITY;
        this.floor = Float.NEGATIVE_INFINITY;
        switch (similarity) {
            case EUCLIDEAN:
                euclidean = true;
                top = Float.NaN;
                return true;
            case COSINE:
                euclidean = false;
                top = 1.0f;
                return true;
            case DOT_PRODUCT:
                euclidean = false;
                if (!Float.isFinite(dotProductBound) || dotProductBound < 0) {
                    unsupportedScoreObserved = true;
                    return false;
                }
                top = (1.0f + dotProductBound) / 2.0f;
                return true;
            default:
                unsupportedScoreObserved = true;
                return false;
        }
    }

    static void validateGamma(float gamma) {
        if (!Float.isFinite(gamma) || gamma < 0.0f) {
            throw new IllegalArgumentException("gamma must be finite and >= 0");
        }
    }

    /**
     * Records a score among the best found. A score above the current zero-distance point raises it, so every
     * distance stays nonnegative.
     */
    void observe(float score) {
        if (!euclidean && score > top && score != Float.POSITIVE_INFINITY) {
            top = score;
        }
    }

    /** Raises the cap to {@code capScore}, the {@value #CAP_FACTOR}{@code k}-th best score found. */
    void raiseCap(float capScore) {
        if (capScore > this.capScore) this.capScore = capScore;
    }

    /**
     * Decides whether the current best unexpanded candidate should terminate
     * the search under the distance-adaptive rule.
     */
    Decision shouldTerminate(float candidateScore, NodeQueue discoveredResults) {
        if (unsupportedScoreObserved) {
            return Decision.FALL_BACK_TO_STANDARD;
        }
        if (discoveredResults.size() < k) {
            return Decision.CONTINUE;
        }
        float kthScore = discoveredResults.topScore();
        if (Float.isNaN(candidateScore) || Float.isNaN(kthScore) || kthScore == Float.POSITIVE_INFINITY) {
            unsupportedScoreObserved = true;
            return Decision.FALL_BACK_TO_STANDARD;
        }
        if (kthScore == Float.NEGATIVE_INFINITY || candidateScore == Float.POSITIVE_INFINITY) {
            return Decision.CONTINUE;
        }
        if (candidateScore == Float.NEGATIVE_INFINITY || candidateScore < capScore || candidateScore < floor) {
            return Decision.TERMINATE;
        }
        if (euclidean) {
            if (!(kthScore > 0.0f)) {
                return Decision.CONTINUE;
            }
            if (!(candidateScore > 0.0f)) {
                return Decision.TERMINATE;
            }
            double kthDistance = Math.max(0.0d, 1.0d / kthScore - 1.0d);
            double candidateDistance = 1.0d / candidateScore - 1.0d;
            return candidateDistance > factor * kthDistance + EPSILON * (1.0d + kthDistance)
                    ? Decision.TERMINATE : Decision.CONTINUE;
        }
        double t = top;
        return t - candidateScore > factor * (t - kthScore) + slack(t)
                ? Decision.TERMINATE : Decision.CONTINUE;
    }

    /**
     * The lowest score a candidate may have and still be continued when the k-th best score found is
     * {@code kthScore}, the cap included. Rounded down, so it never excludes a candidate {@link #shouldTerminate}
     * would continue, and it never falls during a search. When this rule cannot interpret the scores it returns
     * negative infinity, leaving the standard rule to decide.
     */
    float lowestContinuableScore(float kthScore) {
        if (unsupportedScoreObserved || !Float.isFinite(kthScore)) {
            return Float.NEGATIVE_INFINITY;
        }
        double score;
        if (euclidean) {
            if (!(kthScore > 0.0f)) return Float.NEGATIVE_INFINITY;
            double kthDistance = Math.max(0.0d, 1.0d / kthScore - 1.0d);
            score = 1.0d / (1.0d + factor * kthDistance + EPSILON * (1.0d + kthDistance));
        } else {
            double t = top;
            score = t - factor * (t - kthScore) - slack(t);
        }
        float bar = Math.max(capScore, Math.nextDown(Math.nextDown((float) score)));
        if (bar > floor) floor = bar;
        return floor;
    }

    private static double slack(double top) {
        return EPSILON * Math.max(1.0d, Math.abs(top));
    }
}
