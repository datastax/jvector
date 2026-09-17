package io.github.jbellis.jvector.graph;

/**
 * Adaptive termination for graph DOT_PRODUCT similarity scores.
 *
 * <p>The adaptive rule stops when the best unexpanded candidate is farther than
 * the kth-best discovered result by a relative factor:
 *
 * <pre>
 *   d(q, candidate) > (1 + gamma) * d_k
 * </pre>
 *
 * <p>GraphSearcher works with higher-is-better normalized scores. We convert
 * those scores back to a squared-L2-like scale for normalized vectors using
 * {@code distance = 1 - score}. If a query produces scores
 * outside the supported range, callers must fall back to standard termination.
 */
final class AdaptiveTermination {
    static final float DEFAULT_GAMMA = 0.01f;

    private static final double MIN_SCORE = 0.0d;
    private static final double MAX_SCORE = 1.0d;
    private static final double EPSILON = 1e-6d;

    enum Decision {
        CONTINUE,
        TERMINATE,
        FALL_BACK_TO_STANDARD
    }

    private int k;
    private float gamma;
    private boolean unsupportedScoreObserved;
    private boolean rawDotProductScores;

    AdaptiveTermination() {
        reset(1, DEFAULT_GAMMA);
    }

    void reset(int k) {
        reset(k, DEFAULT_GAMMA);
    }

    void reset(int k, float gamma) {
        reset(k, gamma, false);
    }

    void reset(int k, float gamma, boolean rawDotProductScores) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be > 0");
        }
        validateGamma(gamma);

        this.k = k;
        this.gamma = gamma;
        this.unsupportedScoreObserved = false;
        this.rawDotProductScores = rawDotProductScores;
    }

    static void validateGamma(float gamma) {
        if (!Float.isFinite(gamma) || gamma < 0.0f) {
            throw new IllegalArgumentException("gamma must be finite and >= 0");
        }
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

        double candidateDistance = scoreToDistance(candidateScore);
        double kthDistance = scoreToDistance(discoveredResults.topScore());

        if (Double.isNaN(candidateDistance) || Double.isNaN(kthDistance)) {
            unsupportedScoreObserved = true;
            return Decision.FALL_BACK_TO_STANDARD;
        }

        if (Double.isInfinite(kthDistance)) {
            return Decision.CONTINUE;
        }
        if (Double.isInfinite(candidateDistance)) {
            return Decision.TERMINATE;
        }

        double cutoff = (1.0d + gamma) * kthDistance;
        return candidateDistance > cutoff + EPSILON
                ? Decision.TERMINATE
                : Decision.CONTINUE;
    }

    private double scoreToDistance(float score) {
        if (!Float.isFinite(score)) {
            return Double.NaN;
        }
        if (score < (rawDotProductScores ? -1.0d : MIN_SCORE) - EPSILON || score > MAX_SCORE + EPSILON) {
            return Double.NaN;
        }
        double boundedScore = Math.min(MAX_SCORE, Math.max(rawDotProductScores ? -1.0d : MIN_SCORE, score));
        return rawDotProductScores ? (1.0d - boundedScore) / 2.0d : 1.0d - boundedScore;
    }
}
