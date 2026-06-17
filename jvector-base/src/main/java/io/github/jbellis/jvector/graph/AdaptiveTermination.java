package io.github.jbellis.jvector.graph;

/**
 * Distance-adaptive base-layer termination.
 *
 * <p>The adaptive rule stops when the best unexpanded candidate is farther than
 * the kth-best discovered result by a relative factor:
 *
 * <pre>
 *   d(q, candidate) >= (1 + gamma) * d_k
 * </pre>
 *
 * <p>GraphSearcher works with higher-is-better normalized scores. We convert
 * those scores back to a distance-like scale using the common bounded-score
 * convention {@code score = 1 / (1 + distance)}. If a query produces scores
 * outside the supported range, callers must fall back to standard termination.
 */
final class AdaptiveTermination {
    static final float DEFAULT_GAMMA = 0.005f;

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

    AdaptiveTermination() {
        reset(1, DEFAULT_GAMMA);
    }

    void reset(int k) {
        reset(k, DEFAULT_GAMMA);
    }

    void reset(int k, float gamma) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be > 0");
        }
        validateGamma(gamma);

        this.k = k;
        this.gamma = gamma;
        this.unsupportedScoreObserved = false;
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
        return candidateDistance + EPSILON >= cutoff
                ? Decision.TERMINATE
                : Decision.CONTINUE;
    }

    private static double scoreToDistance(float score) {
        if (!Float.isFinite(score)) {
            return Double.NaN;
        }
        if (score < MIN_SCORE - EPSILON || score > MAX_SCORE + EPSILON) {
            return Double.NaN;
        }
        if (score <= EPSILON) {
            return Double.POSITIVE_INFINITY;
        }

        double boundedScore = Math.min(MAX_SCORE, Math.max(MIN_SCORE, score));
        return (1.0d / boundedScore) - 1.0d;
    }
}
