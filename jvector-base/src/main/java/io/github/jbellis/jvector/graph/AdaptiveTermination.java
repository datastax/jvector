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

/**
 * Per-query adaptive early termination based on result-set discovery.
 *
 * <p>
 * This is based on:
 * <ul>
 *   <li>[1] Tommaso Teofili and Jimmy Lin. 2025. Patience in Proximity:
 *       A Simple Early Termination Strategy for HNSW Graph Traversal in
 *       Approximate k-Nearest Neighbor Search. ECIR 2025.
 *       https://doi.org/10.1007/978-3-031-88714-7_39</li>
 *   <li>[2] Tommaso Teofili. 2026. Elasticsearch HNSW: Adaptive early
 *       termination for vector search. Elasticsearch Labs.
 *       https://www.elastic.co/search-labs/blog/hnsw-elasticsearch-adaptive-early-termination</li>
 * </ul>
 *
 * <p>
 * The adaptive variant tracks a smoothed discovery rate per query:
 *
 * <pre>
 *   d_i = alpha * introduced_i / (steps_i * k) + (1 - alpha) * d_{i-1}
 *   threshold_i = mean(d) * thresholdScale * stddev(d)
 *   patience_i = patienceNumerator / (1 + stddev(d))
 * </pre>
 *
 * <p>
 * This class only decides when traversal should terminate after completed visits.
 * It must not be used to skip expansion of a candidate that has already been
 * selected for expansion.
 */
final class AdaptiveTermination {
    static final float DISCOVERY_ALPHA = 0.9f;
    static final float THRESHOLD_SCALE = 0.01f;
    static final float PATIENCE_NUMERATOR = 10.0f;

    private static final float EPSILON = 1e-6f;
    private static final int MIN_OBSERVATIONS = 2;

    private int k;
    private float discovery;

    // Welford state for rolling mean/stddev of discovery.
    private int observations;
    private double mean;
    private double m2;

    private int saturatedVisits;

    AdaptiveTermination() {
        reset(1);
    }

    void reset(int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be > 0");
        }

        this.k = k;
        this.discovery = 0.0f;
        this.observations = 0;
        this.mean = 0.0;
        this.m2 = 0.0;
        this.saturatedVisits = 0;
    }

    /**
     * Records one completed graph-visit interval.
     *
     * @param introduced number of new entries introduced into the tracked result set
     * @param steps number of graph visits represented by this observation
     * @param resultSetFull whether the tracked result set is full enough to stop
     */
    void observe(int introduced, int steps, boolean resultSetFull) {
        int safeIntroduced = Math.max(0, introduced);
        int safeSteps = Math.max(1, steps);

        float rawDiscovery = safeDivide(safeIntroduced, (float) safeSteps * k);
        discovery = DISCOVERY_ALPHA * rawDiscovery + (1.0f - DISCOVERY_ALPHA) * discovery;

        updateStats(discovery);

        if (resultSetFull && observations >= MIN_OBSERVATIONS && isSaturated()) {
            saturatedVisits++;
        } else {
            saturatedVisits = 0;
        }
    }

    boolean shouldTerminate() {
        return observations >= MIN_OBSERVATIONS
                && saturatedVisits > patience();
    }

    private boolean isSaturated() {
        return discovery < saturationThreshold();
    }

    private float saturationThreshold() {
        return (float) mean - THRESHOLD_SCALE * stddev();
    }

    private float patience() {
        return safeDivide(PATIENCE_NUMERATOR, 1.0f + stddev());
    }

    /**
     * Welford's online update for numerically stable mean and variance.
     */
    private void updateStats(float value) {
        observations++;

        double delta = value - mean;
        mean += delta / observations;
        double delta2 = value - mean;
        m2 += delta * delta2;
    }

    private float stddev() {
        if (observations < MIN_OBSERVATIONS) {
            return 0.0f;
        }

        double variance = m2 / (observations - 1);
        if (!Double.isFinite(variance) || variance <= 0.0) {
            return 0.0f;
        }

        return (float) Math.sqrt(variance);
    }

    /**
     * Local finite/zero guard keeps the adaptive equations robust without
     * introducing a dependency on a broader math helper.
     */
    private static float safeDivide(float numerator, float denominator) {
        if (!Float.isFinite(numerator)) {
            return 0.0f;
        }
        if (!Float.isFinite(denominator) || Math.abs(denominator) < EPSILON) {
            return 0.0f;
        }
        return numerator / denominator;
    }
}
