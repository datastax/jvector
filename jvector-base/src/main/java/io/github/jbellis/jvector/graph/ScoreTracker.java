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

import io.github.jbellis.jvector.util.ArrayUtil;
import io.github.jbellis.jvector.util.BoundedLongHeap;
import org.apache.commons.math3.stat.StatUtils;

import static io.github.jbellis.jvector.util.NumericUtils.floatToSortableInt;
import static io.github.jbellis.jvector.util.NumericUtils.sortableIntToFloat;

public interface ScoreTracker {
    class ScoreTrackerFactory {
        private TwoPhaseTracker twoPhaseTracker;
        private RelaxedMonotonicityTracker relaxedMonotonicityTracker;
        private NoOpTracker noOpTracker;

        ScoreTrackerFactory() {
            twoPhaseTracker = null;
            relaxedMonotonicityTracker = null;
            noOpTracker = null;
        }

        public ScoreTracker getScoreTracker(boolean pruneSearch, int rerankK, float threshold) {
            // track scores to predict when we are done with threshold queries
            final ScoreTracker scoreTracker;

            if (threshold > 0) {
                if (twoPhaseTracker == null) {
                    twoPhaseTracker = new ScoreTracker.TwoPhaseTracker();
                } else {
                    twoPhaseTracker.reset(threshold);
                }
                scoreTracker = twoPhaseTracker;
            } else {
                if (pruneSearch) {
                    if (relaxedMonotonicityTracker == null) {
                        relaxedMonotonicityTracker = new ScoreTracker.RelaxedMonotonicityTracker();
                    } else {
                        relaxedMonotonicityTracker.reset(rerankK);
                    }
                    scoreTracker = relaxedMonotonicityTracker;
                } else {
                    if (noOpTracker == null) {
                        noOpTracker = new ScoreTracker.NoOpTracker();
                    }
                    scoreTracker = noOpTracker;
                }
            }
            return scoreTracker;
        }
    }

    ScoreTracker NO_OP = new NoOpTracker();

    void track(float score);

    boolean shouldStop();

    class NoOpTracker implements ScoreTracker {
        @Override
        public void track(float score) { }

        @Override
        public boolean shouldStop() {
            return false;
        }
    }

    /**
     * Follows the methodology of section 3.1 in "VBase: Unifying Online Vector Similarity Search
     * and Relational Queries via Relaxed Monotonicity" to determine when we've left phase 1
     * (finding the local maximum) and entered phase 2 (mostly just finding worse options)
     */
    class TwoPhaseTracker implements ScoreTracker {
        static final int RECENT_SCORES_TRACKED = 500;
        static final int BEST_SCORES_TRACKED = 100;

        // a sliding window of recent scores
        private final double[] recentScores;
        private int recentEntryIndex;

        // Heap of the best scores seen so far
        BoundedLongHeap bestScores;

        // observation count
        private int observationCount;

        private double threshold;

        TwoPhaseTracker(double threshold) {
            this.recentScores = new double[RECENT_SCORES_TRACKED];
            this.bestScores = new BoundedLongHeap(BEST_SCORES_TRACKED);
            this.observationCount = 0;
            this.threshold = threshold;
        }

        TwoPhaseTracker() {
            this(0);
        }

        void reset(double threshold) {
            this.bestScores.clear();
            this.observationCount = 0;
            this.threshold = threshold;
        }

        @Override
        public void track(float score) {
            bestScores.push(floatToSortableInt(score));
            recentScores[recentEntryIndex] = score;
            recentEntryIndex = (recentEntryIndex + 1) % recentScores.length;
            observationCount++;
        }

        @Override
        public boolean shouldStop() {
            // don't stop if we don't have enough data points
            if (observationCount < RECENT_SCORES_TRACKED) {
                return false;
            }
            // evaluation is expensive so only do it 1% of the time
            if (observationCount % 100 != 0) {
                return false;
            }

            // We're in phase 2 if the 99th percentile of the recent scores evaluated is lower
            // than the worst of the best scores seen.
            //
            // (paper suggests using the median of recent scores, but experimentally that is too prone to false positives.
            // 90th does seem to be enough, but 99th doesn't result in much extra work, so we'll be conservative)
            double windowMedian = StatUtils.percentile(recentScores, 99);
            double worstBestScore = sortableIntToFloat((int) bestScores.top());
            return windowMedian < worstBestScore && windowMedian < threshold;
        }
    }

    /**
     * Follows the methodology of section 3.1 in "VBase: Unifying Online Vector Similarity Search
     * and Relational Queries via Relaxed Monotonicity" to determine when we've left phase 1
     * (finding the local maximum) and entered phase 2 (mostly just finding worse options)
     * To compute quantiles quickly, we treat the distribution of the data as Normal,
     * track its mean and variance, and compute quantiles from them as:
     *     mean + SIGMA_FACTOR * sqrt(variance)
     * Empirically, SIGMA_FACTOR=1.75 seems to work reasonably well
     * (approximately the 96th percentile of the Normal distribution).
     */
    class RelaxedMonotonicityTracker implements ScoreTracker {
        private static final double SIGMA_FACTOR = 1.75;

        private static final int BASE_RECENT_SCORES_SIZE = 200;

        // a sliding window of recent scores
        private double[] recentScores;
        private int recentScoresSize;
        private int recentEntryIndex;

        // Heap of the best scores seen so far
        BoundedLongHeap bestScores;

        // observation count
        private int observationCount;

        // the sample mean
        private double mean;

        // the sample variance multiplied by n-1
        private double dSquared;

        /**
         * Constructor
         * @param bestScoresTracked the number of tracked scores used to estimate if we are unlikely to improve
         *                          the results anymore. An empirical rule of thumb is bestScoresTracked=rerankK.
         */
        RelaxedMonotonicityTracker(int bestScoresTracked) {
            this.recentScoresSize = getRecentScoresSize(bestScoresTracked);
            this.recentScores = new double[this.recentScoresSize];
            this.bestScores = new BoundedLongHeap(bestScoresTracked);
            this.observationCount = 0;
            this.mean = 0;
            this.dSquared = 0;
        }

        RelaxedMonotonicityTracker() {
            this(100);
        }

        private static int getRecentScoresSize(int bestScoresTracked) {
            // A quick empirical study yields that the number of recent scores
            // that we need to consider grows by a factor of ~sqrt(bestScoresTracked / 2)
            int factor = (int) Math.round(Math.sqrt(bestScoresTracked / 2.0));
            return BASE_RECENT_SCORES_SIZE * factor;
        }

        void reset(int bestScoresTracked) {
            this.recentScoresSize = getRecentScoresSize(bestScoresTracked);
            if (this.recentScoresSize > recentScores.length) {
                recentScores = ArrayUtil.grow(recentScores, this.recentScoresSize);
            }
            this.bestScores.clear();
            this.observationCount = 0;
            this.mean = 0;
            this.dSquared = 0;
        }

        @Override
        public void track(float score) {
            bestScores.push(floatToSortableInt(score));
            observationCount++;

            // The updates of the sufficient statistics follow
            // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online
            // and
            // https://nestedsoftware.com/2019/09/26/incremental-average-and-standard-deviation-with-sliding-window-470k.176143.html
            if (observationCount <= this.recentScoresSize) {
                // if the buffer is not full yet, use standard Welford method
                var meanDelta = (score - this.mean) / observationCount;
                var newMean = this.mean + meanDelta;

                var dSquaredDelta = ((score - newMean) * (score - this.mean));
                var newDSquared = this.dSquared + dSquaredDelta;

                this.mean = newMean;
                this.dSquared = newDSquared;
            } else {
                // once the buffer is full, adjust Welford method for window size
                var oldScore = recentScores[recentEntryIndex];
                var meanDelta = (score - oldScore) / this.recentScoresSize;
                var newMean = this.mean + meanDelta;

                var dSquaredDelta = ((score - oldScore) * (score - newMean + oldScore - this.mean));
                var newDSquared = this.dSquared + dSquaredDelta;

                this.mean = newMean;
                this.dSquared = newDSquared;
            }
            recentScores[recentEntryIndex] = score;
            recentEntryIndex = (recentEntryIndex + 1) % this.recentScoresSize;
        }

        @Override
        public boolean shouldStop() {
            // don't stop if we don't have enough data points
            if (observationCount < this.recentScoresSize) {
                return false;
            }

            // We're in phase 2 if the q-th percentile of the recent scores evaluated,
            //     mean + SIGMA_FACTOR * sqrt(variance),
            // is lower than the worst of the best scores seen.
            // (paper suggests using the median of recent scores, but experimentally that is too prone to false positives)
            double std = Math.sqrt(this.dSquared / (this.recentScoresSize - 1));
            double windowPercentile = this.mean + SIGMA_FACTOR * std;
            double worstBestScore = sortableIntToFloat((int) bestScores.top());
            return windowPercentile < worstBestScore;
        }
    }
}
