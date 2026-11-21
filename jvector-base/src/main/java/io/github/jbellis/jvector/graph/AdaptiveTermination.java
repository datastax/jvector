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

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.util.Objects;

/**
 * The distance-based stopping rule of Adaptive Beam Search (Al-Jazzazi et al., "Distance Adaptive Beam Search for
 * Provably Accurate Graph-Based Nearest Neighbor Search", <a href="https://arxiv.org/abs/2505.15636">arXiv:2505.15636</a>),
 * for use with {@link GraphSearcher#search(io.github.jbellis.jvector.graph.similarity.SearchScoreProvider, int, int, float, float, io.github.jbellis.jvector.util.Bits, AdaptiveTermination)}.
 * <p>
 * Beam search stops once {@code rerankK} results are closer to the query than the best candidate left to expand,
 * so {@code rerankK} doubles as the beam width and every query does about the same amount of work. This rule instead
 * stops once {@code rerankK} results are closer than the best remaining candidate by a factor of {@code 1 + gamma}:
 * <pre>
 *   (1 + gamma) * distance(query, k-th best result) &lt;= distance(query, best remaining candidate)
 * </pre>
 * A query whose neighborhood is well separated stops early, and one surrounded by many near-ties keeps going, so for
 * the same mean work the hard queries get more of it: recall per node visited improves, while the slowest queries
 * take longer than with a beam of the same mean cost. {@code gamma = 0} is greedy search with a beam of
 * {@code rerankK}; a larger {@code gamma} visits more nodes and finds better results. Search order is unchanged: a
 * search with a larger {@code gamma} expands the same nodes as one with a smaller {@code gamma}, then more.
 * <p>
 * Scores are turned back into distances according to the similarity function, which must be the one the scores
 * were computed with (approximate scores, such as PQ, estimate the same similarity and work too):
 * <ul>
 *   <li>{@link VectorSimilarityFunction#EUCLIDEAN}: the score is {@code 1 / (1 + d^2)}, so {@code d^2 = 1 / score - 1}.</li>
 *   <li>{@link VectorSimilarityFunction#COSINE}: the score is {@code (1 + cos) / 2}, and the distance is the
 *   Euclidean distance between the normalized vectors, {@code d^2 = 2 - 2 cos = 4 (1 - score)}.</li>
 *   <li>{@link VectorSimilarityFunction#DOT_PRODUCT}: as for COSINE, which is exact for unit-length vectors. For
 *   vectors that are not unit length the dot product is not a distance and this rule has no guarantee; a score above
 *   the one of an exact match makes the rule stop as soon as {@code rerankK} results are found.</li>
 * </ul>
 * Instances are immutable and may be shared between threads and searches.
 */
@Experimental
public final class AdaptiveTermination {
    private final VectorSimilarityFunction similarityFunction;
    private final float gamma;
    /** (1 + gamma)^2: the rule is applied to squared distances */
    private final double squaredFactor;

    private AdaptiveTermination(VectorSimilarityFunction similarityFunction, float gamma) {
        this.similarityFunction = similarityFunction;
        this.gamma = gamma;
        this.squaredFactor = (1.0 + gamma) * (1.0 + gamma);
    }

    /**
     * @param similarityFunction the similarity function the search scores are computed with
     * @param gamma how much farther than the k-th best result the best remaining candidate must be before the search
     *              stops, as a fraction of that result's distance; 0 or more. Useful values are small and depend on
     *              the data, because in many dimensions most distances are within a few percent of each other: on
     *              1024-dimension text embeddings, 0.02 to 0.07 spans recall@10 from 0.95 to 0.996. Tune it as you
     *              would a beam width; the work grows quickly with it.
     */
    public static AdaptiveTermination of(VectorSimilarityFunction similarityFunction, float gamma) {
        Objects.requireNonNull(similarityFunction, "similarityFunction");
        if (!(gamma >= 0) || Float.isInfinite(gamma)) {
            throw new IllegalArgumentException("gamma must be finite and >= 0, not " + gamma);
        }
        return new AdaptiveTermination(similarityFunction, gamma);
    }

    public VectorSimilarityFunction getSimilarityFunction() {
        return similarityFunction;
    }

    public float getGamma() {
        return gamma;
    }

    /**
     * Whether a search whose k-th best result has score {@code kthBestScore} can stop instead of expanding the
     * candidate with score {@code candidateScore}, the best one left. Scores are those of the similarity function.
     */
    public boolean shouldStop(float candidateScore, float kthBestScore) {
        return squaredDistance(candidateScore) >= squaredFactor * squaredDistance(kthBestScore);
    }

    /** the squared distance a score stands for, up to a constant factor that the ratio of the rule cancels */
    private double squaredDistance(float score) {
        switch (similarityFunction) {
            case EUCLIDEAN:
                return 1.0 / score - 1.0;
            case COSINE:
            case DOT_PRODUCT:
                return 1.0 - score;
            default:
                throw new AssertionError(similarityFunction);
        }
    }

    @Override
    public String toString() {
        return "AdaptiveTermination(" + similarityFunction + ", gamma=" + gamma + ")";
    }
}
