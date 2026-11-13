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

package io.github.jbellis.jvector.graph.similarity;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/** Encapsulates comparing node distances to a specific vector for GraphSearcher. */
public final class DefaultSearchScoreProvider implements SearchScoreProvider {
    private final ScoreFunction scoreFunction;
    private final ScoreFunction.ExactScoreFunction reranker;
    private final VectorSimilarityFunction adaptiveSimilarity;
    private final float dotProductBound;

    /**
     * @param scoreFunction the primary, fast scoring function
     * <p>
     * No reranking is performed.
     */
    public DefaultSearchScoreProvider(ScoreFunction scoreFunction) {
        this(scoreFunction, null);
    }

    /**
     * @param scoreFunction the primary, fast scoring function
     * @param reranker optional reranking function
     * Generally, reranker will be null iff scoreFunction is an ExactScoreFunction.  However,
     * it is allowed, and sometimes useful, to only perform approximate scoring without reranking.
     * <p>
     * Most often it will be convenient to get the reranker either using `RandomAccessVectorValues.rerankerFor`
     * or `ScoringView.rerankerFor`.
     * <p>
     * Searches with this provider stop by the standard rule, since it cannot tell what its scores measure; use
     * {@link #forSimilarity} to let top-K searches stop adaptively where the similarity function supports it.
     */
    public DefaultSearchScoreProvider(ScoreFunction scoreFunction, ScoreFunction.ExactScoreFunction reranker) {
        this(scoreFunction, reranker, null);
    }

    /**
     * @param adaptiveSimilarity the similarity function {@code scoreFunction} scores by, letting top-K searches stop
     *                           adaptively; null keeps the standard rule
     */
    public DefaultSearchScoreProvider(ScoreFunction scoreFunction, ScoreFunction.ExactScoreFunction reranker,
                                      VectorSimilarityFunction adaptiveSimilarity) {
        this(scoreFunction, reranker, adaptiveSimilarity, Float.NaN);
    }

    /**
     * @param dotProductBound for DOT_PRODUCT scores, an upper bound on the dot product between the query and any
     *                        stored vector, used in place of the score function's own; NaN defers to it
     */
    public DefaultSearchScoreProvider(ScoreFunction scoreFunction, ScoreFunction.ExactScoreFunction reranker,
                                      VectorSimilarityFunction adaptiveSimilarity, float dotProductBound) {
        assert scoreFunction != null;
        this.scoreFunction = scoreFunction;
        this.reranker = reranker;
        this.adaptiveSimilarity = adaptiveSimilarity;
        this.dotProductBound = dotProductBound;
    }

    /**
     * A provider for scores of {@code similarity} whose top-K searches stop adaptively. Dot-product searches need
     * the score function's {@link ScoreFunction#dotProductBound} and keep the standard rule without one.
     * {@link io.github.jbellis.jvector.graph.GraphSearcher#useAdaptiveTermination} turns adaptive stopping off.
     */
    public static DefaultSearchScoreProvider forSimilarity(ScoreFunction scoreFunction,
                                                           ScoreFunction.ExactScoreFunction reranker,
                                                           VectorSimilarityFunction similarity) {
        return new DefaultSearchScoreProvider(scoreFunction, reranker, similarity);
    }

    @Override
    public VectorSimilarityFunction adaptiveSimilarity() { return adaptiveSimilarity; }

    @Override
    public float dotProductBound() {
        return Float.isNaN(dotProductBound) ? scoreFunction.dotProductBound() : dotProductBound;
    }

    public ScoreFunction scoreFunction() {
        return scoreFunction;
    }

    public ScoreFunction.ExactScoreFunction reranker() {
        return reranker;
    }

    public ScoreFunction.ExactScoreFunction exactScoreFunction() {
        return scoreFunction.isExact()
                ? (ScoreFunction.ExactScoreFunction) scoreFunction
                : reranker;
    }

    /**
     * A SearchScoreProvider for a single-pass search based on exact similarity, stopping adaptively where
     * {@code vsf} supports it (see {@link #forSimilarity}).
     * Generally only suitable when your RandomAccessVectorValues is entirely in-memory,
     * e.g. during construction.
     */
    public static DefaultSearchScoreProvider exact(VectorFloat<?> v, VectorSimilarityFunction vsf, RandomAccessVectorValues ravv) {
        return forSimilarity(exactScoreFunction(v, vsf, ravv), null, vsf);
    }

    /** Exact scores of {@code v} against {@code ravv}, safe for concurrent use. */
    static ScoreFunction.ExactScoreFunction exactScoreFunction(VectorFloat<?> v, VectorSimilarityFunction vsf,
                                                               RandomAccessVectorValues ravv) {
        // don't use ESF.reranker, we need thread safety here
        return new ScoreFunction.ExactScoreFunction() {
            @Override
            public float similarityTo(int node2) {
                return vsf.compare(v, ravv.getVector(node2));
            }
        };
    }

    /**
     * A SearchScoreProvider for a single-pass search based on exact similarity.
     * Generally only suitable when your RandomAccessVectorValues is entirely in-memory,
     * e.g. during construction.
     */
    public static DefaultSearchScoreProvider exact(VectorFloat<?> v, int[] graphToRavvOrdMap ,VectorSimilarityFunction vsf, RandomAccessVectorValues ravv) {
        // don't use ESF.reranker, we need thread safety here
        var sf = new ScoreFunction.ExactScoreFunction() {
            @Override
            public float similarityTo(int node2) {
                return vsf.compare(v, ravv.getVector(graphToRavvOrdMap[node2]));
            }
        };
        return forSimilarity(sf, null, vsf);
    }
}