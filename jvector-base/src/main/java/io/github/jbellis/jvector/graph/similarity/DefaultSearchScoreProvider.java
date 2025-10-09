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

    /**
     * Constructs a DefaultSearchScoreProvider with only a primary scoring function.
     * No reranking is performed.
     *
     * @param scoreFunction the primary, fast scoring function
     */
    public DefaultSearchScoreProvider(ScoreFunction scoreFunction) {
        this(scoreFunction, null);
    }

    /**
     * Constructs a DefaultSearchScoreProvider with a primary scoring function and optional reranking.
     * Generally, reranker will be null iff scoreFunction is an ExactScoreFunction. However,
     * it is allowed, and sometimes useful, to only perform approximate scoring without reranking.
     *
     * Most often it will be convenient to get the reranker either using {@code RandomAccessVectorValues.rerankerFor}
     * or {@code ScoringView.rerankerFor}.
     *
     * @param scoreFunction the primary, fast scoring function
     * @param reranker optional reranking function (may be null)
     */
    public DefaultSearchScoreProvider(ScoreFunction scoreFunction, ScoreFunction.ExactScoreFunction reranker) {
        assert scoreFunction != null;
        this.scoreFunction = scoreFunction;
        this.reranker = reranker;
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
     * Creates a SearchScoreProvider for a single-pass search based on exact similarity.
     * Generally only suitable when your RandomAccessVectorValues is entirely in-memory,
     * e.g. during construction.
     *
     * @param v the query vector
     * @param vsf the vector similarity function to use
     * @param ravv the random access vector values to search
     * @return a DefaultSearchScoreProvider configured for exact search
     */
    public static DefaultSearchScoreProvider exact(VectorFloat<?> v, VectorSimilarityFunction vsf, RandomAccessVectorValues ravv) {
        // don't use ESF.reranker, we need thread safety here
        var sf = new ScoreFunction.ExactScoreFunction() {
            @Override
            public float similarityTo(int node2) {
                return vsf.compare(v, ravv.getVector(node2));
            }
        };
        return new DefaultSearchScoreProvider(sf);
    }
}