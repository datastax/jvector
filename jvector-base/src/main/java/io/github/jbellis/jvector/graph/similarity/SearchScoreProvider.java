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

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

/** Encapsulates comparing node distances to a specific vector for GraphSearcher. */
public interface SearchScoreProvider {

    ScoreFunction scoreFunction();

    ScoreFunction.ExactScoreFunction reranker();

    ScoreFunction.ExactScoreFunction exactScoreFunction();

    /**
     * The similarity function whose scores {@link #scoreFunction} returns, when top-K searches with this provider may
     * stop adaptively; null keeps the standard rule. Adaptive stopping is experimental and applies only where the
     * searcher enables it ({@code GraphSearcher.useAdaptiveTermination}). Score functions that require adaptive
     * stopping ({@link ScoreFunction#requiresAdaptiveTermination}) score DOT_PRODUCT and stop adaptively regardless.
     */
    default VectorSimilarityFunction adaptiveSimilarity() { return null; }

    /**
     * For DOT_PRODUCT scores, an upper bound on the dot product between the query and any vector the search can
     * reach; NaN when unknown, which keeps the standard rule. By default the score function's
     * {@link ScoreFunction#dotProductBound}.
     */
    default float dotProductBound() { return scoreFunction().dotProductBound(); }
}