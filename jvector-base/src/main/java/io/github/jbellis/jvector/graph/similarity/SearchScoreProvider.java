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

/** Encapsulates comparing node distances to a specific vector for GraphSearcher. */
public interface SearchScoreProvider {

    /**
     * Returns the primary score function for fast approximate scoring.
     *
     * @return the score function
     */
    ScoreFunction scoreFunction();

    /**
     * Returns the optional reranking function for more accurate scoring.
     *
     * @return the reranker, or null if no reranking is performed
     */
    ScoreFunction.ExactScoreFunction reranker();

    /**
     * Returns the exact score function, either the primary function if it is exact, or the reranker.
     *
     * @return the exact score function
     */
    ScoreFunction.ExactScoreFunction exactScoreFunction();
}