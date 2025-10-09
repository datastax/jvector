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

import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/**
 * Provides an API for encapsulating similarity to another node or vector. Used both for
 * building the graph (as part of NodeSimilarity) or for searching it (used standalone,
 * with a reference to the query vector).
 * <p>
 * ExactScoreFunction and ApproximateScoreFunction are provided for convenience so they
 * can be defined as a simple lambda.
 */
public interface ScoreFunction {
    /** Vector type support for creating vector instances. */
    VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    /**
     * Returns true if the ScoreFunction returns exact, full-resolution scores.
     *
     * @return true if the ScoreFunction returns exact, full-resolution scores
     */
    boolean isExact();

    /**
     * Returns the similarity to one other node.
     *
     * @param node2 the node to compute similarity to
     * @return the similarity to one other node
     */
    float similarityTo(int node2);

    /**
     * Returns the similarity to all of the nodes that the given node has an edge towards.
     * Used when expanding the neighbors of a search candidate for bulk similarity computations.
     *
     * @param node2 the node whose neighbors should be scored
     * @return a vector containing similarity scores to each neighbor
     * @throws UnsupportedOperationException if bulk similarity is not supported
     */
    default VectorFloat<?> edgeLoadingSimilarityTo(int node2) {
        throw new UnsupportedOperationException("bulk similarity not supported");
    }

    /**
     * Returns true if edge loading similarity is supported (i.e., if edgeLoadingSimilarityTo can be called).
     *
     * @return true if `edgeLoadingSimilarityTo` is supported
     */
    default boolean supportsEdgeLoadingSimilarity() {
        return false;
    }

    /**
     * A score function that returns exact, full-resolution similarity scores.
     */
    interface ExactScoreFunction extends ScoreFunction {
        /**
         * Returns true to indicate this is an exact score function.
         *
         * @return true
         */
        default boolean isExact() {
            return true;
        }
    }

    /**
     * A score function that returns approximate similarity scores, potentially using compressed vectors.
     */
    interface ApproximateScoreFunction extends ScoreFunction {
        /**
         * Returns false to indicate this is an approximate score function.
         *
         * @return false
         */
        default boolean isExact() {
            return false;
        }
    }
}
