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
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/**
 * Provides an API for encapsulating similarity to another node or vector.  Used both for
 * building the graph (as part of NodeSimilarity) or for searching it (used standalone,
 * with a reference to the query vector).
 * <p>
 * ExactScoreFunction and ApproximateScoreFunction are provided for convenience so they
 * can be defined as a simple lambda.
 */
public interface ScoreFunction {
    VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    /**
     * @return true if the ScoreFunction returns exact, full-resolution scores
     */
    boolean isExact();

    /**
     * @return the similarity to one other node
     */
    float similarityTo(int node2);

    /**
     * Computes the similarity to the neighborIndex-th neighbor of origin.
     * Before calling this function, enableSimilarityToNeighbors must be called first with the same origin.
     * This function only works if it is called for the same origin node multiple times.
     * Used when expanding the neighbors of a search candidate.
     * @param origin the node we are expanding
     * @param neighborIndex the index of the neighbor we are scoring, a number between 0 and the number of neighbors of the origin node.
     * @return the score
     */
    default float similarityToNeighbor(int origin, int neighborIndex) {
        throw new UnsupportedOperationException("bulk similarity not supported");
    }

    /**
     * Load the corresponding data so that similarityToNeighbor can be used with the neighbors of the origin node.
     */
    default void enableSimilarityToNeighbors(int origin) {}

    /**
     * @return true if `similarityToNeighbor` is supported
     */
    default boolean supportsSimilarityToNeighbors() {
        return false;
    }

    /**
     * Computes the similarity to each of {@code nodes[0..count)}: {@code out[i] = similarityTo(nodes[i])}.
     * <p>
     * Used when expanding the neighbors of a search candidate. Implementations that can score several
     * nodes more cheaply together (e.g. one native call over contiguous vectors) override this and
     * {@link #supportsSimilarityToBatch()}. The default simply loops over {@link #similarityTo(int)}.
     *
     * @param nodes the nodes to score
     * @param count how many entries of {@code nodes} to score
     * @param out   receives the scores; must have room for {@code count} entries
     */
    default void similarityToBatch(int[] nodes, int count, float[] out) {
        for (int i = 0; i < count; i++) {
            out[i] = similarityTo(nodes[i]);
        }
    }

    /**
     * @return true if {@link #similarityToBatch} is meaningfully cheaper than calling
     * {@link #similarityTo(int)} in a loop, so callers should gather neighbors before scoring them
     */
    default boolean supportsSimilarityToBatch() {
        return false;
    }


    interface ExactScoreFunction extends ScoreFunction {
        default boolean isExact() {
            return true;
        }
    }

    interface ApproximateScoreFunction extends ScoreFunction {
        default boolean isExact() {
            return false;
        }
    }
}
