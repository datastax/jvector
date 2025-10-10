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

import java.util.Arrays;
import java.util.Objects;

/**
 * Container class for results of an ANN search, along with associated metrics about the behavior of the search.
 */
public final class SearchResult {
    /** The closest neighbors discovered by the search, sorted best-first. */
    private final NodeScore[] nodes;
    /** The total number of graph nodes visited while performing the search. */
    private final int visitedCount;
    /** The total number of graph nodes expanded while performing the search. */
    private final int expandedCount;
    /** The number of graph nodes expanded while performing the search in the base layer. */
    private final int expandedCountL0;
    /** The number of nodes that were reranked during the search. */
    private final int rerankedCount;
    /** The worst approximate score of the top K nodes in the search result. */
    private final float worstApproximateScoreInTopK;

    /**
     * Constructs a SearchResult with the specified search results and metrics.
     *
     * @param nodes the closest neighbors discovered by the search, sorted best-first
     * @param visitedCount the total number of graph nodes visited while performing the search
     * @param expandedCount the total number of graph nodes expanded while performing the search
     * @param expandedCountL0 the number of graph nodes expanded in the base layer
     * @param rerankedCount the number of nodes that were reranked during the search
     * @param worstApproximateScoreInTopK the worst approximate score in the top K results, or Float.POSITIVE_INFINITY if no reranking occurred
     */
    public SearchResult(NodeScore[] nodes, int visitedCount, int expandedCount, int expandedCountL0, int rerankedCount, float worstApproximateScoreInTopK) {
        this.nodes = nodes;
        this.visitedCount = visitedCount;
        this.expandedCount = expandedCount;
        this.expandedCountL0 = expandedCountL0;
        this.rerankedCount = rerankedCount;
        this.worstApproximateScoreInTopK = worstApproximateScoreInTopK;
    }

    /**
     * Returns the closest neighbors discovered by the search.
     *
     * @return the closest neighbors discovered by the search, sorted best-first
     */
    public NodeScore[] getNodes() {
        return nodes;
    }

    /**
     * Returns the total number of graph nodes visited during the search.
     *
     * @return the total number of graph nodes visited while performing the search
     */
    public int getVisitedCount() {
        return visitedCount;
    }

    /**
     * Returns the total number of graph nodes expanded during the search.
     *
     * @return the total number of graph nodes expanded while performing the search
     */
    public int getExpandedCount() {
        return expandedCount;
    }

    /**
     * Returns the number of graph nodes expanded in the base layer during the search.
     *
     * @return the number of graph nodes expanded while performing the search in the base layer
     */
    public int getExpandedCountBaseLayer() {
        return expandedCountL0;
    }

    /**
     * Returns the number of nodes that were reranked during the search.
     *
     * @return the number of nodes that were reranked during the search
     */
    public int getRerankedCount() {
        return rerankedCount;
    }

    /**
     * Returns the worst approximate score of the top K nodes in the search result.
     * Useful for passing to rerankFloor during search across multiple indexes.
     *
     * @return the worst approximate score of the top K nodes in the search result.  Useful
     * for passing to rerankFloor during search across multiple indexes.  Will be
     * Float.POSITIVE_INFINITY if no reranking was performed or no results were found.
     */
    public float getWorstApproximateScoreInTopK() {
        return worstApproximateScoreInTopK;
    }

    /**
     * Represents a node and its associated similarity score in a search result.
     */
    public static final class NodeScore implements Comparable<NodeScore> {
        /** The node identifier. */
        public final int node;
        /** The similarity score for this node. */
        public final float score;

        /**
         * Constructs a NodeScore with the specified node ID and score.
         *
         * @param node the node identifier
         * @param score the similarity score for this node
         */
        public NodeScore(int node, float score) {
            this.node = node;
            this.score = score;
        }

        @Override
        public String toString() {
            return String.format("NodeScore(%d, %s)", node, score);
        }

        @Override
        public int compareTo(NodeScore o) {
            // Sort by score in descending order (highest score first)
            int scoreCompare = Float.compare(o.score, this.score);
            // If scores are equal, break ties using node id (ascending order)
            return scoreCompare != 0 ? scoreCompare : Integer.compare(node, o.node);
        }

        @Override
        public boolean equals(Object o) {
            if (o == null || getClass() != o.getClass()) return false;
            NodeScore nodeScore = (NodeScore) o;
            return node == nodeScore.node && Float.compare(score, nodeScore.score) == 0;
        }

        @Override
        public int hashCode() {
            return Objects.hash(node, score);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || getClass() != o.getClass()) return false;
        SearchResult that = (SearchResult) o;
        return visitedCount == that.visitedCount && rerankedCount == that.rerankedCount && Float.compare(worstApproximateScoreInTopK, that.worstApproximateScoreInTopK) == 0 && Objects.deepEquals(nodes, that.nodes);
    }

    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(nodes), visitedCount, rerankedCount, worstApproximateScoreInTopK);
    }
}
