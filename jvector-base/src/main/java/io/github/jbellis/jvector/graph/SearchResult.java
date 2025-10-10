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
    private final NodeScore[] nodes;
    private final int visitedCount;
    private final int expandedCount;
    private final int expandedCountL0;
    private final int rerankedCount;
    private final float worstApproximateScoreInTopK;

    /**
     * Creates a new SearchResult containing the results and metrics from an ANN search operation.
     * @param nodes the top scoring neighbors discovered during the search, sorted best-first
     * @param visitedCount the total number of graph nodes visited during the search
     * @param expandedCount the total number of graph nodes expanded (had their neighbors examined)
     * @param expandedCountL0 the number of nodes expanded in the base layer (layer 0)
     * @param rerankedCount the number of nodes that were reranked with exact scores
     * @param worstApproximateScoreInTopK the worst approximate score among the top K results
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
     * @return the closest neighbors discovered by the search, sorted best-first
     */
    public NodeScore[] getNodes() {
        return nodes;
    }

    /**
     * Returns the total number of graph nodes visited during the search operation.
     * @return the total number of graph nodes visited while performing the search
     */
    public int getVisitedCount() {
        return visitedCount;
    }

    /**
     * Returns the total number of graph nodes that had their neighbors examined during search.
     * @return the total number of graph nodes expanded while performing the search
     */
    public int getExpandedCount() {
        return expandedCount;
    }

    /**
     * Returns the number of graph nodes expanded specifically in the base layer.
     * @return the number of graph nodes expanded while performing the search in the base layer
     */
    public int getExpandedCountBaseLayer() {
        return expandedCountL0;
    }

    /**
     * Returns the count of nodes that were reranked with exact similarity scores.
     * @return the number of nodes that were reranked during the search
     */
    public int getRerankedCount() {
        return rerankedCount;
    }

    /**
     * Returns the worst approximate score among the top K results, useful for distributed search.
     * @return the worst approximate score of the top K nodes in the search result.  Useful
     * for passing to rerankFloor during search across multiple indexes.  Will be
     * Float.POSITIVE_INFINITY if no reranking was performed or no results were found.
     */
    public float getWorstApproximateScoreInTopK() {
        return worstApproximateScoreInTopK;
    }

    /**
     * Represents a graph node and its similarity score, used to store search results.
     */
    public static final class NodeScore implements Comparable<NodeScore> {
        /** The ordinal ID of the graph node */
        public final int node;
        /** The similarity score for this node */
        public final float score;

        /**
         * Creates a new NodeScore pairing a node with its similarity score.
         * @param node the ordinal ID of the graph node
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
