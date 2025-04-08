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
    private final int rerankedCount;
    private final float worstApproximateScoreInTopK;

    public SearchResult(NodeScore[] nodes, int visitedCount, int rerankedCount, float worstApproximateScoreInTopK) {
        this.nodes = nodes;
        this.visitedCount = visitedCount;
        this.rerankedCount = rerankedCount;
        this.worstApproximateScoreInTopK = worstApproximateScoreInTopK;
    }

    /**
     * @return the closest neighbors discovered by the search, sorted best-first
     */
    public NodeScore[] getNodes() {
        return nodes;
    }

    /**
     * @return the total number of graph nodes visited while performing the search
     */
    public int getVisitedCount() {
        return visitedCount;
    }

    /**
     * @return the number of nodes that were reranked during the search
     */
    public int getRerankedCount() {
        return rerankedCount;
    }

    /**
     * @return the worst approximate score of the top K nodes in the search result.  Useful
     * for passing to rerankFloor during search across multiple indexes.  Will be
     * Float.POSITIVE_INFINITY if no reranking was performed or no results were found.
     */
    public float getWorstApproximateScoreInTopK() {
        return worstApproximateScoreInTopK;
    }

    public static final class NodeScore implements Comparable<NodeScore> {
        public final int node;
        public final float score;

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
