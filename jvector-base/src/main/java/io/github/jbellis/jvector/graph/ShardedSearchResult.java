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
 * Container class for the results of a {@link MultiGraphSearcher} search: the merged, global
 * top-K across all searched shards, along with aggregated metrics about the underlying per-shard
 * searches.
 */
public final class ShardedSearchResult {
    private final NodeScore[] nodes;
    private final int visitedCount;
    private final int expandedCount;
    private final int rerankedCount;
    private final int roundsUsed;

    public ShardedSearchResult(NodeScore[] nodes, int visitedCount, int expandedCount, int rerankedCount, int roundsUsed) {
        this.nodes = nodes;
        this.visitedCount = visitedCount;
        this.expandedCount = expandedCount;
        this.rerankedCount = rerankedCount;
        this.roundsUsed = roundsUsed;
    }

    /**
     * @return the closest neighbors discovered across all shards, sorted best-first
     */
    public NodeScore[] getNodes() {
        return nodes;
    }

    /**
     * @return the total number of graph nodes visited, summed across all shards
     */
    public int getVisitedCount() {
        return visitedCount;
    }

    /**
     * @return the total number of graph nodes expanded, summed across all shards
     */
    public int getExpandedCount() {
        return expandedCount;
    }

    /**
     * @return the total number of nodes reranked, summed across all shards
     */
    public int getRerankedCount() {
        return rerankedCount;
    }

    /**
     * @return the number of search rounds performed. Phase 1 always performs exactly 1 round
     * (no resume-based refill yet); this field exists so callers don't need to change once
     * later phases add adaptive resume.
     */
    public int getRoundsUsed() {
        return roundsUsed;
    }

    /**
     * A single result, tagged with which shard it came from. {@code node} is an ordinal local to
     * that shard -- ordinals are not comparable or unique across shards, so callers must use
     * {@code shardIndex} to know which shard's vectors/row-mapping {@code node} refers to.
     */
    public static final class NodeScore implements Comparable<NodeScore> {
        /** Index into the shard list passed to {@link MultiGraphSearcher}'s constructor. */
        public final int shardIndex;
        /** Ordinal within shard {@code shardIndex}. Not meaningful outside that shard. */
        public final int node;
        public final float score;

        public NodeScore(int shardIndex, int node, float score) {
            this.shardIndex = shardIndex;
            this.node = node;
            this.score = score;
        }

        @Override
        public String toString() {
            return String.format("NodeScore(shard=%d, node=%d, %s)", shardIndex, node, score);
        }

        @Override
        public int compareTo(NodeScore o) {
            // Sort by score in descending order (highest score first)
            int scoreCompare = Float.compare(o.score, this.score);
            if (scoreCompare != 0) {
                return scoreCompare;
            }
            // Break ties deterministically using shard index, then node id (ascending order)
            int shardCompare = Integer.compare(shardIndex, o.shardIndex);
            return shardCompare != 0 ? shardCompare : Integer.compare(node, o.node);
        }

        @Override
        public boolean equals(Object o) {
            if (o == null || getClass() != o.getClass()) return false;
            NodeScore nodeScore = (NodeScore) o;
            return shardIndex == nodeScore.shardIndex && node == nodeScore.node && Float.compare(score, nodeScore.score) == 0;
        }

        @Override
        public int hashCode() {
            return Objects.hash(shardIndex, node, score);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || getClass() != o.getClass()) return false;
        ShardedSearchResult that = (ShardedSearchResult) o;
        return visitedCount == that.visitedCount && expandedCount == that.expandedCount
                && rerankedCount == that.rerankedCount && roundsUsed == that.roundsUsed
                && Objects.deepEquals(nodes, that.nodes);
    }

    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(nodes), visitedCount, expandedCount, rerankedCount, roundsUsed);
    }
}
