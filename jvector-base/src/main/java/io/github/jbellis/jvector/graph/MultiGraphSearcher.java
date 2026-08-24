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

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Searches multiple independent {@link ImmutableGraphIndex} shards for a single query and merges
 * the results into one global top-K, ranked by score. Intended for callers that split one logical
 * dataset across several physical indexes (e.g. one per segment/sstable) and want a single ranked
 * answer across all of them.
 * <p>
 * This is the Phase 1 implementation described in {@code docs/multi-index-search.md}: every shard
 * is searched independently for its own full top-{@code topK}, and results are merged with no
 * proportional sizing or resume-based refill. This is correct (not just a heuristic) because any
 * vector that belongs in the global top-K must also be in its own shard's local top-K -- removing
 * other shards' vectors from consideration can only improve or maintain a candidate's local rank.
 * Later phases may add proportional per-shard sizing and adaptive resume to reduce redundant work,
 * without changing this correctness property.
 * <p>
 * Not safe for concurrent use by multiple threads -- like {@link GraphSearcher} (the {@link Searcher}
 * implementation this class currently composes), scratch state is reused across calls to {@link #search}.
 */
@Experimental
public class MultiGraphSearcher implements AutoCloseable {
    private final List<Searcher> searchers;

    /**
     * @param shards the graph indexes to search, in the order that
     *               {@link ShardedSearchResult.NodeScore#shardIndex} will refer to them
     */
    public MultiGraphSearcher(List<? extends ImmutableGraphIndex> shards) {
        if (shards.isEmpty()) {
            throw new IllegalArgumentException("MultiGraphSearcher requires at least one shard");
        }
        var searchers = new ArrayList<Searcher>(shards.size());
        for (var shard : shards) {
            searchers.add(shard.searcher());
        }
        this.searchers = searchers;
    }

    /**
     * @return the number of shards this searcher was constructed with
     */
    public int shardCount() {
        return searchers.size();
    }

    /**
     * Searches every shard and returns the merged global top-{@code topK}, best first.
     *
     * @param scoreProviders     one {@link SearchScoreProvider} per shard, in shard order. Each
     *                           closes over the same query but that shard's own vectors/compressor.
     * @param acceptOrdsPerShard one {@link Bits} per shard, in shard order, using ordinals local to
     *                           that shard. Use {@link Bits#ALL} for shards with no per-query filter.
     * @param topK               desired global result count
     * @param rerankK            rerank budget, applied to every shard individually (not divided
     *                           across shards in this phase)
     * @return the merged results, plus metrics summed across all shards
     */
    public ShardedSearchResult search(List<SearchScoreProvider> scoreProviders,
                                       List<Bits> acceptOrdsPerShard,
                                       int topK,
                                       int rerankK)
    {
        if (scoreProviders.size() != searchers.size() || acceptOrdsPerShard.size() != searchers.size()) {
            throw new IllegalArgumentException(String.format(
                    "Expected %d score providers and accept-ords (one per shard), got %d and %d",
                    searchers.size(), scoreProviders.size(), acceptOrdsPerShard.size()));
        }

        var candidates = new ArrayList<ShardedSearchResult.NodeScore>();
        int visitedCount = 0;
        int expandedCount = 0;
        int rerankedCount = 0;
        for (int i = 0; i < searchers.size(); i++) {
            var result = searchers.get(i).search(scoreProviders.get(i), topK, rerankK, 0.0f, 0.0f, acceptOrdsPerShard.get(i));
            visitedCount += result.getVisitedCount();
            expandedCount += result.getExpandedCount();
            rerankedCount += result.getRerankedCount();

            int shardIndex = i;
            for (var nodeScore : result.getNodes()) {
                candidates.add(new ShardedSearchResult.NodeScore(shardIndex, nodeScore.node, nodeScore.score));
            }
        }

        Collections.sort(candidates);
        int resultSize = Math.min(topK, candidates.size());
        var nodes = candidates.subList(0, resultSize).toArray(new ShardedSearchResult.NodeScore[0]);
        return new ShardedSearchResult(nodes, visitedCount, expandedCount, rerankedCount, 1);
    }

    /**
     * Convenience overload using {@link Bits#ALL} for every shard.
     */
    public ShardedSearchResult search(List<SearchScoreProvider> scoreProviders, int topK, int rerankK) {
        return search(scoreProviders, Collections.nCopies(searchers.size(), Bits.ALL), topK, rerankK);
    }

    /**
     * Closes every shard's underlying {@link GraphSearcher}. If more than one fails to close, the
     * first exception is thrown and the rest are attached as suppressed exceptions; every searcher
     * is given a chance to close regardless of earlier failures.
     */
    @Override
    public void close() throws IOException {
        IOException firstFailure = null;
        for (var searcher : searchers) {
            try {
                searcher.close();
            } catch (IOException e) {
                if (firstFailure == null) {
                    firstFailure = e;
                } else {
                    firstFailure.addSuppressed(e);
                }
            }
        }
        if (firstFailure != null) {
            throw firstFailure;
        }
    }
}
