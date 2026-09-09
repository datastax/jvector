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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.function.IntFunction;

/**
 * Searches multiple independent {@link ImmutableGraphIndex} shards for a single query and merges
 * the results into one global top-K, ranked by score. Intended for callers that split one logical
 * dataset across several physical indexes (e.g. one per segment/sstable) and want a single ranked
 * answer across all of them.
 * <p>
 * This is the Phase 2 implementation described in {@code docs/multi-index-search.md}. Each shard's
 * initial ask is sized proportionally to its share of the total vector count (via
 * {@link OverqueryStrategy}), rather than asking every shard for a full local top-{@code topK} --
 * this avoids the redundant work of Phase 1, where every shard fully explored its own top-{@code topK}
 * even though only a fraction of that typically survives the global merge. Like a single-shard
 * approximate search, the result is <b>not</b> guaranteed to be exact: a shard's proportional share is
 * based on its size, not its relevance to the query, so a small shard that happens to be unusually
 * close to the query can legitimately be underrepresented in the result -- the same kind of recall
 * tradeoff a larger overquery factor manages for a single index. If the proportional asks together
 * come up short of {@code topK} candidates overall (e.g. because per-shard {@code acceptOrds} filtered
 * out most of one shard), shards that weren't exhausted are automatically resumed for up to
 * {@link Builder#withMaxResumeRounds} additional rounds to fill the shortfall; this is the only
 * automatic escalation {@link #search} performs; see {@link #resume(int)} for a caller-driven way to
 * ask for more when the caller -- not a raw candidate-count shortfall -- determines more are needed.
 * (An earlier version of this class also resumed a shard whenever its own worst-returned score was
 * still at or above the current merge cutoff, to try to recover exactness; that check is unreliable
 * once a shard's results make up part of the current top-{@code topK}, since its worst score is then
 * close to -- or literally part of -- the cutoff it's being compared against, so it fired on nearly
 * every call and made a single {@link #search} cost close to an exhaustive scan.)
 * <p>
 * {@link #search} starts a new query, discarding any state left over from a previous {@link #search}/
 * {@link #resume} sequence on this instance. Once it returns, {@link #resume(int)} can be called to
 * grow that same query's result count by asking every shard to keep searching from where it left off
 * -- the pull-driven pattern a lazy consumer doing its own downstream filtering (e.g. Cassandra
 * reconciling stale/duplicate rows across sstables) needs when the current top-K doesn't yield enough
 * valid results. It is not valid to call {@link #resume(int)} before calling {@link #search}.
 * <p>
 * Not safe for concurrent use by multiple threads -- like {@link GraphSearcher} (which this class
 * composes, one instance per shard), scratch state is reused across calls to {@link #search} and
 * {@link #resume(int)}.
 * <p>
 * By default, shards are searched sequentially. Since shards are independent (each has its own
 * {@link GraphSearcher} instance, never shared across shards), fan-out across shards can safely be
 * parallelized -- construct via {@link #builder} and call {@link Builder#withExecutor} to supply an
 * {@link ExecutorService} to search shards concurrently. The executor is caller-owned: this class
 * never shuts it down, including from {@link #close()}.
 */
@Experimental
public class MultiGraphSearcher implements AutoCloseable {
    /** Default per-round growth factor applied to a resumed shard's rerank/result budget. */
    private static final double GROWTH_FACTOR = 2.0;

    /** Default number of resume rounds allowed after the initial round (so 3 rounds total). */
    private static final int DEFAULT_MAX_RESUME_ROUNDS = 2;

    private final List<GraphSearcher> searchers;
    private final long[] shardSizes;
    private final ExecutorService executor;
    private final OverqueryStrategy overqueryStrategy;
    private final int maxResumeRounds;

    // Session state for the current query, established by search() and extended by resume(). Per-shard
    // results accumulate across rounds: resume() returns only newly-discovered nodes (continuing the
    // search where it left off), not a replacement for what a shard already returned, so each round's
    // new nodes are appended rather than overwriting prior rounds'.
    private List<SearchResult.NodeScore>[] accumulated;
    private int[] askThisRound;
    private int[] nextBudget;
    private int[] lastReturnedCount;
    private int currentTopK;
    private int roundsUsedTotal;
    private int visitedCountTotal;
    private int expandedCountTotal;
    private int rerankedCountTotal;
    private boolean hasSearched = false;

    /**
     * @param shards the graph indexes to search, in the order that
     *               {@link ShardedSearchResult.NodeScore#shardIndex} will refer to them
     */
    public MultiGraphSearcher(List<? extends ImmutableGraphIndex> shards) {
        this(shards, null, OverqueryStrategy.DEFAULT, DEFAULT_MAX_RESUME_ROUNDS);
    }

    private MultiGraphSearcher(List<? extends ImmutableGraphIndex> shards,
                                ExecutorService executor,
                                OverqueryStrategy overqueryStrategy,
                                int maxResumeRounds)
    {
        if (shards.isEmpty()) {
            throw new IllegalArgumentException("MultiGraphSearcher requires at least one shard");
        }
        var searchers = new ArrayList<GraphSearcher>(shards.size());
        var shardSizes = new long[shards.size()];
        for (int i = 0; i < shards.size(); i++) {
            searchers.add(new GraphSearcher(shards.get(i)));
            shardSizes[i] = shards.get(i).size(0);
        }
        this.searchers = searchers;
        this.shardSizes = shardSizes;
        this.executor = executor;
        this.overqueryStrategy = overqueryStrategy;
        this.maxResumeRounds = maxResumeRounds;
    }

    /**
     * Returns a fluent builder for configuring and constructing a {@link MultiGraphSearcher}.
     *
     * @param shards the graph indexes to search, in the order that
     *               {@link ShardedSearchResult.NodeScore#shardIndex} will refer to them
     */
    public static Builder builder(List<? extends ImmutableGraphIndex> shards) {
        return new Builder(shards);
    }

    /**
     * @return the number of shards this searcher was constructed with
     */
    public int shardCount() {
        return searchers.size();
    }

    /**
     * @return the {@link ImmutableGraphIndex.View} that shard {@code shardIndex}'s internal
     * {@link GraphSearcher} traverses with. Fixed for the lifetime of this instance (each shard's
     * {@code GraphSearcher} is constructed once and reused across every {@link #search}/
     * {@link #resume} call), so it's safe to fetch once and reuse.
     * <p>
     * Needed when building a {@link SearchScoreProvider} whose approximate scoring is stateful and
     * tied to a specific view -- {@code FUSED_PQ}'s neighbor-code reading during traversal is the
     * motivating example, since it reads through the view's own reader position as the search walks
     * neighbors. Such a score provider must be built from <em>this</em> view, not one independently
     * obtained via {@code shard.getView()} on the original {@link ImmutableGraphIndex} -- a
     * separately-obtained view has its own independent reader, and scoring built from it silently
     * reads garbage once the actual search drives a different view's reader position. Score providers
     * with stateless approximate scoring (exact, or {@code CompressedVectors}-based) aren't affected
     * and don't need this.
     *
     * @param shardIndex index into the shard list passed to this searcher's constructor
     */
    public ImmutableGraphIndex.View getView(int shardIndex) {
        return searchers.get(shardIndex).getView();
    }

    /**
     * Searches every shard and returns the merged global top-{@code topK}, best first.
     * <p>
     * Each shard's initial ask is sized proportionally to its share of the total vector count
     * (see {@link OverqueryStrategy}), then shards are adaptively resumed across up to
     * {@link Builder#withMaxResumeRounds} additional rounds until no shard both (a) returned a full
     * local result set and (b) has a worst returned score still competitive with the current global
     * cutoff. If an {@link ExecutorService} was supplied via {@link Builder#withExecutor}, each
     * round's shard calls are dispatched concurrently and this call blocks until the round completes.
     *
     * @param scoreProviders     one {@link SearchScoreProvider} per shard, in shard order. Each
     *                           closes over the same query but that shard's own vectors/compressor.
     * @param acceptOrdsPerShard one {@link Bits} per shard, in shard order, using ordinals local to
     *                           that shard. Use {@link Bits#ALL} for shards with no per-query filter.
     * @param topK               desired global result count
     * @param rerankK            global rerank budget, split proportionally across shards by
     *                           {@link OverqueryStrategy} (default: by shard size, the same split
     *                           applied to {@code topK}) -- {@code rerankK == topK} costs about the
     *                           same, per shard, as a single-shard search at that same overquery factor
     * @return the merged results, plus metrics summed across all shards' work across all rounds
     */
    public ShardedSearchResult search(List<SearchScoreProvider> scoreProviders,
                                       List<Bits> acceptOrdsPerShard,
                                       int topK,
                                       int rerankK)
    {
        int n = searchers.size();
        if (scoreProviders.size() != n || acceptOrdsPerShard.size() != n) {
            throw new IllegalArgumentException(String.format(
                    "Expected %d score providers and accept-ords (one per shard), got %d and %d",
                    n, scoreProviders.size(), acceptOrdsPerShard.size()));
        }

        long totalSize = 0;
        for (long size : shardSizes) {
            totalSize += size;
        }

        // Starting a new query: reset all session state left over from any previous search()/resume()
        // sequence on this instance.
        askThisRound = new int[n];
        nextBudget = new int[n];
        for (int i = 0; i < n; i++) {
            askThisRound[i] = proportionalShare(shardSizes[i], totalSize, topK);
            nextBudget[i] = Math.max(
                    askThisRound[i],
                    overqueryStrategy.initialRerankKFor(i, shardSizes[i], totalSize, topK, rerankK));
        }

        @SuppressWarnings("unchecked")
        List<SearchResult.NodeScore>[] freshAccumulated = new List[n];
        accumulated = freshAccumulated;
        lastReturnedCount = new int[n];
        visitedCountTotal = 0;
        expandedCountTotal = 0;
        rerankedCountTotal = 0;

        int[] allIndices = new int[n];
        for (int i = 0; i < n; i++) {
            allIndices[i] = i;
        }
        var initial = dispatch(allIndices, i -> searchers.get(i).search(
                scoreProviders.get(i), askThisRound[i], nextBudget[i], 0.0f, 0.0f, acceptOrdsPerShard.get(i)));
        for (int i = 0; i < n; i++) {
            var result = initial[i];
            accumulated[i] = new ArrayList<>(List.of(result.getNodes()));
            lastReturnedCount[i] = result.getNodes().length;
            visitedCountTotal += result.getVisitedCount();
            expandedCountTotal += result.getExpandedCount();
            rerankedCountTotal += result.getRerankedCount();
            nextBudget[i] = growBudget(nextBudget[i]);
        }

        hasSearched = true;
        return continueRounds(topK, 1);
    }

    /**
     * Grows the current query's result count by {@code additionalK} and returns the new merged
     * top-{@code (previous topK + additionalK)}, reusing whatever shard results are already
     * accumulated and only asking shards to search further if what's already been found can't satisfy
     * the larger request. Intended for a caller that consumes results lazily and does its own
     * downstream filtering (deduplication, liveness/tombstone checks, ...) -- when that filtering
     * leaves fewer than the desired number of valid results, {@code resume} asks for more without
     * restarting the whole multi-shard search from scratch.
     * <p>
     * Like {@link #search}, shards that still qualify (returned a full batch last round and remain
     * competitive with the current cutoff) may be resumed across multiple rounds, up to
     * {@link Builder#withMaxResumeRounds} rounds for this call.
     *
     * @param additionalK how many more results are wanted, beyond the topK from the previous
     *                    {@link #search}/{@link #resume} call
     * @return the merged top-{@code (previous topK + additionalK)}, plus metrics summed across all
     * shards' work across every round since the initiating {@link #search} call
     * @throws IllegalStateException if called before {@link #search}
     */
    public ShardedSearchResult resume(int additionalK) {
        if (!hasSearched) {
            throw new IllegalStateException("resume() called before search()");
        }
        if (additionalK <= 0) {
            throw new IllegalArgumentException("additionalK must be positive, got " + additionalK);
        }
        return continueRounds(currentTopK + additionalK, roundsUsedTotal);
    }

    /**
     * Shared round loop for {@link #search} (called with {@code startingRound=1}, since the initial
     * per-shard dispatch already happened) and {@link #resume} (called with {@code startingRound} set
     * to the rounds already used, since it continues an existing session). Merges whatever's currently
     * accumulated; if that doesn't yet total {@code topK} candidates, resumes shards that aren't
     * exhausted for up to {@link #maxResumeRounds} more rounds beyond {@code startingRound}, to fill
     * the shortfall.
     * <p>
     * This is deliberately a narrow, unambiguous trigger -- resume only when there is a real shortfall
     * in the total candidate count -- rather than trying to guess whether some shard's own top result
     * might be beaten by exploring further. That guess is unreliable: once a shard's own results make
     * up part of the current global top-{@code topK}, its worst-returned score is, by construction,
     * close to (or literally part of) the merge's own cutoff, so any per-shard "is my worst score still
     * competitive" check is close to tautological and fires on nearly every call -- which previously
     * caused every {@link #search} to escalate to {@link #maxResumeRounds} regardless of whether more
     * searching would help, making a single call cost close to an exhaustive scan instead of a bounded
     * approximate search. Bounding resume to genuine shortfalls keeps {@link #search}'s cost comparable
     * to a single-shard ANN search at the same overquery factor; a caller that wants a shard possibly
     * holding better-but-unretrieved candidates to be explored further should use the caller-facing
     * {@link #resume(int)} explicitly.
     */
    private ShardedSearchResult continueRounds(int topK, int startingRound) {
        int n = searchers.size();
        List<ShardedSearchResult.NodeScore> merged;
        int round = startingRound;
        int maxRound = startingRound + maxResumeRounds;
        while (true) {
            merged = mergeAndTrim(accumulated, topK);

            if (merged.size() >= topK || round >= maxRound) {
                break;
            }

            var resumeIndices = new ArrayList<Integer>();
            for (int i = 0; i < n; i++) {
                boolean notExhausted = lastReturnedCount[i] == askThisRound[i];
                if (notExhausted) {
                    resumeIndices.add(i);
                }
            }
            if (resumeIndices.isEmpty()) {
                break;
            }
            round++;

            int[] resumeArray = resumeIndices.stream().mapToInt(Integer::intValue).toArray();
            for (int i : resumeArray) {
                askThisRound[i] = nextBudget[i];
            }
            var resumed = dispatch(resumeArray, i -> searchers.get(i).resume(askThisRound[i], askThisRound[i]));
            for (int k = 0; k < resumeArray.length; k++) {
                int i = resumeArray[k];
                var result = resumed[k];
                accumulated[i].addAll(List.of(result.getNodes()));
                lastReturnedCount[i] = result.getNodes().length;
                visitedCountTotal += result.getVisitedCount();
                expandedCountTotal += result.getExpandedCount();
                rerankedCountTotal += result.getRerankedCount();
                nextBudget[i] = growBudget(nextBudget[i]);
            }
        }

        currentTopK = topK;
        roundsUsedTotal = round;
        var nodes = merged.toArray(new ShardedSearchResult.NodeScore[0]);
        return new ShardedSearchResult(nodes, visitedCountTotal, expandedCountTotal, rerankedCountTotal, round);
    }

    /**
     * @return {@code max(1, round(topK * shareSize / totalSize))}, or {@code topK} if {@code totalSize}
     * is zero (all shards empty -- the value is moot since every shard will return no results)
     */
    private static int proportionalShare(long shareSize, long totalSize, int topK) {
        if (totalSize == 0) {
            return topK;
        }
        return (int) Math.max(1, Math.round(topK * (double) shareSize / totalSize));
    }

    /** Grows a per-shard budget by {@link #GROWTH_FACTOR}, guaranteeing forward progress. */
    private static int growBudget(int current) {
        return Math.max(current + 1, (int) Math.round(current * GROWTH_FACTOR));
    }

    private static List<ShardedSearchResult.NodeScore> mergeAndTrim(List<SearchResult.NodeScore>[] accumulated, int topK) {
        var candidates = new ArrayList<ShardedSearchResult.NodeScore>();
        for (int shardIndex = 0; shardIndex < accumulated.length; shardIndex++) {
            for (var nodeScore : accumulated[shardIndex]) {
                candidates.add(new ShardedSearchResult.NodeScore(shardIndex, nodeScore.node, nodeScore.score));
            }
        }
        Collections.sort(candidates);
        return candidates.subList(0, Math.min(topK, candidates.size()));
    }

    /**
     * Runs {@code perShard.apply(i)} for each shard index in {@code indices}, either sequentially on
     * the calling thread or concurrently via {@link #executor}, and returns the results in the same
     * order as {@code indices} (not shard order -- callers map results back to shard indices themselves).
     * Only dispatching the indices that actually need work keeps later resume rounds from submitting
     * no-op tasks for shards that aren't being resumed.
     */
    private SearchResult[] dispatch(int[] indices, IntFunction<SearchResult> perShard) {
        return executor == null ? dispatchSequentially(indices, perShard) : dispatchInParallel(indices, perShard);
    }

    private SearchResult[] dispatchSequentially(int[] indices, IntFunction<SearchResult> perShard) {
        var results = new SearchResult[indices.length];
        for (int k = 0; k < indices.length; k++) {
            results[k] = perShard.apply(indices[k]);
        }
        return results;
    }

    private SearchResult[] dispatchInParallel(int[] indices, IntFunction<SearchResult> perShard) {
        var futures = new ArrayList<Future<SearchResult>>(indices.length);
        for (int idx : indices) {
            futures.add(executor.submit(() -> perShard.apply(idx)));
        }

        var results = new SearchResult[indices.length];
        try {
            for (int k = 0; k < futures.size(); k++) {
                results[k] = futures.get(k).get();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted while searching shards", e);
        } catch (ExecutionException e) {
            throw new RuntimeException("Shard search failed", e.getCause());
        }
        return results;
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
     * <p>
     * Does not shut down the {@link ExecutorService} supplied via {@link Builder#withExecutor}, if
     * any -- that executor is caller-owned.
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

    /**
     * Pluggable policy for sizing a shard's initial rerank budget, given its share of the total
     * dataset. The default implementation splits the caller's {@code globalRerankK} proportionally,
     * the same way {@link #search} splits {@code topK}; it's exposed as a strategy so callers can
     * plug in a fixed per-shard floor, a different growth curve, or anything else, without forking
     * this class.
     */
    @FunctionalInterface
    public interface OverqueryStrategy {
        /**
         * @param shardIndex   index of the shard being sized, in the order passed to
         *                     {@link MultiGraphSearcher}'s constructor
         * @param shardSize    number of vectors in this shard ({@code index.size(0)})
         * @param totalSize    number of vectors across all shards
         * @param topK         the caller's requested global result count
         * @param globalRerankK the caller's requested global rerank budget
         * @return the initial rerank budget for this shard. Values less than this shard's
         * proportional share of {@code topK} are clamped up to that share.
         */
        int initialRerankKFor(int shardIndex, long shardSize, long totalSize, int topK, int globalRerankK);

        /**
         * This shard's proportional share of {@code globalRerankK} -- the same proportional-by-size
         * split {@link #search} already applies to {@code topK}, just applied to the caller's actual
         * rerank budget instead of silently substituting a fixed multiplier. This preserves the
         * caller's intended overquery factor: requesting {@code rerankK == topK} (overquery 1x) costs
         * about the same, per shard, as a single-shard search at overquery 1x, rather than always
         * paying for some other multiplier regardless of what was asked for.
         */
        OverqueryStrategy DEFAULT = (shardIndex, shardSize, totalSize, topK, globalRerankK) ->
                proportionalShare(shardSize, totalSize, globalRerankK);
    }

    /**
     * Fluent builder for {@link MultiGraphSearcher}.
     */
    public static final class Builder {
        private final List<? extends ImmutableGraphIndex> shards;
        private ExecutorService executor;
        private OverqueryStrategy overqueryStrategy = OverqueryStrategy.DEFAULT;
        private int maxResumeRounds = DEFAULT_MAX_RESUME_ROUNDS;

        Builder(List<? extends ImmutableGraphIndex> shards) {
            this.shards = shards;
        }

        /**
         * Supplies an executor to search shards concurrently rather than sequentially. The executor
         * is caller-owned -- {@link MultiGraphSearcher} never shuts it down. Not required; if omitted,
         * shards are searched sequentially on the calling thread.
         */
        public Builder withExecutor(ExecutorService executor) {
            this.executor = executor;
            return this;
        }

        /**
         * Supplies the policy used to size each shard's initial rerank budget. Not required; if
         * omitted, {@link OverqueryStrategy#DEFAULT} is used.
         */
        public Builder withOverqueryStrategy(OverqueryStrategy overqueryStrategy) {
            this.overqueryStrategy = overqueryStrategy;
            return this;
        }

        /**
         * Caps how many additional rounds beyond the initial one {@link #search} will use to resume
         * shards that look shortchanged by their proportional initial ask. Not required; defaults to
         * 2 (so 3 rounds total). Zero disables resume entirely, reducing to a single
         * proportional-sizing round with no adaptive refill.
         */
        public Builder withMaxResumeRounds(int maxResumeRounds) {
            if (maxResumeRounds < 0) {
                throw new IllegalArgumentException("maxResumeRounds must be >= 0, got " + maxResumeRounds);
            }
            this.maxResumeRounds = maxResumeRounds;
            return this;
        }

        public MultiGraphSearcher build() {
            return new MultiGraphSearcher(shards, executor, overqueryStrategy, maxResumeRounds);
        }
    }
}
