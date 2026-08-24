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
import io.github.jbellis.jvector.graph.ImmutableGraphIndex.NodeAtLevel;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.BoundedLongHeap;
import io.github.jbellis.jvector.util.GrowableLongHeap;
import org.agrona.collections.Int2ObjectHashMap;
import org.agrona.collections.IntHashSet;

import java.io.IOException;

/**
 * Holds the graph search algorithm and its scratch state. {@link GraphSearcher} is the public,
 * concrete facade that exposes construction and configuration; this class is not part of the public API.
 * <p>
 * Extracted from what was previously a single {@code GraphSearcher} class so that a future second
 * implementation of {@link Searcher} (e.g. searching across multiple independent shards) can reuse
 * this algorithm without reimplementing it or depending on {@code GraphSearcher}'s public surface.
 */
abstract class AbstractSearcher implements Searcher {

    protected ImmutableGraphIndex.View view;

    // Scratch data structures that are used in each search() call. These can be expensive
    // to allocate, so they're cleared and reused across calls.
    private final NodeQueue candidates;
    final NodeQueue approximateResults;
    private final NodeQueue rerankedResults;
    private final IntHashSet visited;
    private final NodesUnsorted evictedResults;

    // Search parameters that we save here for use by resume()
    private Bits acceptOrds;
    private SearchScoreProvider scoreProvider;
    private CachingReranker cachingReranker;

    // Pruning is permanently disabled (see #686 -- the heuristic reduced recall in production and
    // has not shown reliable value). This field is retained, always false, so that a future re-enable
    // decision only has to flip this default rather than re-thread a parameter through the search loop.
    boolean pruneSearch;
    private final ScoreTracker.ScoreTrackerFactory scoreTrackerFactory;

    protected int visitedCount;
    protected int expandedCount;
    protected int expandedCountBaseLayer;

    AbstractSearcher(ImmutableGraphIndex.View view) {
        this.view = view;
        this.candidates = new NodeQueue(new GrowableLongHeap(100), NodeQueue.Order.MAX_HEAP);
        this.evictedResults = new NodesUnsorted(100);
        this.approximateResults = new NodeQueue(new BoundedLongHeap(100), NodeQueue.Order.MIN_HEAP);
        this.rerankedResults = new NodeQueue(new BoundedLongHeap(100), NodeQueue.Order.MIN_HEAP);
        this.visited = new IntHashSet();
        this.pruneSearch = false;
        this.scoreTrackerFactory = new ScoreTracker.ScoreTrackerFactory();
    }

    private void initializeScoreProvider(SearchScoreProvider scoreProvider) {
        this.scoreProvider = scoreProvider;
        if (scoreProvider.reranker() == null) {
            cachingReranker = null;
            return;
        }
        cachingReranker = new CachingReranker(scoreProvider);
    }

    @Override
    @Experimental
    public SearchResult search(SearchScoreProvider scoreProvider,
                               int topK,
                               int refineK,
                               float threshold,
                               float refineFloor,
                               Bits acceptOrds)
    {
        NodeAtLevel entry = view.entryNode();
        if (acceptOrds == null) {
            throw new IllegalArgumentException("Use MatchAllBits to indicate that all ordinals are accepted, instead of null");
        }
        if (refineK < topK) {
            throw new IllegalArgumentException(String.format("refineK %d must be >= topK %d", refineK, topK));
        }
        if (entry == null) {
            return new SearchResult(new SearchResult.NodeScore[0], 0, 0, 0, 0, Float.POSITIVE_INFINITY);
        }
        internalSearch(scoreProvider, entry, topK, refineK, threshold, acceptOrds);
        return reranking(topK, refineK, refineFloor);
    }

    @Override
    public SearchResult search(SearchScoreProvider scoreProvider, int topK, float threshold, Bits acceptOrds) {
        return search(scoreProvider, topK, topK, threshold, 0.0f, acceptOrds);
    }

    @Override
    public SearchResult search(SearchScoreProvider scoreProvider, int topK, Bits acceptOrds) {
        return search(scoreProvider, topK, 0.0f, acceptOrds);
    }

    protected void internalSearch(SearchScoreProvider scoreProvider,
                                  NodeAtLevel entry,
                                  int topK,
                                  int refineK,
                                  float threshold,
                                  Bits acceptOrds)
    {
        initializeInternal(scoreProvider, entry, acceptOrds);

        // Move downward from entry.level to 1
        for (int lvl = entry.level; lvl > 0; lvl--) {
            searchOneLayer(scoreProvider, 1, 0.0f, lvl, Bits.ALL);
            assert approximateResults.size() == 1 : approximateResults.size();
            setEntryPointsFromPreviousLayer();
        }

        searchLayer0(topK, refineK, threshold);
    }

    @Experimental
    public void setEntryPointsFromPreviousLayer() {
        // push the candidates seen so far back onto the queue for the next layer
        // at worst we save recomputing the similarity; at best we might connect to a more distant cluster
        approximateResults.foreach(candidates::push);
        evictedResults.foreach(candidates::push);
        evictedResults.clear();
        approximateResults.clear();
    }

    @Experimental
    public void initializeInternal(SearchScoreProvider scoreProvider, NodeAtLevel entry, Bits rawAcceptOrds) {
        // save search parameters for potential later resume
        initializeScoreProvider(scoreProvider);
        this.acceptOrds = Bits.intersectionOf(rawAcceptOrds, view.liveNodes());

        // reset the scratch data structures
        approximateResults.clear();
        evictedResults.clear();
        candidates.clear();
        visited.clear();

        // Start with entry point
        float score = scoreProvider.scoreFunction().similarityTo(entry.node);
        visited.add(entry.node);
        candidates.push(entry.node, score);

        visitedCount = 0;
        expandedCount = 0;
        expandedCountBaseLayer = 0;
    }

    private boolean stopSearch(NodeQueue localCandidates, ScoreTracker scoreTracker, int refineK, float threshold) {
        float topCandidateScore = localCandidates.topScore();

        // we're done when we have K results and the best candidate is worse than the worst result so far
        if (approximateResults.size() >= refineK && topCandidateScore < approximateResults.topScore()) {
            return true;
        }

        // preserve legacy threshold early termination
        if (threshold > 0 && scoreTracker.shouldStop()) {
            return true;
        }

        return false;
    }

    // Since Astra / Cassandra's usage drives the design decisions here, it's worth being explicit
    // about how that works and why.
    //
    // Astra breaks logical indexes up across multiple physical OnDiskGraphIndex pieces, one per sstable.
    // Each of these pieces is searched independently, and the results are combined.  To avoid doing
    // more work than necessary, Astra assumes that each physical ODGI will contribute responses
    // to the final result in proportion to its size, and only asks for that many results in the initial
    // search.  If this assumption is incorrect, or if the rows found turn out to be deleted or overwritten
    // by later requests (which will be in a different sstable), Astra wants a lightweight way to resume
    // the search where it was left off to get more results.
    //
    // Because Astra uses a nonlinear overquerying strategy (i.e. refineK will be larger in proportion to
    // topK for small values of topK than for large), it's especially important to avoid refining more
    // results than necessary.  Thus, Astra will look at the worstApproximateInTopK value from the first
    // ODGI, and use that as the refineFloor for the next.  Thus, refineFloor helps avoid believed-to-be-
    // unnecessary work in the initial search, but if the caller needs to resume() then that belief was
    // incorrect and is discarded, and there is no reason to pass a refineFloor parameter to resume().
    //
    // Finally: resume() also drives the use of CachingReranker.
    @Experimental
    public void searchOneLayer(SearchScoreProvider scoreProvider,
                        int refineK,
                        float threshold,
                        int level,
                        Bits acceptOrdsThisLayer)
    {
        try {
            assert approximateResults.size() == 0; // should be cleared by setEntryPointsFromPreviousLayer
            approximateResults.setMaxSize(refineK);

            // Pruning is permanently disabled; pruneSearch is always false (see field comment above).
            var scoreTracker = scoreTrackerFactory.getScoreTracker(pruneSearch, refineK, threshold);

            // the main search loop
            while (candidates.size() > 0) {
                if (stopSearch(candidates, scoreTracker, refineK, threshold)) {
                    break;
                }

                // process the top candidate
                float topCandidateScore = candidates.topScore();
                int topCandidateNode = candidates.pop();
                if (acceptOrdsThisLayer.get(topCandidateNode) && topCandidateScore >= threshold) {
                    addTopCandidate(topCandidateNode, topCandidateScore, refineK);
                }

                // skip edge loading if we've found a local maximum and we have enough results
                if (scoreTracker.shouldStop() && candidates.size() >= refineK - approximateResults.size()) {
                    continue;
                }

                if (level == 0) {
                    expandedCountBaseLayer++;
                }
                expandedCount++;

                // score the neighbors of the top candidate and add them to the queue
                var scoreFunction = scoreProvider.scoreFunction();
                ImmutableGraphIndex.NeighborProcessor neighborProcessor = (node2, score) -> {
                    scoreTracker.track(score);
                    candidates.push(node2, score);
                    visitedCount++;
                };
                view.processNeighbors(level, topCandidateNode, scoreFunction, visited::add, neighborProcessor);
            }
        } catch (Throwable t) {
            // clear scratch structures if terminated via throwable, as they may not have been drained
            approximateResults.clear();
            throw t;
        }
    }

    private void searchLayer0(int topK, int refineK, float threshold) {
        rerankedResults.clear();
        rerankedResults.setMaxSize(topK);

        // add evicted results from the last call back to the candidates
        evictedResults.foreach(candidates::push);
        evictedResults.clear();

        searchOneLayer(scoreProvider, refineK, threshold, 0, acceptOrds);
    }

    SearchResult reranking(int topK, int refineK, float refineFloor) {
        assert approximateResults.size() <= refineK;
        NodeQueue popFromQueue;
        float worstApproximateInTopK;
        int refined;
        if (cachingReranker == null) {
            // save the worst candidates in evictedResults for potential resume()
            while (approximateResults.size() > topK) {
                var nScore = approximateResults.topScore();
                var n = approximateResults.pop();
                evictedResults.add(n, nScore);
            }
            refined = 0;
            worstApproximateInTopK = Float.POSITIVE_INFINITY;
            popFromQueue = approximateResults;
        } else {
            int oldRefined = cachingReranker.getRerankCalls();
            worstApproximateInTopK = approximateResults.rerank(topK, cachingReranker, refineFloor, rerankedResults, evictedResults);
            refined = cachingReranker.getRerankCalls() - oldRefined;
            approximateResults.clear();
            popFromQueue = rerankedResults;
        }
        // pop the top K results from the results queue, which has the worst candidates at the top
        assert popFromQueue.size() <= topK;
        var nodes = new SearchResult.NodeScore[popFromQueue.size()];
        for (int i = nodes.length - 1; i >= 0; i--) {
            var nScore = popFromQueue.topScore();
            var n = popFromQueue.pop();
            nodes[i] = new SearchResult.NodeScore(n, nScore);
        }
        assert popFromQueue.size() == 0;

        return new SearchResult(nodes, visitedCount, expandedCount, expandedCountBaseLayer, refined, worstApproximateInTopK);
    }

    SearchResult resume(int topK, int refineK, float threshold, float refineFloor) {
        searchLayer0(topK, refineK, threshold);
        return reranking(topK, refineK, refineFloor);
    }

    @SuppressWarnings("StatementWithEmptyBody")
    private void addTopCandidate(int topCandidateNode, float topCandidateScore, int refineK) {
        // add the new node to the results queue, and any evicted node to evictedResults in case we resume later
        // (push() can't tell us what node was evicted when the queue was already full, so we examine that manually)
        if (approximateResults.size() < refineK) {
            approximateResults.push(topCandidateNode, topCandidateScore);
        } else if (topCandidateScore > approximateResults.topScore()) {
            int evictedNode = approximateResults.topNode();
            float evictedScore = approximateResults.topScore();
            evictedResults.add(evictedNode, evictedScore);
            approximateResults.push(topCandidateNode, topCandidateScore);
        } else {
            // score is exactly equal to the worst candidate in our results, so we don't bother
            // changing the results queue.  (We still want to check its neighbors to see if one of them
            // is better.)
        }
    }

    @Override
    public void close() throws IOException {
        view.close();
    }

    private static class CachingReranker implements ScoreFunction.ExactScoreFunction {
        // this cache never gets cleared out (until a new search reinitializes it),
        // but we expect resume() to be called at most a few times so it's fine
        private final Int2ObjectHashMap<Float> cachedScores;
        private final SearchScoreProvider scoreProvider;
        private int rerankCalls;

        CachingReranker(SearchScoreProvider scoreProvider) {
            this.scoreProvider = scoreProvider;
            cachedScores = new Int2ObjectHashMap<>();
            rerankCalls = 0;
        }

        @Override
        public float similarityTo(int node2) {
            if (cachedScores.containsKey(node2)) {
                return cachedScores.get(node2);
            }
            rerankCalls++;
            float score = scoreProvider.reranker().similarityTo(node2);
            cachedScores.put(node2, Float.valueOf(score));
            return score;
        }

        int getRerankCalls() {
            return rerankCalls;
        }
    }
}
