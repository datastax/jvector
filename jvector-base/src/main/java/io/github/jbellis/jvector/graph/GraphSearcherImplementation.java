/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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
import io.github.jbellis.jvector.graph.similarity.SimilarityFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreBundle;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.BoundedLongHeap;
import io.github.jbellis.jvector.util.GrowableLongHeap;
import io.github.jbellis.jvector.vector.VectorRepresentation;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.Int2ObjectHashMap;
import org.agrona.collections.IntHashSet;

import java.io.IOException;


/**
 * Searches a graph to find nearest neighbors to a query vector.
 * A GraphSearcher is not threadsafe: only one search at a time should be run per GraphSearcher.
 * For more background on the search algorithm, see {@link ImmutableGraphIndex}.
 */
public class GraphSearcherImplementation<Primary extends VectorRepresentation, Secondary extends VectorRepresentation> implements GraphSearcher {
    final private GraphIndexView<Primary, Secondary> view;
    final private SearchScoreBundle<Primary, Secondary> scoreProvider;

    // Scratch data structures that are used in each {@link #searchInternal} call. These can be expensive
    // to allocate, so they're cleared and reused across calls.
    private final NodeQueue candidates;
    final NodeQueue approximateResults;
    private final NodeQueue rerankedResults;
    private final IntHashSet visited;
    private final NodesUnsorted evictedResults;

    // Search parameters that we save here for use by resume()
    private Bits acceptOrds;

    private CachingReranker cachingReranker;

    private boolean pruneSearch;
    private final ScoreTracker.ScoreTrackerFactory scoreTrackerFactory;

    private int visitedCount;
    private int expandedCount;
    private int expandedCountBaseLayer;

    /**
     * Creates a new graph searcher from the given GraphIndex.View
     */
    protected GraphSearcherImplementation(GraphIndexView<Primary, Secondary> view, SearchScoreBundle<Primary, Secondary> scoreProvider) {
        this.view = view;
        this.scoreProvider = scoreProvider;
        this.candidates = new NodeQueue(new GrowableLongHeap(100), NodeQueue.Order.MAX_HEAP);
        this.evictedResults = new NodesUnsorted(100);
        this.approximateResults = new NodeQueue(new BoundedLongHeap(100), NodeQueue.Order.MIN_HEAP);
        this.rerankedResults = new NodeQueue(new BoundedLongHeap(100), NodeQueue.Order.MIN_HEAP);
        this.visited = new IntHashSet();

        this.pruneSearch = true;
        this.scoreTrackerFactory = new ScoreTracker.ScoreTrackerFactory();
    }

    protected int getVisitedCount() {
        return visitedCount;
    }

    protected int getExpandedCount() {
        return expandedCount;
    }

    protected int getExpandedCountBaseLayer() {
        return expandedCountBaseLayer;
    }

    private void initializeScoreProvider() {
        if (scoreProvider.secondaryScoreFunction() == null) {
            cachingReranker = null;
            return;
        }

        cachingReranker = new CachingReranker(scoreProvider);
    }

    @Override
    public void usePruning(boolean usage) {
        pruneSearch = usage;
    }

    @Override
    public SearchResult search(VectorFloat<?> queryVector,
                               int topK,
                               int rerankK,
                               float threshold,
                               float rerankFloor,
                               Bits acceptOrds)
    {
        NodeAtLevel entry = view.entryNode();
        if (acceptOrds == null) {
            throw new IllegalArgumentException("Use MatchAllBits to indicate that all ordinals are accepted, instead of null");
        }
        if (rerankK < topK) {
            throw new IllegalArgumentException(String.format("rerankK %d must be >= topK %d", rerankK, topK));
        }

        if (entry == null) {
            return new SearchResult(new SearchResult.NodeScore[0], 0, 0, 0, 0, Float.POSITIVE_INFINITY);
        }

        internalSearch(queryVector, entry, topK, rerankK, threshold, acceptOrds);
        return reranking(topK, rerankK, rerankFloor);
    }

    /**
     * Performs a search, leaving the results in the internal member variable approximateResults.
     * It does not perform reranking.
     *
     * @param queryVector     the query vector
     * @param entry           the entry point to the graph. Assumed to be not null.
     * @param topK            the number of results to look for. With threshold=0, the search will continue until at least
     *                        `topK` results have been found, or until the entire graph has been searched.
     * @param rerankK         the number of (approximately-scored) results to rerank before returning the best `topK`.
     * @param threshold       the minimum similarity (0..1) to accept; 0 will accept everything. May be used
     *                        with a large topK to find (approximately) all nodes above the given threshold.
     *                        If threshold > 0 then the search will stop when it is probabilistically unlikely
     *                        to find more nodes above the threshold, even if `topK` results have not yet been found.
     * @param acceptOrds      a Bits instance indicating which nodes are acceptable results.
     *                        If {@link Bits#ALL}, all nodes are acceptable.
     *                        It is caller's responsibility to ensure that there are enough acceptable nodes
     *                        that we don't search the entire graph trying to satisfy topK.
     */
    protected void internalSearch(VectorFloat<?> queryVector,
                                  NodeAtLevel entry,
                                  int topK,
                                  int rerankK,
                                  float threshold,
                                  Bits acceptOrds)
    {
        var sf = scoreProvider.primaryScoreFunction();
        sf.fixQuery(queryVector);

        initializeInternal(sf,entry, acceptOrds);

        // Move downward from entry.level to 1
        for (int lvl = entry.level; lvl > 0; lvl--) {
            // Search this layer with minimal parameters since we just want the best candidate
            searchOneLayer(scoreProvider, 1, 0.0f, lvl, Bits.ALL);
            assert approximateResults.size() == 1 : approximateResults.size();
            setEntryPointsFromPreviousLayer();
        }

        // Now do the main search at layer 0
        searchLayer0(topK, rerankK, threshold);;
    }

    void setEntryPointsFromPreviousLayer() {
        // push the candidates seen so far back onto the queue for the next layer
        // at worst we save recomputing the similarity; at best we might connect to a more distant cluster
        approximateResults.foreach(candidates::push);
        evictedResults.foreach(candidates::push);
        evictedResults.clear();
        approximateResults.clear();
    }

    void initializeInternal(SimilarityFunction<Primary> similarityFunction, NodeAtLevel entry, Bits rawAcceptOrds) {
        // save search parameters for potential later resume
        initializeScoreProvider();
        this.acceptOrds = Bits.intersectionOf(rawAcceptOrds, view.liveNodes());

        // reset the scratch data structures
        approximateResults.clear();
        evictedResults.clear();
        candidates.clear();
        visited.clear();

        // Start with entry point
        var vr = view.getPrimaryRepresentation(entry.node);
        float score = similarityFunction.similarityTo(vr);
        visited.add(entry.node);
        candidates.push(entry.node, score);

        visitedCount = 0;
        expandedCount = 0;
        expandedCountBaseLayer = 0;
    }

    private boolean stopSearch(NodeQueue localCandidates, ScoreTracker scoreTracker, int rerankK, float threshold) {
        float topCandidateScore = localCandidates.topScore();
        // we're done when we have K results and the best candidate is worse than the worst result so far
        if (approximateResults.size() >= rerankK && topCandidateScore < approximateResults.topScore()) {
            return true;
        }
        // when querying by threshold, also stop when we are probabilistically unlikely to find more qualifying results
        if (threshold > 0 && scoreTracker.shouldStop()) {
            return true;
        }
        return false;
    }

    /**
     * Performs a single-layer ANN search, expanding from the given candidates queue.
     *
     * @param scoreProvider       the current query's scoring/approximation logic
     * @param rerankK             how many results to over-query for approximate ranking
     * @param threshold           similarity threshold, or 0f if none
     * @param level               which layer to search
     *                            <p>
     *                            Modifies the internal search state.
     *                            When it's done, `approximateResults` contains the best `rerankK` results found at the given layer.
     * @param acceptOrdsThisLayer a Bits instance indicating which nodes are acceptable results.
     *                            If {@link Bits#ALL}, all nodes are acceptable.
     *                            It is caller's responsibility to ensure that there are enough acceptable nodes
     *                            that we don't search the entire graph trying to satisfy topK.
     */
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
    // Because Astra uses a nonlinear overquerying strategy (i.e. rerankK will be larger in proportion to
    // topK for small values of topK than for large), it's especially important to avoid reranking more
    // results than necessary.  Thus, Astra will look at the worstApproximateInTopK value from the first
    // ODGI, and use that as the rerankFloor for the next.  Thus, rerankFloor helps avoid believed-to-be-
    // unnecessary work in the initial search, but if the caller needs to resume() then that belief was
    // incorrect and is discarded, and there is no reason to pass a rerankFloor parameter to resume().
    //
    // Finally: resume() also drives the use of CachingReranker.
    void searchOneLayer(SearchScoreBundle scoreProvider,
                        int rerankK,
                        float threshold,
                        int level,
                        Bits acceptOrdsThisLayer)
    {
        try {
            assert approximateResults.size() == 0; // should be cleared by setEntryPointsFromPreviousLayer
            approximateResults.setMaxSize(rerankK);

            // track scores to predict when we are done with threshold queries
            var scoreTracker = scoreTrackerFactory.getScoreTracker(pruneSearch, rerankK, threshold);

            // the main search loop
            while (candidates.size() > 0) {
                if (stopSearch(candidates, scoreTracker, rerankK, threshold)) {
                    break;
                }

                // process the top candidate
                float topCandidateScore = candidates.topScore();
                int topCandidateNode = candidates.pop();
                if (acceptOrdsThisLayer.get(topCandidateNode) && topCandidateScore >= threshold) {
                    addTopCandidate(topCandidateNode, topCandidateScore, rerankK);
                }

                // skip edge loading if we've found a local maximum and we have enough results
                if (scoreTracker.shouldStop() && candidates.size() >= rerankK - approximateResults.size()) {
                    continue;
                }

                if (level == 0) {
                    expandedCountBaseLayer++;
                }
                expandedCount++;

                // score the neighbors of the top candidate and add them to the queue
                var scoreFunction = scoreProvider.primaryScoreFunction();
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

    private void searchLayer0(int topK, int rerankK, float threshold) {
        // rR is persistent to save on allocations
        rerankedResults.clear();
        rerankedResults.setMaxSize(topK);

        // add evicted results from the last call back to the candidates
        evictedResults.foreach(candidates::push);
        evictedResults.clear();

        searchOneLayer(scoreProvider, rerankK, threshold, 0, acceptOrds);
    }

    SearchResult reranking(int topK, int rerankK, float rerankFloor) {
        // rerank results
        assert approximateResults.size() <= rerankK;
        NodeQueue popFromQueue;
        float worstApproximateInTopK;
        int reranked;
        if (cachingReranker == null) {
            // save the worst candidates in evictedResults for potential resume()
            while (approximateResults.size() > topK) {
                var nScore = approximateResults.topScore();
                var n = approximateResults.pop();
                evictedResults.add(n, nScore);
            }

            reranked = 0;
            worstApproximateInTopK = Float.POSITIVE_INFINITY;
            popFromQueue = approximateResults;
        } else {
            int oldReranked = cachingReranker.getRerankCalls();
            worstApproximateInTopK = approximateResults.rerank(topK, cachingReranker, rerankFloor, rerankedResults, evictedResults);
            reranked = cachingReranker.getRerankCalls() - oldReranked;
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
        // that should be everything
        assert popFromQueue.size() == 0;

        return new SearchResult(nodes, visitedCount, expandedCount, expandedCountBaseLayer, reranked, worstApproximateInTopK);
    }

    SearchResult resume(int topK, int rerankK, float threshold, float rerankFloor) {
        searchLayer0(topK, rerankK, threshold);
        return reranking(topK, rerankK, rerankFloor);
    }

    @SuppressWarnings("StatementWithEmptyBody")
    private void addTopCandidate(int topCandidateNode, float topCandidateScore, int rerankK) {
        // add the new node to the results queue, and any evicted node to evictedResults in case we resume later
        // (push() can't tell us what node was evicted when the queue was already full, so we examine that manually)
        if (approximateResults.size() < rerankK) {
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

    @Experimental
    @Override
    public SearchResult resume(int additionalK, int rerankK) {
        visitedCount = 0;
        expandedCount = 0;
        expandedCountBaseLayer = 0;
        return resume(additionalK, rerankK, 0.0f, 0.0f);
    }

    @Override
    public void close() throws IOException {
        view.close();
    }

    private class CachingReranker implements SimilarityFunction<Secondary> {
        // this cache never gets cleared out (until a new search reinitializes it),
        // but we expect resume() to be called at most a few times so it's fine
        private final Int2ObjectHashMap<Float> cachedScores;
        private final SearchScoreBundle scoreProvider;
        private int rerankCalls;

        public CachingReranker(SearchScoreBundle scoreProvider) {
            this.scoreProvider = scoreProvider;
            cachedScores = new Int2ObjectHashMap<>();
            rerankCalls = 0;
        }

        public int getRerankCalls() {
            return rerankCalls;
        }

        @Override
        public boolean isExact() {
            // TODO complete
            return false;
        }

        @Override
        public void fixQuery(VectorFloat<?> query) {
            // TODO complete
        }

        @Override
        public float similarityTo(Secondary other) {
            // TODO complete
            return 0;
        }

        @Override
        public float similarity(Secondary vec1, Secondary vec2) {
            // TODO complete
            return 0;
        }

        @Override
        public VectorSimilarityFunction getSimilarityFunction() {
            // TODO complete
            return null;
        }

        @Override
        public SimilarityFunction<Secondary> copy() {
            // TODO complete
            return null;
        }

        @Override
        public <Vec2 extends VectorRepresentation> boolean compatible(SimilarityFunction<Vec2> other) {
            // TODO complete
            return false;
        }
    }
}
