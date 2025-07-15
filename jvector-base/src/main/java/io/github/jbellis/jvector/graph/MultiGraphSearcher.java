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
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.BoundedLongHeap;
import io.github.jbellis.jvector.util.GrowableLongHeap;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.Int2ObjectHashMap;
import org.agrona.collections.IntHashSet;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;


/**
 * Experimental!
 * <p>
 * Searches a graph to find nearest neighbors to a query vector. For more background on the
 * search algorithm, see {@link GraphIndex}.
 */
@Experimental
public class MultiGraphSearcher implements Closeable {
    private List<GraphIndex.View> views;

    // Scratch data structures that are used in each {@link #searchInternal} call. These can be expensive
    // to allocate, so they're cleared and reused across calls.
    private final NodeQueue candidates;
    final NodeQueue approximateResults;
    private final NodeQueue rerankedResults;
    private final IntHashSet visited;
    private final NodesUnsorted evictedResults;

    // Search parameters that we save here for use by resume()
    private final MultiviewBits acceptOrds;
    private List<SearchScoreProvider> scoreProviders;
    private CachingReranker cachingReranker;

    private boolean pruneSearch;
    private final ScoreTracker.ScoreTrackerFactory scoreTrackerFactory;

    private final List<GraphSearcher> searchers;

    private int visitedCount;
    private int expandedCount;
    private int expandedCountBaseLayer;

    /**
     * Creates a new graph searcher from the given list of GraphIndex
     */
    public MultiGraphSearcher(List<GraphIndex> graphs) {
        this.views = graphs.stream().map(GraphIndex::getView).collect(Collectors.toList());
        this.searchers = new ArrayList<>();
        for (var view : views) {
            this.searchers.add(new GraphSearcher(view));
        }
        this.candidates = new NodeQueue(new GrowableLongHeap(100), NodeQueue.Order.MAX_HEAP);
        this.evictedResults = new NodesUnsorted(100);
        this.approximateResults = new NodeQueue(new BoundedLongHeap(100), NodeQueue.Order.MIN_HEAP);
        this.rerankedResults = new NodeQueue(new BoundedLongHeap(100), NodeQueue.Order.MIN_HEAP);
        this.visited = new IntHashSet();
        this.pruneSearch = true;
        this.acceptOrds = new MultiviewBits(graphs.size());

        this.pruneSearch = true;
        this.scoreTrackerFactory = new ScoreTracker.ScoreTrackerFactory();
    }

    private void initializeScoreProvider(List<SearchScoreProvider> scoreProviders) {
        this.scoreProviders = scoreProviders;
        if (scoreProviders.stream().anyMatch(sp -> sp.reranker() == null)) {
            cachingReranker = null;
        } else {
            cachingReranker = new CachingReranker(scoreProviders);
        }
    }

    public List<GraphIndex.View> getViews() {
        return views;
    }

    /**
     * When using pruning, we are using a heuristic to terminate the search earlier.
     * In certain cases, it can lead to speedups. This is set to false by default.
     * @param usage a boolean that determines whether we do early termination or not.
     */
    public void usePruning(boolean usage) {
        for (var searcher : searchers) {
            searcher.usePruning(usage);
        }
        pruneSearch = usage;
    }

    /**
     * Convenience function for simple one-off searches.  It is caller's responsibility to make sure that it
     * is the unique owner of the vectors instance passed in here.
     */
    public static MultiSearchResult search(VectorFloat<?> queryVector, int topK, List<RandomAccessVectorValues> vectors, VectorSimilarityFunction similarityFunction, List<GraphIndex> graphs, List<Bits> acceptOrds) {
        try (var searcher = new MultiGraphSearcher(graphs)) {
            var ssps = new ArrayList<SearchScoreProvider>(graphs.size());
            for (var iView = 0; iView < searcher.views.size(); iView++) {
                var ssp = DefaultSearchScoreProvider.exact(queryVector, similarityFunction, vectors.get(iView));
                ssps.add(ssp);
            }
            return searcher.search(ssps, topK, acceptOrds);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Convenience function for simple one-off searches.  It is caller's responsibility to make sure that it
     * is the unique owner of the vectors instance passed in here.
     */
    public static MultiSearchResult search(VectorFloat<?> queryVector, int topK, int rerankK, List<RandomAccessVectorValues> vectors, VectorSimilarityFunction similarityFunction, List<GraphIndex> graphs, List<Bits> acceptOrds) {
        if (acceptOrds.size() != graphs.size()) {
            throw new IllegalArgumentException("Number of acceptOrds: " + acceptOrds.size() + " != " + graphs.size());
        }

        try (var searcher = new MultiGraphSearcher(graphs)) {
            var ssps = new ArrayList<SearchScoreProvider>(graphs.size());
            for (var iView = 0; iView < searcher.views.size(); iView++) {
                var ssp = DefaultSearchScoreProvider.exact(queryVector, similarityFunction, vectors.get(iView));
                ssps.add(ssp);
            }
            return searcher.search(ssps, topK, rerankK, 0.f, acceptOrds);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Sets the view of the graph to be used by the searcher.
     * <p>
     * This method should be used when the searcher operates over a view whose contents might not reflect all changes
     * to the underlying graph, such as {@link OnHeapGraphIndex.ConcurrentGraphIndexView}. This is an optimization over
     * creating a new graph searcher with every update to the view.
     *
     * @param views the new views
     */
    public void setView(List<GraphIndex.View> views) {
        this.views = views;
    }

    /**
     * @param scoreProviders  provides functions to return the similarity of a given node to the query vector
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
     * @return a SearchResult containing the topK results and the number of nodes visited during the search.
     */
    public MultiSearchResult search(List<SearchScoreProvider> scoreProviders,
                                    int topK,
                                    int rerankK,
                                    float threshold,
                                    List<Bits> acceptOrds)
    {
        if (acceptOrds.size() != views.size()) {
            throw new IllegalArgumentException("Number of acceptOrds: " + acceptOrds.size() + " != " + views.size());
        }
        if (acceptOrds.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException("Use MatchAllBits to indicate that all ordinals are accepted, instead of null");
        }
        if (rerankK < topK) {
            throw new IllegalArgumentException(String.format("rerankK %d must be >= topK %d", rerankK, topK));
        }

        if (views.stream().allMatch(v -> v.entryNode() == null)) {
            return new MultiSearchResult(new MultiSearchResult.NodeScore[0], 0, 0, 0, 0, Float.POSITIVE_INFINITY);
        }

        initializeScoreProvider(scoreProviders);
        initializeBits(acceptOrds);

        // reset the scratch data structures
        approximateResults.clear();
        evictedResults.clear();
        candidates.clear();
        visited.clear();

        visitedCount = 0;
        expandedCount = 0;
        expandedCountBaseLayer = 0;

        // TODO this loop could be parallelized
        for (var iView = 0; iView < views.size(); iView++) {
            var searcher = searchers.get(iView);
            var view = views.get(iView);


            var entry = view.entryNode();
            if (entry != null) {
                var sp = scoreProviders.get(iView);

                // Only rerankK matters here, topK is not used
                searcher.internalSearch(sp, entry, topK, topK, threshold, acceptOrds.get(iView));

                int finalIView = iView;
                searcher.approximateResults.foreach((node, score) -> {
                    int internalNodeId = composeInternalNodeId(finalIView, node, views.size());
                    candidates.push(internalNodeId, score);
                    visited.add(internalNodeId);

                });

                visitedCount += searcher.getVisitedCount();
                expandedCount += searcher.getExpandedCount();
                expandedCountBaseLayer += searcher.getExpandedCountBaseLayer();
            }
        }

        // Now do the main search at layer 0
        return resume(topK, rerankK, threshold);
    }

    /**
     * @param scoreProviders  provides functions to return the similarity of a given node to the query vector
     * @param topK            the number of results to look for. With threshold=0, the search will continue until at least
     *                        `topK` results have been found, or until the entire graph has been searched.
     * @param threshold       the minimum similarity (0..1) to accept; 0 will accept everything. May be used
     *                        with a large topK to find (approximately) all nodes above the given threshold.
     *                        If threshold > 0 then the search will stop when it is probabilistically unlikely
     *                        to find more nodes above the threshold, even if `topK` results have not yet been found.
     * @param acceptOrds      a Bits instance indicating which nodes are acceptable results.
     *                        If {@link Bits#ALL}, all nodes are acceptable.
     *                        It is caller's responsibility to ensure that there are enough acceptable nodes
     *                        that we don't search the entire graph trying to satisfy topK.
     * @return a SearchResult containing the topK results and the number of nodes visited during the search.
     */
    public MultiSearchResult search(List<SearchScoreProvider> scoreProviders,
                                    int topK,
                                    float threshold,
                                    List<Bits> acceptOrds) {
        return search(scoreProviders, topK, topK, threshold, acceptOrds);
    }


    /**
     * @param scoreProviders  provides functions to return the similarity of a given node to the query vector for each index
     * @param topK            the number of results to look for. With threshold=0, the search will continue until at least
     *                        `topK` results have been found, or until the entire graph has been searched.
     * @param acceptOrds      a Bits instance indicating which nodes are acceptable results.
     *                        If {@link Bits#ALL}, all nodes are acceptable.
     *                        It is caller's responsibility to ensure that there are enough acceptable nodes
     *                        that we don't search the entire graph trying to satisfy topK.
     * @return a SearchResult containing the topK results and the number of nodes visited during the search.
     */
    public MultiSearchResult search(List<SearchScoreProvider> scoreProviders,
                                    int topK,
                                    List<Bits> acceptOrds)
    {
        return search(scoreProviders, topK, 0.0f, acceptOrds);
    }

    void initializeBits(List<Bits> rawAcceptOrds) {
        for (var iView = 0; iView < views.size(); iView++) {
            var bits = Bits.intersectionOf(rawAcceptOrds.get(iView), views.get(iView).liveNodes());
            this.acceptOrds.add(bits);
        }
    }

    private static int getGraphId(int internalNodeId, int nGraphs) {
        return internalNodeId % nGraphs;
    }

    private static int getNodeId(int internalNodeId, int nGraphs) {
        return internalNodeId / nGraphs;
    }

    private static int composeInternalNodeId(int iView, int nodeId, int nGraphs) {
        return nodeId * nGraphs + iView;
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
     * Modifies the internal search state.
     * When it's done, `approximateResults` contains the best `rerankK` results found at layer 0.
     *
     * @param scoreProviders      the current query's scoring/approximation logic for each index
     * @param rerankK             how many results to over-query for approximate ranking
     * @param threshold           similarity threshold, or 0f if none
     * @param acceptOrdsThisLayer a Bits instance indicating which nodes are acceptable results for each index
     *                            If {@link Bits#ALL}, all nodes are acceptable.
     *                            It is caller's responsibility to ensure that there are enough acceptable nodes
     *                            that we don't search the entire graph trying to satisfy topK.
     */
    void multiSearch(List<SearchScoreProvider> scoreProviders,
                     int rerankK,
                     float threshold,
                     MultiviewBits acceptOrdsThisLayer)
    {
        try {
            assert approximateResults.size() == 0; // should be cleared by setEntryPointsFromPreviousLayer
            approximateResults.setMaxSize(rerankK);

            // track scores to predict when we are done with threshold queries
            var scoreTracker = scoreTrackerFactory.getScoreTracker(pruneSearch, rerankK, threshold);
            VectorFloat<?> similarities = null;

            // the main search loop
            while (candidates.size() > 0) {
                if (stopSearch(candidates, scoreTracker, rerankK, threshold)) {
                    break;
                }

                // process the top candidate
                float topCandidateScore = candidates.topScore();
                int topCandidateNode = candidates.pop();
                int viewId = getGraphId(topCandidateNode, views.size());
                int nodeId = getNodeId(topCandidateNode, views.size());
                if (acceptOrdsThisLayer.get(viewId, nodeId) && topCandidateScore >= threshold) {
                    addTopCandidate(topCandidateNode, topCandidateScore, rerankK);
                }

                // skip edge loading if we've found a local maximum and we have enough results
                if (scoreTracker.shouldStop() && candidates.size() >= rerankK - approximateResults.size()) {
                    continue;
                }

                expandedCountBaseLayer++;
                expandedCount++;

                // score the neighbors of the top candidate and add them to the queue
                var scoreFunction = scoreProviders.get(viewId).scoreFunction();
                var useEdgeLoading = scoreFunction.supportsEdgeLoadingSimilarity();
                if (useEdgeLoading) {
                    similarities = scoreFunction.edgeLoadingSimilarityTo(nodeId);
                }
                int i = 0;
                for (var it = views.get(viewId).getNeighborsIterator(0, nodeId); it.hasNext(); ) {
                    var friendOrd = it.nextInt();
                    int internalNodeId = composeInternalNodeId(viewId, friendOrd, views.size());

                    if (!visited.add(internalNodeId)) {
                        continue;
                    }
                    visitedCount++;

                    float friendSimilarity = useEdgeLoading
                            ? similarities.get(i)
                            : scoreFunction.similarityTo(friendOrd);
                    scoreTracker.track(friendSimilarity);
                    candidates.push(internalNodeId, friendSimilarity);
                    i++;
                }
            }
        } catch (Throwable t) {
            // clear scratch structures if terminated via throwable, as they may not have been drained
            approximateResults.clear();
            throw t;
        }
    }

    MultiSearchResult resume(int topK, int rerankK, float threshold) {
        // rR is persistent to save on allocations
        rerankedResults.clear();
        rerankedResults.setMaxSize(topK);

        // add evicted results from the last call back to the candidates
        evictedResults.foreach(candidates::push);
        evictedResults.clear();

        multiSearch(scoreProviders, rerankK, threshold, acceptOrds);

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
            worstApproximateInTopK = approximateResults.rerank(topK, cachingReranker, 0.0f, rerankedResults, evictedResults);
            reranked = cachingReranker.getRerankCalls() - oldReranked;
            approximateResults.clear();
            popFromQueue = rerankedResults;
        }
        // pop the top K results from the results queue, which has the worst candidates at the top
        assert popFromQueue.size() <= topK;
        var nodes = new MultiSearchResult.NodeScore[popFromQueue.size()];
        for (int i = nodes.length - 1; i >= 0; i--) {
            var nScore = popFromQueue.topScore();
            var n = popFromQueue.pop();
            int viewId = getGraphId(n, views.size());
            int nodeId = getNodeId(n, views.size());
            nodes[i] = new MultiSearchResult.NodeScore(viewId, nodeId, nScore);
        }
        // that should be everything
        assert popFromQueue.size() == 0;

        return new MultiSearchResult(nodes, visitedCount, expandedCount, expandedCountBaseLayer, reranked, worstApproximateInTopK);
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

    /**
     * Experimental!
     * <p>
     * Resume the previous search where it left off and search for the best `additionalK` neighbors.
     * It is NOT valid to call this method before calling
     * `search`, but `resume` may be called as many times as desired once the search is initialized.
     * <p>
     * SearchResult.visitedCount resets with each call to `search` or `resume`.
     */
    @Experimental
    public MultiSearchResult resume(int additionalK, int rerankK) {
        visitedCount = 0;
        expandedCount = 0;
        expandedCountBaseLayer = 0;
        return resume(additionalK, rerankK, 0.0f);
    }

    @Override
    public void close() throws IOException {
        for (GraphIndex.View view : views) {
            view.close();
        }
    }

    private static class CachingReranker implements ScoreFunction.ExactScoreFunction {
        // this cache never gets cleared out (until a new search reinitializes it),
        // but we expect resume() to be called at most a few times so it's fine
        private final Int2ObjectHashMap<Float> cachedScores;
        private final List<SearchScoreProvider> scoreProviders;
        private int rerankCalls;

        public CachingReranker(List<SearchScoreProvider> scoreProviders) {
            this.scoreProviders = scoreProviders;
            cachedScores = new Int2ObjectHashMap<>();
            rerankCalls = 0;
        }

        @Override
        public float similarityTo(int node2) {
            if (cachedScores.containsKey(node2)) {
                return cachedScores.get(node2);
            }
            rerankCalls++;

            int viewId = getGraphId(node2, scoreProviders.size());
            int nodeId = getNodeId(node2, scoreProviders.size());

            float score = scoreProviders.get(viewId).reranker().similarityTo(nodeId);
            cachedScores.put(node2, Float.valueOf(score));
            return score;
        }

        public int getRerankCalls() {
            return rerankCalls;
        }
    }

    private class MultiviewBits {
        private List<Bits> acceptOrds;

        public MultiviewBits(int capacity) {
            acceptOrds = new ArrayList<>(capacity);
        }

        public boolean add(Bits bits) {
            return acceptOrds.add(bits);
        }

        public boolean get(int view, int index) {
            return get(view).get(index);
        }

        public Bits get(int view) {
            return acceptOrds.get(view);
        }
    }
}
