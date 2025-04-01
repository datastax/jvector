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
import io.github.jbellis.jvector.graph.GraphIndex.NodeAtLevel;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.BoundedLongHeap;
import io.github.jbellis.jvector.util.GrowableLongHeap;
import io.github.jbellis.jvector.util.SparseBits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.Int2ObjectHashMap;
import org.agrona.collections.IntHashSet;

import java.io.Closeable;
import java.io.IOException;


/**
 * Searches a graph to find nearest neighbors to a query vector. For more background on the
 * search algorithm, see {@link GraphIndex}.
 */
public class GraphSearcher implements Closeable {
    private boolean pruneSearch;

    private GraphIndex.View view;

    // Scratch data structures that are used in each {@link #searchInternal} call. These can be expensive
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

    /**
     * Creates a new graph searcher from the given GraphIndex
     */
    public GraphSearcher(GraphIndex graph) {
        this(graph.getView());
    }

    private GraphSearcher(GraphIndex.View view) {
        this.view = view;
        this.candidates = new NodeQueue(new GrowableLongHeap(100), NodeQueue.Order.MAX_HEAP);
        this.evictedResults = new NodesUnsorted(100);
        this.approximateResults = new NodeQueue(new BoundedLongHeap(100), NodeQueue.Order.MIN_HEAP);
        this.rerankedResults = new NodeQueue(new BoundedLongHeap(100), NodeQueue.Order.MIN_HEAP);
        this.visited = new IntHashSet();
        this.pruneSearch = false;
    }

    private void initializeScoreProvider(SearchScoreProvider scoreProvider) {
        this.scoreProvider = scoreProvider;
        if (scoreProvider.reranker() == null) {
            cachingReranker = null;
            return;
        }

        cachingReranker = new CachingReranker(scoreProvider);
    }

    public GraphIndex.View getView() {
        return view;
    }

    /**
     * Whe using pruning, we are using a heuristic to terminate the search earlier.
     * In certain cases, it can lead to speedups. This is set to false by default.
     * @param usage a boolean that determines whether we do early termination or not.
     */
    public void usePruning(boolean usage) {
        pruneSearch = usage;
    }

    /**
     * Convenience function for simple one-off searches.  It is caller's responsibility to make sure that it
     * is the unique owner of the vectors instance passed in here.
     */
    public static SearchResult search(VectorFloat<?> queryVector, int topK, RandomAccessVectorValues vectors, VectorSimilarityFunction similarityFunction, GraphIndex graph, Bits acceptOrds) {
        try (var searcher = new GraphSearcher(graph)) {
            var ssp = SearchScoreProvider.exact(queryVector, similarityFunction, vectors);
            return searcher.search(ssp, topK, acceptOrds);
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
     * @param view the new view
     */
    public void setView(GraphIndex.View view) {
        this.view = view;
    }

    /**
     * Call GraphSearcher constructor instead
     */
    @Deprecated
    public static class Builder {
        private final GraphIndex.View view;

        public Builder(GraphIndex.View view) {
            this.view = view;
        }

        public Builder withConcurrentUpdates() {
            return this;
        }

        public GraphSearcher build() {
            return new GraphSearcher(view);
        }
    }

    /**
     * @param scoreProvider   provides functions to return the similarity of a given node to the query vector
     * @param topK            the number of results to look for. With threshold=0, the search will continue until at least
     *                        `topK` results have been found, or until the entire graph has been searched.
     * @param rerankK         the number of (approximately-scored) results to rerank before returning the best `topK`.
     * @param threshold       the minimum similarity (0..1) to accept; 0 will accept everything. May be used
     *                        with a large topK to find (approximately) all nodes above the given threshold.
     *                        If threshold > 0 then the search will stop when it is probabilistically unlikely
     *                        to find more nodes above the threshold, even if `topK` results have not yet been found.
     * @param rerankFloor     (Experimental!) Candidates whose approximate similarity is at least this value
     *                        will be reranked with the exact score (which requires loading a high-res vector from disk)
     *                        and included in the final results.  (Potentially leaving fewer than topK entries
     *                        in the results.)  Other candidates will be discarded, but will be potentially
     *                        resurfaced if `resume` is called.  This is intended for use when combining results
     *                        from multiple indexes.
     * @param acceptOrds      a Bits instance indicating which nodes are acceptable results.
     *                        If {@link Bits#ALL}, all nodes are acceptable.
     *                        It is caller's responsibility to ensure that there are enough acceptable nodes
     *                        that we don't search the entire graph trying to satisfy topK.
     * @return a SearchResult containing the topK results and the number of nodes visited during the search.
     */
    @Experimental
    public SearchResult search(SearchScoreProvider scoreProvider,
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
            return new SearchResult(new SearchResult.NodeScore[0], 0, 0, Float.POSITIVE_INFINITY);
        }

        initializeInternal(scoreProvider, entry, acceptOrds);

        // Move downward from entry.level to 1
        int numVisited = 0;
        for (int lvl = entry.level; lvl > 0; lvl--) {
            // Search this layer with minimal parameters since we just want the best candidate
            numVisited += searchOneLayer(scoreProvider, 1, 0.0f, lvl, Bits.ALL);
            assert approximateResults.size() == 1 : approximateResults.size();
            setEntryPointsFromPreviousLayer();
        }

        // Now do the main search at layer 0
        return resume(numVisited, topK, rerankK, threshold, rerankFloor);
    }

    /**
     * @param scoreProvider   provides functions to return the similarity of a given node to the query vector
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
    public SearchResult search(SearchScoreProvider scoreProvider,
                               int topK,
                               float threshold,
                               Bits acceptOrds) {
        return search(scoreProvider, topK, topK, threshold, 0.0f, acceptOrds);
    }


    /**
     * @param scoreProvider   provides functions to return the similarity of a given node to the query vector
     * @param topK            the number of results to look for. With threshold=0, the search will continue until at least
     *                        `topK` results have been found, or until the entire graph has been searched.
     * @param acceptOrds      a Bits instance indicating which nodes are acceptable results.
     *                        If {@link Bits#ALL}, all nodes are acceptable.
     *                        It is caller's responsibility to ensure that there are enough acceptable nodes
     *                        that we don't search the entire graph trying to satisfy topK.
     * @return a SearchResult containing the topK results and the number of nodes visited during the search.
     */
    public SearchResult search(SearchScoreProvider scoreProvider,
                               int topK,
                               Bits acceptOrds)
    {
        return search(scoreProvider, topK, 0.0f, acceptOrds);
    }

    void setEntryPointsFromPreviousLayer() {
        // push the candidates seen so far back onto the queue for the next layer
        // at worst we save recomputing the similarity; at best we might connect to a more distant cluster
        approximateResults.foreach(candidates::push);
        evictedResults.foreach(candidates::push);
        evictedResults.clear();
        approximateResults.clear();
    }

    void initializeInternal(SearchScoreProvider scoreProvider, NodeAtLevel entry, Bits rawAcceptOrds) {
        // save search parameters for potential later resume
        initializeScoreProvider(scoreProvider);
        this.acceptOrds = Bits.intersectionOf(rawAcceptOrds, view.liveNodes());

        // reset the scratch data structures
        evictedResults.clear();
        candidates.clear();
        visited.clear();

        // Start with entry point
        float score = scoreProvider.scoreFunction().similarityTo(entry.node);
        visited.add(entry.node);
        candidates.push(entry.node, score);
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
    int searchOneLayer(SearchScoreProvider scoreProvider,
                       int rerankK,
                       float threshold,
                       int level,
                       Bits acceptOrdsThisLayer)
    {
        try {
            assert approximateResults.size() == 0; // should be cleared by setEntryPointsFromPreviousLayer
            approximateResults.setMaxSize(rerankK);

            int numVisited = 0;
            // track scores to predict when we are done with threshold queries
            var scoreTracker = threshold > 0
                    ? new ScoreTracker.TwoPhaseTracker(threshold)
                    : pruneSearch ? new ScoreTracker.RelaxedMonotonicityTracker(rerankK) : new ScoreTracker.NoOpTracker();
            VectorFloat<?> similarities = null;

            // the main search loop
            while (candidates.size() > 0) {
                // we're done when we have K results and the best candidate is worse than the worst result so far
                float topCandidateScore = candidates.topScore();
                if (approximateResults.size() >= rerankK && topCandidateScore < approximateResults.topScore()) {
                    break;
                }
                // when querying by threshold, also stop when we are probabilistically unlikely to find more qualifying results
                if (threshold > 0 && scoreTracker.shouldStop()) {
                    break;
                }

                // process the top candidate
                int topCandidateNode = candidates.pop();
                if (acceptOrdsThisLayer.get(topCandidateNode) && topCandidateScore >= threshold) {
                    addTopCandidate(topCandidateNode, topCandidateScore, rerankK);
                }

                // skip edge loading if we've found a local maximum and we have enough results
                if (scoreTracker.shouldStop() && candidates.size() >= rerankK - approximateResults.size()) {
                    continue;
                }

                // score the neighbors of the top candidate and add them to the queue
                var scoreFunction = scoreProvider.scoreFunction();
                var useEdgeLoading = scoreFunction.supportsEdgeLoadingSimilarity();
                if (useEdgeLoading) {
                    similarities = scoreFunction.edgeLoadingSimilarityTo(topCandidateNode);
                }
                int i = 0;
                for (var it = view.getNeighborsIterator(level, topCandidateNode); it.hasNext(); ) {
                    var friendOrd = it.nextInt();
                    if (!visited.add(friendOrd)) {
                        continue;
                    }
                    numVisited++;

                    float friendSimilarity = useEdgeLoading
                            ? similarities.get(i)
                            : scoreFunction.similarityTo(friendOrd);
                    scoreTracker.track(friendSimilarity);
                    candidates.push(friendOrd, friendSimilarity);
                    i++;
                }
            }

            return numVisited;
        } catch (Throwable t) {
            // clear scratch structures if terminated via throwable, as they may not have been drained
            approximateResults.clear();
            throw t;
        }
    }

    SearchResult resume(int numVisited, int topK, int rerankK, float threshold, float rerankFloor) {
        // rR is persistent to save on allocations
        rerankedResults.clear();
        rerankedResults.setMaxSize(topK);

        // add evicted results from the last call back to the candidates
        var previouslyEvicted = evictedResults.size() > 0 ? new SparseBits() : Bits.NONE;
        evictedResults.foreach((node, score) -> {
            candidates.push(node, score);
            ((SparseBits) previouslyEvicted).set(node);
        });
        evictedResults.clear();
        
        numVisited += searchOneLayer(scoreProvider, rerankK, threshold, 0, acceptOrds);

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

        return new SearchResult(nodes, numVisited, reranked, worstApproximateInTopK);
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
    public SearchResult resume(int additionalK, int rerankK) {
        return resume(0, additionalK, rerankK, 0.0f, 0.0f);
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

        public CachingReranker(SearchScoreProvider scoreProvider) {
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

        public int getRerankCalls() {
            return rerankCalls;
        }
    }
}
