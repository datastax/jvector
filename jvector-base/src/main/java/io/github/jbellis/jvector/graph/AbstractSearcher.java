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
import org.agrona.collections.Int2ObjectHashMap;
import org.agrona.collections.IntHashSet;

import java.io.IOException;

/**
 * Base class containing the graph search algorithm. All fields and non-public methods live here;
 * {@link GraphSearcher} is the public facade that exposes configuration and construction.
 */
abstract class AbstractSearcher implements Searcher {

    protected ImmutableGraphIndex.View view;

    // Scratch data structures reused across search() calls
    private final NodeQueue candidates;
    final NodeQueue approximateResults;
    private final NodeQueue refinedResults;
    private final IntHashSet visited;
    private final NodesUnsorted evictedResults;

    // State saved across search/resume calls
    Bits acceptOrds;
    SearchScoreProvider scoreProvider;
    private CachingRefiner cachingRefiner;

    boolean pruneSearch;
    private final ScoreTracker.ScoreTrackerFactory scoreTrackerFactory;

    protected int visitedCount;
    protected int expandedCount;
    protected int expandedCountBaseLayer;

    AbstractSearcher(GraphIndex.View view) {
        this.view = view;
        this.candidates = new NodeQueue(new GrowableLongHeap(100), NodeQueue.Order.MAX_HEAP);
        this.evictedResults = new NodesUnsorted(100);
        this.approximateResults = new NodeQueue(new BoundedLongHeap(100), NodeQueue.Order.MIN_HEAP);
        this.refinedResults = new NodeQueue(new BoundedLongHeap(100), NodeQueue.Order.MIN_HEAP);
        this.visited = new IntHashSet();
        this.pruneSearch = true;
        this.scoreTrackerFactory = new ScoreTracker.ScoreTrackerFactory();
    }

    private void initializeScoreProvider(SearchScoreProvider scoreProvider) {
        this.scoreProvider = scoreProvider;
        if (scoreProvider.refiner() == null) {
            cachingRefiner = null;
            return;
        }
        cachingRefiner = new CachingRefiner(scoreProvider);
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
        return refining(topK, refineK, refineFloor);
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

        for (int lvl = entry.level; lvl > 0; lvl--) {
            searchOneLayer(scoreProvider, 1, 0.0f, lvl, Bits.ALL);
            assert approximateResults.size() == 1 : approximateResults.size();
            setEntryPointsFromPreviousLayer();
        }

        searchLayer0(topK, refineK, threshold);
    }

    void setEntryPointsFromPreviousLayer() {
        approximateResults.foreach(candidates::push);
        evictedResults.foreach(candidates::push);
        evictedResults.clear();
        approximateResults.clear();
    }

    void initializeInternal(SearchScoreProvider scoreProvider, NodeAtLevel entry, Bits rawAcceptOrds) {
        initializeScoreProvider(scoreProvider);
        this.acceptOrds = Bits.intersectionOf(rawAcceptOrds, view.liveNodes());

        approximateResults.clear();
        evictedResults.clear();
        candidates.clear();
        visited.clear();

        float score = scoreProvider.scoreFunction().similarityTo(entry.node);
        visited.add(entry.node);
        candidates.push(entry.node, score);

        visitedCount = 0;
        expandedCount = 0;
        expandedCountBaseLayer = 0;
    }

    private boolean stopSearch(NodeQueue localCandidates, ScoreTracker scoreTracker, int refineK, float threshold) {
        float topCandidateScore = localCandidates.topScore();
        if (approximateResults.size() >= refineK && topCandidateScore < approximateResults.topScore()) {
            return true;
        }
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
    // Finally: resume() also drives the use of CachingRefiner.
    void searchOneLayer(SearchScoreProvider scoreProvider,
                        int refineK,
                        float threshold,
                        int level,
                        Bits acceptOrdsThisLayer)
    {
        try {
            assert approximateResults.size() == 0;
            approximateResults.setMaxSize(refineK);

            var scoreTracker = scoreTrackerFactory.getScoreTracker(pruneSearch, refineK, threshold);

            while (candidates.size() > 0) {
                if (stopSearch(candidates, scoreTracker, refineK, threshold)) {
                    break;
                }

                float topCandidateScore = candidates.topScore();
                int topCandidateNode = candidates.pop();
                if (acceptOrdsThisLayer.get(topCandidateNode) && topCandidateScore >= threshold) {
                    addTopCandidate(topCandidateNode, topCandidateScore, refineK);
                }

                if (scoreTracker.shouldStop() && candidates.size() >= refineK - approximateResults.size()) {
                    continue;
                }

                if (level == 0) {
                    expandedCountBaseLayer++;
                }
                expandedCount++;

                var scoreFunction = scoreProvider.scoreFunction();
                ImmutableGraphIndex.NeighborProcessor neighborProcessor = (node2, score) -> {
                    scoreTracker.track(score);
                    candidates.push(node2, score);
                    visitedCount++;
                };
                view.processNeighbors(level, topCandidateNode, scoreFunction, visited::add, neighborProcessor);
            }
        } catch (Throwable t) {
            approximateResults.clear();
            throw t;
        }
    }

    private void searchLayer0(int topK, int refineK, float threshold) {
        refinedResults.clear();
        refinedResults.setMaxSize(topK);

        evictedResults.foreach(candidates::push);
        evictedResults.clear();

        searchOneLayer(scoreProvider, refineK, threshold, 0, acceptOrds);
    }

    SearchResult refining(int topK, int refineK, float refineFloor) {
        assert approximateResults.size() <= refineK;
        NodeQueue popFromQueue;
        float worstApproximateInTopK;
        int refined;
        if (cachingRefiner == null) {
            while (approximateResults.size() > topK) {
                var nScore = approximateResults.topScore();
                var n = approximateResults.pop();
                evictedResults.add(n, nScore);
            }
            refined = 0;
            worstApproximateInTopK = Float.POSITIVE_INFINITY;
            popFromQueue = approximateResults;
        } else {
            int oldRefined = cachingRefiner.getRefineCalls();
            worstApproximateInTopK = approximateResults.refine(topK, cachingRefiner, refineFloor, refinedResults, evictedResults);
            refined = cachingRefiner.getRefineCalls() - oldRefined;
            approximateResults.clear();
            popFromQueue = refinedResults;
        }
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
        return refining(topK, refineK, refineFloor);
    }

    @SuppressWarnings("StatementWithEmptyBody")
    private void addTopCandidate(int topCandidateNode, float topCandidateScore, int refineK) {
        if (approximateResults.size() < refineK) {
            approximateResults.push(topCandidateNode, topCandidateScore);
        } else if (topCandidateScore > approximateResults.topScore()) {
            int evictedNode = approximateResults.topNode();
            float evictedScore = approximateResults.topScore();
            evictedResults.add(evictedNode, evictedScore);
            approximateResults.push(topCandidateNode, topCandidateScore);
        } else {
            // score is exactly equal to the worst candidate; check neighbors without changing results queue
        }
    }

    @Override
    public void close() throws IOException {
        view.close();
    }

    private static class CachingRefiner implements ScoreFunction.ExactScoreFunction {
        // cache never cleared between resume() calls (we expect at most a few per search)
        private final Int2ObjectHashMap<Float> cachedScores;
        private final SearchScoreProvider scoreProvider;
        private int refineCalls;

        CachingRefiner(SearchScoreProvider scoreProvider) {
            this.scoreProvider = scoreProvider;
            cachedScores = new Int2ObjectHashMap<>();
            refineCalls = 0;
        }

        @Override
        public float similarityTo(int node2) {
            if (cachedScores.containsKey(node2)) {
                return cachedScores.get(node2);
            }
            refineCalls++;
            float score = scoreProvider.refiner().similarityTo(node2);
            cachedScores.put(node2, Float.valueOf(score));
            return score;
        }

        int getRefineCalls() {
            return refineCalls;
        }
    }
}
