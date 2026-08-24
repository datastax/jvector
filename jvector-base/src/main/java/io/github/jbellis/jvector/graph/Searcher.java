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

import java.io.Closeable;

/**
 * Searches an index for nearest neighbors to a query vector.
 * <p>
 * Obtain an instance via {@link ImmutableGraphIndex#searcher()}, {@link GraphSearcher#builder}, or by
 * constructing a {@link GraphSearcher} directly. Instances are <em>not</em> thread-safe -- use one per thread.
 * <p>
 * Configuration of how a search is performed (pruning heuristics, which {@link ImmutableGraphIndex.View}
 * is searched, etc.) is intentionally not part of this interface, since the parameters that make sense
 * differ by implementation; those are exposed on the concrete implementation type instead.
 */
public interface Searcher extends Closeable {

    /**
     * Performs a basic ANN search for the top K nearest neighbors.
     *
     * @param scoreProvider provides functions to return the similarity of a given node to the query vector
     * @param topK          the number of results to look for
     * @param acceptOrds    a Bits instance indicating which nodes are acceptable results.
     *                      If {@link Bits#ALL}, all nodes are acceptable.
     * @return a SearchResult containing the topK results and statistics about the search
     */
    SearchResult search(SearchScoreProvider scoreProvider, int topK, Bits acceptOrds);

    /**
     * Performs an ANN search with a similarity threshold. May be used with a large topK to find
     * (approximately) all nodes above the given threshold; the search will stop early once it is
     * probabilistically unlikely to find more nodes above the threshold, even if topK results have
     * not yet been found.
     *
     * @param scoreProvider provides functions to return the similarity of a given node to the query vector
     * @param topK          the number of results to look for
     * @param threshold     the minimum similarity (0..1) to accept; 0 accepts everything
     * @param acceptOrds    a Bits instance indicating which nodes are acceptable results
     * @return a SearchResult containing the topK results and statistics about the search
     */
    SearchResult search(SearchScoreProvider scoreProvider, int topK, float threshold, Bits acceptOrds);

    /**
     * Full-featured search supporting two-phase (approximate, then exact) scoring.
     *
     * @param scoreProvider provides functions to return the similarity of a given node to the query vector
     * @param topK          the number of results to look for
     * @param refineK       the number of approximately-scored candidates to refine (i.e. rescore exactly)
     *                      before returning the best topK
     * @param threshold     the minimum similarity (0..1) to accept; 0 accepts everything
     * @param refineFloor   (Experimental) candidates whose approximate score is at least this value will
     *                      be refined with exact scoring; others are discarded but may be resurfaced by a
     *                      subsequent resume() call. Intended for use when combining results from multiple indexes.
     * @param acceptOrds    a Bits instance indicating which nodes are acceptable results
     * @return a SearchResult containing the topK results and statistics about the search
     */
    @Experimental
    SearchResult search(SearchScoreProvider scoreProvider,
                        int topK,
                        int refineK,
                        float threshold,
                        float refineFloor,
                        Bits acceptOrds);
}
