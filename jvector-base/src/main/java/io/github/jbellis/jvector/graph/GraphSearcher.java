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
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.Closeable;


/**
 * Searches a graph to find nearest neighbors to a query vector.
 * A GraphSearcher is not threadsafe: only one search at a time should be run per GraphSearcher.
 * For more background on the search algorithm, see {@link ImmutableGraphIndex}.
 */
public interface GraphSearcher extends Closeable {
    /**
     * When using pruning, we are using a heuristic to terminate the search earlier.
     * In certain cases, it can lead to speedups. This is set to false by default.
     * @param usage a boolean that determines whether we do early termination or not.
     */
    void usePruning(boolean usage);

    /**
     * @param queryVector     the query vector
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
    SearchResult search(VectorFloat<?> queryVector,
                        int topK,
                        int rerankK,
                        float threshold,
                        float rerankFloor,
                        Bits acceptOrds);


    /**
     * @param queryVector     the query vector
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
    default SearchResult search(VectorFloat<?> queryVector,
                                int topK,
                                float threshold,
                                Bits acceptOrds) {
        return search(queryVector, topK, topK, threshold, 0.0f, acceptOrds);
    }


    /**
     * @param queryVector     the query vector
     * @param topK            the number of results to look for. With threshold=0, the search will continue until at least
     *                        `topK` results have been found, or until the entire graph has been searched.
     * @param acceptOrds      a Bits instance indicating which nodes are acceptable results.
     *                        If {@link Bits#ALL}, all nodes are acceptable.
     *                        It is caller's responsibility to ensure that there are enough acceptable nodes
     *                        that we don't search the entire graph trying to satisfy topK.
     * @return a SearchResult containing the topK results and the number of nodes visited during the search.
     */
    default SearchResult search(VectorFloat<?> queryVector,
                                int topK,
                                Bits acceptOrds) {
        return search(queryVector, topK, 0.0f, acceptOrds);
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
    SearchResult resume(int additionalK, int rerankK);
}
