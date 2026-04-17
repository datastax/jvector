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
 * Public interface for searching a graph index. Obtain an instance via {@link GraphIndex#searcher()}.
 * <p>
 * Searcher instances are <em>not</em> thread-safe — use one instance per thread.
 */
public interface Searcher extends Closeable {

    /**
     * Searches the index for the {@code topK} nearest neighbors.
     *
     * @param scoreProvider provides similarity scoring against the query
     * @param topK          the number of results to return
     * @param acceptOrds    filter for which nodes are eligible results; use {@link Bits#ALL} for unrestricted search
     * @return a {@link SearchResult} with the topK results and search statistics
     */
    SearchResult search(SearchScoreProvider scoreProvider, int topK, Bits acceptOrds);

    /**
     * Searches the index for the {@code topK} nearest neighbors, stopping early once further
     * improvements are probabilistically unlikely when a similarity threshold is set.
     *
     * @param scoreProvider provides similarity scoring against the query
     * @param topK          the number of results to return
     * @param threshold     minimum similarity to accept; {@code 0} accepts everything
     * @param acceptOrds    filter for which nodes are eligible results
     * @return a {@link SearchResult} with the topK results and search statistics
     */
    SearchResult search(SearchScoreProvider scoreProvider, int topK, float threshold, Bits acceptOrds);

    /**
     * Full-featured search with two-phase scoring support.
     *
     * @param scoreProvider provides similarity scoring against the query
     * @param topK          the number of results to return
     * @param refineK       the number of approximately-scored candidates to refine before returning the best {@code topK}
     * @param threshold     minimum similarity to accept; {@code 0} accepts everything
     * @param refineFloor   (Experimental) candidates whose approximate score meets this floor will be refined with
     *                      exact scoring; others are deferred to a potential {@code resume()} call
     * @param acceptOrds    filter for which nodes are eligible results
     * @return a {@link SearchResult} with the topK results and search statistics
     */
    @Experimental
    SearchResult search(SearchScoreProvider scoreProvider,
                        int topK,
                        int refineK,
                        float threshold,
                        float refineFloor,
                        Bits acceptOrds);
}
