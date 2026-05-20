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
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;


/**
 * Searches a graph-based vector index for nearest neighbors.
 * <p>
 * Obtain an instance via {@link GraphIndex#searcher()} or construct one directly.
 * Instances are <em>not</em> thread-safe — use one per thread.
 *
 * @see Searcher
 */
public class GraphSearcher extends AbstractSearcher {

    /**
     * Creates a new searcher for the given graph, opening a fresh view.
     */
    public GraphSearcher(GraphIndex graph) {
        this(graph.getView());
    }

    /**
     * Creates a new searcher from an existing view.
     */
    public GraphSearcher(GraphIndex.View view) {
        super(view);
    }

    /**
     * Returns the graph view currently being searched.
     */
    public PersistableGraphIndex.View getView() {
        return view;
    }

    /**
     * Replaces the view used by this searcher without allocating a new instance.
     * Useful when the underlying graph is concurrently mutated and the caller wants
     * to periodically refresh the view without creating a new {@code GraphSearcher}.
     *
     * @param view the new view to use
     */
    public void setView(PersistableGraphIndex.View view) {
        this.view = view;
    }

    /**
     * Enables or disables the pruning heuristic that allows early termination of the search.
     * Pruning is enabled by default.
     *
     * @param usage {@code true} to enable pruning, {@code false} to disable
     */
    public void usePruning(boolean usage) {
        pruneSearch = usage;
    }

    /**
     * Experimental! Resumes the previous search to find additional nearest neighbors.
     * <p>
     * May be called any number of times after an initial {@link #search} call.
     * Statistics (visitedCount etc.) reset with each call.
     */
    @Experimental
    public SearchResult resume(int additionalK, int refineK) {
        visitedCount = 0;
        expandedCount = 0;
        expandedCountBaseLayer = 0;
        return resume(additionalK, refineK, 0.0f, 0.0f);
    }

    /**
     * One-off convenience search with exact scoring and no refining.
     * The caller must be the unique owner of the {@code vectors} instance.
     */
    public static SearchResult search(VectorFloat<?> queryVector, int topK, RandomAccessVectorValues vectors, VectorSimilarityFunction similarityFunction, PersistableGraphIndex graph, Bits acceptOrds) {
        try (var searcher = new GraphSearcher(graph)) {
            var ssp = DefaultSearchScoreProvider.exact(queryVector, similarityFunction, vectors);
            return searcher.search(ssp, topK, acceptOrds);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * One-off convenience search with exact scoring and a refine pass.
     * The caller must be the unique owner of the {@code vectors} instance.
     */
    public static SearchResult search(VectorFloat<?> queryVector, int topK, int refineK, RandomAccessVectorValues vectors, VectorSimilarityFunction similarityFunction, PersistableGraphIndex graph, Bits acceptOrds) {
        try (var searcher = new GraphSearcher(graph)) {
            var ssp = DefaultSearchScoreProvider.exact(queryVector, similarityFunction, vectors);
            return searcher.search(ssp, topK, refineK, 0.f, 0.f, acceptOrds);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Deprecated
    public static class Builder {
        private final PersistableGraphIndex.View view;

        public Builder(PersistableGraphIndex.View view) {
            this.view = view;
        }

        public Builder withConcurrentUpdates() {
            return this;
        }

        public GraphSearcher build() {
            return new GraphSearcher(view);
        }
    }
}
