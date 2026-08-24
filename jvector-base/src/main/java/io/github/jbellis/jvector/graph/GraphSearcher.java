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
 * Searches a graph to find nearest neighbors to a query vector. For more background on the
 * search algorithm, see {@link ImmutableGraphIndex}.
 * <p>
 * Obtain an instance via {@link ImmutableGraphIndex#searcher()}, {@link #builder}, or by constructing
 * one directly. Instances are <em>not</em> thread-safe -- use one per thread.
 *
 * @see Searcher
 */
public class GraphSearcher extends AbstractSearcher {

    /**
     * Creates a new graph searcher from the given GraphIndex
     */
    public GraphSearcher(ImmutableGraphIndex graph) {
        this(graph.getView());
    }

    /**
     * Creates a new graph searcher from the given GraphIndex.View
     */
    protected GraphSearcher(ImmutableGraphIndex.View view) {
        super(view);
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

    public ImmutableGraphIndex.View getView() {
        return view;
    }

    /**
     * Exposes the internal approximate-results queue populated by the most recent
     * {@link #searchOneLayer} call. Intended for cross-package internal use (e.g. graph
     * compaction); not part of the stable public API.
     */
    public NodeQueue approximateResults() {
        return approximateResults;
    }

    /**
     * @deprecated TopK and filtered graph-search pruning is disabled because the
     * existing heuristic can reduce recall and has not shown reliable production
     * value. This method is retained for API compatibility and has no effect.
     *
     * Threshold searches, where {@code threshold > 0}, continue to use their
     * legacy threshold early-termination behavior that disregards this setting.
     */
    @Deprecated
    public void usePruning(boolean usage) {
        pruneSearch = false;
    }

    /**
     * Convenience function for simple one-off searches.  It is caller's responsibility to make sure that it
     * is the unique owner of the vectors instance passed in here.
     */
    public static SearchResult search(VectorFloat<?> queryVector, int topK, RandomAccessVectorValues vectors, VectorSimilarityFunction similarityFunction, ImmutableGraphIndex graph, Bits acceptOrds) {
        try (var searcher = new GraphSearcher(graph)) {
            var ssp = DefaultSearchScoreProvider.exact(queryVector, similarityFunction, vectors);
            return searcher.search(ssp, topK, acceptOrds);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Convenience function for simple one-off searches.  It is caller's responsibility to make sure that it
     * is the unique owner of the vectors instance passed in here.
     */
    public static SearchResult search(VectorFloat<?> queryVector, int topK, int refineK, RandomAccessVectorValues vectors, VectorSimilarityFunction similarityFunction, ImmutableGraphIndex graph, Bits acceptOrds) {
        try (var searcher = new GraphSearcher(graph)) {
            var ssp = DefaultSearchScoreProvider.exact(queryVector, similarityFunction, vectors);
            return searcher.search(ssp, topK, refineK, 0.f, 0.f, acceptOrds);
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
    public void setView(ImmutableGraphIndex.View view) {
        this.view = view;
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
    public SearchResult resume(int additionalK, int refineK) {
        visitedCount = 0;
        expandedCount = 0;
        expandedCountBaseLayer = 0;
        return resume(additionalK, refineK, 0.0f, 0.0f);
    }

    /**
     * Returns a fluent builder for configuring and constructing a {@link GraphSearcher}.
     */
    public static Builder builder(ImmutableGraphIndex graph) {
        return new Builder(graph.getView());
    }

    /**
     * Returns a fluent builder for configuring and constructing a {@link GraphSearcher} over an
     * existing view (e.g. a view held open across multiple searches).
     */
    public static Builder builder(ImmutableGraphIndex.View view) {
        return new Builder(view);
    }

    /**
     * Fluent builder for {@link GraphSearcher}.
     */
    public static final class Builder {
        private final ImmutableGraphIndex.View view;

        /**
         * Prefer {@link GraphSearcher#builder(ImmutableGraphIndex)} or
         * {@link GraphSearcher#builder(ImmutableGraphIndex.View)}; this constructor is public for
         * source compatibility with earlier versions that only supported constructing a Builder directly.
         */
        public Builder(ImmutableGraphIndex.View view) {
            this.view = view;
        }

        /**
         * @deprecated pruning is permanently disabled (see {@link GraphSearcher#usePruning}); retained
         * for fluent-API completeness, has no effect.
         */
        @Deprecated
        public Builder usePruning(boolean usage) {
            return this;
        }

        /**
         * @deprecated GraphSearcher has always been usable concurrently by holding one instance per
         * thread; this setting has always been a no-op. Retained for source compatibility.
         */
        @Deprecated
        public Builder withConcurrentUpdates() {
            return this;
        }

        public GraphSearcher build() {
            return new GraphSearcher(view);
        }
    }
}
