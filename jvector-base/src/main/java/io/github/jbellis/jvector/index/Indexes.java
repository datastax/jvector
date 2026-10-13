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

package io.github.jbellis.jvector.index;

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.api.Index;
import io.github.jbellis.jvector.graph.HnswIndexBuilder;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RavvHnswBuilder;
import io.github.jbellis.jvector.graph.ScoreProviderHnswBuilder;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.ivf.IvfIndexBuilder;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

/**
 * Entry point for building a jvector {@link Index}: pick the backing type first, then only its
 * own parameters are available to set.
 * <p>
 * This lives here rather than as static methods on {@link Index} itself so that the
 * {@code io.github.jbellis.jvector.api} package, which holds the backing-agnostic contract
 * ({@link Index} and its searcher), refers to no concrete backing. That keeps the package free to
 * move into a separate contract module later without changes for callers.
 */
public final class Indexes {
    private Indexes() {
    }

    /**
     * Returns a builder for a graph/HNSW index over {@code vectorValues}, compared with
     * {@code similarityFunction}. The vectors may be compressed before building; see
     * {@link HnswIndexBuilder#withCompressionType}.
     */
    public static HnswIndexBuilder hnswBuilder(RandomAccessVectorValues vectorValues, VectorSimilarityFunction similarityFunction) {
        return new RavvHnswBuilder(vectorValues, similarityFunction);
    }

    /**
     * Returns a builder for a graph/HNSW index of vectors of the given {@code dimension}, scored with
     * {@code scoreProvider} (e.g. a PQ- or BQ-based provider). The vectors themselves are supplied as
     * nodes are added.
     */
    public static HnswIndexBuilder hnswBuilder(BuildScoreProvider scoreProvider, int dimension) {
        return new ScoreProviderHnswBuilder(scoreProvider, dimension);
    }

    /** Experimental: IVF is not implemented yet, so the returned builder always refuses to build. */
    @Experimental
    public static IvfIndexBuilder ivfBuilder() {
        return new IvfIndexBuilder();
    }
}
