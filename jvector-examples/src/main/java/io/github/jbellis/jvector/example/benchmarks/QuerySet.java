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

package io.github.jbellis.jvector.example.benchmarks;

import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * The queries a configured system answers, in its dataset's vector type ({@code Q}: float or int8 vectors), and how
 * to score one. Benchmarks measure through it, so each benchmark runs the same way for every vector type.
 */
public interface QuerySet<Q> {
    /** The number of dataset queries. */
    int size();

    /** Dataset query {@code i}. */
    Q get(int i);

    /** The number of base vectors. */
    int baseCount();

    /** A copy of base vector {@code ordinal}, to search as a query. */
    Q baseQuery(int ordinal);

    /** A new query of the dataset's dimension, for {@link #randomize} to fill. */
    Q newQuery();

    /** Fills {@code query} with random nonzero values, for warmup. */
    void randomize(Q query);

    /** A new array for {@code size} queries. */
    Q[] newArray(int size);

    /** The score provider for {@code query} over {@code view}. */
    SearchScoreProvider scoreProviderFor(Q query, ImmutableGraphIndex.View view);

    /**
     * {@code query} as a float vector, for query preparation, which projects float queries only.
     *
     * @throws UnsupportedOperationException if this set's queries are not float vectors
     */
    default VectorFloat<?> asFloat(Q query) {
        throw new UnsupportedOperationException("Query preparation needs float queries");
    }
}
