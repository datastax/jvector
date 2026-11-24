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

package io.github.jbellis.jvector.api;

import io.github.jbellis.jvector.vector.ByteVectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.IOException;
import java.util.Optional;

/**
 * The backing-agnostic handle for a vector index. Every concrete index type (graph/HNSW, IVF, and
 * whatever follows) implements this; callers that don't need to know which backing they hold can
 * program against {@link Index} alone.
 * <p>
 * Callers who <em>do</em> know (or need to recover) the concrete backing use its own interface
 * instead &mdash; e.g. {@code GraphIndex} or {@code IvfIndex} &mdash; which each override
 * {@link #searcher()} to return their own, more specific {@link IndexSearcher} subtype with no
 * cast required.
 */
public interface Index extends AutoCloseable {

    /**
     * Returns an estimate of the memory, in bytes, this index uses on the heap.
     */
    long ramBytesUsed();

    /**
     * Returns a new {@link IndexSearcher} of the type appropriate for this Index. The caller owns it
     * and must close it, e.g. {@code try (IndexSearcher searcher = index.searcher()) { ... }}.
     */
    IndexSearcher searcher();

    /**
     * Returns the similarity function this index was built with, or empty if it doesn't know one: for
     * example, an index built from a custom score provider that names no single function, or one loaded
     * from an on-disk format that doesn't record it.
     * <p>
     * The default implementation returns empty.
     */
    default Optional<VectorSimilarityFunction> similarityFunction() {
        return Optional.empty();
    }

    /**
     * Returns the int8 similarity function this index was built with, for an index of int8 vectors, or empty
     * if it doesn't know one. An index reports at most one of this and {@link #similarityFunction()}.
     * <p>
     * The default implementation returns empty.
     */
    default Optional<ByteVectorSimilarityFunction> byteSimilarityFunction() {
        return Optional.empty();
    }

    /**
     * Releases any resources this index holds. Declared to throw only {@link IOException}, like
     * {@link IndexSearcher#close()}, so code holding just an {@code Index} can use try-with-resources
     * without catching {@code Exception}.
     */
    @Override
    void close() throws IOException;
}
