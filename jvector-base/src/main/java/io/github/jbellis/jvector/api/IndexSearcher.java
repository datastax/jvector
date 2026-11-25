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

import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.Closeable;

/**
 * A backing's search entry point, returned by {@link Index#searcher()}.
 * Each backing (e.g. {@code GraphSearcher} for graph/HNSW indexes, {@code IvfSearcher} for IVF)
 * implements this with its own type-specific search options; a caller holding only {@link Index}
 * narrows to the backing's own index interface (e.g. {@code GraphIndex}) to recover the concrete
 * searcher type instead of casting.
 * <p>
 * A searcher may hold resources (e.g. a graph searcher's view of an on-disk index holds a file
 * reader), so it must be closed when no longer needed; being {@link Closeable} lets code holding
 * only {@link Index} do that with try-with-resources, without knowing the backing.
 */
public interface IndexSearcher extends Closeable {
    /**
     * Searches for the {@code topK} vectors nearest {@code query}, scoring with what the index itself knows: the
     * similarity function it records and the vectors or codes it stores. No similarity function or score provider is
     * passed, so a search can't use a function the index wasn't built for.
     * <p>
     * The default implementation throws {@link UnsupportedOperationException}; backings that can score a query on
     * their own override it.
     *
     * @param rerankK    how many candidates to collect before keeping the best {@code topK}; at least {@code topK}
     * @param acceptOrds which ordinals are acceptable results; {@link Bits#ALL} for all
     * @throws IllegalStateException if the index records no similarity function, or stores nothing to score with;
     *                               search it through the backing's own API with an explicit function or score provider
     */
    default SearchResult search(VectorFloat<?> query, int topK, int rerankK, Bits acceptOrds) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not search by the index's own "
                + "similarity function; use the backing's own search API");
    }
}
