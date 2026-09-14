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

import io.github.jbellis.jvector.util.Accountable;

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
public interface Index extends Accountable, AutoCloseable {

    /**
     * Returns a new {@link IndexSearcher} of the type appropriate for this Index.
     */
    IndexSearcher searcher();
}
