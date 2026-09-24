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

/**
 * Marker interface for a backing's search entry point, returned by {@link Index#searcher()}.
 * Each backing (e.g. {@code GraphSearcher} for graph/HNSW indexes, {@code IvfSearcher} for IVF)
 * implements this with its own type-specific search options; a caller holding only {@link Index}
 * narrows to the backing's own index interface (e.g. {@code GraphIndex}) to recover the concrete
 * searcher type instead of casting this marker.
 */
public interface IndexSearcher {
}
