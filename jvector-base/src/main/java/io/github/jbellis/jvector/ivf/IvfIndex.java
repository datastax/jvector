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

package io.github.jbellis.jvector.ivf;

import io.github.jbellis.jvector.index.Index;

/**
 * An IVF-backed vector index. Mirrors the role {@code GraphIndex} plays for graph/HNSW indexes:
 * the public handle a caller gets back from {@link IvfIndexBuilder#build()}, or recovers from a
 * generic {@link Index} via {@code instanceof IvfIndex} when the caller doesn't already hold the
 * concrete type.
 * <p>
 * Deliberately kept to the public handle surface only &mdash; nothing an internal search
 * algorithm needs (e.g. walking centroids/posting lists) belongs on this interface. That
 * distinction wasn't maintained for {@code GraphIndex} historically (its {@code View}/traversal
 * types ended up here too); IVF has no such legacy shape to preserve, so it starts clean.
 * <p>
 * Still a seam only: there is no concrete implementation yet, pending the IVF design itself.
 */
public interface IvfIndex extends Index {

    @Override
    IvfSearcher searcher();
}
