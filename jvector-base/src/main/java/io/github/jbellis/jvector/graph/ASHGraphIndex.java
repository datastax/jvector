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
import io.github.jbellis.jvector.quantization.AsymmetricHashing;

/**
 * A graph index whose graph is built and searched with ASH codes. ASH is how the index compresses and scores its
 * vectors; the index is still a graph index, and is navigated, searched and written as any other.
 *
 * <p>Its {@link #queryPreparer} prepares queries for its ASH model, and its {@link #getView views} bind those
 * prepared queries with {@link GraphIndex.ScoringView#scoreProviderFor}.</p>
 */
@Experimental
public interface ASHGraphIndex extends PersistableGraphIndex {
    /** True once the index has an ASH model; until then it has no graph to search and no queries to prepare. */
    boolean hasModel();

    /** The model new vectors are encoded with, or null before the index has one. */
    AsymmetricHashing model();
}
