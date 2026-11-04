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
package io.github.jbellis.jvector.graph.similarity;

import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Reusable, model-specific numerical preparation for one query. This state owns
 * its query snapshot and scoring buffers, but no graph view or traversal state.
 *
 * <p>After preparation it can be read by searches using the same model. The
 * owner must not prepare it again until all searches and resumable result
 * iterators borrowing it have finished. Publishing it to another thread requires
 * the usual executor, queue or other happens-before handoff. No executor or
 * request queue is created by this API.
 */
public interface PreparedQuery extends Accountable {
    /** Borrowed FP32 query snapshot for reranking; callers must not modify it. */
    VectorFloat<?> queryVector();
}
