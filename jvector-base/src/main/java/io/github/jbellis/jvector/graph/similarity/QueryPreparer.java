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
 * Prepares independent queries against one immutable model on the calling
 * thread. Distinct output states may be prepared concurrently. The caller owns
 * scheduling, admission, input lifetime during each call, and output reuse.
 */
public interface QueryPreparer extends Accountable {
    /** Allocates one reusable state. Shared model weights are not copied. */
    PreparedQuery newState();

    /** Whether the current backend has a genuine multi-vector preparation path. */
    boolean supportsBatchPreparation();

    /** Snapshots and prepares one input, without participating in caller batching. */
    void prepare(VectorFloat<?> query, PreparedQuery preparedState);

    /**
     * Snapshots and prepares the first {@code count} inputs into distinct states
     * from this model. Zero is a no-op. Counts above eight are supported.
     * Structural validation precedes modification; a computation failure leaves
     * affected states unusable until successfully prepared again. Unused array
     * elements are ignored. Inputs must remain unmodified until this call returns.
     */
    void prepareBatch(VectorFloat<?>[] queries, PreparedQuery[] preparedStates, int count);
}
