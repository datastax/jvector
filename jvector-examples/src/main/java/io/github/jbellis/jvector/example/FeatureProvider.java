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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.quantization.PQVectors;

/**
 * How {@link Grid} writes and uses one on-disk {@link FeatureId}.
 *
 * <p>To benchmark a new feature, implement this in its own file and register it in {@link FeatureProviders}.</p>
 */
interface FeatureProvider {
    FeatureId id();

    /** Adds the feature to one graph index writer and registers the suppliers for its per-node state. */
    void configure(FeatureWriteContext ctx);

    /** Whether the feature can be written from a graph built on full-precision vectors. */
    default boolean supportsInMemoryBuild() {
        return true;
    }

    /** Whether the feature stores vectors the index can rerank with. Compressed builds need at least one. */
    default boolean providesRerankVectors() {
        return false;
    }

    /** Whether searches score approximately from the index itself, so no search-time compression is applied. */
    default boolean providesApproximateScoring() {
        return false;
    }

    /**
     * Rejects build compression this feature cannot be written with, before any graph is built.
     *
     * @param buildPQ the PQ-encoded build vectors, or null when the build used another compressor
     */
    default void checkCompressedBuild(PQVectors buildPQ) {
    }
}
