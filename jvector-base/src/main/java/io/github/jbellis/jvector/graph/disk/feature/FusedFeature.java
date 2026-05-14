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

package io.github.jbellis.jvector.graph.disk.feature;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.disk.CompactionContext;
import io.github.jbellis.jvector.graph.disk.QuantizationCompactionStrategy;
import io.github.jbellis.jvector.util.Accountable;

import java.io.IOException;

/**
 * A fused feature is one that is computed from the neighbors of a node.
 * - writeInline writes the fused features based on the neighbors of the node
 * - writeSource writes the feature of the node itself
 * Implements Quick ADC-style scoring by fusing PQ-encoded neighbors into an OnDiskGraphIndex.
 */
public interface FusedFeature extends Feature {
    default boolean isFused() {
        return true;
    }

    void writeSourceFeature(IndexWriter out, State state) throws IOException;

    interface InlineSource extends Accountable {}

    InlineSource loadSourceFeature(RandomAccessReader in) throws IOException;

    /**
     * Bytes occupied on disk by a single stored code (one neighbor's payload). For fused
     * features {@code featureSize() == codeSize() * maxDegree}. Used by callers that allocate
     * per-thread scratch buffers sized for a single code, e.g. cross-source scoring during
     * compaction.
     */
    int codeSize();

    /**
     * Returns the {@link QuantizationCompactionStrategy} the compactor should run when merging graphs that
     * carry this fused feature. One strategy instance per compaction; it owns any transient state
     * (retrained codebook, pre-encode caches) until the compactor releases it via
     * {@link QuantizationCompactionStrategy#onAfterClose}.
     * <p>
     * Implementations must return a fresh strategy on every call — feature instances themselves
     * are read-mostly objects that may be shared by concurrent readers of the source graph.
     */
    QuantizationCompactionStrategy createCompactionStrategy(CompactionContext ctx);
}
