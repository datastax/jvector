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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.util.FixedBitSet;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;

/**
 * Bundle of inputs that {@link QuantizationCompactionStrategy} implementations need to do their work
 * during a single compaction run. Passed to {@code FusedFeature.createCompactionStrategy(...)}
 * and {@code CompressedVectors.createCompactionStrategy(...)} so strategy implementations can
 * capture exactly the pieces they need without the {@code OnDiskGraphIndexCompactor} leaking
 * through.
 */
public final class CompactionContext {
    public final List<OnDiskGraphIndex> sources;
    /** Parallel to {@link #sources}; {@code null} when no non-fused sidecar input is supplied. */
    public final List<CompressedVectors> sourceCompressed;
    public final List<FixedBitSet> liveNodes;
    public final List<OrdinalMapper> remappers;
    public final int dimension;
    public final int maxOrdinal;
    public final ExecutorService executor;
    public final int taskWindowSize;

    public CompactionContext(
            List<OnDiskGraphIndex> sources,
            List<CompressedVectors> sourceCompressed,
            List<FixedBitSet> liveNodes,
            List<OrdinalMapper> remappers,
            int dimension,
            int maxOrdinal,
            ExecutorService executor,
            int taskWindowSize) {
        this.sources = Collections.unmodifiableList(sources);
        this.sourceCompressed = sourceCompressed == null ? null : Collections.unmodifiableList(sourceCompressed);
        this.liveNodes = Collections.unmodifiableList(liveNodes);
        this.remappers = Collections.unmodifiableList(remappers);
        this.dimension = dimension;
        this.maxOrdinal = maxOrdinal;
        this.executor = executor;
        this.taskWindowSize = taskWindowSize;
    }
}
