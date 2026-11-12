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
import java.util.concurrent.ForkJoinPool;

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
    /**
     * Output ordinal of every source node; {@code null} in the context a strategy is built from,
     * set in the one {@code onRemappersUpdated} delivers once the compactor has assigned them.
     */
    public final List<OrdinalMapper> remappers;
    public final int dimension;
    public final int maxOrdinal;
    public final ForkJoinPool executor;
    public final int taskWindowSize;
    /**
     * The ordinal of {@link #remappers} written at each output ordinal, when the caller chose the output ordinals;
     * {@code null} when the output is numbered by {@link #remappers} itself.
     */
    private final int[] outputToMerged;

    public CompactionContext(
            List<OnDiskGraphIndex> sources,
            List<CompressedVectors> sourceCompressed,
            List<FixedBitSet> liveNodes,
            List<OrdinalMapper> remappers,
            int dimension,
            int maxOrdinal,
            ForkJoinPool executor,
            int taskWindowSize) {
        this(sources, sourceCompressed, liveNodes, remappers, dimension, maxOrdinal, executor, taskWindowSize, null);
    }

    /**
     * @param outputToMerged the ordinal of {@code remappers} written at each output ordinal, or {@code null} when the
     *                       output is numbered by {@code remappers}
     */
    public CompactionContext(
            List<OnDiskGraphIndex> sources,
            List<CompressedVectors> sourceCompressed,
            List<FixedBitSet> liveNodes,
            List<OrdinalMapper> remappers,
            int dimension,
            int maxOrdinal,
            ForkJoinPool executor,
            int taskWindowSize,
            int[] outputToMerged) {
        this.outputToMerged = outputToMerged;
        this.sources = Collections.unmodifiableList(sources);
        this.sourceCompressed = sourceCompressed == null ? null : Collections.unmodifiableList(sourceCompressed);
        this.liveNodes = Collections.unmodifiableList(liveNodes);
        this.remappers = remappers == null ? null : Collections.unmodifiableList(remappers);
        this.dimension = dimension;
        this.maxOrdinal = maxOrdinal;
        this.executor = executor;
        this.taskWindowSize = taskWindowSize;
    }

    /**
     * The ordinal of {@link #remappers} written at {@code outputOrdinal}: what an output laid out by ordinal, such as a
     * compressed sidecar, reads at that position.
     */
    public int mergedOrdinalAt(int outputOrdinal) {
        return outputToMerged == null ? outputOrdinal : outputToMerged[outputOrdinal];
    }
}
