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

import io.github.jbellis.jvector.graph.disk.feature.FusedFeature;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.file.Path;
import java.util.List;

/**
 * Encapsulates the quantization-aware steps the compactor needs to run during a single
 * {@code compact()} invocation. Pulling these behind a strategy lets the compactor body stay
 * scheme-agnostic: it asks the strategy whether to write inline codes, hands it pre/post hooks,
 * and (for sidecar strategies) defers the merged sidecar write to the strategy.
 * <p>
 * One strategy instance per compaction run. Strategies are stateful — they hold the retrained
 * compressor produced by {@link #retrain} and any transient resources (e.g. memory-mapped
 * pre-encode caches) until {@link #onAfterClose} releases them.
 * <p>
 * Two concrete implementations cover all quantization schemes:
 * <ul>
 *     <li>{@link FusedCompactionStrategy} — sources carry a {@link FusedFeature} with inline codes;
 *         the strategy is parameterized by a {@link VectorCompressorRetrainer} and the source's
 *         feature (used as a factory for the merged output's feature via
 *         {@link FusedFeature#withCompressor}). No PQ- or ASH-specific code lives in the strategy.</li>
 *     <li>{@link SidecarCompactionStrategy} — sources ship codes as a non-fused
 *         {@code CompressedVectors} sidecar; the strategy is parameterized by a retrainer plus the
 *         source's {@code CompressedVectors} (used as a format handle).</li>
 * </ul>
 * Adding a new quantization type (e.g. ASH) requires no strategy classes; the new {@code FusedASH}
 * and {@code ASHVectors} just return appropriately-parameterized instances of these two strategies.
 */
public abstract class QuantizationCompactionStrategy {

    /**
     * Singleton strategy for sources that ship no quantization at all (no FUSED_PQ, no sidecar).
     * All hooks are no-ops and {@link #compressor()} returns {@code null}.
     */
    public static final QuantizationCompactionStrategy NONE = new QuantizationCompactionStrategy() {
        @Override
        public void retrain(VectorSimilarityFunction vsf) {
            // no-op
        }

        @Override
        public VectorCompressor<?> compressor() {
            return null;
        }

        @Override
        public String toString() {
            return "QuantizationCompactionStrategy.NONE";
        }
    };

    /**
     * Trains a fresh compressor on a balanced sample of merged source vectors. May be a no-op
     * for strategies that don't carry a compressor (e.g. {@link #NONE}). After this call,
     * {@link #compressor()} returns the retrained compressor.
     */
    public abstract void retrain(VectorSimilarityFunction vsf);

    /** The retrained compressor produced by {@link #retrain}. {@code null} before retrain or for NONE. */
    public abstract VectorCompressor<?> compressor();

    /**
     * Whether this strategy writes codes inline in the graph file (FusedPQ-style). When true, the
     * compactor passes the compressor to {@link CompactWriter} and the strategy expects to drive
     * per-node code emission via the writer's inline-code path.
     */
    public boolean writesCodesInline() {
        return false;
    }

    /**
     * Whether this strategy writes codes to a separate sidecar file (PQVectors-style). When true,
     * the compactor calls {@link #writeSidecar} after the graph file is closed.
     */
    public boolean writesCodesSidecar() {
        return false;
    }

    /**
     * Hook invoked once after {@link CompactWriter#writeHeader()} but before {@code compactLevels}.
     * Inline strategies can use this to pre-encode every live node's code into a transient cache
     * that the writer will copy from during inline writes. No-op by default.
     */
    public void onAfterHeader(CompactWriter writer) throws IOException {
        // no-op
    }

    /**
     * Hook invoked once after {@code compactLevels} returns but before
     * {@link CompactWriter#writeFooter()}. Inline strategies that need to emit a per-graph tail
     * record (e.g. the entry-node PQ code for FusedPQ when there is no hierarchy) do so here.
     * No-op by default.
     */
    public void onAfterLevels(CompactWriter writer, int[] entryNodeSource, List<Integer> maxDegrees) throws IOException {
        // no-op
    }

    /**
     * Hook invoked once after the graph file is closed (in {@code finally}). Strategies can
     * release transient resources (e.g. unmap a pre-encode cache and truncate the output file
     * back to its expected size). No-op by default.
     */
    public void onAfterClose(Path graphPath) {
        // no-op
    }

    /**
     * Writes the merged compressed-vectors sidecar file. Called by the compactor's
     * {@code compact(graphPath, compressedPath)} entry point after the graph is fully written.
     * Throws {@link UnsupportedOperationException} by default; sidecar strategies override.
     */
    public void writeSidecar(Path compressedPath) throws IOException {
        throw new UnsupportedOperationException(this + " does not write a sidecar");
    }

    /**
     * Returns the {@link FusedFeature} the compactor should put in the merged output graph for
     * an inline strategy. {@code null} for non-inline strategies (NONE and any sidecar strategy).
     * Called after {@link #retrain} so the strategy can build the output feature from the
     * retrained compressor.
     */
    public FusedFeature outputFusedFeature(int maxDegree) {
        return null;
    }

    /**
     * For compaction use. Returns the precomputed code cache built by {@link #onAfterHeader},
     * indexed by new ordinal so refinement can memcpy neighbor codes instead of re-encoding them.
     * Returns {@code null} when no cache is held (non-fused strategy, NONE, or graph too large for
     * a single mapping). The returned buffer is shared; callers must {@code .duplicate()} per
     * thread before using.
     */
    public MappedByteBuffer getCodeCache() {
        return null;
    }

    /** For compaction use. Bytes per code in {@link #getCodeCache()}, or {@code 0} when no cache. */
    public int getCacheCodeSize() {
        return 0;
    }

    /**
     * Convenience: returns {@link #compressor()} cast to {@link ProductQuantization}, or
     * {@code null} if no compressor is held. Kept for backward compat with code paths that still
     * thread a typed {@code ProductQuantization} through {@link CompactWriter}.
     */
    protected ProductQuantization compressorAsPQ() {
        VectorCompressor<?> c = compressor();
        return (c instanceof ProductQuantization) ? (ProductQuantization) c : null;
    }
}
