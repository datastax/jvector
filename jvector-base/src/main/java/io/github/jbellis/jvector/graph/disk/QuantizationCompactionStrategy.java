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
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.IOException;
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
    private static final Logger log = LoggerFactory.getLogger(QuantizationCompactionStrategy.class);
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();


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
     * Replaces the strategy's context snapshot after the compactor re-assigns output ordinals
     * (similarity ordinals). Strategies that place codes by output ordinal must adopt the
     * refreshed remappers, or codes land at the caller-proposed ordinals while the graph is
     * written at the reassigned ones. No-op for strategies that hold no context.
     */
    // Optional second compressor whose codes are written to a second scratch cache, right after the
    // first, by the same pre-encode pass (the compactor's wide code). Truncated away with the first.
    protected VectorCompressor<?> secondaryCompressor;
    protected PreEncodedCodeCache secondaryCache;

    /** For compaction use: a second code per node, encoded in the same pre-encode pass. */
    public void setSecondaryCompressor(VectorCompressor<?> compressor) {
        this.secondaryCompressor = compressor;
    }

    /** For compaction use. The secondary code cache, or null. */
    public PreEncodedCodeCache getSecondaryCache() {
        return secondaryCache;
    }

    protected void closeSecondaryCache() {
        if (secondaryCache != null) {
            secondaryCache.close();
            secondaryCache = null;
        }
    }

    public void onRemappersUpdated(CompactionContext refreshed) {
        // no-op by default
    }

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
     * Releases any transient resources still held after a failed compaction (pre-encode cache
     * mappings, un-truncated scratch regions). Idempotent; a successful run releases them in its
     * normal flow, so this only acts when a failure interrupted that flow. No-op by default.
     */
    public void releaseTransientState() {
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
     * indexed by new ordinal so record writes can memcpy neighbor codes instead of re-encoding them.
     * Returns {@code null} when no cache is held (non-fused strategy, NONE, or a pre-encode
     * failure). The returned cache is shared across threads and safe for concurrent use.
     */
    public PreEncodedCodeCache getCodeCache() {
        return null;
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

    /**
     * Encodes every live node with {@code compressor} into a code cache keyed by new ordinal,
     * memory-mapped past the projected end of the output file (the caller truncates the file back
     * to {@link CompactWriter#projectedOutputSize()} when done). Returns null for a degenerate
     * (empty) merge.
     */
    @SuppressWarnings("unchecked")
    protected PreEncodedCodeCache precomputeCodeCache(CompactionContext ctx, CompactWriter writer,
                                                      VectorCompressor<?> compressor, String label) throws IOException {
        final int codeSize = compressor.compressedVectorSize();
        int codeCount = ctx.maxOrdinal + 1;
        long tempSize = PreEncodedCodeCache.sectionBytes(codeCount, codeSize);
        if (codeCount <= 0 || tempSize <= 0) {
            log.info("{} skipped: degenerate cache size {} bytes for {} codes", label, tempSize, codeCount);
            return null;
        }
        final int secondarySize = secondaryCompressor == null ? 0 : secondaryCompressor.compressedVectorSize();
        final long secondaryBytes = secondarySize == 0 ? 0 : PreEncodedCodeCache.sectionBytes(codeCount, secondarySize);
        // Reserved, not simply placed at the projected end: when an inline fused strategy and a
        // sidecar scratch are both active they would otherwise claim the same offset and the
        // larger region would overwrite the smaller one's codes.
        long tempOffset = writer.reserveScratch(tempSize + secondaryBytes);
        final long secondaryOffset = tempOffset + tempSize;
        final PreEncodedCodeCache cache;
        try (FileChannel fc = FileChannel.open(writer.getOutputPath(), StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            fc.write(ByteBuffer.wrap(new byte[]{0}), secondaryOffset + secondaryBytes - 1);
            cache = PreEncodedCodeCache.map(fc, tempOffset, codeCount, codeSize);
            if (secondarySize > 0) {
                secondaryCache = PreEncodedCodeCache.map(fc, secondaryOffset, codeCount, secondarySize);
            }
        }
        final VectorCompressor<ByteSequence<?>> enc = (VectorCompressor<ByteSequence<?>>) compressor;
        final VectorCompressor<ByteSequence<?>> enc2 = (VectorCompressor<ByteSequence<?>>) secondaryCompressor;
        final PreEncodedCodeCache cache2 = secondaryCache;
        List<Callable<Long>> tasks = new ArrayList<>();
        int targetTasks = Math.max(ctx.taskWindowSize * 4, 16);
        // Wide entries (a full-precision merge stores the vectors themselves) are written in new-ordinal
        // order: each source's live nodes occupy one contiguous range of new ordinals, so walking that
        // range and reading the source at random turns the scratch write sequential. Writing in source
        // order instead scatters 6 KB records over a mapping of tens of GB, and the page faults of
        // those random writes cost more than the whole rest of the pass.
        final boolean sequentialWrite = codeSize + secondarySize >= 512;
        for (int s = 0; s < ctx.sources.size() && sequentialWrite; s++) {
            final int sIdx = s;
            final var source = ctx.sources.get(s);
            final var alive = ctx.liveNodes.get(s);
            final var mapper = ctx.remappers.get(s);
            int newLo = Integer.MAX_VALUE, newHi = -1;
            for (int old = alive.nextSetBit(0); old != io.github.jbellis.jvector.util.DocIdSetIterator.NO_MORE_DOCS && old < alive.length(); old = alive.nextSetBit(old + 1)) {
                int n = mapper.oldToNew(old);
                if (n < newLo) newLo = n;
                if (n > newHi) newHi = n;
            }
            if (newHi < 0) continue;
            final int lo0 = newLo, hi0 = newHi + 1;
            int chunkSize = Math.max(256, (hi0 - lo0 + targetTasks - 1) / targetTasks);
            for (int chunkStart = lo0; chunkStart < hi0; chunkStart += chunkSize) {
                final int cStart = chunkStart;
                final int cEnd = Math.min(chunkStart + chunkSize, hi0);
                tasks.add(() -> {
                    ByteSequence<?> code = vectorTypeSupport.createByteSequence(codeSize);
                    ByteSequence<?> code2 = enc2 == null ? null : vectorTypeSupport.createByteSequence(secondarySize);
                    VectorFloat<?> vec = vectorTypeSupport.createFloatVector(ctx.dimension);
                    long count = 0;
                    try (var view = source.getView()) {
                        for (int newOrd = cStart; newOrd < cEnd; newOrd++) {
                            int oldOrd = mapper.newToOld(newOrd);
                            if (oldOrd < 0 || !alive.get(oldOrd)) continue;
                            view.getVectorInto(oldOrd, vec, 0);
                            code.zero();
                            enc.encodeTo(vec, code);
                            cache.put(newOrd, code);
                            if (enc2 != null) {
                                code2.zero();
                                enc2.encodeTo(vec, code2);
                                cache2.put(newOrd, code2);
                            }
                            count++;
                        }
                    }
                    return count;
                });
            }
        }
        for (int s = 0; s < ctx.sources.size() && !sequentialWrite; s++) {
            final int sIdx = s;
            final var source = ctx.sources.get(s);
            final var alive = ctx.liveNodes.get(s);
            final int upper = alive.length();
            int chunkSize = Math.max(256, (upper + targetTasks - 1) / targetTasks);
            for (int chunkStart = 0; chunkStart < upper; chunkStart += chunkSize) {
                final int cStart = chunkStart;
                final int cEnd = Math.min(chunkStart + chunkSize, upper);
                tasks.add(() -> {
                    // stream the chunk's records into the page cache before the encode loop
                    source.prefetchL0Records(cStart, cEnd - 1);
                    ByteSequence<?> code = vectorTypeSupport.createByteSequence(codeSize);
                    ByteSequence<?> code2 = enc2 == null ? null : vectorTypeSupport.createByteSequence(secondarySize);
                    VectorFloat<?> vec = vectorTypeSupport.createFloatVector(ctx.dimension);
                    long count = 0;
                    try (var view = source.getView()) {
                        for (int oldOrd = cStart; oldOrd < cEnd; oldOrd++) {
                            if (!alive.get(oldOrd)) continue;
                            view.getVectorInto(oldOrd, vec, 0);
                            int newOrd = ctx.remappers.get(sIdx).oldToNew(oldOrd);
                            code.zero();
                            enc.encodeTo(vec, code);
                            cache.put(newOrd, code);
                            if (enc2 != null) {
                                code2.zero();
                                enc2.encodeTo(vec, code2);
                                cache2.put(newOrd, code2);
                            }
                            count++;
                        }
                    }
                    return count;
                });
            }
        }
        try {
            long total = 0;
            for (Future<Long> f : ctx.executor.invokeAll(tasks)) {
                total += f.get();
            }
            log.info("{}: {} nodes encoded into {} MB in-output cache across {} mapping(s) (offset {}){}",
                     label, total, tempSize / (1024 * 1024), cache.chunkCount(), tempOffset,
                     secondarySize == 0 ? "" : String.format(", plus %d MB of %d-byte secondary codes", secondaryBytes >> 20, secondarySize));
        } catch (InterruptedException | ExecutionException e) {
            cache.close();
            closeSecondaryCache();
            throw new IOException(label + " failed", e);
        }
        return cache;
    }
}
