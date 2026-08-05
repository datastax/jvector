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
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

/**
 * Generic compaction strategy for any {@link FusedFeature} (PQ today, ASH or other schemes
 * later). Parameterized by:
 * <ul>
 *     <li>a {@link VectorCompressorRetrainer} producing the retrained compressor — the only
 *         scheme-specific knowledge this strategy needs,</li>
 *     <li>the source's {@link FusedFeature}, used as a factory ({@code withCompressor(...)})
 *         to produce the merged output's fused feature.</li>
 * </ul>
 * The pre-encode mmap pass, entry-node-code tail write, and file truncation are all expressed
 * against {@code VectorCompressor.encodeTo} and {@code FusedFeature.codeSize()} — no PQ or ASH
 * specifics live here.
 */
public final class FusedCompactionStrategy extends QuantizationCompactionStrategy {
    private static final Logger log = LoggerFactory.getLogger(FusedCompactionStrategy.class);
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    // Non-final: nulled by releaseSources() after compactGraphImpl so the source graphs reachable
    // through ctx.sources can be GC'd before refinement. onAfterClose must not touch ctx.
    private CompactionContext ctx;
    private final FusedFeature sourceFusedFeature;
    private final VectorCompressorRetrainer retrainer;

    private VectorCompressor<ByteSequence<?>> retrainedCompressor;

    // Transient pre-encode cache: lives in a memory-mapped section appended past the projected end
    // of the output graph file, chunked into <= 1 GiB mappings so it has no single-mapping size cap.
    // Truncated away in onAfterClose. May be PqCodeCache.NONE when the cache is configured off.
    private PqCodeCache codeCache;
    private long cacheTruncateAt;

    // Compose-in / leave-off + chunk-size config; default composes the cache in at 1 GiB chunks.
    private PqCodeCacheConfig pqCodeCacheConfig = PqCodeCacheConfig.DEFAULT;

    /** For compaction use. Sets the {@link PqCodeCache} configuration for this run. */
    public void setPqCodeCacheConfig(PqCodeCacheConfig pqCodeCacheConfig) {
        this.pqCodeCacheConfig = pqCodeCacheConfig;
    }

    public FusedCompactionStrategy(CompactionContext ctx,
                                   FusedFeature sourceFusedFeature,
                                   VectorCompressorRetrainer retrainer) {
        this.ctx = ctx;
        this.sourceFusedFeature = sourceFusedFeature;
        this.retrainer = retrainer;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void retrain(VectorSimilarityFunction vsf) {
        log.info("Retraining fused-quantization compressor on merged sources");
        this.retrainedCompressor = (VectorCompressor<ByteSequence<?>>) (VectorCompressor<?>) retrainer.retrain(vsf);
    }

    @Override
    public VectorCompressor<?> compressor() {
        return retrainedCompressor;
    }

    @Override
    public PqCodeCache getCodeCache() {
        return codeCache;
    }

    @Override
    public boolean writesCodesInline() {
        return true;
    }

    @Override
    public void releaseSources() {
        // ctx is only needed during onAfterHeader/onAfterLevels (pre-encode + entry-node code),
        // which run inside compactGraphImpl. onAfterClose uses only cacheTruncateAt/codeCache.
        // Safe to drop here so ctx.sources' in-heap layers/features are reclaimable before refine.
        ctx = null;
    }

    /**
     * Returns the {@link FusedFeature} the compactor should put in the merged output graph.
     * Constructed from the source's fused feature via {@link FusedFeature#withCompressor},
     * parameterized by the retrained compressor and the merged graph's max degree.
     */
    public FusedFeature outputFusedFeature(int maxDegree) {
        if (retrainedCompressor == null) {
            throw new IllegalStateException("retrain() must be called before outputFusedFeature()");
        }
        return sourceFusedFeature.withCompressor(retrainedCompressor, maxDegree);
    }

    @Override
    public void onAfterHeader(CompactWriter writer) throws IOException {
        if (retrainedCompressor == null) {
            throw new IllegalStateException("retrain() must be called before onAfterHeader()");
        }
        try {
            precomputeCodes(writer);
            if (codeCache != null && codeCache.isActive()) {
                writer.enablePqCodeCache(codeCache);
            }
        } catch (IOException e) {
            // The fallback exists for environmental failures (mapping limits, transient IO) —
            // per-write encoding produces the same output, just slower. Interruption is not
            // environmental: it is the caller cancelling the compaction, and absorbing it here
            // would make compact() run to completion after the one interrupt a host delivers.
            for (Throwable cause = e; cause != null; cause = cause.getCause()) {
                if (cause instanceof InterruptedException) {
                    throw e;
                }
            }
            log.warn("Code pre-encode failed, falling back to per-write encoding: {}", e.getMessage());
        }
    }

    @Override
    public void onAfterLevels(CompactWriter writer, int[] entryNodeSource, List<Integer> maxDegrees) throws IOException {
        // When fused features are present and there is no hierarchy (only L0), the reader expects
        // to find the entry node's own code written after the L0 block, just as
        // AbstractGraphIndexWriter.writeSparseLevels does in its getMaxLevel == 0 branch. Without
        // it, loadInMemoryFeatures reads garbage and hierarchyCachedFeatures is missing the
        // entry node, causing "Node X is not in the hierarchy" on first search.
        if (maxDegrees.size() != 1) {
            return;
        }
        try (var entryView = ctx.sources.get(entryNodeSource[0]).getView()) {
            var entryVec = vectorTypeSupport.createFloatVector(ctx.dimension);
            entryView.getVectorInto(entryNodeSource[1], entryVec, 0);
            var entryCode = vectorTypeSupport.createByteSequence(retrainedCompressor.compressedVectorSize());
            entryCode.zero();
            retrainedCompressor.encodeTo(entryVec, entryCode);
            writer.setEntryNodePqCode(entryCode);
        }
    }

    @Override
    public void onAfterClose(Path graphPath) {
        if (cacheTruncateAt > 0) {
            if (codeCache != null) {
                codeCache.unmap();
            }
            codeCache = null;
            try (FileChannel fc = FileChannel.open(graphPath, StandardOpenOption.WRITE)) {
                if (fc.size() > cacheTruncateAt) {
                    fc.truncate(cacheTruncateAt);
                }
            } catch (IOException e) {
                throw new RuntimeException("Failed to truncate code-cache section from output file " + graphPath, e);
            }
            cacheTruncateAt = 0;
        }
    }

    /// Pre-encode every live node's code into a memory-mapped, chunked cache appended past the
    /// projected output end. Composed in or left off per [PqCodeCacheConfig]; when off, the cache is
    /// [PqCodeCache#NONE] and consumers encode per neighbor write. The chunked mapping has no
    /// single-mapping (2 GiB) size cap, so — unlike the previous single-buffer cache — it is never
    /// silently skipped above ~21M nodes.
    private void precomputeCodes(CompactWriter writer) throws IOException {
        if (!pqCodeCacheConfig.enabled()) {
            log.info("Code pre-encode disabled by configuration; neighbor codes will be encoded per write");
            codeCache = PqCodeCache.NONE;
            return;
        }

        final int cs = retrainedCompressor.compressedVectorSize();
        long numCodes = (long) (ctx.maxOrdinal + 1);
        long tempSize = numCodes * cs;
        if (tempSize <= 0) {
            log.info("Code pre-encode skipped: non-positive cache size {} bytes", tempSize);
            codeCache = PqCodeCache.NONE;
            return;
        }

        long tempOffset = writer.projectedOutputSize();
        cacheTruncateAt = tempOffset;

        try (FileChannel fc = FileChannel.open(writer.getOutputPath(),
                StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            codeCache = PqCodeCache.map(fc, tempOffset, cs, numCodes, pqCodeCacheConfig.maxChunkBytes());
        }

        final PqCodeCache cache = codeCache;
        final VectorCompressor<ByteSequence<?>> compressor = retrainedCompressor;
        List<Callable<Long>> tasks = new ArrayList<>();
        int targetTasks = Math.max(ctx.taskWindowSize * 4, 16);
        for (int s = 0; s < ctx.sources.size(); s++) {
            final int sIdx = s;
            final var source = ctx.sources.get(s);
            final var alive = ctx.liveNodes.get(s);
            final int upper = alive.length();
            int chunkSize = Math.max(256, (upper + targetTasks - 1) / targetTasks);
            for (int chunkStart = 0; chunkStart < upper; chunkStart += chunkSize) {
                final int cStart = chunkStart;
                final int cEnd = Math.min(chunkStart + chunkSize, upper);
                tasks.add(() -> {
                    // Stream this chunk's records into the page cache before the encode loop;
                    // the mapping's MADV_RANDOM otherwise faults them one page at a time.
                    source.prefetchL0Records(cStart, cEnd - 1);
                    ByteSequence<?> code = vectorTypeSupport.createByteSequence(cs);
                    VectorFloat<?> vec = vectorTypeSupport.createFloatVector(ctx.dimension);
                    long count = 0;
                    try (var view = source.getView()) {
                        for (int oldOrd = cStart; oldOrd < cEnd; oldOrd++) {
                            if (!alive.get(oldOrd)) continue;
                            view.getVectorInto(oldOrd, vec, 0);
                            code.zero();
                            compressor.encodeTo(vec, code);
                            int newOrd = ctx.remappers.get(sIdx).oldToNew(oldOrd);
                            cache.putCode(newOrd, code);
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
            log.info("Code pre-encode: {} nodes encoded into {} MB in-output cache across {} chunk(s) (offset {})",
                    total, tempSize / (1024 * 1024), cache.chunkCount(), tempOffset);
        } catch (InterruptedException | ExecutionException e) {
            throw new IOException("Code pre-encode failed", e);
        }
    }
}
