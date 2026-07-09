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
import sun.misc.Unsafe;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
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
    private static final Unsafe UNSAFE = getUnsafe();

    private static Unsafe getUnsafe() {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            return (Unsafe) f.get(null);
        } catch (Exception e) {
            log.warn("FusedCompactionStrategy can't acquire needed Unsafe access; code-cache will not be explicitly unmapped");
            return null;
        }
    }

    // Non-final: nulled by releaseSources() after compactGraphImpl so the source graphs reachable
    // through ctx.sources can be GC'd before refinement. onAfterClose must not touch ctx.
    private CompactionContext ctx;
    private final FusedFeature sourceFusedFeature;
    private final VectorCompressorRetrainer retrainer;

    private VectorCompressor<ByteSequence<?>> retrainedCompressor;

    // Transient pre-encode cache: lives in a memory-mapped section appended past the projected
    // end of the output graph file. Truncated away in onAfterClose. Off-heap; single-mapping
    // limit (2 GB) caps this at ~10M nodes for typical codeSize.
    private MappedByteBuffer codeCache;
    private int cacheCodeSize;
    private long cacheTruncateAt;

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
    public MappedByteBuffer getCodeCache() {
        return codeCache;
    }

    @Override
    public int getCacheCodeSize() {
        return cacheCodeSize;
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
            if (codeCache != null) {
                writer.enablePqCodeCache(codeCache, cacheCodeSize);
            }
        } catch (IOException e) {
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
            if (codeCache != null && UNSAFE != null) {
                try {
                    UNSAFE.invokeCleaner(codeCache);
                } catch (IllegalArgumentException ignored) {
                    // duplicated/indirect buffer; not cleanable
                }
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

    /** Pre-encode every live node's code into a memory-mapped section past the projected output end. */
    private void precomputeCodes(CompactWriter writer) throws IOException {
        cacheCodeSize = retrainedCompressor.compressedVectorSize();
        long tempSize = (long) (ctx.maxOrdinal + 1) * cacheCodeSize;
        if (tempSize <= 0 || tempSize > Integer.MAX_VALUE) {
            log.info("Code pre-encode skipped: required cache size {} bytes exceeds single-mapping limit", tempSize);
            return;
        }

        long tempOffset = writer.projectedOutputSize();
        cacheTruncateAt = tempOffset;
        long totalSize = tempOffset + tempSize;

        try (FileChannel fc = FileChannel.open(writer.getOutputPath(),
                StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            ByteBuffer pad = ByteBuffer.wrap(new byte[]{0});
            fc.write(pad, totalSize - 1);
            codeCache = fc.map(FileChannel.MapMode.READ_WRITE, tempOffset, tempSize);
        }

        final int cs = cacheCodeSize;
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
                            int offset = newOrd * cs;
                            for (int i = 0; i < cs; i++) {
                                codeCache.put(offset + i, code.get(i));
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
            log.info("Code pre-encode: {} nodes encoded into {} MB in-output cache (offset {})",
                    total, tempSize / (1024 * 1024), tempOffset);
        } catch (InterruptedException | ExecutionException e) {
            throw new IOException("Code pre-encode failed", e);
        }
    }
}
