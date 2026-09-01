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

import io.github.jbellis.jvector.disk.BufferedRandomAccessWriter;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;

/**
 * Generic compaction strategy for any non-fused {@link CompressedVectors} sidecar. Parameterized
 * by:
 * <ul>
 *     <li>a {@link VectorCompressorRetrainer} that produces the merged compressor on retrain (the
 *         only scheme-specific knowledge this strategy carries),</li>
 *     <li>a {@code formatHandle} {@link CompressedVectors} from the sources, used only to invoke
 *         {@link CompressedVectors#writeSidecarHeader} and {@link CompressedVectors#sidecarVectorsPerChunk}
 *         — the format hooks that decide the on-disk layout for the merged sidecar.</li>
 * </ul>
 * No PQ-specific (or ASH-specific) code lives here. Adding a new quantization type that ships
 * a sidecar means implementing those two hooks on its {@code CompressedVectors} class plus a
 * retrainer; this strategy and the compactor stay untouched.
 */
public final class SidecarCompactionStrategy extends QuantizationCompactionStrategy {
    private static final Logger log = LoggerFactory.getLogger(SidecarCompactionStrategy.class);
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    private CompactionContext ctx;
    private final CompressedVectors formatHandle;
    private final VectorCompressorRetrainer retrainer;
    private VectorCompressor<?> retrainedCompressor;

    // Pre-encoded merged codes, mmapped past the projected end of the output graph file (same
    // mechanics as FusedCompactionStrategy). Unlike the fused case, the cache must outlive the
    // graph file close: writeSidecar copies from it, so truncation is deferred until then.
    private PreEncodedCodeCache codeCache;
    private int cacheCodeSize;
    private long cacheTruncateAt;
    private Path graphPath;

    public SidecarCompactionStrategy(CompactionContext ctx,
                                     CompressedVectors formatHandle,
                                     VectorCompressorRetrainer retrainer) {
        this.ctx = ctx;
        this.formatHandle = formatHandle;
        this.retrainer = retrainer;
    }

    @Override
    public void onRemappersUpdated(CompactionContext refreshed) {
        this.ctx = refreshed;
    }

    @Override
    public void retrain(VectorSimilarityFunction vsf) {
        log.info("Retraining sidecar compressor ({}) on merged sources",
                formatHandle.getClass().getSimpleName());
        this.retrainedCompressor = retrainer.retrain(vsf);
    }

    @Override
    public VectorCompressor<?> compressor() {
        return retrainedCompressor;
    }

    @Override
    public boolean writesCodesSidecar() {
        return true;
    }

    /**
     * Pre-encodes every live node's merged code into a transient mmapped section of the output
     * graph file (fused-strategy mechanics). The cache serves three consumers: approximate
     * cross-source search scoring, offer/diversity scoring, and {@link #writeSidecar} — which
     * becomes a cache copy instead of a second full re-read+re-encode pass over the sources.
     */
    @Override
    public void onAfterHeader(CompactWriter writer) throws IOException {
        if (retrainedCompressor == null) {
            throw new IllegalStateException("retrain() must be called before onAfterHeader()");
        }
        try {
            precomputeCodes(writer);
        } catch (IOException e) {
            log.warn("Sidecar code pre-encode failed, falling back to re-encode at writeSidecar: {}", e.getMessage());
            closeCache();
        }
    }

    @Override
    public PreEncodedCodeCache getCodeCache() {
        return codeCache;
    }

    @Override
    public int getCacheCodeSize() {
        return codeCache == null ? 0 : cacheCodeSize;
    }

    @Override
    public void onAfterClose(Path graphPath) {
        // Deliberately no truncation here: writeSidecar still needs the cache region. Remember
        // the path so cleanup can truncate after the sidecar is written.
        this.graphPath = graphPath;
    }

    @Override
    public void writeSidecar(Path compressedPath) throws IOException {
        if (retrainedCompressor == null) {
            throw new IllegalStateException("retrain() must be called before writeSidecar()");
        }
        final int vectorsPerChunk = formatHandle.sidecarVectorsPerChunk();
        final int codeSize = retrainedCompressor.compressedVectorSize();
        final int count = ctx.maxOrdinal + 1;
        final int chunkCount = (count + vectorsPerChunk - 1) / vectorsPerChunk;

        log.info("Streaming {} merged ordinals to {} ({} chunks of up to {} entries each{})",
                count, compressedPath, chunkCount, vectorsPerChunk,
                codeCache != null ? ", from pre-encode cache" : "");

        try (var out = new BufferedRandomAccessWriter(compressedPath)) {
            formatHandle.writeSidecarHeader(out, retrainedCompressor, count);

            int parallelism = Math.max(ctx.taskWindowSize, 1);
            for (int batchStart = 0; batchStart < chunkCount; batchStart += parallelism) {
                int batchEnd = Math.min(batchStart + parallelism, chunkCount);
                List<Callable<ByteSequence<?>>> tasks = new ArrayList<>(batchEnd - batchStart);
                for (int c = batchStart; c < batchEnd; c++) {
                    final int chunkStart = c * vectorsPerChunk;
                    final int chunkEnd = Math.min(chunkStart + vectorsPerChunk, count);
                    tasks.add(codeCache != null
                            ? () -> copyChunkFromCache(chunkStart, chunkEnd, codeSize)
                            : () -> encodeChunk(chunkStart, chunkEnd, codeSize, retrainedCompressor));
                }
                for (var f : ctx.executor.invokeAll(tasks)) {
                    vectorTypeSupport.writeByteSequence(out, f.get());
                }
            }
        } catch (InterruptedException | ExecutionException e) {
            throw new IOException("Failed to write compressed sidecar to " + compressedPath, e);
        } finally {
            releaseCacheAndTruncate();
        }
        log.info("Wrote compacted compressed sidecar to {}", compressedPath);
    }

    private ByteSequence<?> copyChunkFromCache(int chunkStart, int chunkEnd, int codeSize) {
        int chunkBytes = (chunkEnd - chunkStart) * codeSize;
        ByteSequence<?> chunk = vectorTypeSupport.createByteSequence(chunkBytes);
        chunk.zero();
        byte[] code = new byte[codeSize];
        for (int newOrd = chunkStart; newOrd < chunkEnd; newOrd++) {
            if (resolveSourceForNewOrd(newOrd) == null) continue;  // hole; slot stays zero
            codeCache.get(newOrd, code);
            int slotOffset = (newOrd - chunkStart) * codeSize;
            for (int b = 0; b < codeSize; b++) {
                chunk.set(slotOffset + b, code[b]);
            }
        }
        return chunk;
    }

    @Override
    public void releaseTransientState() {
        releaseCacheAndTruncate();
    }

    private void closeCache() {
        if (codeCache != null) {
            codeCache.close();
            codeCache = null;
        }
    }

    private void releaseCacheAndTruncate() {
        closeCache();
        if (cacheTruncateAt > 0 && graphPath != null) {
            try (java.nio.channels.FileChannel fc = java.nio.channels.FileChannel.open(
                    graphPath, java.nio.file.StandardOpenOption.WRITE)) {
                if (fc.size() > cacheTruncateAt) {
                    fc.truncate(cacheTruncateAt);
                }
            } catch (IOException e) {
                throw new RuntimeException("Failed to truncate code-cache section from " + graphPath, e);
            }
            cacheTruncateAt = 0;
        }
    }

    /** Mirrors {@code FusedCompactionStrategy.precomputeCodes}; kept separate so the fused path
     *  stays untouched. */
    @SuppressWarnings("unchecked")
    private void precomputeCodes(CompactWriter writer) throws IOException {
        cacheCodeSize = retrainedCompressor.compressedVectorSize();
        int codeCount = ctx.maxOrdinal + 1;
        long tempSize = PreEncodedCodeCache.sectionBytes(codeCount, cacheCodeSize);
        if (codeCount <= 0 || tempSize <= 0) {
            log.info("Sidecar pre-encode skipped: degenerate cache size {} bytes for {} codes", tempSize, codeCount);
            return;
        }

        long tempOffset = writer.projectedOutputSize();
        cacheTruncateAt = tempOffset;
        long totalSize = tempOffset + tempSize;

        try (java.nio.channels.FileChannel fc = java.nio.channels.FileChannel.open(writer.getOutputPath(),
                java.nio.file.StandardOpenOption.READ, java.nio.file.StandardOpenOption.WRITE)) {
            java.nio.ByteBuffer pad = java.nio.ByteBuffer.wrap(new byte[]{0});
            fc.write(pad, totalSize - 1);
            codeCache = PreEncodedCodeCache.map(fc, tempOffset, codeCount, cacheCodeSize);
        }

        final int cs = cacheCodeSize;
        final VectorCompressor<ByteSequence<?>> compressor = (VectorCompressor<ByteSequence<?>>) retrainedCompressor;
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
                            codeCache.put(newOrd, code);
                            count++;
                        }
                    }
                    return count;
                });
            }
        }
        try {
            long total = 0;
            for (var f : ctx.executor.invokeAll(tasks)) {
                total += f.get();
            }
            log.info("Sidecar code pre-encode: {} nodes encoded into {} MB in-output cache ({} mapping(s), offset {})",
                    total, tempSize / (1024 * 1024), codeCache.chunkCount(), tempOffset);
        } catch (InterruptedException | ExecutionException e) {
            throw new IOException("Sidecar code pre-encode failed", e);
        }
    }

    @SuppressWarnings("unchecked")
    private ByteSequence<?> encodeChunk(int chunkStart, int chunkEnd, int codeSize, VectorCompressor<?> compressor) throws IOException {
        int chunkBytes = (chunkEnd - chunkStart) * codeSize;
        ByteSequence<?> chunk = vectorTypeSupport.createByteSequence(chunkBytes);
        chunk.zero();

        // Cast once; valid for all VectorCompressor implementations that produce ByteSequence codes
        // (PQ, future ASH, etc.). VectorCompressor's encode/encodeTo contract guarantees T is the
        // encoded type and for our supported quantization schemes T = ByteSequence<?>.
        VectorCompressor<ByteSequence<?>> byteCompressor = (VectorCompressor<ByteSequence<?>>) compressor;

        OnDiskGraphIndex.View[] views = new OnDiskGraphIndex.View[ctx.sources.size()];
        try {
            VectorFloat<?> vec = vectorTypeSupport.createFloatVector(ctx.dimension);
            ByteSequence<?> code = vectorTypeSupport.createByteSequence(codeSize);
            for (int newOrd = chunkStart; newOrd < chunkEnd; newOrd++) {
                int[] resolved = resolveSourceForNewOrd(newOrd);
                if (resolved == null) continue;  // hole; slot stays zero
                int srcIdx = resolved[0];
                int oldOrd = resolved[1];
                if (views[srcIdx] == null) {
                    views[srcIdx] = (OnDiskGraphIndex.View) ctx.sources.get(srcIdx).getView();
                }
                views[srcIdx].getVectorInto(oldOrd, vec, 0);
                code.zero();
                byteCompressor.encodeTo(vec, code);
                int slotOffset = (newOrd - chunkStart) * codeSize;
                for (int b = 0; b < codeSize; b++) {
                    chunk.set(slotOffset + b, code.get(b));
                }
            }
        } finally {
            for (var v : views) {
                if (v != null) {
                    try { v.close(); } catch (Exception ignore) {}
                }
            }
        }
        return chunk;
    }

    private int[] resolveSourceForNewOrd(int newOrd) {
        for (int s = 0; s < ctx.remappers.size(); s++) {
            int oldOrd = ctx.remappers.get(s).newToOld(newOrd);
            if (oldOrd != OrdinalMapper.OMITTED) {
                return new int[]{s, oldOrd};
            }
        }
        return null;
    }
}
