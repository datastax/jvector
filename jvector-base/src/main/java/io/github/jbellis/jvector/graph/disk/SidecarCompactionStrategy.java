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

    private final CompactionContext ctx;
    private final CompressedVectors formatHandle;
    private final VectorCompressorRetrainer retrainer;
    private VectorCompressor<?> retrainedCompressor;

    public SidecarCompactionStrategy(CompactionContext ctx,
                                     CompressedVectors formatHandle,
                                     VectorCompressorRetrainer retrainer) {
        this.ctx = ctx;
        this.formatHandle = formatHandle;
        this.retrainer = retrainer;
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

    @Override
    public void writeSidecar(Path compressedPath) throws IOException {
        if (retrainedCompressor == null) {
            throw new IllegalStateException("retrain() must be called before writeSidecar()");
        }
        final int vectorsPerChunk = formatHandle.sidecarVectorsPerChunk();
        final int codeSize = retrainedCompressor.compressedVectorSize();
        final int count = ctx.maxOrdinal + 1;
        final int chunkCount = (count + vectorsPerChunk - 1) / vectorsPerChunk;

        log.info("Streaming {} merged ordinals to {} ({} chunks of up to {} entries each)",
                count, compressedPath, chunkCount, vectorsPerChunk);

        try (var out = new BufferedRandomAccessWriter(compressedPath)) {
            formatHandle.writeSidecarHeader(out, retrainedCompressor, count);

            int parallelism = Math.max(ctx.taskWindowSize, 1);
            for (int batchStart = 0; batchStart < chunkCount; batchStart += parallelism) {
                int batchEnd = Math.min(batchStart + parallelism, chunkCount);
                List<Callable<ByteSequence<?>>> tasks = new ArrayList<>(batchEnd - batchStart);
                for (int c = batchStart; c < batchEnd; c++) {
                    final int chunkStart = c * vectorsPerChunk;
                    final int chunkEnd = Math.min(chunkStart + vectorsPerChunk, count);
                    tasks.add(() -> encodeChunk(chunkStart, chunkEnd, codeSize, retrainedCompressor));
                }
                for (var f : ctx.executor.invokeAll(tasks)) {
                    vectorTypeSupport.writeByteSequence(out, f.get());
                }
            }
        } catch (InterruptedException | ExecutionException e) {
            throw new IOException("Failed to write compressed sidecar to " + compressedPath, e);
        }
        log.info("Wrote compacted compressed sidecar to {}", compressedPath);
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
