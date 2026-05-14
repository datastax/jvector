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
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.util.FixedBitSet;
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
import java.util.concurrent.ForkJoinPool;

/**
 * Strategy for compaction when sources ship a non-fused compressed sidecar (e.g.
 * {@link io.github.jbellis.jvector.quantization.PQVectors}). The graph file itself carries no
 * inline codes; this strategy retrains the compressor on a balanced sample of merged source
 * vectors and streams a merged sidecar to the configured compressed path.
 * <p>
 * The sidecar is written as a chunked PQVectors. Each chunk holds {@value #VECTORS_PER_CHUNK}
 * codes; chunks are encoded by parallel workers in batches of {@code parallelism} and written
 * sequentially as each batch completes, so heap residency is bounded by
 * {@code parallelism * chunkBytes} rather than the full {@code (maxOrdinal+1) * codeSize} that
 * an in-memory {@code MutablePQVectors} would hold.
 * <p>
 * Today only {@link ProductQuantization}-backed sidecars are supported; future quantization
 * types (e.g. ASH) will ship their own strategy implementation rather than extending this one.
 */
public final class SidecarPQStrategy extends QuantizationCompactionStrategy {
    private static final Logger log = LoggerFactory.getLogger(SidecarPQStrategy.class);
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    // Match MutablePQVectors' chunk size so the on-disk layout is identical to what
    // PQVectors.load(...) reconstructs into an ImmutablePQVectors.
    private static final int VECTORS_PER_CHUNK = 1024;

    private final List<OnDiskGraphIndex> sources;
    private final List<CompressedVectors> sourceCompressed;
    private final List<FixedBitSet> liveNodes;
    private final List<OrdinalMapper> remappers;
    private final int dimension;
    private final int maxOrdinal;
    private final ForkJoinPool executor;
    private final int taskWindowSize;

    private ProductQuantization retrainedPQ;

    public SidecarPQStrategy(
            List<OnDiskGraphIndex> sources,
            List<CompressedVectors> sourceCompressed,
            List<FixedBitSet> liveNodes,
            List<OrdinalMapper> remappers,
            int dimension,
            int maxOrdinal,
            ForkJoinPool executor,
            int taskWindowSize) {
        this.sources = sources;
        this.sourceCompressed = sourceCompressed;
        this.liveNodes = liveNodes;
        this.remappers = remappers;
        this.dimension = dimension;
        this.maxOrdinal = maxOrdinal;
        this.executor = executor;
        this.taskWindowSize = taskWindowSize;
    }

    @Override
    public void retrain(VectorSimilarityFunction vsf) {
        VectorCompressor<?> firstCompressor = sourceCompressed.get(0).getCompressor();
        if (!(firstCompressor instanceof ProductQuantization)) {
            throw new UnsupportedOperationException(
                    "Compressed-sidecar compaction currently supports only ProductQuantization; got "
                            + firstCompressor.getClass().getSimpleName());
        }
        ProductQuantization basePQ = (ProductQuantization) firstCompressor;
        log.info("Retraining PQ for compacted compressed sidecar (subspaces={}, clusters={})",
                basePQ.getSubspaceCount(), basePQ.getClusterCount());
        PQRetrainer retrainer = new PQRetrainer(sources, liveNodes, dimension);
        this.retrainedPQ = retrainer.retrain(vsf, basePQ);
    }

    @Override
    public VectorCompressor<?> compressor() {
        return retrainedPQ;
    }

    @Override
    public boolean writesCodesSidecar() {
        return true;
    }

    @Override
    public void writeSidecar(Path compressedPath) throws IOException {
        if (retrainedPQ == null) {
            throw new IllegalStateException("retrain() must be called before writeSidecar()");
        }
        final int codeSize = retrainedPQ.compressedVectorSize();
        final int subspaceCount = retrainedPQ.getSubspaceCount();
        final int count = maxOrdinal + 1;
        final int chunkCount = (count + VECTORS_PER_CHUNK - 1) / VECTORS_PER_CHUNK;

        log.info("Streaming {} merged ordinals to {} ({} chunks of up to {} entries each)",
                count, compressedPath, chunkCount, VECTORS_PER_CHUNK);

        try (var out = new BufferedRandomAccessWriter(compressedPath)) {
            // PQVectors header: codebook, count, subspaceCount.
            retrainedPQ.write(out, OnDiskGraphIndex.CURRENT_VERSION);
            out.writeInt(count);
            out.writeInt(subspaceCount);

            // Encode chunks in parallel batches, write batches in order. Heap residency is
            // bounded by `parallelism` chunks in flight at a time.
            int parallelism = Math.max(taskWindowSize, 1);
            for (int batchStart = 0; batchStart < chunkCount; batchStart += parallelism) {
                int batchEnd = Math.min(batchStart + parallelism, chunkCount);
                List<Callable<ByteSequence<?>>> tasks = new ArrayList<>(batchEnd - batchStart);
                for (int c = batchStart; c < batchEnd; c++) {
                    final int chunkStart = c * VECTORS_PER_CHUNK;
                    final int chunkEnd = Math.min(chunkStart + VECTORS_PER_CHUNK, count);
                    tasks.add(() -> encodeChunk(chunkStart, chunkEnd, codeSize, retrainedPQ));
                }
                for (var f : executor.invokeAll(tasks)) {
                    vectorTypeSupport.writeByteSequence(out, f.get());
                }
            }
        } catch (InterruptedException | ExecutionException e) {
            throw new IOException("Failed to write compressed sidecar to " + compressedPath, e);
        }
        log.info("Wrote compacted compressed sidecar to {}", compressedPath);
    }

    /**
     * Encodes a single output chunk (newOrd range {@code [chunkStart, chunkEnd)}) by reading
     * each ordinal's raw vector from its originating source view and PQ-encoding it. Holes
     * (newOrds not covered by any remapper) leave the corresponding slot zero. Views are
     * opened lazily per source touched within the chunk and closed before returning.
     */
    private ByteSequence<?> encodeChunk(int chunkStart, int chunkEnd, int codeSize, ProductQuantization pq) throws IOException {
        int chunkBytes = (chunkEnd - chunkStart) * codeSize;
        ByteSequence<?> chunk = vectorTypeSupport.createByteSequence(chunkBytes);
        chunk.zero();

        OnDiskGraphIndex.View[] views = new OnDiskGraphIndex.View[sources.size()];
        try {
            VectorFloat<?> vec = vectorTypeSupport.createFloatVector(dimension);
            ByteSequence<?> code = vectorTypeSupport.createByteSequence(codeSize);
            for (int newOrd = chunkStart; newOrd < chunkEnd; newOrd++) {
                int[] resolved = resolveSourceForNewOrd(newOrd);
                if (resolved == null) continue;  // hole; slot stays zero
                int srcIdx = resolved[0];
                int oldOrd = resolved[1];
                if (views[srcIdx] == null) {
                    views[srcIdx] = (OnDiskGraphIndex.View) sources.get(srcIdx).getView();
                }
                views[srcIdx].getVectorInto(oldOrd, vec, 0);
                code.zero();
                pq.encodeTo(vec, code);
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

    /**
     * Resolves a merged (new) ordinal back to its (sourceIdx, oldOrd) pair by consulting each
     * source's remapper in turn. Returns {@code null} when the ordinal is a hole (not covered
     * by any source). Each remapper's {@code newToOld} is O(1), so this is O(numSources) per call.
     */
    private int[] resolveSourceForNewOrd(int newOrd) {
        for (int s = 0; s < remappers.size(); s++) {
            int oldOrd = remappers.get(s).newToOld(newOrd);
            if (oldOrd != OrdinalMapper.OMITTED) {
                return new int[]{s, oldOrd};
            }
        }
        return null;
    }
}
