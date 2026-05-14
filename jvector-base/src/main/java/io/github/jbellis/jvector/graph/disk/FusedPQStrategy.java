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
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;

/**
 * Strategy for compaction when sources carry the FUSED_PQ feature. PQ codes live inline in the
 * graph file (Quick-ADC layout, packed per parent's neighbors). This strategy:
 * <ol>
 *     <li>retrains a {@link ProductQuantization} codebook on a balanced sample of merged sources,</li>
 *     <li>pre-encodes every live node's PQ code once into a memory-mapped section appended past
 *         the projected end of the output graph file, so each parent's {@code writeInlineNodeRecord}
 *         can copy neighbor codes from the cache instead of re-encoding ~degree times,</li>
 *     <li>after compaction, when the graph has no hierarchy (level 0 only), writes the entry node's
 *         own PQ code after the L0 block — required by the reader for {@code hierarchyCachedFeatures}
 *         to find the entry node,</li>
 *     <li>on close, drops the mmap reference and truncates the output file back to its expected
 *         size, removing the transient pre-encode section.</li>
 * </ol>
 */
public final class FusedPQStrategy extends QuantizationCompactionStrategy {
    private static final Logger log = LoggerFactory.getLogger(FusedPQStrategy.class);
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final List<OnDiskGraphIndex> sources;
    private final List<FixedBitSet> liveNodes;
    private final List<OrdinalMapper> remappers;
    private final int dimension;
    private final int maxOrdinal;
    private final ForkJoinPool executor;
    private final int taskWindowSize;

    private ProductQuantization retrainedPQ;

    // Transient pre-encode cache: lives in a memory-mapped section appended past the projected
    // end of the output graph file (rather than a separate temp file). Truncated away in
    // onAfterClose. Off-heap so the Java heap is unaffected. Single-mapping limit (2 GB) caps
    // this implementation at ~10M nodes for typical pqCodeSize.
    private MappedByteBuffer pqCache;
    private int pqCacheCodeSize;
    private long pqCacheTruncateAt;  // > 0 means truncate to this size in onAfterClose

    public FusedPQStrategy(
            List<OnDiskGraphIndex> sources,
            List<FixedBitSet> liveNodes,
            List<OrdinalMapper> remappers,
            int dimension,
            int maxOrdinal,
            ForkJoinPool executor,
            int taskWindowSize) {
        this.sources = sources;
        this.liveNodes = liveNodes;
        this.remappers = remappers;
        this.dimension = dimension;
        this.maxOrdinal = maxOrdinal;
        this.executor = executor;
        this.taskWindowSize = taskWindowSize;
    }

    @Override
    public void retrain(VectorSimilarityFunction vsf) {
        PQRetrainer retrainer = new PQRetrainer(sources, liveNodes, dimension);
        // Existing single-arg retrain pulls base PQ from the FUSED_PQ feature on sources[0].
        this.retrainedPQ = retrainer.retrain(vsf);
    }

    @Override
    public VectorCompressor<?> compressor() {
        return retrainedPQ;
    }

    @Override
    public boolean writesCodesInline() {
        return true;
    }

    @Override
    public void onAfterHeader(CompactWriter writer) throws IOException {
        if (retrainedPQ == null) {
            throw new IllegalStateException("retrain() must be called before onAfterHeader()");
        }
        try {
            precomputePQCodes(writer);
            if (pqCache != null) {
                writer.enablePqCodeCache(pqCache, pqCacheCodeSize);
            }
        } catch (IOException e) {
            log.warn("PQ pre-encode failed, falling back to per-write encoding: {}", e.getMessage());
        }
    }

    @Override
    public void onAfterLevels(CompactWriter writer, int[] entryNodeSource, List<Integer> maxDegrees) throws IOException {
        // When FusedPQ is enabled and there is no hierarchy (only L0), the reader expects to
        // find the entry node's own PQ code written after the L0 block, just as
        // AbstractGraphIndexWriter.writeSparseLevels does in its getMaxLevel == 0 branch.
        // Without it, loadInMemoryFeatures reads garbage and hierarchyCachedFeatures is missing
        // the entry node, causing "Node X is not in the hierarchy" on first search.
        if (maxDegrees.size() != 1) {
            return;
        }
        try (var entryView = sources.get(entryNodeSource[0]).getView()) {
            var entryVec = vectorTypeSupport.createFloatVector(dimension);
            entryView.getVectorInto(entryNodeSource[1], entryVec, 0);
            var entryPqCode = vectorTypeSupport.createByteSequence(retrainedPQ.getSubspaceCount());
            entryPqCode.zero();
            retrainedPQ.encodeTo(entryVec, entryPqCode);
            writer.setEntryNodePqCode(entryPqCode);
        }
    }

    @Override
    public void onAfterClose(Path graphPath) {
        // Drop the mmap reference (the actual unmap happens at GC; truncate works on POSIX even
        // with a live mapping). Then truncate the output file to remove the transient PQ-code
        // section appended past projectedOutputSize.
        if (pqCacheTruncateAt > 0) {
            pqCache = null;
            try (FileChannel fc = FileChannel.open(graphPath, StandardOpenOption.WRITE)) {
                if (fc.size() > pqCacheTruncateAt) {
                    fc.truncate(pqCacheTruncateAt);
                }
            } catch (IOException e) {
                log.warn("Failed to truncate PQ cache section from output file {}: {}",
                        graphPath, e.getMessage());
            }
            pqCacheTruncateAt = 0;
        }
    }

    /**
     * Pre-computes PQ codes for every live node and stores them in a memory-mapped section
     * appended past the projected end of the OUTPUT graph file.
     */
    private void precomputePQCodes(CompactWriter writer) throws IOException {
        pqCacheCodeSize = retrainedPQ.getSubspaceCount();
        long tempSize = (long) (maxOrdinal + 1) * pqCacheCodeSize;
        if (tempSize <= 0 || tempSize > Integer.MAX_VALUE) {
            log.info("PQ pre-encode skipped: required cache size {} bytes exceeds single-mapping limit", tempSize);
            return;
        }

        // Place the temp section past the projected end of the output file. The writer never
        // touches this region; it's truncated away in onAfterClose.
        long tempOffset = writer.projectedOutputSize();
        pqCacheTruncateAt = tempOffset;
        long totalSize = tempOffset + tempSize;

        // Open a separate FileChannel on the output file, grow it to fit the temp section,
        // and map only the temp region. The CompactWriter holds an independent handle on the
        // same file via BufferedRandomAccessWriter — POSIX semantics let multiple FDs coexist
        // on the same file; since we only mmap a region the writer never writes to, no
        // coherency issues.
        try (FileChannel fc = FileChannel.open(writer.getOutputPath(),
                StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            ByteBuffer pad = ByteBuffer.wrap(new byte[]{0});
            fc.write(pad, totalSize - 1);
            pqCache = fc.map(FileChannel.MapMode.READ_WRITE, tempOffset, tempSize);
        }

        // Parallel-encode each live node's PQ code into pqCache. Each task takes a slab of
        // ordinals from one source so workers don't share the same source's RandomAccessReader.
        final int pqSize = pqCacheCodeSize;
        final ProductQuantization pq = retrainedPQ;
        List<Callable<Long>> tasks = new ArrayList<>();
        int targetTasks = Math.max(taskWindowSize * 4, 16);
        for (int s = 0; s < sources.size(); s++) {
            final int sIdx = s;
            final var source = sources.get(s);
            final var alive = liveNodes.get(s);
            final int upper = alive.length();
            int chunkSize = Math.max(256, (upper + targetTasks - 1) / targetTasks);
            for (int chunkStart = 0; chunkStart < upper; chunkStart += chunkSize) {
                final int cStart = chunkStart;
                final int cEnd = Math.min(chunkStart + chunkSize, upper);
                tasks.add(() -> {
                    ByteSequence<?> code = vectorTypeSupport.createByteSequence(pqSize);
                    VectorFloat<?> vec = vectorTypeSupport.createFloatVector(dimension);
                    long count = 0;
                    try (var view = source.getView()) {
                        for (int oldOrd = cStart; oldOrd < cEnd; oldOrd++) {
                            if (!alive.get(oldOrd)) continue;
                            view.getVectorInto(oldOrd, vec, 0);
                            code.zero();
                            pq.encodeTo(vec, code);
                            int newOrd = remappers.get(sIdx).oldToNew(oldOrd);
                            int offset = newOrd * pqSize;
                            // Absolute writes are thread-safe across disjoint regions; each
                            // newOrd is unique so workers never write to the same byte.
                            for (int i = 0; i < pqSize; i++) {
                                pqCache.put(offset + i, code.get(i));
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
            for (Future<Long> f : executor.invokeAll(tasks)) {
                total += f.get();
            }
            log.info("PQ pre-encode: {} nodes encoded into {} MB in-output cache (offset {})",
                    total, tempSize / (1024 * 1024), tempOffset);
        } catch (InterruptedException | ExecutionException e) {
            throw new IOException("PQ pre-encode failed", e);
        }
    }
}
