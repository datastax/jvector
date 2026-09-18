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
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;

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

    // Non-final: replaced by onRemappersUpdated when the compactor reassigns ordinals.
    private CompactionContext ctx;
    private final FusedFeature sourceFusedFeature;
    private final VectorCompressorRetrainer retrainer;

    private VectorCompressor<ByteSequence<?>> retrainedCompressor;

    // Transient pre-encode cache: lives in a memory-mapped section appended past the projected
    // end of the output graph file. Truncated away in onAfterClose. Off-heap, and chunked across
    // several mappings so it is not bounded by the 2 GB single-mapping limit — see
    // PreEncodedCodeCache for why that ceiling used to disable the pass entirely.
    private PreEncodedCodeCache codeCache;
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
    public void onRemappersUpdated(CompactionContext refreshed) {
        this.ctx = refreshed;
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
    public PreEncodedCodeCache getCodeCache() {
        return codeCache;
    }

    @Override
    public boolean writesCodesInline() {
        return true;
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
            cacheCodeSize = retrainedCompressor.compressedVectorSize();
            codeCache = precomputeCodeCache(ctx, writer, retrainedCompressor, "Code pre-encode");
            if (codeCache != null) {
                cacheTruncateAt = writer.projectedOutputSize();
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
            if (codeCache != null) {
                codeCache.close();
            }
            codeCache = null;
            closeSecondaryCache();
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
}
