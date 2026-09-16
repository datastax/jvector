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

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.IntStream;
import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedFeature;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.*;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.github.jbellis.jvector.vector.types.ByteSequence;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static java.lang.Math.*;

/**
 * Merges several {@link OnDiskGraphIndex} sources into one on-disk graph.
 * <p>
 * Every live node keeps its own retained edges and gains cross-source edges from a bounded
 * search of the other sources, followed by Vamana-style diversity selection. At the base layer
 * the sources are processed smallest first: a node searches only the sources larger than its
 * own, and each hit is also offered back to the node it found, so the reverse direction of every
 * source pair is supplied by propagation rather than by a second search. The largest source
 * therefore runs no searches at all and only folds the offers it received into its retained
 * edges. Searches traverse the target's PQ codes (fused or sidecar) with exact rescoring of the
 * top candidates, hint the frontier's next records to the page cache while they run, and check
 * the pairwise diversity of offered candidates through their codes so no offerer's vector is read.
 * <p>
 * Ordinals in the output follow the caller's {@link OrdinalMapper}s unless
 * {@link #setReassignOrdinals} is enabled, in which case the compactor numbers nodes by locality
 * and publishes the mapping through {@link #effectiveRemappers()}. Upper layers are merged the
 * same way using a greedy descent to the layer followed by a beam search.
 */
public final class OnDiskGraphIndexCompactor implements Accountable {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final Logger log = LoggerFactory.getLogger(OnDiskGraphIndexCompactor.class);

    // Compaction constants
    private static final float DIVERSITY_ALPHA_STEP = 0.2f;
    private static final int BEAM_WIDTH_MULTIPLIER = 2;
    private static final int TARGET_BATCHES_PER_SOURCE = 40;
    private static final int TARGET_NODES_PER_BATCH = 128;
    private static final int MIN_SEARCH_TOP_K = 2;
    private static final int SEARCH_TOP_K_MULTIPLIER = 4;

    private List<OnDiskGraphIndex> sources;
    // Optional non-fused compressed sidecar, parallel to `sources`. Null when sources carry their
    // quantization inline (FUSED_PQ) or have none. When non-null, compact(Path, Path) retrains the
    // compressor on merged vectors and writes a single merged CompressedVectors to compressedPath.
    private final List<CompressedVectors> sourceCompressed;
    private List<FixedBitSet> liveNodes;
    private final List<Integer> numLiveNodesPerSource;
    private List<OrdinalMapper> remappers;
    private final List<Integer> maxDegrees;

    private final int dimension;
    private int maxOrdinal = -1;
    private int numTotalNodes = 0;
    private final ForkJoinPool executor;
    private final int taskWindowSize;
    private final VectorSimilarityFunction similarityFunction;


    // ---- Pair-asymmetric cross-linking (reverse-edge propagation) ----
    // L0 sources are processed in ascending live-size order with a barrier between sources, and a
    // node searches only sources LARGER than its own. The reverse direction of each source pair is
    // supplied by propagation instead of a search: when node u finds v in a larger source, u is
    // offered as a reverse candidate for v, and v's diversity selection (which runs in a later
    // group, after the barrier) unions those offers with v's retained same-source edges. Similarity
    // is symmetric and offers carry exact scores, so the propagated candidates are exactly what v's
    // own search would have scored — only WHERE candidates come from changes. The larger source of
    // every pair therefore does no cross-source searching at all; under skewed source sizes that
    // population dominates total search count, which is what this trades against the smaller
    // reverse candidate budget (REVERSE_CANDIDATE_SLOTS vs searchTopK per source).
    private static final int REVERSE_CANDIDATE_SLOTS = Integer.getInteger("jvector.compaction.reverseSlots", 16);

    private PreEncodedCodeCache orderingCache;   // non-null only while L0 runs under fused mode

    /**
     * Compactor-assigned ordinals (opt-in via {@link #setReassignOrdinals}): the compactor replaces
     * the caller's remappers with a mapping that numbers each source's live nodes by region (see
     * {@link #buildRegionOrdinalMappers}), sources in ascending-size processing order. Record write
     * offsets follow new ordinals, so processing in the same order makes the writer sequential, and
     * consecutive nodes of every source explore the same region of every target. Callers read the
     * mapping back via {@link #effectiveRemappers()}.
     */
    private boolean reassignOrdinals;
    private boolean ordinalsReassigned;
    private List<OrdinalMapper> effectiveRemappers;
    private int[] sizeRank;            // rank of each source in ascending live-node order
    private int[] l0ProcessOrder;      // source indices in ascending live-node order
    private BandedReverseCandidateBuffer reverseCandidates; // non-null only while L0 is being compacted
    // The sidecar quantization strategy participating in this compaction (NONE when the caller
    // did not supply per-source CompressedVectors). Set by compactGraphImpl; consulted wherever
    // the fused path consults the inline strategy for codes.
    private QuantizationCompactionStrategy activeSidecarStrategy = QuantizationCompactionStrategy.NONE;

    // ---- Cell join: level-0 cross-source candidates from a blocked scan of the code cache ----
    // With compactor-assigned ordinals every source's nodes are grouped by the level-1 node of the
    // largest source's hierarchy they descend to (their cell); each source's nodes of a cell form a
    // contiguous ordinal range and their codes are contiguous in the pre-encoded cache. A node's
    // candidates in another source are found by scanning that source's codes in the node's best
    // cells with an 8-bit lookup table, then rescoring the survivors exactly. Requires reassigned
    // ordinals with a hierarchy in the largest source and a merged code cache; otherwise level 0
    // falls back to the graph search.
    static final int CELL_BUDGET = 8192;      // codes scanned per (node, target) at most; governs how many cells are probed
    static final int CELL_ASSIGN_EF = 8;      // level-1 beam width when assigning a node's cell
    static final int CELL_PROBE_EF = 16;      // level-1 beam width when choosing a node's probe cells
    private HubMap cellMap;                   // resident hub map kept through level 0
    private int[] cellWalkPosition;           // hub node id -> cell (walk position); MAX_VALUE if not level 1
    private int[] cellToPosition;             // cell -> level-1 position in the hub map
    private int[][] cellStart;                // [source][cell] -> first new ordinal of that source's nodes in the cell
    private int cellCount;
    private final LongAdder cellCodesScanned = new LongAdder();
    private final LongAdder cellScans = new LongAdder();
    private final LongAdder cellFallbacks = new LongAdder();

    /** Reverse-offer band width: peak buffer memory is O(band), independent of node count. */
    private static final int OFFER_BAND_WIDTH = 1 << 20;
    private Path spillParent; // parent dir of the compaction output; set by compact()
    final java.util.concurrent.atomic.AtomicLong retainedOnlyNodes = new java.util.concurrent.atomic.AtomicLong();

    /** Orders sources by live-node count (ties by index) and allocates the reverse buffer. */
    private void setupCrossLink() {
        int k = sources.size();
        Integer[] order = new Integer[k];
        for (int s = 0; s < k; s++) order[s] = s;
        Arrays.sort(order, Comparator
                .comparingInt((Integer s) -> numLiveNodesPerSource.get(s))
                .thenComparingInt(s -> s));
        l0ProcessOrder = new int[k];
        sizeRank = new int[k];
        for (int i = 0; i < k; i++) {
            l0ProcessOrder[i] = order[i];
            sizeRank[order[i]] = i;
        }
        reverseCandidates = new BandedReverseCandidateBuffer(sources.size(), maxOrdinal + 1,
                                                             REVERSE_CANDIDATE_SLOTS, OFFER_BAND_WIDTH, spillParent);
        log.info("Cross-link: L0 source order {} (ascending live nodes), {} reverse slots/node, {} ordinals/band",
                Arrays.toString(l0ProcessOrder), REVERSE_CANDIDATE_SLOTS, OFFER_BAND_WIDTH);
    }

    /**
     * Constructs a new OnDiskGraphIndexCompactor for graphs without a non-fused compressed sidecar.
     * Equivalent to calling the 6-arg constructor with {@code sourceCompressed = null}.
     */
    @Experimental
    public OnDiskGraphIndexCompactor(
            List<OnDiskGraphIndex> sources,
            List<FixedBitSet> liveNodes,
            List<OrdinalMapper> remappers,
            VectorSimilarityFunction similarityFunction,
            ForkJoinPool executor) {
        this(sources, null, liveNodes, remappers, similarityFunction, executor);
    }

    /**
     * Constructs a new OnDiskGraphIndexCompactor to merge multiple graph indexes.
     * Initializes thread pool, validates inputs, and prepares metadata for compaction.
     *
     * @param sourceCompressed parallel to {@code sources}, supplying the non-fused compressed
     *                         vectors (e.g. {@link io.github.jbellis.jvector.quantization.PQVectors})
     *                         that ship alongside each graph. Pass {@code null} when sources carry
     *                         quantization inline (FUSED_PQ) or have none. Must not be combined
     *                         with sources that carry the FUSED_PQ feature.
     */
    @Experimental
    public OnDiskGraphIndexCompactor(
            List<OnDiskGraphIndex> sources,
            List<CompressedVectors> sourceCompressed,
            List<FixedBitSet> liveNodes,
            List<OrdinalMapper> remappers,
            VectorSimilarityFunction similarityFunction,
            ForkJoinPool executor) {
        checkBeforeCompact(sources, sourceCompressed, liveNodes, remappers);

        if (executor != null) {
            this.executor = executor;
        } else {
            // Default to the shared physical-core pool. Compaction (PQ encode + parallel record
            // flush) is compute- and memory-bandwidth-bound, so sizing to logical
            // cores oversubscribes hyperthreaded hosts and costs throughput. This pool is
            // process-wide and shared with index construction and quantization; the compactor
            // never owns or shuts it down.
            this.executor = PhysicalCoreExecutor.pool();
        }
        // Track the pool's real parallelism so task-window / backpressure sizing stays correct
        // whether the executor is the shared default or a caller-injected pool.
        this.taskWindowSize = this.executor.getParallelism();

        this.sources = sources;
        this.sourceCompressed = (sourceCompressed == null || sourceCompressed.isEmpty()) ? null : sourceCompressed;
        this.remappers = remappers;
        this.liveNodes = liveNodes;
        this.numLiveNodesPerSource = new ArrayList<>(this.sources.size());
        for (int s = 0; s < this.sources.size(); s++) {
            int numLiveNodes = this.liveNodes.get(s).cardinality();
            this.numTotalNodes += numLiveNodes;
            this.numLiveNodesPerSource.add(numLiveNodes);
        }

        maxDegrees = this.sources.stream()
                .max(Comparator.comparingInt(s -> s.maxDegrees().size()))
                .orElseThrow()
                .maxDegrees();
        dimension = this.sources.get(0).getDimension();
        for (var mapper : remappers) {
            maxOrdinal = max(mapper.maxOrdinal(), maxOrdinal);
        }
        this.similarityFunction = similarityFunction;
    }

    /**
     * Validates that all source indexes have compatible configurations and required features
     * before attempting compaction. Ensures consistent dimensions, max degrees, hierarchical
     * settings, and feature sets across all sources.
     */
    private void checkBeforeCompact(
            List<OnDiskGraphIndex> sources,
            List<CompressedVectors> sourceCompressed,
            List<FixedBitSet> liveNodes,
            List<OrdinalMapper> remappers) {
        validateInputSizes(sources, liveNodes, remappers);
        validateLiveNodesBounds(sources, liveNodes);
        validateGraphConfiguration(sources);
        validateFeatures(sources);
        validateCompressed(sources, sourceCompressed);
    }

    /**
     * Validates that the optional non-fused compressed sidecar list is consistent with
     * {@code sources}: same size, no nulls, identical compressor type across entries, and not
     * combined with FUSED_PQ (which already carries codes inline).
     */
    private void validateCompressed(List<OnDiskGraphIndex> sources, List<CompressedVectors> sourceCompressed) {
        if (sourceCompressed == null || sourceCompressed.isEmpty()) {
            return;
        }
        if (sourceCompressed.size() != sources.size()) {
            throw new IllegalArgumentException("sourceCompressed must have the same size as sources");
        }
        // Inline (fused) and sidecar are mutually exclusive ways to carry quantization codes.
        // Check for any fused feature rather than hard-coding FUSED_PQ so future fused types
        // (e.g. FUSED_ASH) are rejected here without further edits.
        for (var feature : sources.get(0).getFeatures().values()) {
            if (feature.isFused()) {
                throw new IllegalArgumentException(
                        "sourceCompressed cannot be combined with a fused feature ("
                                + feature.id() + "); choose one");
            }
        }
        Class<?> compressorClass = null;
        for (int s = 0; s < sourceCompressed.size(); s++) {
            CompressedVectors cv = Objects.requireNonNull(sourceCompressed.get(s),
                    "sourceCompressed[" + s + "] is null");
            var compressor = Objects.requireNonNull(cv.getCompressor(),
                    "sourceCompressed[" + s + "].getCompressor() is null");
            if (compressorClass == null) {
                compressorClass = compressor.getClass();
            } else if (compressorClass != compressor.getClass()) {
                throw new IllegalArgumentException(
                        "sourceCompressed entries must all use the same compressor type; got "
                                + compressorClass.getSimpleName() + " and "
                                + compressor.getClass().getSimpleName());
            }
        }
    }

    /**
     * Validates that input lists have consistent sizes and are non-null.
     */
    private void validateInputSizes(List<OnDiskGraphIndex> sources,
                                    List<FixedBitSet> liveNodes,
                                    List<OrdinalMapper> remappers) {
        if (sources.isEmpty()) {
            throw new IllegalArgumentException("Must have at least one source");
        }
        Objects.requireNonNull(liveNodes, "liveNodes");
        Objects.requireNonNull(remappers, "remappers");

        if (sources.size() != liveNodes.size()) {
            throw new IllegalArgumentException("sources and liveNodes must have the same size");
        }
        if (sources.size() != remappers.size()) {
            throw new IllegalArgumentException("sources and remappers must have the same size");
        }
    }

    /**
     * Validates that liveNodes bitsets match the size of their corresponding sources.
     */
    private void validateLiveNodesBounds(List<OnDiskGraphIndex> sources, List<FixedBitSet> liveNodes) {
        for (int s = 0; s < sources.size(); ++s) {
            if (liveNodes.get(s).length() != sources.get(s).getIdUpperBound()) {
                throw new IllegalArgumentException("source " + s + " out of bounds: liveNodes length "
                        + liveNodes.get(s).length() + " != idUpperBound " + sources.get(s).getIdUpperBound());
            }
        }
    }

    /**
     * Validates that all sources have consistent graph configuration (dimensions, degrees, hierarchy).
     */
    private void validateGraphConfiguration(List<OnDiskGraphIndex> sources) {
        int dimension = sources.get(0).getDimension();
        var refDegrees = sources.stream()
                .max(Comparator.comparingInt(s -> s.maxDegrees().size()))
                .orElseThrow()
                .maxDegrees();
        var addHierarchy = sources.get(0).isHierarchical();

        for (OnDiskGraphIndex source : sources) {
            if (source.getDimension() != dimension) {
                throw new IllegalArgumentException("sources must have the same dimension");
            }
            int sharedLevels = Math.min(refDegrees.size(), source.maxDegrees().size());
            for (int d = 0; d < sharedLevels; d++) {
                if (!Objects.equals(source.maxDegrees().get(d), refDegrees.get(d))) {
                    throw new IllegalArgumentException("sources must have the same max degrees");
                }
            }
            if (addHierarchy != source.isHierarchical()) {
                throw new IllegalArgumentException("sources must have the same hierarchical setting");
            }
        }
    }

    /**
     * Validates that all sources have compatible features for compaction.
     */
    private void validateFeatures(List<OnDiskGraphIndex> sources) {
        Set<FeatureId> refKeys = sources.get(0).getFeatures().keySet();
        boolean sameFeatures = sources.stream()
                .skip(1)
                .map(s -> s.getFeatures().keySet())
                .allMatch(refKeys::equals);

        if (!sameFeatures) {
            throw new IllegalArgumentException("Each source must have the same features");
        }
        if (!refKeys.contains(FeatureId.INLINE_VECTORS)) {
            throw new IllegalArgumentException("Each source must have the INLINE_VECTORS feature");
        }
    }

    /**
     * When enabled, the compactor chooses the merged ordinals itself and ignores the ordinal values
     * of the caller-supplied remappers (their source/oldOrdinal structure is still used to
     * enumerate nodes). The mapping actually used is available from {@link #effectiveRemappers()}
     * once {@code compact(...)} has started, and callers must translate through it. Letting the
     * compactor choose places similar vectors in adjacent records, keeps record writes sequential,
     * and makes consecutive cross-source searches walk the same neighbourhood of every target,
     * which is where most of the merge time goes. Requires PQ-bearing sources (fused or sidecar
     * codes); silently keeps caller ordinals otherwise.
     */
    @Experimental
    public void setReassignOrdinals(boolean enabled) {
        this.reassignOrdinals = enabled;
    }

    /** Number of level-0 cell scans run so far (one per node and target); for tests and diagnostics. */
    long cellScanCount() {
        return cellScans.sum();
    }

    /** The ordinal mappers in effect: the caller's, or the compactor-assigned mapping. */
    public List<OrdinalMapper> effectiveRemappers() {
        return effectiveRemappers != null ? effectiveRemappers : remappers;
    }

    /**
     * Main compaction entry point. Merges all source indexes into a single output index at the
     * specified path, handling PQ retraining if needed, and writing header, all layers, and footer.
     */
    @Experimental
    public void compact(Path outputPath) throws FileNotFoundException {
        QuantizationCompactionStrategy strategy = detectInlineStrategy();
        try {
            compactGraphImpl(outputPath, strategy);
        } finally {
            strategy.onAfterClose(outputPath);
        }
    }

    /**
     * Compaction entry point for graphs that ship a non-fused compressed sidecar (e.g.
     * {@link io.github.jbellis.jvector.quantization.PQVectors}). Writes the merged graph to
     * {@code graphPath} and the merged compressed vectors to {@code compressedPath}.
     * <p>
     * The compressor is retrained on a balanced sample of merged source vectors, then every live
     * node is re-encoded against the new codebook. Requires that {@code sourceCompressed} was
     * supplied to the constructor.
     */
    @Experimental
    public void compact(Path graphPath, Path compressedPath) throws FileNotFoundException {
        if (sourceCompressed == null) {
            throw new IllegalStateException(
                    "compact(graphPath, compressedPath) requires sourceCompressed to be supplied to the constructor");
        }
        Objects.requireNonNull(compressedPath, "compressedPath");

        // Graph compaction proceeds without fused-PQ retrain (validateCompressed forbids
        // FUSED_PQ when sourceCompressed is set), then the sidecar is written below.
        QuantizationCompactionStrategy inlineStrategy = detectInlineStrategy();
        QuantizationCompactionStrategy sidecarStrategy = detectSidecarStrategy();
        try {
            sidecarStrategy.retrain(similarityFunction);
            activeSidecarStrategy = sidecarStrategy;
            compactGraphImpl(graphPath, inlineStrategy);
            // Record the graph path with the sidecar strategy before writeSidecar: the strategy
            // defers its cache-region truncation until the sidecar copy completes.
            sidecarStrategy.onAfterClose(graphPath);
            sidecarStrategy.writeSidecar(compressedPath);
        } catch (IOException e) {
            throw new RuntimeException("Sidecar compaction failed", e);
        } finally {
            activeSidecarStrategy = QuantizationCompactionStrategy.NONE;
            inlineStrategy.onAfterClose(graphPath);
            // No-op after a successful writeSidecar; releases the cache mapping and truncates
            // the scratch region if a failure interrupted the normal flow.
            sidecarStrategy.releaseTransientState();
        }
    }

    /**
     * Pick the inline-codes strategy by asking the source's fused feature (if any) for its
     * compaction strategy. Returns {@link QuantizationCompactionStrategy#NONE} when no fused feature is
     * present. New fused quantization types extend the compactor purely by implementing
     * {@link FusedFeature#createCompactionStrategy}.
     */
    private QuantizationCompactionStrategy detectInlineStrategy() {
        for (var feature : sources.get(0).getFeatures().values()) {
            if (feature instanceof FusedFeature) {
                return ((FusedFeature) feature).createCompactionStrategy(buildContext());
            }
        }
        return QuantizationCompactionStrategy.NONE;
    }

    /**
     * Pick the sidecar-codes strategy by delegating to the first {@link CompressedVectors}'
     * own factory. Returns {@link QuantizationCompactionStrategy#NONE} when no sidecar input was supplied
     * to the constructor. New sidecar quantization types extend the compactor purely by
     * implementing {@link CompressedVectors#createCompactionStrategy}.
     */
    private QuantizationCompactionStrategy detectSidecarStrategy() {
        if (sourceCompressed == null) {
            return QuantizationCompactionStrategy.NONE;
        }
        return sourceCompressed.get(0).createCompactionStrategy(buildContext());
    }

    /** Snapshot the compactor's state into a {@link CompactionContext} for strategies to consume. */
    private CompactionContext buildContext() {
        return new CompactionContext(sources, sourceCompressed, liveNodes, remappers,
                dimension, maxOrdinal, executor, taskWindowSize);
    }

    /**
     * Internal graph-compaction body. Performs the full graph write but does <em>not</em> shut
     * down {@link #executor}; the public {@code compact(...)} entry points own that lifecycle so
     * follow-on passes (e.g. a sidecar write via {@link SidecarCompactionStrategy}) can keep using
     * the executor.
     * <p>
     * Quantization-aware steps (codebook retrain, pre-encode caches, entry-node tail records,
     * mmap cleanup) are delegated to {@code strategy}. For sources with no inline quantization,
     * pass {@link QuantizationCompactionStrategy#NONE} for a fully no-op strategy hook set.
     */
    private void compactGraphImpl(Path outputPath, QuantizationCompactionStrategy strategy) throws FileNotFoundException {
        this.spillParent = outputPath.toAbsolutePath().getParent();
        strategy.retrain(similarityFunction);

        boolean fusedPQEnabled = strategy.writesCodesInline();
        ProductQuantization pq = strategy.compressorAsPQ();
        if (pq == null) {
            // Sidecar mode: the retrained sidecar compressor is this merge's PQ — the ordinal
            // pass and (via the pre-encode cache) approximate scoring use it exactly as the
            // fused path uses the inline codebook.
            pq = activeSidecarStrategy.compressorAsPQ();
        }
        int maxBaseDegree = java.util.Collections.max(maxDegrees);
        io.github.jbellis.jvector.graph.disk.feature.FusedFeature outputFusedFeature =
                strategy.outputFusedFeature(maxBaseDegree);

        if (reassignOrdinals) {
            if (pq != null) {
                remappers = buildRegionOrdinalMappers(pq);
                effectiveRemappers = remappers;
                ordinalsReassigned = true;
                // the cell join scans the code cache by cell: store it blocked (subspace-major)
                strategy.setBlockedCodeLayout(cellMap != null);
                activeSidecarStrategy.setBlockedCodeLayout(cellMap != null);
                // The strategies snapshotted the caller's remappers at construction; refresh so
                // code placement (pre-encode caches, sidecar order) matches the on-disk ordinals.
                strategy.onRemappersUpdated(buildContext());
                activeSidecarStrategy.onRemappersUpdated(buildContext());
            } else {
                log.info("Ordinal reassignment requested but the sources carry no PQ codebook; keeping caller remappers");
            }
        }

        List<CommonHeader.LayerInfo> layerInfo = computeLayerInfoFromSources();
        int[] entryNodeSource = resolveEntryNodeSource(); // {sourceIdx, originalOrdinal}
        int entryNode = remappers.get(entryNodeSource[0]).oldToNew(entryNodeSource[1]);

        log.info("Writing compacted graph : {} total nodes, maxOrdinal={}, dimension={}, degree={}",
                numTotalNodes, maxOrdinal, dimension, maxDegrees.get(0));
        try (CompactWriter writer = new CompactWriter(outputPath, maxOrdinal, numTotalNodes, 0, layerInfo, entryNode, dimension, maxDegrees, outputFusedFeature)) {
            // Header has to be written first so the writer's position is past the header
            // before any strategy that mmaps past the projected end of the output runs.
            writer.writeHeader();
            strategy.onAfterHeader(writer);
            activeSidecarStrategy.onAfterHeader(writer);

            // Approximate cross-source scoring is available whenever a merged code cache exists:
            // fused (inline codes) or sidecar (strategy pre-encode cache built just above).
            boolean compressedPrecision = fusedPQEnabled || activeSidecarStrategy.getCodeCache() != null;
            compactLevels(writer, similarityFunction, fusedPQEnabled, compressedPrecision, pq);

            strategy.onAfterLevels(writer, entryNodeSource, maxDegrees);

            writer.writeFooter();
            log.info("Compaction complete: {}", outputPath);
        } catch (IOException | ExecutionException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Returns {sourceIdx, originalOrdinal} for the entry node of the compacted graph.
     * The chosen node must exist at maxLevel (since the on-disk format sets entryNode.level =
     * maxLevel). Prefers the designated entry node of any source whose maxLevel equals the global
     * maxLevel; if all such entry nodes are deleted, falls back to the first live node at maxLevel
     * across all sources.
     */
    private int[] resolveEntryNodeSource() {
        int maxLevel = sources.stream().mapToInt(OnDiskGraphIndex::getMaxLevel).max().orElse(0);

        // The on-disk format sets entryNode.level = layerInfo.size() - 1 (i.e. maxLevel).
        // So the chosen node must actually have neighbors written at maxLevel — meaning it
        // must exist at maxLevel in its source.  Prefer the designated entry node of a
        // maxLevel source; fall back to any live node that is at maxLevel.
        for (int s = 0; s < sources.size(); s++) {
            if (sources.get(s).getMaxLevel() == maxLevel) {
                int originalEntry = sources.get(s).getView().entryNode().node;
                if (liveNodes.get(s).get(originalEntry)) {
                    return new int[]{s, originalEntry};
                }
            }
        }

        // Entry nodes were all deleted: scan for any live node that exists at maxLevel.
        for (int s = 0; s < sources.size(); s++) {
            if (sources.get(s).getMaxLevel() < maxLevel) continue;
            NodesIterator it = sources.get(s).getNodes(maxLevel);
            while (it.hasNext()) {
                int node = it.next();
                if (liveNodes.get(s).get(node)) {
                    return new int[]{s, node};
                }
            }
        }

        throw new IllegalStateException("No live nodes found at maxLevel=" + maxLevel);
    }

    /**
     * Compacts all hierarchical levels of the graph, processing each level in batches.
     * For level 0 (base layer), writes inline vectors and neighbors. For upper layers,
     * writes only graph structure and optional PQ codes.
     */
    private void compactLevels(CompactWriter writer,
                                 VectorSimilarityFunction similarityFunction,
                                 boolean fusedPQEnabled,
                                 boolean compressedPrecision,
                                 ProductQuantization pq)
            throws IOException, ExecutionException, InterruptedException {

        int maxUpperDegree = 0;
        for (int level = 1; level < maxDegrees.size(); level++) {
            maxUpperDegree = Math.max(maxUpperDegree, maxDegrees.get(level));
        }

        int baseSearchTopK = Math.max(MIN_SEARCH_TOP_K, ((maxDegrees.get(0) + sources.size() - 1) / sources.size()) * SEARCH_TOP_K_MULTIPLIER);
        int baseMaxCandidateSize = baseSearchTopK * (sources.size() - 1) + maxDegrees.get(0) + REVERSE_CANDIDATE_SLOTS;
        int upperMaxPerSourceTopK = maxUpperDegree == 0 ? 0 : Math.max(MIN_SEARCH_TOP_K, ((maxUpperDegree + sources.size() - 1) / sources.size()) * SEARCH_TOP_K_MULTIPLIER);
        int upperMaxCandidateSize = upperMaxPerSourceTopK * sources.size();
        int maxCandidateSize = Math.max(baseMaxCandidateSize, upperMaxCandidateSize);
        int scratchDegree = Math.max(maxDegrees.get(0), Math.max(1, maxUpperDegree));
        final ThreadLocal<Scratch> threadLocalScratch = ThreadLocal.withInitial(() ->
            new Scratch(maxCandidateSize, scratchDegree, dimension, sources, pq)
        );

        setupCrossLink();
        orderingCache = fusedPQEnabled ? writer.pqCodeCache() : activeSidecarStrategy.getCodeCache();

        for (int level = 0; level < maxDegrees.size(); level++) {
            int searchTopK = Math.max(MIN_SEARCH_TOP_K, ((maxDegrees.get(level) + sources.size() - 1) / sources.size()) * SEARCH_TOP_K_MULTIPLIER);
            int beamWidth = Math.max(maxDegrees.get(level), searchTopK) * BEAM_WIDTH_MULTIPLIER;
            if (level == 0) log.info("Cross-link search budget: searchTopK={} per target, beamWidth={}", searchTopK, beamWidth);

            CompactionParams params = new CompactionParams(fusedPQEnabled, compressedPrecision, searchTopK, beamWidth, pq);

            if (level == 0) {
                log.info("Compacting level 0 (base layer)");

                ExecutorCompletionService<List<WriteResult>> ecs =
                        new ExecutorCompletionService<>(executor);

                java.util.function.Consumer<BatchSpec> submitOne = (bs) -> {
                    ecs.submit(() -> {
                        Scratch scratch = threadLocalScratch.get();
                        return computeBaseBatch(writer, bs, scratch, params);
                    });
                };

                var wropts = EnumSet.of(StandardOpenOption.WRITE, StandardOpenOption.READ);
                try (FileChannel fc = FileChannel.open(writer.getOutputPath(), wropts)) {
                    java.util.function.Consumer<List<WriteResult>> writeResults = (results) -> {
                        try {
                            for (WriteResult r : results) {
                                ByteBuffer b = r.data;
                                long pos = r.fileOffset;
                                while (b.hasRemaining()) {
                                    int n = fc.write(b, pos);
                                    pos += n;
                                }
                            }
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    };

                    // Sources run smallest-first, one group at a time: the drain between groups is
                    // the barrier that guarantees every reverse-candidate offer into a source has
                    // completed before that source's own nodes read them.
                    for (int s : l0ProcessOrder) {
                        runBatchesWithBackpressure(buildBatchesForSource(s, 0), ecs, submitOne, writeResults);
                    }
                }

                log.info("Cross-link reverse propagation: {} offers onto {} touched of {} nodes ({} slots/node), {} retained-only fast-path nodes",
                        reverseCandidates.offered(), reverseCandidates.touchedTargets(),
                        maxOrdinal + 1, REVERSE_CANDIDATE_SLOTS, retainedOnlyNodes.get());
                if (cellMap != null) {
                    long scans = Math.max(1, cellScans.sum());
                    log.info("Cell join: {} scans, {} codes/scan, {} nodes without a cell (graph search)",
                             cellScans.sum(), cellCodesScanned.sum() / scans, cellFallbacks.sum());
                }
                reverseCandidates.close();
                reverseCandidates = null; // consumed entirely within L0; scales with node count
                orderingCache = null;
                writer.offsetAfterInline();

            } else {
                final int lvl = level;
                log.info("Compacting upper layer {}", level);
                List<BatchSpec> batches = buildBatches(level);

                ExecutorCompletionService<List<UpperLayerWriteResult>> ecs =
                        new ExecutorCompletionService<>(executor);

                java.util.function.Consumer<BatchSpec> submitOne = (bs) -> {
                    ecs.submit(() -> {
                        Scratch scratch = threadLocalScratch.get();
                        return computeUpperBatchForLevel(bs, lvl, scratch, params);
                    });
                };

                runBatchesWithBackpressure(
                        batches,
                        ecs,
                        submitOne,
                        (results) -> {
                            try {
                                for (UpperLayerWriteResult r : results) {
                                    writer.writeUpperLayerNode(
                                            lvl,
                                            r.ordinal,
                                            r.neighbors,
                                            r.pqCode
                                    );
                                }
                            } catch (IOException e) {
                                throw new RuntimeException(e);
                            }
                        }
                );
            }
        }

        Scratch s = threadLocalScratch.get();
        s.close();
        threadLocalScratch.remove();
    }

    /**
     * Divides nodes at a given level across all source indexes into processing batches
     * for parallel execution. Each batch contains a subset of nodes from one source.
     */
    private List<BatchSpec> buildBatches(int level) {
        List<BatchSpec> batches = new ArrayList<>();
        for (int s = 0; s < sources.size(); ++s) {
            batches.addAll(buildBatchesForSource(s, level));
        }
        return batches;
    }

    /**
     * Builds the processing batches for one source at one level. Split out from
     * {@link #buildBatches} so L0 compaction can run sources one group at a time in size order
     * (the cross-link barrier); upper layers still batch all sources together.
     */
    private List<BatchSpec> buildBatchesForSource(int s, int level) {
        List<BatchSpec> batches = new ArrayList<>();
        var source = sources.get(s);
        if (level > source.getMaxLevel()) return batches;

        int[] nodes;
        int numNodes;
        if (level == 0) {
            // Enumerate live L0 nodes from the in-memory liveNodes bitset. source.getNodes(0)
            // seeks and reads a 4-byte id at every node's record offset — a full random disk
            // scan of the source (the dominant cost of full-precision compaction disk-cold,
            // where nothing warms the cache first), and unnecessary: liveNodes already holds
            // exactly the live ordinals. Also skips dead nodes up front rather than in-batch.
            FixedBitSet alive = liveNodes.get(s);
            numNodes = alive.cardinality();
            nodes = new int[numNodes];
            int i = 0;
            for (int n = alive.nextSetBit(0);
                 n != DocIdSetIterator.NO_MORE_DOCS;
                 n = alive.nextSetBit(n + 1)) {
                nodes[i++] = n;
            }
            if (ordinalsReassigned && numNodes > 1) {
                // Merged ordinals were assigned in region order, so ordering processing by
                // new ordinal gives locality AND sequential record writes at once.
                OrdinalMapper mapper = remappers.get(s);
                long[] keyed = new long[numNodes];
                for (int k = 0; k < numNodes; k++) {
                    keyed[k] = ((long) mapper.oldToNew(nodes[k]) << 32) | (nodes[k] & 0xFFFFFFFFL);
                }
                Arrays.parallelSort(keyed);
                for (int k = 0; k < numNodes; k++) {
                    nodes[k] = (int) keyed[k];
                }
                log.info("L0 source {}: {} nodes in region-ordinal order", s, numNodes);
            }
            // Similarity-ordered scheduling: sort searching sources' nodes by the leading bytes
            // of their PQ code, so consecutive searches walk overlapping target regions. The
            // largest source runs no searches and keeps ordinal order (contiguous record
            // streaming matters more there).
            boolean searches = reverseCandidates == null || sizeRank[s] < sources.size() - 1;
            if (!ordinalsReassigned && orderingCache != null && searches && orderingCache.codeSize() >= 4 && numNodes > 1) {
                OrdinalMapper mapper = remappers.get(s);
                byte[] code = new byte[orderingCache.codeSize()];
                // Two-level order: similarity-sort WITHIN coarse ordinal chunks. A record's write
                // offset follows its ordinal, so a global similarity sort scatters the (single
                // threaded) writer's pwrites across the whole L0 region and random-page writeback
                // throttling becomes the pipeline ceiling; chunking bounds the write window while
                // consecutive nodes remain similar within each chunk.
                int segStart = 0;
                while (segStart < numNodes) {
                    int chunk = nodes[segStart] >>> 22;
                    int segEnd = segStart + 1;
                    while (segEnd < numNodes && (nodes[segEnd] >>> 22) == chunk) {
                        segEnd++;
                    }
                    int len = segEnd - segStart;
                    if (len > 1) {
                        long[] keyed = new long[len];
                        for (int k = 0; k < len; k++) {
                            orderingCache.get(mapper.oldToNew(nodes[segStart + k]), code);
                            long key = ((code[0] & 0xFFL) << 24) | ((code[1] & 0xFFL) << 16)
                                     | ((code[2] & 0xFFL) << 8) | (code[3] & 0xFFL);
                            keyed[k] = (key << 32) | (nodes[segStart + k] & 0xFFFFFFFFL);
                        }
                        Arrays.parallelSort(keyed);
                        for (int k = 0; k < len; k++) {
                            nodes[segStart + k] = (int) keyed[k];
                        }
                    }
                    segStart = segEnd;
                }
                log.info("L0 source {}: {} nodes in similarity order within {}-node ordinal chunks",
                        s, numNodes, 1 << 22);
            }
        } else {
            NodesIterator sourceNodes = source.getNodes(level);
            numNodes = sourceNodes.size();
            nodes = new int[numNodes];
            int i = 0;
            while (sourceNodes.hasNext()) {
                nodes[i++] = sourceNodes.next();
            }
        }

        int numBatches = max(TARGET_BATCHES_PER_SOURCE, (numNodes + TARGET_NODES_PER_BATCH - 1) / TARGET_NODES_PER_BATCH);
        if (numBatches > numNodes) numBatches = numNodes;
        int batchSize = numBatches == 0 ? 0 : (numNodes + numBatches - 1) / numBatches;
        for (int b = 0; b < numBatches; ++b) {
            int start = min(numNodes, batchSize * b);
            int end = min(numNodes, batchSize * (b + 1));
            batches.add(new BatchSpec(s, nodes, start, end));
        }

        return batches;
    }

    /**
     * Processes a batch of base layer (level 0) nodes from one source index. For each live node,
     * gathers candidates from all sources, applies diversity selection, and creates write results
     * containing the full node record data.
     */
   private List<WriteResult> computeBaseBatch(CompactWriter writer,
                                              BatchSpec bs,
                                              Scratch scratch,
                                              CompactionParams params) throws IOException {

        List<WriteResult> out = new ArrayList<>(bs.end - bs.start);
        if (bs.end > bs.start) {
            // Stream this batch's own records into the page cache before processing. Search
            // reads into other sources are data-dependent and stay demand-faulted, but each
            // node's own record read (adjacency + vector) is fully predictable. Under
            // similarity ordering the batch's ordinals are scattered, so only prefetch when
            // they still form a reasonably dense range.
            int lo = Integer.MAX_VALUE;
            int hi = -1;
            for (int i = bs.start; i < bs.end; i++) {
                lo = Math.min(lo, bs.nodes[i]);
                hi = Math.max(hi, bs.nodes[i]);
            }
            if ((long) hi - lo <= 8L * (bs.end - bs.start)) {
                sources.get(bs.sourceIdx).prefetchL0Records(lo, hi);
            }
        }

        for (int i = bs.start; i < bs.end; i++) {
            int node = bs.nodes[i];
            if (!liveNodes.get(bs.sourceIdx).get(node)) continue;

            out.add(processBaseNode(node, bs.sourceIdx, scratch, writer, params));
        }

        return out;
    }

    /**
     * Read-free hub pass: reverse-offer candidates keep the exact score their offerer computed, and
     * their pairwise diversity checks use a symmetric code-code similarity over the merged PQ, so
     * folding an offer never reads the offerer's vector. Available whenever the merged codes are
     * cached and the similarity function has a code-code form.
     */
    private boolean codeDiversityAvailable(CompactionParams params) {
        return orderingCache != null && params.pq != null && SymmetricCodeSimilarity.supports(similarityFunction);
    }

    /** Symmetric code-code similarity over the merged PQ, built once per compaction on first use. */
    private volatile SymmetricCodeSimilarity codeSimilarity;

    private SymmetricCodeSimilarity codeSimilarity(ProductQuantization pq) {
        SymmetricCodeSimilarity c = codeSimilarity;
        if (c == null) {
            synchronized (this) {
                c = codeSimilarity;
                if (c == null) {
                    codeSimilarity = c = new SymmetricCodeSimilarity(pq, similarityFunction);
                }
            }
        }
        return c;
    }

    /**
     * Processes a batch of upper layer nodes from one source index. Similar to base layer
     * processing but returns only ordinal, neighbors, and optional PQ code (no inline vectors).
     */
    private List<UpperLayerWriteResult> computeUpperBatchForLevel(
            BatchSpec bs,
            int level,
            Scratch scratch,
            CompactionParams params
    ) {
        List<UpperLayerWriteResult> results =
                new ArrayList<>(bs.end - bs.start);

        for (int i = bs.start; i < bs.end; i++) {
            int node = bs.nodes[i];

            if (!liveNodes.get(bs.sourceIdx).get(node)) continue;

            results.add(processUpperNode(node, bs.sourceIdx, level, scratch, params));
        }

        return results;
    }

    /**
     * Processes a single base layer node: retrieves its vector, gathers diverse candidates from
     * all sources, selects best neighbors using diversity criteria, remaps ordinals, and returns
     * the complete write result for this node.
     */
    private WriteResult processBaseNode(
            int node,
            int sourceIdx,
            Scratch scratch,
            CompactWriter writer,
            CompactionParams params
    ) throws IOException {

        // Retained-only fast path: a node of the largest source runs no forward searches, so if
        // it also received no reverse candidates its candidate set is exactly its retained
        // same-source edges — and re-running diversity over an already-diversity-selected edge
        // set is a fixed point. Skip selection entirely: filter dead neighbors, remap, write.
        if (reverseCandidates != null && sizeRank[sourceIdx] == sources.size() - 1) {
            int newOrdinal = remappers.get(sourceIdx).oldToNew(node);
            if (reverseCandidates.countAt(sourceIdx, newOrdinal) == 0) {
                return writeRetainedOnlyRecord(node, sourceIdx, newOrdinal, scratch, writer);
            }
        }

        var sourceView = (OnDiskGraphIndex.View) scratch.gs[sourceIdx].getView();
        sourceView.getVectorInto(node, scratch.baseVec, 0);

        int candSize = gatherCandidates(node, 0, sourceIdx, scratch, scratch.baseVec, params);

        int[] order = IntStream.range(0, candSize).toArray();
        sortOrderByScoreDesc(order, scratch.candScore, candSize);

        var selected = scratch.selectedCache;

        var provider = new CompactVamanaDiversityProvider(similarityFunction, 1.2f);
        if (codeDiversityAvailable(params)) {
            provider.withCodes(orderingCache, remappers, codeSimilarity(params.pq), scratch.candCodeOnly);
        }
        provider.retainDiverse(
                scratch.candSrc,
                scratch.candNode,
                scratch.candScore,
                order,
                candSize,
                maxDegrees.get(0),
                selected,
                scratch.tmpVec,
                scratch.gs
        );

        // Reverse-edge propagation (as in single-graph Vamana insertion): this node offers
        // itself only to the cross-source neighbors its own selection KEPT, not to everything
        // its searches surfaced. Each target folds its accumulated reverse edges into its one
        // diversity pass when its group runs; scores are exact and similarity is symmetric, so
        // the offer carries the score the target's own search would have computed.
        if (reverseCandidates != null) {
            for (int k = 0; k < selected.size; k++) {
                int ssrc = selected.sourceIdx[k];
                if (ssrc != sourceIdx && sizeRank[ssrc] > sizeRank[sourceIdx]) {
                    int targetNew = remappers.get(ssrc).oldToNew(selected.nodes[k]);
                    reverseCandidates.offer(ssrc, targetNew, sourceIdx, node, selected.scores[k]);
                }
            }
        }

        // remap
        for (int k = 0; k < selected.size; k++) {
            selected.nodes[k] =
                    remappers.get(selected.sourceIdx[k])
                            .oldToNew(selected.nodes[k]);
        }

        int newOrdinal = remappers.get(sourceIdx).oldToNew(node);

        return writer.writeInlineNodeRecord(
                newOrdinal,
                scratch.baseVec,
                selected,
                scratch.pqCode
        );
    }

    /**
     * Writes a record whose neighbors are the node's live retained same-source edges, unchanged
     * and in their original order — used by the retained-only fast path. Neighbor vectors are
     * read only when the writer must encode per-neighbor codes from them (fused output without
     * the pre-encoded code cache); otherwise the only read is the node's own record.
     */
    private WriteResult writeRetainedOnlyRecord(int node, int sourceIdx, int newOrdinal,
                                                Scratch scratch, CompactWriter writer) throws IOException {
        var view = (OnDiskGraphIndex.View) scratch.gs[sourceIdx].getView();
        view.getVectorInto(node, scratch.baseVec, 0);
        FixedBitSet alive = liveNodes.get(sourceIdx);
        OrdinalMapper mapper = remappers.get(sourceIdx);
        var selected = scratch.selectedCache;
        selected.reset();
        boolean needVecs = writer.needsNeighborVectors();

        var it = view.getNeighborsIterator(0, node);
        while (it.hasNext()) {
            int nb = it.nextInt();
            if (!alive.get(nb)) continue;
            if (needVecs) {
                view.getVectorInto(nb, scratch.tmpVec, 0);
                selected.add(sourceIdx, view, nb, 0f, scratch.tmpVec);
            } else {
                selected.sourceIdx[selected.size] = sourceIdx;
                selected.views[selected.size] = view;
                selected.nodes[selected.size] = nb;
                selected.scores[selected.size] = 0f;
                selected.size++;
            }
        }
        for (int k = 0; k < selected.size; k++) {
            selected.nodes[k] = mapper.oldToNew(selected.nodes[k]);
        }
        retainedOnlyNodes.incrementAndGet();
        return writer.writeInlineNodeRecord(newOrdinal, scratch.baseVec, selected, scratch.pqCode);
    }

    /**
     * Processes a single upper layer node: similar to base layer processing but only returns
     * graph structure (ordinal and neighbors) and optional PQ encoding for level 1.
     */
    private UpperLayerWriteResult processUpperNode(
            int node,
            int sourceIdx,
            int level,
            Scratch scratch,
            CompactionParams params
    ) {
        var sourceView = (OnDiskGraphIndex.View) scratch.gs[sourceIdx].getView();
        sourceView.getVectorInto(node, scratch.baseVec, 0);

        int candSize = gatherCandidates(node, level, sourceIdx, scratch, scratch.baseVec, params);

        int[] order = IntStream.range(0, candSize).toArray();
        sortOrderByScoreDesc(order, scratch.candScore, candSize);

        var selected = scratch.selectedCache;

        new CompactVamanaDiversityProvider(similarityFunction, 1.2f)
                .retainDiverse(
                        scratch.candSrc,
                        scratch.candNode,
                        scratch.candScore,
                        order,
                        candSize,
                        maxDegrees.get(level),
                        selected,
                        scratch.tmpVec,
                        scratch.gs
                );

        // remap
        for (int k = 0; k < selected.size; k++) {
            selected.nodes[k] =
                    remappers.get(selected.sourceIdx[k])
                            .oldToNew(selected.nodes[k]);
        }

        int newOrdinal = remappers.get(sourceIdx).oldToNew(node);

        ByteSequence<?> pqCode = maybeEncodePQ(level, scratch, params);

        return new UpperLayerWriteResult(newOrdinal, selected, pqCode);
    }

    /**
     * Encodes a vector using Product Quantization if enabled and the level is 1.
     * Returns null otherwise.
     */
    private ByteSequence<?> maybeEncodePQ(int level, Scratch scratch, CompactionParams params) {
        if (!params.fusedPQEnabled || level != 1) {
            return null;
        }

        scratch.pqCode.zero();
        params.pq.encodeTo(scratch.baseVec, scratch.pqCode);
        return scratch.pqCode.copy();
    }

    /**
     * Collects neighbor candidates for a node from all source indexes. For the source containing
     * the node, uses existing neighbors; for other sources, performs graph search. Returns the
     * total number of candidates gathered.
     */
    private int gatherCandidates(
            int node,
            int level,
            int sourceIdx,
            Scratch scratch,
            VectorFloat<?> baseVec,
            CompactionParams params
    ) {
        int candSize = 0;
        Arrays.fill(scratch.candCodeOnly, false);

        for (int ss = 0; ss < sources.size(); ss++) {
            var searchView = (OnDiskGraphIndex.View) scratch.gs[ss].getView();
            var indexAlive = liveNodes.get(ss);

            if (ss == sourceIdx) {
                candSize = gatherFromSameSource(node, level, ss, searchView, indexAlive,
                                                 baseVec, scratch, candSize);
            } else {
                // Cross-link: at L0 only search LARGER sources; candidates from smaller sources
                // arrive via reverse propagation (consumed below), offered when those sources'
                // nodes searched this one in an earlier group.
                if (level == 0 && reverseCandidates != null && sizeRank[ss] < sizeRank[sourceIdx]) {
                    continue;
                }
                candSize = gatherFromOtherSource(node, sourceIdx, level, ss, searchView, indexAlive,
                                                  baseVec, scratch, candSize, params);
            }
        }

        if (level == 0 && reverseCandidates != null) {
            int offersStart = candSize;
            candSize = reverseCandidates.appendTo(sourceIdx, remappers.get(sourceIdx).oldToNew(node),
                    scratch.candSrc, scratch.candNode, scratch.candScore, candSize);
            // Offers carry the exact score their offerer computed; their pairwise diversity checks
            // run on codes, so their vectors are never read here.
            if (codeDiversityAvailable(params)) {
                Arrays.fill(scratch.candCodeOnly, offersStart, candSize, true);
            }
        }

        return candSize;
    }

    /**
     * Gathers candidates from the same source index that contains the node.
     * Simply iterates through existing neighbors.
     */
    private int gatherFromSameSource(int node, int level, int sourceIdx,
                                     OnDiskGraphIndex.View searchView, FixedBitSet indexAlive,
                                     VectorFloat<?> baseVec, Scratch scratch, int candSize) {
        var it = searchView.getNeighborsIterator(level, node);
        while (it.hasNext()) {
            int nb = it.nextInt();
            if (!indexAlive.get(nb)) continue;

            searchView.getVectorInto(nb, scratch.tmpVec, 0);

            scratch.candSrc[candSize] = sourceIdx;
            scratch.candNode[candSize] = nb;
            scratch.candScore[candSize] = similarityFunction.compare(baseVec, scratch.tmpVec);
            candSize++;
        }
        return candSize;
    }

    /**
     * Gathers candidates from a different source index via graph search.
     */
    private int gatherFromOtherSource(int node, int nodeSourceIdx, int level, int sourceIdx,
                                      OnDiskGraphIndex.View searchView, FixedBitSet indexAlive,
                                      VectorFloat<?> baseVec, Scratch scratch, int candSize,
                                      CompactionParams params) {
        SearchScoreProvider ssp = buildCrossSourceScoreProvider(
                params.compressedPrecision,
                sources.get(sourceIdx),
                sourceIdx,
                searchView,
                baseVec,
                scratch.tmpVec,
                similarityFunction,
                params.pq
        );

        if (level == 0 && cellMap != null && orderingCache != null) {
            candSize = gatherFromOtherSourceByCells(node, nodeSourceIdx, sourceIdx, searchView, baseVec, scratch, candSize, params);
            if (scratch.probeCount > 0) {
                return candSize;
            }
            cellFallbacks.increment();   // node without a cell: graph search below
        }

        if (level == 0) {
            // rerankK = searchTopK, not beamWidth: the wider beam's extra candidates are
            // largely pruned by diversity selection, so the doubled approximate-phase cost
            // buys almost no recall.
            SearchResult results = scratch.gs[sourceIdx].search(
                    ssp, params.searchTopK, params.searchTopK, 0f, 0f, indexAlive
            );
            for (var r : results.getNodes()) {
                scratch.candSrc[candSize] = sourceIdx;
                scratch.candNode[candSize] = r.node;
                scratch.candScore[candSize] =
                        params.compressedPrecision
                                ? rescore(searchView, r.node, baseVec, scratch.tmpVec)
                                : r.score;
                candSize++;
            }
        } else {
            var entry = searchView.entryNode();
            if (level > entry.level) return candSize;
            scratch.gs[sourceIdx].initializeInternal(ssp, entry, Bits.ALL);

            // Descend greedily through levels above the target level, so the search at
            // `level` starts from the best-known region rather than the global entry node.
            // This mirrors how GraphSearcher.searchInternal navigates the hierarchy.
            for (int l = entry.level; l > level; l--) {
                scratch.gs[sourceIdx].searchOneLayer(ssp, 1, 0f, l, Bits.ALL);
                scratch.gs[sourceIdx].setEntryPointsFromPreviousLayer();
            }

            scratch.gs[sourceIdx].searchOneLayer(
                    ssp, params.searchTopK, 0f, level, indexAlive
            );

            int prev_candSize = candSize;
            candSize = appendApproximateResults(
                    scratch.gs[sourceIdx].approximateResults(),
                    sourceIdx,
                    scratch,
                    candSize
            );

            if (params.compressedPrecision) {
                for (int i = prev_candSize; i < candSize; i++) {
                    scratch.candScore[i] = rescore(
                            searchView,
                            scratch.candNode[i],
                            baseVec,
                            scratch.tmpVec
                    );
                }
            }
        }

        return candSize;
    }

    /**
     * Cross-source candidates by cell scan. The node's own cell is implied by its reassigned
     * ordinal; a small beam over the hub's level-1 graph seeded there ranks the cells it scores
     * (once per node, shared by all targets). The target's codes in those cells are scanned best
     * cell first with the node's 8-bit table, until {@link #CELL_BUDGET} codes, keeping the top
     * {@code searchTopK}; those are rescored exactly.
     * Leaves {@code probeCount == 0} when the node has no cell, in which case the caller falls
     * back to the graph search.
     */
    private int gatherFromOtherSourceByCells(int node, int nodeSourceIdx, int targetIdx,
                                             OnDiskGraphIndex.View targetView, VectorFloat<?> baseVec,
                                             Scratch scratch, int candSize, CompactionParams params) {
        if (scratch.cellScorer == null) {
            scratch.cellScorer = cellMap.scorer();
            scratch.topIds = new int[params.searchTopK];
            scratch.topScores = new float[params.searchTopK];
            scratch.probeCells = new int[beamScoredCapacity(CELL_PROBE_EF, cellMap.degree[1])];
        }
        HubMap.Scorer scorer = scratch.cellScorer;
        if (scratch.probeNode != node || scratch.probeSrc != nodeSourceIdx) {
            scratch.probeNode = node;
            scratch.probeSrc = nodeSourceIdx;
            scratch.probeCount = 0;
            scorer.setQuery(baseVec);
            int ownCell = cellOf(nodeSourceIdx, remappers.get(nodeSourceIdx).oldToNew(node));
            if (ownCell < 0) {
                return candSize;
            }
            selectCellsByBeam(scratch, scorer, cellToPosition[ownCell]);
        }
        if (scratch.probeCount == 0) {
            return candSize;
        }

        final int[] starts = cellStart[targetIdx];
        final int k = params.searchTopK;
        final int[] ids = scratch.topIds;
        final float[] sc = scratch.topScores;
        final int codeSize = orderingCache.codeSize();
        final int msub = cellMap.subspaceCount;
        final byte[] lut8 = scorer.lut8();
        int size = 0;
        long scanned = 0;
        for (int i = 0; i < scratch.probeCount; i++) {
            int cell = scratch.probeCells[i];
            int lo = starts[cell], hi = starts[cell + 1];
            int n = hi - lo;
            if (n <= 0) continue;
            if (scanned > 0 && scanned + n > CELL_BUDGET) {
                break;
            }
            // whole 64-code blocks covering [lo, hi): the kernel scores every code in them,
            // only the in-range ones are ranked
            int blockCount = ((hi - 1) >>> 6) - (lo >>> 6) + 1;
            int needBytes = blockCount * PreEncodedCodeCache.BLOCK * codeSize;
            if (scratch.cellBlock.length < needBytes) {
                scratch.cellBlock = new byte[Math.max(needBytes, scratch.cellBlock.length * 2)];
                scratch.cellBlockSums = new short[scratch.cellBlock.length / codeSize];
            }
            int blockStart = orderingCache.copyBlocks(lo, hi, scratch.cellBlock);
            PqScanKernel.scan(scratch.cellBlock, blockCount, msub, lut8, scratch.cellBlockSums);
            final short[] sums = scratch.cellBlockSums;
            final short[] norms;
            if (cellMap.cosine) {
                // second additive pass: decoded squared norms, so the ratio ranks like cosine
                if (scratch.cellBlockNorms.length < scratch.cellBlockSums.length) {
                    scratch.cellBlockNorms = new short[scratch.cellBlockSums.length];
                }
                PqScanKernel.scan(scratch.cellBlock, blockCount, msub, cellMap.mag8, scratch.cellBlockNorms);
                norms = scratch.cellBlockNorms;
            } else {
                norms = null;
            }
            final float dScale = scorer.lut8Scale, dOffset = scorer.lut8Offset;
            final float nScale = cellMap.mag8Scale, nOffset = cellMap.mag8Offset + cellMap.centerNorm2;
            scanned += n;
            for (int newOrd = lo; newOrd < hi; newOrd++) {
                int rel = newOrd - blockStart;
                float score;
                if (norms != null) {
                    float dot = (sums[rel] & 0xFFFF) / dScale + dOffset;
                    float n2 = (norms[rel] & 0xFFFF) / nScale + nOffset;
                    score = dot / (float) Math.sqrt(Math.max(1e-12f, n2));
                } else {
                    score = sums[rel] & 0xFFFF;   // higher is better (table oriented that way)
                }
                if (size < k) {
                    int pos = size++;
                    while (pos > 0) {
                        int parent = (pos - 1) >>> 1;
                        if (sc[parent] <= score) break;
                        sc[pos] = sc[parent]; ids[pos] = ids[parent]; pos = parent;
                    }
                    sc[pos] = score; ids[pos] = newOrd;
                } else if (score > sc[0]) {
                    int pos = 0;
                    while (true) {
                        int child = 2 * pos + 1;
                        if (child >= size) break;
                        if (child + 1 < size && sc[child + 1] < sc[child]) child++;
                        if (sc[child] >= score) break;
                        sc[pos] = sc[child]; ids[pos] = ids[child]; pos = child;
                    }
                    sc[pos] = score; ids[pos] = newOrd;
                }
            }
        }
        cellCodesScanned.add(scanned);
        cellScans.increment();

        OrdinalMapper mapper = remappers.get(targetIdx);
        for (int i = 0; i < size; i++) {
            int old = mapper.newToOld(ids[i]);
            scratch.candSrc[candSize] = targetIdx;
            scratch.candNode[candSize] = old;
            scratch.candScore[candSize] = rescore(targetView, old, baseVec, scratch.tmpVec);
            candSize++;
        }
        return candSize;
    }


    /** Cell (walk position) whose ordinal range of {@code source} contains {@code newOrdinal}; -1 if none. */
    private int cellOf(int source, int newOrdinal) {
        int[] starts = cellStart[source];
        int lo = 0, hi = cellCount;   // cells [0, cellCount) plus the no-cell bucket at cellCount
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (starts[mid + 1] <= newOrdinal) lo = mid + 1; else hi = mid;
        }
        return lo < cellCount && starts[lo] <= newOrdinal && newOrdinal < starts[lo + 1] ? lo : -1;
    }

    /**
     * Best-first search over the hub's level-1 graph from {@code startPos}, width
     * {@link #CELL_PROBE_EF}. Every level-1 node it scores becomes a probe cell, best first; the
     * scan then takes cells in that order until {@link #CELL_BUDGET} codes, so the number of cells
     * probed adapts to their size.
     */
    private void selectCellsByBeam(Scratch scratch, HubMap.Scorer scorer, int startPos) {
        final int ef = CELL_PROBE_EF;
        final int stride = cellMap.degree[1];
        final int[] adj = cellMap.adjacency[1];
        if (scratch.beamPos == null) {
            scratch.beamPos = new int[ef];
            scratch.beamScore = new float[ef];
            scratch.beamExpanded = new boolean[ef];
            scratch.l1Visited = new int[visitedCapacity(ef, stride)];
            scratch.scoredKeys = new long[beamScoredCapacity(ef, stride)];
        }
        final int[] visited = scratch.l1Visited;
        Arrays.fill(visited, -1);
        final int mask = visited.length - 1;
        int[] pos = scratch.beamPos;
        float[] sc = scratch.beamScore;
        boolean[] expanded = scratch.beamExpanded;
        long[] scored = scratch.scoredKeys;   // (sortable score << 32 | position) for every scored cell
        int scoredCount = 0;
        int size = 0;
        float s0 = scorer.score(1, startPos);
        pos[0] = startPos; sc[0] = s0; expanded[0] = false; size = 1;
        visitedAdd(visited, mask, startPos);
        scored[scoredCount++] = packScored(s0, startPos);
        // replaced entries re-enter as unexpanded, so bound the expansions (2 * ef is ample: the
        // beam has converged long before) and size every array for that bound
        for (int expansions = 0; expansions < 2 * ef; expansions++) {
            int bi = -1;
            for (int i = 0; i < size; i++) {
                if (!expanded[i] && (bi < 0 || sc[i] > sc[bi])) bi = i;
            }
            if (bi < 0) break;
            expanded[bi] = true;
            int p = pos[bi];
            for (int j = 0; j < stride; j++) {
                int nb = adj[p * stride + j];
                if (nb < 0) break;
                int np = cellMap.position(1, nb);
                if (np < 0 || !visitedAdd(visited, mask, np)) continue;
                float score = scorer.score(1, np);
                scored[scoredCount++] = packScored(score, np);
                if (size < ef) {
                    pos[size] = np; sc[size] = score; expanded[size] = false; size++;
                } else {
                    int wi = 0;
                    for (int i = 1; i < size; i++) if (sc[i] < sc[wi]) wi = i;
                    if (score > sc[wi]) {
                        pos[wi] = np; sc[wi] = score; expanded[wi] = false;
                    }
                }
            }
        }
        Arrays.sort(scored, 0, scoredCount);
        int n = 0;
        for (int i = scoredCount - 1; i >= 0; i--) {   // best first
            int cell = cellWalkPosition[cellMap.nodes[1][(int) scored[i]]];
            if (cell >= 0 && cell < cellCount) scratch.probeCells[n++] = cell;
        }
        scratch.probeCount = n;
    }

    /** Packs a float score into a sortable long key with the position in the low 32 bits. */
    private static long packScored(float score, int position) {
        int bits = Float.floatToIntBits(score);
        int sortable = bits ^ ((bits >> 31) & 0x7fffffff);   // order-preserving int for floats
        return ((long) sortable << 32) | (position & 0xFFFFFFFFL);
    }

    /** Most nodes a beam of width {@code ef} can score: {@code 2 * ef} expansions of {@code stride} plus the seed. */
    private static int beamScoredCapacity(int ef, int stride) {
        return 2 * ef * stride + 1;
    }

    /** Power-of-two capacity for a visited set holding a beam's scored nodes at low load. */
    private static int visitedCapacity(int ef, int stride) {
        return Integer.highestOneBit(Math.max(64, 4 * beamScoredCapacity(ef, stride)) - 1) << 1;
    }


    /** Best-scoring position at {@code level} found by a best-first beam of width {@code ef} from {@code startPos}. */
    private static int bestByBeam(HubMap map, HubMap.Scorer scorer, int level, int startPos, int ef, int[] visited) {
        final int landingPos = startPos;
        int stride = map.degree[level];
        int[] adj = map.adjacency[level];
        int[] pos = new int[ef];
        float[] sc = new float[ef];
        boolean[] expanded = new boolean[ef];
        Arrays.fill(visited, -1);
        final int mask = visited.length - 1;
        int size = 0;
        pos[0] = landingPos; sc[0] = scorer.score(level, landingPos); size = 1;
        visitedAdd(visited, mask, landingPos);
        int bestPos = landingPos; float bestScore = sc[0];
        int[] nbPos = new int[stride];
        float[] nbScore = new float[stride];
        for (int expansions = 0; expansions < 2 * ef; expansions++) {
            int bi = -1;
            for (int i = 0; i < size; i++) if (!expanded[i] && (bi < 0 || sc[i] > sc[bi])) bi = i;
            if (bi < 0) break;
            expanded[bi] = true;
            int p = pos[bi];
            // phase 1: collect unvisited neighbours and score them in a tight loop (misses overlap)
            int cnt = 0;
            for (int j = 0; j < stride; j++) {
                int nb = adj[p * stride + j];
                if (nb < 0) break;
                int np = map.position(level, nb);
                if (np < 0 || !visitedAdd(visited, mask, np)) continue;
                nbPos[cnt++] = np;
            }
            for (int j = 0; j < cnt; j++) {
                nbScore[j] = scorer.score(level, nbPos[j]);
            }
            // phase 2: beam bookkeeping
            for (int j = 0; j < cnt; j++) {
                int np = nbPos[j];
                float score = nbScore[j];
                if (score > bestScore) { bestScore = score; bestPos = np; }
                if (size < ef) {
                    pos[size] = np; sc[size] = score; expanded[size] = false; size++;
                } else {
                    int wi = 0;
                    for (int i = 1; i < size; i++) if (sc[i] < sc[wi]) wi = i;
                    if (score > sc[wi]) { pos[wi] = np; sc[wi] = score; expanded[wi] = false; }
                }
            }
        }
        return bestPos;
    }

    /** Adds {@code key} to the open-addressing set (power-of-two length, -1 = empty); false if present. */
    private static boolean visitedAdd(int[] set, int mask, int key) {
        int i = (key * 0x9E3779B1) >>> 1 & mask;
        for (int probes = 0; probes <= mask; probes++) {
            int v = set[i];
            if (v == key) return false;
            if (v == -1) { set[i] = key; return true; }
            i = (i + 1) & mask;
        }
        return false;   // full (cannot happen with the sized capacities); treat as visited
    }

    /**
     * Recomputes the exact similarity between the base vector and a node's vector, used to
     * rerank approximate PQ-scored search results.
     */
    private float rescore(OnDiskGraphIndex.View view,
                         int node,
                         VectorFloat<?> base,
                         VectorFloat<?> tmp) {
        view.getVectorInto(node, tmp, 0);
        return similarityFunction.compare(base, tmp);
    }

    /**
     * Executes batches with controlled concurrency using a sliding window approach. Prevents
     * overwhelming memory by limiting the number of in-flight tasks while maintaining high
     * throughput via the completion service.
     */
    private <T> void runBatchesWithBackpressure(
            List<BatchSpec> batches,
            ExecutorCompletionService<List<T>> ecs,
            java.util.function.Consumer<BatchSpec> submitOne,
            java.util.function.Consumer<List<T>> onComplete
    ) throws InterruptedException, ExecutionException {

        final int total = batches.size();
        int nextToSubmit = 0;
        int inFlight = 0;

        // initial window
        while (inFlight < taskWindowSize && nextToSubmit < total) {
            submitOne.accept(batches.get(nextToSubmit++));
            inFlight++;
        }

        int completed = 0;
        while (completed < total) {
            List<T> results = ecs.take().get();
            onComplete.accept(results);

            completed++;
            inFlight--;

            if (nextToSubmit < total) {
                submitOne.accept(batches.get(nextToSubmit++));
                inFlight++;
            }
            if (completed % 10 == 0) {
                log.debug("Compaction I/O progress: {}/{} batches written to disk", completed, total);
            }
        }
    }

    /**
     * Appends search results from a NodeQueue to the candidate arrays, returning the updated
     * candidate count.
     */
    private int appendApproximateResults(NodeQueue queue,
                                         int sourceIdx,
                                         Scratch scratch,
                                         int candSize) {
        final int ss = sourceIdx;
        final int[] idx = new int[] { candSize };

        queue.foreach((nb, score) -> {
            scratch.candSrc[idx[0]] = ss;
            scratch.candNode[idx[0]] = nb;
            scratch.candScore[idx[0]] = score;
            idx[0]++;
        });

        return idx[0];
    }

    /**
     * Computes layer metadata for the compacted graph by counting live nodes at each level
     * across all source indexes.
     */
    private List<CommonHeader.LayerInfo> computeLayerInfoFromSources() {
        int maxLevel = sources.stream().mapToInt(OnDiskGraphIndex::getMaxLevel).max().orElse(0);
        List<CommonHeader.LayerInfo> layerInfo = new ArrayList<>(maxLevel + 1);
        for (int level = 0; level <= maxLevel; level++) {
            int count = 0;
            for (int s = 0; s < sources.size(); s++) {
                if (level > sources.get(s).getMaxLevel()) continue;
                if (level == 0) {
                    // Every live node is present at level 0 (HNSW base layer invariant),
                    // so count directly from the in-memory bitset instead of scanning node
                    // records on disk (which touches gigabytes of source data on a cold cache).
                    count += liveNodes.get(s).cardinality();
                } else {
                    NodesIterator it = sources.get(s).getNodes(level);
                    FixedBitSet alive = liveNodes.get(s);
                    while (it.hasNext()) {
                        int node = it.next();
                        if (alive.get(node)) count++;
                    }
                }
            }
            layerInfo.add(new CommonHeader.LayerInfo(count, maxDegrees.get(level)));
        }
        return layerInfo;
    }

    /**
     * Creates a score provider for searching across different source indexes. Uses approximate
     * PQ-based scoring if compressedPrecision is enabled, otherwise uses exact scoring.
     */
    private SearchScoreProvider buildCrossSourceScoreProvider(boolean compressedPrecision,
                                                              OnDiskGraphIndex searchSource,
                                                              int searchSourceIdx,
                                                              OnDiskGraphIndex.View searchView,
                                                              VectorFloat<?> baseVec,
                                                              VectorFloat<?> tmpVec,
                                                              VectorSimilarityFunction similarityFunction,
                                                              ProductQuantization mergedPq) {
        if (compressedPrecision && searchSource.getFeatures().containsKey(FeatureId.FUSED_PQ)) {
            ScoreFunction.ExactScoreFunction reranker =
                node2 -> {
                    searchView.getVectorInto(node2, tmpVec, 0);
                    return similarityFunction.compare(baseVec, tmpVec);
                };
            var asf = ((FusedPQ) searchSource.getFeatures().get(FeatureId.FUSED_PQ)).approximateScoreFunctionFor(baseVec, similarityFunction, searchView, reranker);

            return new DefaultSearchScoreProvider(asf);
        }
        // Sidecar parity: no fused feature, but the merged code cache exists (built by the
        // sidecar strategy's pre-encode pass). Score approximately from the cache via a per-query
        // PQ lookup table, rerank exactly from inline vectors — same economics as fused mode
        // without touching the fused code path.
        PreEncodedCodeCache sidecarCache = activeSidecarStrategy.getCodeCache();
        if (compressedPrecision && sidecarCache != null && mergedPq != null
                && lutSupported(similarityFunction)) {
            // Single-arg provider, mirroring the fused branch: candidates are exact-rescored
            // downstream (compressedPrecision gates), and an in-search reranker would consume
            // the approximate-results queue those collection paths read.
            var asf = cacheLutScoreFunction(baseVec, similarityFunction, mergedPq, sidecarCache,
                    remappers.get(searchSourceIdx), liveNodes.get(searchSourceIdx));
            return new DefaultSearchScoreProvider(asf);
        }

        var sf = new ScoreFunction.ExactScoreFunction() {
            @Override
            public float similarityTo(int node2) {
                searchView.getVectorInto(node2, tmpVec, 0);
                return similarityFunction.compare(baseVec, tmpVec);
            }
        };
        return new DefaultSearchScoreProvider(sf);
    }

    private static boolean lutSupported(VectorSimilarityFunction f) {
        return f == VectorSimilarityFunction.DOT_PRODUCT || f == VectorSimilarityFunction.EUCLIDEAN;
    }

    /**
     * Per-query PQ lookup-table scorer over the merged pre-encode cache. Builds the standard
     * asymmetric-distance table (one sub-score per subspace centroid), then scores a source-graph
     * node by translating its ordinal through the remapper and summing table entries for its
     * cached code. Similarity conversion mirrors {@link VectorSimilarityFunction}:
     * DOT_PRODUCT (1+dot)/2, EUCLIDEAN 1/(1+d^2).
     */
    private ScoreFunction.ApproximateScoreFunction cacheLutScoreFunction(VectorFloat<?> query,
                                                                         VectorSimilarityFunction f,
                                                                         ProductQuantization pq,
                                                                         PreEncodedCodeCache cache,
                                                                         OrdinalMapper mapper,
                                                                         FixedBitSet alive) {
        final int msub = pq.getSubspaceCount();
        final int clusters = pq.getClusterCount();
        // Center-adjusted PQ encodes (v - globalCentroid): a decoded vector is
        // centroid + concat(subcentroids). For EUCLIDEAN, center the query so per-subspace
        // distances compose; for DOT, score subspaces against the raw query and add the
        // constant dot(query, centroid) term.
        final VectorFloat<?> center = pq.getGlobalCentroid();
        float dotConstant = 0;
        VectorFloat<?> q = query;
        if (center != null) {
            if (f == VectorSimilarityFunction.EUCLIDEAN) {
                VectorFloat<?> centered = vectorTypeSupport.createFloatVector(dimension);
                for (int i = 0; i < dimension; i++) {
                    centered.set(i, query.get(i) - center.get(i));
                }
                q = centered;
            } else { // DOT_PRODUCT
                for (int i = 0; i < dimension; i++) {
                    dotConstant += query.get(i) * center.get(i);
                }
            }
        }
        final float[] lut = new float[msub * clusters];
        int queryOffset = 0;
        for (int m = 0; m < msub; m++) {
            int sz = pq.getSubvectorSize(m);
            var cb = pq.getCodebookVector(m);
            for (int c = 0; c < clusters; c++) {
                float acc = 0;
                int base = c * sz;
                if (f == VectorSimilarityFunction.DOT_PRODUCT) {
                    for (int i = 0; i < sz; i++) {
                        acc += q.get(queryOffset + i) * cb.get(base + i);
                    }
                } else { // EUCLIDEAN
                    for (int i = 0; i < sz; i++) {
                        float d = q.get(queryOffset + i) - cb.get(base + i);
                        acc += d * d;
                    }
                }
                lut[m * clusters + c] = acc;
            }
            queryOffset += sz;
        }
        final byte[] codeScratch = new byte[cache.codeSize()];
        final boolean dot = f == VectorSimilarityFunction.DOT_PRODUCT;
        final float constant = dotConstant;
        return node -> {
            // Dead nodes have no merged ordinal (and no cached code); the search may still
            // traverse them. Score them at the floor — they are excluded from results by the
            // alive filter, and candidates are exact-rescored before diversity regardless.
            if (!alive.get(node)) {
                return 0f;
            }
            int newOrd = mapper.oldToNew(node);
            cache.get(newOrd, codeScratch);
            float sum = 0;
            for (int m = 0; m < msub; m++) {
                sum += lut[m * clusters + (codeScratch[m] & 0xFF)];
            }
            return dot ? (1 + sum + constant) / 2 : 1 / (1 + sum);
        };
    }

    /**
     * Estimates the RAM usage of this compactor instance.
     * Accounts for data structures used during compaction including bitsets, remappers,
     * executor overhead, and per-thread scratch space.
     */
    @Override
    public long ramBytesUsed() {
        int OH = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        int REF = RamUsageEstimator.NUM_BYTES_OBJECT_REF;

        // Shallow size of this object (header + fields)
        // Current fields: sources, liveNodes, numLiveNodesPerSource, remappers, maxDegrees,
        //                dimension(int), maxOrdinal(int), numTotalNodes(int),
        //                executor, taskWindowSize(int), similarityFunction
        long size = OH + 8L * REF + Integer.BYTES * 4;

        // liveNodes: FixedBitSet per source
        if (liveNodes != null) {
            for (var entry : liveNodes) {
                size += entry.ramBytesUsed();
            }
        }

        // numLiveNodesPerSource: ArrayList of Integers
        size += OH + REF + (long) numLiveNodesPerSource.size() * (OH + Integer.BYTES);

        // remappers: each MapMapper holds an oldToNew HashMap and newToOld Int2IntHashMap.
        if (remappers != null) {
            for (var mapper : remappers) {
                // Object overhead + two maps with int key/value pairs
                // HashMap entry: ~32 bytes each; Int2IntHashMap: ~16 bytes per entry
                if (mapper instanceof OrdinalMapper.MapMapper) {
                    // rough estimate: the mapper stores two maps over all mapped ordinals
                    size += OH + (long) (maxOrdinal + 1) * 48;
                }
            }
        }

        // maxDegrees: small list of integers
        size += OH + REF + (long) maxDegrees.size() * (OH + Integer.BYTES);

        // Cross-link reverse-candidate buffer (present only while L0 is being compacted)
        if (reverseCandidates != null) {
            size += reverseCandidates.ramBytesUsed();
        }

        // executor: a shared pool (default) or caller-injected — not owned by the compactor, so it
        // contributes no pool allocation here. Scratch space still scales with its parallelism.
        int numThreads = taskWindowSize;

        // Scratch space: ThreadLocal instances (one per active thread)
        // Each Scratch contains:
        //   - candSrc, candNode, candScore arrays
        //   - SelectedVecCache (with its own arrays and vector copies)
        //   - tmpVec, baseVec (VectorFloat instances)
        //   - GraphSearcher array (one per source)
        //   - pqCode ByteSequence
        size += estimateScratchSpacePerThread() * numThreads;

        return size;
    }

    /**
     * Estimates the RAM usage of a single Scratch instance.
     */
    private long estimateScratchSpacePerThread() {
        int OH = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        int REF = RamUsageEstimator.NUM_BYTES_OBJECT_REF;

        // Calculate maxCandidateSize and maxDegree (same logic as in compactLevels)
        int maxUpperDegree = 0;
        for (int level = 1; level < maxDegrees.size(); level++) {
            maxUpperDegree = Math.max(maxUpperDegree, maxDegrees.get(level));
        }
        int baseSearchTopK = Math.max(MIN_SEARCH_TOP_K, ((maxDegrees.get(0) + sources.size() - 1) / sources.size()) * SEARCH_TOP_K_MULTIPLIER);
        int baseMaxCandidateSize = baseSearchTopK * (sources.size() - 1) + maxDegrees.get(0) + REVERSE_CANDIDATE_SLOTS;
        int upperMaxPerSourceTopK = maxUpperDegree == 0 ? 0 : Math.max(MIN_SEARCH_TOP_K, ((maxUpperDegree + sources.size() - 1) / sources.size()) * SEARCH_TOP_K_MULTIPLIER);
        int upperMaxCandidateSize = upperMaxPerSourceTopK * sources.size();
        int maxCandidateSize = Math.max(baseMaxCandidateSize, upperMaxCandidateSize);
        int scratchDegree = Math.max(maxDegrees.get(0), Math.max(1, maxUpperDegree));

        long scratchSize = OH + 6L * REF;

        // candSrc, candNode, candScore, candCodeOnly arrays
        scratchSize += (long) maxCandidateSize * Integer.BYTES; // candSrc
        scratchSize += (long) maxCandidateSize * Integer.BYTES; // candNode
        scratchSize += (long) maxCandidateSize * Float.BYTES;   // candScore
        scratchSize += maxCandidateSize;                        // candCodeOnly

        // SelectedVecCache
        scratchSize += OH + 5L * REF + Integer.BYTES; // SelectedVecCache object
        scratchSize += (long) scratchDegree * Integer.BYTES;  // sourceIdx array
        scratchSize += (long) scratchDegree * REF;            // views array
        scratchSize += (long) scratchDegree * Integer.BYTES;  // nodes array
        scratchSize += (long) scratchDegree * Float.BYTES;    // scores array
        scratchSize += (long) scratchDegree * REF;            // vecs array
        scratchSize += (long) scratchDegree * (OH + dimension * Float.BYTES); // VectorFloat instances

        // tmpVec and baseVec
        scratchSize += 2L * (OH + dimension * Float.BYTES);

        // GraphSearcher array (one per source)
        scratchSize += (long) sources.size() * REF;
        // Each GraphSearcher has internal state - rough estimate
        scratchSize += (long) sources.size() * (OH + 10L * REF);

        // Per-thread scratch ByteSequence holding one code's worth of bytes, for each fused
        // feature carried by the graph. Generalized over fused types so new quantizations
        // (e.g. FUSED_ASH) don't need an edit here.
        for (var feature : sources.get(0).getFeatures().values()) {
            if (feature instanceof FusedFeature) {
                scratchSize += OH + ((FusedFeature) feature).codeSize();
            }
        }

        return scratchSize;
    }

    /**
     * Encapsulates common parameters used throughout the compaction process.
     */
    private static final class CompactionParams {
        final boolean fusedPQEnabled;
        final boolean compressedPrecision;
        final int searchTopK;
        final int beamWidth;
        final ProductQuantization pq;

        CompactionParams(boolean fusedPQEnabled, boolean compressedPrecision,
                        int searchTopK, int beamWidth, ProductQuantization pq) {
            this.fusedPQEnabled = fusedPQEnabled;
            this.compressedPrecision = compressedPrecision;
            this.searchTopK = searchTopK;
            this.beamWidth = beamWidth;
            this.pq = pq;
        }
    }

    /**
     * Sorts an index array by descending score values using quicksort.
     */
    private static void sortOrderByScoreDesc(int[] order, float[] score, int size) {
        quicksort(order, score, 0, size - 1);
    }

    /**
     * Tail-recursive quicksort implementation for sorting by score in descending order.
     */
    private static void quicksort(int[] order, float[] score, int lo, int hi) {
        while (lo < hi) {
            int p = partition(order, score, lo, hi);
            // recurse smaller side first (limits stack)
            if (p - lo < hi - p) {
                quicksort(order, score, lo, p - 1);
                lo = p + 1;
            } else {
                quicksort(order, score, p + 1, hi);
                hi = p - 1;
            }
        }
    }

    /**
     * Partitions the order array for quicksort using descending score comparison.
     */
    private static int partition(int[] order, float[] score, int lo, int hi) {
        float pivot = score[order[hi]];
        int i = lo;
        for (int j = lo; j < hi; j++) {
            if (score[order[j]] > pivot) { // DESC
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
                i++;
            }
        }
        int t = order[i];
        order[i] = order[hi];
        order[hi] = t;
        return i;
    }

    static final class WriteResult {
        final int newOrdinal;
        final long fileOffset;
        final ByteBuffer data;

        WriteResult(int newOrdinal, long fileOffset, ByteBuffer data) {
            this.newOrdinal = newOrdinal;
            this.fileOffset = fileOffset;
            this.data = data;
        }
    };

    private static final class UpperLayerWriteResult {
        final int ordinal;
        final int[] neighbors;
        final ByteSequence<?> pqCode;

        UpperLayerWriteResult(int ordinal, SelectedVecCache cache, ByteSequence<?> pqCode) {
            this.ordinal = ordinal;
            this.neighbors = Arrays.copyOf(cache.nodes, cache.size);
            this.pqCode = pqCode == null ? null : pqCode.copy();
        }
    };

    /** Array-backed OrdinalMapper for one source of the compactor-assigned similarity mapping. */
    private static final class ArrayOrdinalMapper implements OrdinalMapper {
        private final int src;
        private final int[] oldToNew;      // per-source, indexed by old ordinal
        private final int[] newToOldAll;   // global, indexed by new ordinal
        private final int[] newToSrcAll;   // global, indexed by new ordinal
        private final int maxOrdinal;

        ArrayOrdinalMapper(int src, int[] oldToNew, int[] newToOldAll, int[] newToSrcAll, int maxOrdinal) {
            this.src = src;
            this.oldToNew = oldToNew;
            this.newToOldAll = newToOldAll;
            this.newToSrcAll = newToSrcAll;
            this.maxOrdinal = maxOrdinal;
        }

        @Override
        public int maxOrdinal() {
            return maxOrdinal;
        }

        @Override
        public int oldToNew(int oldOrdinal) {
            return oldToNew[oldOrdinal];
        }

        @Override
        public int newToOld(int newOrdinal) {
            if (newOrdinal < 0 || newOrdinal >= newToSrcAll.length || newToSrcAll[newOrdinal] != src) {
                return OMITTED;
            }
            return newToOldAll[newOrdinal];
        }
    }

    /**
     * Resident copy of the largest source's upper layers (levels 1 and above): adjacency and PQ
     * codes of every upper-layer node, so region keys can be assigned by a code-scored greedy
     * descent without touching the source file per query.
     */
    private static final class HubMap {
        private final ProductQuantization pq;
        private final VectorSimilarityFunction lutFunction;
        final int subspaceCount;
        final int clusterCount;
        final boolean euclidean;
        final boolean cosine;
        // COSINE: squared norm of the decoded vector is additive over subspaces too. magnitudes[m*K+c]
        // = |centroid|^2 (+ 2 dot(globalCentroid_m, centroid) when center-adjusted); centerNorm2 is the
        // constant part. mag8 is the same table quantized for the scan kernel (per-subspace minimum
        // shift, one shared scale), with mag8Scale/mag8Offset undoing the quantization of a sum.
        final VectorFloat<?> magnitudes;
        final float centerNorm2;
        final byte[] mag8;
        final float mag8Scale, mag8Offset;
        final int topLevel;
        int entryNode;
        // per level (index 0 unused): node ids, adjacency with stride degree[level] (-1 padded),
        // and PQ codes with stride subspaceCount, all indexed by the node's position in nodes[level]
        final int[] degree;
        final int[][] nodes;
        final int[][] adjacency;
        final ByteSequence<?>[] codes;
        // node id -> position: a flat array for level 1 (the bulk of the upper layers), maps above it
        int[] level1Position;
        final Map<Integer, Integer>[] upperPosition;
        // diagnostics for the summary log line
        final LongAdder descents = new LongAdder();
        final LongAdder moves = new LongAdder();

        @SuppressWarnings("unchecked")
        HubMap(OnDiskGraphIndex hub, ProductQuantization pq, VectorSimilarityFunction similarityFunction) {
            this.pq = pq;
            // squared distance for EUCLIDEAN, dot product otherwise; COSINE divides the dot product by
            // the decoded norm from the magnitude table below
            this.lutFunction = similarityFunction == VectorSimilarityFunction.EUCLIDEAN
                    ? VectorSimilarityFunction.EUCLIDEAN : VectorSimilarityFunction.DOT_PRODUCT;
            this.subspaceCount = pq.getSubspaceCount();
            this.clusterCount = pq.getClusterCount();
            this.euclidean = lutFunction == VectorSimilarityFunction.EUCLIDEAN;
            this.cosine = similarityFunction == VectorSimilarityFunction.COSINE;
            if (cosine) {
                magnitudes = vectorTypeSupport.createFloatVector(subspaceCount * clusterCount);
                VectorFloat<?> center = pq.getGlobalCentroid();
                int offset = 0;
                for (int m = 0; m < subspaceCount; m++) {
                    int size = pq.getSubvectorSize(m);
                    VectorFloat<?> cb = pq.getCodebookVector(m);
                    VectorUtil.calculatePartialSelfMagnitudes(cb, m, size, clusterCount, magnitudes);
                    if (center != null) {
                        for (int c = 0; c < clusterCount; c++) {
                            int i = m * clusterCount + c;
                            magnitudes.set(i, magnitudes.get(i) + 2 * VectorUtil.dotProduct(cb, c * size, center, offset, size));
                        }
                    }
                    offset += size;
                }
                centerNorm2 = center == null ? 0f : VectorUtil.dotProduct(center, center);
                // quantize once: sum of quantized entries q_m relates to the float sum as
                // sum_m (q_m / scale + min_m)  =>  float = N / scale + offset
                mag8 = new byte[subspaceCount * clusterCount];
                float maxRange = 0, offsetSum = 0;
                float[] mins = new float[subspaceCount];
                for (int m = 0; m < subspaceCount; m++) {
                    float mn = Float.POSITIVE_INFINITY, mx = Float.NEGATIVE_INFINITY;
                    for (int c = 0; c < clusterCount; c++) {
                        float v = magnitudes.get(m * clusterCount + c);
                        mn = Math.min(mn, v);
                        mx = Math.max(mx, v);
                    }
                    mins[m] = mn;
                    offsetSum += mn;
                    maxRange = Math.max(maxRange, mx - mn);
                }
                float scale = maxRange > 0 ? 255f / maxRange : 0f;
                for (int m = 0; m < subspaceCount; m++) {
                    float off = 0.5f - mins[m] * scale;
                    for (int c = 0; c < clusterCount; c++) {
                        int q = (int) (magnitudes.get(m * clusterCount + c) * scale + off);
                        mag8[m * clusterCount + c] = (byte) (q > 255 ? 255 : q);
                    }
                }
                mag8Scale = scale;
                mag8Offset = offsetSum;
            } else {
                magnitudes = null;
                centerNorm2 = 0f;
                mag8 = null;
                mag8Scale = 0f;
                mag8Offset = 0f;
            }
            this.topLevel = hub.getMaxLevel();
            this.degree = new int[topLevel + 1];
            this.nodes = new int[topLevel + 1][];
            this.adjacency = new int[topLevel + 1][];
            this.codes = new ByteSequence<?>[topLevel + 1];
            this.upperPosition = new Map[topLevel + 1];
        }

        /** Per-thread query state: the query's partial-sum table over the codebooks. */
        final class Scorer {
            private final VectorFloat<?> partialSums = vectorTypeSupport.createFloatVector(subspaceCount * clusterCount);
            private float[] lutArray;   // flat copy of partialSums for the code scan kernel (built on demand)
            private boolean lutArrayValid;

            private byte[] lut8;
            private boolean lut8Valid;
            float lut8Scale, lut8Offset;   // float sum = D / lut8Scale + lut8Offset (D = sum of table bytes)
            private final float[] lut8Mins = new float[subspaceCount];

            /**
             * 8-bit table for the scan kernel: per subspace the values are shifted
             * by their minimum (a per-query constant) and all subspaces share one scale, so sums of
             * table entries rank codes like sums of the float table. Higher is better; for EUCLIDEAN
             * the distances are negated before quantization.
             */
            byte[] lut8() {
                if (!lut8Valid) {
                    float[] f = lut();
                    if (lut8 == null) lut8 = new byte[subspaceCount * clusterCount];
                    boolean neg = lutFunction == VectorSimilarityFunction.EUCLIDEAN;
                    float maxRange = 0;
                    final float sign = neg ? -1f : 1f;
                    for (int m = 0; m < subspaceCount; m++) {
                        float mn = Float.POSITIVE_INFINITY, mx = Float.NEGATIVE_INFINITY;
                        int base = m * clusterCount;
                        for (int c = 0; c < clusterCount; c++) {
                            float v = sign * f[base + c];
                            mn = Math.min(mn, v);
                            mx = Math.max(mx, v);
                        }
                        lut8Mins[m] = mn;
                        maxRange = Math.max(maxRange, mx - mn);
                    }
                    final float scale = maxRange > 0 ? 255f / maxRange : 0f;
                    float offsetSum = 0;
                    for (int m = 0; m < subspaceCount; m++) offsetSum += lut8Mins[m];
                    lut8Scale = scale;
                    lut8Offset = offsetSum;
                    for (int m = 0; m < subspaceCount; m++) {
                        int base = m * clusterCount;
                        float off = 0.5f - lut8Mins[m] * scale;
                        for (int c = 0; c < clusterCount; c++) {
                            int q = (int) (sign * f[base + c] * scale + off);
                            lut8[base + c] = (byte) (q > 255 ? 255 : q);   // q >= 0 by construction
                        }
                    }
                    lut8Valid = true;
                }
                return lut8;
            }

            /** The current query's lookup table as a flat float[] ({@code m * clusterCount + c}). */
            float[] lut() {
                if (!lutArrayValid) {
                    float[] backing = VectorUtil.backingArray(partialSums);
                    if (backing != null) {
                        lutArray = backing;                       // no copy: heap-backed under both providers
                    } else {
                        if (lutArray == null) lutArray = new float[subspaceCount * clusterCount];
                        for (int i = 0; i < lutArray.length; i++) lutArray[i] = partialSums.get(i);
                    }
                    lutArrayValid = true;
                }
                return lutArray;
            }

            void setQuery(VectorFloat<?> query) {
                lutArrayValid = false;
                lut8Valid = false;
                // A center-adjusted PQ encodes (v - globalCentroid). For EUCLIDEAN the query must be
                // centered so per-subspace distances compose; for the dot-product ranking the raw
                // query is correct (dot(q, centroid) is a per-query constant), and centering it
                // would make the partial sums code-dependent.
                VectorFloat<?> center = pq.getGlobalCentroid();
                VectorFloat<?> q = center != null && lutFunction == VectorSimilarityFunction.EUCLIDEAN
                        ? VectorUtil.sub(query, center) : query;
                int offset = 0;
                for (int m = 0; m < subspaceCount; m++) {
                    int size = pq.getSubvectorSize(m);
                    VectorUtil.calculatePartialSums(pq.getCodebookVector(m), m, size, clusterCount, q, offset, lutFunction, partialSums);
                    offset += size;
                }
            }

                    /** Higher is closer. */
            float score(int level, int position) {
                int off = position * subspaceCount;
                float sum = VectorUtil.assembleAndSum(partialSums, clusterCount, codes[level], off, subspaceCount);
                if (cosine) {
                    float n2 = VectorUtil.assembleAndSum(magnitudes, clusterCount, codes[level], off, subspaceCount) + centerNorm2;
                    return sum / (float) Math.sqrt(Math.max(1e-12f, n2));   // query norm is a per-query constant
                }
                return lutFunction == VectorSimilarityFunction.EUCLIDEAN ? -sum : sum;
            }


                }

        Scorer scorer() {
            return new Scorer();
        }

        int position(int level, int node) {
            if (level == 1) {
                return node >= 0 && node < level1Position.length ? level1Position[node] : -1;
            }
            Integer p = upperPosition[level].get(node);
            return p == null ? -1 : p;
        }

        /**
         * Greedy descent from the entry node down to level 1 under the scorer's current query.
         *
         * @return the level-1 node the descent lands on
         */
        int descend(Scorer scorer) {
            int current = entryNode;
            long moved = 0;
            for (int level = topLevel; level >= 1; level--) {
                int position = position(level, current);
                if (position < 0) {
                    break;
                }
                float currentScore = scorer.score(level, position);
                int stride = degree[level];
                int[] adj = adjacency[level];
                boolean improved = true;
                while (improved) {
                    improved = false;
                    for (int j = 0; j < stride; j++) {
                        int neighbor = adj[position * stride + j];
                        if (neighbor < 0) {
                            break;
                        }
                        int neighborPosition = position(level, neighbor);
                        if (neighborPosition < 0) {
                            continue;
                        }
                        float score = scorer.score(level, neighborPosition);
                        if (score > currentScore) {
                            currentScore = score;
                            current = neighbor;
                            position = neighborPosition;
                            improved = true;
                            moved++;
                        }
                    }
                }
            }
            descents.increment();
            moves.add(moved);
            return current;
        }

        /**
         * Breadth-first walk over the level-1 graph, restarting at the lowest unvisited position
         * when a component is exhausted.
         *
         * @return walk position per node id (Integer.MAX_VALUE for nodes not on level 1)
         */
        int[] walkPositions() {
            int[] l1 = nodes[1];
            int n = l1.length;
            int stride = degree[1];
            int[] adj = adjacency[1];
            int[] positionOf = new int[level1Position.length];
            Arrays.fill(positionOf, Integer.MAX_VALUE);
            boolean[] seen = new boolean[n];
            int[] queue = new int[n];
            int head = 0, tail = 0, emitted = 0, nextUnseen = 0;
            while (emitted < n) {
                if (head == tail) {
                    while (nextUnseen < n && seen[nextUnseen]) {
                        nextUnseen++;
                    }
                    if (nextUnseen >= n) {
                        break;
                    }
                    seen[nextUnseen] = true;
                    queue[tail++] = nextUnseen;
                }
                int x = queue[head++];
                positionOf[l1[x]] = emitted++;
                for (int j = 0; j < stride; j++) {
                    int neighbor = adj[x * stride + j];
                    if (neighbor < 0) {
                        break;
                    }
                    int y = position(1, neighbor);
                    if (y >= 0 && !seen[y]) {
                        seen[y] = true;
                        queue[tail++] = y;
                    }
                }
            }
            return positionOf;
        }
    }

    /** Loads the largest source's upper layers into a {@link HubMap}, encoding every upper-layer node with {@code pq}. */
    private HubMap buildHubMap(OnDiskGraphIndex hub, ProductQuantization pq) {
        long t0 = System.nanoTime();
        HubMap h = new HubMap(hub, pq, similarityFunction);
        try (var view = hub.getView()) {
            h.entryNode = view.entryNode().node;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        int subspaceCount = pq.getSubspaceCount();
        for (int level = 1; level <= h.topLevel; level++) {
            NodesIterator it = hub.getNodes(level);
            int[] nodes = new int[it.size()];
            int n = 0;
            while (it.hasNext()) {
                nodes[n++] = it.next();
            }
            h.nodes[level] = nodes;
            if (level == 1) {
                h.level1Position = new int[hub.getIdUpperBound()];
                Arrays.fill(h.level1Position, -1);
                for (int i = 0; i < n; i++) {
                    h.level1Position[nodes[i]] = i;
                }
            } else {
                h.upperPosition[level] = new HashMap<>(n * 2);
                for (int i = 0; i < n; i++) {
                    h.upperPosition[level].put(nodes[i], i);
                }
            }
            int stride = hub.getDegree(level);
            h.degree[level] = stride;
            int[] adj = new int[n * stride];
            Arrays.fill(adj, -1);
            ByteSequence<?> codes = vectorTypeSupport.createByteSequence(n * subspaceCount);
            int lvl = level;
            int chunks = Math.max(1, Math.min(1024, n / 4096));
            int chunk = (n + chunks - 1) / chunks;
            List<Callable<Void>> tasks = new ArrayList<>();
            for (int c = 0; c < chunks; c++) {
                int lo = c * chunk;
                int hi = Math.min(n, lo + chunk);
                if (lo >= hi) {
                    continue;
                }
                tasks.add(() -> {
                    VectorFloat<?> vec = vectorTypeSupport.createFloatVector(dimension);
                    ByteSequence<?> code = vectorTypeSupport.createByteSequence(subspaceCount);
                    try (var view = hub.getView()) {
                        for (int i = lo; i < hi; i++) {
                            var neighbors = view.getNeighborsIterator(lvl, nodes[i]);
                            int j = 0;
                            while (neighbors.hasNext() && j < stride) {
                                adj[i * stride + j++] = neighbors.nextInt();
                            }
                            view.getVectorInto(nodes[i], vec, 0);
                            pq.encodeTo(vec, code);
                            codes.copyFrom(code, 0, i * subspaceCount, subspaceCount);
                        }
                    }
                    return null;
                });
            }
            joinAll(tasks);
            h.adjacency[level] = adj;
            h.codes[level] = codes;
        }
        log.info("Region ordinals: resident hub map built (levels 1..{}, {} level-1 nodes) in {} ms",
                 h.topLevel, h.nodes[1].length, (System.nanoTime() - t0) / 1_000_000);
        return h;
    }

    /**
     * Builds the compactor-assigned ordinal mapping. Live nodes are numbered source by source in
     * ascending-size processing order, and within a source by a locality key: the walk position
     * (breadth-first over the largest source's level-1 graph) of the level-1 node a code-scored
     * greedy descent lands on, or the node's PQ-code prefix when the largest source has no
     * hierarchy. Batches
     * processed in the same order therefore write records sequentially, and consecutive nodes of
     * every source explore the same region of every target. Dead nodes are numbered after all
     * live nodes, preserving a total bijection.
     */
    private List<OrdinalMapper> buildRegionOrdinalMappers(ProductQuantization pq) {
        long t0 = System.nanoTime();
        int numSources = sources.size();
        long totalOrdinals = 0;
        for (OnDiskGraphIndex src : sources) {
            totalOrdinals += src.size(0);
        }
        if (totalOrdinals > Integer.MAX_VALUE) {
            throw new IllegalStateException("merged ordinal space exceeds int range: " + totalOrdinals);
        }

        // ascending-size processing order, matching setupCrossLink
        Integer[] order = new Integer[numSources];
        for (int i = 0; i < numSources; i++) order[i] = i;
        Arrays.sort(order, Comparator
                .comparingInt((Integer i) -> numLiveNodesPerSource.get(i))
                .thenComparingInt(i -> i));

        int[] newToOldAll = new int[(int) totalOrdinals];
        int[] newToSrcAll = new int[(int) totalOrdinals];
        int[][] oldToNewPerSource = new int[numSources][];
        int next = 0;
        // region mode: the largest source is the hub; every live node's key is the walk position
        // of the level-1 hub node a code-scored greedy descent lands on
        OnDiskGraphIndex hubSource = sources.get(order[numSources - 1]);
        boolean regionMode = hubSource.getMaxLevel() >= 1;
        HubMap hubMap = null;
        int[] walkPosition = null;
        if (regionMode) {
            hubMap = buildHubMap(hubSource, pq);
            walkPosition = hubMap.walkPositions();
            log.info("Region ordinals: hub source {} (maxLevel {}), {} level-1 nodes walked",
                     order[numSources - 1], hubSource.getMaxLevel(), hubMap.nodes[1].length);
        } else {
            log.info("Region ordinals: the largest source has no hierarchy; falling back to PQ-prefix ordinals");
        }
        HubMap hubMapRef = hubMap;
        int[] walkPositionRef = walkPosition;
        int prefixBytes = Math.min(4, pq.getSubspaceCount());
        if (regionMode && SymmetricCodeSimilarity.supports(similarityFunction)) {
            // dot product, Euclidean and cosine (cosine adds a second additive pass over decoded norms)
            cellMap = hubMap;
            cellWalkPosition = walkPosition;
            cellCount = hubMap.nodes[1].length;
            cellStart = new int[numSources][];
            cellToPosition = new int[cellCount];
            for (int pos = 0; pos < cellCount; pos++) {
                int w = walkPosition[hubMap.nodes[1][pos]];
                if (w >= 0 && w < cellCount) cellToPosition[w] = pos;
            }
        }
        for (int oi = 0; oi < numSources; oi++) {
            int s = order[oi];
            OnDiskGraphIndex source = sources.get(s);
            int size = source.size(0);
            FixedBitSet alive = liveNodes.get(s);
            int[] oldToNew = new int[size];
            oldToNewPerSource[s] = oldToNew;

            // one streaming pass: locality key per live node, packed with the ordinal
            int liveCount = numLiveNodesPerSource.get(s);
            long[] keyed = new long[liveCount];
            AtomicInteger fill = new AtomicInteger();
            int window = 1 << 18;
            List<Callable<Void>> tasks = new ArrayList<>();
            for (int from = 0; from < size; from += window) {
                final int lo = from;
                final int hi = Math.min(size, from + window);
                tasks.add(() -> {
                    source.prefetchL0Records(lo, hi - 1);
                    VectorFloat<?> vec = vectorTypeSupport.createFloatVector(dimension);
                    ByteSequence<?> code = vectorTypeSupport.createByteSequence(pq.getSubspaceCount());
                    HubMap.Scorer scorer = regionMode ? hubMapRef.scorer() : null;
                    int[] beamVisited = regionMode ? new int[visitedCapacity(CELL_ASSIGN_EF, hubMapRef.degree[1])] : null;
                    try (var view = (OnDiskGraphIndex.View) source.getView()) {
                        for (int node = lo; node < hi; node++) {
                            if (!alive.get(node)) continue;
                            view.getVectorInto(node, vec, 0);
                            long key;
                            if (regionMode) {
                                scorer.setQuery(vec);
                                // cell = best level-1 node found by a small beam from the descent's landing
                                int landing = hubMapRef.descend(scorer);
                                int lp = hubMapRef.position(1, landing);
                                if (lp >= 0) {
                                    landing = hubMapRef.nodes[1][bestByBeam(hubMapRef, scorer, 1, lp, CELL_ASSIGN_EF, beamVisited)];
                                }
                                int pos = landing >= 0 && landing < walkPositionRef.length ? walkPositionRef[landing] : Integer.MAX_VALUE;
                                key = pos & 0xFFFFFFFFL;
                            } else {
                                pq.encodeTo(vec, code);
                                key = 0;
                                for (int b = 0; b < prefixBytes; b++) {
                                    key = (key << 8) | (code.get(b) & 0xFFL);
                                }
                            }
                            keyed[fill.getAndIncrement()] = (key << 32) | (node & 0xFFFFFFFFL);
                        }
                    }
                    return null;
                });
            }
            joinAll(tasks);
            Arrays.parallelSort(keyed, 0, fill.get());

            int[] starts = cellStart != null ? new int[cellCount + 2] : null;
            int cell = 0;
            for (int k = 0; k < fill.get(); k++) {
                if (starts != null) {
                    long key = keyed[k] >>> 32;
                    int c = key >= cellCount ? cellCount : (int) key;
                    while (cell <= c) {
                        starts[cell++] = next;
                    }
                }
                int old = (int) keyed[k];
                oldToNew[old] = next;
                newToOldAll[next] = old;
                newToSrcAll[next] = s;
                next++;
            }
            if (starts != null) {
                while (cell < starts.length) {
                    starts[cell++] = next;
                }
                cellStart[s] = starts;
            }
        }
        // dead nodes last, any order
        for (int oi = 0; oi < numSources; oi++) {
            int s = order[oi];
            FixedBitSet alive = liveNodes.get(s);
            int size = sources.get(s).size(0);
            int[] oldToNew = oldToNewPerSource[s];
            for (int node = 0; node < size; node++) {
                if (alive.get(node)) continue;
                oldToNew[node] = next;
                newToOldAll[next] = node;
                newToSrcAll[next] = s;
                next++;
            }
        }

        this.maxOrdinal = next - 1;
        List<OrdinalMapper> mappers = new ArrayList<>(numSources);
        for (int s = 0; s < numSources; s++) {
            mappers.add(new ArrayOrdinalMapper(s, oldToNewPerSource[s], newToOldAll, newToSrcAll, maxOrdinal));
        }
        if (regionMode) {
            long descents = Math.max(1, hubMap.descents.sum());
            log.info("Region ordinals: {} descents, {} moves/descent", descents, String.format("%.1f", hubMap.moves.sum() / (double) descents));
        }
        log.info("Region ordinals assigned ({}): {} ordinals across {} sources in {} ms", regionMode ? "region order" : "PQ-prefix order",
                next, numSources, (System.nanoTime() - t0) / 1_000_000);
        return mappers;
    }

    private void joinAll(List<Callable<Void>> tasks) {
        try {
            for (var f : executor.invokeAll(tasks)) {
                f.get();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        } catch (ExecutionException e) {
            throw new RuntimeException(e.getCause());
        }
    }

    private static final class Scratch implements AutoCloseable {
        final int[] candSrc, candNode;
        final float[] candScore;
        final boolean[] candCodeOnly;   // per-candidate: pairwise diversity checks via codes (reverse offers)
        final SelectedVecCache selectedCache;
        final VectorFloat<?> tmpVec, baseVec;
        final GraphSearcher[] gs;
        final ByteSequence<?> pqCode;
        // cell-join experiment state (allocated lazily)
        HubMap.Scorer cellScorer;
        byte[] cellBlock = new byte[0];        // blocked codes of the cell slice being scanned
        short[] cellBlockSums = new short[0];  // scan-kernel output (u16 per code)
        short[] cellBlockNorms = new short[0]; // COSINE: second pass over the norm table
        int[] probeCells = new int[0];
        int probeCount;
        // level-1 beam state (beam probe selection): generation-stamped visited marks over level-1 positions
        int[] beamPos;
        float[] beamScore;
        boolean[] beamExpanded;
        long[] scoredKeys; // scored cells of the current beam, packed for sorting
        int[] l1Visited;   // open-addressing set of level-1 positions, sized 4x the beam's maximum visit count
        int probeNode = -1, probeSrc = -1;
        int[] topIds;
        float[] topScores;

        /**
         * Constructs scratch space with buffers sized for the maximum expected candidates and degree.
         */
        Scratch(int maxCandidateSize, int maxDegree, int dimension, List<OnDiskGraphIndex> sources, ProductQuantization pq) {
            this.candSrc = new int[maxCandidateSize];
            this.candNode = new int[maxCandidateSize];
            this.candScore = new float[maxCandidateSize];
            this.candCodeOnly = new boolean[maxCandidateSize];
            this.selectedCache = new SelectedVecCache(maxDegree, dimension);
            this.tmpVec = vectorTypeSupport.createFloatVector(dimension);
            this.baseVec = vectorTypeSupport.createFloatVector(dimension);
            this.pqCode = (pq == null) ? null : vectorTypeSupport.createByteSequence(pq.getSubspaceCount());

            this.gs = new GraphSearcher[sources.size()];
            for (int i = 0; i < sources.size(); i++) {
                gs[i] = new GraphSearcher.Builder(sources.get(i).getView()).build();
                gs[i].usePruning(false);
            }
        }

        /**
         * Closes all graph searchers and resets the cache.
         */
        @Override
        public void close() throws IOException {
            for (var s : gs) s.close();
            selectedCache.reset();
        }
    }

    /**
     * Specification for a batch of nodes to be processed from one source index.
     */
    private static final class BatchSpec {
        final int sourceIdx;
        final int[] nodes;              // materialized node ids for this source
        final int start;
        final int end;

        BatchSpec(int sourceIdx, int[] nodes, int start, int end) {
            this.sourceIdx = sourceIdx;
            this.nodes = nodes;
            this.start = start;
            this.end = end;
        }
    }

    /**
     * Provides Vamana-style diversity filtering for neighbor selection during compaction.
     */
    private static final class CompactVamanaDiversityProvider {
        /**
         * the diversity threshold; 1.0 is equivalent to HNSW; Vamana uses 1.2 or more
         */
        public final float alpha;

        /**
         * used to compute diversity
         */
        public final VectorSimilarityFunction vsf;

        /**
         * Create a new diversity provider
         */
        public CompactVamanaDiversityProvider(VectorSimilarityFunction vsf, float alpha) {
            this.vsf = vsf;
            this.alpha = alpha;
        }

        // Optional code-based path: candidates flagged in {@code codeOnly} have exact scores against the
        // node but are compared with already-selected neighbours through their merged-PQ codes, so
        // their vectors are never read. Codes come from the pre-encode cache by merged ordinal.
        private PreEncodedCodeCache codeCache;
        private List<OrdinalMapper> codeRemappers;
        private SymmetricCodeSimilarity codeSimilarity;
        private boolean[] codeOnly;
        private byte[] candCode;

        CompactVamanaDiversityProvider withCodes(PreEncodedCodeCache cache, List<OrdinalMapper> remappers,
                                                 SymmetricCodeSimilarity similarity, boolean[] codeOnlyFlags) {
            this.codeCache = cache;
            this.codeRemappers = remappers;
            this.codeSimilarity = similarity;
            this.codeOnly = codeOnlyFlags;
            this.candCode = new byte[cache.codeSize()];
            return this;
        }

        private void fetchCode(int src, int node) {
            codeCache.get(codeRemappers.get(src).oldToNew(node), candCode);
        }

        /**
         * Selects diverse neighbors from the candidates listed in {@code order[0..orderSize)}, using a
         * gradually increasing alpha threshold so that the nearest candidates are prioritized. Fills
         * {@code selectedCache}; the candidate arrays are not modified.
         */
        public void retainDiverse(int[] candSrc, int[] candNode, float[] candScore, int[] order, int orderSize,
                                  int maxDegree, SelectedVecCache selectedCache, VectorFloat<?> tmp, GraphSearcher[] gs) {
            selectedCache.reset();
            if (orderSize == 0) return;
            int nSelected = 0;

            // add diverse candidates, gradually increasing alpha to the threshold
            // (so that the nearest candidates are prioritized)
            float currentAlpha = 1.0f;
            while (currentAlpha <= alpha + 1E-6 && nSelected < maxDegree) {
                for (int i = 0; i < orderSize && nSelected < maxDegree; i++) {
                    int ci = order[i];
                    int cSrc = candSrc[ci];
                    int cNode = candNode[ci];
                    float cScore = candScore[ci];

                    OnDiskGraphIndex.View cView = (OnDiskGraphIndex.View) gs[cSrc].getView();
                    boolean codeOnlyCandidate = codeSimilarity != null && codeOnly[ci];
                    if (!codeOnlyCandidate) {
                        cView.getVectorInto(cNode, tmp, 0);
                    }
                    // The candidate's code is needed whenever a code-based comparison can occur: the
                    // candidate itself has no vector, or a selected neighbour has none.
                    boolean codeFetched = codeSimilarity != null && (codeOnlyCandidate || selectedCache.anyCodeOnly);
                    if (codeFetched) {
                        fetchCode(cSrc, cNode);
                    }
                    if (isDiverse(cView, cNode, tmp, cScore, currentAlpha, selectedCache, codeOnlyCandidate)) {
                        selectedCache.add(cSrc, cView, cNode, cScore, tmp);
                        if (codeSimilarity != null) {
                            if (!codeFetched) {
                                fetchCode(cSrc, cNode);
                            }
                            selectedCache.setCode(candCode, codeOnlyCandidate);
                        }
                        nSelected++;
                    }
                }

                currentAlpha += DIVERSITY_ALPHA_STEP;
            }
        }

        /**
         * Checks if a candidate is diverse enough by ensuring it's closer to the base node
         * than to any already-selected neighbor (scaled by alpha threshold). Pairs in which either
         * side is a code-only entry are compared through their codes.
         */
        private boolean isDiverse(OnDiskGraphIndex.View cView, int cNode, VectorFloat<?> cVec, float cScore, float alpha, SelectedVecCache selectedCache) {
            return isDiverse(cView, cNode, cVec, cScore, alpha, selectedCache, false);
        }

        private boolean isDiverse(OnDiskGraphIndex.View cView, int cNode, VectorFloat<?> cVec, float cScore, float alpha, SelectedVecCache selectedCache, boolean candCodeOnly) {
            for (int j = 0; j < selectedCache.size; j++) {
                if (selectedCache.views[j] == cView && selectedCache.nodes[j] == cNode) {
                    return false; // already selected; don't add a duplicate
                }
                float sim = codeSimilarity != null && (candCodeOnly || selectedCache.codeOnly[j])
                        ? codeSimilarity.similarity(candCode, selectedCache.codes[j])
                        : vsf.compare(cVec, selectedCache.vecs[j]);
                if (sim > cScore * alpha) {
                    return false;
                }
            }
            return true;
        }

    }

    /**
     * Cache for storing selected diverse neighbors along with their metadata and vector copies.
     */
    static final class SelectedVecCache {
        int[] sourceIdx;
        OnDiskGraphIndex.View[] views;
        int[] nodes;
        float[] scores;
        VectorFloat<?>[] vecs;
        int size;
        // merged-PQ code of each entry and whether the entry was selected from its code alone
        // (no vector copy); anyCodeOnly short-circuits the code path when no such entry exists
        byte[][] codes;
        boolean[] codeOnly;
        boolean anyCodeOnly;

        /**
         * Constructs a cache with the specified capacity and vector dimension.
         */
        SelectedVecCache(int capacity, int dimension) {
            codes = new byte[capacity][];
            codeOnly = new boolean[capacity];
            sourceIdx = new int[capacity];
            views = new OnDiskGraphIndex.View[capacity];
            nodes = new int[capacity];
            scores = new float[capacity];
            vecs = new VectorFloat<?>[capacity];
            for(int c = 0; c < capacity; ++c) {
                vecs[c] = vectorTypeSupport.createFloatVector(dimension);
            }
            size = 0;
        }

        /**
         * Resets the cache for reuse.
         */
        void reset() {
            anyCodeOnly = false;
            size = 0;
        }

        /**
         * Adds a selected neighbor to the cache, copying its vector.
         */
        void add(int source, OnDiskGraphIndex.View view, int node, float score, VectorFloat<?> vec) {
            sourceIdx[size] = source;
            views[size] = view;
            nodes[size] = node;
            scores[size] = score;
            vecs[size].copyFrom(vec, 0, 0, vec.length());
            codeOnly[size] = false;
            size++;
        }

        /**
         * Attaches the merged-PQ code of the entry just added (index {@code size - 1}).
         */
        void setCode(byte[] code, boolean isCodeOnly) {
            int i = size - 1;
            if (codes[i] == null || codes[i].length != code.length) {
                codes[i] = new byte[code.length];
            }
            System.arraycopy(code, 0, codes[i], 0, code.length);
            codeOnly[i] = isCodeOnly;
            if (isCodeOnly) {
                anyCodeOnly = true;
            }
        }
    }

}

