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
import java.util.concurrent.atomic.AtomicLong;
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
import io.github.jbellis.jvector.vector.types.FloatArray;
import io.github.jbellis.jvector.util.BoundedLongHeap;
import io.github.jbellis.jvector.util.NumericUtils;
import org.agrona.collections.Int2IntHashMap;
import org.agrona.collections.IntHashSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static java.lang.Math.*;

/**
 * Merges several {@link OnDiskGraphIndex} sources into one on-disk graph.
 * <p>
 * Every live node keeps its own retained edges and gains cross-source edges, followed by
 * Vamana-style diversity selection. At the base layer the sources are processed smallest first:
 * a node gathers candidates only from the sources larger than its own, and each candidate is
 * also offered back to the node it found, so the reverse direction of every source pair is
 * supplied by propagation rather than by a second pass. The largest source therefore gathers
 * nothing itself and only folds the offers it received into its retained edges. Base-layer
 * candidates come from the cell join: the merge holds every live vector in a scratch region of
 * the output, ordered so that each source's members of every cell (a level-1 node of the largest
 * source's hierarchy) are contiguous, and a node's candidates are found by streaming the cells a
 * beam over that hierarchy picks and scoring their vectors exactly. Only the pairwise diversity
 * comparisons run on a compact wide code. The graph search remains for the upper layers and for
 * a node without a cell.
 * <p>
 * Ordinals in the output are assigned by the compactor: each source's live nodes are numbered by
 * region so the cell join can scan a cell's members contiguously, deleted nodes get no ordinal,
 * and the mapping is published through {@link #ordinalMappers()} once {@link #compact} has run.
 * Upper layers are merged with a greedy descent to the layer followed by a beam search.
 */
public final class OnDiskGraphIndexCompactor implements Accountable {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final Logger log = LoggerFactory.getLogger(OnDiskGraphIndexCompactor.class);

    // Compaction constants

    private static final float DIVERSITY_ALPHA = 1.2f;
    private static final int TARGET_BATCHES_PER_SOURCE = 40;
    private static final int TARGET_NODES_PER_BATCH = 128;

    /**
     * The merge scans and scores on the vectors themselves, always: the cell walk streams each
     * target's cells in ordinal order, so every candidate score is exact and the scan reads
     * sequentially whether or not the store is resident. Only the finalize's pairwise diversity
     * comparisons, the one place vectors are re-read at random, use a compact wide code that stays
     * resident. One configuration for every input type and every memory regime; the input type
     * decides only whether the output carries codes.
     */
    // The raw-vector store every merge scans: a sidecar strategy whose "code" is the vector.
    private QuantizationCompactionStrategy vectorStore = QuantizationCompactionStrategy.NONE;
    // full-precision merges scan cells for a range of nodes at once (the cell walk); the range is bounded by the
    // memory its query vectors take (about 3 GB) and by this many nodes
    private static final int EXACT_JOIN_MAX_RANGE = 1 << 22;
    private static final int MIN_SEARCH_TOP_K = 2;
    private static final int SEARCH_TOP_K_MULTIPLIER = 4;

    private List<OnDiskGraphIndex> sources;
    // Optional non-fused compressed sidecar, parallel to `sources`. Null when sources carry their
    // quantization inline (FUSED_PQ) or have none. When non-null, compact(Path, Path) retrains the
    // compressor on merged vectors and writes a single merged CompressedVectors to compressedPath.
    private final List<CompressedVectors> sourceCompressed;
    private List<FixedBitSet> liveNodes;
    private final List<Integer> numLiveNodesPerSource;
    /**
     * Compactor-assigned ordinals, built by {@link #buildRegionOrdinalMappers} at the start of
     * {@link #compact}: each source's live nodes numbered by region, sources in ascending-size
     * processing order. Record write offsets follow new ordinals, so processing in the same order
     * makes the writer sequential, and consecutive nodes of every source explore the same region
     * of every target. Callers read the mapping back via {@link #ordinalMappers()}.
     */
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
    private static final int REVERSE_CANDIDATE_SLOTS = 16;


    private int[] sizeRank;            // rank of each source in ascending live-node order
    private int[] l0ProcessOrder;      // source indices in ascending live-node order
    private BandedReverseCandidateBuffer reverseCandidates; // non-null only while L0 is being compacted
    // The sidecar quantization strategy participating in this compaction (NONE when the caller
    // did not supply per-source CompressedVectors). Set by compactGraphImpl; consulted wherever
    // the fused path consults the inline strategy for codes.
    private QuantizationCompactionStrategy activeSidecarStrategy = QuantizationCompactionStrategy.NONE;
    // Wide code: a second, near-lossless PQ (2 dimensions per subspace) trained at merge time and kept in a
    // scratch cache by new ordinal, encoded by the same pre-encode pass as the scan code. At level 0 the cell
    // join reranks its candidates and scores the node's retained edges from it, and the diversity checks
    // compare it, so no record but the node's own is read.
    private ProductQuantization pqWide;
    // Full-precision sources (no quantization anywhere): the scratch holds the vectors themselves, the
    // hub map holds the upper layers' vectors, and level 0 scans, scores and diversifies on vectors.
    private boolean exactMode;
    private volatile ExactJoin exactJoin;   // full-precision merge: scan results of the node range level 0 is processing
    private final LongAdder finalizeRowsTouched = new LongAdder(), finalizeRowsRequested = new LongAdder();
    private ThreadLocal<Scratch> scratchLocal;  // the per-thread scratch of the current compaction (set by compactLevels)
    // What level 0 evaluates candidates from: the wide code for coded sources, the vectors
    // themselves for full-precision ones. Non-null only while level 0 runs.
    private PreEncodedCodeCache candidateStore;
    // Full-precision merge above the memory budget: the compact code the diversity test reads,
    // while the walk keeps reading candidateStore (the vectors themselves).
    private PreEncodedCodeCache diversityCache;
    private final LongAdder l0WideScores = new LongAdder();
    /**
     * Dimensions per subspace of the wide code. Four, measured: on cap-6M (1536-d), cohere-10M
     * (1024-d) and dpr-gemma-10m (768-d) the diversity test scored best at four dimensions per
     * subspace, beating both the coarser code the old flat 192-subspace cap produced (by 2.6 pt at
     * 1536-d) and the finer two-dimensional code that cap was meant to approximate (by 0.06 to
     * 0.61 pt). Finer is not better here because the test asks whether two candidates are too
     * similar, and a coarser code collides near-duplicates onto one centroid, which is exactly the
     * answer the pruning rule wants.
     */
    private static final int WIDE_DIMS_PER_SUBSPACE = 4;

    // ---- Cell join: level-0 cross-source candidates from an exact scan of the cell members ----
    // With compactor-assigned ordinals every source's nodes are grouped by the level-1 node of the
    // largest source's hierarchy they descend to (their cell); each source's nodes of a cell form a
    // contiguous ordinal range and their codes are contiguous in the pre-encoded cache. A node's
    // candidates in another source are found by scanning that source's codes in the node's best
    // cells with an 8-bit lookup table, then rescoring the survivors exactly. Requires reassigned
    // ordinals with a hierarchy in the largest source and a merged code cache; otherwise level 0
    // falls back to the graph search.
    // Codes scanned per (node, target) at most; governs how many cells are probed. This is the
    // cell join's recall/time operating point: it caps how deeply the probed cells are scanned,
    // while the probe beam below bounds which cells are probed at all.
    static final int CELL_BUDGET = 4096;
    static final int CELL_ASSIGN_EF = 8;      // level-1 beam width when assigning a node's cell
    static final int CELL_PROBE_EF = 16;      // level-1 beam width when choosing a node's probe cells
    private HubMap cellMap;                   // resident hub map kept through level 0
    private int[] cellWalkPosition;           // hub node id -> cell (walk position); MAX_VALUE if not level 1
    private int[] cellToPosition;             // cell -> level-1 position in the hub map
    private int[][] cellStart;                // [source][cell] -> first new ordinal of that source's nodes in the cell
    private int cellCount;
    private final LongAdder cellCodesScanned = new LongAdder();
    private final LongAdder cellScans = new LongAdder();

    /** Reverse-offer band width: peak buffer memory is O(band), independent of node count. */
    private static final int OFFER_BAND_WIDTH = 1 << 20;
    private Path spillParent; // parent dir of the compaction output; set by compact()
    final AtomicLong retainedOnlyNodes = new AtomicLong();

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
            VectorSimilarityFunction similarityFunction,
            ForkJoinPool executor) {
        this(sources, null, liveNodes, similarityFunction, executor);
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
            VectorSimilarityFunction similarityFunction,
            ForkJoinPool executor) {
        checkBeforeCompact(sources, sourceCompressed, liveNodes);

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
        // Output ordinals are dense over the live nodes (buildRegionOrdinalMappers numbers them
        // 0..live-1), so the extent is known before the mapping itself exists.
        maxOrdinal = numTotalNodes - 1;
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
            List<FixedBitSet> liveNodes) {
        validateInputSizes(sources, liveNodes);
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
    private void validateInputSizes(List<OnDiskGraphIndex> sources, List<FixedBitSet> liveNodes) {
        if (sources.isEmpty()) {
            throw new IllegalArgumentException("Must have at least one source");
        }
        Objects.requireNonNull(liveNodes, "liveNodes");
        if (sources.size() != liveNodes.size()) {
            throw new IllegalArgumentException("sources and liveNodes must have the same size");
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

    /** Number of level-0 cell scans run so far (one per node and target); for tests and diagnostics. */
    long cellScanCount() {
        return cellScans.sum();
    }

    /**
     * The output ordinal of every source node, one mapper per source: {@code oldToNew} of a live
     * node is its ordinal in the compacted graph, of a deleted node {@link OrdinalMapper#OMITTED}.
     * Available once {@link #compact} has assigned the ordinals.
     */
    public List<OrdinalMapper> ordinalMappers() {
        if (remappers == null) {
            throw new IllegalStateException("ordinals are assigned by compact()");
        }
        return remappers;
    }

    /**
     * Main compaction entry point. Merges all source indexes into a single output index at the
     * specified path, handling PQ retraining if needed, and writing header, all layers, and footer.
     */
    @Experimental
    public void compact(Path outputPath) throws FileNotFoundException {
        QuantizationCompactionStrategy strategy = detectInlineStrategy();
        // Whatever the sources carry, the merge itself runs on the vectors: the scratch holds them
        // in merged-ordinal order and the whole merge is exact. A fused output still gets its codes
        // from the inline strategy's pre-encode, which is what that output format needs.
        QuantizationCompactionStrategy scratch = SidecarCompactionStrategy.scratchVectors(buildContext());
        scratch.retrain(similarityFunction);
        try {
            activeSidecarStrategy = scratch;
            vectorStore = scratch;
            compactGraphImpl(outputPath, strategy);
        } finally {
            activeSidecarStrategy = QuantizationCompactionStrategy.NONE;
            vectorStore = QuantizationCompactionStrategy.NONE;
            strategy.onAfterClose(outputPath);
            scratch.onAfterClose(outputPath);
            scratch.releaseTransientState();
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
        QuantizationCompactionStrategy scratch = SidecarCompactionStrategy.scratchVectors(buildContext());
        try {
            sidecarStrategy.retrain(similarityFunction);
            scratch.retrain(similarityFunction);
            activeSidecarStrategy = sidecarStrategy;
            vectorStore = scratch;
            compactGraphImpl(graphPath, inlineStrategy);
            // Record the graph path with the sidecar strategy before writeSidecar: the strategy
            // defers its cache-region truncation until the sidecar copy completes.
            sidecarStrategy.onAfterClose(graphPath);
            sidecarStrategy.writeSidecar(compressedPath);
        } catch (IOException e) {
            throw new RuntimeException("Sidecar compaction failed", e);
        } finally {
            activeSidecarStrategy = QuantizationCompactionStrategy.NONE;
            vectorStore = QuantizationCompactionStrategy.NONE;
            inlineStrategy.onAfterClose(graphPath);
            scratch.onAfterClose(graphPath);
            scratch.releaseTransientState();
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

        exactMode = vectorStore.compressor() instanceof RawVectorCode;
        assert exactMode : "the merge always runs on a raw-vector store";
        remappers = buildRegionOrdinalMappers(pq);
        if (cellMap != null) {
            log.info("Level-0 candidates are scanned and scored on the vectors themselves ({} B per node in scratch)",
                     4 * dimension);
            pqWide = new PQRetrainer(sources, liveNodes, dimension).train(wideSubspaces());
            if (pqWide != null) {
                vectorStore.setSecondaryCompressor(pqWide);
                log.info("Diversity comparisons use a {}-subspace wide code ({} B per node); every candidate score stays exact",
                         pqWide.getSubspaceCount(), pqWide.getSubspaceCount());
            } else {
                log.info("Too few vectors to train the wide code; diversity comparisons stay on the vectors");
            }
        }
        // The strategies were built before the ordinals existed; refresh so code placement
        // (pre-encode caches, sidecar order) follows the on-disk ordinals.
        strategy.onRemappersUpdated(buildContext());
        activeSidecarStrategy.onRemappersUpdated(buildContext());
        if (vectorStore != activeSidecarStrategy) {
            vectorStore.onRemappersUpdated(buildContext());
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
            if (vectorStore != activeSidecarStrategy) {
                vectorStore.onAfterHeader(writer);
            }

            // Approximate cross-source scoring is available whenever a merged code cache exists:
            // fused (inline codes) or sidecar (strategy pre-encode cache built just above).
            boolean compressedPrecision = fusedPQEnabled || activeSidecarStrategy.getCodeCache() != null;
            candidateStore = strategy.getSecondaryCache() != null ? strategy.getSecondaryCache() : activeSidecarStrategy.getSecondaryCache();
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
        scratchLocal = threadLocalScratch;

        setupCrossLink();
        if (exactMode && cellMap != null) {
            // The vector store is always the sidecar scratch. Naming it explicitly matters when an
            // writer's fused code cache, and reading 4d bytes from a code-sized entry underflows.
            candidateStore = vectorStore.getCodeCache();
            if (pqWide != null) {
                diversityCache = vectorStore.getSecondaryCache();
                log.info("Diversity code cache {}", diversityCache == null ? "unavailable; diversity comparisons fall back to the vectors" : "ready");
            }
        }

        for (int level = 0; level < maxDegrees.size(); level++) {
            int searchTopK = Math.max(MIN_SEARCH_TOP_K, ((maxDegrees.get(level) + sources.size() - 1) / sources.size()) * SEARCH_TOP_K_MULTIPLIER);
            if (level == 0) log.info("Cross-link search budget: searchTopK={} per target", searchTopK);

            CompactionParams params = new CompactionParams(fusedPQEnabled, compressedPrecision, searchTopK, pq);

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
                    // A batch's records are consecutive ordinals, hence adjacent on disk: gather each
                    // contiguous run into one positional write instead of one syscall per record.
                    final ByteBuffer run = ByteBuffer.allocateDirect(4 << 20);
                    java.util.function.Consumer<List<WriteResult>> writeResults = (results) -> {
                        try {
                            long runStart = -1, runEnd = -1;
                            for (WriteResult r : results) {
                                int len = r.data.remaining();
                                if (runStart >= 0 && (r.fileOffset != runEnd || run.remaining() < len)) {
                                    run.flip();
                                    writeFully(fc, run, runStart);
                                    run.clear();
                                    runStart = -1;
                                }
                                if (len > run.capacity()) {
                                    writeFully(fc, r.data, r.fileOffset);
                                    continue;
                                }
                                if (runStart < 0) {
                                    runStart = runEnd = r.fileOffset;
                                }
                                run.put(r.data);
                                runEnd += len;
                            }
                            if (runStart >= 0) {
                                run.flip();
                                writeFully(fc, run, runStart);
                                run.clear();
                            }
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    };

                    // Sources run smallest-first, one group at a time: the drain between groups is
                    // the barrier that guarantees every reverse-candidate offer into a source has
                    // completed before that source's own nodes read them.
                    for (int s : l0ProcessOrder) {
                        List<BatchSpec> batches = buildBatchesForSource(s, 0);
                        boolean walk = exactMode && cellMap != null && candidateStore != null
                                && sizeRank[s] < sources.size() - 1 && !batches.isEmpty();
                        if (!walk) {
                            runBatchesWithBackpressure(batches, ecs, submitOne, writeResults);
                            continue;
                        }
                        // Full-precision merge: the batches are in new-ordinal order and a source's live
                        // nodes are one contiguous range, so a group of consecutive batches is a range of
                        // new ordinals. The cell walk scans for the whole range before its batches run.
                        OrdinalMapper m = remappers.get(s);
                        int range = exactJoinRange();
                        int gStart = 0;
                        while (gStart < batches.size()) {
                            BatchSpec first = batches.get(gStart);
                            int lo = m.oldToNew(first.nodes[first.start]);
                            int gEnd = gStart, hi = lo;
                            while (gEnd < batches.size()) {
                                BatchSpec b = batches.get(gEnd);
                                int last = m.oldToNew(b.nodes[b.end - 1]);
                                if (gEnd > gStart && last + 1 - lo > range) break;
                                hi = last + 1;
                                gEnd++;
                            }
                            exactJoin = cellWalk(s, lo, hi, params.searchTopK);
                            runBatchesWithBackpressure(batches.subList(gStart, gEnd), ecs, submitOne, writeResults);
                            exactJoin = null;
                            gStart = gEnd;
                        }
                    }
                }

                log.info("Cross-link: {} of {} nodes took the retained-only fast path", retainedOnlyNodes.get(), maxOrdinal + 1);
                if (cellMap != null) {
                    long scans = Math.max(1, cellScans.sum());
                    if (finalizeRowsRequested.sum() > 0) {
                        log.info("Finalize prefetch: {} rows touched for {} requested ({}x shared)",
                                 finalizeRowsTouched.sum(), finalizeRowsRequested.sum(),
                                 String.format("%.2f", finalizeRowsRequested.sum() / (double) finalizeRowsTouched.sum()));
                    }
                    log.info("Cell join: {} scans, {} codes/scan", cellScans.sum(), cellCodesScanned.sum() / scans);
                }
                reverseCandidates.close();
                reverseCandidates = null; // consumed entirely within L0; scales with node count
                if (candidateStore != null) {
                    log.info("Wide code: {} candidate scores from the wide cache", l0WideScores.sum());
                    candidateStore = null;   // the strategy unmaps it with its own cache
                }
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
            if (numNodes > 1) {
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

        // Warming is worth ~2x above the memory budget and pure cost below it, so it follows the
        // same decision as the diversity representation.
        if (exactMode && candidateStore != null && exactJoin != null) {
            prefetchFinalizeRows(bs, scratch);
        }
        for (int i = bs.start; i < bs.end; i++) {
            int node = bs.nodes[i];
            if (!liveNodes.get(bs.sourceIdx).get(node)) continue;
            out.add(processBaseNode(node, bs.sourceIdx, scratch, writer, params));
        }

        return out;
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
        if (candidateStore != null && scratch.wide == null) {
            scratch.wide = new WideDecoder(candidateStore, exactMode, pqWide, dimension);
        }
        // Guarded separately from `wide`: the cell walk creates `wide` on its own threads, so a
        // shared guard would leave those threads without a diversity decoder.
        if (scratch.wide != null && scratch.div == null) {
            // candidate vectors for the diversity test come from the compact code,
            // never from the vector store; every score the merge uses is still exact.
            scratch.div = diversityCache != null ? new WideDecoder(diversityCache, false, pqWide, dimension) : scratch.wide;
        }

        int candSize = gatherCandidates(node, 0, sourceIdx, scratch, scratch.baseVec, params);

        int[] order = IntStream.range(0, candSize).toArray();
        sortOrderByScoreDesc(order, scratch.candScore, candSize);

        var selected = scratch.selectedCache;

        var provider = new CompactVamanaDiversityProvider(similarityFunction, DIVERSITY_ALPHA);
        if (candidateStore != null) {
            provider.withCandidateVectors(scratch.candVec, scratch.candHasVec);
            if (diversityCache != null) {
                provider.withCodedCandidateVectors(scratch.baseVec);
            }
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

        new CompactVamanaDiversityProvider(similarityFunction, DIVERSITY_ALPHA)
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
        Arrays.fill(scratch.candHasVec, false);

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
            if (exactMode && candidateStore != null) {
                // full-precision merge: the offerer's vector comes from the vector store
                for (int i = offersStart; i < candSize; i++) {
                    scratch.div.decode(remappers.get(scratch.candSrc[i]).oldToNew(scratch.candNode[i]), scratch.candVec[i]);
                    scratch.candHasVec[i] = true;
                }
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
        final boolean wide = level == 0 && candidateStore != null;
        final OrdinalMapper mapper = wide ? remappers.get(sourceIdx) : null;
        while (it.hasNext()) {
            int nb = it.nextInt();
            if (!indexAlive.get(nb)) continue;

            if (wide) {
                VectorFloat<?> v = scratch.candVec[candSize];
                scratch.div.decode(mapper.oldToNew(nb), v);
                scratch.candHasVec[candSize] = true;
                scratch.candScore[candSize] = similarityFunction.compare(baseVec, v);
                l0WideScores.increment();
            } else {
                searchView.getVectorInto(nb, scratch.tmpVec, 0);
                scratch.candScore[candSize] = similarityFunction.compare(baseVec, scratch.tmpVec);
            }
            scratch.candSrc[candSize] = sourceIdx;
            scratch.candNode[candSize] = nb;
            candSize++;
        }
        return candSize;
    }


    /**
     * Level 0: hands a node the cell walk's results for one target - the best {@code searchTopK}
     * ordinals with their exact scores - decoding each candidate's diversity representation as it
     * goes. A node without a cell has no walk results and simply gets no candidates from this target.
     */
    private int gatherFromCellWalk(int node, int nodeSourceIdx, int targetIdx, Scratch scratch, int candSize) {
            // full-precision merge: the survivors were computed for this node's range by the cell walk
            ExactJoin j = exactJoin;
            int slot = j.slot(targetIdx, remappers.get(nodeSourceIdx).oldToNew(node));
            OrdinalMapper mapper = remappers.get(targetIdx);
            int base = slot * j.k, size = j.sizes[slot];
            for (int e = 0; e < size; e++) {
                long key = j.heaps[base + e];
                int newOrd = (int) key;
                scratch.candSrc[candSize] = targetIdx;
                scratch.candNode[candSize] = mapper.newToOld(newOrd);
                scratch.div.decode(newOrd, scratch.candVec[candSize]);
                scratch.candHasVec[candSize] = true;
                scratch.candScore[candSize] = Float.intBitsToFloat((int) (key >>> 32));
                l0WideScores.increment();
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
        if (level == 0 && cellMap != null) {
            return gatherFromCellWalk(node, nodeSourceIdx, sourceIdx, scratch, candSize);
        }

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

        if (level == 0) {
            // rerankK = searchTopK: a wider beam's extra candidates are largely pruned by
            // diversity selection, so the doubled approximate-phase cost buys almost no recall.
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
     * Probe cells for the current node: a beam from {@code startPos} over the level-1 graph; every
     * level-1 node it scores becomes a probe cell, best first. The scan takes cells in that order
     * until {@link #CELL_BUDGET} codes, so the number of cells probed adapts to their size.
     */
    private void selectCellsByBeam(Scratch scratch, HubMap.Scorer scorer, int startPos) {
        final int stride = cellMap.degree[1];
        if (scratch.l1Visited == null) {
            scratch.l1Visited = new IntHashSet(2 * beamScoredCapacity(CELL_PROBE_EF, stride));
            scratch.scoredKeys = new long[beamScoredCapacity(CELL_PROBE_EF, stride)];
        }
        long[] scored = scratch.scoredKeys;
        int scoredCount = beam(cellMap, scorer, 1, startPos, CELL_PROBE_EF, scratch.l1Visited, scored);
        Arrays.sort(scored, 0, scoredCount);
        int n = 0;
        for (int i = scoredCount - 1; i >= 0; i--) {   // best first
            int cell = cellWalkPosition[cellMap.nodes[1][(int) scored[i]]];
            if (cell >= 0 && cell < cellCount) scratch.probeCells[n++] = cell;
        }
        scratch.probeCount = n;
    }

    /** The best-scoring level-1 position a beam of width {@code ef} from {@code startPos} scores. */
    private static int bestLevel1ByBeam(HubMap map, HubMap.Scorer scorer, int startPos, int ef, IntHashSet visited, long[] scored) {
        int count = beam(map, scorer, 1, startPos, ef, visited, scored);
        long best = scored[0];
        for (int i = 1; i < count; i++) if (scored[i] > best) best = scored[i];
        return (int) best;
    }

    /**
     * Best-first search over {@code level} of the hub map by code score, width {@code ef}. Every
     * scored position is appended to {@code scored} as a sortable key (score in the high bits,
     * position in the low 32); returns how many. Replaced beam entries re-enter as unexpanded, so
     * expansions are capped at {@code 2 * ef} (the beam has converged long before) and the caller
     * sizes {@code scored} for that bound ({@link #beamScoredCapacity}).
     */
    private static int beam(HubMap map, HubMap.Scorer scorer, int level, int startPos, int ef, IntHashSet visited, long[] scored) {
        final int stride = map.degree[level];
        final int[] adj = map.adjacency[level];
        int[] pos = new int[ef];
        float[] sc = new float[ef];
        boolean[] expanded = new boolean[ef];
        int[] nbPos = new int[stride];
        visited.clear();
        int size = 0, scoredCount = 0;
        float s0 = scorer.score(level, startPos);
        pos[0] = startPos; sc[0] = s0; size = 1;
        visited.add(startPos);
        scored[scoredCount++] = packScored(s0, startPos);
        for (int expansions = 0; expansions < 2 * ef; expansions++) {
            int bi = -1;
            for (int i = 0; i < size; i++) {
                if (!expanded[i] && (bi < 0 || sc[i] > sc[bi])) bi = i;
            }
            if (bi < 0) break;
            expanded[bi] = true;
            int p = pos[bi];
            // score the unvisited neighbours in one tight loop first (their cache misses overlap)
            int cnt = 0;
            for (int j = 0; j < stride; j++) {
                int nb = adj[p * stride + j];
                if (nb < 0) break;
                int np = map.position(level, nb);
                if (np >= 0 && visited.add(np)) nbPos[cnt++] = np;
            }
            for (int j = 0; j < cnt; j++) {
                scored[scoredCount + j] = packScored(scorer.score(level, nbPos[j]), nbPos[j]);
            }
            for (int j = 0; j < cnt; j++) {
                long key = scored[scoredCount + j];
                int np = nbPos[j];
                float score = unpackScore(key);
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
            scoredCount += cnt;
        }
        return scoredCount;
    }

    /** Packs a float score into a sortable long key with the position in the low 32 bits. */
    private static long packScored(float score, int position) {
        return ((long) NumericUtils.floatToSortableInt(score) << 32) | (position & 0xFFFFFFFFL);
    }

    private static float unpackScore(long key) {
        return NumericUtils.sortableIntToFloat((int) (key >>> 32));
    }

    /** Most positions a beam of width {@code ef} can score: {@code 2 * ef} expansions of {@code stride} plus the seed. */
    private static int beamScoredCapacity(int ef, int stride) {
        return 2 * ef * stride + 1;
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
    private static void writeFully(FileChannel fc, ByteBuffer b, long pos) throws IOException {
        while (b.hasRemaining()) {
            pos += fc.write(b, pos);
        }
    }

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
     * Per-query PQ lookup-table scorer over the merged pre-encode cache: the query's partial sums
     * over the codebooks ({@link VectorUtil#calculatePartialSums}), assembled per source-graph
     * node from its cached code. Similarity conversion mirrors {@link VectorSimilarityFunction}:
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
        // Center-adjusted PQ encodes (v - globalCentroid). For EUCLIDEAN, center the query so
        // per-subspace distances compose; for DOT, score subspaces against the raw query and add
        // the constant dot(query, centroid) term.
        final VectorFloat<?> center = pq.getGlobalCentroid();
        final boolean dot = f == VectorSimilarityFunction.DOT_PRODUCT;
        VectorFloat<?> q = query;
        float dotConstant = 0;
        if (center != null) {
            if (dot) {
                dotConstant = VectorUtil.dotProduct(query, center);
            } else {
                q = VectorUtil.sub(query, center);
            }
        }
        final VectorFloat<?> partialSums = vectorTypeSupport.createFloatVector(msub * clusters);
        int queryOffset = 0;
        for (int m = 0; m < msub; m++) {
            int sz = pq.getSubvectorSize(m);
            VectorUtil.calculatePartialSums(pq.getCodebookVector(m), m, sz, clusters, q, queryOffset, f, partialSums);
            queryOffset += sz;
        }
        final ByteSequence<?> code = vectorTypeSupport.createByteSequence(cache.codeSize());
        final byte[] codeBytes = code.get() instanceof byte[] ? (byte[]) code.get() : new byte[cache.codeSize()];
        final float constant = dotConstant;
        return node -> {
            // Dead nodes have no merged ordinal (and no cached code); the search may still
            // traverse them. Score them at the floor: they are excluded from results by the
            // alive filter, and candidates are exact-rescored before diversity regardless.
            if (!alive.get(node)) {
                return 0f;
            }
            cache.get(mapper.oldToNew(node), codeBytes);
            if (codeBytes != code.get()) {
                for (int m = 0; m < msub; m++) code.set(m, codeBytes[m]);
            }
            float sum = VectorUtil.assembleAndSum(partialSums, clusters, code, 0, msub);
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

        // remappers: one oldToNew array per source plus the shared newToOld / newToSrc arrays
        if (remappers != null) {
            for (var source : sources) {
                size += OH + REF + (long) source.getIdUpperBound() * Integer.BYTES;
            }
            size += 2 * (OH + REF + (long) (maxOrdinal + 1) * Integer.BYTES);
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

        // candSrc, candNode, candScore arrays
        scratchSize += (long) maxCandidateSize * Integer.BYTES; // candSrc
        scratchSize += (long) maxCandidateSize * Integer.BYTES; // candNode
        scratchSize += (long) maxCandidateSize * Float.BYTES;   // candScore

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
        final ProductQuantization pq;

        CompactionParams(boolean fusedPQEnabled, boolean compressedPrecision, int searchTopK, ProductQuantization pq) {
            this.fusedPQEnabled = fusedPQEnabled;
            this.compressedPrecision = compressedPrecision;
            this.searchTopK = searchTopK;
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

    /** Loads the largest source's upper layers (adjacency and vectors) into a {@link HubMap}. */
    private HubMap buildHubMap(OnDiskGraphIndex hub) {
        long t0 = System.nanoTime();
        HubMap h = new HubMap(hub, similarityFunction, dimension);
        try (var view = hub.getView()) {
            h.entryNode = view.entryNode().node;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
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
                h.upperPosition[level] = new Int2IntHashMap(n * 2, 0.65f, -1);
                for (int i = 0; i < n; i++) {
                    h.upperPosition[level].put(nodes[i], i);
                }
            }
            int stride = hub.getDegree(level);
            h.degree[level] = stride;
            int[] adj = new int[n * stride];
            Arrays.fill(adj, -1);
            if ((long) n * dimension > Integer.MAX_VALUE - 8) {
                throw new IllegalStateException("hub map: level " + level + " too large (" + n + " nodes x " + dimension + ")");
            }
            final VectorFloat<?> levelVectors = vectorTypeSupport.createFloatVector(n * dimension);
            final float[] levelNorms = h.cosine ? new float[n] : null;
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
                    try (var view = hub.getView()) {
                        for (int i = lo; i < hi; i++) {
                            var neighbors = view.getNeighborsIterator(lvl, nodes[i]);
                            int j = 0;
                            while (neighbors.hasNext() && j < stride) {
                                adj[i * stride + j++] = neighbors.nextInt();
                            }
                            view.getVectorInto(nodes[i], vec, 0);
                            levelVectors.copyFrom(vec, 0, i * dimension, dimension);
                            if (levelNorms != null) {
                                levelNorms[i] = (float) Math.sqrt(VectorUtil.dotProduct(vec, vec));
                            }
                        }
                    }
                    return null;
                });
            }
            joinAll(tasks);
            h.adjacency[level] = adj;
            h.vectors[level] = levelVectors;
            h.norms[level] = levelNorms;
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
    /** The similarity functions the cell walk's scoring handles: dot product, Euclidean and cosine. */
    private static boolean cellsSupport(VectorSimilarityFunction f) {
        return f == VectorSimilarityFunction.DOT_PRODUCT || f == VectorSimilarityFunction.EUCLIDEAN
                || f == VectorSimilarityFunction.COSINE;
    }

    /** Subspace count of the wide code: {@link #WIDE_DIMS_PER_SUBSPACE} dimensions each, never fewer than 8. */
    private int wideSubspaces() {
        return Math.max(8, dimension / WIDE_DIMS_PER_SUBSPACE);
    }

    private List<OrdinalMapper> buildRegionOrdinalMappers(ProductQuantization pq) {
        long t0 = System.nanoTime();
        int numSources = sources.size();
        // The output ordinal space is dense over LIVE nodes: a deleted node gets no ordinal, so no
        // record, code slot or sidecar entry is spent on it. Every ordinal-indexed structure the
        // writer and strategies size from maxOrdinal shrinks with deletions accordingly.
        long totalOrdinals = 0;
        for (int n : numLiveNodesPerSource) {
            totalOrdinals += n;
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
            hubMap = buildHubMap(hubSource);
            walkPosition = hubMap.walkPositions();
            log.info("Region ordinals: hub source {} (maxLevel {}), {} level-1 nodes walked",
                     order[numSources - 1], hubSource.getMaxLevel(), hubMap.nodes[1].length);
        } else {
            log.info("Region ordinals: the largest source has no hierarchy; falling back to PQ-prefix ordinals");
        }
        HubMap hubMapRef = hubMap;
        int[] walkPositionRef = walkPosition;
        int prefixBytes = pq == null ? 0 : Math.min(4, pq.getSubspaceCount());
        if (regionMode && cellsSupport(similarityFunction)) {
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
            // enough windows to keep every worker busy on small sources, at most 256K nodes each
            int window = Math.max(4096, Math.min(1 << 18, size / (4 * taskWindowSize) + 1));
            List<Callable<Void>> tasks = new ArrayList<>();
            for (int from = 0; from < size; from += window) {
                final int lo = from;
                final int hi = Math.min(size, from + window);
                tasks.add(() -> {
                    source.prefetchL0Records(lo, hi - 1);
                    VectorFloat<?> vec = vectorTypeSupport.createFloatVector(dimension);
                    ByteSequence<?> code = pq == null ? null : vectorTypeSupport.createByteSequence(pq.getSubspaceCount());
                    HubMap.Scorer scorer = regionMode ? hubMapRef.scorer() : null;
                    IntHashSet beamVisited = regionMode ? new IntHashSet(2 * beamScoredCapacity(CELL_ASSIGN_EF, hubMapRef.degree[1])) : null;
                    long[] beamScored = regionMode ? new long[beamScoredCapacity(CELL_ASSIGN_EF, hubMapRef.degree[1])] : null;
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
                                    landing = hubMapRef.nodes[1][bestLevel1ByBeam(hubMapRef, scorer, lp, CELL_ASSIGN_EF, beamVisited, beamScored)];
                                }
                                int pos = landing >= 0 && landing < walkPositionRef.length ? walkPositionRef[landing] : Integer.MAX_VALUE;
                                key = pos & 0xFFFFFFFFL;
                            } else {
                                key = 0;
                                if (pq != null) {
                                    pq.encodeTo(vec, code);
                                    for (int b = 0; b < prefixBytes; b++) {
                                        key = (key << 8) | (code.get(b) & 0xFFL);
                                    }
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
        // deleted nodes: no ordinal
        for (int oi = 0; oi < numSources; oi++) {
            int s = order[oi];
            FixedBitSet alive = liveNodes.get(s);
            int size = sources.get(s).size(0);
            int[] oldToNew = oldToNewPerSource[s];
            for (int node = 0; node < size; node++) {
                if (!alive.get(node)) {
                    oldToNew[node] = OrdinalMapper.OMITTED;
                }
            }
        }
        assert next == totalOrdinals : next + " ordinals assigned for " + totalOrdinals + " live nodes";

        this.maxOrdinal = next - 1;
        List<OrdinalMapper> mappers = new ArrayList<>(numSources);
        for (int s = 0; s < numSources; s++) {
            mappers.add(new ArrayOrdinalMapper(s, oldToNewPerSource[s], newToOldAll, newToSrcAll, maxOrdinal));
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

    /**
     * Full-precision merge: touches every vector-store row this batch will read during its per-node
     * finalize — the cell walk's survivors and each node's retained same-source edges — in ordinal
     * order, once per distinct row. The finalize's own reads then hit warm pages, and rows shared by
     * several nodes of the batch (common, since a batch's nodes are ordinal neighbours probing the
     * same cells) are fetched once instead of once per node.
     */
    private void prefetchFinalizeRows(BatchSpec bs, Scratch scratch) {
        final ExactJoin j = exactJoin;
        final int src = bs.sourceIdx;
        final FixedBitSet alive = liveNodes.get(src);
        final OrdinalMapper mapper = remappers.get(src);
        final int perNode = j.targets * j.k + maxDegrees.get(0) + 8;
        final int need = (bs.end - bs.start) * perNode;
        if (scratch.finalizeOrds == null || scratch.finalizeOrds.length < need) {
            scratch.finalizeOrds = new int[need];
        }
        final int[] ords = scratch.finalizeOrds;
        int count = 0;
        var view = (OnDiskGraphIndex.View) scratch.gs[src].getView();
        for (int i = bs.start; i < bs.end; i++) {
            int node = bs.nodes[i];
            if (!alive.get(node)) continue;
            int newOrd = mapper.oldToNew(node);
            for (int t = 0; t < sources.size(); t++) {
                if (j.targetIndex[t] < 0) continue;
                int slot = j.slot(t, newOrd);
                int base = slot * j.k, size = j.sizes[slot];
                for (int e = 0; e < size && count < ords.length; e++) {
                    ords[count++] = (int) j.heaps[base + e];
                }
            }
            var it = view.getNeighborsIterator(0, node);
            while (it.hasNext() && count < ords.length) {
                int nb = it.nextInt();
                if (alive.get(nb)) {
                    ords[count++] = mapper.oldToNew(nb);
                }
            }
        }
        if (count == 0) {
            return;
        }
        Arrays.sort(ords, 0, count);
        // The finalize reads the diversity representation: the compact code when there is one,
        // else the vector store itself.
        final PreEncodedCodeCache warmTarget = diversityCache != null ? diversityCache : candidateStore;
        int distinct = 0, prev = -1;
        for (int i = 0; i < count; i++) {
            if (ords[i] != prev) {
                warmTarget.warm(ords[i]);
                prev = ords[i];
                distinct++;
            }
        }
        finalizeRowsTouched.add(distinct);
        finalizeRowsRequested.add(count);
    }

    /** Node range size of the cell walk: its query vectors take about 3 GB at most. */
    private int exactJoinRange() {
        return (int) Math.max(1 << 16, Math.min(EXACT_JOIN_MAX_RANGE, (3L << 30) / (4L * dimension)));
    }

    /**
     * The cell scan of a full-precision merge as a join: for the nodes {@code [lo, hi)} of source
     * invert them into per-cell prober lists, then walk each larger target's cells in ordinal order
     * once, streaming each cell's vectors from the store and scoring them against all the nodes that
     * probe the cell. Every store vector is read once per range; the per-node top-k are those of a
     * per-node scan.
     */
    private ExactJoin cellWalk(int src, int lo, int hi, int k) {
        final long t0 = System.nanoTime();
        final ExactJoin j = new ExactJoin(src, lo, hi, k, sources.size(), sizeRank);
        final int n = j.n;
        final boolean needNorms = similarityFunction != VectorSimilarityFunction.DOT_PRODUCT;
        final VectorFloat<?>[] queries = new VectorFloat<?>[n];
        final float[] queryNorm2 = new float[n];
        // phase 1: query vectors and probe cells; (cell << 32 | node index) pairs per target, per task
        final int tasks1 = Math.max(1, Math.min(taskWindowSize * 4, n / 256));
        final int chunk = (n + tasks1 - 1) / tasks1;
        final long[][][] pairsByTask = new long[tasks1][j.targets][];
        final int[][] countByTask = new int[tasks1][j.targets];
        List<Callable<Void>> phase1 = new ArrayList<>();
        for (int c = 0; c < tasks1; c++) {
            final int task = c, from = c * chunk, to = Math.min(n, from + chunk);
            if (from >= to) continue;
            phase1.add(() -> {
                Scratch sc = scratchLocal.get();
                if (sc.cellScorer == null) {
                    sc.cellScorer = cellMap.scorer();
                    sc.probeCells = new int[beamScoredCapacity(CELL_PROBE_EF, cellMap.degree[1])];
                }
                if (sc.wide == null) sc.wide = new WideDecoder(candidateStore, exactMode, pqWide, dimension);
                long[][] pairs = pairsByTask[task];
                int[] count = countByTask[task];
                for (int t = 0; t < j.targets; t++) pairs[t] = new long[Math.max(64, (to - from) * 6)];
                for (int i = from; i < to; i++) {
                    int newOrd = lo + i;
                    VectorFloat<?> q = vectorTypeSupport.createFloatVector(dimension);
                    sc.wide.decode(newOrd, q);
                    queries[i] = q;
                    queryNorm2[i] = needNorms ? VectorUtil.dotProduct(q, q) : 0f;
                    int ownCell = cellOf(src, newOrd);
                    if (ownCell < 0) continue;
                    sc.cellScorer.setQuery(q);
                    sc.probeCount = 0;
                    selectCellsByBeam(sc, sc.cellScorer, cellToPosition[ownCell]);
                    sc.probeNode = -1;
                    for (int target = 0; target < sources.size(); target++) {
                        int ti = j.targetIndex[target];
                        if (ti < 0) continue;
                        int[] starts = cellStart[target];
                        long scanned = 0;
                        for (int p = 0; p < sc.probeCount; p++) {
                            int cell = sc.probeCells[p];
                            int cn = starts[cell + 1] - starts[cell];
                            if (cn <= 0) continue;
                            if (scanned > 0 && scanned + cn > CELL_BUDGET) break;
                            scanned += cn;
                            if (count[ti] == pairs[ti].length) pairs[ti] = Arrays.copyOf(pairs[ti], pairs[ti].length * 2);
                            pairs[ti][count[ti]++] = ((long) cell << 32) | i;
                        }
                        cellCodesScanned.add(scanned);
                        cellScans.increment();
                    }
                }
                return null;
            });
        }
        joinAll(phase1);
        final long t1 = System.nanoTime();
        // phase 2: per target, sort the pairs by cell and walk the cells in ordinal order
        long pairTotal = 0;
        for (int target = 0; target < sources.size(); target++) {
            final int ti = j.targetIndex[target];
            if (ti < 0) continue;
            int total = 0;
            for (int c = 0; c < tasks1; c++) total += countByTask[c][ti];
            final long[] pairs = new long[total];
            int at = 0;
            for (int c = 0; c < tasks1; c++) {
                int cnt = countByTask[c][ti];
                if (cnt > 0) System.arraycopy(pairsByTask[c][ti], 0, pairs, at, cnt);
                at += cnt;
                pairsByTask[c][ti] = null;
            }
            if (total == 0) continue;
            Arrays.parallelSort(pairs);
            pairTotal += total;
            final int[] starts = cellStart[target];
            // split the sorted pairs into contiguous cell ranges for the workers
            int tasks2 = Math.max(1, Math.min(taskWindowSize * 8, total / 4096));
            List<Callable<Void>> phase2 = new ArrayList<>();
            int p = 0;
            for (int c = 0; c < tasks2 && p < total; c++) {
                int q = Math.min(total, (int) ((long) total * (c + 1) / tasks2));
                if (q <= p) continue;
                while (q < total && (pairs[q] >>> 32) == (pairs[q - 1] >>> 32)) q++;   // do not split a cell
                final int from = p, to = q;
                p = q;
                phase2.add(() -> {
                    Scratch sc = scratchLocal.get();
                    if (sc.wide == null) sc.wide = new WideDecoder(candidateStore, exactMode, pqWide, dimension);
                    final VectorFloat<?> tmp = sc.tmpVec;
                    final VectorFloat<?>[] chunkQueries = new VectorFloat<?>[EXACT_QUERY_CHUNK];
                    final float[] chunkScores = new float[EXACT_QUERY_CHUNK];
                    int[] probers = new int[256];
                    int a = from;
                    while (a < to) {
                        int cell = (int) (pairs[a] >>> 32);
                        int b = a;
                        while (b < to && (int) (pairs[b] >>> 32) == cell) b++;
                        int count = b - a;
                        if (probers.length < count) probers = new int[count * 2];
                        for (int r = a; r < b; r++) probers[r - a] = (int) pairs[r];
                        final int cLo = starts[cell], cHi = starts[cell + 1];
                        for (int c0 = 0; c0 < count; c0 += EXACT_QUERY_CHUNK) {
                            final int cn = Math.min(EXACT_QUERY_CHUNK, count - c0);
                            for (int r = 0; r < cn; r++) chunkQueries[r] = queries[probers[c0 + r]];
                            for (int newOrd = cLo; newOrd < cHi; newOrd++) {
                                sc.wide.decode(newOrd, tmp);
                                VectorUtil.dotProductMulti(tmp, chunkQueries, cn, chunkScores);
                                float vNorm2 = needNorms ? VectorUtil.dotProduct(tmp, tmp) : 0f;
                                for (int r = 0; r < cn; r++) {
                                    int i = probers[c0 + r];
                                    float dot = chunkScores[r];
                                    float score;
                                    if (!needNorms) {
                                        score = (1 + dot) / 2;
                                    } else if (similarityFunction == VectorSimilarityFunction.EUCLIDEAN) {
                                        score = 1 / (1 + Math.max(0f, queryNorm2[i] + vNorm2 - 2 * dot));
                                    } else {
                                        score = (1 + dot / (float) Math.sqrt(Math.max(1e-12, (double) queryNorm2[i] * vNorm2))) / 2;
                                    }
                                    int slot = ti * n + i;
                                    if (score > j.thresholds[slot]) {
                                        j.offer(slot, newOrd, score);
                                    }
                                }
                            }
                        }
                        a = b;
                    }
                    return null;
                });
            }
            joinAll(phase2);
        }
        log.info("Cell walk: source {} ordinals [{}, {}): probes {} ms, {} (cell, node) pairs over {} targets, walk {} ms",
                 src, lo, hi, (t1 - t0) / 1_000_000, pairTotal, j.targets, (System.nanoTime() - t1) / 1_000_000);
        return j;
    }

    private static final int EXACT_QUERY_CHUNK = 32;   // queries scored together against each vector; 32 x 1.5 KB stays in L1

    private static final class Scratch implements AutoCloseable {
        final int[] candSrc, candNode;
        final float[] candScore;
        final SelectedVecCache selectedCache;
        final VectorFloat<?> tmpVec, baseVec;
        final GraphSearcher[] gs;
        final ByteSequence<?> pqCode;
        WideDecoder wide;               // decodes the scan store (raw vectors in exact mode) for the walk
        WideDecoder div;                // decodes the diversity representation into candVec slots
        final VectorFloat<?>[] candVec; // per-candidate decoded vector (wide code), used for scoring and diversity
        final boolean[] candHasVec;
        // cell join state (allocated on first use)
        HubMap.Scorer cellScorer;
        int[] probeCells;
        int probeCount;
        long[] scoredKeys;      // positions scored by the current probe beam, packed with their scores
        IntHashSet l1Visited;   // visited set for the probe beam
        int probeNode = -1, probeSrc = -1;
        int[] finalizeOrds;     // full-precision merge: the batch's candidate rows, for the ordered prefetch

        /**
         * Constructs scratch space with buffers sized for the maximum expected candidates and degree.
         */
        Scratch(int maxCandidateSize, int maxDegree, int dimension, List<OnDiskGraphIndex> sources, ProductQuantization pq) {
            this.candSrc = new int[maxCandidateSize];
            this.candNode = new int[maxCandidateSize];
            this.candScore = new float[maxCandidateSize];
            this.candVec = new VectorFloat<?>[maxCandidateSize];
            for (int i = 0; i < maxCandidateSize; i++) {
                candVec[i] = vectorTypeSupport.createFloatVector(dimension);
            }
            this.candHasVec = new boolean[maxCandidateSize];
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

}

