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
import java.util.stream.IntStream;
import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedFeature;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.*;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.disk.SimpleReader;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.github.jbellis.jvector.vector.types.ByteSequence;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static java.lang.Math.*;

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


    // Non-final so releaseSourcesBeforeRefine() can drop the strong reference once compactGraphImpl
    // has consumed them, letting the source graphs' in-heap upper-layer adjacency + feature buffers
    // be reclaimed before refineCompactedGraph loads a second full graph. Read only during
    // compaction (validation, compactGraphImpl, cost estimation) — never after refinement starts.
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
    private boolean refineAfterCompaction = false;

    /**
     * Whether to run the second-pass neighbor refinement after the merged graph is written
     * (default false). Refinement is a navigability pass: it has no measurable effect on
     * recall, but it improves query latency on the merged index at the cost of a significant
     * fraction of total compaction time. Enable it when search latency matters more than
     * compaction throughput.
     */
    @Experimental
    public void setRefineAfterCompaction(boolean refineAfterCompaction) {
        this.refineAfterCompaction = refineAfterCompaction;
    }

    // ---- Full-precision cross-source search seeding (always on for full-precision compaction) ----
    // The full-precision cross-source search descends from the global entry node and reads a full
    // vector (getVectorInto) on every hop — a disk fault per hop at >RAM scale. Seeding warm-starts
    // the layer-0 beam from entry points near the query, produced during compaction: when node u
    // searches source t, a finished same-source neighbor k of u contributes the members of its
    // merged adjacency that belong to t (they are near k, hence near u). This cuts the hop count and
    // therefore the full-vector reads. Seeds are scored with full vectors (a few per partition),
    // which is cheap against the descent hops they replace. Only the WHERE-candidates-come-from
    // changes; diversity, scores fed to it, and the output are unaffected. Distinct from the fused
    // ADC path — this targets the full-vector-read regime where hop count dominates.
    //
    // Memory: a finished node's merged edges are NOT mirrored in heap — they are read back from the
    // output file (which already holds them) on demand, so per-node heap is just the done-flag and
    // the two ordinal maps (~12 bytes/node) rather than ~142 with a degree-wide adjacency mirror.
    // The done-flag is published only after the record is physically written, so an output-file
    // read of a flagged node always sees committed data.
    private static final int SEEDS_PER_PARTITION = 4;
    private static final int SEED_POOL_CAPACITY = 64;
    private boolean seedingActive;
    private int seedDegree;
    private java.util.concurrent.atomic.AtomicIntegerArray doneFlag;
    private int[] srcOfNewOrd;
    private int[] oldOfNewOrd;
    private CompactWriter seedWriter;   // for neighborCountFileOffset; set during L0 compaction
    final java.util.concurrent.atomic.AtomicLong seededSearches = new java.util.concurrent.atomic.AtomicLong();
    final java.util.concurrent.atomic.AtomicLong coldSearches = new java.util.concurrent.atomic.AtomicLong();

    /** Builds the seeding structures (done-flag + ordinal maps; no adjacency mirror). */
    private void setupSeeding(int baseDegree) {
        int bound = maxOrdinal + 1;
        seedDegree = baseDegree;
        doneFlag = new java.util.concurrent.atomic.AtomicIntegerArray(bound);
        srcOfNewOrd = new int[bound];
        oldOfNewOrd = new int[bound];
        Arrays.fill(srcOfNewOrd, -1);
        for (int s = 0; s < sources.size(); s++) {
            FixedBitSet alive = liveNodes.get(s);
            OrdinalMapper mapper = remappers.get(s);
            for (int oldOrd = 0; oldOrd < alive.length(); oldOrd++) {
                if (!alive.get(oldOrd)) continue;
                int newOrd = mapper.oldToNew(oldOrd);
                srcOfNewOrd[newOrd] = s;
                oldOfNewOrd[newOrd] = oldOrd;
            }
        }
        seedingActive = true;
        log.info("Full-precision search seeding enabled ({} nodes, degree {}; edges read from output)",
                bound, baseDegree);
    }

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

    private PreEncodedCodeCache orderingCache;   // non-null only while L0 runs under fused mode

    // Bounded cluster search: consecutive (similarity-ordered) nodes share one anchor search per
    // target source. The anchor searches CLUSTER_MARGIN deeper than needed; a member rescores the
    // anchor's list against its own query and skips its search only when the triangle inequality
    // certifies that no point outside the shared list can reach its top-K: with delta the
    // query-to-anchor angular distance and ThetaD the anchor's worst kept angle, certification is
    // memberKthAngle <= ThetaD - delta. Fallback is a normal cold search (which becomes the new
    // anchor). Valid for angular similarities (normalized DOT_PRODUCT / COSINE) and EUCLIDEAN,
    // where the underlying distance is a metric; disabled otherwise.
    // On-demand anchor depth: anchors search plain searchTopK; when a member's certificate
    // fails, the anchor search is resumed in CLUSTER_EXT_STEP increments (growing ThetaD for
    // this and all later members) up to CLUSTER_MARGIN total extra results, before the member
    // falls back to its own search. Exact duplicates certify with no extension at all.
    private static final int CLUSTER_MARGIN = 32;
    private static final int CLUSTER_EXT_STEP = 16;
    final java.util.concurrent.atomic.AtomicLong clusterResumes = new java.util.concurrent.atomic.AtomicLong();
    final java.util.concurrent.atomic.AtomicLong clusterCertified = new java.util.concurrent.atomic.AtomicLong();
    final java.util.concurrent.atomic.AtomicLong clusterAnchors = new java.util.concurrent.atomic.AtomicLong();

    // Similarity-assigned merged ordinals: the compactor replaces the caller's remappers with a
    // mapping that numbers each source's live nodes in PQ-code order (sources in ascending-size
    // processing order). Record write offsets follow new ordinals, so processing in the same
    // order makes the writer sequential, and similar vectors become adjacent records in the
    // output — locality that also benefits post-compaction searches. Callers opting in must
    // read the mapping back via {@link #effectiveRemappers()}.
    private static final boolean SIMILARITY_ORDINALS_DEFAULT =
            Boolean.parseBoolean(System.getProperty("jvector.compaction.similarityOrdinals", "false"));
    private boolean similarityOrdinals = SIMILARITY_ORDINALS_DEFAULT;
    private boolean similarityOrdinalsActive;
    private List<OrdinalMapper> effectiveRemappers;   // survives releaseSourcesBeforeRefine
    private int[] sizeRank;            // rank of each source in ascending live-node order
    private int[] l0ProcessOrder;      // source indices in ascending live-node order
    private ReverseCandidateBuffer reverseCandidates; // non-null only while L0 is being compacted
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
        reverseCandidates = new ReverseCandidateBuffer(maxOrdinal + 1, REVERSE_CANDIDATE_SLOTS);
        log.info("Cross-link: L0 source order {} (ascending live nodes), {} reverse slots/node",
                Arrays.toString(l0ProcessOrder), REVERSE_CANDIDATE_SLOTS);
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
            // flush + refinement) is compute- and memory-bandwidth-bound, so sizing to logical
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
        if (sources.size() < 2) {
            throw new IllegalArgumentException("Must have at least two sources");
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
     * When enabled, the compactor assigns merged ordinals itself, numbering nodes in vector
     * similarity order, and ignores the ordinal values of the caller-supplied remappers (their
     * source/oldOrdinal structure is still used to enumerate nodes). The mapping actually used
     * is available from {@link #effectiveRemappers()} after {@code compact(...)} begins.
     * Requires fused-PQ sources; silently keeps caller ordinals otherwise.
     */
    @Experimental
    public void setSimilarityOrdinals(boolean enabled) {
        this.similarityOrdinals = enabled;
    }

    /** The ordinal mappers in effect (the caller's, or the compactor-assigned similarity mapping). */
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
            releaseSourcesBeforeRefine(strategy);
            if (refineAfterCompaction) {
                refineCompactedGraph(outputPath, strategy);
            }
        } finally {
            // Delayed until after refinement so refineCompactedGraph can read from the pre-encoded
            // code cache appended past the projected EOF; onAfterClose unmaps it and truncates.
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
            compactGraphImpl(graphPath, inlineStrategy);
            if (refineAfterCompaction) {
                refineCompactedGraph(graphPath, inlineStrategy);
            }
            sidecarStrategy.writeSidecar(compressedPath);
        } catch (IOException e) {
            throw new RuntimeException("Sidecar compaction failed", e);
        } finally {
            inlineStrategy.onAfterClose(graphPath);
        }
    }

    /**
     * For compaction use. Drops the compactor's strong references to the source graphs and their
     * per-source live-node / remapper sidecars, and tells the strategy to release its
     * {@link CompactionContext} hold on the same. Called between {@code compactGraphImpl} and
     * {@code refineCompactedGraph} so the source graphs' in-heap upper-layer adjacency and feature
     * buffers become GC-eligible before refinement loads a second full graph — the peak that was
     * OOM-ing on memory-tight hosts. The underlying {@code ReaderSupplier}s are still owned and
     * closed by the caller (per {@link OnDiskGraphIndex#close()}'s contract), so we only drop
     * references, never close. Not used by the sidecar {@code compact(graphPath, compressedPath)}
     * path: {@code SidecarCompactionStrategy.writeSidecar} re-reads source vectors after refinement.
     */
    private void releaseSourcesBeforeRefine(QuantizationCompactionStrategy strategy) {
        strategy.releaseSources();
        sources = null;
        liveNodes = null;
        remappers = null;
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
        strategy.retrain(similarityFunction);

        boolean fusedPQEnabled = strategy.writesCodesInline();
        ProductQuantization pq = strategy.compressorAsPQ();
        boolean compressedPrecision = fusedPQEnabled;
        int maxBaseDegree = java.util.Collections.max(maxDegrees);
        io.github.jbellis.jvector.graph.disk.feature.FusedFeature outputFusedFeature =
                strategy.outputFusedFeature(maxBaseDegree);

        if (similarityOrdinals) {
            if (fusedPQEnabled && pq != null && pq.getSubspaceCount() >= 4) {
                remappers = buildSimilarityOrdinalMappers(pq);
                effectiveRemappers = remappers;
                similarityOrdinalsActive = true;
            } else {
                log.info("similarityOrdinals requested but unavailable (requires fused PQ codes); keeping caller remappers");
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

            compactLevels(writer, similarityFunction, fusedPQEnabled, compressedPrecision, pq);
            if (seedingActive) {
                long seeded = seededSearches.get(), cold = coldSearches.get();
                log.info("Full-precision seeding: {} seeded / {} cold cross-source searches ({}% seeded)",
                        seeded, cold, cold + seeded == 0 ? 0 : 100 * seeded / (seeded + cold));
            }

            strategy.onAfterLevels(writer, entryNodeSource, maxDegrees);

            writer.writeFooter();
            log.info("Compaction complete: {}", outputPath);
        } catch (IOException | ExecutionException | InterruptedException e) {
            throw new RuntimeException(e);
        }
        // strategy.onAfterClose is deferred to the public compact() entry points so refinement
        // can read from the still-mapped pre-encode cache section past the projected EOF.
    }

    /**
     * Second pass over the just-written compacted graph. Mirrors
     * {@link io.github.jbellis.jvector.graph.GraphIndexBuilder}'s {@code cleanup()} refinement
     * step: when the merged graph has a hierarchy, iterates only level-1 nodes (which are also
     * in L0); for each node, descends greedily through upper layers and beam-searches level 0
     * carrying entry points layer-to-layer, then rewrites the L0 neighbor list (and the inline
     * per-neighbor PQ codes for fused-PQ outputs) in place. When the merged graph has no
     * hierarchy, falls back to iterating all live L0 nodes.
     * <p>
     * The refinement search uses approximate PQ scoring with an exact reranker when fused-PQ is
     * available (matching the cross-source path in {@code compactLevels}); otherwise it falls
     * back to exact-only scoring backed by inline vectors.
     * <p>
     * For fused-PQ outputs the per-neighbor code write is a memcpy from the
     * {@link QuantizationCompactionStrategy#getCodeCache() pre-encode cache} keyed by new
     * ordinal — no per-neighbor {@code encodeTo} call. The cache lives in the same file past
     * the projected EOF and is truncated away by {@code onAfterClose} once refinement returns.
     * <p>
     * Only L0 records are written. Upper-layer neighbor lists live in an in-memory map after
     * load and have no addressable file offset, so they're left as written by compactLevels.
     */
    private void refineCompactedGraph(Path outputPath, QuantizationCompactionStrategy strategy) {
        log.info("Refining compacted graph: {}", outputPath);
        long t0 = System.nanoTime();

        final int baseDegree = maxDegrees.get(0);
        final boolean hasFusedPQ = strategy.writesCodesInline();
        @SuppressWarnings("unchecked")
        final VectorCompressor<ByteSequence<?>> compressor =
                hasFusedPQ ? (VectorCompressor<ByteSequence<?>>) (VectorCompressor<?>) strategy.compressor() : null;
        final int pqCodeSize = hasFusedPQ ? compressor.compressedVectorSize() : 0;

        final int searchTopK = Math.max(MIN_SEARCH_TOP_K,
                baseDegree * SEARCH_TOP_K_MULTIPLIER);
        final int beamWidth = Math.max(baseDegree, searchTopK) * BEAM_WIDTH_MULTIPLIER;

        // Code cache may or may not be present; capture once so refineOneNode can take the fast path.
        // The cache is shared across threads; refineOneNode duplicates per call (cheap; no per-thread
        // state to track and the duplicates are tiny GC-friendly ByteBuffer wrappers).
        final PreEncodedCodeCache codeCache = hasFusedPQ ? strategy.getCodeCache() : null;
        final int cacheCodeSize = hasFusedPQ ? strategy.getCacheCodeSize() : 0;

        try (var supplier = new SimpleReader.Supplier(outputPath);
             FileChannel fc = FileChannel.open(outputPath, StandardOpenOption.WRITE, StandardOpenOption.READ)) {

            // useFooter=false because the file's logical EOF (where the v6 footer trailer sits) is
            // before the still-attached pre-encode cache section. loadFromFooter() would seek to
            // the actual file length and read garbage as the magic.
            OnDiskGraphIndex mergedGraph = OnDiskGraphIndex.load(supplier, 0, false);

            // Pick the iteration set: when there's a hierarchy, refine only L1 nodes (each also
            // lives in L0, so their L0 record is what we rewrite). Mirrors GraphIndexBuilder's
            // cleanup() which gates improveConnections() on `graph.getMaxLevel() > 0` and iterates
            // `nodeStream(1)`. When there's no hierarchy, fall back to all L0 nodes.
            int[] liveOrdinals;
            int iterationLevel = mergedGraph.getMaxLevel() > 0 ? 1 : 0;
            try (var collectView = mergedGraph.getView()) {
                NodesIterator it = mergedGraph.getNodes(iterationLevel);
                liveOrdinals = new int[it.size()];
                int n = 0;
                while (it.hasNext()) liveOrdinals[n++] = it.next();
            }

            final ThreadLocal<RefineScratch> tls = ThreadLocal.withInitial(() ->
                    new RefineScratch(mergedGraph, baseDegree, dimension, searchTopK, pqCodeSize));

            ExecutorCompletionService<Integer> ecs = new ExecutorCompletionService<>(executor);

            int total = liveOrdinals.length;
            int targetBatches = Math.max(taskWindowSize * 4, 16);
            int batchSize = Math.max(1, (total + targetBatches - 1) / targetBatches);

            final int[] ords = liveOrdinals;
            final boolean fpq = hasFusedPQ;
            final int codeSize = pqCodeSize;
            final VectorCompressor<ByteSequence<?>> cmp = compressor;
            final int bw = beamWidth;
            final PreEncodedCodeCache cache = codeCache;
            final int cacheSz = cacheCodeSize;
            final OnDiskGraphIndex graphRef = mergedGraph;

            log.info("Refining {} live nodes at level {} (hierarchy maxLevel={}, fusedPQ={}, codeCache={})",
                    total, iterationLevel, mergedGraph.getMaxLevel(), fpq, cache != null);

            int submitted = 0;
            for (int start = 0; start < total; start += batchSize) {
                final int s = start;
                final int e = Math.min(start + batchSize, total);
                ecs.submit(() -> {
                    RefineScratch scratch = tls.get();
                    for (int i = s; i < e; i++) {
                        int node = ords[i];
                        refineOneNode(node, scratch, fc, baseDegree, fpq, codeSize, cmp, bw,
                                graphRef, cache, cacheSz);
                    }
                    return e - s;
                });
                submitted++;
            }

            int completed = 0;
            int nodesDone = 0;
            int progressStep = Math.max(1, total / 10);
            int nextProgress = progressStep;
            while (completed < submitted) {
                nodesDone += ecs.take().get();
                completed++;
                if (nodesDone >= nextProgress) {
                    log.info("Refinement progress: {}/{} nodes", nodesDone, total);
                    nextProgress += progressStep;
                }
            }

            // Per-thread scratches live in worker-thread ThreadLocals; closing the supplier in
            // try-with-resources tears down the underlying mapping, so any later access would
            // fail anyway. The references will be GC'd when the worker threads die.
        } catch (IOException | InterruptedException | ExecutionException e) {
            throw new RuntimeException("Refinement failed", e);
        }

        log.info("Refinement complete in {} ms", (System.nanoTime() - t0) / 1_000_000);
    }

    /**
     * Refines a single node by mirroring {@code GraphIndexBuilder.improveConnections}:
     * descend greedily through upper layers carrying entry points layer-to-layer, then beam
     * search at L0. Diversity selection + in-place L0 record rewrite happen at the end.
     * <p>
     * The {@code SearchScoreProvider} uses approximate PQ scoring with an exact reranker when
     * fused-PQ is available; otherwise exact-only via the inline-vector reranker. Diversity
     * always runs over exact scores (so we rescore approximate results after the L0 beam).
     */
    private void refineOneNode(int node,
                               RefineScratch scratch,
                               FileChannel fc,
                               int baseDegree,
                               boolean hasFusedPQ,
                               int pqCodeSize,
                               VectorCompressor<ByteSequence<?>> compressor,
                               int beamWidth,
                               OnDiskGraphIndex mergedGraph,
                               PreEncodedCodeCache codeCache,
                               int cacheCodeSize) {
        OnDiskGraphIndex.View view = scratch.view;
        view.getVectorInto(node, scratch.queryVec, 0);

        // Build score provider for this query. Reranker reads the candidate's inline FP vector
        // (via view.getVectorInto into a worker-private tmp) and computes exact similarity.
        ScoreFunction.ExactScoreFunction reranker = node2 -> {
            view.getVectorInto(node2, scratch.tmpVec, 0);
            return similarityFunction.compare(scratch.queryVec, scratch.tmpVec);
        };
        SearchScoreProvider ssp;
        if (hasFusedPQ) {
            FusedPQ fpq = (FusedPQ) mergedGraph.getFeatures().get(FeatureId.FUSED_PQ);
            var asf = fpq.approximateScoreFunctionFor(scratch.queryVec, similarityFunction, view, reranker);
            ssp = new DefaultSearchScoreProvider(asf, reranker);
        } else {
            ssp = new DefaultSearchScoreProvider(reranker);
        }

        Bits excludeSelf = idx -> idx != node;

        // Per-layer descent. Mirrors GraphSearcher.internalSearch: greedy single-best through
        // each upper layer, then a beam search at layer 0. Entry points carry forward via
        // setEntryPointsFromPreviousLayer so the L0 beam starts from the best-known region
        // rather than the global entry node — much cheaper than the previous full search().
        GraphSearcher gs = scratch.searcher;
        var entry = view.entryNode();
        gs.initializeInternal(ssp, entry, excludeSelf);
        for (int lvl = entry.level; lvl > 0; lvl--) {
            gs.searchOneLayer(ssp, 1, 0f, lvl, excludeSelf);
            gs.setEntryPointsFromPreviousLayer();
        }
        gs.searchOneLayer(ssp, beamWidth, 0f, 0, excludeSelf);

        // Collect candidates. Start with the node's existing L0 edges (rescored exact) so
        // refinement never drops an edge that the search happened to miss — matches the
        // existing+search union pattern from GraphIndexBuilder.insertDiverse.
        scratch.candSize = 0;
        var existing = view.getNeighborsIterator(0, node);
        while (existing.hasNext()) {
            int nb = existing.nextInt();
            if (nb == node) continue;
            view.getVectorInto(nb, scratch.tmpVec, 0);
            scratch.candNode[scratch.candSize] = nb;
            scratch.candScore[scratch.candSize] = similarityFunction.compare(scratch.queryVec, scratch.tmpVec);
            scratch.candSize++;
        }
        // Pull search results from approximateResults. When fused-PQ is on the scores there are
        // approximate; rescore exact for correct diversity comparison against existing edges.
        final boolean rescore = hasFusedPQ;
        gs.approximateResults().foreach((nb, approxScore) -> {
            if (nb == node) return;
            for (int k = 0; k < scratch.candSize; k++) {
                if (scratch.candNode[k] == nb) return; // de-dupe against existing edges
            }
            if (scratch.candSize >= scratch.candNode.length) return;
            float s;
            if (rescore) {
                view.getVectorInto(nb, scratch.tmpVec, 0);
                s = similarityFunction.compare(scratch.queryVec, scratch.tmpVec);
            } else {
                s = approxScore;
            }
            scratch.candNode[scratch.candSize] = nb;
            scratch.candScore[scratch.candSize] = s;
            scratch.candSize++;
        });

        if (scratch.candSize == 0) {
            // No live neighbors found — leave the existing record alone.
            return;
        }

        // Sort candidates by descending score.
        int[] order = scratch.order;
        for (int k = 0; k < scratch.candSize; k++) order[k] = k;
        sortOrderByScoreDesc(order, scratch.candScore, scratch.candSize);

        // Vamana diversity selection with progressively-relaxed alpha.
        int selectedSize = retainDiverseSingleSource(
                view, order, scratch.candNode, scratch.candScore, scratch.candSize,
                baseDegree, scratch.selectedNodes, scratch.selectedVecs, scratch.tmpVec);

        // Build the trailing-section bytes (PQ codes block — if any — followed by count + neighbors).
        ByteBuffer rec = scratch.recordBuffer;
        rec.clear();

        long writeOffset;
        if (hasFusedPQ) {
            // PQ codes block sits between the inline vector and the neighbor count.
            writeOffset = view.offsetFor(node, FeatureId.FUSED_PQ);
            if (codeCache != null) {
                // Memcpy from the pre-encoded cache (indexed by new ordinal). Avoids one FP
                // vector read AND one PQ encode per selected neighbor. The cache resolves the
                // ordinal to a per-thread view internally, so workers don't race.
                byte[] codeBuf = scratch.pqCodeBytes;
                for (int k = 0; k < selectedSize; k++) {
                    int newOrd = scratch.selectedNodes[k];
                    codeCache.get(newOrd, codeBuf);
                    rec.put(codeBuf, 0, cacheCodeSize);
                }
            } else {
                // Fallback: re-encode from the selected neighbor's inline vector. Same as before
                // the cache-reuse optimization. Used when the cache wasn't built (graph too large
                // for a single mapping, or pre-encode failure).
                ByteSequence<?> codeOut = scratch.pqCode;
                for (int k = 0; k < selectedSize; k++) {
                    view.getVectorInto(scratch.selectedNodes[k], scratch.tmpVec, 0);
                    codeOut.zero();
                    compressor.encodeTo(scratch.tmpVec, codeOut);
                    for (int b = 0; b < pqCodeSize; b++) {
                        rec.put(codeOut.get(b));
                    }
                }
            }
            // Pad remaining slots with zero codes (matches CompactWriter's zeroPQ behavior).
            int padSlots = baseDegree - selectedSize;
            for (int s = 0; s < padSlots; s++) {
                for (int b = 0; b < pqCodeSize; b++) rec.put((byte) 0);
            }
        } else {
            writeOffset = view.neighborsOffsetFor(0, node);
        }

        // Neighbor count + ordinals (-1 padding for unused slots).
        rec.putInt(selectedSize);
        for (int k = 0; k < selectedSize; k++) rec.putInt(scratch.selectedNodes[k]);
        for (int k = selectedSize; k < baseDegree; k++) rec.putInt(-1);

        rec.flip();
        try {
            while (rec.hasRemaining()) {
                int n = fc.write(rec, writeOffset);
                writeOffset += n;
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Single-source Vamana diversity selection. Mirrors {@link CompactVamanaDiversityProvider}
     * but operates on one merged graph rather than per-source views, so candidates are bare
     * (node, score) pairs.
     *
     * @return the number of selected neighbors written into {@code selectedNodes}.
     */
    private int retainDiverseSingleSource(OnDiskGraphIndex.View view,
                                          int[] order, int[] candNode, float[] candScore, int candSize,
                                          int maxDegree, int[] selectedNodes,
                                          VectorFloat<?>[] selectedVecs, VectorFloat<?> tmp) {
        if (candSize == 0) return 0;
        int nSelected = 0;
        float currentAlpha = 1.0f;
        final float alpha = 1.2f;
        while (currentAlpha <= alpha + 1E-6 && nSelected < maxDegree) {
            for (int i = 0; i < candSize && nSelected < maxDegree; i++) {
                int ci = order[i];
                int cNode = candNode[ci];
                float cScore = candScore[ci];

                view.getVectorInto(cNode, tmp, 0);

                boolean diverse = true;
                for (int j = 0; j < nSelected; j++) {
                    if (selectedNodes[j] == cNode) { diverse = false; break; }
                    if (similarityFunction.compare(tmp, selectedVecs[j]) > cScore * currentAlpha) {
                        diverse = false;
                        break;
                    }
                }
                if (diverse) {
                    selectedNodes[nSelected] = cNode;
                    selectedVecs[nSelected].copyFrom(tmp, 0, 0, tmp.length());
                    nSelected++;
                }
            }
            currentAlpha += DIVERSITY_ALPHA_STEP;
        }
        return nSelected;
    }

    /** Per-thread scratch space for refinement. One per worker thread, populated lazily via ThreadLocal. */
    private static final class RefineScratch {
        final OnDiskGraphIndex.View view;
        final GraphSearcher searcher;
        final VectorFloat<?> queryVec;
        final VectorFloat<?> tmpVec;
        final int[] candNode;
        final float[] candScore;
        final int[] order;
        int candSize;
        final int[] selectedNodes;
        final VectorFloat<?>[] selectedVecs;
        final ByteSequence<?> pqCode;
        // Heap byte buffer for memcpy from the precomputed code cache into the record buffer.
        final byte[] pqCodeBytes;
        final ByteBuffer recordBuffer;

        RefineScratch(OnDiskGraphIndex mergedGraph, int baseDegree, int dimension, int searchTopK, int pqCodeSize) {
            this.view = mergedGraph.getView();
            this.searcher = new GraphSearcher(mergedGraph);
            this.searcher.usePruning(false);
            this.queryVec = vectorTypeSupport.createFloatVector(dimension);
            this.tmpVec = vectorTypeSupport.createFloatVector(dimension);
            // Candidates = existing neighbors (up to baseDegree) ∪ search results (up to searchTopK).
            int cap = searchTopK + baseDegree + 16;
            this.candNode = new int[cap];
            this.candScore = new float[cap];
            this.order = new int[cap];
            this.selectedNodes = new int[baseDegree];
            this.selectedVecs = new VectorFloat<?>[baseDegree];
            for (int i = 0; i < baseDegree; i++) {
                this.selectedVecs[i] = vectorTypeSupport.createFloatVector(dimension);
            }
            this.pqCode = pqCodeSize > 0 ? vectorTypeSupport.createByteSequence(pqCodeSize) : null;
            this.pqCodeBytes = pqCodeSize > 0 ? new byte[pqCodeSize] : null;
            // Trailing section to rewrite: optional PQ codes block + count + neighbor ids.
            int recordBytes = (pqCodeSize > 0 ? baseDegree * pqCodeSize : 0) + Integer.BYTES + baseDegree * Integer.BYTES;
            this.recordBuffer = ByteBuffer.allocate(recordBytes).order(java.nio.ByteOrder.BIG_ENDIAN);
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
        // When seeding, each thread reads finished nodes' edges from the output file — give Scratch
        // the path so it can open its own read channel.
        Path seedOutputPath = !fusedPQEnabled ? writer.getOutputPath() : null;
        final ThreadLocal<Scratch> threadLocalScratch = ThreadLocal.withInitial(() ->
            new Scratch(maxCandidateSize, scratchDegree, dimension, sources, pq, seedOutputPath)
        );

        // Seeding only helps the full-precision cross-source search (fused already scores hops via
        // RAM-resident ADC codes); enable it there when requested.
        // Seeding is always on for full-precision compaction (fused already scores hops via
        // RAM-resident ADC codes, so it gains nothing there).
        if (!fusedPQEnabled && !seedingActive) {
            setupSeeding(maxDegrees.get(0));
        }
        if (seedingActive) {
            seedWriter = writer; // exposes neighborCountFileOffset for reading finished nodes' edges
        }

        setupCrossLink();
        orderingCache = fusedPQEnabled ? writer.pqCodeCache() : null;

        for (int level = 0; level < maxDegrees.size(); level++) {
            int searchTopK = Math.max(MIN_SEARCH_TOP_K, ((maxDegrees.get(level) + sources.size() - 1) / sources.size()) * SEARCH_TOP_K_MULTIPLIER);
            int beamWidth = Math.max(maxDegrees.get(level), searchTopK) * BEAM_WIDTH_MULTIPLIER;

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
                                // Publish AFTER the write so a seed reader that sees the
                                // flag can safely read this node's edges from the file.
                                if (seedingActive) {
                                    doneFlag.set(r.newOrdinal, 1);
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
                        reverseCandidates.offered.sum(), reverseCandidates.touchedTargets.sum(),
                        maxOrdinal + 1, REVERSE_CANDIDATE_SLOTS, retainedOnlyNodes.get());
                if (clusterCertified.get() + clusterAnchors.get() > 0) {
                    log.info("Cluster search: {} certified from {} anchor searches, {} resumes ({} total)",
                            clusterCertified.get(), clusterAnchors.get(), clusterResumes.get(),
                            clusterCertified.get() + clusterAnchors.get());
                }
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
            if (similarityOrdinalsActive && numNodes > 1) {
                // Merged ordinals were assigned in similarity order, so ordering processing by
                // new ordinal gives similarity locality AND sequential record writes at once.
                OrdinalMapper mapper = remappers.get(s);
                long[] keyed = new long[numNodes];
                for (int k = 0; k < numNodes; k++) {
                    keyed[k] = ((long) mapper.oldToNew(nodes[k]) << 32) | (nodes[k] & 0xFFFFFFFFL);
                }
                Arrays.parallelSort(keyed);
                for (int k = 0; k < numNodes; k++) {
                    nodes[k] = (int) keyed[k];
                }
                log.info("L0 source {}: {} nodes in similarity-ordinal order", s, numNodes);
            }
            // Similarity-ordered scheduling: sort searching sources' nodes by the leading bytes
            // of their PQ code, so consecutive searches walk overlapping target regions. The
            // largest source runs no searches and keeps ordinal order (contiguous record
            // streaming matters more there).
            boolean searches = reverseCandidates == null || sizeRank[s] < sources.size() - 1;
            if (!similarityOrdinalsActive && orderingCache != null && searches && orderingCache.codeSize() >= 4 && numNodes > 1) {
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
        scratch.resetChainSeeds();
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
            if (reverseCandidates.countAt(newOrdinal) == 0) {
                return writeRetainedOnlyRecord(node, sourceIdx, newOrdinal, scratch, writer);
            }
        }

        var sourceView = (OnDiskGraphIndex.View) scratch.gs[sourceIdx].getView();
        sourceView.getVectorInto(node, scratch.baseVec, 0);

        int candSize = gatherCandidates(node, 0, sourceIdx, scratch, scratch.baseVec, params);

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
                        maxDegrees.get(0),
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

        // Note: the done-flag that makes this node's edges available as seeds is set after the
        // record is physically written (in the L0 write callback), not here — so a reader of a
        // flagged node's edges from the output file always sees committed data.

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
     * Bounded cluster search for one (node, target) pair. If a valid anchor exists and the
     * triangle-inequality certificate passes — the member's k-th best rescored distance is at
     * most the anchor's worst kept distance minus the query-to-anchor distance — the member's
     * top-k comes from the shared list with no search. Otherwise runs a cold search
     * {@link #CLUSTER_MARGIN} deeper than k, which becomes the new anchor. Candidates appended
     * are exactly k either way, with exact scores.
     */
    private int clusterSearchL0(int node, int targetIdx, OnDiskGraphIndex.View searchView,
                                FixedBitSet indexAlive, VectorFloat<?> baseVec, Scratch scratch,
                                int candSize, CompactionParams params, SearchScoreProvider ssp) {
        int k = params.searchTopK;

        if (scratch.clusterAnchorValid[targetIdx]) {
            float qaSim = similarityFunction.compare(baseVec, scratch.clusterAnchorQuery[targetIdx]);
            double delta = metricDistance(qaSim);
            int verified = 0;   // prefix of the anchor list already scored against this member
            while (true) {
                int n = scratch.clusterCount[targetIdx];
                double thetaD = metricDistance(scratch.clusterWorstSim[targetIdx]);
                if (delta < thetaD) {
                    // score any anchor-list entries this member hasn't scored yet
                    int[] nodes = scratch.clusterNodes[targetIdx];
                    float[] ms = scratch.clusterMemberScores;
                    for (int i = verified; i < n; i++) {
                        ms[i] = indexAlive.get(nodes[i])
                                ? rescore(searchView, nodes[i], baseVec, scratch.tmpVec)
                                : Float.NEGATIVE_INFINITY;
                    }
                    verified = n;
                    int live = 0;
                    for (int i = 0; i < n; i++) {
                        if (ms[i] != Float.NEGATIVE_INFINITY) live++;
                    }
                    if (live >= k) {
                        Integer[] ord = scratch.clusterOrder;
                        for (int i = 0; i < n; i++) ord[i] = i;
                        Arrays.sort(ord, 0, n, (a, b) -> Float.compare(ms[b], ms[a]));
                        double memberKth = metricDistance(ms[ord[k - 1]]);
                        if (memberKth <= thetaD - delta) {
                            clusterCertified.incrementAndGet();
                            for (int j = 0; j < k; j++) {
                                int idx = ord[j];
                                scratch.candSrc[candSize] = targetIdx;
                                scratch.candNode[candSize] = nodes[idx];
                                scratch.candScore[candSize] = ms[idx];
                                candSize++;
                            }
                            return candSize;
                        }
                    }
                }
                // not certified at current depth: extend the anchor's search if budget remains
                if (scratch.clusterExtended[targetIdx] >= CLUSTER_MARGIN) {
                    break;
                }
                int got = extendAnchor(targetIdx, searchView, scratch, params);
                if (got == 0) {
                    break; // anchor's search is exhausted; no deeper ThetaD exists
                }
            }
        }

        return clusterFallbackSearch(node, targetIdx, searchView, indexAlive, baseVec,
                                     scratch, candSize, params, ssp);
    }

    /**
     * Extends the current anchor's search by up to {@link #CLUSTER_EXT_STEP} further results via
     * {@link GraphSearcher#resume}, exact-rescoring them against the ANCHOR's query and growing
     * the stored list (and so ThetaD) for this and all later members. Returns how many results
     * the resume produced; 0 means the search space is exhausted at this beam's reach.
     */
    private int extendAnchor(int targetIdx, OnDiskGraphIndex.View searchView, Scratch scratch,
                             CompactionParams params) {
        clusterResumes.incrementAndGet();
        SearchResult more = scratch.gs[targetIdx].resume(CLUSTER_EXT_STEP, CLUSTER_EXT_STEP);
        int[] nodes = scratch.clusterNodes[targetIdx];
        float[] ascore = scratch.clusterAnchorScores[targetIdx];
        int at = scratch.clusterCount[targetIdx];
        int got = 0;
        float worst = scratch.clusterWorstSim[targetIdx];
        VectorFloat<?> anchorQuery = scratch.clusterAnchorQuery[targetIdx];
        for (var r : more.getNodes()) {
            float ex = params.fusedPQEnabled
                    ? rescoreAgainst(searchView, r.node, anchorQuery, scratch.tmpVec)
                    : r.score;
            nodes[at] = r.node;
            ascore[at] = ex;
            worst = Math.min(worst, ex);
            at++;
            got++;
        }
        scratch.clusterCount[targetIdx] = at;
        scratch.clusterWorstSim[targetIdx] = worst;
        scratch.clusterExtended[targetIdx] += got;
        return got;
    }

    /** Exact similarity of {@code node} to an arbitrary query vector (not the current baseVec). */
    private float rescoreAgainst(OnDiskGraphIndex.View view, int node, VectorFloat<?> query,
                                 VectorFloat<?> tmp) {
        view.getVectorInto(node, tmp, 0);
        return similarityFunction.compare(query, tmp);
    }

    /** Cold search k+m deep for a node that could not be certified; becomes the new anchor. */
    private int clusterFallbackSearch(int node, int targetIdx, OnDiskGraphIndex.View searchView,
                                      FixedBitSet indexAlive, VectorFloat<?> baseVec, Scratch scratch,
                                      int candSize, CompactionParams params, SearchScoreProvider ssp) {
        int k = params.searchTopK;
        clusterAnchors.incrementAndGet();
        int deep = k;
        SearchResult results = scratch.gs[targetIdx].search(ssp, deep, deep, 0f, 0f, indexAlive);
        int[] nodes = scratch.clusterNodes[targetIdx];
        float[] ascore = scratch.clusterAnchorScores[targetIdx];
        int stored = 0;
        for (var r : results.getNodes()) {
            nodes[stored] = r.node;
            ascore[stored] = params.fusedPQEnabled
                    ? rescore(searchView, r.node, baseVec, scratch.tmpVec)
                    : r.score;
            stored++;
        }
        // rank the anchor list by exact score (defines both the top-k and the worst-kept bound)
        final int fstored = stored;
        Integer[] ord = scratch.clusterOrder;
        for (int i = 0; i < fstored; i++) ord[i] = i;
        Arrays.sort(ord, 0, fstored, (a, b) -> Float.compare(ascore[b], ascore[a]));
        int[] tmpN = new int[fstored];
        float[] tmpS = new float[fstored];
        for (int i = 0; i < fstored; i++) {
            tmpN[i] = nodes[ord[i]];
            tmpS[i] = ascore[ord[i]];
        }
        System.arraycopy(tmpN, 0, nodes, 0, fstored);
        System.arraycopy(tmpS, 0, ascore, 0, fstored);
        scratch.clusterCount[targetIdx] = fstored;
        scratch.clusterWorstSim[targetIdx] = fstored > 0 ? ascore[fstored - 1] : Float.NEGATIVE_INFINITY;
        scratch.clusterExtended[targetIdx] = 0;
        scratch.clusterAnchorQuery[targetIdx].copyFrom(baseVec, 0, 0, baseVec.length());
        scratch.clusterAnchorValid[targetIdx] = fstored >= k;

        int emit = Math.min(k, fstored);
        for (int j = 0; j < emit; j++) {
            scratch.candSrc[candSize] = targetIdx;
            scratch.candNode[candSize] = nodes[j];
            scratch.candScore[candSize] = ascore[j];
            candSize++;
        }
        return candSize;
    }

    /**
     * Collects up to {@link #SEEDS_PER_PARTITION} entry points in {@code targetSourceIdx} for node
     * {@code u}'s search of that source, taken from {@code u}'s finished same-source neighbors'
     * merged edges into the target, scored with full vectors. Writes into
     * {@code scratch.seedNodes}/{@code seedScores}; returns the seed count.
     */
    private int gatherSeeds(int node, int nodeSourceIdx, int targetSourceIdx, VectorFloat<?> baseVec,
                            OnDiskGraphIndex.View targetView, FixedBitSet targetAlive, Scratch scratch) {
        var uView = (OnDiskGraphIndex.View) scratch.gs[nodeSourceIdx].getView();
        var it = uView.getNeighborsIterator(0, node);
        OrdinalMapper uMapper = remappers.get(nodeSourceIdx);
        FixedBitSet uAlive = liveNodes.get(nodeSourceIdx);
        int poolSize = 0;
        while (it.hasNext() && poolSize < SEED_POOL_CAPACITY) {
            int nb = it.nextInt();
            if (!uAlive.get(nb)) continue;
            int nbNew = uMapper.oldToNew(nb);
            if (doneFlag.get(nbNew) == 0) continue; // neighbor not finished; no merged edges yet
            // Read the finished neighbor's merged edges (new ordinals) back from the output file
            // — it already holds them — instead of a heap-resident adjacency mirror.
            int deg = readCompactedNeighbors(nbNew, scratch);
            ByteBuffer buf = scratch.seedEdgeBuf;
            for (int j = 0; j < deg && poolSize < SEED_POOL_CAPACITY; j++) {
                int m = buf.getInt();
                if (m < 0) continue; // padding slot
                if (srcOfNewOrd[m] != targetSourceIdx) continue;
                int mOld = oldOfNewOrd[m];
                if (!targetAlive.get(mOld)) continue;
                boolean dup = false;
                for (int p = 0; p < poolSize; p++) {
                    if (scratch.seedPool[p] == mOld) { dup = true; break; }
                }
                if (!dup) scratch.seedPool[poolSize++] = mOld;
            }
        }
        // Score the pool with full vectors, keep the top SEEDS_PER_PARTITION (descending score).
        int seedCount = 0;
        for (int p = 0; p < poolSize; p++) {
            int cand = scratch.seedPool[p];
            targetView.getVectorInto(cand, scratch.tmpVec, 0);
            float score = similarityFunction.compare(baseVec, scratch.tmpVec);
            if (seedCount < SEEDS_PER_PARTITION) {
                int pos = seedCount++;
                while (pos > 0 && scratch.seedScores[pos - 1] < score) {
                    scratch.seedScores[pos] = scratch.seedScores[pos - 1];
                    scratch.seedNodes[pos] = scratch.seedNodes[pos - 1];
                    pos--;
                }
                scratch.seedScores[pos] = score;
                scratch.seedNodes[pos] = cand;
            } else if (score > scratch.seedScores[SEEDS_PER_PARTITION - 1]) {
                int pos = SEEDS_PER_PARTITION - 1;
                while (pos > 0 && scratch.seedScores[pos - 1] < score) {
                    scratch.seedScores[pos] = scratch.seedScores[pos - 1];
                    scratch.seedNodes[pos] = scratch.seedNodes[pos - 1];
                    pos--;
                }
                scratch.seedScores[pos] = score;
                scratch.seedNodes[pos] = cand;
            }
        }
        return seedCount;
    }

    /**
     * Reads finished node {@code newOrd}'s merged neighbor list from the output file into
     * {@code scratch.seedEdgeBuf}, leaving the buffer positioned at the first neighbor int;
     * returns the neighbor count. On-disk ints are big-endian (ByteBuffer's default order).
     * Safe because the caller only reads flagged nodes, whose records are already written.
     */
    private int readCompactedNeighbors(int newOrd, Scratch scratch) {
        long off = seedWriter.neighborCountFileOffset(newOrd);
        ByteBuffer buf = scratch.seedEdgeBuf;
        int need = (seedDegree + 1) * Integer.BYTES;
        buf.clear().limit(need);
        try {
            int got = 0;
            while (got < need) {
                int r = scratch.outputChannel.read(buf, off + got);
                if (r <= 0) break;
                got += r;
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        buf.flip();
        return buf.getInt(); // count; buffer now positioned at the first neighbor
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
            candSize = reverseCandidates.appendTo(remappers.get(sourceIdx).oldToNew(node),
                    scratch.candSrc, scratch.candNode, scratch.candScore, candSize);
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
        // The whole candidate list is known before any vector is read, so batch-hint every
        // record and let the reads below overlap in the device queue instead of paying one
        // fault of latency each, serially. (The extra iterator pass re-reads adjacency the
        // first pass just faulted in — RAM-cheap.)
        var hintIt = searchView.getNeighborsIterator(level, node);
        var source = sources.get(sourceIdx);
        while (hintIt.hasNext()) {
            int nb = hintIt.nextInt();
            if (indexAlive.get(nb)) {
                source.willNeedL0Record(nb);
            }
        }
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
                searchView,
                baseVec,
                scratch.tmpVec,
                similarityFunction
        );

        if (level == 0) {
            int prevCandSize = candSize;
            boolean clusterMode = !seedingActive && orderingCache != null
                    && clusterSearchUsable() && params.fusedPQEnabled;
            if (clusterMode) {
                candSize = clusterSearchL0(node, sourceIdx, searchView, indexAlive, baseVec,
                                           scratch, candSize, params, ssp);
                // fall through to the common reverse-offer block below
            } else {
                // Seeding: warm-start the L0 beam from finished neighbors' edges into this source,
                // skipping the hierarchy descent (and its per-hop full-vector reads). Falls back to
                // a normal search when no seeds are available.
                int seedCount = seedingActive
                        ? gatherSeeds(node, nodeSourceIdx, sourceIdx, baseVec, searchView, indexAlive, scratch)
                        : 0;
                if (seedCount > 0) {
                    seededSearches.incrementAndGet();
                    scratch.gs[sourceIdx].initializeWithSeeds(ssp, indexAlive,
                            scratch.seedNodes, scratch.seedScores, seedCount);
                    scratch.gs[sourceIdx].searchOneLayer(ssp, params.searchTopK, 0f, 0, indexAlive);
                    candSize = appendApproximateResults(
                            scratch.gs[sourceIdx].approximateResults(), sourceIdx, scratch, candSize);
                } else {
                    if (seedingActive) {
                        coldSearches.incrementAndGet();
                    }
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
                                params.fusedPQEnabled
                                        ? rescore(searchView, r.node, baseVec, scratch.tmpVec)
                                        : r.score;
                        candSize++;
                    }
                }
            }

            // Cross-link: each found node in the (larger) target also learns about this node.
            // Scores here are exact in both scoring modes (fused results were rescored above),
            // and similarity is symmetric, so the offer carries the score the target's own
            // search would have computed.
            if (reverseCandidates != null) {
                OrdinalMapper targetMapper = remappers.get(sourceIdx);
                for (int i = prevCandSize; i < candSize; i++) {
                    reverseCandidates.offer(targetMapper.oldToNew(scratch.candNode[i]),
                            nodeSourceIdx, node, scratch.candScore[i]);
                }
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

            if (params.fusedPQEnabled) {
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
     * Converts a jvector similarity score to the underlying metric distance used by the
     * cluster-search certificates: angular distance for DOT_PRODUCT/COSINE (normalized inputs),
     * Euclidean distance for EUCLIDEAN. Returns NaN for similarities with no metric backing.
     */
    private double metricDistance(float sim) {
        switch (similarityFunction) {
            case DOT_PRODUCT:
            case COSINE: {
                double cos = 2.0 * sim - 1.0;
                return Math.acos(Math.max(-1.0, Math.min(1.0, cos)));
            }
            case EUCLIDEAN: {
                double d2 = 1.0 / Math.max(1e-9, sim) - 1.0;
                return Math.sqrt(Math.max(0.0, d2));
            }
            default:
                return Double.NaN;
        }
    }

    private boolean clusterSearchUsable() {
        switch (similarityFunction) {
            case DOT_PRODUCT:
            case COSINE:
            case EUCLIDEAN:
                return true;
            default:
                return false;
        }
    }

    /**
     * Recomputes exact similarity score between the base vector and a node's vector,
     * used to refine approximate PQ-based search results.
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
                                                              OnDiskGraphIndex.View searchView,
                                                              VectorFloat<?> baseVec,
                                                              VectorFloat<?> tmpVec,
                                                              VectorSimilarityFunction similarityFunction) {
        if (compressedPrecision) {
            ScoreFunction.ExactScoreFunction reranker =
                node2 -> {
                    searchView.getVectorInto(node2, tmpVec, 0);
                    return similarityFunction.compare(baseVec, tmpVec);
                };
            var asf = ((FusedPQ) searchSource.getFeatures().get(FeatureId.FUSED_PQ)).approximateScoreFunctionFor(baseVec, similarityFunction, searchView, reranker);

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

        // liveNodes: FixedBitSet per source. May be null after releaseSourcesBeforeRefine().
        if (liveNodes != null) {
            for (var entry : liveNodes) {
                size += entry.ramBytesUsed();
            }
        }

        // numLiveNodesPerSource: ArrayList of Integers
        size += OH + REF + (long) numLiveNodesPerSource.size() * (OH + Integer.BYTES);

        // remappers: each MapMapper holds an oldToNew HashMap and newToOld Int2IntHashMap.
        // May be null after releaseSourcesBeforeRefine().
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


    /**
     * Thread-local scratch space containing reusable buffers and search state for processing nodes.
     */
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
     * Bounded per-node pool of cross-source neighbor candidates discovered by smaller-source
     * searches, keyed by the target's merged (new) ordinal. Offers keep the top-{@code slots}
     * entries by score. Writers are the smaller sources' worker threads; the reader is the
     * target's own processing task, which runs in a later source group — the inter-group barrier
     * (completion-service drain on the main thread) orders every offer before the read, so
     * {@link #appendTo} needs no locking.
     */
    private static final class ReverseCandidateBuffer {
        /**
         * Per-target slot block, allocated on first offer and released as soon as the target
         * consumes it. Layout: {@code [count | src[slots] | oldOrd[slots] | scoreBits[slots]]},
         * scores stored via {@link Float#floatToRawIntBits}. Targets never discovered by a
         * smaller source's search cost only a null reference, and a consumed target's block is
         * dropped inside {@link #appendTo}, so heap tracks the not-yet-consumed touched targets
         * rather than the merged graph's node count — RAM that above-cache-size merges would
         * otherwise take from the page cache serving cross-source searches. The reference array
         * also removes the {@code ordinal * slots} indexing of a dense layout, which overflows
         * int past floor(2^31 / slots) total nodes (~134M at 16 slots).
         */
        final int slots;
        final int[][] blocks;   // per target ordinal; null until first offer, nulled on consume
        final Object[] locks = new Object[1024];
        final java.util.concurrent.atomic.LongAdder offered = new java.util.concurrent.atomic.LongAdder();
        final java.util.concurrent.atomic.LongAdder touchedTargets = new java.util.concurrent.atomic.LongAdder();

        ReverseCandidateBuffer(int numOrdinals, int slots) {
            this.slots = slots;
            this.blocks = new int[numOrdinals][];
            for (int i = 0; i < locks.length; i++) locks[i] = new Object();
        }

        void offer(int targetNewOrd, int src, int oldOrd, float score) {
            offered.increment();
            synchronized (locks[targetNewOrd & (locks.length - 1)]) {
                int[] b = blocks[targetNewOrd];
                if (b == null) {
                    b = new int[1 + 3 * slots];
                    blocks[targetNewOrd] = b;
                    touchedTargets.increment();
                }
                int n = b[0];
                for (int i = 0; i < n; i++) {
                    if (b[1 + i] == src && b[1 + slots + i] == oldOrd) return;
                }
                if (n < slots) {
                    b[1 + n] = src;
                    b[1 + slots + n] = oldOrd;
                    b[1 + 2 * slots + n] = Float.floatToRawIntBits(score);
                    b[0] = n + 1;
                    return;
                }
                int minIdx = 0;
                float minScore = Float.intBitsToFloat(b[1 + 2 * slots]);
                for (int i = 1; i < slots; i++) {
                    float s = Float.intBitsToFloat(b[1 + 2 * slots + i]);
                    if (s < minScore) { minScore = s; minIdx = i; }
                }
                if (score > minScore) {
                    b[1 + minIdx] = src;
                    b[1 + slots + minIdx] = oldOrd;
                    b[1 + 2 * slots + minIdx] = Float.floatToRawIntBits(score);
                }
            }
        }

        /** Number of accumulated reverse candidates for a target; final once the target's group runs. */
        int countAt(int targetNewOrd) {
            int[] b = blocks[targetNewOrd];
            return b == null ? 0 : b[0];
        }

        /**
         * Appends the target's reverse candidates to the candidate arrays and releases the
         * target's block; returns new candSize. Safe without the offer lock: every offer for
         * this target completed before its source's group started (inter-group barrier), and
         * each target is processed exactly once, by one thread.
         */
        int appendTo(int targetNewOrd, int[] candSrc, int[] candNode, float[] candScore, int candSize) {
            int[] b = blocks[targetNewOrd];
            if (b == null) {
                return candSize;
            }
            int n = b[0];
            for (int i = 0; i < n; i++) {
                candSrc[candSize] = b[1 + i];
                candNode[candSize] = b[1 + slots + i];
                candScore[candSize] = Float.intBitsToFloat(b[1 + 2 * slots + i]);
                candSize++;
            }
            blocks[targetNewOrd] = null;
            return candSize;
        }

        long ramBytesUsed() {
            // References plus an upper bound on live blocks (16B array header + payload);
            // consumed and never-touched targets hold no block.
            long blockBytes = 16 + (long) (1 + 3 * slots) * Integer.BYTES;
            return (long) blocks.length * 8 + touchedTargets.sum() * blockBytes;
        }
    }

    /**
     * Builds the similarity-assigned ordinal mapping: one streaming pass per source computes a
     * 4-byte PQ-code prefix per live node; live nodes are then numbered in (prefix, ordinal)
     * order, source by source in ascending-size processing order — so batch processing in the
     * same order writes records sequentially and places similar vectors in adjacent records.
     * Dead nodes are numbered after all live nodes, preserving a total bijection.
     */
    private List<OrdinalMapper> buildSimilarityOrdinalMappers(ProductQuantization pq) {
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

        for (int oi = 0; oi < numSources; oi++) {
            int s = order[oi];
            OnDiskGraphIndex source = sources.get(s);
            int size = source.size(0);
            FixedBitSet alive = liveNodes.get(s);
            int[] oldToNew = new int[size];
            oldToNewPerSource[s] = oldToNew;

            // one streaming pass: 4-byte code prefix per live node, packed with the ordinal
            int liveCount = numLiveNodesPerSource.get(s);
            long[] keyed = new long[liveCount];
            java.util.concurrent.atomic.AtomicInteger fill = new java.util.concurrent.atomic.AtomicInteger();
            int window = 1 << 18;
            List<java.util.concurrent.Callable<Void>> tasks = new ArrayList<>();
            for (int from = 0; from < size; from += window) {
                final int lo = from;
                final int hi = Math.min(size, from + window);
                tasks.add(() -> {
                    source.prefetchL0Records(lo, hi - 1);
                    VectorFloat<?> vec = vectorTypeSupport.createFloatVector(dimension);
                    ByteSequence<?> code = vectorTypeSupport.createByteSequence(pq.getSubspaceCount());
                    try (var view = (OnDiskGraphIndex.View) source.getView()) {
                        for (int node = lo; node < hi; node++) {
                            if (!alive.get(node)) continue;
                            view.getVectorInto(node, vec, 0);
                            pq.encodeTo(vec, code);
                            long key = ((code.get(0) & 0xFFL) << 24) | ((code.get(1) & 0xFFL) << 16)
                                     | ((code.get(2) & 0xFFL) << 8) | (code.get(3) & 0xFFL);
                            keyed[fill.getAndIncrement()] = (key << 32) | (node & 0xFFFFFFFFL);
                        }
                    }
                    return null;
                });
            }
            joinAll(tasks);
            Arrays.parallelSort(keyed, 0, fill.get());

            for (int k = 0; k < fill.get(); k++) {
                int old = (int) keyed[k];
                oldToNew[old] = next;
                newToOldAll[next] = old;
                newToSrcAll[next] = s;
                next++;
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
        log.info("Similarity ordinals assigned: {} ordinals across {} sources in {} ms",
                next, numSources, (System.nanoTime() - t0) / 1_000_000);
        return mappers;
    }

    private void joinAll(List<java.util.concurrent.Callable<Void>> tasks) {
        try {
            for (var f : executor.invokeAll(tasks)) {
                f.get();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        } catch (java.util.concurrent.ExecutionException e) {
            throw new RuntimeException(e.getCause());
        }
    }

    private static final class Scratch implements AutoCloseable {

        final int[] candSrc, candNode;
        final float[] candScore;
        final SelectedVecCache selectedCache;
        final VectorFloat<?> tmpVec, baseVec;
        final GraphSearcher[] gs;
        final ByteSequence<?> pqCode;
        // Seeding scratch: candidate seed pool (old ordinals in the target source), the chosen top
        // seeds with their exact scores, a per-thread read channel on the output file, and a buffer
        // sized for one record's neighbor-count + list.
        final int[] seedPool = new int[SEED_POOL_CAPACITY];
        final int[] seedNodes = new int[SEEDS_PER_PARTITION];
        final float[] seedScores = new float[SEEDS_PER_PARTITION];
        // Bounded cluster search: per-target anchor (query copy, result ordinals, the anchor's
        // exact scores) and scratch for member rescoring. Reset at batch boundaries.
        final VectorFloat<?>[] clusterAnchorQuery;   // [source]; null slot = no anchor
        final boolean[] clusterAnchorValid;
        final int[][] clusterNodes;                  // [source][k+m]
        final float[][] clusterAnchorScores;         // [source][k+m]
        final int[] clusterCount;
        final float[] clusterWorstSim;     // worst exact score in the anchor list (defines ThetaD)
        final int[] clusterExtended;       // results added by resume() for the current anchor
        final float[] clusterMemberScores;           // [k+m] scratch
        final Integer[] clusterOrder;                // [k+m] scratch for sorting members
        final FileChannel outputChannel;      // null unless seeding
        final ByteBuffer seedEdgeBuf;         // null unless seeding

        /**
         * Constructs scratch space with buffers sized for the maximum expected candidates and degree.
         */
        Scratch(int maxCandidateSize, int maxDegree, int dimension, List<OnDiskGraphIndex> sources,
                ProductQuantization pq, Path seedOutputPath) {
            this.candSrc = new int[maxCandidateSize];
            this.candNode = new int[maxCandidateSize];
            this.candScore = new float[maxCandidateSize];
            this.selectedCache = new SelectedVecCache(maxDegree, dimension);
            this.tmpVec = vectorTypeSupport.createFloatVector(dimension);
            this.baseVec = vectorTypeSupport.createFloatVector(dimension);
            this.pqCode = (pq == null) ? null : vectorTypeSupport.createByteSequence(pq.getSubspaceCount());

            this.gs = new GraphSearcher[sources.size()];
            for (int i = 0; i < sources.size(); i++) {
                gs[i] = new GraphSearcher.Builder(FrontierPrefetchingView.wrap(sources.get(i))).build();
                gs[i].usePruning(false);
            }
            int clusterCap = maxCandidateSize; // >= searchTopK + CLUSTER_MARGIN
            this.clusterAnchorQuery = new VectorFloat<?>[sources.size()];
            this.clusterAnchorValid = new boolean[sources.size()];
            this.clusterNodes = new int[sources.size()][];
            this.clusterAnchorScores = new float[sources.size()][];
            this.clusterCount = new int[sources.size()];
            this.clusterWorstSim = new float[sources.size()];
            this.clusterExtended = new int[sources.size()];
            this.clusterMemberScores = new float[clusterCap];
            this.clusterOrder = new Integer[clusterCap];
            for (int i = 0; i < sources.size(); i++) {
                this.clusterAnchorQuery[i] = vectorTypeSupport.createFloatVector(dimension);
                this.clusterNodes[i] = new int[clusterCap];
                this.clusterAnchorScores[i] = new float[clusterCap];
            }

            if (seedOutputPath != null) {
                try {
                    this.outputChannel = FileChannel.open(seedOutputPath, StandardOpenOption.READ);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
                // count int + up to maxDegree neighbor ints
                this.seedEdgeBuf = ByteBuffer.allocate(Integer.BYTES * (maxDegree + 1));
            } else {
                this.outputChannel = null;
                this.seedEdgeBuf = null;
            }
        }

        /** Forgets all cluster anchors; called at batch boundaries. */
        void resetChainSeeds() {
            Arrays.fill(clusterAnchorValid, false);
            Arrays.fill(clusterCount, 0);
        }

        /**
         * Closes all graph searchers and resets the cache.
         */
        @Override
        public void close() throws IOException {
            for (var s : gs) s.close();
            selectedCache.reset();
            if (outputChannel != null) outputChannel.close();
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

        /**
         * Selects diverse neighbors from candidates using gradually increasing alpha threshold.
         * Update `selected` with the diverse members of `neighbors`.  `neighbors` is not modified
         * It assumes that the i-th neighbor with 0 {@literal <=} i {@literal <} diverseBefore is already diverse.
         */
        public void retainDiverse(int[] candSrc, int[] candNode, float[] candScore, int[] order, int orderSize, int maxDegree, SelectedVecCache selectedCache, VectorFloat<?> tmp, GraphSearcher[] gs) {
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
                    cView.getVectorInto(cNode, tmp, 0);
                    if (isDiverse(cView, cNode, tmp, cScore, currentAlpha, selectedCache)) {
                        selectedCache.add(cSrc, cView, cNode, cScore, tmp);
                        nSelected++;
                    }
                }

                currentAlpha += DIVERSITY_ALPHA_STEP;
            }
        }

        /**
         * Checks if a candidate is diverse enough by ensuring it's closer to the base node
         * than to any already-selected neighbor (scaled by alpha threshold).
         */
        private boolean isDiverse(OnDiskGraphIndex.View cView, int cNode, VectorFloat<?> cVec, float cScore, float alpha, SelectedVecCache selectedCache) {
            for (int j = 0; j < selectedCache.size; j++) {
                if (selectedCache.views[j] == cView && selectedCache.nodes[j] == cNode) {
                    return false; // already selected; don't add a duplicate
                }
                if (vsf.compare(cVec, selectedCache.vecs[j]) > cScore * alpha) {
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

        /**
         * Constructs a cache with the specified capacity and vector dimension.
         */
        SelectedVecCache(int capacity, int dimension) {
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
            size++;
        }
    }

}

