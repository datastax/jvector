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
import io.github.jbellis.jvector.util.work.ProgressLimiter;
import io.github.jbellis.jvector.util.work.WorkLimiter;
import io.github.jbellis.jvector.util.work.WorkStage;
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

/**
 * Merges multiple {@link OnDiskGraphIndex} sources into a single compacted index, preserving the
 * layer hierarchy, remapping ordinals, and — when quantization features are present — retraining
 * and re-encoding codes. Entry points are the {@code compact(...)} overloads; parallelism and
 * throttling are supplied by the caller via the executor constructor argument and
 * {@link #setProgressLimiter}.
 *
 * <p><b>Source lifecycle contract.</b> The caller owns every source's
 * {@link io.github.jbellis.jvector.disk.ReaderSupplier} and must keep the suppliers — and the
 * files and mappings underneath them — open and unchanged until the {@code compact(...)} call
 * returns or throws. That is sufficient: on every path, including batch failure and interrupt,
 * the compactor drains its in-flight work before the call unwinds, so no source access survives
 * it. Closing, truncating, or rewriting a source before then is undefined behavior; for
 * raw-unmapping suppliers (see {@link io.github.jbellis.jvector.disk.ReaderSupplier#close()}) it
 * manifests as a native JVM fault rather than an exception.</p>
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
    private final Executor executor;
    private final int taskWindowSize;
    private final VectorSimilarityFunction similarityFunction;
    private boolean refineAfterCompaction = false;

    // Embedder progress + work-admission control surface (see io.github.jbellis.jvector.util.work).
    // Default UNLIMITED = no observation and no throttling → byte-identical output and equivalent
    // timing to no SPI installed.
    private volatile ProgressLimiter limiter = ProgressLimiter.UNLIMITED;

    /**
     * The stages a compaction reports progress and acquires work admission against. Names are
     * stable so an embedder can distinguish them. The write side of {@code MERGE_LEVELS} carries the
     * graph-body IO — there is no separate flush phase.
     */
    @Experimental
    public enum Phase implements WorkStage {
        /** Merging source graphs level-by-level and writing the compacted graph body. */
        MERGE_LEVELS,
        /** Second pass refining neighborhoods in the compacted graph. */
        REFINE
    }

    /**
     * Installs an embedder control surface for progress observation and work admission. Both facets
     * are optional (see {@link ProgressLimiter}); passing {@code null} restores
     * {@link ProgressLimiter#UNLIMITED}. Returns {@code this} for chaining.
     *
     * <p>Admission ({@code acquire}) is invoked only on the orchestrating thread — the caller of
     * {@code compact} — never on a pool worker, so a blocking limiter back-pressures batch dispatch
     * without a {@code ForkJoinPool.ManagedBlocker}. The unit of {@code acquire} for this consumer
     * is bytes about to be written.
     */
    @Experimental
    public OnDiskGraphIndexCompactor setProgressLimiter(ProgressLimiter limiter) {
        this.limiter = (limiter == null) ? ProgressLimiter.UNLIMITED : limiter;
        return this;
    }

    /**
     * The compactor's task window: the number of batches kept in flight, equal to the injected
     * pool's {@code getParallelism()} (or the {@link PhysicalCoreExecutor#pool()} default's). It
     * bounds both compaction concurrency and the in-flight memory window; embedders can log or
     * assert it to confirm the parallelism they injected.
     */
    @Experimental
    public int getTaskWindowSize() {
        return taskWindowSize;
    }

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

    /**
     * Primary constructor: merges multiple graph indexes using any {@link Executor} with an
     * explicit in-flight window.
     *
     * @param sourceCompressed parallel to {@code sources}, supplying the non-fused compressed
     *                         vectors (e.g. {@link io.github.jbellis.jvector.quantization.PQVectors})
     *                         that ship alongside each graph. Pass {@code null} when sources carry
     *                         quantization inline (FUSED_PQ) or have none. Must not be combined
     *                         with sources that carry the FUSED_PQ feature.
     * @param executor runs compaction batches submitted via an internal
     *                 {@link java.util.concurrent.ExecutorCompletionService} while the calling
     *                 thread blocks for their completion. Safe to pass a {@link ForkJoinPool}
     *                 (work-stealing + managed blocking make submit-and-block safe) or a
     *                 <b>caller-runs / same-thread</b> executor (e.g. {@code Runnable::run}), which
     *                 runs each batch synchronously on the calling thread — no worker threads, no
     *                 separate pool. It is <b>not</b> safe to pass a bounded
     *                 {@code ThreadPoolExecutor} that is <i>also</i> running the calling thread: the
     *                 caller blocks in {@code take()} while its sub-tasks queue behind it, which can
     *                 thread-starvation-deadlock (worst case a single-thread pool). Embedders that
     *                 want their own bounded pool without a dedicated fan-out pool should pass a
     *                 caller-runs executor and derive parallelism from the number of <i>concurrent</i>
     *                 compactions instead. Pass {@code null} to use the shared
     *                 {@link PhysicalCoreExecutor#pool()} default. The compactor never owns or shuts
     *                 down the executor.
     * @param taskWindowSize bounds the number of in-flight batches (both concurrency and peak
     *                 write-side memory). {@code <= 0} derives it from a {@link ForkJoinPool}'s
     *                 {@code getParallelism()}, else defaults to 1 (serial).
     */
    @Experimental
    public OnDiskGraphIndexCompactor(
            List<OnDiskGraphIndex> sources,
            List<CompressedVectors> sourceCompressed,
            List<FixedBitSet> liveNodes,
            List<OrdinalMapper> remappers,
            VectorSimilarityFunction similarityFunction,
            Executor executor,
            int taskWindowSize) {
        checkBeforeCompact(sources, sourceCompressed, liveNodes, remappers);

        // Default to the shared physical-core pool. Compaction (PQ encode + parallel record flush +
        // refinement) is compute- and memory-bandwidth-bound, so sizing to logical cores
        // oversubscribes hyperthreaded hosts. This pool is process-wide and shared with index
        // construction and quantization; the compactor never owns or shuts it down.
        this.executor = (executor != null) ? executor : PhysicalCoreExecutor.pool();
        this.taskWindowSize = resolveWindow(this.executor, taskWindowSize);

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
     * Convenience {@link Executor} constructor without a non-fused compressed sidecar. Equivalent
     * to the 7-arg form with {@code sourceCompressed = null}.
     */
    @Experimental
    public OnDiskGraphIndexCompactor(
            List<OnDiskGraphIndex> sources,
            List<FixedBitSet> liveNodes,
            List<OrdinalMapper> remappers,
            VectorSimilarityFunction similarityFunction,
            Executor executor,
            int taskWindowSize) {
        this(sources, null, liveNodes, remappers, similarityFunction, executor, taskWindowSize);
    }

    private static int resolveWindow(Executor executor, int requested) {
        if (requested > 0) return requested;
        if (executor instanceof ForkJoinPool) return ((ForkJoinPool) executor).getParallelism();
        return 1;   // arbitrary Executor with no declared parallelism → serial window
    }

    /**
     * Adapts an arbitrary {@link Executor} to the {@link ExecutorService} the quantization
     * strategies need for {@code invokeAll} during PQ pre-encode. Tasks run on whatever thread
     * the executor dispatches to (the calling thread, for a caller-runs {@code Runnable::run});
     * no pool is created, and the lifecycle methods are inert since the compactor never owns the
     * executor. Always returns the {@link DrainingExecutorService} wrapper — even over a real
     * {@code ExecutorService} — because the wrapper's {@code invokeAll} must own the unwind
     * semantics; see there.
     */
    private static ExecutorService asExecutorService(Executor executor) {
        return new DrainingExecutorService(executor);
    }

    /**
     * The adapter behind {@link #asExecutorService}. Its {@code invokeAll} carries the same
     * drain-before-unwind guarantee as the batch loops ({@code awaitAbandoned}): on interrupt,
     * every submitted task is awaited before the {@code InterruptedException} propagates — so no
     * strategy fan-out can outlive {@code compact()} and keep reading source graphs (or writing
     * the pre-encode cache that {@code onAfterClose} unmaps) after the caller regains control.
     * Stock {@code invokeAll} instead cancels-with-interrupt and returns immediately, abandoning
     * running tasks, which do not poll the interrupt flag.
     */
    private static final class DrainingExecutorService extends AbstractExecutorService {
        private final Executor executor;

        DrainingExecutorService(Executor executor) {
            this.executor = executor;
        }

        @Override
        public void execute(Runnable command) {
            executor.execute(command);
        }

        @Override
        public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> tasks) throws InterruptedException {
            List<Future<T>> futures = new ArrayList<>(tasks.size());
            try {
                for (Callable<T> task : tasks) {
                    RunnableFuture<T> f = newTaskFor(task);
                    futures.add(f);
                    executor.execute(f);
                }
            } catch (Throwable t) {
                if (awaitDone(futures)) {
                    Thread.currentThread().interrupt();
                }
                throw t;
            }
            if (awaitDone(futures)) {
                // Interrupt status is consumed by the exception, matching stock invokeAll; a set
                // flag would also make the unwind's own channel work (e.g. the code-cache
                // truncate in onAfterClose) fail with ClosedByInterruptException.
                throw new InterruptedException("interrupted during invokeAll; submitted tasks were drained first");
            }
            return futures;
        }

        /**
         * Blocks until every future is done — uninterruptibly, the same hang-beats-crash choice
         * as {@code awaitAbandoned} — swallowing per-task outcomes (callers inspect the futures,
         * matching the {@code invokeAll} contract). Nothing is cancelled, deliberately:
         * {@code FutureTask.cancel} cannot distinguish queued from running (a running task's
         * state is still NEW), so cancelling would detach a future from its still-running body
         * and recreate the abandonment this wrapper exists to prevent. Queued tasks therefore
         * run to completion on the caller's executor before the unwind proceeds. Returns whether
         * an interrupt was received while waiting.
         */
        private static boolean awaitDone(List<? extends Future<?>> futures) {
            boolean interrupted = Thread.interrupted();
            for (Future<?> f : futures) {
                while (!f.isDone()) {
                    try {
                        f.get();
                    } catch (InterruptedException e) {
                        interrupted = true;
                    } catch (ExecutionException | CancellationException ignored) {
                        // the outcome stays in the future for the caller
                    }
                }
            }
            return interrupted;
        }

        // lifecycle is inert: the compactor never owns the executor
        @Override
        public void shutdown() {
        }

        @Override
        public List<Runnable> shutdownNow() {
            return Collections.emptyList();
        }

        @Override
        public boolean isShutdown() {
            return false;
        }

        @Override
        public boolean isTerminated() {
            return false;
        }

        @Override
        public boolean awaitTermination(long timeout, TimeUnit unit) {
            return false;
        }
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
     * Main compaction entry point. Merges all source indexes into a single output index at the
     * specified path, handling PQ retraining if needed, and writing header, all layers, and footer.
     * Any pre-existing file at {@code outputPath} is truncated first: the destination is wholly
     * jvector-owned, and a leftover longer file would keep its old footer at the file end, which
     * the footer-based default load would otherwise silently trust.
     * <p>
     * Source lifecycle: the caller keeps every source {@code ReaderSupplier} open until this call
     * returns or throws — which is sufficient; see the class-level contract.
     */
    @Experimental
    public void compact(Path outputPath) throws FileNotFoundException {
        compact(outputPath, 0L);
    }

    /**
     * No-copy compaction entry point: writes the compacted graph <b>into {@code outputPath} at
     * {@code startOffset}</b>, leaving any bytes in {@code [0, startOffset)} untouched. An embedder
     * that wraps the graph in its own container reserves its header by passing the header size as
     * {@code startOffset}, so jvector's body lands directly inside the container — removing the
     * temp-file-and-copy. jvector writes only {@code [startOffset, projectedSize)} and never reads
     * or clobbers the reserved prefix.
     * <p>
     * With {@code startOffset > 0} the file is opened read/write and <b>not</b> truncated, so a
     * prefix the embedder pre-wrote survives — and therefore a container longer than the new body
     * keeps its stale tail. Never load such a container footer-first through a whole-file reader
     * (the footer search trusts the file <i>end</i>): use
     * {@code OnDiskGraphIndex.load(supplier, startOffset, false)}, or a region-bounded
     * {@code ReaderSupplier} whose {@code length()} is the region end. With
     * {@code startOffset == 0} the destination is wholly jvector-owned and any pre-existing file
     * is truncated before writing, so a reused path cannot leave a stale tail — or a stale,
     * still-valid footer — behind the new graph. {@code compact(path, 0)} equals
     * {@link #compact(Path)}.
     * <p>
     * Source lifecycle: the caller keeps every source {@code ReaderSupplier} open until this call
     * returns or throws — which is sufficient; see the class-level contract.
     *
     * @param outputPath  the file to write into
     * @param startOffset the byte offset at which jvector's output begins; {@code 0} for a
     *                    standalone file
     */
    @Experimental
    public void compact(Path outputPath, long startOffset) throws FileNotFoundException {
        if (startOffset < 0) {
            throw new IllegalArgumentException("startOffset must be >= 0, got " + startOffset);
        }
        if (startOffset == 0) {
            truncateStandaloneDestination(outputPath);
        }
        QuantizationCompactionStrategy strategy = detectInlineStrategy();
        try {
            compactGraphImpl(outputPath, startOffset, strategy);
            releaseSourcesBeforeRefine(strategy);
            if (refineAfterCompaction) {
                refineCompactedGraph(outputPath, startOffset, strategy);
            }
        } finally {
            // Delayed until after refinement so refineCompactedGraph can read from the pre-encoded
            // code cache appended past the projected EOF; onAfterClose unmaps it and truncates.
            strategy.onAfterClose(outputPath);
        }
    }

    /**
     * No-copy compaction into an embedder-supplied {@link CompactionDestination}: writes the graph
     * body into the destination's container at its reserved offset, then {@code commit}s the body
     * length on success (so the embedder can finalize its footer/checksum), or — on any failure —
     * closes the target without committing (so the embedder discards the partial output). The
     * compactor writes into {@code target.file()} at {@code target.startOffset()} using its own
     * random-access + memory-mapped IO; the destination expresses <i>where</i>, not <i>how</i>.
     * <p>
     * Source lifecycle: the caller keeps every source {@code ReaderSupplier} open until this call
     * returns or throws — which is sufficient; see the class-level contract.
     *
     * @return the graph body length written, i.e. {@code size(file) - startOffset}
     */
    @Experimental
    public long compact(CompactionDestination destination) throws IOException {
        try (CompactionDestination.Target target = destination.open()) {
            Path file = target.file();
            long base = target.startOffset();
            compact(file, base);
            long bodyLength = java.nio.file.Files.size(file) - base;
            target.commit(bodyLength);
            return bodyLength;
        }
    }

    /**
     * Compaction entry point for graphs that ship a non-fused compressed sidecar (e.g.
     * {@link io.github.jbellis.jvector.quantization.PQVectors}). Writes the merged graph to
     * {@code graphPath} and the merged compressed vectors to {@code compressedPath}. Both are
     * wholly jvector-owned standalone destinations: any pre-existing file at either path is
     * truncated first.
     * <p>
     * The compressor is retrained on a balanced sample of merged source vectors, then every live
     * node is re-encoded against the new codebook. Requires that {@code sourceCompressed} was
     * supplied to the constructor.
     * <p>
     * Source lifecycle: the caller keeps every source {@code ReaderSupplier} open until this call
     * returns or throws — which is sufficient; see the class-level contract.
     */
    @Experimental
    public void compact(Path graphPath, Path compressedPath) throws FileNotFoundException {
        if (sourceCompressed == null) {
            throw new IllegalStateException(
                    "compact(graphPath, compressedPath) requires sourceCompressed to be supplied to the constructor");
        }
        Objects.requireNonNull(compressedPath, "compressedPath");

        // Both outputs are wholly jvector-owned standalone files; clear stale content the same
        // way compact(Path) does. The graph reload is footer-based and provably corruptible by a
        // stale tail, and the sidecar writer likewise opens "rw" without truncating.
        truncateStandaloneDestination(graphPath);
        truncateStandaloneDestination(compressedPath);

        // Graph compaction proceeds without fused-PQ retrain (validateCompressed forbids
        // FUSED_PQ when sourceCompressed is set), then the sidecar is written below.
        QuantizationCompactionStrategy inlineStrategy = detectInlineStrategy();
        QuantizationCompactionStrategy sidecarStrategy = detectSidecarStrategy();
        try {
            sidecarStrategy.retrain(similarityFunction);
            compactGraphImpl(graphPath, 0L, inlineStrategy);
            if (refineAfterCompaction) {
                refineCompactedGraph(graphPath, 0L, inlineStrategy);
            }
            sidecarStrategy.writeSidecar(compressedPath);
        } catch (IOException e) {
            throw new RuntimeException("Sidecar compaction failed", e);
        } finally {
            inlineStrategy.onAfterClose(graphPath);
        }
    }

    /**
     * Clears any pre-existing file at a wholly jvector-owned (standalone) destination so its
     * stale tail cannot survive past this compaction. Footer-based loads locate metadata
     * relative to the file END, so a stale-but-still-valid footer there would silently resurrect
     * the old graph's structure over the new bytes. Never applied to embedded destinations
     * ({@code startOffset > 0}), whose containers belong to the embedder.
     */
    private static void truncateStandaloneDestination(Path destination) {
        try (FileChannel ch = FileChannel.open(destination, StandardOpenOption.WRITE, StandardOpenOption.CREATE)) {
            ch.truncate(0);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to truncate pre-existing standalone destination " + destination, e);
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
        // PQ retrain runs on the compaction's own executor: forkJoin when it's a ForkJoinPool, else
        // caller-runs — so a non-pool / caller-runs merge keeps retrain on the caller's thread rather
        // than leaking to an all-core pool. The abstract executor is still wrapped for the strategies'
        // drain-safe invokeAll.
        ParallelExecutor computeExecutor = (executor instanceof ForkJoinPool)
                ? ParallelExecutor.forkJoin((ForkJoinPool) executor)
                : ParallelExecutor.callerRuns();
        return new CompactionContext(sources, sourceCompressed, liveNodes, remappers,
                dimension, maxOrdinal, asExecutorService(executor), computeExecutor, taskWindowSize);
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
    private void compactGraphImpl(Path outputPath, long startOffset, QuantizationCompactionStrategy strategy) throws FileNotFoundException {
        strategy.retrain(similarityFunction);

        boolean fusedPQEnabled = strategy.writesCodesInline();
        ProductQuantization pq = strategy.compressorAsPQ();
        boolean compressedPrecision = fusedPQEnabled;
        int maxBaseDegree = java.util.Collections.max(maxDegrees);
        io.github.jbellis.jvector.graph.disk.feature.FusedFeature outputFusedFeature =
                strategy.outputFusedFeature(maxBaseDegree);

        List<CommonHeader.LayerInfo> layerInfo = computeLayerInfoFromSources();
        int[] entryNodeSource = resolveEntryNodeSource(); // {sourceIdx, originalOrdinal}
        int entryNode = remappers.get(entryNodeSource[0]).oldToNew(entryNodeSource[1]);

        log.info("Writing compacted graph : {} total nodes, maxOrdinal={}, dimension={}, degree={}",
                numTotalNodes, maxOrdinal, dimension, maxDegrees.get(0));
        try (CompactWriter writer = new CompactWriter(outputPath, maxOrdinal, numTotalNodes, startOffset, layerInfo, entryNode, dimension, maxDegrees, outputFusedFeature)) {
            // Header has to be written first so the writer's position is past the header
            // before any strategy that mmaps past the projected end of the output runs.
            writer.writeHeader();
            strategy.onAfterHeader(writer);

            compactLevels(writer, similarityFunction, fusedPQEnabled, compressedPrecision, pq);

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
    private void refineCompactedGraph(Path outputPath, long startOffset, QuantizationCompactionStrategy strategy) {
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
        final java.nio.MappedByteBuffer codeCache = hasFusedPQ ? strategy.getCodeCache() : null;
        final int cacheCodeSize = hasFusedPQ ? strategy.getCacheCodeSize() : 0;

        try (var supplier = new SimpleReader.Supplier(outputPath);
             FileChannel fc = FileChannel.open(outputPath, StandardOpenOption.WRITE, StandardOpenOption.READ)) {

            // useFooter=false because the file's logical EOF (where the v6 footer trailer sits) is
            // before the still-attached pre-encode cache section. loadFromFooter() would seek to
            // the actual file length and read garbage as the magic.
            OnDiskGraphIndex mergedGraph = OnDiskGraphIndex.load(supplier, startOffset, false);

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
            final java.nio.MappedByteBuffer cache = codeCache;
            final int cacheSz = cacheCodeSize;
            final OnDiskGraphIndex graphRef = mergedGraph;

            log.info("Refining {} live nodes at level {} (hierarchy maxLevel={}, fusedPQ={}, codeCache={})",
                    total, iterationLevel, mergedGraph.getMaxLevel(), fpq, cache != null);

            // Windowed submit/drain (mirrors runBatchesWithBackpressure) so a blocking WorkLimiter —
            // a rate limiter, or a semaphore that releases permits on Grant.close() — is
            // deadlock-free: admission is on the orchestrator before each submit, and exactly one
            // grant is released per completed batch. This also bounds in-flight refine batches (and
            // thus peak memory) to taskWindowSize, which the prior submit-all loop did not. Amount is
            // an estimate of the per-node record bytes rewritten. UNLIMITED makes it all no-ops.
            final long refinedRecordSize = (long) dimension * Float.BYTES
                    + (long) baseDegree * Integer.BYTES + codeSize;
            final int nBatches = (total + batchSize - 1) / batchSize;
            java.util.ArrayDeque<WorkLimiter.Grant> grants = new java.util.ArrayDeque<>();

            int nextStart = 0, inFlight = 0, completed = 0, nodesDone = 0;
            int progressStep = Math.max(1, total / 10);
            int nextProgress = progressStep;
            try {
                while (completed < nBatches) {
                    while (inFlight < taskWindowSize && nextStart < total) {
                        final int s = nextStart;
                        final int e = Math.min(nextStart + batchSize, total);
                        nextStart = e;
                        grants.add(limiter.acquire((long) (e - s) * refinedRecordSize)); // may park the orchestrator
                        ecs.submit(() -> {
                            RefineScratch scratch = tls.get();
                            for (int i = s; i < e; i++) {
                                int node = ords[i];
                                refineOneNode(node, scratch, fc, baseDegree, fpq, codeSize, cmp, bw,
                                        graphRef, cache, cacheSz);
                            }
                            return e - s;
                        });
                        inFlight++;
                    }
                    Future<Integer> finished = ecs.take();
                    inFlight--;   // finished, whatever its outcome — get() below may throw
                    nodesDone += finished.get();
                    completed++;
                    WorkLimiter.Grant g = grants.poll();
                    if (g != null) g.close();
                    limiter.onProgress(Phase.REFINE, nodesDone, total);
                    if (nodesDone >= nextProgress) {
                        log.info("Refinement progress: {}/{} nodes", nodesDone, total);
                        nextProgress += progressStep;
                    }
                }
            } catch (Throwable t) {
                // Same drain-before-unwind as runBatchesWithBackpressure: the enclosing
                // try-with-resources releases the output supplier, the channel, and (for fused
                // strategies) the pre-encode cache as this exception propagates, which must not
                // happen underneath still-running refine batches.
                awaitAbandoned(ecs, inFlight);
                throw t;
            } finally {
                for (WorkLimiter.Grant g : grants) g.close();
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
                               java.nio.MappedByteBuffer codeCache,
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
                // vector read AND one PQ encode per selected neighbor. duplicate() gives this
                // call its own position cursor without racing other workers.
                ByteBuffer cacheView = codeCache.duplicate();
                byte[] codeBuf = scratch.pqCodeBytes;
                for (int k = 0; k < selectedSize; k++) {
                    int newOrd = scratch.selectedNodes[k];
                    cacheView.position(newOrd * cacheCodeSize);
                    cacheView.get(codeBuf, 0, cacheCodeSize);
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
        int baseMaxCandidateSize = baseSearchTopK * (sources.size() - 1) + maxDegrees.get(0);
        int upperMaxPerSourceTopK = maxUpperDegree == 0 ? 0 : Math.max(MIN_SEARCH_TOP_K, ((maxUpperDegree + sources.size() - 1) / sources.size()) * SEARCH_TOP_K_MULTIPLIER);
        int upperMaxCandidateSize = upperMaxPerSourceTopK * sources.size();
        int maxCandidateSize = Math.max(baseMaxCandidateSize, upperMaxCandidateSize);
        int scratchDegree = Math.max(maxDegrees.get(0), Math.max(1, maxUpperDegree));
        final ThreadLocal<Scratch> threadLocalScratch = ThreadLocal.withInitial(() ->
            new Scratch(maxCandidateSize, scratchDegree, dimension, sources, pq)
        );

        // MERGE_LEVELS progress denominator: base-layer live nodes. Level 0 (the bulk) runs first,
        // so progress[0] climbs 0 -> numTotalNodes across it, then holds while the small upper tail
        // runs (completed is clamped to the total). progress persists across all level calls.
        long[] mergeProgress = { 0L, numTotalNodes };

        for (int level = 0; level < maxDegrees.size(); level++) {
            List<BatchSpec> batches = buildBatches(level);
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

                    runBatchesWithBackpressure(
                            batches,
                            ecs,
                            submitOne,
                            (results) -> {
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
                            },
                            Phase.MERGE_LEVELS,
                            (results) -> {                       // exact bytes: read before the write consumes the buffers
                                long s = 0;
                                for (WriteResult r : results) s += r.data.remaining();
                                return s;
                            },
                            mergeProgress
                    );
                }

                writer.offsetAfterInline();

            } else {
                final int lvl = level;
                log.info("Compacting upper layer {}", level);

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
                        },
                        Phase.MERGE_LEVELS,
                        (results) -> {                           // estimate: neighbor ids + optional PQ code per node
                            long s = 0;
                            for (UpperLayerWriteResult r : results)
                                s += (long) r.neighbors.length * Integer.BYTES + (r.pqCode == null ? 0 : r.pqCode.length());
                            return s;
                        },
                        mergeProgress
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
            var source = sources.get(s);
            if (level > source.getMaxLevel()) continue;
            NodesIterator sourceNodes = source.getNodes(level);
            int numNodes = sourceNodes.size();
            int[] nodes = new int[numNodes];
            int i = 0;
            while (sourceNodes.hasNext()) {
                nodes[i++] = sourceNodes.next();
            }

            int numBatches = max(TARGET_BATCHES_PER_SOURCE, (numNodes + TARGET_NODES_PER_BATCH - 1) / TARGET_NODES_PER_BATCH);
            if (numBatches > numNodes) numBatches = numNodes;
            int batchSize = (numNodes + numBatches - 1) / numBatches;
            for (int b = 0; b < numBatches; ++b) {
                int start = min(numNodes, batchSize * b);
                int end = min(numNodes, batchSize * (b + 1));
                batches.add(new BatchSpec(s, nodes, start, end));
            }
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
            // node's own record read (adjacency + vector) is fully predictable.
            sources.get(bs.sourceIdx).prefetchL0Records(bs.nodes[bs.start], bs.nodes[bs.end - 1]);
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

        return writer.writeInlineNodeRecord(
                newOrdinal,
                scratch.baseVec,
                selected,
                scratch.pqCode
        );
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
                candSize = gatherFromOtherSource(node, level, ss, searchView, indexAlive,
                                                  baseVec, scratch, candSize, params);
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
    private int gatherFromOtherSource(int node, int level, int sourceIdx,
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
            // rerankK = searchTopK, not beamWidth: the wider beam's extra candidates are largely
            // pruned by diversity selection, so the doubled approximate-phase cost buys almost
            // no recall.
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
            java.util.function.Consumer<List<T>> onComplete,
            WorkStage stage,
            java.util.function.ToLongFunction<List<T>> batchBytes,
            long[] progress
    ) throws InterruptedException, ExecutionException {

        final int total = batches.size();
        int nextToSubmit = 0;
        int inFlight = 0;
        int completed = 0;

        try {
            // initial window
            while (inFlight < taskWindowSize && nextToSubmit < total) {
                submitOne.accept(batches.get(nextToSubmit++));
                inFlight++;
            }

            while (completed < total) {
                Future<List<T>> finished = ecs.take();
                inFlight--;   // finished, whatever its outcome — get() below may throw
                List<T> results = finished.get();

                // Admission runs on this (orchestrating) thread, before the write. A blocking limiter
                // back-pressures dispatch/consume while in-flight workers keep computing — no
                // ManagedBlocker. The amount is the bytes this batch is about to write, read here before
                // onComplete consumes the buffers. For the default UNLIMITED limiter both calls are no-ops.
                long amount = batchBytes.applyAsLong(results);
                try (WorkLimiter.Grant g = limiter.acquire(amount)) {
                    onComplete.accept(results);
                }

                completed++;

                progress[0] = Math.min(progress[0] + results.size(), progress[1]);
                limiter.onProgress(stage, progress[0], progress[1]);

                if (nextToSubmit < total) {
                    submitOne.accept(batches.get(nextToSubmit++));
                    inFlight++;
                }
                if (completed % 10 == 0) {
                    log.debug("Compaction I/O progress: {}/{} batches written to disk", completed, total);
                }
            }
        } catch (Throwable t) {
            // A failed batch or an interrupt must not unwind while other batches are still
            // running: the caller may release the source graphs the moment compact() returns or
            // throws, and a read abandoned mid-flight against a since-unmapped source faults
            // natively instead of throwing. Block until every submitted batch has finished, then
            // let the original failure propagate.
            awaitAbandoned(ecs, inFlight);
            throw t;
        }
    }

    /**
     * Blocks until {@code remaining} already-submitted batches have finished, discarding their
     * results. Runs on the orchestrating thread during an exceptional unwind, before the
     * exception can reach any scope that releases resources the batches still read (the caller's
     * source graphs, the output supplier and channel, the pre-encode cache): a batch abandoned
     * mid-read whose mapping is then closed faults natively instead of throwing. The wait is
     * deliberately unbounded — a hung batch leaves a thread-dumpable hang, which is strictly
     * better than the use-after-unmap crash a timeout would reintroduce; batches never wait on
     * the orchestrator, so the drain cannot deadlock. An interrupt received while draining is
     * remembered and re-asserted on exit, never dropped.
     */
    private static void awaitAbandoned(ExecutorCompletionService<?> ecs, int remaining) {
        boolean interrupted = Thread.interrupted();
        int drained = 0;
        while (drained < remaining) {
            try {
                Future<?> finished = ecs.take();
                drained++;
                try {
                    finished.get();
                } catch (ExecutionException e) {
                    log.debug("Discarding abandoned batch failure during compaction unwind", e.getCause());
                } catch (InterruptedException e) {
                    interrupted = true;
                } catch (CancellationException e) {
                    // nothing cancels these futures today; tolerate rather than mask the unwind
                }
            } catch (InterruptedException e) {
                interrupted = true;
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
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
        int baseMaxCandidateSize = baseSearchTopK * (sources.size() - 1) + maxDegrees.get(0);
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
    private static final class Scratch implements AutoCloseable {

        final int[] candSrc, candNode;
        final float[] candScore;
        final SelectedVecCache selectedCache;
        final VectorFloat<?> tmpVec, baseVec;
        final GraphSearcher[] gs;
        final ByteSequence<?> pqCode;

        /**
         * Constructs scratch space with buffers sized for the maximum expected candidates and degree.
         */
        Scratch(int maxCandidateSize, int maxDegree, int dimension, List<OnDiskGraphIndex> sources, ProductQuantization pq) {
            this.candSrc = new int[maxCandidateSize];
            this.candNode = new int[maxCandidateSize];
            this.candScore = new float[maxCandidateSize];
            this.selectedCache = new SelectedVecCache(maxDegree, dimension);
            this.tmpVec = vectorTypeSupport.createFloatVector(dimension);
            this.baseVec = vectorTypeSupport.createFloatVector(dimension);
            this.pqCode = (pq == null) ? null : vectorTypeSupport.createByteSequence(pq.getSubspaceCount());

            this.gs = new GraphSearcher[sources.size()];
            for (int i = 0; i < sources.size(); i++) {
                gs[i] = new GraphSearcher(sources.get(i));
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

