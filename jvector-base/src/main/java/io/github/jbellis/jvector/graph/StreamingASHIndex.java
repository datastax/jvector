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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.PreparedQuery;
import io.github.jbellis.jvector.graph.similarity.QueryPreparer;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.ASHQueryProjection;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.MutableASHVectors;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.DocIdSetIterator;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.util.OrdinalReservoir;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.util.ThreadSafeGrowableBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * An in-memory graph index that is built and searched while vectors stream in, holding no copy of the vectors.
 *
 * <p>The caller owns the vectors and supplies them through a {@link RandomAccessVectorValues} that can read any
 * ordinal it has added. The index keeps only the graph and an ASH code per vector: the graph is built and searched
 * with ASH scoring, and the best candidates of each search are reranked with exact scores read from the caller's
 * vectors.</p>
 *
 * <p><b>Models.</b> An index starts with a model the caller passes in, typically one trained on an earlier index
 * over the same kind of data, or cold. A cold index first builds and searches its graph with exact scores read from
 * the caller's vectors, as an uncompressed index does, while it keeps a {@link OrdinalReservoir uniform sample} of
 * ordinals. With {@link Options#train} it trains its first model on that sample once {@link Options#trainAfter}
 * ordinals have been offered; without it, it fixes a random-projection model, which needs no training, after
 * {@link Options#coldStartSize}. It then encodes the vectors it holds with that model, rescores the graph's edges with
 * ASH scores and continues with ASH scoring. Later, with {@link Options#train}, it trains newer models when due (see
 * {@link Options}) and moves to them: new pages of codes are encoded with the new model at once, and existing pages
 * are re-encoded from the caller's vectors and replaced one at a time (see {@link MutableASHVectors}). The graph is
 * never rebuilt.</p>
 *
 * <p><b>Cost.</b> Every operation's cost is bounded. A cold index's first model is fixed while inserts wait, once:
 * encoding the vectors it holds and rescoring their edges. An insert encodes its vector, inserts it into the graph and
 * re-encodes at most {@link Options#reencodePerInsert} earlier vectors for a model change in progress, skipping that
 * work rather than waiting when another insert is doing it. Training runs on the library's physical-core pool. A search
 * prepares its query once per model in use (two during a model change) and is never blocked.</p>
 *
 * <p><b>Concurrency.</b> {@link #add}, {@link #markDeleted} and {@link #search} may be called from many threads.
 * Each ordinal is added once, and its vector must be readable from the caller's vectors from before the call until
 * the index is discarded; for a deleted ordinal the caller may instead return null, and the vector then gets an
 * all-zero code if its page is re-encoded. Ordinals should be dense. A search sees every vector whose {@code add}
 * returned before the search began, and may see vectors being added concurrently.</p>
 *
 * <p>Similarity: ASH scoring supports dot product, which is also cosine similarity for unit-length vectors.</p>
 */
@Experimental
public final class StreamingASHIndex implements Accountable {
    /** Settings for a streaming index; the defaults suit most data. */
    public static final class Options {
        /**
         * Vectors a cold index scores exactly before it fixes a random-projection model, when it does not train; also
         * the smallest sample a first model is trained on, below which it uses a random projection.
         */
        public int coldStartSize = 1024;
        /** Ordinals sampled for training; 0 means 20 times the dimension, the sample ASH training uses. */
        public int reservoirSize = 0;
        /**
         * Whether the index trains a model on its sample when one is due (see {@link #trainAfter} and
         * {@link #retrainGrowth}) and moves to it. Training runs on the library's physical-core pool, as all ASH
         * training does. Without it, the index trains only when the caller calls {@link #retrain}.
         */
        public boolean train = false;
        /** The optimizer of trained models. */
        public int optimizer = AsymmetricHashing.ITQ;
        /** Offered ordinals before a cold index trains its first model; 0 means the reservoir size. */
        public long trainAfter = 0;
        /**
         * Retrain when the index has grown by this factor since its model was trained; 0 never retrains. An inherited
         * model counts as trained at {@code trainAfter} vectors.
         */
        public double retrainGrowth = 0;
        /** Earlier vectors an insert re-encodes, at most, while the index moves to a newer model. */
        public int reencodePerInsert = 2;
        /**
         * How models the index trains project queries ({@link AsymmetricHashing#setQueryOptions}); null keeps the
         * current model's setting, or for the first model the {@code jvector.ash.queryProjection} default.
         */
        public ASHQueryProjection queryProjection = null;
        /**
         * The largest group of concurrent queries a trained model projects together; 0 keeps the current model's
         * setting, or for the first model the {@code jvector.ash.queryBatchSize} default.
         */
        public int queryBatchSize = 0;
        /**
         * Inserts a cold index runs at once while it trains its first model, so the training, on the physical-core
         * pool, gets the CPU it needs even when inserts would use every core; others wait for a slot. 0 means half
         * the available processors, and at least 1.
         */
        public int insertsWhileTraining = 0;
    }

    private final Supplier<RandomAccessVectorValues> vectors;
    private final RandomAccessVectorValues source;
    private final int dimension;
    private final VectorSimilarityFunction similarity;
    private final Function<BuildScoreProvider, GraphIndexBuilder> builders;
    private final int encodedBits;
    private final int bitsPerDimension;
    private final Options options;
    private final long trainAfter;
    private final OrdinalReservoir reservoir;

    /** Null until the index has a model; published once, before {@link #exact} is cleared. */
    private volatile Live live;
    /** A cold index's graph with exact scores, until the index has a model; then null. */
    private volatile Exact exact;
    /** Held by inserts and deletes into the exact graph; held exclusively while the first model is fixed. */
    private final ReentrantReadWriteLock firstModel =
            new ReentrantReadWriteLock();
    /** The mean of the vectors added, kept as they arrive; see {@link #mean}. */
    private final VectorMean.Running mean;
    /** How long fixing the first model held inserts, in nanoseconds; 0 until then. */
    private volatile long firstModelPauseNanos;

    // Memory allocated outside inserts (the sample, models, and the codes the first model gives the vectors already
    // held) is reported on the insert path, heap through add() and native through takeNativeBytes(). Guarded by
    // itself; the charge is re-checked when the models change and on about one insert in 1024.
    private final Object charges = new Object();
    private long reportedHeap;
    private long reportedNative;
    private long firstModelCodeBytes;
    private volatile boolean chargesDue = true;
    private final java.util.concurrent.atomic.AtomicLong pendingNative = new java.util.concurrent.atomic.AtomicLong();
    /** Insert slots while the first model trains; see {@link Options#insertsWhileTraining}. */
    private final Semaphore trainingSlots;

    private final AtomicBoolean training = new AtomicBoolean();
    /** Trainings run one at a time across the JVM's indexes, each using the whole physical-core pool. */
    private static final ReentrantLock TRAINING = new ReentrantLock();
    /** Vectors in the index when its current model was trained or inherited, for the growth rule. */
    private volatile long trainedAt;
    /** True while the current model is a cold start's random projection, which the first training replaces. */
    private volatile boolean provisional;

    // Model change in progress, guarded by migration.
    private final ReentrantLock migration = new ReentrantLock();
    private volatile int target = -1;
    private MutableASHVectors.PageRebuild rebuilding;
    private int scanFrom;
    /** After a scan finds no page to replace, the vector count at which to scan again (the last page fills). */
    private int nextScanAt;

    /** The parts of a live index, published together. */
    private static final class Live {
        final MutableASHVectors codes;
        final BuildScoreProvider scores;
        final GraphIndexBuilder builder;
        final ExplicitThreadLocal<GraphSearcher> searchers;

        Live(MutableASHVectors codes, BuildScoreProvider scores, GraphIndexBuilder builder) {
            this.codes = codes;
            this.scores = scores;
            this.builder = builder;
            this.searchers = ExplicitThreadLocal.withInitial(() -> new GraphSearcher(builder.getGraph()));
        }
    }

    /** The parts of a cold index before its first model. */
    private static final class Exact {
        final GraphIndexBuilder builder;
        final ExplicitThreadLocal<GraphSearcher> searchers;
        /** Ordinals added, including those with no vector, which get the all-zero code. */
        final ThreadSafeGrowableBitSet added =
                new ThreadSafeGrowableBitSet(1024);

        Exact(GraphIndexBuilder builder) {
            this.builder = builder;
            this.searchers = ExplicitThreadLocal.withInitial(() -> new GraphSearcher(builder.getGraph()));
        }
    }

    private StreamingASHIndex(RandomAccessVectorValues vectors, VectorSimilarityFunction similarity,
                              Function<BuildScoreProvider, GraphIndexBuilder> builders, AsymmetricHashing model,
                              int encodedBits, int bitsPerDimension, Options options) {
        Objects.requireNonNull(vectors, "vectors");
        if (similarity != VectorSimilarityFunction.DOT_PRODUCT)
            throw new IllegalArgumentException("ASH streaming indexes support DOT_PRODUCT similarity");
        this.options = Objects.requireNonNull(options, "options");
        this.source = vectors;
        this.vectors = vectors.threadLocalSupplier();
        this.dimension = vectors.dimension();
        this.similarity = similarity;
        this.builders = Objects.requireNonNull(builders, "builders");
        this.encodedBits = encodedBits;
        this.bitsPerDimension = bitsPerDimension;
        int reservoirSize = options.reservoirSize > 0 ? options.reservoirSize : 20 * dimension;
        this.reservoir = new OrdinalReservoir(reservoirSize);
        this.trainAfter = options.trainAfter > 0 ? options.trainAfter : reservoirSize;
        this.mean = new VectorMean.Running(dimension);
        this.trainingSlots = new Semaphore(options.insertsWhileTraining > 0
                ? options.insertsWhileTraining
                : Math.max(1, Runtime.getRuntime().availableProcessors() / 2));
        if (model != null) {
            if (model.originalDimension != dimension) throw new IllegalArgumentException("Model dimension does not match the vectors");
            trainedAt = trainAfter;
            live = startLive(model);
        } else {
            if (options.coldStartSize < 1) throw new IllegalArgumentException("coldStartSize must be positive");
            exact = new Exact(builders.apply(BuildScoreProvider.randomAccessScoreProvider(source, similarity)));
        }
    }

    /**
     * An index that encodes vectors with {@code model} until it trains a newer one.
     *
     * @param vectors  the caller's vectors, read by ordinal; see the class comment
     * @param builders creates the graph builder from the index's score provider, with the caller's graph settings
     *                 (degree, beam width, hierarchy, executors)
     */
    public static StreamingASHIndex withModel(RandomAccessVectorValues vectors, VectorSimilarityFunction similarity,
                                              AsymmetricHashing model,
                                              Function<BuildScoreProvider, GraphIndexBuilder> builders,
                                              Options options) {
        Objects.requireNonNull(model, "model");
        return new StreamingASHIndex(vectors, similarity, builders, model, model.encodedBits, model.bitsPerDimension,
                options);
    }

    /**
     * An index that starts cold; see the class comment.
     *
     * @param encodedBits      ASH code size, including the header, for every model this index uses
     * @param bitsPerDimension stored bits per projected dimension
     */
    public static StreamingASHIndex coldStart(RandomAccessVectorValues vectors, VectorSimilarityFunction similarity,
                                              int encodedBits, int bitsPerDimension,
                                              Function<BuildScoreProvider, GraphIndexBuilder> builders,
                                              Options options) {
        return new StreamingASHIndex(vectors, similarity, builders, null, encodedBits, bitsPerDimension, options);
    }

    private Live startLive(AsymmetricHashing model) {
        var codes = new MutableASHVectors(model);
        var scores = codes.buildScoreProvider(source);
        return new Live(codes, scores, builders.apply(scores));
    }

    // ---------------------------------------------------------------------------------------------
    // Adding and deleting

    /**
     * Adds the caller's vector at {@code ordinal}; see the class comment.
     *
     * @return the memory the insert added to the index: its graph node and its code. Fixing a cold index's first
     *         model encodes the vectors it holds at once, counted by {@link #ramBytesUsed} but not here.
     */
    public long add(int ordinal) {
        Live l = live;
        if (l == null) {
            // While the first model trains, inserts take turns so the training gets the CPU.
            boolean slot = training.get();
            if (slot) trainingSlots.acquireUninterruptibly();
            long bytes = 0;
            var lock = firstModel.readLock();
            lock.lock();
            try {
                l = live;
                if (l == null) {
                    Exact e = exact;
                    e.added.set(ordinal);
                    reservoir.offer(ordinal);
                    VectorFloat<?> v = vectors.get().getVector(ordinal);
                    if (v != null) {
                        mean.accept(v);
                        bytes = e.builder.addGraphNode(ordinal, v);
                    }
                }
            } finally {
                lock.unlock();
                if (slot) trainingSlots.release();
            }
            if (l == null) {
                afterExactInsert();
                return bytes + chargeOutsideInserts();
            }
        }
        long bytes = insert(l, ordinal);
        afterInsert(l);
        return bytes + chargeOutsideInserts();
    }

    /**
     * The heap allocated outside inserts since it was last reported, also moving native memory allocated since then to
     * {@link #takeNativeBytes}. Cheap unless a recount is due: when the models change, and on about one insert in 1024,
     * which catches projections a backend allocates when first used.
     */
    private long chargeOutsideInserts() {
        if (!chargesDue && (java.util.concurrent.ThreadLocalRandom.current().nextInt() & 1023) != 0) return 0;
        synchronized (charges) {
            chargesDue = false;
            long heap = RamUsageEstimator.shallowSizeOfInstance(StreamingASHIndex.class) + reservoir.ramBytesUsed()
                    + firstModelCodeBytes;
            long nativeBytes = 0;
            Live l = live;
            if (l != null) {
                heap += l.codes.modelHeapBytes();
                nativeBytes = l.codes.nativeBytesUsed();
            }
            long heapDelta = Math.max(0, heap - reportedHeap);
            reportedHeap += heapDelta;
            long nativeDelta = Math.max(0, nativeBytes - reportedNative);
            reportedNative += nativeDelta;
            if (nativeDelta > 0) pendingNative.addAndGet(nativeDelta);
            return heapDelta;
        }
    }

    /**
     * Native memory the index allocated, such as a model's native projections, since this was last called; the
     * caller charges it to whatever accounts for its off-heap memory. Heap is reported through {@link #add} instead.
     */
    public long takeNativeBytes() {
        return pendingNative.getAndSet(0);
    }

    /** Fixes the first model on the physical-core pool once it is due. */
    private void afterExactInsert() {
        long due = options.train ? trainAfter : options.coldStartSize;
        if (reservoir.offered() < due || !training.compareAndSet(false, true)) return;
        try {
            PhysicalCoreExecutor.pool().execute(() -> {
                try {
                    fixFirstModel();
                } finally {
                    training.set(false);
                }
            });
        } catch (RuntimeException e) {
            training.set(false);
            throw e;
        }
    }

    /**
     * Trains (or, without {@link Options#train} or on a small sample, fixes a random projection for) the first model,
     * then, while inserts wait, encodes every vector the index holds with it and rescores the graph's edges with
     * ASH scores. Searches keep using the exact graph until the index publishes the rescored one.
     */
    private void fixFirstModel() {
        if (live != null) return;
        int[] sample = reservoir.sample();
        if (sample.length == 0) return;
        boolean trained = options.train && sample.length >= options.coldStartSize;
        AsymmetricHashing model = trainNextModel(trained ? options.optimizer : AsymmetricHashing.RANDOM, null);
        configureQueries(model, null);
        var lock = firstModel.writeLock();
        lock.lock();
        long pauseStart = System.nanoTime();
        try {
            if (live != null) return;
            Exact e = exact;
            var codes = new MutableASHVectors(model);
            var added = e.added;
            int upper = added.length();
            PhysicalCoreExecutor.pool().submit(() -> java.util.stream.IntStream.range(0, upper).parallel()
                    .filter(added::get).forEach(ordinal -> {
                        VectorFloat<?> v = vectors.get().getVector(ordinal);
                        if (v == null) codes.setZero(ordinal);
                        else codes.encodeAndSet(ordinal, v);
                    })).join();
            var scores = codes.buildScoreProvider(source);
            var builder = GraphIndexBuilder.rescore(e.builder, scores);
            var deleted = ((OnHeapGraphIndex) e.builder.getGraph()).getDeletedNodes();
            for (int i = deleted.nextSetBit(0); i != DocIdSetIterator.NO_MORE_DOCS;
                 i = i + 1 < deleted.length() ? deleted.nextSetBit(i + 1) : DocIdSetIterator.NO_MORE_DOCS)
                builder.markNodeDeleted(i);
            trainedAt = codes.count();
            provisional = options.optimizer != AsymmetricHashing.RANDOM
                    && (!trained || options.optimizer != model.optimizer);
            synchronized (charges) {
                firstModelCodeBytes = (long) added.cardinality() * codes.bytesPerVector();
            }
            live = new Live(codes, scores, builder);
            exact = null;
            chargesDue = true;
            firstModelPauseNanos = System.nanoTime() - pauseStart;
        } finally {
            lock.unlock();
        }
    }

    private long insert(Live l, int ordinal) {
        VectorFloat<?> v = vectors.get().getVector(ordinal);
        if (v == null) l.codes.setZero(ordinal);
        else {
            mean.accept(v);
            l.codes.encodeAndSet(ordinal, v);
        }
        reservoir.offer(ordinal);
        return l.builder.addGraphNode(ordinal, l.scores.searchProviderFor(ordinal)) + l.codes.bytesPerVector();
    }

    /**
     * The mean of every vector added, kept as vectors arrive, so writing the index needs no pass over the vectors
     * to center them (as NVQ does); deleted vectors stay in it. Null before any vector with a value was added.
     */
    public VectorFloat<?> mean() {
        return mean.mean();
    }

    /**
     * Marks {@code ordinal} deleted: searches no longer return it, and graph cleanup removes it. Its code stays until
     * then, since searches may still pass through it.
     */
    public void markDeleted(int ordinal) {
        Live l = live;
        if (l == null) {
            var lock = firstModel.readLock();
            lock.lock();
            try {
                l = live;
                if (l == null) {
                    exact.builder.markNodeDeleted(ordinal);
                    return;
                }
            } finally {
                lock.unlock();
            }
        }
        l.builder.markNodeDeleted(ordinal);
    }

    // ---------------------------------------------------------------------------------------------
    // Training and model changes

    private void afterInsert(Live l) {
        if (options.train && isTrainingDue(l) && training.compareAndSet(false, true)) {
            try {
                PhysicalCoreExecutor.pool().execute(() -> {
                    try {
                        retrain();
                    } finally {
                        training.set(false);
                    }
                });
            } catch (RuntimeException e) {
                training.set(false);
                throw e;
            }
        }
        if (target >= 0 && options.reencodePerInsert > 0) reencode(l, options.reencodePerInsert, false);
    }

    private boolean isTrainingDue(Live l) {
        if (target >= 0) return false; // finish one model change before starting another
        if (provisional) return reservoir.offered() >= trainAfter;
        return options.retrainGrowth > 0 && l.codes.count() >= options.retrainGrowth * Math.max(1, trainedAt);
    }

    /**
     * Trains a model on the sampled ordinals and starts moving the index to it: pages created from now on use it,
     * and inserts re-encode earlier pages. Trains as {@link #trainNextModel} does.
     *
     * @return the new model, or null when the index has no model yet
     */
    public AsymmetricHashing retrain() {
        Live l = live;
        if (l == null) return null;
        AsymmetricHashing model = trainNextModel(options.optimizer, null);
        if (model == null) return null;
        configureQueries(model, l.codes.getCompressor());
        migration.lock();
        try {
            int id = l.codes.addModel(model);
            l.codes.useModel(id);
            chargesDue = true;
            trainedAt = l.codes.count();
            provisional = false;
            rebuilding = null;
            scanFrom = 0;
            nextScanAt = 0;
            target = id;
            // Keep the time with two models short: while both are in use, scoring across them projects source
            // vectors, which slows inserts. Every complete page moves now, in parallel on the physical-core pool,
            // as other ASH encoding does; inserts move only the pages that were still filling.
            reencodeCompletePages(l, id);
        } finally {
            migration.unlock();
        }
        return model;
    }

    /** Gives a model the index trained the query settings of the options, else of the model it replaces. */
    private void configureQueries(AsymmetricHashing model, AsymmetricHashing current) {
        var precision = options.queryProjection != null ? options.queryProjection
                : current != null ? current.getQueryProjection() : null;
        int batch = options.queryBatchSize > 0 ? options.queryBatchSize
                : current != null ? current.getQueryBatchSize() : 0;
        if (precision != null || batch > 0)
            model.setQueryOptions(precision != null ? precision : model.getQueryProjection(),
                    batch > 0 ? batch : model.getQueryBatchSize());
    }

    /** Re-encodes every complete page not on model {@code id}, in parallel; called with the migration lock held. */
    private void reencodeCompletePages(Live l, int id) {
        var codes = l.codes;
        var pages = new ArrayList<MutableASHVectors.PageRebuild>();
        for (int p = 0; p < codes.pageCount(); p++) {
            var rebuild = codes.rebuild(p, id);
            if (rebuild != null) pages.add(rebuild);
        }
        if (pages.isEmpty()) return;
        PhysicalCoreExecutor.pool().submit(() -> pages.parallelStream().forEach(rebuild -> {
            rebuild.encode(codes.vectorsPerPage(), vectors.get());
            rebuild.publish();
        })).join();
    }

    /**
     * Re-encodes up to {@code max} vectors for the model change in progress. Never waits: another thread doing the
     * same work makes this a no-op.
     *
     * @param quiesced true when no vectors are being added, so the last, incomplete page may be replaced too
     * @return true if the index is no longer changing models
     */
    private boolean reencode(Live l, int max, boolean quiesced) {
        if (!(quiesced ? lock() : migration.tryLock())) return target < 0;
        try {
            int id = target;
            if (id < 0) return true;
            var view = vectors.get();
            int left = max;
            while (left > 0) {
                if (rebuilding == null) {
                    if (!quiesced && l.codes.count() < nextScanAt) return false;
                    rebuilding = nextRebuild(l.codes, id, quiesced);
                    if (rebuilding == null) {
                        int[] inUse = l.codes.modelsInUse();
                        if (inUse.length == 1 && inUse[0] == id) target = -1;
                        // Only incomplete pages remain on older models; look again once another page has filled.
                        else nextScanAt = l.codes.count() + l.codes.vectorsPerPage();
                        return target < 0;
                    }
                }
                left -= rebuilding.encode(left, view);
                if (rebuilding.isFull()) {
                    rebuilding.publish();
                    rebuilding = null;
                }
            }
            return false;
        } finally {
            migration.unlock();
        }
    }

    private boolean lock() {
        migration.lock();
        return true;
    }

    /** The next page not yet on model {@code id}, scanning round the pages from where the last scan stopped. */
    private MutableASHVectors.PageRebuild nextRebuild(MutableASHVectors codes, int id, boolean quiesced) {
        int pages = codes.pageCount();
        for (int k = 0; k < pages; k++) {
            int index = (scanFrom + k) % pages;
            var rebuild = quiesced ? codes.rebuildQuiesced(index, id) : codes.rebuild(index, id);
            if (rebuild != null) {
                scanFrom = index + 1;
                return rebuild;
            }
        }
        return null;
    }

    /**
     * Readies the index to be written: every vector ends encoded with a trained model, since a written index can be
     * searched for a long time. A training in progress is waited for; a cold index still scoring exactly fixes its
     * first model now, on whatever sample it has; an index on a random projection it did not choose trains its model
     * now; then every page not on the newest model is re-encoded, in parallel. Call only while no vectors are being
     * added.
     */
    public void prepareForFlush() {
        while (training.get()) LockSupport.parkNanos(1_000_000);
        if (live == null) {
            fixFirstModel();
            if (live == null) return; // nothing was added
        }
        if (provisional) retrain();
        finishModelChange();
    }

    /**
     * Finishes a model change in progress, including the last page, so the index has one model again, as writing it
     * requires. Pages are re-encoded in parallel on the library's physical-core pool, as other ASH encoding is.
     * Call only while no vectors are being added.
     */
    public void finishModelChange() {
        Live l = live;
        if (l == null) return;
        migration.lock();
        try {
            int id = target;
            if (id < 0) return;
            rebuilding = null; // a partly built replacement is simply rebuilt below
            var codes = l.codes;
            var pages = new ArrayList<MutableASHVectors.PageRebuild>();
            for (int p = 0; p < codes.pageCount(); p++) {
                var rebuild = codes.rebuildQuiesced(p, id);
                if (rebuild != null) pages.add(rebuild);
            }
            PhysicalCoreExecutor.pool().submit(() -> pages.parallelStream().forEach(rebuild -> {
                rebuild.encode(codes.vectorsPerPage(), vectors.get());
                rebuild.publish();
            })).join();
            target = -1;
        } finally {
            migration.unlock();
        }
    }

    /** True while the index is moving its pages to a newer model. */
    public boolean isChangingModel() {
        return target >= 0;
    }

    /** Vectors in the index when its current model was trained; for an inherited model, the training threshold. */
    public long vectorsAtLastTraining() {
        return trainedAt;
    }

    /** How long fixing a cold index's first model held inserts, in nanoseconds; 0 before then or when warm. */
    public long firstModelPauseNanos() {
        return firstModelPauseNanos;
    }

    /** True while a model is being trained on the executor. */
    public boolean isTraining() {
        return training.get();
    }

    /**
     * Trains a model on the sampled ordinals, for this index ({@link #retrain}) or the caller's next index over the
     * same kind of data. Reads the sampled vectors from the caller's vectors; holds none afterward. Trainings
     * run one at a time across the JVM's streaming indexes, as each uses the whole pool.
     *
     * @param optimizer {@link AsymmetricHashing#ITQ} or {@link AsymmetricHashing#RANDOM}
     * @param executor  the pool training runs in, whose parallelism native linear algebra also uses; null for the
     *                  library's physical-core pool, as ASH training uses elsewhere
     * @return the new model, or null when nothing has been added yet
     */
    public AsymmetricHashing trainNextModel(int optimizer, ForkJoinPool executor) {
        int[] sample = reservoir.sample();
        if (sample.length == 0) return null;
        TRAINING.lock();
        try {
            var ravv = new RemappedRandomAccessVectorValues(vectors.get(), sample);
            return AsymmetricHashing.initialize(ravv, optimizer, encodedBits, 1, bitsPerDimension,
                    executor == null ? PhysicalCoreExecutor.pool() : executor);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } finally {
            TRAINING.unlock();
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Searching

    /**
     * The {@code topK} nearest vectors to {@code query}: an ASH graph search for {@code rerankK} candidates,
     * reranked with exact scores from the caller's vectors. Before a cold index has its model, a graph search with
     * exact scores.
     */
    public SearchResult search(VectorFloat<?> query, int topK, int rerankK, Bits acceptOrds) {
        Live l = live;
        if (l == null) {
            Exact e = exact;
            if (e != null) {
                var searcher = e.searchers.get();
                searcher.setView(e.builder.getGraph().getView());
                return searcher.search(DefaultSearchScoreProvider.exact(query, similarity, vectors.get()), topK,
                        rerankK, 0f, 0f, acceptOrds);
            }
            l = live; // published before the exact graph is dropped
        }
        return search(l, l.codes.precomputedScoreFunctionFor(query, similarity), query, topK, rerankK, acceptOrds);
    }

    /**
     * The scores for searching this index with a searcher of the caller's own, such as one that resumes searches:
     * exact scores before a cold index has its model, then ASH scores reranked with exact scores from the caller's
     * vectors. Search the current {@link #graph}; either may be read first, as both kinds of scores cover every node
     * of either graph.
     */
    public SearchScoreProvider searchScoreProviderFor(VectorFloat<?> query) {
        Live l = live;
        if (l == null) {
            if (exact != null) return DefaultSearchScoreProvider.exact(query, similarity, vectors.get());
            l = live; // published before the exact graph is dropped
        }
        return new DefaultSearchScoreProvider(l.codes.precomputedScoreFunctionFor(query, similarity),
                vectors.get().rerankerFor(query, similarity));
    }

    /**
     * A preparer that projects queries for this index, several at once when they arrive together, as a queued
     * search does; null while a cold index scores exactly, when there is nothing to prepare.
     */
    public QueryPreparer queryPreparer() {
        Live l = live;
        return l == null ? null : l.codes.queryPreparer();
    }

    /** As {@link #search(VectorFloat, int, int, Bits)}, for a query prepared by this index's {@link #queryPreparer}. */
    public SearchResult search(PreparedQuery query, int topK, int rerankK,
                               Bits acceptOrds) {
        Live l = live; // a preparer exists only once the index has its model
        if (l == null) throw new IllegalStateException("The index has no model to prepare queries for");
        var vector = query.queryVector();
        return search(l, l.codes.scoreFunctionFor(query), vector, topK, rerankK, acceptOrds);
    }

    private SearchResult search(Live l, ScoreFunction.ApproximateScoreFunction score,
                                VectorFloat<?> query, int topK, int rerankK, Bits acceptOrds) {
        var provider = new DefaultSearchScoreProvider(score, vectors.get().rerankerFor(query, similarity));
        var searcher = l.searchers.get();
        searcher.setView(l.builder.getGraph().getView());
        return searcher.search(provider, topK, rerankK, 0f, 0f, acceptOrds);
    }

    // ---------------------------------------------------------------------------------------------
    // Accessors

    /** True once the index has fixed its first model. */
    public boolean hasModel() {
        return live != null;
    }

    /** The model new vectors are encoded with, or null while a cold index scores exactly. */
    public AsymmetricHashing model() {
        Live l = live;
        return l == null ? null : l.codes.getCompressor();
    }

    /** The index's codes, or null while a cold index scores exactly. */
    public MutableASHVectors codes() {
        Live l = live;
        return l == null ? null : l.codes;
    }

    /** The graph; before a cold index has its model, the graph with exact scores. */
    public OnHeapGraphIndex graph() {
        return (OnHeapGraphIndex) builder().getGraph();
    }

    /** The graph builder, for finishing the graph (cleanup) or writing it; see {@link #graph}. */
    public GraphIndexBuilder builder() {
        Live l = live;
        if (l != null) return l.builder;
        Exact e = exact;
        return e != null ? e.builder : live.builder;
    }

    /** The sampled ordinals training uses. */
    public OrdinalReservoir reservoir() {
        return reservoir;
    }

    /** Graph, codes and sample; the caller's vectors are not counted. */
    @Override
    public long ramBytesUsed() {
        return heapBytesUsed() + nativeBytesUsed();
    }

    /** The part of {@link #ramBytesUsed} on the Java heap: graph, codes, models' heap and sample. */
    public long heapBytesUsed() {
        long bytes = RamUsageEstimator.shallowSizeOfInstance(StreamingASHIndex.class) + reservoir.ramBytesUsed();
        Live l = live;
        if (l == null) {
            Exact e = exact;
            if (e != null) return bytes + e.builder.getGraph().ramBytesUsed();
            l = live;
        }
        return bytes + l.codes.ramBytesUsed() - l.codes.nativeBytesUsed() + l.builder.getGraph().ramBytesUsed();
    }

    /** The part of {@link #ramBytesUsed} in native memory: the models' native projections. */
    public long nativeBytesUsed() {
        Live l = live;
        return l == null ? 0 : l.codes.nativeBytesUsed();
    }
}
