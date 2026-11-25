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
import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OnDiskParallelGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OnDiskSequentialGraphIndexWriter;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.PreparedQuery;
import io.github.jbellis.jvector.graph.similarity.QueryPreparer;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.ASHQueryProjection;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.MutableASHVectors;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.DocIdSetIterator;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.util.OrdinalReservoir;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.util.ThreadSafeGrowableBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
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
 * <p><b>A graph index.</b> The index is an {@link ASHGraphIndex}: its structure is that of its current
 * {@link #graph}, its {@link #getView views} score queries with its ASH codes and rerank them from the caller's
 * vectors, its {@link #queryPreparer} prepares queries for its model, and it is written as its graph is. Before a cold
 * index has its model the graph is empty, so searches through a view find nothing; {@link #search} scores the
 * recorded vectors exactly until then.</p>
 *
 * <p><b>Models.</b> An index starts with a model the caller passes in, typically one trained on an earlier index
 * over the same kind of data, or cold. A cold index first only records the ordinals it is given, and searches them
 * by scoring each exactly, while it keeps a {@link OrdinalReservoir uniform sample} of ordinals. With
 * {@link Options#train} it trains its first model on that sample once {@link Options#trainAfter} ordinals have been
 * offered; without it, it fixes a random-projection model, which needs no training, after
 * {@link Options#coldStartSize}. It then encodes the recorded vectors with that model and inserts them into its graph
 * in parallel, with ASH scores, while new ordinals are still recorded, and continues with ASH scoring once they are
 * all in. Later, with {@link Options#train}, it trains newer models when due (see
 * {@link Options}) and moves to them: new pages of codes are encoded with the new model at once, and existing pages
 * are re-encoded from the caller's vectors and replaced one at a time (see {@link MutableASHVectors}). The graph is
 * never rebuilt.</p>
 *
 * <p><b>Cost.</b> Every operation's cost is bounded. Until a cold index has its first model, an insert only records its
 * ordinal, and a search scores every recorded vector. Once the model is trained, each insert also places one recorded
 * vector into the new graph while the pool places the rest, so the backlog shrinks however fast inserts arrive, and
 * inserts never wait for training: a caller with a write timeout, such as a database, sees no stall. Inserts wait only
 * while the last few recorded vectors join the graph, once, and, as a safeguard, if recorded vectors reach many times
 * the number that makes the model due before it is trained. Once the index has a
 * model, an insert encodes its vector, inserts it into the graph and
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
 * <p>Similarity: the index's similarity, DOT_PRODUCT, EUCLIDEAN or COSINE, is fixed when it is created; every model
 * it trains is trained for it, and every scorer uses it. Vectors need not be unit length. A COSINE index rejects zero
 * vectors and queries with IllegalArgumentException; callers validate them first.</p>
 */
@Experimental
public final class StreamingASHIndex implements ASHGraphIndex {
    /** Settings for a streaming index; the defaults suit most data. */
    public static final class Options {
        /**
         * Vectors a cold index records before it fixes a random-projection model, when it does not train; also the
         * smallest sample a first model is trained on, below which it uses a random projection.
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

    /** Null until the index has a model; published once, before {@link #cold} is cleared. */
    private volatile Live live;
    /** A cold index's recorded ordinals, until the index has a model; then null. */
    private volatile Cold cold;
    /** Held by inserts and deletes while the index is cold; held exclusively while the first model is published. */
    private final ReentrantReadWriteLock firstModel =
            new ReentrantReadWriteLock();
    /** The mean of the vectors added, kept as they arrive; see {@link #mean}. */
    private final VectorMean.Running mean;
    /** How long publishing the first model held inserts, in nanoseconds; 0 until then. */
    private volatile long firstModelPauseNanos;
    /** Recorded vectors still to join the graph when inserts are held to add the rest; see {@link #fixFirstModel}. */
    private static final int CATCH_UP = 256;
    /** Vectors a cold index records, at most, before inserts wait for its first model to be trained: a safeguard far
     * above what arrives while a model trains, so it bounds the vectors a search scores if training cannot keep up. */
    private final long recordLimit;
    /** Times the number that makes the first model due that a cold index records before inserts wait for it. */
    private static final int RECORD_LIMIT_FACTOR = 64;
    private final java.util.concurrent.atomic.AtomicLong recorded = new java.util.concurrent.atomic.AtomicLong();

    // Memory allocated outside inserts (the sample, models, and the graph nodes and codes the first model gives the
    // recorded vectors) is reported on the insert path, heap through add() and native through takeNativeBytes().
    // Guarded by itself; the charge is re-checked when the models change and on about one insert in 1024.
    private final Object charges = new Object();
    private long reportedHeap;
    private long reportedNative;
    private long firstModelBytes;
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
    /** The first model's graph while the recorded vectors join it; see {@link #fixFirstModel}. */
    private static final class CatchUp {
        final Live next;
        /** Recorded ordinals placed in {@link #next}, or claimed by the thread placing them. */
        final ThreadSafeGrowableBitSet placed;
        final java.util.concurrent.atomic.AtomicLong bytes = new java.util.concurrent.atomic.AtomicLong();
        /** Where inserts look for the next recorded ordinal to place. */
        final java.util.concurrent.atomic.AtomicInteger cursor = new java.util.concurrent.atomic.AtomicInteger();

        CatchUp(Live next, int capacity) {
            this.next = next;
            this.placed = new ThreadSafeGrowableBitSet(capacity);
        }
    }

    private static final class Cold {
        /** Ordinals added, including those with no vector, which get the all-zero code. */
        final ThreadSafeGrowableBitSet added = new ThreadSafeGrowableBitSet(1024);
        /** Ordinals deleted while the index is cold. */
        final ThreadSafeGrowableBitSet deleted = new ThreadSafeGrowableBitSet(1024);
        /** What {@link #builder} returns until the first model is published: an empty graph. */
        final GraphIndexBuilder empty;
        /** Set once the first model is trained, while the recorded vectors join its graph; inserts help. */
        volatile CatchUp catchUp;

        Cold(GraphIndexBuilder empty) {
            this.empty = empty;
        }
    }

    private StreamingASHIndex(RandomAccessVectorValues vectors, VectorSimilarityFunction similarity,
                              Function<BuildScoreProvider, GraphIndexBuilder> builders, AsymmetricHashing model,
                              int encodedBits, int bitsPerDimension, Options options) {
        Objects.requireNonNull(vectors, "vectors");
        Objects.requireNonNull(similarity, "similarity");
        if (model != null && model.similarityFunction() != similarity)
            throw new IllegalArgumentException("The model was trained for " + model.similarityFunction()
                    + "; the index is " + similarity);
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
        this.recordLimit = (long) RECORD_LIMIT_FACTOR * Math.max(1, options.train ? trainAfter : options.coldStartSize);
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
            cold = new Cold(builders.apply(BuildScoreProvider.randomAccessScoreProvider(source, similarity)));
        }
    }

    /**
     * An index that encodes vectors with {@code model} until it trains a newer one. Its similarity is the model's.
     *
     * @param vectors  the caller's vectors, read by ordinal; see the class comment
     * @param builders creates the graph builder from the index's score provider, with the caller's graph settings
     *                 (degree, beam width, hierarchy, executors)
     */
    public static StreamingASHIndex withModel(RandomAccessVectorValues vectors, AsymmetricHashing model,
                                              Function<BuildScoreProvider, GraphIndexBuilder> builders,
                                              Options options) {
        Objects.requireNonNull(model, "model");
        return new StreamingASHIndex(vectors, model.similarityFunction(), builders, model, model.encodedBits,
                model.bitsPerDimension, options);
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

    /**
     * COSINE is undefined for a zero vector, so a COSINE index rejects zero vectors and queries, as
     * {@link VectorUtil#l2normalize} does; callers validate vectors before adding or searching them.
     */
    private void checkVector(VectorFloat<?> v) {
        if (similarity == VectorSimilarityFunction.COSINE && v != null && !(VectorUtil.dotProduct(v, v) > 0f))
            throw new IllegalArgumentException("COSINE similarity is undefined for a zero vector");
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
     *         model gives the recorded vectors their nodes and codes outside inserts; that memory is reported by later
     *         calls.
     */
    public long add(int ordinal) {
        Live l = live;
        if (l == null && recorded.get() >= recordLimit && awaitingCatchUp()) {
            // A cold index that has recorded its limit waits for its first model, retrying it if a fix failed.
            while ((l = live) == null && awaitingCatchUp()) {
                afterColdInsert();
                LockSupport.parkNanos(100_000);
            }
        }
        if (l == null) {
            // While the first model trains, inserts take turns so the training gets the CPU.
            boolean slot = training.get();
            if (slot) trainingSlots.acquireUninterruptibly();
            var lock = firstModel.readLock();
            lock.lock();
            try {
                l = live;
                if (l == null) {
                    VectorFloat<?> v = vectors.get().getVector(ordinal);
                    checkVector(v);
                    reservoir.offer(ordinal);
                    if (v != null) mean.accept(v);
                    Cold c = cold;
                    c.added.set(ordinal); // last, so a search that finds the ordinal can read its vector
                    recorded.incrementAndGet();
                    // Once the first model is trained, each insert places one recorded vector, so the backlog
                    // cannot outgrow the catch-up however fast inserts arrive.
                    CatchUp u = c.catchUp;
                    if (u != null) placeOneRecorded(c, u);
                }
            } finally {
                lock.unlock();
                if (slot) trainingSlots.release();
            }
            if (l == null) {
                afterColdInsert();
                return chargeOutsideInserts();
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
                    + firstModelBytes;
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
    private void afterColdInsert() {
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
     * True while the index is cold and its first model has not started catching up. The first model is published by
     * setting {@code live} and then clearing {@code cold}, so a null {@code cold} means {@code live} is set.
     */
    private boolean awaitingCatchUp() {
        Cold c = cold;
        return c != null && c.catchUp == null;
    }

    /**
     * Trains (or, without {@link Options#train} or on a small sample, fixes a random projection for) the first model,
     * then encodes the recorded vectors with it and inserts them into a new graph with ASH scores, in parallel, while
     * inserts keep recording. Once few are left, inserts wait while the rest join and the index publishes its graph;
     * until then searches score the recorded vectors exactly.
     */
    private void fixFirstModel() {
        if (live != null) return;
        int[] sample = reservoir.sample();
        if (sample.length == 0) return;
        boolean trained = options.train && sample.length >= options.coldStartSize;
        AsymmetricHashing model = trainNextModel(trained ? options.optimizer : AsymmetricHashing.RANDOM, null);
        configureQueries(model, null);
        Cold c = cold;
        Live next = startLive(model);
        var u = new CatchUp(next, Math.max(1024, c.added.length()));
        c.catchUp = u;
        // Catch up with the recorded vectors while inserts continue and help; each round takes what is left.
        for (int round = 0; round < 8 && insertRecorded(c, u) > CATCH_UP; round++) { }
        var lock = firstModel.writeLock();
        lock.lock();
        long pauseStart = System.nanoTime();
        try {
            if (live != null) return;
            insertRecorded(c, u);
            for (int i = c.deleted.nextSetBit(0); i != DocIdSetIterator.NO_MORE_DOCS;
                 i = i + 1 < c.deleted.length() ? c.deleted.nextSetBit(i + 1) : DocIdSetIterator.NO_MORE_DOCS)
                next.builder.markNodeDeleted(i);
            trainedAt = next.codes.count();
            provisional = options.optimizer != AsymmetricHashing.RANDOM
                    && (!trained || options.optimizer != model.optimizer);
            synchronized (charges) {
                firstModelBytes = u.bytes.get();
            }
            live = next;
            cold = null;
            chargesDue = true;
            firstModelPauseNanos = System.nanoTime() - pauseStart;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Encodes and inserts, in parallel on the physical-core pool, the recorded ordinals not yet placed.
     *
     * @return how many it found to place
     */
    private int insertRecorded(Cold c, CatchUp u) {
        var added = c.added;
        int upper = added.length();
        int[] todo = java.util.stream.IntStream.range(0, upper).filter(o -> added.get(o) && !u.placed.get(o)).toArray();
        if (todo.length == 0) return 0;
        PhysicalCoreExecutor.pool().submit(() -> java.util.Arrays.stream(todo).parallel().forEach(ordinal -> {
            if (!u.placed.getAndSet(ordinal)) u.bytes.addAndGet(place(u.next, ordinal));
        })).join();
        return todo.length;
    }

    /** Places the next recorded ordinal no one has placed yet, if the insert finds one nearby. */
    private void placeOneRecorded(Cold c, CatchUp u) {
        int from = u.cursor.get();
        for (int tries = 0; tries < 64; tries++) {
            int o = c.added.nextSetBit(from);
            if (o == DocIdSetIterator.NO_MORE_DOCS) return;
            u.cursor.accumulateAndGet(o + 1, Math::max);
            if (!u.placed.getAndSet(o)) {
                u.bytes.addAndGet(place(u.next, o));
                return;
            }
            from = o + 1;
        }
    }

    private long insert(Live l, int ordinal) {
        VectorFloat<?> v = vectors.get().getVector(ordinal);
        checkVector(v);
        if (v != null) mean.accept(v);
        reservoir.offer(ordinal);
        return place(l, ordinal);
    }

    /** Encodes the vector at {@code ordinal} and inserts it into the graph; returns the memory both added. */
    private long place(Live l, int ordinal) {
        VectorFloat<?> v = vectors.get().getVector(ordinal);
        if (v == null) l.codes.setZero(ordinal);
        else l.codes.encodeAndSet(ordinal, v);
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
                    cold.deleted.set(ordinal);
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
            return AsymmetricHashing.initialize(ravv, similarity, optimizer, encodedBits, 1, bitsPerDimension,
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
     * reranked with exact scores from the caller's vectors. Before a cold index has its model, every recorded vector
     * scored exactly, so the results are exact; callers that search the graph themselves use this until
     * {@link #hasModel}.
     */
    public SearchResult search(VectorFloat<?> query, int topK, int rerankK, Bits acceptOrds) {
        checkVector(query);
        Live l = live;
        if (l == null) {
            Cold c = cold;
            if (c != null) return searchRecorded(c, query, topK, acceptOrds);
            l = live; // published before the recorded ordinals are dropped
        }
        return search(l, l.codes.precomputedScoreFunctionFor(query, similarity), query, topK, rerankK, acceptOrds);
    }

    /** The {@code topK} best of a cold index's recorded vectors, each scored exactly. */
    private SearchResult searchRecorded(Cold c, VectorFloat<?> query, int topK, Bits acceptOrds) {
        var source = vectors.get();
        var best = new java.util.PriorityQueue<SearchResult.NodeScore>(
                java.util.Comparator.comparingDouble((SearchResult.NodeScore n) -> n.score));
        var added = c.added;
        int scored = 0;
        for (int o = added.nextSetBit(0); o != DocIdSetIterator.NO_MORE_DOCS;
             o = o + 1 < added.length() ? added.nextSetBit(o + 1) : DocIdSetIterator.NO_MORE_DOCS) {
            if (c.deleted.get(o) || !acceptOrds.get(o)) continue;
            VectorFloat<?> v = source.getVector(o);
            if (v == null) continue;
            float score = similarity.compare(query, v);
            scored++;
            if (best.size() < topK) best.add(new SearchResult.NodeScore(o, score));
            else if (score > best.peek().score) {
                best.poll();
                best.add(new SearchResult.NodeScore(o, score));
            }
        }
        var nodes = new SearchResult.NodeScore[best.size()];
        for (int i = nodes.length - 1; i >= 0; i--) nodes[i] = best.poll();
        return new SearchResult(nodes, scored, 0, 0, 0, Float.POSITIVE_INFINITY);
    }

    /**
     * The scores for searching this index with a searcher of the caller's own, such as one that resumes searches:
     * ASH scores reranked with exact scores from the caller's vectors. Search the current {@link #graph}. Before a
     * cold index has its model its graph is empty, and these are exact scores; search with {@link #search} until
     * {@link #hasModel}.
     */
    public SearchScoreProvider searchScoreProviderFor(VectorFloat<?> query) {
        checkVector(query);
        Live l = live;
        if (l == null) {
            if (cold != null) return DefaultSearchScoreProvider.exact(query, similarity, vectors.get());
            l = live; // published before the recorded ordinals are dropped
        }
        return new DefaultSearchScoreProvider(l.codes.precomputedScoreFunctionFor(query, similarity),
                vectors.get().rerankerFor(query, similarity));
    }

    /**
     * A preparer that projects queries for this index's current model, several at once when they arrive together, as
     * a queued search does; empty while a cold index scores exactly, when there is nothing to prepare.
     *
     * <p>The preparer is for the model that is current when it is obtained. Queries it prepares stay usable while the
     * index changes to a newer model, and are rejected once the change finishes and their model is no longer used; get
     * a new preparer when {@link #model} changes.</p>
     *
     * <p>Before this index was a {@link GraphIndex}, this method returned the preparer or null. It returns
     * {@link Optional} because {@link GraphIndex#queryPreparer} does: every graph index, ASH or not, is asked for its
     * preparer the same way, and code written against {@code GraphIndex} works on this index unchanged. Java does not
     * allow two methods that differ only in return type, and keeping the nullable form under another name would leave
     * two methods doing one thing, so the signature changed; the class is experimental. Callers that checked for null
     * use {@code queryPreparer().orElse(null)} or {@link Optional#ifPresent}.</p>
     */
    @Override
    public Optional<QueryPreparer> queryPreparer() {
        Live l = live;
        return l == null ? Optional.empty() : Optional.of(l.codes.queryPreparer());
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
        // No threshold or rerank floor: unclamped dot-product scores can be negative.
        return searcher.search(provider, topK, rerankK, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, acceptOrds);
    }

    // ---------------------------------------------------------------------------------------------
    // Accessors

    /** True once the index has fixed its first model. */
    @Override
    public boolean hasModel() {
        return live != null;
    }

    /** The model new vectors are encoded with, or null while a cold index scores exactly. */
    @Override
    public AsymmetricHashing model() {
        Live l = live;
        return l == null ? null : l.codes.getCompressor();
    }

    /** The index's codes, or null while a cold index scores exactly. */
    public MutableASHVectors codes() {
        Live l = live;
        return l == null ? null : l.codes;
    }

    /** The graph; empty until a cold index has its model. */
    public OnHeapGraphIndex graph() {
        return (OnHeapGraphIndex) builder().getGraph();
    }

    /** The graph builder, for finishing the graph (cleanup) or writing it; see {@link #graph}. */
    public GraphIndexBuilder builder() {
        Live l = live;
        if (l != null) return l.builder;
        Cold c = cold;
        return c != null ? c.empty : live.builder;
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

    /**
     * The part of {@link #ramBytesUsed} on the Java heap: graph, codes, models' heap and sample; for a cold index, its
     * record of ordinals.
     */
    public long heapBytesUsed() {
        long bytes = RamUsageEstimator.shallowSizeOfInstance(StreamingASHIndex.class) + reservoir.ramBytesUsed();
        Live l = live;
        if (l == null) {
            Cold c = cold;
            if (c != null) return bytes + (c.added.length() + c.deleted.length()) / 8 + c.empty.getGraph().ramBytesUsed();
            l = live;
        }
        return bytes + l.codes.ramBytesUsed() - l.codes.nativeBytesUsed() + l.builder.getGraph().ramBytesUsed();
    }

    /** The part of {@link #ramBytesUsed} in native memory: the models' native projections. */
    public long nativeBytesUsed() {
        Live l = live;
        return l == null ? 0 : l.codes.nativeBytesUsed();
    }

    // ---------------------------------------------------------------------------------------------
    // GraphIndex: the structure is the current graph's; see the class comment

    /**
     * A searcher whose {@link GraphSearcher#search(VectorFloat, int, int, Bits) search(query, topK, rerankK, acceptOrds)}
     * is this index's {@link #search(VectorFloat, int, int, Bits)}: it scores a cold index's recorded vectors exactly,
     * and searches the current graph with the current model, however long the searcher is kept. Its other searches
     * search the graph as of the view it was created with, as for any graph searcher.
     */
    @Override
    public GraphSearcher searcher() {
        return new GraphSearcher(this) {
            @Override
            public SearchResult search(VectorFloat<?> queryVector, int topK, int rerankK, Bits acceptOrds) {
                return StreamingASHIndex.this.search(queryVector, topK, rerankK, acceptOrds);
            }
        };
    }

    /**
     * A view of the current graph that also scores queries: with ASH codes once the index has a model, reranked
     * exactly from the caller's vectors. It scores with the model in use when it was taken; take a new view to see
     * a model fixed since. Before a cold index has its model the graph is empty and the view finds nothing.
     */
    @Override
    public View getView() {
        Live l = live;
        var g = l == null ? graph() : l.builder.getGraph();
        return new IndexView(l, g.getView());
    }

    @Override
    public NodesIterator getNodes(int level) {
        return graph().getNodes(level);
    }

    @Override
    public int maxDegree() {
        return graph().maxDegree();
    }

    @Override
    public List<Integer> maxDegrees() {
        return graph().maxDegrees();
    }

    @Override
    public int getDimension() {
        return dimension;
    }

    @Override
    public int getIdUpperBound() {
        return graph().getIdUpperBound();
    }

    @Override
    public boolean containsNode(int nodeId) {
        return graph().containsNode(nodeId);
    }

    @Override
    public boolean isHierarchical() {
        return graph().isHierarchical();
    }

    @Override
    public int getMaxLevel() {
        return graph().getMaxLevel();
    }

    @Override
    public int getDegree(int level) {
        return graph().getDegree(level);
    }

    @Override
    public double getAverageDegree(int level) {
        return graph().getAverageDegree(level);
    }

    @Override
    public int size(int level) {
        return graph().size(level);
    }

    /** The index's similarity, fixed when it is created. */
    @Override
    public Optional<VectorSimilarityFunction> similarityFunction() {
        return Optional.of(similarity);
    }

    /** Releases nothing: the caller owns the vectors, and the graph and codes are reclaimed with the index. */
    @Override
    public void close() {
    }

    // The writers write the current graph, as writing graph() does. The index holds no vectors: pass them to the
    // writer, as inline vectors or features. Fused ASH codes need one model: call prepareForFlush first. Deleted
    // nodes must be removed first with builder().cleanup().

    /** Writes the current graph; see {@link PersistableGraphIndex#getParallelWriterBuilder}. */
    @Override
    public OnDiskParallelGraphIndexWriter.Builder getParallelWriterBuilder(Path path) throws FileNotFoundException {
        return graph().getParallelWriterBuilder(path);
    }

    /** Writes the current graph; see {@link PersistableGraphIndex#getWriterBuilder(Path)}. */
    @Override
    public OnDiskGraphIndexWriter.Builder getWriterBuilder(Path path) throws FileNotFoundException {
        return graph().getWriterBuilder(path);
    }

    /** Writes the current graph; see {@link PersistableGraphIndex#getWriterBuilder(IndexWriter)}. */
    @Override
    public OnDiskSequentialGraphIndexWriter.Builder getWriterBuilder(IndexWriter out) {
        return graph().getWriterBuilder(out);
    }

    /** Writes the current graph with {@code vectors} inline; see {@link PersistableGraphIndex#writeTo}. */
    @Override
    public void writeTo(Path path, RandomAccessVectorValues vectors) throws IOException {
        graph().writeTo(path, vectors);
    }

    /** A view of a graph, scoring with the codes of the index state it was taken from; null codes while cold. */
    private final class IndexView implements ScoringView {
        private final Live live;
        private final View graphView;

        IndexView(Live live, View graphView) {
            this.live = live;
            this.graphView = graphView;
        }

        @Override
        public SearchScoreProvider scoreProviderFor(PreparedQuery query, boolean rerank) {
            if (live == null) throw new IllegalStateException("The index has no model to prepare queries for");
            var score = live.codes.scoreFunctionFor(query);
            var reranker = rerank ? rerankerFor(query.queryVector(), similarity) : null;
            return new DefaultSearchScoreProvider(score, reranker, similarity);
        }

        @Override
        public ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf) {
            checkVector(queryVector);
            return vectors.get().rerankerFor(queryVector, vsf);
        }

        @Override
        public ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> queryVector,
                                                                                  VectorSimilarityFunction vsf) {
            if (live == null) throw new UnsupportedOperationException("The index has no model to score with yet");
            checkVector(queryVector);
            return live.codes.precomputedScoreFunctionFor(queryVector, vsf);
        }

        @Override
        public boolean hasApproximateScores() {
            return live != null;
        }

        @Override
        public NodesIterator getNeighborsIterator(int level, int node) {
            return graphView.getNeighborsIterator(level, node);
        }

        @Override
        public void processNeighbors(int level, int node, ScoreFunction scoreFunction, IntMarker visited,
                                     NeighborProcessor neighborProcessor) {
            graphView.processNeighbors(level, node, scoreFunction, visited, neighborProcessor);
        }

        @Override
        @Deprecated
        public int size() {
            return graphView.size();
        }

        @Override
        public NodeAtLevel entryNode() {
            return graphView.entryNode();
        }

        @Override
        public Bits liveNodes() {
            return graphView.liveNodes();
        }

        @Override
        public int getIdUpperBound() {
            return graphView.getIdUpperBound();
        }

        @Override
        public boolean contains(int level, int node) {
            return graphView.contains(level, node);
        }

        @Override
        public void close() throws IOException {
            graphView.close();
        }
    }
}
