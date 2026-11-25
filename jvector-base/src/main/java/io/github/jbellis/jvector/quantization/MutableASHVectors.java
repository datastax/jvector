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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.PreparedQuery;
import io.github.jbellis.jvector.graph.similarity.QueryPreparer;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

/**
 * ASH codes for an index that grows while it is built and searched, and whose model can change while it does.
 *
 * <p><b>Pages.</b> Codes are kept in pages of a fixed power-of-two number of vectors. A page holds its vectors'
 * codes, their decoded headers and the centered offsets symmetric scoring needs, and records the model that
 * encoded every vector in it. Stored data never moves: adding a vector writes into its page, and the page
 * table only grows.</p>
 *
 * <p><b>Changing models.</b> {@link #addModel} registers a model and {@link #useModel} makes it the one new pages
 * are encoded with. A page takes the current model when it is created, so the switch takes effect at the next
 * page. Existing pages are moved to a newer model one at a time with {@link #reencodePage}: the replacement is
 * built privately, from the caller's vectors, and published with a single release store, so a reader sees the
 * whole old page or the whole new one and never a mix. A scorer records which model encoded each vector it scores
 * and prepares each model it meets once; scoring is exact to each vector's own model throughout a change.
 * Codes from different models are never compared directly: a symmetric score across models scores the source
 * vector, read from the caller, against the target's code. {@link #snapshot} needs a single model.</p>
 *
 * <p><b>Concurrency.</b> Vectors may be added from many threads, each ordinal once. A vector may be scored once its
 * addition happens-before the scoring, for example because the adding thread then inserted the ordinal into a
 * concurrent graph in which the scoring thread found it. Pages may be re-encoded concurrently with additions and
 * searches. Ordinals should be dense; a missing ordinal scores as an all-zero code.</p>
 *
 * <p>All models of one collection must share a code shape: original dimension, projected dimension and bits per
 * dimension, with one landmark. Scores of a model's codes are bit-for-bit those of {@link ASHVectors} with the same
 * codes.</p>
 */
@Experimental
public final class MutableASHVectors implements MutableCompressedVectors<VectorFloat<?>> {
    /** Target page size in bytes; pages stay under G1's smallest humongous threshold. */
    static final int DEFAULT_PAGE_BYTES = 256 << 10;
    /**
     * Most vectors per page. A page is the unit of a model change: a new model takes effect at the next page, and
     * re-encoding replaces a page at a time. Small codes would otherwise fit tens of thousands to a page.
     */
    static final int MAX_VECTORS_PER_PAGE = 1024;
    private static final VarHandle PAGE = MethodHandles.arrayElementVarHandle(Page[].class);
    private static final VectorUtilSupport VU = VectorizationProvider.getInstance().getVectorUtilSupport();

    private final int originalDimension;
    private final int quantizedDim;
    private final int bits;
    private final int signWords;
    private final int bodyBytes;
    private final int pageShift;
    private final int pageMask;
    private final boolean simdSymmetric;

    /** Registered models by id; grows by copy under the lock. */
    private volatile Model[] models;
    /** The id new pages are encoded with. */
    private volatile int current;

    /** The page table; replaced (never shrunk) under the lock when it grows. */
    private volatile Page[] pages = new Page[4];
    /** One past the highest ordinal set. */
    private final AtomicInteger size = new AtomicInteger();
    private volatile long ramBytesUsed;
    /** Pages per model id, guarded by this; and the ids with pages, ascending, republished when that set changes. */
    private final int[] pagesPerModel = new int[256];
    private volatile int[] inUseIds = new int[0];
    private final AtomicLong reencodedMissing = new AtomicLong();

    private final ThreadLocal<AsymmetricHashing.QuantizedVector> scratch;
    /** Batch scratch for page rebuilds, which every model of this collection's shape can use. */
    private final ThreadLocal<AsymmetricHashing.EncodingBatch> encodingBatch;
    private final ThreadLocal<int[]> batchSlots;

    /** A model and what scoring with it needs, prepared once. */
    static final class Model {
        final int id;
        final AsymmetricHashing ash;
        final ASHScorer scorer;
        final float landmarkDot;
        final AsymmetricHashing.QuantizedVector zero;
        final float zeroCenteredOffset;

        Model(int id, AsymmetricHashing ash) {
            this.id = id;
            this.ash = ash;
            this.scorer = new ASHScorer(ash);
            this.landmarkDot = ash.symmetricLandmarkTerm(0, 0);
            this.zero = AsymmetricHashing.QuantizedVector.createEmpty(ash.quantizedDim, ash.bitsPerDimension);
            this.zeroCenteredOffset = ash.centeredOffset(zero);
        }
    }

    /** A page of codes encoded with one model. Its arrays are written only before it is complete. */
    static final class Page {
        final int model;
        final long[] signs;
        final byte[] body;
        final float[] scales;
        final float[] offsets;
        final float[] centered;
        final byte[] landmarks;
        /** Slots set so far; a page is complete when every slot is set. */
        final AtomicInteger filled = new AtomicInteger();

        Page(int model, int vectors, int signWords, int bodyBytes) {
            this.model = model;
            this.signs = signWords == 0 ? null : new long[vectors * signWords];
            this.body = bodyBytes == 0 ? null : new byte[vectors * bodyBytes];
            this.scales = new float[vectors];
            this.offsets = new float[vectors];
            this.centered = new float[vectors];
            this.landmarks = new byte[vectors];
        }

        long bytes() {
            return RamUsageEstimator.shallowSizeOfInstance(Page.class) + RamUsageEstimator.shallowSizeOfInstance(AtomicInteger.class)
                    + (signs == null ? 0 : RamUsageEstimator.sizeOf(signs)) + (body == null ? 0 : RamUsageEstimator.sizeOf(body))
                    + RamUsageEstimator.sizeOf(scales) + RamUsageEstimator.sizeOf(offsets)
                    + RamUsageEstimator.sizeOf(centered) + RamUsageEstimator.sizeOf(landmarks);
        }
    }

    /** An empty collection whose vectors will be encoded with {@code ash} until another model is chosen. */
    public MutableASHVectors(AsymmetricHashing ash) {
        this(ash, DEFAULT_PAGE_BYTES);
    }

    /** As {@link #MutableASHVectors(AsymmetricHashing)} with a chosen page size; tests use small pages. */
    MutableASHVectors(AsymmetricHashing ash, int pageBytes) {
        Objects.requireNonNull(ash, "ash");
        if (ash.landmarkCount != 1) throw new IllegalArgumentException("Growable ASH codes need one landmark");
        this.originalDimension = ash.originalDimension;
        this.quantizedDim = ash.quantizedDim;
        this.bits = ash.bitsPerDimension;
        this.signWords = AsymmetricHashing.QuantizedVector.signWordsForDims(quantizedDim, bits);
        this.bodyBytes = AsymmetricHashing.QuantizedVector.bodyBytesForDims(quantizedDim, bits);
        int stride = Math.max(1, Math.max(signWords * Long.BYTES, bodyBytes));
        this.pageShift = 31 - Integer.numberOfLeadingZeros(Math.max(1, Math.min(MAX_VECTORS_PER_PAGE, pageBytes / stride)));
        this.pageMask = (1 << pageShift) - 1;
        this.simdSymmetric = VU.supportsAshSymmetricScoring() && (bits == 1 || VU.supportsAshSymmetricDotAt());
        this.models = new Model[]{new Model(0, ash)};
        this.current = 0;
        this.scratch = ThreadLocal.withInitial(() ->
                AsymmetricHashing.QuantizedVector.createEmpty(quantizedDim, bits));
        this.encodingBatch = ThreadLocal.withInitial(ash::newEncodingBatch);
        this.batchSlots = ThreadLocal.withInitial(() -> new int[encodingBatch.get().capacity()]);
        this.ramBytesUsed = RamUsageEstimator.shallowSizeOfInstance(MutableASHVectors.class)
                + RamUsageEstimator.shallowSizeOf(pages);
    }

    // ---------------------------------------------------------------------------------------------
    // Models

    /** Registers {@code ash}, which must have this collection's code shape, and returns its id. */
    public synchronized int addModel(AsymmetricHashing ash) {
        Objects.requireNonNull(ash, "ash");
        if (ash.originalDimension != originalDimension || ash.quantizedDim != quantizedDim
                || ash.bitsPerDimension != bits || ash.landmarkCount != 1)
            throw new IllegalArgumentException("A new model must have the collection's code shape and one landmark");
        if (ash.similarityFunction() != similarityFunction())
            throw new IllegalArgumentException("A new model must have the collection's similarity, "
                    + similarityFunction() + "; got " + ash.similarityFunction());
        Model[] m = models;
        if (m.length == 256) throw new IllegalStateException("At most 256 models per collection");
        Model[] next = Arrays.copyOf(m, m.length + 1);
        next[m.length] = new Model(m.length, ash);
        models = next;
        return m.length;
    }

    /** Encodes pages created from now on with model {@code id}; see the class comment. */
    public void useModel(int id) {
        Objects.checkIndex(id, models.length);
        current = id;
    }

    /** The id new pages are encoded with. */
    public int currentModel() {
        return current;
    }

    /** The model with {@code id}. */
    public AsymmetricHashing model(int id) {
        return models[id].ash;
    }

    /** The model new pages are encoded with. */
    @Override
    public AsymmetricHashing getCompressor() {
        return models[current].ash;
    }

    /** The similarity every model of this collection was trained for, which every scorer of it uses. */
    @Override
    public VectorSimilarityFunction similarityFunction() {
        return models[0].ash.similarityFunction();
    }

    // ---------------------------------------------------------------------------------------------
    // Pages

    /** The page holding {@code node}, or null if none was created. Acquire, so a swapped page is seen whole. */
    private Page page(int node) {
        Page[] t = pages;
        int p = node >>> pageShift;
        if (p < t.length) {
            Page page = (Page) PAGE.getAcquire(t, p);
            if (page != null) return page;
        }
        return pageSlow(p);
    }

    private synchronized Page pageSlow(int p) {
        Page[] t = pages;
        return p < t.length ? t[p] : null;
    }

    /** The page for {@code node}, creating it, and pages before it, with the current model if needed. */
    private Page pageForWrite(int node) {
        Page page = page(node);
        if (page != null) return page;
        synchronized (this) {
            int p = node >>> pageShift;
            Page[] t = pages;
            if (p >= t.length) {
                int length = t.length;
                while (length <= p) length *= 2;
                long before = RamUsageEstimator.shallowSizeOf(t);
                t = Arrays.copyOf(t, length);
                ramBytesUsed += RamUsageEstimator.shallowSizeOf(t) - before;
            }
            for (int i = 0; i <= p; i++) {
                if (t[i] == null) {
                    Page created = new Page(current, pageMask + 1, signWords, bodyBytes);
                    ramBytesUsed += created.bytes();
                    countPage(created.model, +1);
                    PAGE.setRelease(t, i, created);
                }
            }
            pages = t;
            return t[p];
        }
    }

    /** Writes a code into its slot of {@code page}. */
    private void store(Page page, int slot, AsymmetricHashing.QuantizedVector code, float centered) {
        if (signWords != 0) System.arraycopy(code.binaryVector, 0, page.signs, slot * signWords, signWords);
        if (bodyBytes != 0) System.arraycopy(code.extraBits, 0, page.body, slot * bodyBytes, bodyBytes);
        page.scales[slot] = code.scale;
        page.offsets[slot] = code.offset;
        page.centered[slot] = centered;
        page.landmarks[slot] = code.landmark;
    }

    private void extendSize(int ordinal) {
        int n;
        while ((n = size.get()) <= ordinal && !size.compareAndSet(n, ordinal + 1)) {
            // another writer raised it
        }
    }

    /** Encodes {@code vector} with its page's model and stores the code at {@code ordinal}. */
    @Override
    public void encodeAndSet(int ordinal, VectorFloat<?> vector) {
        if (ordinal < 0) throw new IndexOutOfBoundsException("Negative ordinal " + ordinal);
        Page page = pageForWrite(ordinal);
        var code = scratch.get();
        float centered = models[page.model].ash.encodeToWithCenteredOffset(vector, code);
        store(page, ordinal & pageMask, code, centered);
        page.filled.incrementAndGet();
        extendSize(ordinal);
    }

    /** Stores the all-zero code, the code of a missing vector, at {@code ordinal}. */
    @Override
    public void setZero(int ordinal) {
        if (ordinal < 0) throw new IndexOutOfBoundsException("Negative ordinal " + ordinal);
        Page page = pageForWrite(ordinal);
        Model m = models[page.model];
        store(page, ordinal & pageMask, m.zero, m.zeroCenteredOffset);
        page.filled.incrementAndGet();
        extendSize(ordinal);
    }

    /** Pages created so far. */
    public int pageCount() {
        Page[] t = pages;
        int n = 0;
        while (n < t.length && PAGE.getAcquire(t, n) != null) n++;
        return n;
    }

    /** Vectors per page. */
    public int vectorsPerPage() {
        return pageMask + 1;
    }

    /** The model id of page {@code index}, or -1 if that page does not exist. */
    public int pageModel(int index) {
        Page page = index < pages.length ? (Page) PAGE.getAcquire(pages, index) : null;
        return page == null ? -1 : page.model;
    }

    /** True when every slot of page {@code index} has been set. */
    public boolean isPageComplete(int index) {
        Page page = index < pages.length ? (Page) PAGE.getAcquire(pages, index) : null;
        return page != null && page.filled.get() == pageMask + 1;
    }

    /**
     * Replaces complete page {@code index} with one encoded by model {@code id}, reading its vectors from
     * {@code vectors}, in one call. See {@link #rebuild} to spread the work out.
     *
     * @return false if the page is missing, incomplete, already encoded with that model, or was replaced meanwhile
     */
    public boolean reencodePage(int index, int id, RandomAccessVectorValues vectors) {
        var rebuild = rebuild(index, id);
        if (rebuild == null) return false;
        rebuild.encode(vectorsPerPage(), vectors);
        return rebuild.publish();
    }

    /**
     * Starts replacing complete page {@code index} with one encoded by model {@code id}. The replacement is built
     * privately, a few vectors at a time, and published whole; see the class comment.
     *
     * @return the rebuild, or null if the page is missing, incomplete or already encoded with that model
     */
    public PageRebuild rebuild(int index, int id) {
        return rebuild(index, id, false);
    }

    /**
     * As {@link #rebuild(int, int)}, also for a page that is not complete, such as the last page. Only for a
     * caller that adds no vectors until the rebuild is published: an addition to the old page meanwhile is lost.
     * Ordinals at or beyond {@link #count()} get the all-zero code.
     */
    public PageRebuild rebuildQuiesced(int index, int id) {
        return rebuild(index, id, true);
    }

    private PageRebuild rebuild(int index, int id, boolean quiesced) {
        Objects.checkIndex(id, models.length);
        Page old = index < pages.length ? (Page) PAGE.getAcquire(pages, index) : null;
        if (old == null || old.model == id || (!quiesced && old.filled.get() != pageMask + 1)) return null;
        return new PageRebuild(index, old, models[id]);
    }

    /** Slots re-encoded as all-zero codes because the caller had no vector for them, such as deleted vectors. */
    public long reencodedMissing() {
        return reencodedMissing.get();
    }

    /** A replacement page being built; used by one thread at a time. */
    public final class PageRebuild {
        private final int index;
        private final Page old;
        private final Page fresh;
        private final Model model;
        private int next;

        private PageRebuild(int index, Page old, Model model) {
            this.index = index;
            this.old = old;
            this.model = model;
            this.fresh = new Page(model.id, pageMask + 1, signWords, bodyBytes);
        }

        /** The page being replaced. */
        public int index() {
            return index;
        }

        /** True when every vector of the page has been encoded. */
        public boolean isFull() {
            return next > pageMask;
        }

        /**
         * Encodes up to {@code max} more of the page's vectors, read from {@code vectors}; an ordinal the caller has
         * no vector for (null) gets the all-zero code.
         *
         * @return the number encoded
         */
        public int encode(int max, RandomAccessVectorValues vectors) {
            int base = index << pageShift, done = 0, n = count();
            // Vectors are encoded in batches, as a fixed store's are; each batch's slots are remembered in order.
            var batch = encodingBatch.get();
            int[] slots = batchSlots.get();
            batch.clear();
            for (; done < max && next <= pageMask; done++, next++) {
                if (base + next >= n) {
                    store(fresh, next, model.zero, model.zeroCenteredOffset);
                    continue;
                }
                VectorFloat<?> v = vectors.getVector(base + next);
                if (v == null) {
                    store(fresh, next, model.zero, model.zeroCenteredOffset);
                    reencodedMissing.incrementAndGet();
                } else {
                    slots[batch.size()] = next;
                    model.ash.addToBatch(batch, v);
                    if (batch.isFull()) storeBatch(batch, slots);
                }
            }
            storeBatch(batch, slots);
            return done;
        }

        private void storeBatch(AsymmetricHashing.EncodingBatch batch, int[] slots) {
            model.ash.encodeBatch(batch);
            for (int i = 0; i < batch.size(); i++) store(fresh, slots[i], batch.code(i), batch.centeredOffset(i));
            batch.clear();
        }

        /**
         * Publishes the full replacement with one release store.
         *
         * @return false if the page was replaced by another rebuild meanwhile
         * @throws IllegalStateException if the replacement is not full
         */
        public boolean publish() {
            if (!isFull()) throw new IllegalStateException("Page " + index + " is not fully re-encoded");
            fresh.filled.set(old.filled.get());
            synchronized (MutableASHVectors.this) {
                Page[] t = pages;
                if (t[index] != old) return false;
                ramBytesUsed += fresh.bytes() - old.bytes();
                countPage(fresh.model, +1);
                countPage(old.model, -1);
                PAGE.setRelease(t, index, fresh);
            }
            return true;
        }
    }

    /** Called with the lock held. */
    private void countPage(int model, int delta) {
        int before = pagesPerModel[model];
        pagesPerModel[model] = before + delta;
        if ((before == 0) != (before + delta == 0)) {
            int[] ids = new int[256];
            int n = 0;
            for (int i = 0; i < 256; i++) if (pagesPerModel[i] > 0) ids[n++] = i;
            inUseIds = Arrays.copyOf(ids, n);
        }
    }

    /** The memory one vector's code takes in a page: its sign words, body bytes, scale, offsets and landmark. */
    public long bytesPerVector() {
        return (long) signWords * Long.BYTES + bodyBytes + 3L * Float.BYTES + Byte.BYTES;
    }

    /** Ids of the models that encode at least one page, ascending. */
    public int[] modelsInUse() {
        return inUseIds.clone();
    }

    @Override
    public int count() {
        return size.get();
    }

    // ---------------------------------------------------------------------------------------------
    // Query scoring

    @Override
    public ScoreFunction.AdaptiveApproximateScoreFunction scoreFunctionFor(VectorFloat<?> query,
                                                                          VectorSimilarityFunction similarityFunction) {
        getCompressor().checkSimilarity(similarityFunction);
        return new QueryScoreFunction(query);
    }

    @Override
    public ScoreFunction.AdaptiveApproximateScoreFunction precomputedScoreFunctionFor(VectorFloat<?> query,
                                                                                     VectorSimilarityFunction similarityFunction) {
        return scoreFunctionFor(query, similarityFunction);
    }

    /**
     * A preparer for queries to this collection, for its current model: it projects queries, several at once when
     * they arrive together, so {@link #scoreFunctionFor(PreparedQuery)} scores without projecting again. Queued
     * search uses it to prepare waiting queries in one batch. A preparer made before a model change still works:
     * the other models in use are prepared from the query vector when scored.
     */
    public QueryPreparer queryPreparer() {
        return new ASHQueryPreparer(getCompressor());
    }

    /**
     * As {@link #scoreFunctionFor(VectorFloat, VectorSimilarityFunction)} in the collection's similarity, for a query
     * prepared by
     * a {@link #queryPreparer} of this collection; the score function borrows the prepared state until its search
     * ends.
     */
    public ScoreFunction.AdaptiveApproximateScoreFunction scoreFunctionFor(PreparedQuery query) {
        if (!(query instanceof ASHQueryPreparer.State))
            throw new IllegalArgumentException("Prepared query belongs to another kind of index");
        var ash = ((ASHQueryPreparer.State) query).ash;
        Model[] ms = models;
        for (Model m : ms) {
            if (m.ash == ash) return new QueryScoreFunction(query.queryVector(), m.id,
                    ASHQueryPreparer.state(ash, query, true).prepared);
        }
        throw new IllegalArgumentException("Prepared query belongs to a model this collection does not use");
    }

    /** A query prepared for one model: the same arithmetic {@link ASHVectors} uses for that model's codes. */
    private abstract static class ModelQuery {
        abstract float similarityTo(Page page, int slot);

        abstract float dotProductBound();
    }

    /** Through the model's code scorer, as {@link ASHVectors} scores multibit codes and forced-scalar 1-bit codes. */
    private final class CodeScorerQuery extends ModelQuery {
        private final ASHScorer.CodeScorer f;
        private final ASHScoreScale scale;
        private final float bound;

        CodeScorerQuery(Model m, VectorFloat<?> query) {
            this(m, m.scorer.precomputeQuery(query));
        }

        CodeScorerQuery(Model m, ASHScorer.QueryPrecompute qp) {
            scale = m.ash.scores();
            f = m.scorer.codeScorerFor(qp);
            bound = m.scorer.dotProductBound(qp);
        }

        @Override
        float similarityTo(Page page, int slot) {
            return scale.similarity(f.dot(page.signs, slot * signWords, page.body, slot * bodyBytes,
                    page.scales[slot], page.offsets[slot], page.landmarks[slot] & 0xFF));
        }

        @Override
        float dotProductBound() {
            return bound;
        }
    }

    /** The masked-add kernel over sign words, as {@link ASHVectors} scores 1-bit codes when the backend has it. */
    private final class MaskedAddQuery extends ModelQuery {
        private final float[] qProj;
        private final float sumQProj;
        private final float dotQMu;
        private final float bound;
        private final ASHScoreScale scale;
        private final int words = AsymmetricHashing.QuantizedVector.wordsForDims(quantizedDim);

        MaskedAddQuery(Model m, VectorFloat<?> query) {
            scale = m.ash.scores();
            float[] q = new float[originalDimension];
            query = m.ash.loadQuery(query, q, null);
            qProj = new float[quantizedDim];
            m.ash.stiefelTransform.projectQuery(q, qProj);
            sumQProj = ASHScorer.sum(qProj);
            dotQMu = m.ash.queryLandmarkTerm(query, 0);
            bound = bound(m, (float) Math.sqrt(VectorUtil.dotProduct(query, query)));
        }

        MaskedAddQuery(Model m, ASHScorer.QueryPrecompute qp) {
            scale = m.ash.scores();
            qProj = qp.qProj;
            sumQProj = qp.sumQProj;
            dotQMu = qp.dotQMuByLandmark[0];
            bound = bound(m, qp.queryNorm);
        }

        private float bound(Model m, float norm) {
            float maxNorm = m.ash.maxEncodedNorm();
            return (Float.isNaN(norm) ? maxNorm : norm) * maxNorm;
        }

        @Override
        float similarityTo(Page page, int slot) {
            float maskedAdd = VU.ashMaskedAdd_512(qProj, 0, page.signs, slot * signWords, quantizedDim, words);
            return scale.similarity(page.scales[slot] * (2f * maskedAdd - sumQProj) + dotQMu + page.offsets[slot]);
        }

        @Override
        float dotProductBound() {
            return bound;
        }
    }

    /** The query path {@link ASHVectors} would use for {@code m}'s codes, in {@code m}'s similarity. */
    private ModelQuery prepare(Model m, VectorFloat<?> query) {
        return maskedAdd() ? new MaskedAddQuery(m, query) : new CodeScorerQuery(m, query);
    }

    /** As {@link #prepare(Model, VectorFloat)}, from a query already prepared for {@code m}. */
    private ModelQuery prepare(Model m, ASHScorer.QueryPrecompute qp) {
        return maskedAdd() ? new MaskedAddQuery(m, qp) : new CodeScorerQuery(m, qp);
    }

    /** Whether 1-bit codes are scored with the masked-add kernel, as {@link ASHVectors} scores them. */
    private boolean maskedAdd() {
        if (bits != 1) return false;
        var kernel = ASHVectors.AshSingleKernel.fromProperty();
        boolean simd = kernel != ASHVectors.AshSingleKernel.SCALAR && VU.supportsAshMaskedLoad();
        if (kernel == ASHVectors.AshSingleKernel.SIMD && !simd)
            throw new IllegalStateException("ASH single-vector SIMD is unavailable on " + VU.getClass().getName());
        return simd;
    }

    /**
     * Scores a raw query against each node with the node's own model. Every model in use is prepared up front,
     * so the bound covers every node; a model that comes into use later is prepared when first met.
     */
    private final class QueryScoreFunction implements ScoreFunction.AdaptiveApproximateScoreFunction {
        private final VectorFloat<?> query;
        private ModelQuery[] prepared = new ModelQuery[models.length];
        private float bound = Float.NEGATIVE_INFINITY;

        QueryScoreFunction(VectorFloat<?> query) {
            this(query, -1, null);
        }

        /** With {@code qp} already prepared for model {@code preparedId}; the other models are prepared here. */
        QueryScoreFunction(VectorFloat<?> query, int preparedId, ASHScorer.QueryPrecompute qp) {
            this.query = query;
            if (preparedId >= 0) {
                if (preparedId >= prepared.length) prepared = Arrays.copyOf(prepared, preparedId + 1);
                use(preparedId, MutableASHVectors.this.prepare(models[preparedId], qp));
            }
            int[] inUse = inUseIds;
            if (inUse.length == 0 && current != preparedId) prepare(current);
            for (int id : inUse) if (id != preparedId) prepare(id);
        }

        private void use(int id, ModelQuery q) {
            prepared[id] = q;
            float b = q.dotProductBound();
            if (!(b <= bound)) bound = b;
        }

        private ModelQuery prepare(int id) {
            if (id >= prepared.length) prepared = Arrays.copyOf(prepared, models.length);
            ModelQuery q = MutableASHVectors.this.prepare(models[id], query);
            use(id, q);
            return q;
        }

        @Override
        public float similarityTo(int node) {
            Page page = page(node);
            var p = prepared;
            ModelQuery q = page.model < p.length ? p[page.model] : null;
            if (q == null) q = prepare(page.model);
            return q.similarityTo(page, node & pageMask);
        }

        @Override
        public float dotProductBound() {
            return bound;
        }

        @Override
        public VectorSimilarityFunction similarityFunction() {
            return MutableASHVectors.this.similarityFunction();
        }

        @Override
        public boolean isExact() {
            return false;
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Symmetric (code-to-code) scoring

    /**
     * Scores node {@code a} against other nodes: code to code when both have the same model; otherwise
     * {@code a}'s vector, read from {@code vectors}, against the other node's code.
     */
    ScoreFunction.AdaptiveApproximateScoreFunction symmetricFunctionFor(int a, Supplier<RandomAccessVectorValues> vectors) {
        return new SymmetricScoreFunction(a, page(a), vectors);
    }

    /** Per-thread buffer that batch scoring gathers codes into; it grows to the largest batch. */
    private static final ThreadLocal<byte[][]> GATHER = ThreadLocal.withInitial(() -> new byte[][]{new byte[0]});

    private static byte[] gather(int bytes) {
        byte[][] holder = GATHER.get();
        if (holder[0].length < bytes) holder[0] = new byte[Math.max(bytes, 2 * holder[0].length)];
        return holder[0];
    }

    private final class SymmetricScoreFunction implements ScoreFunction.AdaptiveApproximateScoreFunction {
        private final int node;
        private final Page source;
        private final int sourceSlot;
        private final Model model;
        private final ASHScoreScale scoreScale;
        private final float scale;
        private final float centered;
        private final Supplier<RandomAccessVectorValues> vectors;
        private final boolean batch;
        private ScoreFunction.AdaptiveApproximateScoreFunction crossModel;

        SymmetricScoreFunction(int node, Page source, Supplier<RandomAccessVectorValues> vectors) {
            this.node = node;
            this.source = source;
            this.sourceSlot = node & pageMask;
            this.model = models[source.model];
            this.scoreScale = model.ash.scores();
            this.scale = source.scales[sourceSlot];
            this.centered = source.centered[sourceSlot];
            this.vectors = vectors;
            this.batch = simdSymmetric && bits != 1 && VU.supportsAshSymmetricDotRows();
        }

        private float codeDot(Page target, int slot) {
            if (simdSymmetric) {
                return bits == 1
                        ? VU.ashSymmetricBinaryDotAt(source.signs, sourceSlot * signWords, target.signs, slot * signWords, quantizedDim)
                        : VU.ashSymmetricDotAt(source.body, sourceSlot * bodyBytes, target.body, slot * bodyBytes, quantizedDim, bits);
            }
            return ASHSymmetricScorer.codeDot(source.signs, sourceSlot * signWords, source.body, sourceSlot * bodyBytes,
                    target.signs, slot * signWords, target.body, slot * bodyBytes, quantizedDim, bits);
        }

        /** The source vector against a node of another model. */
        private float crossModel(int target) {
            if (crossModel == null) crossModel = new QueryScoreFunction(vectors.get().getVector(node));
            return crossModel.similarityTo(target);
        }

        @Override
        public float similarityTo(int target) {
            Page page = page(target);
            if (page.model != model.id) return crossModel(target);
            int slot = target & pageMask;
            float dot = (scale * page.scales[slot]) * codeDot(page, slot)
                    + (centered + page.centered[slot]) + model.landmarkDot;
            return similarity(dot);
        }

        /** The similarity of a same-model estimate, in the source model's units. */
        private float similarity(float dot) {
            return scoreScale.similarity(dot);
        }

        @Override
        public VectorSimilarityFunction similarityFunction() {
            return model.ash.similarityFunction();
        }

        @Override
        public boolean supportsBatchScoring() {
            return batch;
        }

        /** Gathers same-model codes into one buffer and scores them in one call; same scores as one at a time. */
        @Override
        public void similarityTo(int[] nodes, int count, float[] scores) {
            // Without the rows kernel (1-bit codes, or no SIMD), score one at a time.
            if (!batch) {
                for (int i = 0; i < count; i++) scores[i] = similarityTo(nodes[i]);
                return;
            }
            byte[] rows = gather(count * bodyBytes);
            Page[] targets = new Page[count];
            int same = 0;
            for (int i = 0; i < count; i++) {
                Page page = page(nodes[i]);
                targets[i] = page;
                if (page.model == model.id) {
                    System.arraycopy(page.body, (nodes[i] & pageMask) * bodyBytes, rows, same * bodyBytes, bodyBytes);
                    same++;
                }
            }
            float[] dots = same == count ? scores : new float[same];
            if (same > 0) VU.ashSymmetricDotRows(source.body, sourceSlot * bodyBytes, rows, bodyBytes, same,
                    quantizedDim, bits, dots);
            for (int i = 0, k = 0; i < count; i++) {
                Page page = targets[i];
                if (page.model != model.id) {
                    scores[i] = crossModel(nodes[i]);
                    continue;
                }
                int slot = nodes[i] & pageMask;
                float dot = (scale * page.scales[slot]) * dots[k++] + (centered + page.centered[slot]) + model.landmarkDot;
                scores[i] = similarity(dot);
            }
        }
    }

    @Override
    public ScoreFunction.AdaptiveApproximateScoreFunction diversityFunctionFor(int node, VectorSimilarityFunction similarityFunction) {
        getCompressor().checkSimilarity(similarityFunction);
        return symmetricFunctionFor(node, () -> {
            throw new IllegalStateException("Scoring across models needs the source vectors; use buildScoreProvider");
        });
    }

    /** The largest length any model in use has encoded; with one model, that model's. */
    private float maxEncodedNorm() {
        float m = Float.NaN;
        for (int id : inUseIds) {
            float n = models[id].ash.maxEncodedNorm();
            if (!(n <= m)) m = n;
        }
        return Float.isNaN(m) ? models[current].ash.maxEncodedNorm() : m;
    }

    /**
     * Graph construction scoring over these codes: symmetric between codes of one model, and from the caller's
     * vector across models, as the class comment describes.
     *
     * @param vectors the caller's vectors, read only when scoring across models
     */
    public BuildScoreProvider buildScoreProvider(RandomAccessVectorValues vectors) {
        var views = vectors.threadLocalSupplier();
        return new BuildScoreProvider() {
            @Override
            public Optional<VectorSimilarityFunction> searchSimilarityFunction() {
                return Optional.of(similarityFunction());
            }

            @Override
            public boolean isExact() {
                return false;
            }

            @Override
            public VectorFloat<?> approximateCentroid() {
                return models[current].ash.landmarks[0].copy();
            }

            @Override
            public SearchScoreProvider searchProviderFor(VectorFloat<?> vector) {
                return new DefaultSearchScoreProvider(scoreFunctionFor(vector, similarityFunction()));
            }

            @Override
            public SearchScoreProvider searchProviderFor(int node) {
                float m = maxEncodedNorm();
                return new DefaultSearchScoreProvider(symmetricFunctionFor(node, views), null,
                        similarityFunction(), m * m);
            }

            @Override
            public SearchScoreProvider diversityProviderFor(int node) {
                return searchProviderFor(node);
            }
        };
    }

    // ---------------------------------------------------------------------------------------------
    // Snapshots and accounting

    /** A fixed copy of every vector added so far, in ordinal order; requires a single model in use. */
    public ASHVectors snapshot() {
        int n = count();
        int[] identity = new int[n];
        for (int i = 0; i < n; i++) identity[i] = i;
        return snapshot(identity);
    }

    /**
     * A fixed copy holding the vector at ordinal {@code newToOld[i]} at position {@code i}, for writing an index
     * whose ordinals were renumbered. An entry of {@link OrdinalMapper#OMITTED} marks a hole in the new ordinal
     * space and gets the all-zero code, as {@link #setZero} stores. Every vector must be encoded with one model:
     * finish moving pages to a newer model first.
     */
    public ASHVectors snapshot(int[] newToOld) {
        int[] inUse = inUseIds;
        if (inUse.length > 1)
            throw new IllegalStateException("A snapshot needs a single model; pages use models " + Arrays.toString(inUse));
        Model m = models[inUse.length == 0 ? current : inUse[0]];
        var codes = ASHCodes.allocate(newToOld.length, quantizedDim, bits);
        float[] centered = new float[newToOld.length];
        var code = AsymmetricHashing.QuantizedVector.createEmpty(quantizedDim, bits);
        int n = count();
        for (int i = 0; i < newToOld.length; i++) {
            if (newToOld[i] == OrdinalMapper.OMITTED) {
                codes.set(i, m.zero);
                centered[i] = m.zeroCenteredOffset;
                continue;
            }
            int old = Objects.checkIndex(newToOld[i], n);
            Page page = page(old);
            int slot = old & pageMask;
            if (signWords != 0) System.arraycopy(page.signs, slot * signWords, code.binaryVector, 0, signWords);
            if (bodyBytes != 0) System.arraycopy(page.body, slot * bodyBytes, code.extraBits, 0, bodyBytes);
            code.scale = page.scales[slot];
            code.offset = page.offsets[slot];
            code.landmark = page.landmarks[slot];
            codes.set(i, code);
            centered[i] = page.centered[slot];
        }
        return new ASHVectors(m.ash, codes, centered);
    }

    /** A copy of the stored code at {@code ordinal}. */
    public AsymmetricHashing.QuantizedVector get(int ordinal) {
        Objects.checkIndex(ordinal, count());
        Page page = page(ordinal);
        int slot = ordinal & pageMask;
        var code = AsymmetricHashing.QuantizedVector.createEmpty(quantizedDim, bits);
        if (signWords != 0) System.arraycopy(page.signs, slot * signWords, code.binaryVector, 0, signWords);
        if (bodyBytes != 0) System.arraycopy(page.body, slot * bodyBytes, code.extraBits, 0, bodyBytes);
        code.scale = page.scales[slot];
        code.offset = page.offsets[slot];
        code.landmark = page.landmarks[slot];
        return code;
    }

    /** The model id that encoded {@code ordinal}. */
    public int modelOf(int ordinal) {
        return page(ordinal).model;
    }

    /** Writes a {@link #snapshot()}, which needs a single model. */
    @Override
    public void write(IndexWriter out, int version) throws IOException {
        snapshot().write(out, version);
    }

    @Override
    public int getOriginalSize() {
        return originalDimension * Float.BYTES;
    }

    @Override
    public int getCompressedSize() {
        return models[current].ash.compressedVectorSize();
    }

    /** Pages, page table, models and this object. Does not allocate. */
    @Override
    public long ramBytesUsed() {
        long bytes = ramBytesUsed;
        for (Model m : models) bytes += m.ash.ramBytesUsed();
        return bytes;
    }

    /** The part of {@link #ramBytesUsed} held in native memory: the models' native projections. */
    public long nativeBytesUsed() {
        long bytes = 0;
        for (Model m : models) bytes += m.ash.nativeBytesUsed();
        return bytes;
    }

    /** The heap the models use, which pages of codes do not include. */
    public long modelHeapBytes() {
        long bytes = 0;
        for (Model m : models) bytes += m.ash.ramBytesUsed() - m.ash.nativeBytesUsed();
        return bytes;
    }

    @Override
    public String toString() {
        return "MutableASHVectors{count=" + count() + ", models=" + models.length + ", current=" + models[current].ash + '}';
    }
}
