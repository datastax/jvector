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

import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Full-precision vectors stored back to back in a few large pages, so scoring reads a row in place: there is no
 * per-vector object to reach first, and neighboring rows share pages instead of being scattered across the heap.
 * <p>
 * Rows are padded to a multiple of 16 floats (64 bytes). Pages hold power-of-two row counts that double from a
 * first page up to a maximum and then stay at the maximum, so a row's page and offset come from a few bit
 * operations, and a growing store never holds more spare rows than it has rows, nor more than one page. Every
 * {@link VectorSimilarityFunction} is scored with the existing offset-based kernels
 * ({@link VectorUtil#dotProduct(VectorFloat, int, VectorFloat, int, int)},
 * {@link VectorUtil#squareL2Distance(VectorFloat, int, VectorFloat, int, int)} and
 * {@link VectorUtil#cosine(VectorFloat, int, VectorFloat, int, int)}), with the same score transforms as
 * {@link VectorSimilarityFunction#compare}; batches use the row-gather kernels. Pages are storage vectors
 * ({@link VectorTypeSupport#createStorageVector}), which a native backend places outside the Java heap, so a batch
 * whose rows span pages is still one kernel call through a table of the pages' addresses; other backends score a
 * batch one page at a time.
 * <p>
 * A store is either a fixed copy ({@link #copyOf}) or appendable ({@link #appendable}). An appendable store takes
 * rows with {@link #add} from any number of threads, each ordinal once, in any order; readers never lock. A row is
 * visible to another thread once its writer publishes the ordinal through a happens-before edge, as adding the node
 * to a concurrent graph does, so a graph can be built and searched while rows arrive. {@link #getVector} returns a
 * fresh copy and is meant for occasional access; hot paths should use {@link #scoreFunctionFor}.
 */
public final class FlatVectorValues implements BatchedVectorValues {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final VectorUtilSupport vus = VectorizationProvider.getInstance().getVectorUtilSupport();
    /**
     * Floats per page of a fixed copy by default: 4 GiB, so most copies are one page and a batch of rows is one
     * kernel call. Larger copies split a batch by page.
     */
    static final int DEFAULT_PAGE_FLOATS = 1 << 30;
    /**
     * Floats per full page of an appendable store by default: 64 MiB. A batch spanning pages is one kernel call on a
     * native backend, so the size only bounds the spare rows of a store that stops growing.
     */
    public static final int DEFAULT_APPENDABLE_PAGE_FLOATS = 1 << 24;
    /** Floats in the first page of an appendable store by default: 1 MiB, so a small store stays small. */
    static final int DEFAULT_FIRST_PAGE_FLOATS = 1 << 18;
    private static final int ROW_ALIGNMENT_FLOATS = 16;

    private final int dimension;
    private final int stride;
    /** log2 of the rows in the first page and in a full page. */
    private final int firstShift, fullShift;
    private final int fullMask;
    /** Whether {@link #add} is allowed. */
    private final boolean appendable;
    /** Every page up to the highest ordinal added; replaced, never mutated in place, when it grows. */
    private volatile VectorFloat<?>[] pages;
    /** The pages with the backend's handle for scoring rows across them in one call; published after pages. */
    private volatile PagedRows pagedRows;

    /** A set of pages and the {@link VectorUtilSupport#pagedRowTable} handle for exactly those pages. */
    private static final class PagedRows {
        final VectorFloat<?>[] pages;
        final Object table;

        PagedRows(VectorFloat<?>[] pages, Object table) {
            this.pages = pages;
            this.table = table;
        }
    }
    private final AtomicInteger rowCount = new AtomicInteger();
    private volatile long allocatedFloats;

    private FlatVectorValues(int dimension, int firstPageFloats, int fullPageFloats, boolean appendable) {
        if (dimension <= 0) throw new IllegalArgumentException("dimension must be positive: " + dimension);
        this.dimension = dimension;
        this.stride = (dimension + ROW_ALIGNMENT_FLOATS - 1) / ROW_ALIGNMENT_FLOATS * ROW_ALIGNMENT_FLOATS;
        int fullRows = rowsFor(fullPageFloats, stride);
        int firstRows = Math.min(rowsFor(firstPageFloats, stride), fullRows);
        this.firstShift = Integer.numberOfTrailingZeros(firstRows);
        this.fullShift = Integer.numberOfTrailingZeros(fullRows);
        this.fullMask = fullRows - 1;
        this.appendable = appendable;
        this.pages = new VectorFloat<?>[0];
    }

    private static int rowsFor(int pageFloats, int stride) {
        if (pageFloats <= 0) throw new IllegalArgumentException("page size must be positive: " + pageFloats);
        return Integer.highestOneBit(Math.max(1, pageFloats / stride));
    }

    /** Copies every vector of {@code source} into a new, fixed flat store. */
    public static FlatVectorValues copyOf(RandomAccessVectorValues source) {
        return copyOf(source, DEFAULT_PAGE_FLOATS);
    }

    static FlatVectorValues copyOf(RandomAccessVectorValues source, int maxPageFloats) {
        var flat = new FlatVectorValues(source.dimension(), maxPageFloats, maxPageFloats, false);
        int size = source.size();
        if (size > 0) flat.allocateThrough(flat.pageOf(size - 1), size);
        var pages = flat.pages;
        for (int node = 0; node < size; node++) {
            var v = source.getVector(node);
            if (v == null) throw new IllegalArgumentException("Missing vector for node " + node);
            pages[flat.pageOf(node)].copyFrom(v, 0, flat.offset(node), flat.dimension);
        }
        flat.rowCount.set(size);
        return flat;
    }

    /** An empty store that takes rows with {@link #add}, with default page sizes. */
    public static FlatVectorValues appendable(int dimension) {
        return appendable(dimension, DEFAULT_FIRST_PAGE_FLOATS, DEFAULT_APPENDABLE_PAGE_FLOATS);
    }

    /**
     * An empty store that takes rows with {@link #add}. Pages start at about {@code firstPageFloats} floats and
     * double up to about {@code fullPageFloats}; both are rounded down to a power-of-two row count of at least one.
     */
    public static FlatVectorValues appendable(int dimension, int firstPageFloats, int fullPageFloats) {
        return new FlatVectorValues(dimension, firstPageFloats, fullPageFloats, true);
    }

    /**
     * Stores {@code vector} as row {@code ordinal}. Safe to call from many threads at once, for distinct ordinals
     * in any order; each ordinal must be added once, before it is published to readers.
     *
     * @return the bytes of page memory this call allocated, usually zero
     */
    public long add(int ordinal, VectorFloat<?> vector) {
        if (!appendable) throw new UnsupportedOperationException("This flat store is a fixed copy");
        if (ordinal < 0) throw new IllegalArgumentException("ordinal must be nonnegative: " + ordinal);
        if (vector.length() != dimension) {
            throw new IllegalArgumentException("vector dimension " + vector.length() + " != " + dimension);
        }
        int page = pageOf(ordinal);
        var current = pages;
        long allocated = 0;
        if (page >= current.length) {
            allocated = allocateThrough(page, -1);
            current = pages;
        }
        current[page].copyFrom(vector, 0, offset(ordinal), dimension);
        rowCount.incrementAndGet();
        return allocated;
    }

    /**
     * Allocates every missing page up to {@code lastPage}, then publishes the larger directory. The last page of a
     * fixed copy holds only {@code size}'s rows; appendable pages are always full.
     *
     * @return the bytes allocated
     */
    private synchronized long allocateThrough(int lastPage, int size) {
        var current = pages;
        if (lastPage < current.length) return 0;
        var grown = java.util.Arrays.copyOf(current, lastPage + 1);
        long floats = 0;
        for (int p = current.length; p <= lastPage; p++) {
            int rows = rowsInPage(p);
            if (size >= 0 && p == lastPage) rows = size - firstRowOf(p);
            grown[p] = vts.createStorageVector(rows * stride);
            floats += (long) rows * stride;
        }
        allocatedFloats += floats;
        pages = grown;
        Object table = grown.length > 1 ? vus.pagedRowTable(grown) : null;
        pagedRows = table == null ? null : new PagedRows(grown, table);
        return floats * Float.BYTES;
    }

    /** Pages 0..k hold the first page's rows, then doubling counts up to a full page, then full pages. */
    private int pageOf(int node) {
        if (node < (1 << firstShift)) return 0;
        if (node >>> fullShift != 0) return (node >>> fullShift) + fullShift - firstShift;
        return 31 - Integer.numberOfLeadingZeros(node) - firstShift + 1;
    }

    private int rowOf(int node) {
        if (node < (1 << firstShift)) return node;
        if (node >>> fullShift != 0) return node & fullMask;
        return node - Integer.highestOneBit(node);
    }

    private int rowsInPage(int page) {
        int doublings = fullShift - firstShift;
        if (page == 0) return 1 << firstShift;
        return page <= doublings ? 1 << (firstShift + page - 1) : 1 << fullShift;
    }

    private int firstRowOf(int page) {
        int doublings = fullShift - firstShift;
        if (page == 0) return 0;
        if (page <= doublings) return 1 << (firstShift + page - 1);
        return (page - doublings) << fullShift;
    }

    private int offset(int node) {
        return rowOf(node) * stride;
    }

    private VectorFloat<?> page(int node) {
        return pages[pageOf(node)];
    }

    /** The rows added so far; for a fixed copy, all of them. */
    @Override
    public int size() {
        return rowCount.get();
    }

    @Override
    public int dimension() {
        return dimension;
    }

    /** Bytes of page memory allocated, including rows not yet added. */
    public long ramBytesUsed() {
        return allocatedFloats * Float.BYTES;
    }

    /** Returns a copy of the row; prefer {@link #scoreFunctionFor} or {@link #getVectorInto} on hot paths. */
    @Override
    public VectorFloat<?> getVector(int nodeId) {
        var v = vts.createFloatVector(dimension);
        v.copyFrom(page(nodeId), offset(nodeId), 0, dimension);
        return v;
    }

    @Override
    public void getVectorInto(int node, VectorFloat<?> destinationVector, int offset) {
        destinationVector.copyFrom(page(node), offset(node), offset, dimension);
    }

    /** Reads the rows in order into one vector that each step lends to the caller, so a scan allocates nothing. */
    @Override
    public VectorCursor openRange(int startInclusive, int endExclusive) {
        java.util.Objects.checkFromToIndex(startInclusive, endExclusive, size());
        return new RowCursor(null, startInclusive, endExclusive - startInclusive);
    }

    /** As {@link #openRange}, in request order and keeping duplicates. */
    @Override
    public VectorCursor openSelection(int[] ordinals, int offset, int count) {
        java.util.Objects.checkFromIndexSize(offset, count, ordinals.length);
        for (int i = offset; i < offset + count; i++) java.util.Objects.checkIndex(ordinals[i], size());
        return new RowCursor(ordinals, offset, count);
    }

    /** Lends each requested row in one reused vector, valid until the next step or close. */
    private final class RowCursor implements VectorCursor {
        private final int[] ordinals;
        private final int start;
        private final int count;
        private final VectorFloat<?> row = vts.createFloatVector(dimension);
        private int position = -1;
        private boolean closed;

        RowCursor(int[] ordinals, int start, int count) {
            this.ordinals = ordinals;
            this.start = start;
            this.count = count;
        }

        @Override
        public boolean next() {
            if (closed) throw new IllegalStateException("Cursor is closed");
            if (position == count) return false;
            if (++position == count) return false;
            getVectorInto(ordinal(), row, 0);
            return true;
        }

        @Override
        public int ordinal() {
            checkPosition();
            return ordinals == null ? start + position : ordinals[start + position];
        }

        @Override
        public VectorFloat<?> vector() {
            checkPosition();
            return row;
        }

        @Override
        public boolean isValueShared() {
            return true;
        }

        @Override
        public void close() {
            closed = true;
        }

        private void checkPosition() {
            if (closed || position < 0 || position >= count) throw new IllegalStateException("No current vector");
        }
    }

    @Override
    public boolean isValueShared() {
        return false;
    }

    @Override
    public FlatVectorValues copy() {
        return this;
    }

    /** Exact scores of {@code query} against stored rows, read in place. */
    public ScoreFunction.ExactScoreFunction scoreFunctionFor(VectorFloat<?> query, VectorSimilarityFunction similarity) {
        if (query.length() != dimension) {
            throw new IllegalArgumentException("query dimension " + query.length() + " != " + dimension);
        }
        return new QueryScoreFunction(query, similarity);
    }

    /** Exact scores of stored row {@code node1} against other stored rows, read in place. */
    public AnchoredScoreFunction scoreFunctionFor(int node1, VectorSimilarityFunction similarity) {
        return new AnchoredScoreFunction(similarity).anchorAt(node1);
    }

    private static void requireSupported(VectorSimilarityFunction similarity) {
        switch (similarity) {
            case DOT_PRODUCT:
            case EUCLIDEAN:
            case COSINE:
                return;
            default:
                throw new UnsupportedOperationException("Unsupported similarity function " + similarity);
        }
    }

    /** The score of one row: the raw kernel value at a row offset, with the {@link VectorSimilarityFunction#compare} transform. */
    private float score(VectorSimilarityFunction similarity, VectorFloat<?> a, int aOffset, int node) {
        var page = page(node);
        int offset = offset(node);
        switch (similarity) {
            case DOT_PRODUCT:
                return (1 + VectorUtil.dotProduct(a, aOffset, page, offset, dimension)) / 2;
            case EUCLIDEAN:
                return 1 / (1 + VectorUtil.squareL2Distance(a, aOffset, page, offset, dimension));
            default:
                return (1 + VectorUtil.cosine(a, aOffset, page, offset, dimension)) / 2;
        }
    }

    /** Raw kernel values of {@code query} against rows spread over pages, then the score transform. */
    private void scorePagedRows(VectorSimilarityFunction similarity, VectorFloat<?> query, PagedRows paged,
                                int[] pageOf, int[] rowInPage, int count, float[] out) {
        switch (similarity) {
            case DOT_PRODUCT:
                vus.dotProductPagedRows(query, paged.pages, paged.table, pageOf, rowInPage, stride, count, dimension, out);
                for (int i = 0; i < count; i++) out[i] = (1 + out[i]) / 2;
                return;
            case EUCLIDEAN:
                vus.squareDistancePagedRows(query, paged.pages, paged.table, pageOf, rowInPage, stride, count, dimension, out);
                for (int i = 0; i < count; i++) out[i] = 1 / (1 + out[i]);
                return;
            default:
                vus.cosinePagedRows(query, paged.pages, paged.table, pageOf, rowInPage, stride, count, dimension, out);
                for (int i = 0; i < count; i++) out[i] = (1 + out[i]) / 2;
        }
    }

    /** Raw kernel values of {@code query} against {@code count} rows of one page, then the score transform. */
    private void scoreRows(VectorSimilarityFunction similarity, VectorFloat<?> query, VectorFloat<?> page,
                           int[] rows, int count, float[] out) {
        switch (similarity) {
            case DOT_PRODUCT:
                VectorUtil.dotProductRows(query, page, stride, rows, count, dimension, out);
                for (int i = 0; i < count; i++) out[i] = (1 + out[i]) / 2;
                return;
            case EUCLIDEAN:
                VectorUtil.squareL2DistanceRows(query, page, stride, rows, count, dimension, out);
                for (int i = 0; i < count; i++) out[i] = 1 / (1 + out[i]);
                return;
            default:
                VectorUtil.cosineRows(query, page, stride, rows, count, dimension, out);
                for (int i = 0; i < count; i++) out[i] = (1 + out[i]) / 2;
        }
    }

    /**
     * Batch scoring shared by the score functions: one kernel call per page present in the batch, using buffers in
     * {@code scratch} to split the batch when there is more than one page.
     */
    private void scoreMany(VectorSimilarityFunction similarity, VectorFloat<?> query, int[] nodes, int count,
                           float[] scores, BatchScratch scratch) {
        var pages = this.pages;
        if (pages.length == 1) {
            // Every node is in page 0, where a node's row is the node itself.
            scoreRows(similarity, query, pages[0], nodes, count, scores);
            return;
        }
        if (count == 0) return;
        // The store may have grown past one page after the caller looked; a one-off buffer covers that race.
        if (scratch == null) scratch = new BatchScratch();
        scratch.ensure(count);
        var paged = pagedRows;
        if (paged != null) {
            // One kernel call for the whole batch, through the table of page addresses. A node the caller can see
            // was added after its page, and so after this table, was published.
            for (int i = 0; i < count; i++) {
                scratch.pending[i] = pageOf(nodes[i]);
                scratch.rows[i] = rowOf(nodes[i]);
            }
            scorePagedRows(similarity, query, paged, scratch.pending, scratch.rows, count, scores);
            return;
        }
        int pending = count;
        for (int i = 0; i < count; i++) scratch.pending[i] = i;
        while (pending > 0) {
            int page = pageOf(nodes[scratch.pending[0]]);
            int n = 0, kept = 0;
            for (int j = 0; j < pending; j++) {
                int position = scratch.pending[j];
                int node = nodes[position];
                if (pageOf(node) == page) {
                    scratch.rows[n] = rowOf(node);
                    scratch.positions[n++] = position;
                } else {
                    scratch.pending[kept++] = position;
                }
            }
            scoreRows(similarity, query, pages[page], scratch.rows, n, scratch.scores);
            for (int k = 0; k < n; k++) scores[scratch.positions[k]] = scratch.scores[k];
            pending = kept;
        }
    }

    /** Buffers for a batch whose rows span pages: page and row indexes, or the pieces of a batch split by page. */
    private static final class BatchScratch {
        int[] pending = new int[0];
        int[] rows = new int[0];
        int[] positions = new int[0];
        float[] scores = new float[0];

        void ensure(int count) {
            if (pending.length >= count) return;
            pending = new int[count];
            rows = new int[count];
            positions = new int[count];
            scores = new float[count];
        }
    }

    /** Exact scores of a query against stored rows; batches go to the row-gather kernels. Not safe for concurrent use. */
    private final class QueryScoreFunction implements ScoreFunction.ExactScoreFunction {
        private final VectorFloat<?> query;
        private final VectorSimilarityFunction similarity;
        private BatchScratch scratch;

        QueryScoreFunction(VectorFloat<?> query, VectorSimilarityFunction similarity) {
            requireSupported(similarity);
            this.query = query;
            this.similarity = similarity;
        }

        @Override
        public float similarityTo(int node2) {
            return score(similarity, query, 0, node2);
        }

        @Override
        public void similarityTo(int[] nodes, int count, float[] scores) {
            if (scratch == null && pages.length > 1) scratch = new BatchScratch();
            scoreMany(similarity, query, nodes, count, scores, scratch);
        }

        @Override
        public boolean supportsBatchScoring() {
            return true;
        }
    }

    /**
     * Exact scores of one stored row against the others. The anchor row can be moved with {@link #anchorAt}, so a
     * caller that scores against many anchors in turn, such as a diversity check, allocates one of these instead of
     * one per anchor. Not safe for concurrent use.
     */
    public final class AnchoredScoreFunction implements ScoreFunction.ExactScoreFunction {
        private final VectorSimilarityFunction similarity;
        private int anchor;
        private VectorFloat<?> anchorPage;
        private int anchorOffset;
        /** A copy of the anchor row for batch scoring, which reads its query from offset 0; made on first use. */
        private VectorFloat<?> anchorCopy;
        private boolean anchorCopied;
        private BatchScratch scratch;

        private AnchoredScoreFunction(VectorSimilarityFunction similarity) {
            requireSupported(similarity);
            this.similarity = similarity;
        }

        /** Scores against {@code node1} from now on; returns this. */
        public AnchoredScoreFunction anchorAt(int node1) {
            anchor = node1;
            anchorPage = page(node1);
            anchorOffset = offset(node1);
            anchorCopied = false;
            return this;
        }

        /** True when this scores rows of {@code store} with {@code similarity}, so it can be re-anchored for that use. */
        public boolean scores(FlatVectorValues store, VectorSimilarityFunction similarity) {
            return store == FlatVectorValues.this && similarity == this.similarity;
        }

        @Override
        public float similarityTo(int node2) {
            return score(similarity, anchorPage, anchorOffset, node2);
        }

        @Override
        public void similarityTo(int[] nodes, int count, float[] scores) {
            if (!anchorCopied) {
                if (anchorCopy == null) anchorCopy = vts.createFloatVector(dimension);
                getVectorInto(anchor, anchorCopy, 0);
                anchorCopied = true;
            }
            if (scratch == null && pages.length > 1) scratch = new BatchScratch();
            scoreMany(similarity, anchorCopy, nodes, count, scores, scratch);
        }

        @Override
        public boolean supportsBatchScoring() {
            return true;
        }
    }

    @Override
    public ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf) {
        return scoreFunctionFor(queryVector, vsf);
    }

    @Override
    public String toString() {
        return String.format("FlatVectorValues(size=%d, dimension=%d, stride=%d, pages=%d)", size(), dimension, stride, pages.length);
    }
}
