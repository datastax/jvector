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
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/**
 * Full-precision vectors stored back to back in a few large pages, so scoring reads a row in place: there is no
 * per-vector object to reach first, and neighboring rows share pages instead of being scattered across the heap.
 * <p>
 * Rows are padded to a multiple of 16 floats (64 bytes) and each page holds a power-of-two number of rows, so a
 * row's page and offset come from a shift and a mask. Every {@link VectorSimilarityFunction} is scored with the
 * existing offset-based kernels ({@link VectorUtil#dotProduct(VectorFloat, int, VectorFloat, int, int)},
 * {@link VectorUtil#squareL2Distance(VectorFloat, int, VectorFloat, int, int)} and
 * {@link VectorUtil#cosine(VectorFloat, int, VectorFloat, int, int)}), with the same score transforms as
 * {@link VectorSimilarityFunction#compare}.
 * <p>
 * Instances are immutable after {@link #copyOf} and safe for concurrent readers. {@link #getVector} returns a
 * fresh copy and is meant for occasional access; hot paths should use {@link #scoreFunctionFor}.
 */
public final class FlatVectorValues implements RandomAccessVectorValues {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    /**
     * Floats per page by default: 4 GiB, so most stores are one page and a batch of rows is one kernel call. Larger
     * stores split a batch by page.
     */
    static final int DEFAULT_PAGE_FLOATS = 1 << 30;
    private static final int ROW_ALIGNMENT_FLOATS = 16;

    private final int size;
    private final int dimension;
    private final int stride;
    private final int pageShift;
    private final int pageMask;
    private final VectorFloat<?>[] pages;

    private FlatVectorValues(int size, int dimension, int maxPageFloats) {
        if (dimension <= 0) throw new IllegalArgumentException("dimension must be positive: " + dimension);
        this.size = size;
        this.dimension = dimension;
        this.stride = (dimension + ROW_ALIGNMENT_FLOATS - 1) / ROW_ALIGNMENT_FLOATS * ROW_ALIGNMENT_FLOATS;
        int rowsPerPage = Integer.highestOneBit(Math.max(1, maxPageFloats / stride));
        this.pageShift = Integer.numberOfTrailingZeros(rowsPerPage);
        this.pageMask = rowsPerPage - 1;
        int pageCount = (int) (((long) size + rowsPerPage - 1) / rowsPerPage);
        this.pages = new VectorFloat<?>[pageCount];
        for (int p = 0; p < pageCount; p++) {
            int rows = Math.min(rowsPerPage, size - p * rowsPerPage);
            pages[p] = vts.createFloatVector(rows * stride);
        }
    }

    /** Copies every vector of {@code source} into a new flat store. */
    public static FlatVectorValues copyOf(RandomAccessVectorValues source) {
        return copyOf(source, DEFAULT_PAGE_FLOATS);
    }

    static FlatVectorValues copyOf(RandomAccessVectorValues source, int maxPageFloats) {
        var flat = new FlatVectorValues(source.size(), source.dimension(), maxPageFloats);
        for (int node = 0; node < flat.size; node++) {
            var v = source.getVector(node);
            if (v == null) throw new IllegalArgumentException("Missing vector for node " + node);
            flat.pages[node >>> flat.pageShift].copyFrom(v, 0, flat.offset(node), flat.dimension);
        }
        return flat;
    }

    private int offset(int node) {
        return (node & pageMask) * stride;
    }

    private VectorFloat<?> page(int node) {
        return pages[node >>> pageShift];
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int dimension() {
        return dimension;
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
     * Batch scoring shared by the score functions. With one page, node ids are the page's rows and the batch is one
     * kernel call. Otherwise the nodes are split by page, one call per page present, using buffers in {@code scratch}.
     */
    private void scoreMany(VectorSimilarityFunction similarity, VectorFloat<?> query, int[] nodes, int count,
                           float[] scores, BatchScratch scratch) {
        if (pages.length == 1) {
            scoreRows(similarity, query, pages[0], nodes, count, scores);
            return;
        }
        scratch.ensure(count);
        int pending = count;
        for (int i = 0; i < count; i++) scratch.pending[i] = i;
        while (pending > 0) {
            int page = nodes[scratch.pending[0]] >>> pageShift;
            int n = 0, kept = 0;
            for (int j = 0; j < pending; j++) {
                int position = scratch.pending[j];
                int node = nodes[position];
                if (node >>> pageShift == page) {
                    scratch.rows[n] = node & pageMask;
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

    /** Buffers for splitting a batch across pages; allocated only for stores of more than one page. */
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
        return String.format("FlatVectorValues(size=%d, dimension=%d, stride=%d, pages=%d)", size, dimension, stride, pages.length);
    }
}
