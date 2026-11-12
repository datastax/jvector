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
    /** Floats per page by default: 16 MiB. */
    static final int DEFAULT_PAGE_FLOATS = 1 << 22;
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
        switch (similarity) {
            case DOT_PRODUCT:
                return node -> (1 + VectorUtil.dotProduct(query, 0, page(node), offset(node), dimension)) / 2;
            case EUCLIDEAN:
                return node -> 1 / (1 + VectorUtil.squareL2Distance(query, 0, page(node), offset(node), dimension));
            case COSINE:
                return node -> (1 + VectorUtil.cosine(query, 0, page(node), offset(node), dimension)) / 2;
            default:
                throw new UnsupportedOperationException("Unsupported similarity function " + similarity);
        }
    }

    /** Exact scores of stored row {@code node1} against other stored rows, read in place. */
    public AnchoredScoreFunction scoreFunctionFor(int node1, VectorSimilarityFunction similarity) {
        return new AnchoredScoreFunction(similarity).anchorAt(node1);
    }

    /**
     * Exact scores of one stored row against the others. The anchor row can be moved with {@link #anchorAt}, so a
     * caller that scores against many anchors in turn, such as a diversity check, allocates one of these instead of
     * one per anchor. Not safe for concurrent use.
     */
    public final class AnchoredScoreFunction implements ScoreFunction.ExactScoreFunction {
        private final VectorSimilarityFunction similarity;
        private VectorFloat<?> anchorPage;
        private int anchorOffset;

        private AnchoredScoreFunction(VectorSimilarityFunction similarity) {
            switch (similarity) {
                case DOT_PRODUCT:
                case EUCLIDEAN:
                case COSINE:
                    break;
                default:
                    throw new UnsupportedOperationException("Unsupported similarity function " + similarity);
            }
            this.similarity = similarity;
        }

        /** Scores against {@code node1} from now on; returns this. */
        public AnchoredScoreFunction anchorAt(int node1) {
            anchorPage = page(node1);
            anchorOffset = offset(node1);
            return this;
        }

        /** True when this scores rows of {@code store} with {@code similarity}, so it can be re-anchored for that use. */
        public boolean scores(FlatVectorValues store, VectorSimilarityFunction similarity) {
            return store == FlatVectorValues.this && similarity == this.similarity;
        }

        @Override
        public float similarityTo(int node2) {
            var page2 = page(node2);
            int offset2 = offset(node2);
            switch (similarity) {
                case DOT_PRODUCT:
                    return (1 + VectorUtil.dotProduct(anchorPage, anchorOffset, page2, offset2, dimension)) / 2;
                case EUCLIDEAN:
                    return 1 / (1 + VectorUtil.squareL2Distance(anchorPage, anchorOffset, page2, offset2, dimension));
                default:
                    return (1 + VectorUtil.cosine(anchorPage, anchorOffset, page2, offset2, dimension)) / 2;
            }
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
