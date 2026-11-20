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

import java.nio.ByteBuffer;

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * One target source's part of a cell walk in {@link OnDiskGraphIndexCompactor}'s full-precision merge, as handed to a
 * {@link CompactionAccelerator}.
 * <p>
 * The walk has {@link #queryCount()} queries: the vectors of a range of new ordinals of the source being merged. Each
 * query probes some cells of the target source, and each cell is a contiguous range of new ordinals
 * ({@link #cellStart} to {@link #cellEnd}) whose raw vectors {@link #copyRows} reads from the merge's scratch region.
 * {@link #pairs()} lists every (cell, query) pair, sorted by cell. For each query, the result is the {@link #topK()}
 * best (score, new ordinal) over all members of all cells it probes, set with {@link #setTopK}.
 */
@Experimental
public final class CellWalk {

    private final ExactJoin join;
    private final int targetIndex;
    private final VectorFloat<?>[] queries;
    private final long[] pairs;
    private final int pairCount;
    private final int[] cellStarts;
    private final PreEncodedCodeCache rows;
    private final int dimension;
    private final VectorSimilarityFunction similarity;

    CellWalk(ExactJoin join, int targetIndex, VectorFloat<?>[] queries, long[] pairs, int pairCount, int[] cellStarts,
             PreEncodedCodeCache rows, int dimension, VectorSimilarityFunction similarity) {
        this.join = join;
        this.targetIndex = targetIndex;
        this.queries = queries;
        this.pairs = pairs;
        this.pairCount = pairCount;
        this.cellStarts = cellStarts;
        this.rows = rows;
        this.dimension = dimension;
        this.similarity = similarity;
    }

    /** Number of queries; query indices run from 0. */
    public int queryCount() {
        return join.n;
    }

    public int dimension() {
        return dimension;
    }

    public VectorSimilarityFunction similarity() {
        return similarity;
    }

    /** Candidates kept per query. */
    public int topK() {
        return join.k;
    }

    /** Copies query {@code i}'s vector into {@code dst}. A query is null (and skipped here) when its node has no cell. */
    public boolean query(int i, float[] dst) {
        VectorFloat<?> q = queries[i];
        if (q == null) {
            return false;
        }
        for (int d = 0; d < dimension; d++) {
            dst[d] = q.get(d);
        }
        return true;
    }

    /** The (cell, query) pairs, {@code (long) cell << 32 | query}, sorted; only the first {@link #pairCount()} are used. */
    public long[] pairs() {
        return pairs;
    }

    public int pairCount() {
        return pairCount;
    }

    /** Number of cells; cell indices run from 0. */
    public int cellCount() {
        return cellStarts.length - 1;
    }

    /** First new ordinal of {@code cell}'s members in the target. */
    public int cellStart(int cell) {
        return cellStarts[cell];
    }

    /** One past the last new ordinal of {@code cell}'s members in the target. */
    public int cellEnd(int cell) {
        return cellStarts[cell + 1];
    }

    /**
     * Copies the raw vectors of new ordinals {@code [fromOrdinal, fromOrdinal + count)}, as little-endian float32
     * rows of {@link #dimension()} values, into {@code dst} at its position, advancing it.
     */
    public void copyRows(int fromOrdinal, int count, ByteBuffer dst) {
        rows.copyRows(fromOrdinal, count, dst);
    }

    /**
     * JVector's score of a query and a member from their dot product and squared norms: {@code (1 + dot) / 2} for
     * DOT_PRODUCT, {@code (1 + cos) / 2} for COSINE, {@code 1 / (1 + squared distance)} for EUCLIDEAN.
     */
    public float score(float dot, float queryNorm2, float memberNorm2) {
        switch (similarity) {
            case DOT_PRODUCT:
                return (1 + dot) / 2;
            case EUCLIDEAN:
                return 1 / (1 + Math.max(0f, queryNorm2 + memberNorm2 - 2 * dot));
            default:
                return (1 + dot / (float) Math.sqrt(Math.max(1e-12, (double) queryNorm2 * memberNorm2))) / 2;
        }
    }

    /**
     * Sets query {@code i}'s candidates: {@code count} keys from {@code keys[from]}, each
     * {@code (long) Float.floatToRawIntBits(score) << 32 | newOrdinal}, at most {@link #topK()}, in any order.
     * Calls for different queries may run concurrently.
     */
    public void setTopK(int i, long[] keys, int from, int count) {
        if (count > join.k) {
            throw new IllegalArgumentException(count + " candidates for a top " + join.k);
        }
        int slot = targetIndex * join.n + i;
        System.arraycopy(keys, from, join.heaps, slot * join.k, count);
        join.sizes[slot] = count;
        join.thresholds[slot] = count == join.k ? Float.intBitsToFloat((int) (min(keys, from, count) >>> 32)) : Float.NEGATIVE_INFINITY;
    }

    private static long min(long[] keys, int from, int count) {
        long m = Long.MAX_VALUE;
        for (int e = from; e < from + count; e++) {
            m = Math.min(m, keys[e]);
        }
        return m;
    }
}
