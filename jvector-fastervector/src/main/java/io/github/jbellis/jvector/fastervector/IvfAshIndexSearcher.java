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

package io.github.jbellis.jvector.fastervector;

import io.github.jbellis.jvector.fastervector.ffi.FvecCapi;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * A reusable searcher that owns its scratch workspace.
 * <p>
 * Obtain via {@link IvfAshIndex#searcherBuilder()} or the one-shot
 * {@link IvfAshIndex#search(Vecs, int, int, FvecThreadPool)}.
 * <p>
 * {@link #close()} is idempotent.  The parent {@link IvfAshIndex} tracks all
 * live searchers and closes them before deleting the native index.
 * <p>
 * <b>Dim note:</b> for the build path the index dim is known and stored on this
 * object; queries whose {@link Vecs#dim()} differs are rejected.  For the open
 * path dim is not exposed by the C API (unless set via
 * {@link IvfAshIndex.OpenBuilder#dim(int)}), so the query dim is trusted.
 * <p>
 * <b>Threads:</b> a searcher runs one search at a time, as the native searcher requires; use
 * one searcher per thread, or {@link IvfAshBatchingSearcher} to share one. It may be created
 * on one thread and used on another.
 */
public final class IvfAshIndexSearcher implements AutoCloseable {

    private final int topk;
    /** Stored dim (0 if unknown — open path). Validated against the query dim when non-zero. */
    private final int dim;
    private final IvfAshIndex parent;

    /** Raw native pointer. Null if closed. */
    private volatile MemorySegment nativePtr;

    /*
     * Off-heap query and result buffers, kept between searches and grown to the largest batch
     * searched, so a search neither allocates nor zero-fills them. A shared arena, since the
     * searcher may be used on a thread other than the one that created it; it is replaced when
     * the buffers grow and closed by close(). Allocated on the first search.
     */
    private Arena buffers;
    private MemorySegment queryBuf;
    private MemorySegment labelsBuf;
    private MemorySegment scoresBuf;

    IvfAshIndexSearcher(MemorySegment nativePtr, int topk, int dim, IvfAshIndex parent) {
        this.nativePtr = nativePtr;
        this.topk = topk;
        this.dim = dim;
        this.parent = parent;
    }

    /** Number of nearest neighbours per query. */
    public int topk() {
        return topk;
    }

    /** The index dimension this searcher checks queries against, or 0 if it isn't known. */
    int dim() {
        return dim;
    }

    /**
     * Search all vectors of {@code queries}.
     * <p>
     * The queries are read through a fresh cursor into the searcher's off-heap row-major
     * {@code nq × dim} float32 buffer, which it keeps for later searches. The buffer and the
     * result buffers grow to the largest batch searched and are freed by {@link #close()}.
     *
     * @param queries query vectors; {@link Vecs#nvecs()} is the number of queries
     * @return {@link SearchResult} with one list of up to {@code topk} {@link NodeScore}s per
     *         query; empty slots (label {@code -1}) are truncated, so a list may be shorter
     *         than {@code topk}
     * @throws IllegalStateException    if this searcher is already closed
     * @throws IllegalArgumentException if {@code queries.dim()} does not match the index dim,
     *                                  or the cursor yields fewer vectors than {@code nvecs()}
     * @throws FvecException            if the native call fails
     */
    public SearchResult search(Vecs queries) {
        MemorySegment ptr = requireOpen();

        int qdim = queries.dim();
        if (dim != 0 && qdim != dim) {
            throw new IllegalArgumentException(
                    "query dim mismatch: index dim=" + dim + ", queries dim=" + qdim);
        }
        long nq = queries.nvecs();

        ensureBuffers(nq * qdim, nq * (long) topk);
        long read = BuildChunks.fillChunk(queries.getCursor(), qdim, nq, queryBuf);
        if (read != nq) {
            throw new IllegalArgumentException(
                    "queries cursor yielded " + read + " vectors, expected " + nq);
        }

        int rc = FvecCapi.fvec_ivfash_searcher_search(ptr, nq, queryBuf, labelsBuf, scoresBuf);
        if (rc != 0) {
            throw new FvecException("fvec_ivfash_searcher_search failed (rc=" + rc + ")");
        }

        // Each row is sized exactly and wrapped rather than copied: both lists are unmodifiable
        // views of arrays nothing else holds
        List<List<NodeScore>> results = new ArrayList<>((int) nq);
        for (long q = 0; q < nq; q++) {
            long base = q * topk;
            int n = 0;
            // Empty slots (label -1) come last, since results are best-first
            while (n < topk && labelsBuf.getAtIndex(ValueLayout.JAVA_LONG, base + n) != -1) {
                n++;
            }
            NodeScore[] row = new NodeScore[n];
            for (int k = 0; k < n; k++) {
                row[k] = new NodeScore(labelsBuf.getAtIndex(ValueLayout.JAVA_LONG, base + k),
                                       scoresBuf.getAtIndex(ValueLayout.JAVA_FLOAT, base + k));
            }
            results.add(Collections.unmodifiableList(Arrays.asList(row)));
        }
        return new SearchResult(Collections.unmodifiableList(results));
    }

    /**
     * Make the buffers hold at least {@code queryFloats} query floats and {@code resultSlots}
     * results. Growing replaces the arena; the old buffers are freed with it.
     */
    private void ensureBuffers(long queryFloats, long resultSlots) {
        if (buffers != null
                && queryBuf.byteSize() >= queryFloats * Float.BYTES
                && labelsBuf.byteSize() >= resultSlots * Long.BYTES) {
            return;
        }
        long floats = Math.max(queryFloats, queryBuf == null ? 0 : queryBuf.byteSize() / Float.BYTES);
        long slots = Math.max(resultSlots, labelsBuf == null ? 0 : labelsBuf.byteSize() / Long.BYTES);
        Arena grown = Arena.ofShared();
        MemorySegment q = grown.allocate(ValueLayout.JAVA_FLOAT, Math.max(floats, 1));
        MemorySegment l = grown.allocate(ValueLayout.JAVA_LONG, Math.max(slots, 1));
        MemorySegment s = grown.allocate(ValueLayout.JAVA_FLOAT, Math.max(slots, 1));
        if (buffers != null) {
            buffers.close();
        }
        buffers = grown;
        queryBuf = q;
        labelsBuf = l;
        scoresBuf = s;
    }

    /**
     * Free the buffers and delete the native searcher. Idempotent.
     *
     * @throws IllegalStateException if a search is running on another thread; the searcher
     *                               stays open
     */
    @Override
    public synchronized void close() {
        MemorySegment p = nativePtr;
        if (p != null) {
            // First, so a search in its native call makes this throw (the call keeps the arena
            // alive) instead of the native searcher being deleted under it
            if (buffers != null) {
                buffers.close();
                buffers = null;
            }
            nativePtr = null;
            FvecCapi.fvec_ivfash_searcher_delete(p);
            parent.onSearcherClosed(this);
        }
    }

    /** Returns the native pointer or throws if closed. */
    private MemorySegment requireOpen() {
        MemorySegment p = nativePtr;
        if (p == null) {
            throw new IllegalStateException("IvfAshIndexSearcher is already closed");
        }
        return p;
    }

    // ── Builder ────────────────────────────────────────────────────────────────

    /**
     * Builder for {@link IvfAshIndexSearcher}.
     * Obtain via {@link IvfAshIndex#searcherBuilder()}.
     */
    public static final class Builder {
        private final IvfAshIndex index;
        private Integer topk;
        private Integer nprobe;
        private FvecThreadPool searchThreadPool; // null = 1 thread

        Builder(IvfAshIndex index) {
            this.index = index;
        }

        /** Number of nearest neighbours per query. Required. */
        public Builder topk(int topk) {
            this.topk = topk;
            return this;
        }

        /** Number of coarse centroids to probe per query. Required. */
        public Builder nprobe(int nprobe) {
            this.nprobe = nprobe;
            return this;
        }

        /**
         * Thread pool for search. Only its thread count is used: each search call starts that
         * many I/O threads and scores on fastervector's shared scan pool of that size. If not
         * supplied, the searcher uses 1 thread (an internal 1-thread pool is passed, since the C
         * API's {@code NULL} would mean all CPUs at the pinned fastervector commit).
         */
        public Builder searchThreadPool(FvecThreadPool pool) {
            this.searchThreadPool = pool;
            return this;
        }

        /**
         * Build the searcher.
         *
         * @throws IllegalStateException if topk or nprobe are not set, or the index is closed
         * @throws FvecException         if the native call fails
         */
        public IvfAshIndexSearcher build() {
            if (topk == null) throw new IllegalStateException("topk must be set");
            if (nprobe == null) throw new IllegalStateException("nprobe must be set");
            return index.createSearcher(topk, nprobe, searchThreadPool);
        }
    }
}
