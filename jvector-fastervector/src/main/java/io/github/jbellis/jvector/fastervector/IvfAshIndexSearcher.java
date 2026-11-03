package io.github.jbellis.jvector.fastervector;

import io.github.jbellis.jvector.fastervector.ffi.FvecCapi;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
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
 */
public final class IvfAshIndexSearcher implements AutoCloseable {

    private final int topk;
    /** Stored dim (0 if unknown — open path). Validated against the query dim when non-zero. */
    private final int dim;
    private final IvfAshIndex parent;

    /** Raw native pointer. Null if closed. */
    private volatile MemorySegment nativePtr;

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

    /**
     * Search all vectors of {@code queries}.
     * <p>
     * The queries are read through a fresh cursor into a temporary off-heap
     * row-major {@code nq × dim} float32 buffer, which is released before returning.
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

        long resultLen = nq * (long) topk;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment qvecs = arena.allocate(nq * qdim * Float.BYTES);
            long read = BuildChunks.fillChunk(queries.getCursor(), qdim, nq, qvecs);
            if (read != nq) {
                throw new IllegalArgumentException(
                        "queries cursor yielded " + read + " vectors, expected " + nq);
            }
            MemorySegment labelsBuf = arena.allocate(resultLen * Long.BYTES);
            MemorySegment scoresBuf = arena.allocate(resultLen * Float.BYTES);

            int rc = FvecCapi.fvec_ivfash_searcher_search(ptr, nq, qvecs, labelsBuf, scoresBuf);
            if (rc != 0) {
                throw new FvecException("fvec_ivfash_searcher_search failed (rc=" + rc + ")");
            }

            List<List<NodeScore>> results = new ArrayList<>((int) nq);
            for (long q = 0; q < nq; q++) {
                List<NodeScore> row = new ArrayList<>(topk);
                for (int k = 0; k < topk; k++) {
                    long i = q * topk + k;
                    long label = labelsBuf.get(ValueLayout.JAVA_LONG, i * Long.BYTES);
                    if (label == -1) {
                        break; // empty slot; results are best-first so the rest are empty too
                    }
                    row.add(new NodeScore(label, scoresBuf.get(ValueLayout.JAVA_FLOAT, i * Float.BYTES)));
                }
                results.add(List.copyOf(row));
            }
            return new SearchResult(List.copyOf(results));
        }
    }

    /**
     * Delete the native searcher. Idempotent.
     */
    @Override
    public synchronized void close() {
        MemorySegment p = nativePtr;
        if (p != null) {
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
        private FvecThreadPool searchThreadPool; // null = use internal 1-thread pool

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
         * Thread pool for search.  If not supplied, an internal 1-thread pool
         * is created and released right after searcher creation (matching the
         * Rust behaviour of {@code unwrap_or(1)}).
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
