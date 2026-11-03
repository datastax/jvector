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

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A built IVF-ASH index stored on SSD.
 * <p>
 * Construct via {@link #builder()} (new build) or {@link #prepareOpen()} (reload).
 * <p>
 * <b>Lifetime rule:</b> all {@link IvfAshIndexSearcher}s derived from this
 * index must be closed before — or concurrently with — this index.  {@link #close()}
 * closes all outstanding searchers first (they are tracked internally), then
 * deletes the native index.  {@link #close()} is idempotent.
 */
public final class IvfAshIndex implements AutoCloseable {

    /** Raw native pointer to the {@code IvfAshHandle}. Null if closed. */
    private volatile MemorySegment nativePtr;

    /** Dimension of the dataset; 0 if unknown (opened without knowing). */
    private final int dim;

    /** Live searchers; closed before the index is destroyed. */
    private final Set<IvfAshIndexSearcher> liveSearchers =
            Collections.newSetFromMap(new ConcurrentHashMap<>());

    private IvfAshIndex(MemorySegment nativePtr, int dim) {
        this.nativePtr = nativePtr;
        this.dim = dim;
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /**
     * Close all outstanding searchers then delete the native index.
     * Idempotent.
     */
    @Override
    public synchronized void close() {
        if (nativePtr == null) return;
        // Close live searchers first (the native searcher holds its own Arc to the
        // index, so this is not required for memory safety, but frees resources eagerly)
        for (IvfAshIndexSearcher s : new ArrayList<>(liveSearchers)) {
            s.close();
        }
        MemorySegment p = nativePtr;
        nativePtr = null;
        FvecCapi.fvec_ivfash_index_delete(p);
    }

    /** Called by IvfAshIndexSearcher when it is closed. */
    void onSearcherClosed(IvfAshIndexSearcher searcher) {
        liveSearchers.remove(searcher);
    }

    /** Returns the native pointer or throws if closed. */
    MemorySegment requireOpen() {
        MemorySegment p = nativePtr;
        if (p == null) {
            throw new IllegalStateException("IvfAshIndex is already closed");
        }
        return p;
    }

    // ── Searcher creation ──────────────────────────────────────────────────────

    /**
     * Start building a searcher for this index.
     * <pre>
     *   var searcher = index.searcherBuilder()
     *       .topk(10).nprobe(32).build();
     * </pre>
     */
    public IvfAshIndexSearcher.Builder searcherBuilder() {
        return new IvfAshIndexSearcher.Builder(this);
    }

    /**
     * One-shot search: create a searcher, run all queries, return results.
     * The temporary searcher and its internal 1-thread pool (if {@code pool} is null)
     * are closed after the call.
     *
     * @param queries query vectors; {@link Vecs#nvecs()} is the number of queries
     * @param topk    neighbours per query
     * @param nprobe  centroids to probe per query
     * @param pool    thread pool, or {@code null} for a 1-thread pool
     */
    public SearchResult search(Vecs queries, int topk, int nprobe, FvecThreadPool pool) {
        try (IvfAshIndexSearcher searcher = searcherBuilder()
                .topk(topk).nprobe(nprobe).searchThreadPool(pool).build()) {
            return searcher.search(queries);
        }
    }

    /**
     * Create a new searcher.  Internal method called by the builder.
     * The capi.rs doc says the pool pointer is only read during this call
     * (only {@code nthreads()} is consumed), so the caller may release the
     * pool immediately after; the searcher does NOT own the passed-in pool.
     */
    IvfAshIndexSearcher createSearcher(int topk, int nprobe, FvecThreadPool userPool) {
        MemorySegment indexPtr = requireOpen();
        // Rust unwrap_or(1): if no pool, create 1-thread pool internally.
        // Passing NULL to C API means "all CPUs", so we must NOT pass null here.
        boolean ownPool = (userPool == null);
        FvecThreadPool pool = ownPool ? FvecThreadPool.create(1) : userPool;
        try {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment out = arena.allocate(ValueLayout.ADDRESS);
                int rc = FvecCapi.fvec_ivfash_index_create_searcher(
                        indexPtr, topk, nprobe, pool.nativePtr(), out);
                if (rc != 0) {
                    throw new FvecException("fvec_ivfash_index_create_searcher failed (rc=" + rc + ")");
                }
                MemorySegment searcherPtr = out.get(ValueLayout.ADDRESS, 0);
                IvfAshIndexSearcher searcher = new IvfAshIndexSearcher(
                        searcherPtr, topk, dim, this);
                liveSearchers.add(searcher);
                return searcher;
            }
        } finally {
            // Pool pointer was only read during creation; we can release the
            // internal pool immediately. User-provided pools are not owned here.
            if (ownPool) {
                pool.close();
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Build builder
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Start building a new index.
     * <pre>
     *   IvfAshIndex index = IvfAshIndex.builder()
     *       .nlist(256)
     *       .ssdPaths(List.of(Path.of("/mnt/ssd")))
     *       .vectors(myVecs)
     *       .build();
     * </pre>
     */
    public static BuildBuilder builder() {
        return new BuildBuilder();
    }

    /** Builder for constructing a new IVF-ASH index from a {@link Vecs} dataset. */
    public static final class BuildBuilder {
        private Integer nlist;
        private float storageFactor = 1.0f;
        private List<Path> ssdPaths;
        private String subfolder = "";
        private FvecThreadPool buildThreadPool; // null = use NULL (all CPUs)
        private Vecs vectors;

        private BuildBuilder() {}

        /** Number of IVF coarse centroids. Required. */
        public BuildBuilder nlist(int nlist) {
            this.nlist = nlist;
            return this;
        }

        /** Compression level. Default 1.0 (quant_dim = dim/2, 32× compression). */
        public BuildBuilder storageFactor(float storageFactor) {
            this.storageFactor = storageFactor;
            return this;
        }

        /** SSD mount directories (at least one required). */
        public BuildBuilder ssdPaths(List<Path> paths) {
            this.ssdPaths = paths;
            return this;
        }

        /** Optional subfolder within each SSD mount. Default "". */
        public BuildBuilder subfolder(String subfolder) {
            this.subfolder = subfolder;
            return this;
        }

        /**
         * Thread pool for the build.  If not supplied, {@code NULL} is passed
         * to the C API, which uses all available CPU cores (matching Rust's
         * {@code available_parallelism} behaviour — confirmed via
         * {@code capi.rs} {@code num_procs()}).
         */
        public BuildBuilder buildThreadPool(FvecThreadPool pool) {
            this.buildThreadPool = pool;
            return this;
        }

        /** Dataset to build from. Required. */
        public BuildBuilder vectors(Vecs vectors) {
            this.vectors = vectors;
            return this;
        }

        /**
         * Execute the three-pass streaming build and return the ready index.
         *
         * @throws IllegalStateException if required fields are missing
         * @throws FvecException         if a native call fails
         * @throws IOException           if SSD directories cannot be created
         */
        public IvfAshIndex build() throws IOException {
            // Validate required fields (before native load — these throw IllegalStateException)
            if (nlist == null) throw new IllegalStateException("nlist must be set");
            if (vectors == null) throw new IllegalStateException("vectors must be set");
            if (ssdPaths == null || ssdPaths.isEmpty()) {
                throw new IllegalStateException("ssdPaths must contain at least one path");
            }
            NativeHolder.ensureLoaded();

            long nvecs = vectors.nvecs();
            int dim = vectors.dim();
            long chunkSize = BuildChunks.effectiveChunkSize();

            // Resolve SSD paths: create dirs, build dev/manifest paths
            List<String> devPaths = new ArrayList<>(ssdPaths.size());
            for (int i = 0; i < ssdPaths.size(); i++) {
                Path dir = ssdPaths.get(i).resolve(subfolder);
                Files.createDirectories(dir);
                devPaths.add(dir.resolve("dev" + i + ".ash.blk").toString());
            }
            String manifestPath = ssdPaths.get(0).resolve(subfolder)
                    .resolve("dev0.ash.blk.manifest").toString();

            long numChunks = BuildChunks.numChunks(nvecs, chunkSize);

            // Pre-allocate the reusable off-heap chunk buffer.
            // Size = min(chunkSize, nvecs) × dim × 4 bytes.
            // Using ofShared() so it can be passed to confined arenas' close without
            // issues, and because we reuse it across multiple passes.
            long bufVecs = Math.min(chunkSize, Math.max(nvecs, 1));
            try (Arena chunkArena = Arena.ofShared()) {
                MemorySegment chunkBuf = chunkArena.allocate((long) dim * bufVecs * Float.BYTES);

                // ── Pass 0: train on first chunk ──────────────────────────────────
                System.out.println(">> [Pass 0] Reading chunk 0 for IVF-ASH training\u2026");
                VecsCursor cursor0 = vectors.getCursor();
                long n0 = BuildChunks.fillChunk(cursor0, dim, chunkSize, chunkBuf);
                if (n0 == 0) {
                    throw new FvecException("dataset is empty — cannot train");
                }
                System.out.println(">> [Pass 0] Training on " + n0 + " vectors\u2026");

                MemorySegment poolPtr = buildThreadPool != null
                        ? buildThreadPool.nativePtr() : MemorySegment.NULL;

                // Build string segments in an arena that lasts for the call
                try (Arena strArena = Arena.ofConfined()) {
                    MemorySegment manifestSeg = strArena.allocateFrom(manifestPath);
                    MemorySegment devPathsArray = allocateDevPathsArray(strArena, devPaths);

                    MemorySegment phase0Out = strArena.allocate(ValueLayout.ADDRESS);
                    int rc = FvecCapi.fvec_ivfash_create_builder_phase0(
                            dim, (long) nlist, storageFactor, poolPtr,
                            n0, chunkBuf,
                            manifestSeg, devPaths.size(), devPathsArray,
                            phase0Out);
                    if (rc != 0) {
                        throw new FvecException("fvec_ivfash_create_builder_phase0 failed (rc=" + rc + ")");
                    }
                    MemorySegment phase0Handle = phase0Out.get(ValueLayout.ADDRESS, 0);
                    System.out.println(">> [Pass 0] Training complete.");

                    // ── Pass 1: coarse assignment ─────────────────────────────────
                    System.out.println(">> [Pass 1] Coarse assignment pass\u2026");
                    VecsCursor cursor1 = vectors.getCursor();
                    long chunkIdx = 0;
                    while (true) {
                        long nVec = BuildChunks.fillChunk(cursor1, dim, chunkSize, chunkBuf);
                        if (nVec == 0) break;
                        rc = FvecCapi.fvec_ivfash_builder_phase0_assign_chunk(
                                phase0Handle, nVec, chunkBuf);
                        if (rc != 0) {
                            // Handle is still valid on assign_chunk failure (not consuming)
                            throw new FvecException(
                                    "fvec_ivfash_builder_phase0_assign_chunk failed (rc=" + rc + ")");
                        }
                        chunkIdx++;
                        System.out.print("\r   [Pass 1] " + chunkIdx + "/" + numChunks);
                        System.out.flush();
                    }
                    System.out.println();
                    System.out.println(">> [Pass 1] Assignment complete.");

                    // Transition Phase 0 → Phase 1 (consumes phase0Handle unconditionally)
                    MemorySegment phase1Out = strArena.allocate(ValueLayout.ADDRESS);
                    // Must null out phase0Handle before this call — it is consumed even on failure
                    rc = FvecCapi.fvec_ivfash_builder_phase0_done_assign(phase0Handle, phase1Out);
                    if (rc != 0) {
                        // phase0Handle is now invalid; abandon (documented leak)
                        throw new FvecException(
                                "fvec_ivfash_builder_phase0_done_assign failed (rc=" + rc + ")");
                    }
                    MemorySegment phase1Handle = phase1Out.get(ValueLayout.ADDRESS, 0);

                    // ── Pass 2: encode + stream to SSD ────────────────────────────
                    System.out.println(">> [Pass 2] Encode & SSD stream pass\u2026");
                    VecsCursor cursor2 = vectors.getCursor();
                    chunkIdx = 0;
                    while (true) {
                        long nVec = BuildChunks.fillChunk(cursor2, dim, chunkSize, chunkBuf);
                        if (nVec == 0) break;
                        rc = FvecCapi.fvec_ivfash_builder_phase1_add(
                                phase1Handle, nVec, chunkBuf);
                        if (rc != 0) {
                            // Not consuming — but we abandon on error anyway
                            throw new FvecException(
                                    "fvec_ivfash_builder_phase1_add failed (rc=" + rc + ")");
                        }
                        chunkIdx++;
                        System.out.print("\r   [Pass 2] " + chunkIdx + "/" + numChunks);
                        System.out.flush();
                    }
                    System.out.println();

                    // Finalize (consumes phase1Handle unconditionally)
                    MemorySegment indexOut = strArena.allocate(ValueLayout.ADDRESS);
                    rc = FvecCapi.fvec_ivfash_builder_phase1_done(phase1Handle, indexOut);
                    if (rc != 0) {
                        throw new FvecException(
                                "fvec_ivfash_builder_phase1_done failed (rc=" + rc + ")");
                    }
                    MemorySegment indexPtr = indexOut.get(ValueLayout.ADDRESS, 0);
                    System.out.println(">> SSD index built successfully.");
                    return new IvfAshIndex(indexPtr, dim);
                }
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Open builder
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Start the open (reload) path.
     * <pre>
     *   IvfAshIndex index = IvfAshIndex.prepareOpen()
     *       .manifestPath("/mnt/ssd/dev0.ash.blk.manifest")
     *       .devPaths(List.of("/mnt/ssd/dev0.ash.blk"))
     *       .open();
     * </pre>
     */
    public static OpenBuilder prepareOpen() {
        return new OpenBuilder();
    }

    /** Builder for reopening an existing index from disk. */
    public static final class OpenBuilder {
        private String manifestPath;
        private List<String> devPaths;
        private FvecThreadPool buildThreadPool; // null = use NULL (all CPUs)
        private int dim; // 0 = unknown (derived from the query buffer at search time)

        private OpenBuilder() {}

        /**
         * Optional vector dimension of the index.  The C API does not expose it, so
         * without this the searcher trusts the dim of the query {@link Vecs}.
         * When set, the searcher rejects queries whose {@link Vecs#dim()} differs.
         *
         * @throws IllegalArgumentException if {@code dim <= 0}
         */
        public OpenBuilder dim(int dim) {
            if (dim <= 0) {
                throw new IllegalArgumentException("dim must be positive, got " + dim);
            }
            this.dim = dim;
            return this;
        }

        /** Configured dim; 0 if not set. */
        int dim() {
            return dim;
        }

        /** Path to the {@code .manifest} file. Required. */
        public OpenBuilder manifestPath(String manifestPath) {
            this.manifestPath = manifestPath;
            return this;
        }

        /** Per-device {@code .ash.blk} paths in build order. Required. */
        public OpenBuilder devPaths(List<String> devPaths) {
            this.devPaths = devPaths;
            return this;
        }

        /**
         * Thread pool for coarse-probe at search time.
         * If not supplied, {@code NULL} is passed (all CPUs).
         */
        public OpenBuilder buildThreadPool(FvecThreadPool pool) {
            this.buildThreadPool = pool;
            return this;
        }

        /**
         * Open the index.
         *
         * @throws IllegalStateException if required fields are missing
         * @throws FvecException         if the native call fails
         */
        public IvfAshIndex open() {
            if (manifestPath == null || manifestPath.isBlank()) {
                throw new IllegalStateException("manifestPath must be set");
            }
            if (devPaths == null || devPaths.isEmpty()) {
                throw new IllegalStateException("devPaths must contain at least one path");
            }
            NativeHolder.ensureLoaded();

            System.out.println(">> [open] Reconstructing index from " + manifestPath + "\u2026");

            MemorySegment poolPtr = buildThreadPool != null
                    ? buildThreadPool.nativePtr() : MemorySegment.NULL;

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment manifestSeg = arena.allocateFrom(manifestPath);
                MemorySegment devPathsArray = allocateDevPathsArray(arena, devPaths);
                MemorySegment out = arena.allocate(ValueLayout.ADDRESS);

                int rc = FvecCapi.fvec_ivfash_index_open(
                        manifestSeg, devPaths.size(), devPathsArray, poolPtr, out);
                if (rc != 0) {
                    throw new FvecException("fvec_ivfash_index_open failed (rc=" + rc + ")");
                }
                MemorySegment indexPtr = out.get(ValueLayout.ADDRESS, 0);
                System.out.println(">> [open] Index loaded.");
                // dim = 0 unless set via dim(int): the C API does not expose it
                return new IvfAshIndex(indexPtr, dim);
            }
        }
    }

    // ── Shared utility ─────────────────────────────────────────────────────────

    /**
     * Allocate a null-terminated C string array in {@code arena} from a Java list.
     * Returns a {@code const char *const *} MemorySegment suitable for the C API.
     */
    private static MemorySegment allocateDevPathsArray(Arena arena, List<String> paths) {
        int n = paths.size();
        MemorySegment arr = arena.allocate(ValueLayout.ADDRESS.byteSize() * n);
        for (int i = 0; i < n; i++) {
            MemorySegment strSeg = arena.allocateFrom(paths.get(i));
            arr.setAtIndex(ValueLayout.ADDRESS, i, strSeg);
        }
        return arr;
    }
}
