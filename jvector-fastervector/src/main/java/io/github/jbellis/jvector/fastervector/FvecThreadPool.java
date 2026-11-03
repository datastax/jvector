package io.github.jbellis.jvector.fastervector;

import io.github.jbellis.jvector.fastervector.ffi.FvecCapi;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * A native thread pool wrapping {@code FvecThreadPool}.
 * <p>
 * Create with {@link #create(int)} and close when done. Implements
 * {@link AutoCloseable}; {@link #close()} is idempotent.
 */
public final class FvecThreadPool implements AutoCloseable {

    private final int nthreads;
    /** Raw pointer to the native {@code FvecThreadPool}. Null if closed. */
    private volatile MemorySegment nativePtr;

    private FvecThreadPool(int nthreads, MemorySegment nativePtr) {
        this.nthreads = nthreads;
        this.nativePtr = nativePtr;
    }

    /**
     * Create a new thread pool with the given number of worker threads.
     *
     * @param nthreads positive thread count
     * @return a new {@code FvecThreadPool}
     * @throws FvecException if the native call fails
     */
    public static FvecThreadPool create(int nthreads) {
        NativeHolder.ensureLoaded();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(ValueLayout.ADDRESS);
            int rc = FvecCapi.fvec_create_thread_pool(nthreads, out);
            if (rc != 0) {
                throw new FvecException("fvec_create_thread_pool failed (rc=" + rc + ")");
            }
            MemorySegment ptr = out.get(ValueLayout.ADDRESS, 0);
            return new FvecThreadPool(nthreads, ptr);
        }
    }

    /** Number of threads this pool was created with. */
    public int nthreads() {
        return nthreads;
    }

    /**
     * Returns the raw native pointer (for passing to other C-API functions).
     * Package-private — not part of the public API.
     *
     * It is the caller's responsibility to ensure that the pool is not closed
     * while the pointer is in use, as this can lead to native memory corruption.
     *
     * @throws IllegalStateException if already closed
     */
    MemorySegment nativePtr() {
        MemorySegment p = nativePtr;
        if (p == null) {
            throw new IllegalStateException("FvecThreadPool is already closed");
        }
        return p;
    }

    /**
     * Frees the native thread pool. Idempotent — safe to call multiple times.
     */
    @Override
    public synchronized void close() {
        MemorySegment p = nativePtr;
        if (p != null) {
            nativePtr = null;
            FvecCapi.fvec_delete_thread_pool(p);
        }
    }
}
