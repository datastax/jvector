package io.github.jbellis.jvector.fastervector.utils;

import io.github.jbellis.jvector.fastervector.Vecs;
import io.github.jbellis.jvector.fastervector.VecsCursor;

import java.lang.foreign.MemorySegment;

/**
 * Shared base for in-memory (Java heap) {@link Vecs} implementations. Subclasses only
 * supply how to copy one vector into a segment; cursor handling lives here.
 */
abstract class HeapVecs implements Vecs {

    private final long nvecs;
    private final int dim;

    HeapVecs(long nvecs, int dim) {
        if (dim <= 0) {
            throw new IllegalArgumentException("dim must be positive: " + dim);
        }
        this.nvecs = nvecs;
        this.dim = dim;
    }

    /** Copy vector {@code ord} (guaranteed in range) into the start of {@code out}. */
    abstract void copyInto(long ord, MemorySegment out);

    @Override public long nvecs() { return nvecs; }
    @Override public int dim() { return dim; }

    @Override
    public VecsCursor getCursor() {
        return new VecsCursor() {
            private long ord = 0;

            @Override public long ord() { return ord; }

            @Override
            public void seek(long n) {
                if (n < 0 || n > nvecs) {
                    throw new IllegalArgumentException("Seek out of range: " + n);
                }
                ord = n;
            }

            @Override
            public boolean nextVectorInto(MemorySegment out) {
                if (ord >= nvecs) return false;
                copyInto(ord, out);
                ord++;
                return true;
            }
        };
    }
}
