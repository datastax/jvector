package io.github.jbellis.jvector.fastervector.utils;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * In-memory {@link io.github.jbellis.jvector.fastervector.Vecs} over a flat {@code float[]} with known
 * dimension and stride: vector {@code i} occupies {@code data[i*stride .. i*stride+dim)}.
 * The array is not copied and must not be mutated while this object is in use.
 * <p>
 * The vector count is the number of whole vectors that fit:
 * {@code (data.length - dim) / stride + 1}, or 0 if {@code data.length < dim}.
 */
public final class FlatArrayVecs extends HeapVecs {

    private final float[] data;
    private final int stride;

    /** Densely packed vectors (stride == dim). */
    public FlatArrayVecs(float[] data, int dim) {
        this(data, dim, dim);
    }

    /**
     * @param stride floats between the starts of consecutive vectors; must be {@code >= dim}
     * @throws IllegalArgumentException if {@code dim <= 0} or {@code stride < dim}
     */
    public FlatArrayVecs(float[] data, int dim, int stride) {
        super(count(data.length, dim, stride), dim);
        this.data = data;
        this.stride = stride;
    }

    private static long count(int len, int dim, int stride) {
        if (dim <= 0) {
            throw new IllegalArgumentException("dim must be positive: " + dim);
        }
        if (stride < dim) {
            throw new IllegalArgumentException("stride (" + stride + ") must be >= dim (" + dim + ")");
        }
        return len < dim ? 0 : (len - dim) / (long) stride + 1;
    }

    @Override
    void copyInto(long ord, MemorySegment out) {
        MemorySegment.copy(data, (int) (ord * stride), out, ValueLayout.JAVA_FLOAT_UNALIGNED, 0L, dim());
    }
}
