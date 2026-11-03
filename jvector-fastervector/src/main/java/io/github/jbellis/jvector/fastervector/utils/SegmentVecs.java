package io.github.jbellis.jvector.fastervector.utils;

import java.lang.foreign.MemorySegment;

/**
 * {@link io.github.jbellis.jvector.fastervector.Vecs} over a densely packed row-major float32
 * {@link MemorySegment}: vector {@code i} occupies bytes
 * {@code [i*dim*4, (i+1)*dim*4)}. The segment is not copied and must remain
 * accessible (e.g. its arena open) while this object is in use.
 * <p>
 * The vector count is the number of whole vectors that fit in the segment.
 */
public final class SegmentVecs extends HeapVecs {

    private final MemorySegment data;

    /** @throws IllegalArgumentException if {@code dim <= 0} */
    public SegmentVecs(MemorySegment data, int dim) {
        super(dim <= 0 ? 0 : data.byteSize() / ((long) dim * Float.BYTES), dim);
        this.data = data;
    }

    @Override
    void copyInto(long ord, MemorySegment out) {
        long vecBytes = (long) dim() * Float.BYTES;
        MemorySegment.copy(data, ord * vecBytes, out, 0L, vecBytes);
    }
}
