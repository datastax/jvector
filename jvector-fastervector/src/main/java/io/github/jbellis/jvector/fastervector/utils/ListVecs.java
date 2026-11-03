package io.github.jbellis.jvector.fastervector.utils;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;

/**
 * In-memory {@link io.github.jbellis.jvector.fastervector.Vecs} over a {@code List<float[]>}.
 * The dimension is taken from the first vector; every vector must have that length.
 * The list's element references are snapshotted at construction (so any {@code List}
 * type is fine and later list changes are ignored), but the arrays themselves are not
 * copied and must not be mutated while this object is in use.
 */
public final class ListVecs extends HeapVecs {

    private final float[][] rows;

    /** @throws IllegalArgumentException if empty, or vector lengths differ or are zero */
    public ListVecs(List<float[]> vectors) {
        this(vectors.toArray(new float[0][]));
    }

    private ListVecs(float[][] rows) {
        super(rows.length, dimOf(rows));
        this.rows = rows;
        for (int i = 0; i < rows.length; i++) {
            if (rows[i].length != dim()) {
                throw new IllegalArgumentException(
                    "Vector " + i + " has length " + rows[i].length + ", expected " + dim());
            }
        }
    }

    private static int dimOf(float[][] rows) {
        if (rows.length == 0) {
            throw new IllegalArgumentException("Cannot infer dimension from an empty vector set");
        }
        return rows[0].length;
    }

    @Override
    void copyInto(long ord, MemorySegment out) {
        MemorySegment.copy(rows[(int) ord], 0, out, ValueLayout.JAVA_FLOAT_UNALIGNED, 0L, dim());
    }
}
