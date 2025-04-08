package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import java.util.function.Supplier;

public class SubsetRandomAccessVectorValues implements RandomAccessVectorValues {
    private final RandomAccessVectorValues parent;
    private final int offset;
    private final int size;

    /**
     * Constructs a subset view of the parent RandomAccessVectorValues.
     * @param parent The original RandomAccessVectorValues (e.g. the lazy ravv).
     * @param start  The starting index (inclusive) of the subset.
     * @param end    The ending index (exclusive) of the subset.
     */
    public SubsetRandomAccessVectorValues(RandomAccessVectorValues parent, int start, int end) {
        if (start < 0 || end > parent.size() || start > end) {
            throw new IllegalArgumentException("Invalid subset range: " + start + " to " + end);
        }
        this.parent = parent;
        this.offset = start;
        this.size = end - start;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int dimension() {
        return parent.dimension();
    }

    @Override
    public VectorFloat<?> getVector(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Index " + index + " out of range for subset size " + size);
        }
        // Delegate to the parent, shifting the index by offset.
        return parent.getVector(index + offset);
    }

    @Override
    public boolean isValueShared() {
        return parent.isValueShared();
    }

    @Override
    public RandomAccessVectorValues copy() {
        // Create a copy of the parent and use the same offset and size.
        return new SubsetRandomAccessVectorValues(parent.copy(), offset, offset + size);
    }

    @Override
    public Supplier<RandomAccessVectorValues> threadLocalSupplier() {
        return () -> this;
    }
}

