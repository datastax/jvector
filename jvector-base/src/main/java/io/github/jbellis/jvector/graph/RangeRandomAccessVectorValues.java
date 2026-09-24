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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.Objects;

/**
 * A re-based view over a contiguous ordinal range {@code [fromOrdinal, toOrdinal)} of a backing
 * {@link RandomAccessVectorValues}. Ordinal {@code i} of the view reads ordinal {@code fromOrdinal + i}
 * of the backing RAVV.
 * <p>
 * The view does not copy vectors. Its size is fixed at construction time, so a backing RAVV that grows
 * afterwards (for example a {@link ListRandomAccessVectorValues} over a list that is still being appended to)
 * is not reflected in the view. Value-sharing semantics are inherited from the backing RAVV.
 * <p>
 * This is the default implementation returned by {@link RandomAccessVectorValues#range(int, int)}; implementations
 * with a cheaper native ranged read may override that method instead.
 */
public final class RangeRandomAccessVectorValues implements RandomAccessVectorValues {
    private final RandomAccessVectorValues ravv;
    private final int fromOrdinal;
    private final int size;

    /**
     * Creates a view over ordinals {@code [fromOrdinal, toOrdinal)} of {@code ravv}.
     *
     * @param ravv        the backing RAVV
     * @param fromOrdinal the first backing ordinal of the range, inclusive
     * @param toOrdinal   the last backing ordinal of the range, exclusive
     * @throws IndexOutOfBoundsException if the range is not within {@code [0, ravv.size()]}
     */
    public RangeRandomAccessVectorValues(RandomAccessVectorValues ravv, int fromOrdinal, int toOrdinal) {
        Objects.checkFromToIndex(fromOrdinal, toOrdinal, ravv.size());
        this.ravv = ravv;
        this.fromOrdinal = fromOrdinal;
        this.size = toOrdinal - fromOrdinal;
    }

    /**
     * @return the backing ordinal that view ordinal {@code 0} maps to
     */
    public int fromOrdinal() {
        return fromOrdinal;
    }

    /**
     * @return the exclusive upper bound of the backing ordinal range
     */
    public int toOrdinal() {
        return fromOrdinal + size;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int dimension() {
        return ravv.dimension();
    }

    @Override
    public VectorFloat<?> getVector(int nodeId) {
        return ravv.getVector(fromOrdinal + Objects.checkIndex(nodeId, size));
    }

    @Override
    public void getVectorInto(int node, VectorFloat<?> destinationVector, int offset) {
        ravv.getVectorInto(fromOrdinal + Objects.checkIndex(node, size), destinationVector, offset);
    }

    @Override
    public boolean isValueShared() {
        return ravv.isValueShared();
    }

    /**
     * Copies the backing RAVV and wraps the copy in an equivalent view. If the backing RAVV is un-shared
     * and returns itself from {@link RandomAccessVectorValues#copy()}, this view returns itself as well.
     */
    @Override
    public RandomAccessVectorValues copy() {
        RandomAccessVectorValues copied = ravv.copy();
        return copied == ravv ? this : new RangeRandomAccessVectorValues(copied, fromOrdinal, toOrdinal());
    }

    /**
     * Narrows this view without adding a level of indirection: the result reads the backing RAVV directly
     * at {@code [this.fromOrdinal + fromOrdinal, this.fromOrdinal + toOrdinal)}.
     */
    @Override
    public RandomAccessVectorValues range(int fromOrdinal, int toOrdinal) {
        Objects.checkFromToIndex(fromOrdinal, toOrdinal, size);
        return new RangeRandomAccessVectorValues(ravv, this.fromOrdinal + fromOrdinal, this.fromOrdinal + toOrdinal);
    }
}
