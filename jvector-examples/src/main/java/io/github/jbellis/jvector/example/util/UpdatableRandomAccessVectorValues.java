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

package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.ArrayList;
import java.util.List;

/**
 * A mutable implementation of {@link RandomAccessVectorValues} that allows vectors to be
 * added dynamically. This implementation stores vectors in memory using an {@link ArrayList}
 * and is suitable for scenarios where the vector collection needs to grow over time.
 *
 * <p>This class is thread-safe for read operations but not for concurrent modifications.
 * Multiple threads can safely read vectors using {@link #getVector(int)}, but adding vectors
 * via {@link #add(VectorFloat)} should be externally synchronized if concurrent access is needed.</p>
 */
public class UpdatableRandomAccessVectorValues implements RandomAccessVectorValues {
    private final List<VectorFloat<?>> data;
    private final int dimensions;

    /**
     * Creates a new updatable vector collection with the specified dimensionality.
     * Initializes the internal storage with a capacity of 1024 vectors.
     *
     * @param dimensions the dimensionality of vectors that will be stored
     */
    public UpdatableRandomAccessVectorValues(int dimensions) {
        this.data = new ArrayList<>(1024);
        this.dimensions = dimensions;
    }

    /**
     * Adds a vector to this collection. The vector must have the same dimensionality
     * as specified in the constructor.
     *
     * @param vector the vector to add to this collection
     * @throws IllegalArgumentException if the vector's dimension does not match the expected dimension
     */
    public void add(VectorFloat<?> vector) {
        data.add(vector);
    }

    @Override
    public int size() {
        return data.size();
    }

    @Override
    public int dimension() {
        return dimensions;
    }

    @Override
    public VectorFloat<?> getVector(int targetOrd) {
        return data.get(targetOrd);
    }

    @Override
    public boolean isValueShared() {
        return false;
    }

    @Override
    public RandomAccessVectorValues copy() {
        return this;
    }
}
