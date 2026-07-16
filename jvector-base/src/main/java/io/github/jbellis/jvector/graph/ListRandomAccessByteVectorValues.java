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

import io.github.jbellis.jvector.vector.types.ByteSequence;

import java.util.List;

/**
 * A List-backed implementation of the {@link RandomAccessByteVectorValues} interface.
 * <p>
 * It is acceptable to provide this class to a GraphBuilder, and then continue
 * to add vectors to the backing List as you add to the graph.
 * <p>
 * This will be as threadsafe as the provided List.
 */
public class ListRandomAccessByteVectorValues implements RandomAccessByteVectorValues {
    private final List<ByteSequence<?>> vectors;
    private final int dimension;

    /**
     * Construct a new instance of {@link ListRandomAccessByteVectorValues}.
     *
     * @param vectors   a (potentially mutable) list of byte vectors.
     * @param dimension the dimension of the vectors.
     */
    public ListRandomAccessByteVectorValues(List<ByteSequence<?>> vectors, int dimension) {
        this.vectors = vectors;
        this.dimension = dimension;
    }

    @Override
    public int size() {
        return vectors.size();
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public ByteSequence<?> getVector(int nodeId) {
        return vectors.get(nodeId);
    }

    @Override
    public boolean isValueShared() {
        return false;
    }

    @Override
    public ListRandomAccessByteVectorValues copy() {
        return this;
    }
}
