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

import io.github.jbellis.jvector.graph.representations.RandomAccessVectorRepresentations;
import io.github.jbellis.jvector.vector.VectorRepresentation;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.List;
import java.util.Map;

/**
 * A List-backed implementation of the {@link RandomAccessVectorRepresentations} interface.
 * <p>
 * It is acceptable to provide this class to a GraphBuilder, and then continue
 * to add vectors to the backing List as you add to the graph.
 * <p>
 * This will be as threadsafe as the provided List.
 */
public class ListRandomAccessVectorRepresentations<Vec extends VectorRepresentation> implements RandomAccessVectorRepresentations<Vec> {
    private final List<Vec> vectors;
    private final int dimension;

    /**
     * Construct a new instance of {@link ListRandomAccessVectorRepresentations}.
     *
     * @param vectors   a (potentially mutable) list of float vectors.
     * @param dimension the dimension of the vectors.
     */
    public ListRandomAccessVectorRepresentations(List<Vec> vectors, int dimension) {
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
    public Vec getVector(int targetOrd) {
        return vectors.get(targetOrd);
    }

    @Override
    public boolean isValueShared() {
        return false;
    }

    @Override
    public ListRandomAccessVectorRepresentations copy() {
        return this;
    }

    @Override
    public long ramBytesUsed() {
        long bytesUsed = 0;
        for (Vec v : vectors) {
            bytesUsed += Integer.BYTES + v.ramBytesUsed();
        }
        return bytesUsed;
    }
}
