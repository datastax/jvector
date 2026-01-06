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

import io.github.jbellis.jvector.graph.representations.RandomAccessVectorRepresentations;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.ArrayList;
import java.util.List;

public class UpdatableRandomAccessVectorRepresentations implements RandomAccessVectorRepresentations {
    private final List<VectorFloat<?>> data;
    private final int dimensions;

    public UpdatableRandomAccessVectorRepresentations(int dimensions) {
        this.data = new ArrayList<>(1024);
        this.dimensions = dimensions;
    }

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
    public RandomAccessVectorRepresentations copy() {
        return this;
    }
}
