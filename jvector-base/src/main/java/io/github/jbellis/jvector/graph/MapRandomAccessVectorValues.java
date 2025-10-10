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

import java.util.Map;

/**
 * RandomAccessValues backed by a Map.  This can be more useful than `ListRandomAccessVectorValues`
 * for handling concurrent inserts.
 * <p>
 * It is acceptable to provide this class to a GraphBuilder, and then continue
 * to add vectors to the backing Map as you add to the graph.
 * <p>
 * This will be as threadsafe as the provided Map.
 */
public class MapRandomAccessVectorValues implements RandomAccessVectorValues {
    private final Map<Integer, VectorFloat<?>> map;
    private final int dimension;

    /**
     * Constructs a MapRandomAccessVectorValues backed by the provided map.
     * @param map the map storing node ID to vector mappings
     * @param dimension the dimensionality of all vectors in the map
     */
    public MapRandomAccessVectorValues(Map<Integer, VectorFloat<?>> map, int dimension) {
        this.map = map;
        this.dimension = dimension;
    }

    @Override
    public int size() {
        return map.size();
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public VectorFloat<?> getVector(int nodeId) {
        return map.get(nodeId);
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
