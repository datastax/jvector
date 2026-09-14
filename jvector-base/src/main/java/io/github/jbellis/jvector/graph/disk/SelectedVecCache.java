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

package io.github.jbellis.jvector.graph.disk;

import java.util.*;
import java.util.concurrent.*;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.util.*;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import static java.lang.Math.*;

/**
 * Cache for storing selected diverse neighbors along with their metadata and vector copies.
 */
final class SelectedVecCache {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    int[] sourceIdx;
    OnDiskGraphIndex.View[] views;
    int[] nodes;
    float[] scores;
    VectorFloat<?>[] vecs;
    int size;

    /**
     * Constructs a cache with the specified capacity and vector dimension.
     */
    SelectedVecCache(int capacity, int dimension) {
        sourceIdx = new int[capacity];
        views = new OnDiskGraphIndex.View[capacity];
        nodes = new int[capacity];
        scores = new float[capacity];
        vecs = new VectorFloat<?>[capacity];
        for(int c = 0; c < capacity; ++c) {
            vecs[c] = vectorTypeSupport.createFloatVector(dimension);
        }
        size = 0;
    }

    /**
     * Resets the cache for reuse.
     */
    void reset() {
        size = 0;
    }

    /**
     * Adds a selected neighbor to the cache, copying its vector.
     */
    void add(int source, OnDiskGraphIndex.View view, int node, float score, VectorFloat<?> vec) {
        sourceIdx[size] = source;
        views[size] = view;
        nodes[size] = node;
        scores[size] = score;
        vecs[size].copyFrom(vec, 0, 0, vec.length());
        size++;
    }
}
