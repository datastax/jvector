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

package io.github.jbellis.jvector.graph.similarity;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.Int2ObjectHashMap;

/**
 * This is NOT a general "make vectors go faster" class.  It is used specifically by diversity computations
 * when loading vectors from disk, because in that specific scenario the same vector is usually loaded multiple times
 * as different neighbors are scored.
 */
class CachingVectorValues implements RandomAccessVectorValues {
    private final PQVectors cv;
    private final int dimension;
    private final Int2ObjectHashMap<VectorFloat<?>> cache;
    private final RandomAccessVectorValues ravv;

    public CachingVectorValues(PQVectors cv, int dimension, Int2ObjectHashMap<VectorFloat<?>> cache, RandomAccessVectorValues ravv) {
        this.cv = cv;
        this.dimension = dimension;
        this.cache = cache;
        this.ravv = ravv;
    }

    @Override
    public int size() {
        return cv.count();
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public boolean isValueShared() {
        return false;
    }

    @Override
    public RandomAccessVectorValues copy() {
        return this;
    }

    @Override
    public void getVectorInto(int nodeId, VectorFloat<?> result, int offset) {
        // getVectorInto is only called by reranking, not diversity code
        throw new UnsupportedOperationException();
    }

    @Override
    public VectorFloat<?> getVector(int nodeId) {
        return cache.computeIfAbsent(nodeId, (int n) -> {
            var v = ravv.getVector(n);
            return ravv.isValueShared() ? v.copy() : v;
        });
    }
}
