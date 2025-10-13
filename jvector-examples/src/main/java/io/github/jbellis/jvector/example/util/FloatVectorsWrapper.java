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
import io.github.jbellis.jvector.vector.ArrayVectorFloat;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.nosqlbench.vectordata.spec.datasets.types.FloatVectors;

import java.util.function.Supplier;

/// Wrapper that adapts a nosqlbench FloatVectors instance to implement RandomAccessVectorValues
public class FloatVectorsWrapper implements RandomAccessVectorValues {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    
    private final FloatVectors floatVectors;
    private final int dimension;

    public FloatVectorsWrapper(FloatVectors floatVectors) {
        this.floatVectors = floatVectors;
        this.dimension = floatVectors.getVectorDimensions();
    }
    
    @Override
    public int size() {
        return floatVectors.getCount();
    }
    
    @Override
    public int dimension() {
        return floatVectors.getVectorDimensions();
    }
    
    @Override
    public VectorFloat<?> getVector(int nodeId) {
        return vts.createFloatVector(floatVectors.get(nodeId));
    }
    
    @Override
    public boolean isValueShared() {
        return true;
    }
    
    @Override
    public RandomAccessVectorValues copy() {
        return new FloatVectorsWrapper(floatVectors);
    }
    
    @Override
    public Supplier<RandomAccessVectorValues> threadLocalSupplier() {
        return () -> new FloatVectorsWrapper(floatVectors);
    }
}