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