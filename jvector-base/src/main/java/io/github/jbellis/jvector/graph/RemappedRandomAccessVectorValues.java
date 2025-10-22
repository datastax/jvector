package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.Arrays;

public class RemappedRandomAccessVectorValues implements RandomAccessVectorValues {
    private final RandomAccessVectorValues ravv;
    private final int[] graphToRavvOrdMap;

    public RemappedRandomAccessVectorValues(RandomAccessVectorValues ravv, int[] graphToRavvOrdMap) {
        this.ravv = ravv;
        this.graphToRavvOrdMap = graphToRavvOrdMap;
    }

    @Override
    public int size() {
        return graphToRavvOrdMap.length;
    }

    @Override
    public int dimension() {
        return ravv.dimension();
    }

    @Override
    public VectorFloat<?> getVector(int node) {
        return ravv.getVector(graphToRavvOrdMap[node]);
    }

    @Override
    public boolean isValueShared() {
        return ravv.isValueShared();
    }

    @Override
    public RandomAccessVectorValues copy() {
        return new RemappedRandomAccessVectorValues(ravv.copy(), Arrays.copyOf(graphToRavvOrdMap, graphToRavvOrdMap.length));
    }

    @Override
    public void getVectorInto(int node, VectorFloat<?> result, int offset) {
        ravv.getVectorInto(graphToRavvOrdMap[node], result, offset);
    }
}
