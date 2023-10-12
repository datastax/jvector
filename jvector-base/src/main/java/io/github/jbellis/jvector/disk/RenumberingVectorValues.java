package io.github.jbellis.jvector.disk;

import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;

import java.util.HashMap;
import java.util.Map;

class RenumberingVectorValues<T> implements RandomAccessVectorValues<T> {
    private final RandomAccessVectorValues<T> ravv;
    private final Map<Integer, Integer> newToOldMap;

    public RenumberingVectorValues(OnHeapGraphIndex<T> graph, RandomAccessVectorValues<T> ravv) {
        this.ravv = ravv;
        this.newToOldMap = new HashMap<>();
        int nextOrdinal = 0;
        for (int i = 0; i < ravv.size(); i++) {
            if (graph.containsNode(i)) {
                newToOldMap.put(nextOrdinal++, i);
            }
        }
    }

    @Override
    public int size() {
        return newToOldMap.size();
    }

    @Override
    public int dimension() {
        return ravv.dimension();
    }

    @Override
    public T vectorValue(int targetOrd) {
        return ravv.vectorValue(newToOldMap.get(targetOrd));
    }

    @Override
    public boolean isValueShared() {
        return ravv.isValueShared();
    }

    @Override
    public RandomAccessVectorValues<T> copy() {
        throw new UnsupportedOperationException();
    }
}
