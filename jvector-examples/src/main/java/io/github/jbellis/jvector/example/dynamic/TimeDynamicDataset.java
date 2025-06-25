package io.github.jbellis.jvector.example.dynamic;

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class TimeDynamicDataset {
    public DynamicDataset getTimeDynamicDataset(String name,
                                                VectorSimilarityFunction similarityFunction,
                                                List<Integer> timestamps,
                                                List<VectorFloat<?>> baseVectors,
                                                List<VectorFloat<?>> queryVectors,
                                                int epochs,
                                                int deletionLag,
                                                int topK) {
        if (epochs < 1) {
            throw new IllegalArgumentException("epochs must be at least 1");
        }
        if (deletionLag < 1) {
            throw new IllegalArgumentException("deletionLag must be at least 1");
        }

        var indices = argsort(timestamps, true);

        List<List<Integer>> batches = new ArrayList<>(epochs);
        int batchSize = indices.length / epochs;
        for (int start = 0; start < epochs; start += batchSize) {
            int end = Math.min(start + batchSize, indices.length);
            var list = new ArrayList<Integer>(batchSize);
            for (int idx = start; idx < end; idx++) {
                list.add(indices[idx]);
            }
            batches.add(list);
        }

        return new AbstractDynamicDataset(name, similarityFunction, batches, deletionLag, baseVectors, queryVectors, topK);
    }

    private static Integer[] argsort(final List<Integer> a, final boolean ascending) {
        Integer[] indices = new Integer[a.size()];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(final Integer i1, final Integer i2) {
                return (ascending ? 1 : -1) * Double.compare(a.get(i1), a.get(i2));
            }
        });
        return indices;
    }

}
