package io.github.jbellis.jvector.example.dynamic;

import io.github.jbellis.jvector.example.util.QueryBundle;
import io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class ClusteredDynamicDataset {
    public DynamicDataset getClusteredDynamicDataset(String name,
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

        var kmeans = new KMeansPlusPlusClusterer(baseVectors.toArray(VectorFloat<?>[]::new), epochs);
        kmeans.cluster(10, 0);

        List<List<Integer>> batches = new ArrayList<>(epochs);
        for (int i = 0; i < baseVectors.size(); i++) {
            int epoch = kmeans.getNearestCluster(baseVectors.get(i));
            if (batches.get(epoch) == null) {
                batches.add(epoch, new ArrayList<>());
            }
            batches.get(epoch).add(i);
        }

        return new AbstractDynamicDataset(name, similarityFunction, batches, deletionLag, baseVectors, queryVectors, topK);
    }
}
