package io.github.jbellis.jvector.example.dynamic;

import io.github.jbellis.jvector.example.util.Dataset;
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
    public static DynamicDataset create(Dataset ds,
                                        int epochs,
                                        int deletionLag,
                                        int topK) {
        return create(ds.getName(),
                      ds.getSimilarityFunction(),
                      ds.getBaseVectors(),
                      ds.getQueryBundle().queryVectors,
                      epochs,
                deletionLag,
                topK);
    }

    public static DynamicDataset create(String name,
                                        VectorSimilarityFunction similarityFunction,
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
        for (int ep = 0; ep < epochs; ep++) {
            batches.add(new ArrayList<>());
        }
        for (int i = 0; i < baseVectors.size(); i++) {
            int epoch = kmeans.getNearestCluster(baseVectors.get(i));
            batches.get(epoch).add(i);
        }

        return new AbstractDynamicDataset(name, similarityFunction, batches, deletionLag, baseVectors, queryVectors, topK);
    }
}
