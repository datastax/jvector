package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.List;

public class QueryBundle {
    public final List<VectorFloat<?>> queryVectors;
    public final List<? extends List<Integer>> groundTruth;

    public QueryBundle(List<VectorFloat<?>> queryVectors, List<? extends List<Integer>> groundTruth) {
        if (queryVectors.isEmpty()) {
            throw new IllegalArgumentException("Query vectors must not be empty");
        }
        if (groundTruth.isEmpty()) {
            throw new IllegalArgumentException("Ground truth vectors must not be empty");
        }
        if (queryVectors.size() != groundTruth.size()) {
            throw new IllegalArgumentException("Query and ground truth lists must be the same size");
        }

        this.queryVectors = queryVectors;
        this.groundTruth = groundTruth;
    }
}
