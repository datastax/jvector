package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;

public class LazyFvecsDataSet extends DataSet {
    public final Path datasetFilePath;
    public final int vectorCount;
    public final int dimension;
    public final VectorTypeSupport vectorTypeSupport;

    public LazyFvecsDataSet(String name,
                            VectorSimilarityFunction similarityFunction,
                            List<VectorFloat<?>> queryVectors,
                            List<? extends List<Integer>> groundTruth,
                            Path datasetFilePath,
                            int vectorCount,
                            int dimension,
                            VectorTypeSupport vectorTypeSupport) {
        // Pass an empty list for baseVectors since we load them lazily.  Set the allowEmptyBase flag.
        super(name, similarityFunction, List.of(), queryVectors, groundTruth, true);
        this.datasetFilePath = datasetFilePath;
        this.vectorCount = vectorCount;
        this.dimension = dimension;
        this.vectorTypeSupport = vectorTypeSupport;
        System.out.format("Lazy Loading: %d base vectors of %d dims from %s%n", vectorCount, dimension, datasetFilePath);
    }

    /**
     * Returns a lazy-loading RandomAccessVectorValues that reads base vectors on demand.
     */
    @Override
    public RandomAccessVectorValues getBaseRavv() {
        try {
            return new LazyFvecsRandomAccessVectorValues(datasetFilePath, dimension, vectorCount, vectorTypeSupport);
        } catch (IOException e) {
            throw new RuntimeException("Failed to create lazy vector values", e);
        }
    }

    @Override
    public int getDimension() {
        return this.dimension;
    }
}

