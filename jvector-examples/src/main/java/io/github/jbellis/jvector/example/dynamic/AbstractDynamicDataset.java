package io.github.jbellis.jvector.example.dynamic;

import io.github.jbellis.jvector.example.util.QueryBundle;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class AbstractDynamicDataset implements DynamicDataset {
    private final String name;
    private final VectorSimilarityFunction similarityFunction;

    // The points belonging to each batch
    private final List<List<Integer>> batches;
    // At time t, all vectors inserted in the (t - deletionLag) epoch are moved
    private final int deletionLag;

    private final List<VectorFloat<?>> baseVectors;
    private RandomAccessVectorValues baseRavv; // wrapper around baseVectors
    private final List<VectorFloat<?>> queryVectors;

    private List<List<List<Integer>>> epochGroundTruth;

    private final int topK;

    protected AbstractDynamicDataset(String name,
                                     VectorSimilarityFunction similarityFunction,
                                     List<List<Integer>> batches,
                                     int deletionLag,
                                     List<VectorFloat<?>> baseVectors,
                                     List<VectorFloat<?>> queryVectors,
                                     int topK) {
        this.name = name;
        this.similarityFunction = similarityFunction;
        this.batches = batches;
        this.deletionLag = deletionLag;
        this.baseVectors = baseVectors;
        this.queryVectors = queryVectors;
        this.topK = topK;

        computeEpochGroundTruth();
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public VectorSimilarityFunction getSimilarityFunction() {
        return similarityFunction;
    }

    @Override
    public int epochs() {
        return batches.size();
    }

    @Override
    public List<Integer> insertions(int epoch) {
        return batches.get(epoch);
    }

    @Override
    public List<Integer> deletions(int epoch) {
        if (epoch < deletionLag) {
            return new ArrayList<>();
        }
        return batches.get(epoch - deletionLag);
    }

    @Override
    public QueryBundle getQueryBundle(int epoch) {
        return new QueryBundle(queryVectors, epochGroundTruth.get(epoch));
    }

    private void computeEpochGroundTruth() {
        epochGroundTruth = new ArrayList<>(epochs());
        for (int epoch = 0; epoch < epochs(); epoch++) {
            List<List<Integer>> groundTruth = new ArrayList<>(queryVectors.size());

            List<Integer> elementsInBatch = new ArrayList<>();
            for (int ep = 0; ep <= epoch; ep++) {
                elementsInBatch.addAll(insertions(ep));
            }
            for (int ep = 0; ep <= epoch; ep++) {
                elementsInBatch.removeAll(deletions(ep));
            }
            System.out.println("Epoch " + epoch + ": " + elementsInBatch.size() + " elements in batch");

            for (var query : queryVectors) {
                List<VectorFloat<?>> vectorsEpoch0 = elementsInBatch.stream().map(baseVectors::get).collect(Collectors.toList());
                Integer[] gt = computeGroundTruth(similarityFunction, vectorsEpoch0, query, topK);
                var temp = Arrays.stream(gt).map(elementsInBatch::get).collect(Collectors.toList());
                groundTruth.add(temp);
            }
            System.out.println("Epoch " + epoch + ": " + groundTruth.size() + " queries");
            epochGroundTruth.add(groundTruth);
        }
    }

    @Override
    public RandomAccessVectorValues getBaseRavv() {
        if (baseRavv == null) {
            baseRavv = new ListRandomAccessVectorValues(baseVectors, getDimension());
        }
        return baseRavv;
    }

    @Override
    public VectorFloat<?> getBaseVector(int ordinal) {
        return baseVectors.get(ordinal);
    }

    @Override
    public int getDimension() {
        return baseVectors.get(0).length();
    }

    public static Integer[] computeGroundTruth(VectorSimilarityFunction vsf, List<VectorFloat<?>> vectors, VectorFloat<?> query, int topK) {
        double[] groundTruthDistances = vectors.stream().mapToDouble(vec -> vsf.compare(query, vec)).toArray();
        Integer[] idx = argsort(groundTruthDistances, false);
        idx = Arrays.copyOfRange(idx, 0, topK);
        return idx;
    }

    public static Integer[] argsort(final double[] a, final boolean ascending) {
        Integer[] indices = new Integer[a.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(final Integer i1, final Integer i2) {
                return (ascending ? 1 : -1) * Double.compare(a[i1], a[i2]);
            }
        });
        return indices;
    }
}
