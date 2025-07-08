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
        return batches.get(epoch - deletionLag);
    }

    @Override
    public QueryBundle getQueryBundle(int epoch) {
        // TODO for now, the ground truth is computed on the fly
        List<List<Integer>> groundTruth = new ArrayList<>(queryVectors.size());
        for (var query : queryVectors) {
            Integer[] gt = computeGroundTruth(similarityFunction, baseVectors, batches.get(epoch), query, topK);
            var temp = new ArrayList<Integer>(gt.length);
            temp.addAll(Arrays.asList(gt));
            groundTruth.add(temp);
        }
        return new QueryBundle(queryVectors, groundTruth);
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

    public static Integer[] computeGroundTruth(VectorSimilarityFunction vsf, List<VectorFloat<?>> vectors, List<Integer> batch, VectorFloat<?> query, int topK) {
        double[] groundTruthDistances = batch.stream().mapToDouble(i -> vsf.compare(query, vectors.get(i))).toArray();
        Integer[] idx = argsort(groundTruthDistances, false);
        idx = Arrays.copyOfRange(idx, 0, topK);
        for (int i = 0; i < idx.length; i++) {
            idx[i] = batch.get(idx[i]);
        }
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
