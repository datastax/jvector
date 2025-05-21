package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

public interface DataSet {
  /**
   Return a dataset containing the given vectors, scrubbed free from zero vectors and normalized to
   unit length.
   Note: This only scrubs and normalizes for dot product similarity.
   */
  static DataSet getScrubbedDataSet(
      String pathStr,
      VectorSimilarityFunction vsf,
      List<VectorFloat<?>> baseVectors,
      List<VectorFloat<?>> queryVectors,
      List<List<Integer>> groundTruth
  )
  {
    // remove zero vectors and duplicates, noting that this will change the indexes of the ground truth answers
    List<VectorFloat<?>> scrubbedBaseVectors;
    List<VectorFloat<?>> scrubbedQueryVectors;
    List<ArrayList<Integer>> gtSet;
    scrubbedBaseVectors = new ArrayList<>(baseVectors.size());
    scrubbedQueryVectors = new ArrayList<>(queryVectors.size());
    gtSet = new ArrayList<>(groundTruth.size());
    var uniqueVectors = new TreeSet<VectorFloat<?>>((a, b) -> {
      assert a.length() == b.length();
      for (int i = 0; i < a.length(); i++) {
        if (a.get(i) < b.get(i)) {
          return -1;
        }
        if (a.get(i) > b.get(i)) {
          return 1;
        }
      }
      return 0;
    });
    Map<Integer, Integer> rawToScrubbed = new HashMap<>();
    {
      int j = 0;
      for (int i = 0; i < baseVectors.size(); i++) {
        VectorFloat<?> v = baseVectors.get(i);
        var valid = (vsf == VectorSimilarityFunction.EUCLIDEAN) || Math.abs(normOf(v)) > 1e-5;
        if (valid && uniqueVectors.add(v)) {
          scrubbedBaseVectors.add(v);
          rawToScrubbed.put(i, j++);
        }
      }
    }
    // also remove zero query vectors and query vectors that are present in the base set
    for (int i = 0; i < queryVectors.size(); i++) {
      VectorFloat<?> v = queryVectors.get(i);
      var valid = (vsf == VectorSimilarityFunction.EUCLIDEAN) || Math.abs(normOf(v)) > 1e-5;
      var dupe = uniqueVectors.contains(v);
      if (valid && !dupe) {
        scrubbedQueryVectors.add(v);
        var gt = new ArrayList<Integer>();
        for (int j : groundTruth.get(i)) {
          gt.add(rawToScrubbed.get(j));
        }
        gtSet.add(gt);
      }
    }

    // now that the zero vectors are removed, we can normalize if it looks like they aren't already
    if (vsf == VectorSimilarityFunction.DOT_PRODUCT) {
      if (Math.abs(normOf(baseVectors.get(0)) - 1.0) > 1e-5) {
        normalizeAll(scrubbedBaseVectors);
        normalizeAll(scrubbedQueryVectors);
      }
    }

    assert scrubbedQueryVectors.size() == gtSet.size();
    return new DataSet(pathStr, vsf, scrubbedBaseVectors, scrubbedQueryVectors, gtSet);
  }

  static void normalizeAll(Iterable<VectorFloat<?>> vectors) {
    for (VectorFloat<?> v : vectors) {
      VectorUtil.l2normalize(v);
    }
  }

  static float normOf(VectorFloat<?> baseVector) {
    float norm = 0;
    for (int i = 0; i < baseVector.length(); i++) {
      norm += baseVector.get(i) * baseVector.get(i);
    }
    return (float) Math.sqrt(norm);
  }

  int getDimension();

  RandomAccessVectorValues getBaseRavv();

  String getName();

  VectorSimilarityFunction getSimilarityFunction();

  List<VectorFloat<?>> getBaseVectors();

  List<VectorFloat<?>> getQueryVectors();

  List<? extends List<Integer>> getGroundTruth();
}
