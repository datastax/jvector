package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.jhdf.api.Dataset;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

public interface DataSet {

  default void normalizeAll(Iterable<VectorFloat<?>> vectors) {
    for (VectorFloat<?> v : vectors) {
      VectorUtil.l2normalize(v);
    }
  }

  default float normOf(VectorFloat<?> baseVector) {
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

  public static interface CanScrub {
    DataSet  scrub();
  }
}
