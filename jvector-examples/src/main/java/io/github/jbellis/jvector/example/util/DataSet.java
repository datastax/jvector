/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.nosqlbench.vectordata.TestDataView;
import io.nosqlbench.vectordata.internalapi.datasets.FloatVectors;
import io.nosqlbench.vectordata.internalapi.datasets.views.NeighborIndices;
import io.nosqlbench.vectordata.internalapi.datasets.views.QueryVectors;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

public class DataSet {
  public final String name;
  public final VectorSimilarityFunction similarityFunction;
  public final List<VectorFloat<?>> baseVectors;
  public final List<VectorFloat<?>> queryVectors;
  public final List<? extends Set<Integer>> groundTruth;
  private RandomAccessVectorValues baseRavv;
  private static final VectorTypeSupport VECTOR_TYPE_SUPPORT =
      VectorizationProvider.getInstance().getVectorTypeSupport();

  public DataSet(
      String name,
      VectorSimilarityFunction similarityFunction,
      List<VectorFloat<?>> queryVectors,
      List<VectorFloat<?>> baseVectors,
      List<? extends Set<Integer>> groundTruth
  )
  {
    if (baseVectors.isEmpty()) {
      throw new IllegalArgumentException("Base vectors must not be empty");
    }
    if (queryVectors.isEmpty()) {
      throw new IllegalArgumentException("Query vectors must not be empty");
    }
    if (groundTruth.isEmpty()) {
      throw new IllegalArgumentException("Ground truth vectors must not be empty");
    }

    if (baseVectors.get(0).length() != queryVectors.get(0).length()) {
      throw new IllegalArgumentException("Base and query vectors must have the same dimensionality");
    }
    if (queryVectors.size() != groundTruth.size()) {
      throw new IllegalArgumentException("Query and ground truth lists must be the same size");
    }

    this.name = name;
    this.similarityFunction = similarityFunction;
    this.baseVectors = baseVectors;
    this.queryVectors = queryVectors;
    this.groundTruth = groundTruth;

    System.out.format(
        "%n%s: %d base and %d query vectors created, dimensions %d%n",
        name,
        baseVectors.size(),
        queryVectors.size(),
        baseVectors.get(0).length()
    );
  }

  public DataSet(TestDataView testdata) {
    this(
        testdata.getName(),
        getSimilarityFunction(testdata),
        getQueryVectors(testdata),
        getBaseVectors(testdata),
        getGroundTruth(testdata)
    );
  }

  private static List<? extends Set<Integer>> getGroundTruth(TestDataView testdata) {
    NeighborIndices ints = testdata.getNeighborIndices()
        .orElseThrow(() -> new IllegalArgumentException("No ground truth found in testdata"));
    List<Set<Integer>> gt = new ArrayList<>(ints.getCount());
    ints.forEach(neighborSet -> {
      Set<Integer> intSet = new HashSet<>(neighborSet.length);
      for (int i : neighborSet) {
        intSet.add(i);
      }
      gt.add(intSet);
    });
    return gt;
  }

  private static VectorSimilarityFunction getSimilarityFunction(TestDataView testdata) {
    VectorSimilarityFunction similarityFunction = switch (testdata.getDistanceFunction()) {
      case COSINE -> VectorSimilarityFunction.COSINE;
      case EUCLIDEAN -> VectorSimilarityFunction.EUCLIDEAN;
      case DOT_PRODUCT -> VectorSimilarityFunction.DOT_PRODUCT;
    };
    return similarityFunction;
  }

  private static List<VectorFloat<?>> getBaseVectors(TestDataView testdata) {
    FloatVectors floats = testdata.getBaseVectors()
        .orElseThrow(() -> new IllegalArgumentException("No base vectors found in testdata"));
    List<VectorFloat<?>> vflist = new ArrayList<>(floats.getCount());
    floats.forEach(vf -> vflist.add(VECTOR_TYPE_SUPPORT.createFloatVector(vf)));
    return vflist;
  }

  private static List<VectorFloat<?>> getQueryVectors(TestDataView testdata) {
    QueryVectors floats = testdata.getQueryVectors()
        .orElseThrow(() -> new IllegalArgumentException("No query vectors found in testdata"));
    List<VectorFloat<?>> vflist = new ArrayList<>(floats.getCount());
    floats.forEach(vf -> vflist.add(VECTOR_TYPE_SUPPORT.createFloatVector(vf)));
    return vflist;
  }

  /**
   Return a dataset containing the given vectors, scrubbed free from zero vectors and normalized to
   unit length.
   Note: This only scrubs and normalizes for dot product similarity.
   */
  public static DataSet getScrubbedDataSet(
      String pathStr,
      VectorSimilarityFunction vsf,
      List<VectorFloat<?>> baseVectors,
      List<VectorFloat<?>> queryVectors,
      List<Set<Integer>> groundTruth
  )
  {
    // remove zero vectors and duplicates, noting that this will change the indexes of the ground truth answers
    List<VectorFloat<?>> scrubbedBaseVectors;
    List<VectorFloat<?>> scrubbedQueryVectors;
    List<HashSet<Integer>> gtSet;
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
        var gt = new HashSet<Integer>();
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
    return new DataSet(pathStr, vsf, scrubbedQueryVectors, scrubbedBaseVectors, gtSet);
  }

  private static void normalizeAll(Iterable<VectorFloat<?>> vectors) {
    for (VectorFloat<?> v : vectors) {
      VectorUtil.l2normalize(v);
    }
  }

  private static float normOf(VectorFloat<?> baseVector) {
    float norm = 0;
    for (int i = 0; i < baseVector.length(); i++) {
      norm += baseVector.get(i) * baseVector.get(i);
    }
    return (float) Math.sqrt(norm);
  }

  public int getDimension() {
    return baseVectors.get(0).length();
  }

  public RandomAccessVectorValues getBaseRavv() {
    if (baseRavv == null) {
      baseRavv = new ListRandomAccessVectorValues(baseVectors, getDimension());
    }
    return baseRavv;
  }
}
