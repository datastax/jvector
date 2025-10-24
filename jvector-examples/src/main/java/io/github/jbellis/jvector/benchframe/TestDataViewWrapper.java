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

package io.github.jbellis.jvector.benchframe;

import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.FloatVectorsWrapper;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.nosqlbench.vectordata.discovery.TestDataView;
import io.nosqlbench.vectordata.spec.datasets.types.NeighborIndices;
import io.nosqlbench.vectordata.spec.datasets.types.QueryVectors;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class TestDataViewWrapper implements DataSet {
  public final TestDataView view;
  private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

  public TestDataViewWrapper(TestDataView view) {
    this.view = view;
  }

  @Override
  public String getName() {
    return view.getName();
  }

  @Override
  public VectorSimilarityFunction getSimilarityFunction() {
    var df = view.getDistanceFunction();
    switch (df) {
        case EUCLIDEAN: return VectorSimilarityFunction.EUCLIDEAN;
        case COSINE: return VectorSimilarityFunction.COSINE;
        case DOT_PRODUCT: return VectorSimilarityFunction.DOT_PRODUCT;
        default: throw new IllegalArgumentException("Unknown distance function " + df);
    }
  }

  @Override
  public List<VectorFloat<?>> getBaseVectors() {
    throw new RuntimeException("This method should not be called. Use getBaseRavv() instead.");
  }

  @Override
  public List<VectorFloat<?>> getQueryVectors() {
    QueryVectors queryVectors = view.getQueryVectors().orElseThrow(() -> new RuntimeException("unable to load query vectors"));
    ArrayList<VectorFloat<?>> vectorFlaots = new ArrayList<>(queryVectors.getCount());
    for (float[] qv : queryVectors) {
      vectorFlaots.add(vts.createFloatVector(qv));
    }
    return vectorFlaots;

  }

  @Override
  public List<? extends List<Integer>> getGroundTruth() {
    Optional<NeighborIndices> gt = view.getNeighborIndices();

    return List.of();
  }

  @Override
  public int getDimension() {
    return view.getBaseVectors().get().getVectorDimensions();
  }

  @Override
  public RandomAccessVectorValues getBaseRavv() {
    return view.getBaseVectors().map(FloatVectorsWrapper::new).orElseThrow(() -> new RuntimeException("unable to load float vectors"));
  }
}
