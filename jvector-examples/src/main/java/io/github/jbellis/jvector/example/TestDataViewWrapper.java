package io.github.jbellis.jvector.example;

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
    return switch (df) {
      case EUCLIDEAN -> VectorSimilarityFunction.EUCLIDEAN;
      case COSINE -> VectorSimilarityFunction.COSINE;
      case DOT_PRODUCT -> VectorSimilarityFunction.DOT_PRODUCT;
      default -> throw new IllegalArgumentException("Unknown distance function " + df);
    };
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
