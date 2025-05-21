package io.github.jbellis.jvector.example.testrig;


import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.Grid;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.Hdf5Loader;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedADC;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.nosqlbench.vectordata.VectorTestData;
import io.nosqlbench.vectordata.discovery.TestDataView;
import io.nosqlbench.vectordata.downloader.DatasetEntry;
import io.nosqlbench.vectordata.spec.datasets.types.BaseVectors;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

public class BenchHarness implements Runnable {

  private final DatasetEntry datasetEntry;
  private final String profile;

  public BenchHarness(
      io.nosqlbench.vectordata.downloader.DatasetEntry datasetEntry,
      String profile
  )
  {
    this.datasetEntry = datasetEntry;
    this.profile = profile;
  }

  @Override
  public void run() {
    TestDataView testDataView = datasetEntry.select().profile("default");
    BaseVectors bv = testDataView.getBaseVectors().orElseThrow();
    float[] v1 = bv.get(1);
    System.out.println(Arrays.toString(v1));
  }


}
