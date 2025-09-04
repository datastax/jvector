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
import io.nosqlbench.nbdatatools.api.concurrent.ProgressIndicator;
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
import java.util.concurrent.*;

public class BenchHarness implements Runnable {

  private final DatasetEntry datasetEntry;
  private final String profile;
  private final int concurrency;
  private final ExecutorService virtualThreadExecutor;
  private final Semaphore semaphore;

  public BenchHarness(
      io.nosqlbench.vectordata.downloader.DatasetEntry datasetEntry,
      String profile
  )
  {
    this(datasetEntry, profile, 1);
  }

  public BenchHarness(
      io.nosqlbench.vectordata.downloader.DatasetEntry datasetEntry,
      String profile,
      int concurrency
  )
  {
    this.datasetEntry = datasetEntry;
    this.profile = profile;
    this.concurrency = concurrency;
    this.virtualThreadExecutor = Executors.newVirtualThreadPerTaskExecutor();
    this.semaphore = new Semaphore(concurrency);
  }

  @Override
  public void run() {
    TestDataView testDataView = datasetEntry.select().profile(profile);
    smokeTestDataLoad(testDataView);
  }

  private void smokeTestDataLoad(TestDataView testDataView) {
      BaseVectors bv = testDataView.getBaseVectors().orElseThrow();

      System.out.println("Prebuffering...");
      CompletableFuture<Void> prebuffer = bv.prebuffer();
      if (prebuffer instanceof ProgressIndicator<?>) {
          ((ProgressIndicator<?>)prebuffer).monitorProgress(1000);
      }
      prebuffer.join();
      System.out.println("Prebuffered");

      float[] v1 = bv.get(1);
      System.out.println(Arrays.toString(v1));

    float[] vend = bv.get(bv.getCount() - 1);
    System.out.println(Arrays.toString(vend));

    /// Create tasks for processing vectors concurrently
    CompletableFuture<?>[] futures = new CompletableFuture[100];

    for (int i = 0; i < 100; i++) {
      final int index = i;
      futures[i] = CompletableFuture.runAsync(() -> {
        try {
          semaphore.acquire();
      try {
            /// This will be a stepping through the space of vectors
            int idx = (int) ((float)index / 100 * bv.getCount());
            float[] v = bv.get(idx);
            System.out.println(Arrays.toString(v));
          } finally {
            semaphore.release();
          }
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new RuntimeException(e);
        }
      }, virtualThreadExecutor);
    }

    /// Wait for all tasks to complete
    CompletableFuture.allOf(futures).join();

    /// Shutdown the executor
    virtualThreadExecutor.shutdown();
    try {
      if (!virtualThreadExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
        virtualThreadExecutor.shutdownNow();
      }
    } catch (InterruptedException e) {
      virtualThreadExecutor.shutdownNow();
      Thread.currentThread().interrupt();
    }
  }


}
