package io.github.jbellis.jvector.testrig;


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
import io.nosqlbench.vectordata.TestDataGroup;
import io.nosqlbench.vectordata.TestDataView;
import io.nosqlbench.vectordata.download.DatasetEntry;
import io.nosqlbench.vectordata.download.DownloadProgress;
import io.nosqlbench.vectordata.download.DownloadResult;
import io.nosqlbench.vectordata.download.DownloadStatus;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
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
  private final String datasetProfile;

  public BenchHarness(DatasetEntry datasetEntry, String profile) {
    this.datasetEntry = datasetEntry;
    this.datasetProfile = profile;
  }

  @Override
  public void run() {

    String hdf5Dir = Hdf5Loader.HDF5_DIR;
    Path dspath = Path.of(hdf5Dir).resolve(datasetEntry.name());

    DownloadProgress download = datasetEntry.download(dspath);
    DownloadResult result;
    while (true) {
      try {
        result = download.poll(5, TimeUnit.SECONDS);
        System.out.println("result:" + result);
        if (result != null)
          break;
      } catch (InterruptedException e) {
        throw new RuntimeException(e);
      }
      System.out.println(download);
    }
    if (result.status() == DownloadStatus.FAILED) {
      throw new RuntimeException(
          "Failed to download dataset: " + datasetEntry.name() + " " + result);
    }

    TestDataGroup datagroup = result.getDataGroup()
        .orElseThrow(() -> new RuntimeException("No data group found for " + datasetEntry.name()));
    TestDataView dataview = datagroup.getProfileOptionally(datasetProfile)
        .orElseThrow(() -> new RuntimeException(
            "No data view found for " + datasetEntry.name() + " profile named '" + datasetProfile + "'"));
    DataSet ds = new DataSet(dataview);

    //    io.github.jbellis.jvector.example.util.Hdf5Loader.maybeDownloadHdf5("glove-100-angular.hdf5");

    Set<FeatureId> features = EnumSet.of(FeatureId.NVQ_VECTORS);
    var mGrid = List.of(32); // List.of(16, 24, 32, 48, 64, 96, 128);
    var efConstructionGrid = List.of(100); // List.of(60, 80, 100, 120, 160, 200, 400, 600, 800);
    var overqueryGrid = List.of(1.0, 2.0, 5.0); // rerankK = oq * topK
    var neighborOverflowGrid = List.of(1.2f); // List.of(1.2f, 2.0f);
    var addHierarchyGrid = List.of(true); // List.of(false, true);
    var usePruningGrid = List.of(true); // List.of(false, true);

    //    DataSet ds = Hdf5Loader.load(dspath.toString());
    try {
      Grid.runOneGraph(
          List.of(features),
          mGrid.get(0),
          efConstructionGrid.get(0),
          neighborOverflowGrid.get(0),
          addHierarchyGrid.get(0),
          null,
          List.of(),
          overqueryGrid,
          usePruningGrid,
          ds,
          null
      );
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }


  public static void runOneGraph(
      List<? extends Set<FeatureId>> featureSets,
      int M,
      int efConstruction,
      float neighborOverflow,
      boolean addHierarchy,
      VectorCompressor<?> buildCompressor,
      List<Function<DataSet, CompressorParameters>> compressionGrid,
      List<Double> efSearchOptions,
      List<Boolean> usePruningGrid,
      DataSet ds,
      Path testDirectory
  ) throws IOException
  {
    Map<Set<FeatureId>, GraphIndex> indexes;
    if (buildCompressor == null) {
      indexes = buildInMemory(
          featureSets,
          M,
          efConstruction,
          neighborOverflow,
          addHierarchy,
          ds,
          testDirectory
      );
    } else {
      indexes = buildOnDisk(
          featureSets,
          M,
          efConstruction,
          neighborOverflow,
          addHierarchy,
          ds,
          testDirectory,
          buildCompressor
      );
    }

    try {
      for (var cpSupplier : compressionGrid) {
        var compressor = Grid.getCompressor(cpSupplier, ds);
        CompressedVectors cv;
        if (compressor == null) {
          cv = null;
          System.out.format("Uncompressed vectors%n");
        } else {
          long start = System.nanoTime();
          cv = compressor.encodeAll(ds.getBaseRavv());
          System.out.format(
              "%s encoded %d vectors [%.2f MB] in %.2fs%n",
              compressor,
              ds.baseVectors.size(),
              (cv.ramBytesUsed() / 1024f / 1024f),
              (System.nanoTime() - start) / 1_000_000_000.0
          );
        }

        indexes.forEach((features, index) -> {
          try (var cs = new Grid.ConfiguredSystem(
              ds,
              index,
              cv,
              index instanceof OnDiskGraphIndex ? ((OnDiskGraphIndex) index).getFeatureSet() :
                  Set.of()
          ))
          {
            testConfiguration(cs, efSearchOptions, usePruningGrid);
          } catch (Exception e) {
            throw new RuntimeException(e);
          }
        });
      }
      for (var index : indexes.values()) {
        index.close();
      }
    } finally {
      for (int n = 0; n < featureSets.size(); n++) {
        Files.deleteIfExists(testDirectory.resolve("graph" + n));
      }
    }
  }

  private static Map<Set<FeatureId>, GraphIndex> buildInMemory(
      List<? extends Set<FeatureId>> featureSets,
      int M,
      int efConstruction,
      float neighborOverflow,
      boolean addHierarchy,
      DataSet ds,
      Path testDirectory
  ) throws IOException
  {
    var floatVectors = ds.getBaseRavv();
    Map<Set<FeatureId>, GraphIndex> indexes = new HashMap<>();
    long start;
    var bsp = BuildScoreProvider.randomAccessScoreProvider(floatVectors, ds.similarityFunction);
    GraphIndexBuilder builder = new GraphIndexBuilder(
        bsp,
        floatVectors.dimension(),
        M,
        efConstruction,
        neighborOverflow,
        1.2f,
        addHierarchy,
        PhysicalCoreExecutor.pool(),
        ForkJoinPool.commonPool()
    );
    start = System.nanoTime();
    var onHeapGraph = builder.build(floatVectors);
    System.out.format(
        "Build (%s) M=%d overflow=%.2f ef=%d in %.2fs%n",
        "full res",
        M,
        neighborOverflow,
        efConstruction,
        (System.nanoTime() - start) / 1_000_000_000.0
    );
    for (int i = 0; i <= onHeapGraph.getMaxLevel(); i++) {
      System.out.format(
          "  L%d: %d nodes, %.2f avg degree%n",
          i,
          onHeapGraph.getLayerSize(i),
          onHeapGraph.getAverageDegree(i)
      );
    }
    int n = 0;
    for (var features : featureSets) {
      if (features.contains(FeatureId.FUSED_ADC)) {
        System.out.println("Skipping Fused ADC feature when building in memory");
        continue;
      }
      var graphPath = testDirectory.resolve("graph" + n++);
      var bws = builderWithSuppliers(features, onHeapGraph, graphPath, floatVectors, null);
      try (var writer = bws.builder.build()) {
        start = System.nanoTime();
        writer.write(bws.suppliers);
        System.out.format(
            "Wrote %s in %.2fs%n",
            features,
            (System.nanoTime() - start) / 1_000_000_000.0
        );
      }

      var index = OnDiskGraphIndex.load(ReaderSupplierFactory.open(graphPath));
      indexes.put(features, index);
    }
    return indexes;
  }

  private static Map<Set<FeatureId>, GraphIndex> buildOnDisk(
      List<? extends Set<FeatureId>> featureSets,
      int M,
      int efConstruction,
      float neighborOverflow,
      boolean addHierarchy,
      DataSet ds,
      Path testDirectory,
      VectorCompressor<?> buildCompressor
  ) throws IOException
  {
    var floatVectors = ds.getBaseRavv();

    var pq = (PQVectors) buildCompressor.encodeAll(floatVectors);
    var bsp = BuildScoreProvider.pqBuildScoreProvider(ds.similarityFunction, pq);
    GraphIndexBuilder builder = new GraphIndexBuilder(
        bsp,
        floatVectors.dimension(),
        M,
        efConstruction,
        neighborOverflow,
        1.2f,
        addHierarchy
    );

    // use the inline vectors index as the score provider for graph construction
    Map<Set<FeatureId>, OnDiskGraphIndexWriter> writers = new HashMap<>();
    Map<Set<FeatureId>, Map<FeatureId, IntFunction<Feature.State>>> suppliers = new HashMap<>();
    OnDiskGraphIndexWriter scoringWriter = null;
    int n = 0;
    for (var features : featureSets) {
      var graphPath = testDirectory.resolve("graph" + n++);
      var bws = builderWithSuppliers(
          features,
          builder.getGraph(),
          graphPath,
          floatVectors,
          pq.getCompressor()
      );
      var writer = bws.builder.build();
      writers.put(features, writer);
      suppliers.put(features, bws.suppliers);
      if (features.contains(FeatureId.INLINE_VECTORS) || features.contains(FeatureId.NVQ_VECTORS)) {
        scoringWriter = writer;
      }
    }
    if (scoringWriter == null) {
      throw new IllegalStateException(
          "Bench looks for either NVQ_VECTORS or INLINE_VECTORS feature set for scoring compressed builds.");
    }

    // build the graph incrementally
    long start = System.nanoTime();
    var vv = floatVectors.threadLocalSupplier();
    PhysicalCoreExecutor.pool().submit(() -> {
      IntStream.range(0, floatVectors.size()).parallel().forEach(node -> {
        writers.forEach((features, writer) -> {
          try {
            var stateMap = new EnumMap<FeatureId, Feature.State>(FeatureId.class);
            suppliers.get(features).forEach((featureId, supplier) -> {
              stateMap.put(featureId, supplier.apply(node));
            });
            writer.writeInline(node, stateMap);
          } catch (IOException e) {
            throw new UncheckedIOException(e);
          }
        });
        builder.addGraphNode(node, vv.get().getVector(node));
      });
    }).join();
    builder.cleanup();
    // write the edge lists and close the writers
    // if our feature set contains Fused ADC, we need a Fused ADC write-time supplier (as we don't have neighbor information during writeInline)
    writers.entrySet().stream().parallel().forEach(entry -> {
      var writer = entry.getValue();
      var features = entry.getKey();
      Map<FeatureId, IntFunction<Feature.State>> writeSuppliers;
      if (features.contains(FeatureId.FUSED_ADC)) {
        writeSuppliers = new EnumMap<>(FeatureId.class);
        var view = builder.getGraph().getView();
        writeSuppliers.put(FeatureId.FUSED_ADC, ordinal -> new FusedADC.State(view, pq, ordinal));
      } else {
        writeSuppliers = Map.of();
      }
      try {
        writer.write(writeSuppliers);
        writer.close();
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
    });
    builder.close();
    System.out.format(
        "Build and write %s in %ss%n",
        featureSets,
        (System.nanoTime() - start) / 1_000_000_000.0
    );

    // open indexes
    Map<Set<FeatureId>, GraphIndex> indexes = new HashMap<>();
    n = 0;
    for (var features : featureSets) {
      var graphPath = testDirectory.resolve("graph" + n++);
      var index = OnDiskGraphIndex.load(ReaderSupplierFactory.open(graphPath));
      indexes.put(features, index);
    }
    return indexes;
  }

  private static BuilderWithSuppliers builderWithSuppliers(
      Set<FeatureId> features,
      OnHeapGraphIndex onHeapGraph,
      Path outPath,
      RandomAccessVectorValues floatVectors,
      ProductQuantization pq
  ) throws FileNotFoundException
  {
    var identityMapper = new OrdinalMapper.IdentityMapper(floatVectors.size() - 1);
    var builder =
        new OnDiskGraphIndexWriter.Builder(onHeapGraph, outPath).withMapper(identityMapper);
    Map<FeatureId, IntFunction<Feature.State>> suppliers = new EnumMap<>(FeatureId.class);
    for (var featureId : features) {
      switch (featureId) {
        case INLINE_VECTORS:
          builder.with(new InlineVectors(floatVectors.dimension()));
          suppliers.put(
              FeatureId.INLINE_VECTORS,
              ordinal -> new InlineVectors.State(floatVectors.getVector(ordinal))
          );
          break;
        case FUSED_ADC:
          if (pq == null) {
            System.out.println("Skipping Fused ADC feature due to null ProductQuantization");
            continue;
          }
          // no supplier as these will be used for writeInline, when we don't have enough information to fuse neighbors
          builder.with(new FusedADC(onHeapGraph.maxDegree(), pq));
          break;
        case NVQ_VECTORS:
          var nvq = NVQuantization.compute(floatVectors, 2);
          builder.with(new NVQ(nvq));
          suppliers.put(
              FeatureId.NVQ_VECTORS,
              ordinal -> new NVQ.State(nvq.encode(floatVectors.getVector(ordinal)))
          );
          break;

      }
    }
    return new BuilderWithSuppliers(builder, suppliers);
  }

  private static class BuilderWithSuppliers {
    public final OnDiskGraphIndexWriter.Builder builder;
    public final Map<FeatureId, IntFunction<Feature.State>> suppliers;

    public BuilderWithSuppliers(
        OnDiskGraphIndexWriter.Builder builder,
        Map<FeatureId, IntFunction<Feature.State>> suppliers
    )
    {
      this.builder = builder;
      this.suppliers = suppliers;
    }
  }

  private static void testConfiguration(
      Grid.ConfiguredSystem cs,
      List<Double> efSearchOptions,
      List<Boolean> usePruningGrid
  )
  {
    var topK = cs.ds.groundTruth.get(0).size();
    int queryRuns = 2;
    System.out.format("Using %s:%n", cs.getIndex());
    for (var overquery : efSearchOptions) {
      int rerankK = (int) (topK * overquery);
      for (var usePruning : usePruningGrid) {
        var startTime = System.nanoTime();
        var pqr = Grid.performQueries(cs, topK, rerankK, usePruning, queryRuns);
        var stopTime = System.nanoTime();
        var recall = ((double) pqr.topKFound) / (queryRuns * cs.ds.queryVectors.size() * topK);
        System.out.format(
            " Query top %d/%d recall %.4f in %.2fms after %.2f nodes visited (AVG) and %.2f nodes expanded with pruning=%b%n",
            topK,
            rerankK,
            recall,
            (stopTime - startTime) / (queryRuns * 1_000_000.0),
            (double) pqr.nodesVisited / (queryRuns * cs.ds.queryVectors.size()),
            (double) pqr.nodesExpanded / (queryRuns * cs.ds.queryVectors.size()),
            usePruning
        );
      }
    }
  }


}
