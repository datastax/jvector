package io.github.jbellis.jvector.bench;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.benchmarks.datasets.SiftSmall;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.TimeUnit;

import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.openjdk.jmh.annotations.*;
        import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(1)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Threads(1)
public class CompactorBenchmark {
    private static final Logger log = LoggerFactory.getLogger(CompactorBenchmark.class);
    private RandomAccessVectorValues ravv;
    private List<VectorFloat<?>> baseVectors;
    private List<VectorFloat<?>> queryVectors;
    private List<List<Integer>> groundTruth;
    private GraphIndexBuilder graphIndexBuilder;
    List<OnDiskGraphIndex> graphs = new ArrayList<>();
    private ImmutableGraphIndex graphIndex;
    private ImmutableGraphIndex compactedGraphIndex;
    int originalDimension;
    VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.COSINE;
    private Path tempDir;
    private int numSources;
    private int numVectorsPerSource;

    @Setup(Level.Iteration)
    public void setup() throws IOException {
        var siftPath = "siftsmall";
        baseVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_base.fvecs", siftPath));
        queryVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_query.fvecs", siftPath));
        groundTruth = SiftLoader.readIvecs(String.format("%s/siftsmall_groundtruth.ivecs", siftPath));
        log.info("base vectors size: {}, query vectors size: {}, loaded, dimensions {}",
                baseVectors.size(), queryVectors.size(), baseVectors.get(0).length());
        originalDimension = baseVectors.get(0).length();
        // wrap the raw vectors in a RandomAccessVectorValues
        ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);
        // score provider using the raw, in-memory vectors
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);

        graphIndexBuilder = new GraphIndexBuilder(bsp,
                ravv.dimension(),
                16, // graph degree
                100, // construction search depth
                1.2f, // allow degree overflow during construction by this factor
                1.2f, // relax neighbor diversity requirement by this factor
                true); // add the hierarchy
        graphIndex = graphIndexBuilder.build(ravv);

        tempDir = Files.createTempDirectory("compact-bench");
        numSources = 2;
        numVectorsPerSource = baseVectors.size() / numSources;
        for(int i = 0; i < numSources; i++) {
            List<VectorFloat<?>> vectorsPerSource = new ArrayList<>(baseVectors.subList(i * numVectorsPerSource, (i + 1) * numVectorsPerSource));
            Path outputPath = tempDir.resolve("per-source-graph-" + i);
            var ravvPerSource = new ListRandomAccessVectorValues(vectorsPerSource, originalDimension);
            BuildScoreProvider bspPerSource = BuildScoreProvider.randomAccessScoreProvider(ravvPerSource, VectorSimilarityFunction.EUCLIDEAN);
            var graphIndexBuilderPerSource = new GraphIndexBuilder(bspPerSource,
                    ravv.dimension(),
                    16, // graph degree
                    100, // construction search depth
                    1.2f, // allow degree overflow during construction by this factor
                    1.2f, // relax neighbor diversity requirement by this factor
                    true); // add the hierarchy
            var graph = graphIndexBuilderPerSource.build(ravvPerSource);
            OnDiskGraphIndex.write(graph, ravvPerSource, outputPath);
        }



    }

    @TearDown(Level.Iteration)
    public void tearDown() throws IOException {
        baseVectors.clear();
        queryVectors.clear();
        groundTruth.clear();
        graphIndexBuilder.close();
        graphs.clear();
        graphIndex.close();
    }

    @Benchmark
    public void testCompactWithRandomQueryVectors(Blackhole blackhole) throws IOException {
        log.info("compactWithRandomQueryVectors");
        List<ReaderSupplier> rss = new ArrayList<>();
        for(int i = 0; i < numSources; ++i) {
            var outputPathPerSource = tempDir.resolve("per-source-graph-" + i);
            rss.add(ReaderSupplierFactory.open(outputPathPerSource.toAbsolutePath()));
            var onDiskGraph = OnDiskGraphIndex.load(rss.get(i));
            graphs.add(onDiskGraph);
        }

        var outputPath = tempDir.resolve("compact-graph");
        var compactor = new OnDiskGraphIndexCompactor(graphs);
        int globalOrdinal = 0;
        for(int n = 0; n < numSources; ++n) {
            Map<Integer, Integer> map = new HashMap<>();
            for(int i = 0; i < numVectorsPerSource; ++i) {
                map.put(i, globalOrdinal++);
            }
            var remapper = new OrdinalMapper.MapMapper(map);
            compactor.setRemapper(graphs.get(n), remapper);
        }
        compactor.compact(outputPath, similarityFunction);

        ReaderSupplier rs;
        try {
            rs = ReaderSupplierFactory.open(outputPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        var compactGraph = OnDiskGraphIndex.load(rs);
        // Your benchmark code here
        List<SearchResult> retrieved = new ArrayList<>();
        for(int n = 0; n < queryVectors.size(); ++n) {
            retrieved.add(GraphSearcher.search(queryVectors.get(n),
                    10, // number of results
                    ravv, // vectors we're searching, used for scoring
                    VectorSimilarityFunction.EUCLIDEAN, // how to score
                    graphIndex,
                    Bits.ALL)); // valid ordinals to consider
        }
        List<SearchResult> compactedRetrieved = new ArrayList<>();
        for(int n = 0; n < queryVectors.size(); ++n) {
            compactedRetrieved.add(GraphSearcher.search(queryVectors.get(n),
                    10, // number of results
                    ravv, // vectors we're searching, used for scoring
                    VectorSimilarityFunction.EUCLIDEAN, // how to score
                    compactGraph,
                    Bits.ALL)); // valid ordinals to consider
        }
        var recall = AccuracyMetrics.recallFromSearchResults(groundTruth, retrieved, 10, 10);
        var crecall = AccuracyMetrics.recallFromSearchResults(groundTruth, compactedRetrieved, 10, 10);
        log.info("Whole Graph Recall: {}", recall);
        log.info("Compacted Graph Recall: {}", crecall);
        blackhole.consume(retrieved);
        graphs.clear();
    }
}
