package io.github.jbellis.jvector.example.tutorial;

import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.disk.MemorySegmentReader;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.SearchResult.NodeScore;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.ash.AbstractAshVectors;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

public class Ashtray3 {
    public static void main(String[] args) throws IOException {
        var ds = DataSets.loadDataSet("ada002-100k").orElseThrow().getDataSet();
        var base = ds.getBaseVectors().stream().limit(1000).collect(Collectors.toList());
        var baseRavv = new ListRandomAccessVectorValues(base, ds.getDimension());
        // var baseRavv = ds.getBaseRavv();

        // var query = ds.getQueryVectors();
        var query = ds.getQueryVectors().stream().limit(100).collect(Collectors.toList());

        // var compressor = new PQFactory().create(baseRavv);
        var ashOptimizer = AsymmetricHashing.RANDOM;
        var ashBits = 1536;
        var ashLandmarks = 1;
        var bitDepth = 4;

        System.out.println("computing ASH");
        var ash = AsymmetricHashing.initialize(baseRavv, ashOptimizer, ashBits, bitDepth, ashLandmarks);
        System.out.println("encoding ASH");
        var ashv = ash.encodeAll(baseRavv, ForkJoinPool.commonPool());

        // var pq = ProductQuantization.compute(baseRavv, 192, 256, false);
        // var ashv = pq.encodeAll(baseRavv, ForkJoinPool.commonPool());

        var vsf = ds.getSimilarityFunction();
        assert vsf == VectorSimilarityFunction.DOT_PRODUCT;
        var graph = createGraph(ashv, baseRavv, vsf);
        // var graph = createGraph(baseRavv, vsf);

        System.out.println("Getting gt");
        // var gt = ds.getGroundTruth();
        var gt = query.stream()
            .parallel()
            .map(q -> {
                var sf = baseRavv.rerankerFor(q, vsf);
                return IntStream.range(0, base.size())
                    .mapToObj(i -> new NodeScore(i, sf.similarityTo(i)))
                    .sorted()
                    .limit(10)
                    .map(ns -> ns.node)
                    .collect(Collectors.toList());
            })
            .collect(Collectors.toList());

        System.out.println("Searching");
        var topk = 10;
        var oq = 1.0;
        var rrk = (int) (topk * oq);

        List<SearchResult> pred;
        try (var searchers = ExplicitThreadLocal.withInitial(() -> {
                var searcher = new GraphSearcher(graph);
                searcher.usePruning(false);
                return searcher;
            })
        ) {
            pred = IntStream.range(0, query.size())
                // .parallel()
                .mapToObj(qid -> {
                    var q = query.get(qid);
                    var searcher = searchers.get();
                    var asf = ashv.precomputedScoreFunctionFor(q, vsf);
                    var graphRavv = (RandomAccessVectorValues) searcher.getView();
                    var reranker = graphRavv.rerankerFor(q, vsf);
                    var ssp = new DefaultSearchScoreProvider(asf, reranker);

                    SearchResult sr;
                    sr = searcher.search(ssp, topk, rrk, 0.0f, 0.0f, Bits.ALL);
                    return sr;
                })
                .collect(Collectors.toList());

        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        double recall = AccuracyMetrics.recallFromSearchResults(gt, pred, topk, topk);
        System.out.println("Recall: " + recall);
    }

    static ImmutableGraphIndex createGraph(
            AbstractAshVectors<?> ashv,
            RandomAccessVectorValues ravv,
            VectorSimilarityFunction vsf
    ) throws IOException {
        var dim = ravv.dimension();
        var M = 32;
        var ef = 100;
        var nOv = 1.2f;
        var alpha = 1.2f;
        // occassionally causes assertion errors without this?
        var useHierarchy = false;

        var bsp = ashv.createBuildScoreProvider(vsf);
        // var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, vsf);

        System.out.println("Building graph");
        ImmutableGraphIndex graph;
        try (
            var builder = new GraphIndexBuilder(bsp, dim, M, ef, nOv, alpha, useHierarchy);
        ) {
            graph = builder.build(ravv);
        }

        var graphPath = Files.createTempFile("ashtray-", ".graph");
        // var graphPath = Path.of("ashtray.jvgraph");
        Files.deleteIfExists(graphPath);

        try (
            var writer = new OnDiskGraphIndexWriter.Builder(graph, graphPath)
                .with(new InlineVectors(dim))
                .build();
        ) {
            writer.write(Map.of(FeatureId.INLINE_VECTORS, i -> new InlineVectors.State(ravv.getVector(i))));
        }

        var rs = new MemorySegmentReader.Supplier(graphPath);
        return OnDiskGraphIndex.load(rs);
    }
}