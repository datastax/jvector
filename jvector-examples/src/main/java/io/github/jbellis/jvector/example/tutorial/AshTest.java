package io.github.jbellis.jvector.example.tutorial;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult.NodeScore;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.ash.AbstractAshVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

public class AshTest {
    public static void main(String[] args) throws IOException {
        System.out.println("Hi AshTest");
        var ds = DataSets.loadDataSet("ada002-100k").orElseThrow().getDataSet();

        var maxBase = 1_000_000;
        var base = ds.getBaseVectors().stream().limit(maxBase).collect(Collectors.toList());
        var ravv = new ListRandomAccessVectorValues(base, ds.getDimension());

        var maxQuery = 100;
        var query = ds.getQueryVectors().stream().limit(maxQuery).collect(Collectors.toList());

        var optimizer = AsymmetricHashing.RANDOM;
        // var optimizer = AsymmetricHashing.ITQ;

        var landmarkCount = 1;
        int bitDepth = 1;

        // float bitRatio = 1.0f;
        // boolean includeHeaderBitsInCalc = false;

        // int totalEncodedBits = (int) Math.round(ds.getDimension() * bitRatio);
        // int totalQuantizeBits = totalEncodedBits - (includeHeaderBitsInCalc ? AsymmetricHashing.HEADER_BITS : 0);
        // int quantizeDim = (int) Math.round(totalQuantizeBits / (float) bitDepth);
        // int encodedBits = quantizeDim + AsymmetricHashing.HEADER_BITS;

        var encodedBits = ds.getDimension() + AsymmetricHashing.HEADER_BITS;

        System.out.println("Initializing ASH");
        var ash = AsymmetricHashing.initialize(ravv, optimizer, encodedBits, bitDepth, landmarkCount);

        System.out.println("Encoding ash");
        var ashv = ash.encodeAll(ravv, ForkJoinPool.commonPool());

        var vsf = VectorSimilarityFunction.DOT_PRODUCT;
        var predk = 10;
        var gtk = 10;

        System.out.println("Getting / Computing gt");

        var gt = ds.getGroundTruth()
                .stream()
                .limit(maxQuery)
                .map(g -> g.stream().limit(gtk).collect(Collectors.toList()))
                .collect(Collectors.toList());

        // var gt = query.stream()
        //     .parallel()
        //     .map(q -> {
        //         var sf = ravv.rerankerFor(q, vsf);
        //         return IntStream.range(0, base.size())
        //             .mapToObj(i -> new NodeScore(i, sf.similarityTo(i)))
        //             .sorted()
        //             .limit(gtk)
        //             .map(ns -> ns.node)
        //             .collect(Collectors.toList());
        //     })
        //     .collect(Collectors.toList());

        System.out.println("Computing pred");
        var startNs = System.nanoTime();
        var pred = search(ashv, query, vsf, predk);
        var durationNs = System.nanoTime() - startNs;
        var durationMs = durationNs / 1e6;

        var recall = AccuracyMetrics.recallFromResults(gt, pred, gtk, predk);
        System.out.println("Recall: " + recall);
        System.out.println("Duration: " + durationMs + " ms");
    }

    static List<List<Integer>> search(CompressedVectors ashv, List<VectorFloat<?>> query, VectorSimilarityFunction vsf, int predk) {
        return query.stream()
            .map(q -> {
                var sf = ashv.scoreFunctionFor(q, vsf);
                // var sf = ravv.rerankerFor(q, vsf);
                return IntStream.range(0, ashv.count())
                    .mapToObj(i -> new NodeScore(i, sf.similarityTo(i)))
                    .sorted()
                    .limit(predk)
                    .map(ns -> ns.node)
                    .collect(Collectors.toList());
            })
            .collect(Collectors.toList());
    }
}
