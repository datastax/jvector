package io.github.jbellis.jvector.example.tutorial;

import java.io.IOException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult.NodeScore;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

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

        float bitRatio = 1.0f;
        int bitDepth = 1;
        boolean includeHeaderBitsInCalc = true;

        int totalEncodedBits = (int) Math.round(ds.getDimension() * bitRatio);
        int totalQuantizeBits = totalEncodedBits - (includeHeaderBitsInCalc ? AsymmetricHashing.HEADER_BITS : 0);
        int quantizeDim = (int) Math.round(totalQuantizeBits / (float) bitDepth);
        int encodedBits = quantizeDim + AsymmetricHashing.HEADER_BITS;

        // var encodedBits = ds.getDimension() + AsymmetricHashing.HEADER_BITS;

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
        //     .map(q -> {
        //         var sf = ravv.rerankerFor(q, vsf);
        //         return IntStream.range(0, base.size())
        //             .parallel()
        //             .mapToObj(i -> new NodeScore(i, sf.similarityTo(i)))
        //             .sorted()
        //             .limit(gtk)
        //             .map(ns -> ns.node)
        //             .collect(Collectors.toList());
        //     })
        //     .collect(Collectors.toList());

        System.out.println("Computing pred");
        var pred = query.stream()
            .map(q -> {
                var sf = ashv.scoreFunctionFor(q, vsf);
                return IntStream.range(0, base.size())
                    .mapToObj(i -> new NodeScore(i, sf.similarityTo(i)))
                    .sorted()
                    .limit(predk)
                    .map(ns -> ns.node)
                    .collect(Collectors.toList());
            })
            .collect(Collectors.toList());
        
        var recall = AccuracyMetrics.recallFromResults(gt, pred, gtk, predk);
        System.out.println("Recall: " + recall);
    }
}
