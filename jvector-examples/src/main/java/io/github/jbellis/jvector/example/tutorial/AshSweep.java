package io.github.jbellis.jvector.example.tutorial;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.example.util.AccuracyMetrics;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult.NodeScore;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.ash.AbstractAshVectors;
import io.github.jbellis.jvector.quantization.ash.AshVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

public class AshSweep {
    
    private static class SweepResult {
        String dataset;
        String optimizer;
        float quantizeDimRatio;
        int quantizeDim;
        int bitDepth;
        int landmarkCount;
        int predk;
        int encodedBits;
        double recall;
        long initTimeMs;
        long encodeTimeMs;
        long predTimeNs;
        float qps;
        int nbase;
        int nq;
        
        @Override
        public String toString() {
            return String.format("%s,%s,%.2f,%d,%d,%d,%d,%d,%.4f,%d,%d,%f,%d,%d,%f",
                dataset, optimizer, quantizeDimRatio, quantizeDim, bitDepth, landmarkCount, predk,
                encodedBits, recall, initTimeMs, encodeTimeMs, predTimeNs / 1e6, nbase, nq, qps);
        }
    }
    
    public static void main(String[] args) throws IOException {
        System.out.println("Starting ASH Parameter Sweep");
        
        // Define parameter sweep ranges
        String[] datasets = {
            "cohere-english-v3-100k",
            "cohere-english-v3-1M",
            "openai-v3-large-1536-100k",
            "openai-v3-large-3072-100k",
        };
        String[] optimizers = {"RANDOM", "ITQ"};
        float[] quantizeDimRatios = {1.0f, 2.0f};
        int[] bitDepths = {2, 4};
        int[] landmarkCounts = {1};
        int[] predks = {10};
        var vsf = VectorSimilarityFunction.DOT_PRODUCT;
        var gtk = 10;
        var maxBase = 100_000_000;
        var maxQuery = 100000;
        
        // Setup CSV files
        String csvFileBasePath = "./local/ash_parameter_sweep_results";
        String csvFile = csvFileBasePath + ".csv";
        String partCsvFile = csvFile + ".part.csv";
        
        // Initialize the partial CSV file with header
        try (PrintWriter writer = new PrintWriter(new FileWriter(partCsvFile))) {
            writer.println("dataset,optimizer,quantizeDimRatio,quantizeDim,bitDepth,landmarkCount,predk,encodedBits," +
                "recall,initTimeMs,encodeTimeMs,predTimeMs,nbase,nq,QPS");
        }
        
        int totalCombinations = datasets.length * optimizers.length * quantizeDimRatios.length *
                                bitDepths.length * landmarkCounts.length * predks.length;
        int currentCombination = 0;
        
        // Parameter sweep - dataset is outermost loop
        for (String datasetName : datasets) {
            System.out.println("\n=== Loading dataset: " + datasetName + " ===");
            var ds = DataSets.loadDataSet(datasetName).orElseThrow().getDataSet();
            
            var base = ds.getBaseVectors().stream().limit(maxBase).collect(Collectors.toList());
            var ravv = new ListRandomAccessVectorValues(base, ds.getDimension());
            
            var query = ds.getQueryVectors().stream().limit(maxQuery).collect(Collectors.toList());
            
            // Prepare ground truth once per dataset
            System.out.println("Computing ground truth for " + datasetName);
            var gt = ds.getGroundTruth()
                    .stream()
                    .limit(maxQuery)
                    .map(g -> g.stream().limit(gtk).collect(Collectors.toList()))
                    .collect(Collectors.toList());
            
            for (String optimizerName : optimizers) {
                var optimizer = optimizerName.equals("RANDOM") ?
                    AsymmetricHashing.RANDOM : AsymmetricHashing.ITQ;
                
                for (float quantizeDimRatio : quantizeDimRatios) {
                    // Calculate quantizeDim from ratio: quantizeDim = dimension / ratio
                    int quantizeDim = (int) Math.round(ds.getDimension() / quantizeDimRatio);
                    
                    for (int bitDepth : bitDepths) {
                        for (int landmarkCount : landmarkCounts) {
                            // Initialize ASH once for this parameter combination
                            // (independent of predk)
                            System.out.printf("\nRunning: dataset=%s, optimizer=%s, " +
                                "quantizeDimRatio=%.2f (quantizeDim=%d), bitDepth=%d, landmarkCount=%d%n",
                                datasetName, optimizerName, quantizeDimRatio, quantizeDim, bitDepth, landmarkCount);
                            
                            try {
                                // Calculate encoding parameters
                                int encodedBits = quantizeDim + AsymmetricHashing.HEADER_BITS;
                                
                                // Initialize ASH
                                long startInit = System.currentTimeMillis();
                                var ash = AsymmetricHashing.initialize(ravv, optimizer, encodedBits, 
                                    bitDepth, landmarkCount);
                                long initTimeMs = System.currentTimeMillis() - startInit;
                                
                                // Encode vectors
                                long startEncode = System.currentTimeMillis();
                                var ashv = ash.encodeAll(ravv, ForkJoinPool.commonPool());
                                long encodeTimeMs = System.currentTimeMillis() - startEncode;
                                
                                System.out.printf("  Initialized and encoded (init=%dms, encode=%dms)%n",
                                    initTimeMs, encodeTimeMs);
                                
                                // Now sweep over predk values using the same ash and ashv
                                for (int predk : predks) {
                                    currentCombination++;
                                    System.out.printf("  [%d/%d] Testing predk=%d... ",
                                        currentCombination, totalCombinations, predk);
                                    
                                    SweepResult result = computeRecall(
                                        datasetName, ravv, query, gt, ashv, vsf,
                                        optimizerName, quantizeDimRatio, quantizeDim, bitDepth, landmarkCount,
                                        predk, gtk, encodedBits,
                                        initTimeMs, encodeTimeMs
                                    );
                                    
                                    // Write result immediately to partial CSV
                                    appendResultToCSV(result, partCsvFile);
                                    
                                    System.out.printf("recall=%.4f (predTime=%fms, qps=%f)%n",
                                        result.recall, result.predTimeNs / 1e6, result.qps);
                                }
                            } catch (Exception e) {
                                System.err.printf("  Error running experiment: %s%n", e.getMessage());
                                e.printStackTrace();
                                // Skip all predk values for this combination
                                currentCombination += predks.length;
                            }
                        }
                    }
                }
            }
        }
        
        // Copy partial CSV to final CSV
        System.out.println("\nFinalizing results...");
        try {
            Files.copy(Path.of(partCsvFile), Path.of(csvFile), StandardCopyOption.REPLACE_EXISTING);
            System.out.println("Sweep complete! Results written to " + csvFile);
        } catch (IOException e) {
            System.err.println("Error copying partial results to final file: " + e.getMessage());
            System.out.println("Results are available in " + partCsvFile);
        }
    }
    
    private static SweepResult computeRecall(
            String datasetName,
            ListRandomAccessVectorValues ravv,
            List<VectorFloat<?>> query,
            List<List<Integer>> gt,
            AshVectors ashv,
            VectorSimilarityFunction vsf,
            String optimizerName,
            float quantizeDimRatio,
            int quantizeDim,
            int bitDepth,
            int landmarkCount,
            int predk,
            int gtk,
            int encodedBits,
            long initTimeMs,
            long encodeTimeMs) {
        
        SweepResult result = new SweepResult();
        result.dataset = datasetName;
        result.optimizer = optimizerName;
        result.quantizeDimRatio = quantizeDimRatio;
        result.quantizeDim = quantizeDim;
        result.bitDepth = bitDepth;
        result.landmarkCount = landmarkCount;
        result.predk = predk;
        result.encodedBits = encodedBits;
        result.initTimeMs = initTimeMs;
        result.encodeTimeMs = encodeTimeMs;

        // warmup
        var pred = getPred(query, ashv, ravv, vsf, predk);

        // Compute predictions
        int reps = 3;
        long[] elapsedNs = new long[reps];
        for (int i = 0; i < reps; i++) {
            long startPredNs = System.nanoTime();
            pred = getPred(query, ashv, ravv, vsf, predk);
            elapsedNs[i] = System.nanoTime() - startPredNs;
        }
        result.predTimeNs = (long) Arrays.stream(elapsedNs).average().orElseThrow();
        // Arrays.stream(elapsedNs).sum() 
        result.qps = (float) (query.size() / (result.predTimeNs * 1e-9));
        System.out.println(query.size() + " " + ravv.size());

        result.nbase = ravv.size();
        result.nq = query.size();

        // Calculate recall
        // result.recall = AccuracyMetrics.recallFromResults(gt, pred, gtk, predk);
        result.recall = AccuracyMetrics.recallFromResults(gt, pred, gtk, predk);

        return result;
    }

    private static final List<List<Integer>> getPred(List<VectorFloat<?>> query, AshVectors ashv, RandomAccessVectorValues ravv, VectorSimilarityFunction vsf, int predk) {
        var pred = query
            .parallelStream()
            .map(q -> {
                PriorityQueue<NodeScore> pq = new PriorityQueue<>(predk, (a, b) -> Float.compare(a.score, b.score));
                var sf = ashv.scoreFunctionFor(q, vsf);
                for (int i = 0; i < ravv.size(); i++) {
                    float score = sf.similarityTo(i);
                    if (pq.size() < predk) {
                        pq.add(new NodeScore(i, score));
                    } else if (pq.peek().score < score) {
                        pq.poll();
                        pq.add(new NodeScore(i, score));
                    }
                }
                return pq.stream().sorted().map(ns -> ns.node).collect(Collectors.toList());
            })
            .collect(Collectors.toList());
        return pred;
    }

    private static void appendResultToCSV(SweepResult result, String filename) throws IOException {
        // Append mode - add result to existing file
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename, true))) {
            writer.println(result.toString());
        }
    }
}
