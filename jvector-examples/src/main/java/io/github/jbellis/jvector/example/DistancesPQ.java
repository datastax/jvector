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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

public class DistancesPQ {

    /**
     * PQ benchmark aligned with DistancesASH:
     *
     *  - Same datasets
     *  - Same query normalization
     *  - Same brute-force scan pattern
     *  - Same parallel executor
     *  - Same compression budget (≈ d/32 bytes per vector)
     *
     * Fast PQ path = PQDecoder (precomputedScoreFunctionFor).
     */
    public static void testPQEncodings(String filenameBase, String filenameQueries, String filenameGT) throws IOException {
        // ------------------------------------------------------------
        // Benchmark configuration (runtime flags)
        // ------------------------------------------------------------
        final boolean RUN_RECALL_CHECK =
                Boolean.parseBoolean(System.getProperty("jvector.bench.recall", "false"));

        final int RECALL_K =
                Integer.getInteger("jvector.bench.recall.k", 10);

        final boolean RUN_ACCURACY_CHECK =
                Boolean.parseBoolean(System.getProperty("jvector.bench.accuracy-check", "false"));
        final boolean RUN_FLOAT_SCORING =
                Boolean.parseBoolean(System.getProperty("jvector.bench.float-scoring", "false"));

        List<VectorFloat<?>> vectors = SiftLoader.readFvecs(filenameBase);
        List<VectorFloat<?>> queries = SiftLoader.readFvecs(filenameQueries);

        final List<List<Integer>> groundTruth;
        if (RUN_RECALL_CHECK) {
            if (filenameGT == null || filenameGT.trim().isEmpty()) {
                throw new IllegalArgumentException(
                        "Recall is enabled but no ground-truth ivec file was provided"
                );
            }

            groundTruth = SiftLoader.readIvecs(filenameGT);

            if (groundTruth.size() < queries.size()) {
                throw new IllegalArgumentException(
                        "Ground truth contains " + groundTruth.size()
                                + " rows but there are " + queries.size() + " queries"
                );
            }

            for (int i = 0; i < queries.size(); i++) {
                if (groundTruth.get(i).size() < RECALL_K) {
                    throw new IllegalArgumentException(
                            "Ground truth row " + i + " has only "
                                    + groundTruth.get(i).size()
                                    + " entries, but recall.k=" + RECALL_K
                    );
                }
            }
        } else {
            groundTruth = java.util.Collections.emptyList();
        }

        // ------------------------------------------------------------
        // Parallel executor (identical to DistancesASH)
        // ------------------------------------------------------------
//        ForkJoinPool simdExecutor = PhysicalCoreExecutor.pool(); // For production
        ForkJoinPool simdExecutor = new ForkJoinPool(180); // For profiling
        int parallelism = simdExecutor.getParallelism();
        int chunkSize = Math.max(1, (queries.size() + parallelism - 1) / parallelism);

        // ------------------------------------------------------------
        // PQ parameters
        // ------------------------------------------------------------
        // M is derived from the input dimension:
        //
        //   M = dimension / mFactor
        //
        // With K=256, each subspace code is one byte, so PQ stores M bytes
        // per vector. Since the input is float32, the exact compression ratio is:
        //
        //   (dimension * 4 bytes) / M bytes = 4 * mFactor
        //
        // This requires dimension to be evenly divisible by mFactor.
        int dimension = vectors.get(0).length();

        final int mFactor = Integer.getInteger("jvector.pq.mFactor", 8);
        if (mFactor <= 0) {
            throw new IllegalArgumentException(
                    "jvector.pq.mFactor must be positive; got " + mFactor
            );
        }
        if (dimension % mFactor != 0) {
            throw new IllegalArgumentException(
                    "dimension must be evenly divisible by jvector.pq.mFactor: "
                            + "dimension=" + dimension
                            + ", mFactor=" + mFactor
            );
        }

        final int M = dimension / mFactor;
        final int K = 256;                 // 1 byte per subspace
        final boolean globallyCenter = false;
        final float anisotropicThreshold = 0.0f;

        final int inputBytesPerVector = dimension * Float.BYTES;
        final int compressedBytesPerVector = M;
        final double compressionRatio =
                (double) inputBytesPerVector / compressedBytesPerVector;

        System.out.println("\tPQ params:");
        System.out.println("\t  dimension = " + dimension);
        System.out.println("\t  mFactor = " + mFactor);
        System.out.println("\t  M = " + M + " subspaces");
        System.out.println("\t  K = " + K + " clusters");
        System.out.println("\t  Input size = " + inputBytesPerVector + " bytes/vector");
        System.out.println("\t  PQ size = " + compressedBytesPerVector + " bytes/vector");
        System.out.println(
                "\t  Compression ratio = "
                        + String.format(java.util.Locale.ROOT, "%.3f", compressionRatio)
                        + "x"
        );

        // ------------------------------------------------------------
        // Train PQ + encode (timed)
        // ------------------------------------------------------------
        RandomAccessVectorValues ravv =
                new ListRandomAccessVectorValues(vectors, dimension);

        long pqTrainStart = System.nanoTime();
        ProductQuantization pq = ProductQuantization.compute(
                ravv,
                M,
                K,
                globallyCenter,
                anisotropicThreshold,
                simdExecutor,
                ForkJoinPool.commonPool()
        );
        long pqTrainEnd = System.nanoTime();

        long pqEncodeStart = System.nanoTime();
        CompressedVectors pqAny = pq.encodeAll(ravv, simdExecutor);
        long pqEncodeEnd = System.nanoTime();

        PQVectors pqVecs = (PQVectors) pqAny;

        System.out.println("\tPQ training took " + (pqTrainEnd - pqTrainStart) / 1e9 + " seconds");
        System.out.println("\tPQ encoding took " + (pqEncodeEnd - pqEncodeStart) / 1e9 + " seconds");
        double encSeconds = (pqEncodeEnd - pqEncodeStart) / 1e9;
        double encThroughput = vectors.size() / encSeconds;

        System.out.println(
                "\tEncoding throughput = "
                        + String.format(java.util.Locale.ROOT, "%.3f", encThroughput)
                        + " vectors/sec"
        );


        // ============================================================
        // [1] Accuracy (NOT timed)
        //     Compare unscaled dot products:
        //       PQ returns (1 + dp)/2  -> dp = 2*score - 1
        // ============================================================
        if (RUN_ACCURACY_CHECK) {
            List<ForkJoinTask<double[]>> errorTasks = new ArrayList<>();

            for (int start = 0; start < queries.size(); start += chunkSize) {
                final int s = start;
                final int e = Math.min(start + chunkSize, queries.size());

                errorTasks.add(simdExecutor.submit(() -> {
                    double localError = 0.0;
                    long localCount = 0;

                    for (int i = s; i < e; i++) {
                        VectorFloat<?> q = queries.get(i);
                        ScoreFunction.ApproximateScoreFunction f =
                                pqVecs.precomputedScoreFunctionFor(q, VectorSimilarityFunction.DOT_PRODUCT);

                        for (int j = 0; j < vectors.size(); j++) {
                            float trueDot = VectorUtil.dotProduct(q, vectors.get(j));
                            float approxScaled = f.similarityTo(j);
                            float approxDot = 2.0f * approxScaled - 1.0f;

                            localError += Math.abs(approxDot - trueDot);
                            localCount++;
                        }
                    }
                    return new double[]{localError, localCount};
                }));
            }

            double totalError = 0.0;
            long totalCount = 0;
            for (ForkJoinTask<double[]> t : errorTasks) {
                double[] r = t.join();
                totalError += r[0];
                totalCount += (long) r[1];
            }

            System.out.println("\tAverage absolute dot-product error = " + (totalError / totalCount));
        }

        // ============================================================
        // [1b] Recall@K run (NOT timed)
        // ============================================================
        if (RUN_RECALL_CHECK) {
            final int[] atValues = {10, 15, 20, 30, 40, 50};
            final int maxAt = atValues[atValues.length - 1];

            List<ForkJoinTask<double[]>> recallTasks = new ArrayList<>();

            System.out.println("\t[stage] Computing " + RECALL_K + "-Recall@K...");
            System.out.flush();

            for (int start = 0; start < queries.size(); start += chunkSize) {
                final int s = start;
                final int e = Math.min(start + chunkSize, queries.size());

                recallTasks.add(simdExecutor.submit(() -> {
                    double[] localTotalRecall = new double[atValues.length];

                    for (int i = s; i < e; i++) {
                        ScoreFunction.ApproximateScoreFunction f =
                                pqVecs.precomputedScoreFunctionFor(
                                        queries.get(i),
                                        VectorSimilarityFunction.DOT_PRODUCT
                                );

                        // Min-heap keeping the top maxAt approximate results.
                        var topCandidates =
                                new java.util.PriorityQueue<long[]>((a, b) ->
                                        Float.compare(
                                                Float.intBitsToFloat((int) a[0]),
                                                Float.intBitsToFloat((int) b[0])
                                        ));

                        for (int j = 0; j < vectors.size(); j++) {
                            float score = f.similarityTo(j);

                            if (topCandidates.size() < maxAt) {
                                topCandidates.add(new long[]{Float.floatToRawIntBits(score), j});
                            } else if (score > Float.intBitsToFloat((int) topCandidates.peek()[0])) {
                                topCandidates.poll();
                                topCandidates.add(new long[]{Float.floatToRawIntBits(score), j});
                            }
                        }

                        int[] topIndices = new int[topCandidates.size()];
                        for (int rank = topCandidates.size() - 1; rank >= 0; rank--) {
                            topIndices[rank] = (int) topCandidates.poll()[1];
                        }

                        int[] queryGT = groundTruth.get(i).stream()
                                .mapToInt(Integer::intValue)
                                .toArray();

                        for (int aIdx = 0; aIdx < atValues.length; aIdx++) {
                            int at = atValues[aIdx];

                            java.util.Set<Integer> topAtSet = new java.util.HashSet<>();
                            for (int r = 0; r < Math.min(at, topIndices.length); r++) {
                                topAtSet.add(topIndices[r]);
                            }

                            int matches = 0;
                            java.util.Set<Integer> gtSeen = new java.util.HashSet<>(RECALL_K * 2);

                            // Avoid double-counting duplicate GT ids.
                            for (int g = 0; g < RECALL_K; g++) {
                                int gtId = queryGT[g];
                                if (gtSeen.add(gtId) && topAtSet.contains(gtId)) {
                                    matches++;
                                }
                            }

                            localTotalRecall[aIdx] += (double) matches / RECALL_K;
                        }
                    }

                    return localTotalRecall;
                }));
            }

            double[] totalRecall = new double[atValues.length];
            for (ForkJoinTask<double[]> t : recallTasks) {
                double[] local = t.join();
                for (int aIdx = 0; aIdx < atValues.length; aIdx++) {
                    totalRecall[aIdx] += local[aIdx];
                }
            }

            for (int aIdx = 0; aIdx < atValues.length; aIdx++) {
                System.out.format(
                        "\tPQ %d-recall@%d = %.4f%n",
                        RECALL_K,
                        atValues[aIdx],
                        totalRecall[aIdx] / queries.size()
                );
            }
        }

        // ============================================================
        // [2] Fast PQ scan timing (PQDecoder / ADC)
        // ============================================================
        List<ForkJoinTask<Double>> pqTasks = new ArrayList<>();
        long pqStart = System.nanoTime();

        for (int start = 0; start < queries.size(); start += chunkSize) {
            final int s = start;
            final int e = Math.min(start + chunkSize, queries.size());

            pqTasks.add(simdExecutor.submit(() -> {
                double localSum = 0.0;
                for (int i = s; i < e; i++) {
                    ScoreFunction.ApproximateScoreFunction f =
                            pqVecs.precomputedScoreFunctionFor(
                                    queries.get(i),
                                    VectorSimilarityFunction.DOT_PRODUCT
                            );
                    for (int j = 0; j < vectors.size(); j++) {
                        localSum += f.similarityTo(j);
                    }
                }
                return localSum;
            }));
        }

        double pqDummy = 0.0;
        for (ForkJoinTask<Double> t : pqTasks) {
            pqDummy += t.join();
        }
        // Prevent dead-code elimination
        System.out.println("\tdummyAccumulator = " + (float) (pqDummy));
        System.out.println("--");

        long pqEnd = System.nanoTime();
        System.out.println("\tPQDecoder scan took " + (pqEnd - pqStart) / 1e9 + " seconds");

        double blockSeconds = (pqEnd - pqStart) / 1e9;
        long totalDotProducts = (long) queries.size() * (long) vectors.size();

        double scoreThroughput = totalDotProducts / blockSeconds;

        System.out.println(
                "\tDecoder scan throughput = "
                        + String.format(java.util.Locale.ROOT, "%.3f", scoreThroughput)
                        + " dot-products/sec"
                        + " ("
                        + String.format(java.util.Locale.ROOT, "%.3f", scoreThroughput / 1e6)
                        + " Mdot/s)"
        );

        // ============================================================
        // [3] Float dot-product baseline
        // ============================================================
        if (RUN_FLOAT_SCORING) {
            List<ForkJoinTask<Double>> floatTasks = new ArrayList<>();
            long floatStart = System.nanoTime();

            for (int start = 0; start < queries.size(); start += chunkSize) {
                final int s = start;
                final int e = Math.min(start + chunkSize, queries.size());

                floatTasks.add(simdExecutor.submit(() -> {
                    double localSum = 0.0;
                    for (int i = s; i < e; i++) {
                        VectorFloat<?> q = queries.get(i);
                        for (int j = 0; j < vectors.size(); j++) {
                            localSum += VectorUtil.dotProduct(q, vectors.get(j));
                        }
                    }
                    return localSum;
                }));
            }

            double floatDummy = 0.0;
            for (ForkJoinTask<Double> t : floatTasks) {
                floatDummy += t.join();
            }

            long floatEnd = System.nanoTime();
            System.out.println("\tFloat dot-product scan took " + (floatEnd - floatStart) / 1e9 + " seconds");

            // Prevent dead-code elimination
            System.out.println("\tdummyAccumulator = " + (float) (floatDummy));
            System.out.println("--");
        }
    }

    // ------------------------------------------------------------------
    // Dataset runners (mirrors DistancesASH)
    // ------------------------------------------------------------------

    public static void runCohere100k() throws IOException {
        System.out.println("Running Cohere-100k");

        var baseVectors = "./fvec/cohere-100k/cohere_embed-english-v3.0_1024_base_vectors_100000.fvec";
        var queryVectors = "./fvec/cohere-100k/cohere_embed-english-v3.0_1024_query_vectors_10000.fvec";
        var gtVectors = ""; // Set this to the GT file for this exact base/query ordering before enabling recall.

        testPQEncodings(baseVectors, queryVectors, gtVectors);
    }

    public static void runADA() throws IOException {
        System.out.println("Running ada_002");

        var baseVectors = "./fvec/ada-002/ada_002_100000_base_vectors.fvec";
        var queryVectors = "./fvec/ada-002/ada_002_100000_query_vectors_10000.fvec";
        var gtVectors = "./fvec/ada-002/ada_002_100000_indices_query_10000.ivec";

        testPQEncodings(baseVectors, queryVectors, gtVectors);
    }

    public static void runADANoZeros() throws IOException {
        System.out.println("Running ada-002-no-zeros");

        var baseVectors = "./fvec/ada-002-no-zeros/ada_002_100000_base_vectors_no_zeros.fvec";
        var queryVectors = "./fvec/ada-002-no-zeros/ada_002_100000_query_vectors_10000_no_zeros.fvec";
        var gtVectors = "./fvec/ada-002-no-zeros/ada-002_gt_no_zeros.ivec";

        testPQEncodings(baseVectors, queryVectors, gtVectors);
    }

    public static void runOpenai1536() throws IOException {
        System.out.println("Running text-embedding-3-large_1536");

        var baseVectors = "./fvec/openai-v3-large-1536-100k/openai_v3_large_1536_100k_base_98716.fvecs";
        var queryVectors = "./fvec/openai-v3-large-1536-100k/openai_v3_large_1536_100k_query_10000.fvecs";
        var gtVectors = "./fvec/openai-v3-large-1536-100k/openai_v3_large_1536_100k_gt_ip_100.ivecs";

        testPQEncodings(baseVectors, queryVectors, gtVectors);
    }

    public static void runOpenai3072() throws IOException {
        System.out.println("Running text-embedding-3-large_3072");

        var baseVectors = "./fvec/openai-v3-large-3072-100k/text-embedding-3-large_3072_100000_base_vectors.fvec";
        var queryVectors = "./fvec/openai-v3-large-3072-100k/text-embedding-3-large_3072_100000_query_vectors_10000.fvec";
        var gtVectors = "./fvec/openai-v3-large-3072-100k/text-embedding-3-large_3072_100000_indices_query_10000.ivec";

        testPQEncodings(baseVectors, queryVectors, gtVectors);
    }

    public static void runCap6m() throws IOException {
        System.out.println("Running cap-6m");

        var baseVectors = "./fvec/cap-6m/Caselaw_gte-Qwen2-1.5B_embeddings_base_6m_norm_shuffle.fvecs";
        var queryVectors = "./fvec/cap-6m/Caselaw_gte-Qwen2-1.5B_embeddings_query_10k_norm_shuffle.fvecs";
        var gtVectors = "./fvec/cap-6m/cap_6m_gt_norm_shuffle_ip_k100.ivecs";

        testPQEncodings(baseVectors, queryVectors, gtVectors);
    }

    public static void runCohere10m() throws IOException {
        System.out.println("Running cohere-10m");

        var baseVectors = "./fvec/cohere-10m/cohere_wiki_en_flat_base_10m_norm.fvecs";
        var queryVectors = "./fvec/cohere-10m/cohere_wiki_en_flat_query_10k_norm.fvecs";
        var gtVectors = "./fvec/cohere-10m/cohere_wiki_en_flat_gt_10m_ip_k100.ivecs";

        testPQEncodings(baseVectors, queryVectors, gtVectors);
    }

    public static void main(String[] args) throws IOException {
//        runCohere100k();
//        runADA();
//        runADANoZeros();
        runOpenai1536();
//        runOpenai3072();
//        runCap6m();
//        runCohere10m();
    }
}
