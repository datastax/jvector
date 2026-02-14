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
    public static void testPQEncodings(String filenameBase, String filenameQueries) throws IOException {
        // ------------------------------------------------------------
        // Benchmark configuration (runtime flags)
        // ------------------------------------------------------------
        final boolean RUN_ACCURACY_CHECK =
                Boolean.parseBoolean(System.getProperty("jvector.bench.accuracy-check", "false"));
        final boolean RUN_FLOAT_SCORING =
                Boolean.parseBoolean(System.getProperty("jvector.bench.float-scoring", "false"));

        // Define the benchmark size
        int maxQueries = 10_000;
        int maxVectors = 10_000_000;

        int queryCountInFile = SiftLoader.countFvecs(filenameQueries);
        int vectorCountInFile = SiftLoader.countFvecs(filenameBase);

        // Will benchmark what was requested or the maximum available vectors (if less)
        int nQueries = Math.min(maxQueries, queryCountInFile);
        int nVectors = Math.min(maxVectors, vectorCountInFile);

        List<VectorFloat<?>> vectors = SiftLoader.readFvecs(filenameBase, nVectors);
        List<VectorFloat<?>> queries = SiftLoader.readFvecs(filenameQueries, nQueries);

        // ------------------------------------------------------------
        // Remove zero-norm vectors (match DistancesASH)
        // ------------------------------------------------------------
        int beforeBase = vectors.size();
        List<VectorFloat<?>> filteredBase = new ArrayList<>(vectors.size());
        for (VectorFloat<?> v : vectors) {
            if (VectorUtil.dotProduct(v, v) > 0f) {
                filteredBase.add(v);
            }
        }
        vectors = filteredBase;

        int beforeQuery = queries.size();
        List<VectorFloat<?>> filteredQueries = new ArrayList<>(queries.size());
        for (VectorFloat<?> q : queries) {
            if (VectorUtil.dotProduct(q, q) > 0f) {
                filteredQueries.add(q);
            }
        }
        queries = filteredQueries;

        System.out.println("\tRemoved " + (beforeBase - vectors.size()) + " zero base vectors and "
                + (beforeQuery - queries.size()) + " zero query vectors");

        // ------------------------------------------------------------
        // Update counts after filtering
        // ------------------------------------------------------------
        final int finalVectorCount = vectors.size();
        final int finalQueryCount = queries.size();

        // Match ASH: normalize queries only
        for (VectorFloat<?> q : queries) {
            VectorUtil.l2normalize(q);
        }

        int dimension = vectors.get(0).length();

        final List<VectorFloat<?>> finalVectors = vectors;
        final List<VectorFloat<?>> finalQueries = queries;

        System.out.format("\t%d base and %d query vectors loaded, dimension=%d%n",
                vectors.size(), queries.size(), dimension);

        // ------------------------------------------------------------
        // Parallel executor (identical to DistancesASH)
        // ------------------------------------------------------------
//        ForkJoinPool simdExecutor = PhysicalCoreExecutor.pool(); // For production
        ForkJoinPool simdExecutor = new ForkJoinPool(180); // For profiling
        int parallelism = simdExecutor.getParallelism();
        int chunkSize = Math.max(1, (finalQueryCount + parallelism - 1) / parallelism);

        // ------------------------------------------------------------
        // PQ parameters (EXACT bit-for-bit match with ASH)
        // ------------------------------------------------------------
        // ASH total size = encodedBits (includes header + payload)
        // PQ total size  = M * 8 bits (1 byte per subspace)
        //
        // We REQUIRE exact equality for fair benchmarking.
        int HEADER_BITS = 72;
        int encodedBitsASH = (dimension / 4) + HEADER_BITS; // HEADER_BITS = 72
        if ((encodedBitsASH & 7) != 0) {
            throw new IllegalArgumentException(
                    "PQ requires encodedBits to be a multiple of 8 for exact bit matching; got " + encodedBitsASH
            );
        }

        final int M = encodedBitsASH / 8;
        final int K = 256;                 // 1 byte per subspace
        final boolean globallyCenter = false;
        final float anisotropicThreshold = 0.0f;

        System.out.println("\tPQ params:");
        System.out.println("\t  M = " + M + " subspaces");
        System.out.println("\t  K = " + K + " clusters");
        System.out.println("\t  Compression = " + (M * 8) + " bits (" + M + " bytes)");
        System.out.println("\tASH reference = " + encodedBitsASH + " bits (" + (encodedBitsASH / 8) + " bytes)");


        // ------------------------------------------------------------
        // Train PQ + encode (timed)
        // ------------------------------------------------------------
        RandomAccessVectorValues ravv =
                new ListRandomAccessVectorValues(finalVectors, dimension);

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
        double encThroughput = finalVectorCount / encSeconds;

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

            for (int start = 0; start < finalQueryCount; start += chunkSize) {
                final int s = start;
                final int e = Math.min(start + chunkSize, finalQueryCount);

                errorTasks.add(simdExecutor.submit(() -> {
                    double localError = 0.0;
                    long localCount = 0;

                    for (int i = s; i < e; i++) {
                        VectorFloat<?> q = finalQueries.get(i);
                        ScoreFunction.ApproximateScoreFunction f =
                                pqVecs.precomputedScoreFunctionFor(q, VectorSimilarityFunction.DOT_PRODUCT);

                        for (int j = 0; j < finalVectorCount; j++) {
                            float trueDot = VectorUtil.dotProduct(q, finalVectors.get(j));
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
        // [2] Fast PQ scan timing (PQDecoder / ADC)
        // ============================================================
        List<ForkJoinTask<Double>> pqTasks = new ArrayList<>();
        long pqStart = System.nanoTime();

        for (int start = 0; start < finalQueryCount; start += chunkSize) {
            final int s = start;
            final int e = Math.min(start + chunkSize, finalQueryCount);

            pqTasks.add(simdExecutor.submit(() -> {
                double localSum = 0.0;
                for (int i = s; i < e; i++) {
                    ScoreFunction.ApproximateScoreFunction f =
                            pqVecs.precomputedScoreFunctionFor(
                                    finalQueries.get(i),
                                    VectorSimilarityFunction.DOT_PRODUCT
                            );
                    for (int j = 0; j < finalVectorCount; j++) {
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
        long totalDotProducts = (long) finalQueryCount * (long) finalVectorCount;

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

            for (int start = 0; start < finalQueryCount; start += chunkSize) {
                final int s = start;
                final int e = Math.min(start + chunkSize, finalQueryCount);

                floatTasks.add(simdExecutor.submit(() -> {
                    double localSum = 0.0;
                    for (int i = s; i < e; i++) {
                        VectorFloat<?> q = finalQueries.get(i);
                        for (int j = 0; j < finalVectorCount; j++) {
                            localSum += VectorUtil.dotProduct(q, finalVectors.get(j));
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
        testPQEncodings(
                "./fvec/cohere-100k/cohere_embed-english-v3.0_1024_base_vectors_100000.fvec",
                "./fvec/cohere-100k/cohere_embed-english-v3.0_1024_query_vectors_10000.fvec"
        );
    }

    public static void runADA() throws IOException {
        System.out.println("Running ada_002");
        testPQEncodings(
                "./fvec/ada-002/ada_002_100000_base_vectors.fvec",
                "./fvec/ada-002/ada_002_100000_query_vectors_10000.fvec"
        );
    }

    public static void runADANoZeros() throws IOException {
        System.out.println("Running ada-002-no-zeros");

        testPQEncodings(
                "./fvec/ada-002-no-zeros/ada_002_100000_base_vectors_no_zeros.fvec",
                "./fvec/ada-002-no-zeros/ada_002_100000_query_vectors_10000_no_zeros.fvec"
        );
    }

    public static void runOpenai1536() throws IOException {
        System.out.println("Running text-embedding-3-large_1536");
        testPQEncodings(
                "./fvec/openai-v3-large-1536-100k/text-embedding-3-large_1536_100000_base_vectors.fvec",
                "./fvec/openai-v3-large-1536-100k/text-embedding-3-large_1536_100000_query_vectors_10000.fvec"
        );
    }

    public static void runOpenai3072() throws IOException {
        System.out.println("Running text-embedding-3-large_3072");
        testPQEncodings(
                "./fvec/openai-v3-large-3072-100k/text-embedding-3-large_3072_100000_base_vectors.fvec",
                "./fvec/openai-v3-large-3072-100k/text-embedding-3-large_3072_100000_query_vectors_10000.fvec"
        );
    }

    public static void runCap6m() throws IOException {
        System.out.println("Running cap-6m");

        var baseVectors = "./fvec/cap-6m/Caselaw_gte-Qwen2-1.5B_embeddings_base_6m_norm_shuffle.fvecs";
        var queryVectors = "./fvec/cap-6m/Caselaw_gte-Qwen2-1.5B_embeddings_query_10k_norm_shuffle.fvecs";
        testPQEncodings(baseVectors, queryVectors);
    }

    public static void runCohere10m() throws IOException {
        System.out.println("Running cohere-10m");

        var baseVectors = "./fvec/cohere-10m/cohere_wiki_en_flat_base_10m_norm.fvecs";
        var queryVectors = "./fvec/cohere-10m/cohere_wiki_en_flat_query_10k_norm.fvecs";
        testPQEncodings(baseVectors, queryVectors);
    }

    public static void main(String[] args) throws IOException {
//        runCohere100k();
//        runADA();
//        runADANoZeros();
//        runOpenai1536();
//        runOpenai3072();
//        runCap6m();
        runCohere10m();
    }
}
