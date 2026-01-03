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
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.quantization.ASHBlockScorer;
import io.github.jbellis.jvector.quantization.CompressedVectors;

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.quantization.ASHVectors;

import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;

import java.io.IOException;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

public class DistancesASH {
    private static void printScorerInfo(String label, ASHBlockScorer scorer) {
        System.out.println(
                "\t[" + label + "] scorer implementation = "
                        + scorer.getClass().getName()
        );
    }

    public static void testASHEncodings(String filenameBase, String filenameQueries) throws IOException {
        // ------------------------------------------------------------
        // Benchmark configuration (runtime flags)
        // ------------------------------------------------------------
        final boolean RUN_SANITY_CHECK =
                Boolean.parseBoolean(System.getProperty("jvector.bench.sanity-check", "false"));

        final boolean RUN_ACCURACY_CHECK =
                Boolean.parseBoolean(System.getProperty("jvector.bench.accuracy", "true"));

        final boolean RUN_SCALAR_SCORING =
                Boolean.parseBoolean(System.getProperty("jvector.bench.scalar-scoring", "false"));

        final boolean RUN_FLOAT_SCORING =
                Boolean.parseBoolean(System.getProperty("jvector.bench.float-scoring", "false"));

        // Block sizes to benchmark
        final int[] BLOCK_SIZES = {16, 32, 64}; // 16, 32, and/or 64

        // Define the benchmark size
        int maxQueries = 1_000;
        int maxVectors = 100_000;

        int queryCountInFile = SiftLoader.countFvecs(filenameQueries);
        int vectorCountInFile = SiftLoader.countFvecs(filenameBase);

        // Will benchmark what was requested or the maximum available vectors (if less)
        int nQueries = Math.min(maxQueries, queryCountInFile);
        int nVectors = Math.min(maxVectors, vectorCountInFile);

        List<VectorFloat<?>> vectors = SiftLoader.readFvecs(filenameBase, nVectors);
        List<VectorFloat<?>> queries = SiftLoader.readFvecs(filenameQueries, nQueries);

        // ------------------------------------------------------------------
        // Remove zero-norm vectors (undefined for ASH / Eq. 11)
        // ------------------------------------------------------------------
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

        // ASH normalization policy:
        //
        // - Base vectors x are NOT globally normalized.
        // - For encoding, we compute μ on raw x.
        // - The residual (x − μ) is normalized ONCE inside the binarizer,
        //   producing \hat{x} = (x − μ) / ||x − μ|| (ASH Paper, Eq. 6).
        // - Queries may be L2-normalized in the benchmark (standard practice),
        //   but are NOT normalized inside the encoder.
        // - No other normalization steps are applied.
        for (VectorFloat<?> q : queries) VectorUtil.l2normalize(q);

        int dimension = vectors.get(0).length();
        int encodedBits = dimension / 4; // sweep later if desired

        System.out.println(
                "\toriginalDim = " + dimension +
                        ", encodedBits = " + encodedBits
        );

        final List<VectorFloat<?>> finalQueries = queries;
        final List<VectorFloat<?>> finalVectors = vectors;

        System.out.format("\t%d base and %d query vectors loaded, dimension=%d%n",
                vectors.size(), queries.size(), dimension);

        // ------------------------------------------------------------------
        // Build ASH
        // ------------------------------------------------------------------
        var ravv = new ListRandomAccessVectorValues(vectors, dimension);
        var ash = AsymmetricHashing.initialize(ravv, AsymmetricHashing.RANDOM, encodedBits);

        long startTime = System.nanoTime();
        CompressedVectors ashVecs = ash.encodeAll(ravv);
        long endTime = System.nanoTime();
        System.out.println("\tEncoding took " + (endTime - startTime) / 1e9 + " seconds");

        double encSeconds = (endTime - startTime) / 1e9;
        double encThroughput = finalVectorCount / encSeconds;

        System.out.println(
                "\tEncoding throughput = "
                        + String.format(java.util.Locale.ROOT, "%.3f", encThroughput)
                        + " vectors/sec"
        );

        ASHVectors ashVectors = (ASHVectors) ashVecs;
        VectorFloat<?> q0 = finalQueries.get(0);

        if(RUN_SANITY_CHECK) {

            ASHBlockScorer ref = ashVectors.blockScorerFor(q0, VectorSimilarityFunction.DOT_PRODUCT);
            printScorerInfo("scalar-ref", ref);

            final int sanityBlockSize = BLOCK_SIZES[0];
            ASHBlockScorer blk = ashVectors.blockScorerFor(q0, VectorSimilarityFunction.DOT_PRODUCT, sanityBlockSize);
            printScorerInfo("block", blk);


            float[] a = new float[128];
            float[] b = new float[128];
            ref.scoreRange(0, 128, a);
            blk.scoreRange(0, 128, b);

            final float EPS = 1e-6f;

            for (int i = 0; i < 128; i++) {
                float diff = Math.abs(a[i] - b[i]);
                if (diff > EPS) {
                    throw new AssertionError(
                            "Mismatch at " + i +
                                    ": ref=" + a[i] +
                                    ", blk=" + b[i] +
                                    ", diff=" + diff);
                }
            }
        }

        // ------------------------------------------------------------------
        // Shared parallelization setup (executor-honoring)
        // ------------------------------------------------------------------
         ForkJoinPool simdExecutor = PhysicalCoreExecutor.pool(); // For production
        // ForkJoinPool simdExecutor = new ForkJoinPool(4); // For profiling

        int parallelism = simdExecutor.getParallelism();
        int chunkSize = Math.max(1, (finalQueryCount + parallelism - 1) / parallelism);

        // ==================================================================
        // [1] Accuracy run (NOT timed)
        // ==================================================================
        if(RUN_ACCURACY_CHECK) {
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
                                ashVecs.scoreFunctionFor(q, VectorSimilarityFunction.DOT_PRODUCT);

                        for (int j = 0; j < finalVectorCount; j++) {
                            VectorFloat<?> v = finalVectors.get(j);
                            float trueDot = VectorUtil.dotProduct(q, v);
                            float approxDot = f.similarityTo(j);

                            localError += Math.abs(approxDot - trueDot);
                            localCount++;
                        }
                    }
                    return new double[]{localError, localCount};
                }));
            }

            double distanceError = 0.0;
            long count = 0;

            for (ForkJoinTask<double[]> t : errorTasks) {
                double[] r = t.join();
                distanceError += r[0];
                count += (long) r[1];
            }

            distanceError /= count;
            System.out.println("\tAverage absolute dot-product error = " + distanceError);
        }

        // ==================================================================
        // [2] Scalar ASH scoring timing (reference)
        // ==================================================================
        if (RUN_SCALAR_SCORING) {
            List<ForkJoinTask<Double>> ashTasks = new ArrayList<>();

            long ashStart = System.nanoTime();

            for (int start = 0; start < finalQueryCount; start += chunkSize) {
                final int s = start;
                final int e = Math.min(start + chunkSize, finalQueryCount);

                ashTasks.add(simdExecutor.submit(() -> {
                    double localSum = 0.0;

                    for (int i = s; i < e; i++) {
                        VectorFloat<?> q = finalQueries.get(i);
                        ScoreFunction.ApproximateScoreFunction f =
                                ashVecs.scoreFunctionFor(q, VectorSimilarityFunction.DOT_PRODUCT);

                        for (int j = 0; j < finalVectorCount; j++) {
                            localSum += f.similarityTo(j);
                        }
                    }
                    return localSum;
                }));
            }

            double ashDummy = 0.0;
            for (ForkJoinTask<Double> t : ashTasks) {
                ashDummy += t.join();
            }

            long ashEnd = System.nanoTime();

            System.out.println("\tScalar ASH dot-product computations took "
                    + (ashEnd - ashStart) / 1e9 + " seconds");
            System.out.println("\tdummyAccumulator = " + (float) (ashDummy));
            System.out.println("--");
        }

        // ==================================================================
        // [3] Float dot-product scoring (ground truth baseline)
        // ==================================================================
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

            System.out.println("\tFloat dot-product computations took "
                    + (floatEnd - floatStart) / 1e9 + " seconds");

            // Prevent dead-code elimination
            System.out.println("\tdummyAccumulator = " + (float) (floatDummy));
            System.out.println("--");
        }

        // ==================================================================
        // [4] Block scoring timing (single mode per JVM run)
        // NOTE: Choose the block kernel using:
        //   -Djvector.ash.blockKernel=scalar|simd|auto
        // ==================================================================
        final String kernelMode =
                System.getProperty("jvector.ash.blockKernel", "auto").toLowerCase();

        for (int blockSize : BLOCK_SIZES) {

            // Print scorer implementation once per blockSize (diagnostic)
            {
                ASHBlockScorer scorer =
                        ashVectors.blockScorerFor(
                                finalQueries.get(0),
                                VectorSimilarityFunction.DOT_PRODUCT,
                                blockSize
                        );

                printScorerInfo(kernelMode + " (blockSize=" + blockSize + ")", scorer);
            }

            List<ForkJoinTask<Double>> blockTasks = new ArrayList<>();
            long blockStart = System.nanoTime();

            for (int start = 0; start < finalQueryCount; start += chunkSize) {
                final int s = start;
                final int e = Math.min(start + chunkSize, finalQueryCount);

                blockTasks.add(simdExecutor.submit(() -> {
                    double localSum = 0.0;
                    final float[] scores = new float[blockSize];

                    for (int qi = s; qi < e; qi++) {
                        ASHBlockScorer scorer =
                                ashVectors.blockScorerFor(
                                        finalQueries.get(qi),
                                        VectorSimilarityFunction.DOT_PRODUCT,
                                        blockSize
                                );

                        int j = 0;
                        while (j + blockSize <= finalVectorCount) {
                            scorer.scoreRange(j, blockSize, scores);
                            for (int k = 0; k < blockSize; k++) {
                                localSum += scores[k];
                            }
                            j += blockSize;
                        }

                        // Tail uses scalar per-vector scorer (correctness-preserving)
                        if (j < finalVectorCount) {
                            ScoreFunction.ApproximateScoreFunction f =
                                    ashVecs.scoreFunctionFor(
                                            finalQueries.get(qi),
                                            VectorSimilarityFunction.DOT_PRODUCT
                                    );
                            for (; j < finalVectorCount; j++) {
                                localSum += f.similarityTo(j);
                            }
                        }
                    }

                    return localSum;
                }));
            }

            double blockDummy = 0.0;
            for (ForkJoinTask<Double> t : blockTasks) {
                blockDummy += t.join();
            }

            long blockEnd = System.nanoTime();

            System.out.println(
                    "\tBlock " + kernelMode + " (blockSize=" + blockSize + ") took "
                            + (blockEnd - blockStart) / 1e9 + " seconds"
            );

            double blockSeconds = (blockEnd - blockStart) / 1e9;
            long totalDotProducts = (long) finalQueryCount * (long) finalVectorCount;

            double scoreThroughput = totalDotProducts / blockSeconds;

            System.out.println(
                    "\tBlock " + kernelMode + " throughput (blockSize=" + blockSize + ") = "
                            + String.format(java.util.Locale.ROOT, "%.3f", scoreThroughput)
                            + " dot-products/sec"
                            + " ("
                            + String.format(java.util.Locale.ROOT, "%.3f", scoreThroughput / 1e6)
                            + " Mdot/s)"
            );

            System.out.println("\tdummyAccumulator = " + (float) (blockDummy));
            System.out.println("--");
        }
    }

    public static void runSIFT() throws IOException {
        System.out.println("Running siftsmall");

        var baseVectors = "siftsmall/siftsmall_base.fvecs";
        var queryVectors = "siftsmall/siftsmall_query.fvecs";
        testASHEncodings(baseVectors, queryVectors);
    }

    public static void runGIST() throws IOException {
        System.out.println("Running GIST");

        var baseVectors = "./fvec/gist/gist_base.fvecs";
        var queryVectors = "./fvec/gist/gist_query.fvecs";
        testASHEncodings(baseVectors, queryVectors);
    }

    public static void runCohere100k() throws IOException {
        System.out.println("Running Cohere-100k");

        var baseVectors = "./fvec/cohere-100k/cohere_embed-english-v3.0_1024_base_vectors_100000.fvec";
        var queryVectors = "./fvec/cohere-100k/cohere_embed-english-v3.0_1024_query_vectors_10000.fvec";
        testASHEncodings(baseVectors, queryVectors);
    }

    public static void runADA() throws IOException {
        System.out.println("Running ada_002");

        var baseVectors = "./fvec/ada-002/ada_002_100000_base_vectors.fvec";
        var queryVectors = "./fvec/ada-002/ada_002_100000_query_vectors_10000.fvec";
        testASHEncodings(baseVectors, queryVectors);
    }

    public static void runColbert() throws IOException {
        System.out.println("Running colbertv2");

        var baseVectors = "./fvec/wikipedia_squad/1M/colbertv2.0_128_base_vectors_1000000.fvec";
        var queryVectors = "./fvec/wikipedia_squad/1M/colbertv2.0_128_query_vectors_100000.fvec";
        testASHEncodings(baseVectors, queryVectors);
    }

    public static void runOpenai1536() throws IOException {
        System.out.println("Running text-embedding-3-large_1536");

        var baseVectors = "./fvec/openai-v3-large-1536-100k/text-embedding-3-large_1536_100000_base_vectors.fvec";
        var queryVectors = "./fvec/openai-v3-large-1536-100k/text-embedding-3-large_1536_100000_query_vectors_10000.fvec";
        testASHEncodings(baseVectors, queryVectors);
    }

    public static void runOpenai3072() throws IOException {
        System.out.println("Running text-embedding-3-large_3072");

        var baseVectors = "./fvec/openai-v3-large-3072-100k/text-embedding-3-large_3072_100000_base_vectors.fvec";
        var queryVectors = "./fvec/openai-v3-large-3072-100k/text-embedding-3-large_3072_100000_query_vectors_10000.fvec";
        testASHEncodings(baseVectors, queryVectors);
    }

    public static void runCap6m() throws IOException {
        System.out.println("Running cap-6m");

        var baseVectors = "./fvec/cap-6m/Caselaw_gte-Qwen2-1.5B_embeddings_base_6m_norm_shuffle.fvecs";
        var queryVectors = "./fvec/cap-6m/Caselaw_gte-Qwen2-1.5B_embeddings_query_10k_norm_shuffle.fvecs";
        testASHEncodings(baseVectors, queryVectors);
    }

    public static void runCohere10m() throws IOException {
        System.out.println("Running cohere-10m");

        var baseVectors = "./fvec/cohere-10m/cohere_wiki_en_flat_base_10m_norm.fvecs";
        var queryVectors = "./fvec/cohere-10m/cohere_wiki_en_flat_query_10k_norm.fvecs";
        testASHEncodings(baseVectors, queryVectors);
    }

    public static void main(String[] args) throws IOException {
//        runSIFT();
//        runGIST();
//        runColbert();
        runCohere100k();
        runADA();
        runOpenai1536();
        runOpenai3072();
//        runCap6m();
//        runCohere10m();
    }
}
