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
import io.github.jbellis.jvector.quantization.CompressedVectors;

import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;
import java.util.ArrayList;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;



import static java.lang.Math.abs;

public class DistancesASH {
    public static void testASHEncodings(String filenameBase, String filenameQueries) throws IOException {

        List<VectorFloat<?>> vectors = SiftLoader.readFvecs(filenameBase);
        List<VectorFloat<?>> queries = SiftLoader.readFvecs(filenameQueries);

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

        // ASH normalization policy:
        //
        // - Base vectors x are NOT globally normalized.
        // - For encoding, we compute μ on raw x.
        // - The residual (x − μ) is normalized ONCE inside the binarizer,
        //   producing \hat{x} = (x − μ) / ||x − μ|| (Eq. 6).
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

        int nQueries = Math.min(1000, queries.size());
        int nVectors = Math.min(100_000, vectors.size());

        vectors = vectors.subList(0, nVectors);
        queries = queries.subList(0, nQueries);

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

        // ------------------------------------------------------------------
        // Shared parallelization setup (executor-honoring)
        // ------------------------------------------------------------------
        ForkJoinPool simdExecutor = PhysicalCoreExecutor.pool();

        int parallelism = simdExecutor.getParallelism();
        int chunkSize = Math.max(1, (nQueries + parallelism - 1) / parallelism);

        // ==================================================================
        // [1] Accuracy run (NOT timed)
        // ==================================================================
        List<ForkJoinTask<double[]>> errorTasks = new ArrayList<>();

        for (int start = 0; start < nQueries; start += chunkSize) {
            final int s = start;
            final int e = Math.min(start + chunkSize, nQueries);

            errorTasks.add(simdExecutor.submit(() -> {
                double localError = 0.0;
                long localCount = 0;

                for (int i = s; i < e; i++) {
                    VectorFloat<?> q = finalQueries.get(i);
                    ScoreFunction.ApproximateScoreFunction f =
                            ashVecs.scoreFunctionFor(q, VectorSimilarityFunction.DOT_PRODUCT);

                    for (int j = 0; j < nVectors; j++) {
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

        // ==================================================================
        // [2] ASH distance timing (Eq. 11 only)
        // ==================================================================
        List<ForkJoinTask<Double>> ashTasks = new ArrayList<>();

        long ashStart = System.nanoTime();

        for (int start = 0; start < nQueries; start += chunkSize) {
            final int s = start;
            final int e = Math.min(start + chunkSize, nQueries);

            ashTasks.add(simdExecutor.submit(() -> {
                double localSum = 0.0;

                for (int i = s; i < e; i++) {
                    VectorFloat<?> q = finalQueries.get(i);
                    ScoreFunction.ApproximateScoreFunction f =
                            ashVecs.scoreFunctionFor(q, VectorSimilarityFunction.DOT_PRODUCT);

                    for (int j = 0; j < nVectors; j++) {
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

        System.out.println("\tASH dot-product computations took "
                + (ashEnd - ashStart) / 1e9 + " seconds");

        // ==================================================================
        // [3] Float dot-product timing (ground truth baseline)
        // ==================================================================
        List<ForkJoinTask<Double>> floatTasks = new ArrayList<>();

        long floatStart = System.nanoTime();

        for (int start = 0; start < nQueries; start += chunkSize) {
            final int s = start;
            final int e = Math.min(start + chunkSize, nQueries);

            floatTasks.add(simdExecutor.submit(() -> {
                double localSum = 0.0;

                for (int i = s; i < e; i++) {
                    VectorFloat<?> q = finalQueries.get(i);
                    for (int j = 0; j < nVectors; j++) {
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
        System.out.println("\tdummyAccumulator = " + (float) (ashDummy + floatDummy));
        System.out.println("--");
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

        var baseVectors = "./fvec/wikipedia_squad/100k/ada_002_100000_base_vectors.fvec";
        var queryVectors = "./fvec/wikipedia_squad/100k/ada_002_100000_query_vectors_10000.fvec";
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
        var queryVectors = "./fvec/openai-v3-large-3072-100k/text-embedding-3-large_3072_100000_base_vectors.fvec";
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
    }
}