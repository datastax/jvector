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

import java.io.IOException;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

public class DistancesASH {

    static final String blockMode = System.getProperty("jvector.ash.blockKernel", "auto").toLowerCase();
    static final String singleMode = System.getProperty("jvector.ash.singleKernel", "auto").toLowerCase();

    private static void printScorerInfo(String label, Object scorer) {
        var backend =
                io.github.jbellis.jvector.vector.VectorizationProvider.getInstance()
                        .getVectorUtilSupport();

        System.out.println(
                "\t[" + label + "] scorer implementation = "
                        + (scorer instanceof ASHBlockScorer
                            ? ((ASHBlockScorer) scorer).description() : scorer.getClass().getName())
                        + " (singleKernel=" + singleMode
                        + ", blockKernel=" + blockMode
                        + ", supportsAshMaskedLoad=" + backend.supportsAshMaskedLoad()
                        + ", supportsAshLutScoring=" + backend.supportsAshLutScoring()
                        + ", supportsAshProjectionScoring=" + backend.supportsAshProjectionScoring()
                        + ", " + backend.ashKernelDescription()
                        + ")"
        );
    }

    private static void logProgress(String msg) {
        System.out.println(msg);
        System.out.flush();
    }

    /** Warm each measured path independently; recall only warms single-vector scoring. */
    private static void warmupASH(ASHVectors vectors, List<VectorFloat<?>> queries,
                                  ForkJoinPool executor, int blockSize) {
        int requested = Integer.parseInt(System.getProperty("jvector.bench.scoringWarmupQueries", "128"));
        if (requested < 0) throw new IllegalArgumentException("scoringWarmupQueries must be nonnegative");
        int queryCount = Math.min(requested, queries.size());
        if (queryCount == 0) return;
        int workers = Math.min(executor.getParallelism(), queryCount);
        List<ForkJoinTask<Double>> tasks = new ArrayList<>();
        for (int worker = 0; worker < workers; worker++) {
            final int first = worker;
            tasks.add(executor.submit(() -> {
                double checksum = 0;
                float[] scores = blockSize == 0 ? null : new float[blockSize];
                for (int q = first; q < queryCount; q += workers) {
                    if (blockSize == 0) {
                        var scorer = vectors.scoreFunctionFor(queries.get(q), VectorSimilarityFunction.DOT_PRODUCT);
                        for (int i = 0; i < vectors.count(); i++) checksum += scorer.similarityTo(i);
                    } else {
                        var scorer = vectors.blockScorerFor(queries.get(q), VectorSimilarityFunction.DOT_PRODUCT, blockSize);
                        for (int start = 0; start < vectors.count(); start += blockSize) {
                            int count = Math.min(blockSize, vectors.count() - start);
                            scorer.scoreRange(start, count, scores);
                            for (int i = 0; i < count; i++) checksum += scores[i];
                        }
                    }
                }
                return checksum;
            }));
        }
        double checksum = 0;
        for (var task : tasks) checksum += task.join();
        System.out.printf(java.util.Locale.ROOT, "\t%s warmup: %d queries, checksum=%.6f (excluded from timing)%n",
                blockSize == 0 ? "Single" : "Block", queryCount, checksum);
    }

    private static int parseOptimizerFromProperty() {
        String opt = System.getProperty("jvector.ash.optimizer", "random").trim().toLowerCase();

        if ("random".equals(opt)) return AsymmetricHashing.RANDOM;
        if ("itq".equals(opt)) return AsymmetricHashing.ITQ;

        throw new IllegalArgumentException(
                "Unknown jvector.ash.optimizer=" + opt + " (expected random|itq)"
        );
    }

    private static String optimizerName(int optimizer) {
        if (optimizer == AsymmetricHashing.RANDOM) return "RANDOM";
        if (optimizer == AsymmetricHashing.ITQ) return "ITQ";
        return "UNKNOWN(" + optimizer + ")";
    }

    public static void testASHEncodings(String filenameBase, String filenameQueries, String filenameGT) throws IOException {
        // ------------------------------------------------------------
        // Benchmark configuration (runtime flags)
        // ------------------------------------------------------------
        final boolean RUN_SANITY_CHECK =
                Boolean.parseBoolean(System.getProperty("jvector.bench.sanity-check", "false"));

        final boolean RUN_RECALL_CHECK =
                Boolean.parseBoolean(System.getProperty("jvector.bench.recall", "false"));

        final int RECALL_K =
                Integer.getInteger("jvector.bench.recall.k", 10);

        final boolean RUN_ACCURACY_CHECK =
                Boolean.parseBoolean(System.getProperty("jvector.bench.accuracy", "false"));

        final boolean RUN_SINGLE_SCORING = Boolean.parseBoolean(System.getProperty(
                "jvector.bench.single-scoring", System.getProperty("jvector.bench.scalar-scoring", "true")));

        final boolean RUN_FLOAT_SCORING =
                Boolean.parseBoolean(System.getProperty("jvector.bench.float-scoring", "false"));

        // ASH header bits
        final int HEADER_BITS = AsymmetricHashing.HEADER_BITS;
        System.out.println("\tASH header: " + HEADER_BITS + " bits (scale=16, offset=16, landmark=8)");

        // Block sizes to benchmark
        final int[] BLOCK_SIZES = {32}; // Supported fused/multi-bit block capacities: 8, 16, 32.

        // How many ASH landmarks to use, C = [1, 64]
        final int landmarkCount = 1;

        List<VectorFloat<?>> vectors = SiftLoader.readFvecs(filenameBase);
        List<VectorFloat<?>> queries = SiftLoader.readFvecs(filenameQueries);
        final List<List<Integer>> groundTruth;
        if (RUN_RECALL_CHECK) {
            if (filenameGT == null || filenameGT.isBlank()) {
                throw new IllegalArgumentException(
                        "Recall requires a ground-truth filename");
            }
            groundTruth = SiftLoader.readIvecs(filenameGT);
        } else {
            groundTruth = List.of();
        }

        // ASH normalization policy:
        //
        // - Base vectors x are NOT globally normalized.
        // - For encoding, we compute μ on raw x.
        // - The residual (x − μ) is normalized ONCE inside the binarizer,
        //   producing \hat{x} = (x − μ) / ||x − μ|| (ASH Paper, Eq. 6).
        // - Queries may be L2-normalized in the benchmark (standard practice),
        //   but are NOT normalized inside the encoder.
        // - No other normalization steps are applied.

        int dimension = vectors.get(0).length();

        final int bitsPerDimension =
                Integer.getInteger("jvector.ash.bitsPerDimension", 4);

        final int quantizedDimensions =
                Integer.getInteger("jvector.ash.quantizedDimensions", dimension / 2);

        if (bitsPerDimension != 1
                && bitsPerDimension != 2
                && bitsPerDimension != 4) {
            throw new IllegalArgumentException(
                    "bitsPerDimension must be 1, 2, or 4: " + bitsPerDimension);
        }

        if (quantizedDimensions <= 0 || quantizedDimensions > dimension) {
            throw new IllegalArgumentException(
                    "Invalid quantizedDimensions=" + quantizedDimensions
                            + " for original dimension=" + dimension);
        }

        int payloadBits = Math.multiplyExact(quantizedDimensions, bitsPerDimension);
        int encodedBits = Math.addExact(HEADER_BITS, payloadBits);

        System.out.println(
                "\toriginalDim=" + dimension
                        + ", quantizedDim=" + quantizedDimensions
                        + ", bitsPerDimension=" + bitsPerDimension
                        + ", payloadBits=" + payloadBits
                        + ", encodedBits=" + encodedBits);

        final List<VectorFloat<?>> finalQueries = queries;
        final List<VectorFloat<?>> finalVectors = vectors;

        System.out.format("\t%d base and %d query vectors loaded, dimension=%d%n",
                vectors.size(), queries.size(), dimension);

        // ------------------------------------------------------------------
        // Build ASH (centroids + optional training), then encode
        // ------------------------------------------------------------------
        var ravv = new ListRandomAccessVectorValues(vectors, dimension);

        // Choose optimizer at runtime:
        //   -Djvector.ash.optimizer=random|itq
        final int optimizer = parseOptimizerFromProperty();

        logProgress("\tASH optimizer = " + optimizerName(optimizer));

        logProgress("\t[stage] ASH initialize: starting (centroids + training)");
        long initStart = System.nanoTime();
        var ash = AsymmetricHashing.initialize(
                ravv, optimizer, encodedBits, landmarkCount, bitsPerDimension);
        long initEnd = System.nanoTime();
        logProgress("\t[stage] ASH initialize: done in " + (initEnd - initStart) / 1e9 + " seconds");

        logProgress("\t[stage] ASH encodeAll: starting");
        long startTime = System.nanoTime();
        CompressedVectors ashVecs = ash.encodeAll(ravv);
        long endTime = System.nanoTime();
        logProgress("\t[stage] ASH encodeAll: done in " + (endTime - startTime) / 1e9 + " seconds");


        double encSeconds = (endTime - startTime) / 1e9;
        double encThroughput = vectors.size() / encSeconds;

        System.out.println(
                "\tEncoding throughput = "
                        + String.format(java.util.Locale.ROOT, "%.3f", encThroughput)
                        + " vectors/sec"
        );

        ASHVectors ashVectors = (ASHVectors) ashVecs;
        VectorFloat<?> q0 = finalQueries.get(0);

        // ------------------------------------------------------------------
        // Landmark reorder (debug + timing)
        // ------------------------------------------------------------------
        final boolean REORDER_BY_LANDMARK =
                Boolean.parseBoolean(System.getProperty("jvector.ash.reorderByLandmark", "true"));

        final ASHVectors ashVectorsFinal;
        final CompressedVectors ashVecsFinal;
        final int[] newToOldFinal;

        if (REORDER_BY_LANDMARK) {
            long stat0 = System.nanoTime();
            var beforeStats = ashVectors.landmarkRunStats();
            long stat1 = System.nanoTime();

            System.out.println("\tLandmark run stats BEFORE reorder: " + beforeStats
                    + " (computed in " + (stat1 - stat0) / 1e9 + " s)");

            long t0 = System.nanoTime();
            ASHVectors.LandmarkOrder order = ashVectors.reorderByLandmarkFast();
            long t1 = System.nanoTime();

            ASHVectors reordered = order.vectors;

            long stat2 = System.nanoTime();
            var afterStats = reordered.landmarkRunStats();
            long stat3 = System.nanoTime();

            System.out.println("\tLandmark reorder took " + (t1 - t0) / 1e9 + " s");
            System.out.println("\tLandmark run stats AFTER  reorder: " + afterStats
                    + " (computed in " + (stat3 - stat2) / 1e9 + " s)");
            System.out.println("--");

            ashVectorsFinal = reordered;
            ashVecsFinal = reordered;         // CompressedVectors view used by scoreFunctionFor
            newToOldFinal = order.newToOld;   // map reordered ordinal -> original base ordinal
        } else {
            System.out.println("\tLandmark reorder disabled (-Djvector.ash.reorderByLandmark=false)");
            ashVectorsFinal = ashVectors;
            ashVecsFinal = ashVectors;
            newToOldFinal = null;
        }

        {
            ScoreFunction.ApproximateScoreFunction single =
                    ashVecsFinal.scoreFunctionFor(q0, VectorSimilarityFunction.DOT_PRODUCT);
            printScorerInfo("single", single);
        }

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
        // Default to one scoring thread for kernel/accumulator comparisons.
        // For aggregate throughput, set -Djvector.bench.scoringThreads=N,
        // where N is the number of physical cores available to this process.
        // This controls accuracy, recall, and scoring tasks, not encodeAll.
        final int scoringThreads =
                Integer.parseInt(
                        System.getProperty("jvector.bench.scoringThreads", "1").trim());

        if (scoringThreads <= 0) {
            throw new IllegalArgumentException(
                    "scoringThreads must be positive: " + scoringThreads);
        }

        final ForkJoinPool simdExecutor = new ForkJoinPool(scoringThreads);

        int parallelism = simdExecutor.getParallelism();
        int chunkSize = Math.max(1, (queries.size() + parallelism - 1) / parallelism);

        System.out.println("\tScoring parallelism = " + parallelism);

        // ==================================================================
        // [1] Accuracy run (NOT timed)
        // ==================================================================
        if (RUN_ACCURACY_CHECK) {
            List<ForkJoinTask<double[]>> errorTasks = new ArrayList<>();

            for (int start = 0; start < queries.size(); start += chunkSize) {
                final int s = start;
                final int e = Math.min(start + chunkSize, queries.size());

                errorTasks.add(simdExecutor.submit(() -> {
                    double localError = 0.0;
                    long localCount = 0;

                    for (int i = s; i < e; i++) {
                        VectorFloat<?> q = finalQueries.get(i);
                        ScoreFunction.ApproximateScoreFunction f =
                                ashVecsFinal.scoreFunctionFor(q, VectorSimilarityFunction.DOT_PRODUCT);

                        for (int j = 0; j < vectors.size(); j++) {
                            final int baseOrd = (newToOldFinal == null) ? j : newToOldFinal[j];
                            VectorFloat<?> v = finalVectors.get(baseOrd);

                            float trueSimilarity = VectorSimilarityFunction.DOT_PRODUCT.compare(q, v);
                            float approxSimilarity = f.similarityTo(j);

                            localError += Math.abs(approxSimilarity - trueSimilarity);
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
            System.out.println("\tAverage absolute DOT_PRODUCT similarity error = " + distanceError);
        }

        // ==================================================================
        // [1b] Recall@K run (Parallelized)
        // ==================================================================
        if (RUN_RECALL_CHECK) {
            int[] atValues = {10, 15, 20, 30, 40, 50};
            int maxAt = 50;
            List<ForkJoinTask<double[]>> recallTasks = new ArrayList<>();

            logProgress("\t[stage] Computing " + RECALL_K + "-Recall@K...");
            for (int start = 0; start < queries.size(); start += chunkSize) {
                final int s = start;
                final int e = Math.min(start + chunkSize, queries.size());

                recallTasks.add(simdExecutor.submit(() -> {
                    double[] localTotalRecall = new double[atValues.length];
                    for (int i = s; i < e; i++) {
                        VectorFloat<?> q = finalQueries.get(i);
                        ScoreFunction.ApproximateScoreFunction f = ashVecsFinal.scoreFunctionFor(q, VectorSimilarityFunction.DOT_PRODUCT);

                        // Min-heap to keep top 50 results (storing score as int bits and ordinal as long)
                        var topCandidates = new java.util.PriorityQueue<long[]>((a, b) -> Float.compare(Float.intBitsToFloat((int) a[0]), Float.intBitsToFloat((int) b[0])));
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
                        for (int rank = topCandidates.size() - 1; rank >= 0; rank--) topIndices[rank] = (int) topCandidates.poll()[1];

                        int[] queryGT = groundTruth.get(i).stream()
                                .mapToInt(Integer::intValue)
                                .toArray();
                        for (int aIdx = 0; aIdx < atValues.length; aIdx++) {
                            int at = atValues[aIdx];
                            java.util.Set<Integer> topAtSet = new java.util.HashSet<>();
                            for (int r = 0; r < Math.min(at, topIndices.length); r++) {
                                topAtSet.add((newToOldFinal == null) ? topIndices[r] : newToOldFinal[topIndices[r]]);
                            }
                            int matches = 0;
                            java.util.HashSet<Integer> gtSeen = new java.util.HashSet<>(RECALL_K * 2);
                            // Compute and don't count duplicates (if present in GT or retrieved IDs)
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
                for (int aIdx = 0; aIdx < atValues.length; aIdx++) totalRecall[aIdx] += local[aIdx];
            }

            for (int aIdx = 0; aIdx < atValues.length; aIdx++) {
                System.out.format("\tASH %d-recall@%d = %.4f%n", RECALL_K, atValues[aIdx], totalRecall[aIdx] / queries.size());
            }
        }

        // ==================================================================
        // [2] Single-vector ASH scoring, using the selected singleKernel
        // ==================================================================
        if (RUN_SINGLE_SCORING) {
            warmupASH(ashVectorsFinal, finalQueries, simdExecutor, 0);
            List<ForkJoinTask<Double>> ashTasks = new ArrayList<>();

            long ashStart = System.nanoTime();

            for (int start = 0; start < queries.size(); start += chunkSize) {
                final int s = start;
                final int e = Math.min(start + chunkSize, queries.size());

                ashTasks.add(simdExecutor.submit(() -> {
                    double localSum = 0.0;

                    for (int i = s; i < e; i++) {
                        VectorFloat<?> q = finalQueries.get(i);
                        ScoreFunction.ApproximateScoreFunction f =
                                ashVecsFinal.scoreFunctionFor(q, VectorSimilarityFunction.DOT_PRODUCT);

                        for (int j = 0; j < vectors.size(); j++) {
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

            double singleSeconds = (ashEnd - ashStart) / 1e9;
            long totalDotProducts = (long) queries.size() * (long) vectors.size();
            double singleThroughput = totalDotProducts / singleSeconds;

            System.out.println("\tSingle " + singleMode + " scoring took "
                    + singleSeconds + " seconds");

            System.out.println(
                    "\tSingle " + singleMode + " throughput = "
                            + String.format(java.util.Locale.ROOT, "%.3f", singleThroughput)
                            + " dot-products/sec"
                            + " ("
                            + String.format(java.util.Locale.ROOT, "%.3f", singleThroughput / 1e6)
                            + " Mdot/s)"
            );

            System.out.println("\tdummyAccumulator = " + (float) (ashDummy));
            System.out.println("--");

        }

        // ==================================================================
        // [3] Float dot-product scoring (ground truth baseline)
        // ==================================================================
        if (RUN_FLOAT_SCORING) {
            List<ForkJoinTask<Double>> floatTasks = new ArrayList<>();

            long floatStart = System.nanoTime();

            for (int start = 0; start < queries.size(); start += chunkSize) {
                final int s = start;
                final int e = Math.min(start + chunkSize, queries.size());

                floatTasks.add(simdExecutor.submit(() -> {
                    double localSum = 0.0;

                    for (int i = s; i < e; i++) {
                        VectorFloat<?> q = finalQueries.get(i);
                        for (int j = 0; j < vectors.size(); j++) {
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
                        ashVectorsFinal.blockScorerFor(
                                finalQueries.get(0),
                                VectorSimilarityFunction.DOT_PRODUCT,
                                blockSize
                        );

                printScorerInfo(kernelMode + " (blockSize=" + blockSize + ")", scorer);
            }

            warmupASH(ashVectorsFinal, finalQueries, simdExecutor, blockSize);
            List<ForkJoinTask<Double>> blockTasks = new ArrayList<>();
            long blockStart = System.nanoTime();

            for (int start = 0; start < queries.size(); start += chunkSize) {
                final int s = start;
                final int e = Math.min(start + chunkSize, queries.size());

                blockTasks.add(simdExecutor.submit(() -> {
                    double localSum = 0.0;
                    final float[] scores = new float[blockSize];

                    for (int qi = s; qi < e; qi++) {
                        ASHBlockScorer scorer =
                                ashVectorsFinal.blockScorerFor(
                                        finalQueries.get(qi),
                                        VectorSimilarityFunction.DOT_PRODUCT,
                                        blockSize
                                );

                        int j = 0;
                        while (j + blockSize <= vectors.size()) {
                            scorer.scoreRange(j, blockSize, scores);
                            for (int k = 0; k < blockSize; k++) {
                                localSum += scores[k];
                            }
                            j += blockSize;
                        }

                        // Exercise the selected block kernel for partial batches as well.
                        if (j < vectors.size()) {
                            int count = vectors.size() - j;
                            scorer.scoreRange(j, count, scores);
                            for (int k = 0; k < count; k++) localSum += scores[k];
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
            long totalDotProducts = (long) queries.size() * (long) vectors.size();

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
        var gtVectors = "";
        testASHEncodings(baseVectors, queryVectors, gtVectors);
    }

    public static void runGIST() throws IOException {
        System.out.println("Running GIST");

        var baseVectors = "./fvec/gist/gist_base.fvecs";
        var queryVectors = "./fvec/gist/gist_query.fvecs";
        var gtVectors = "";
        testASHEncodings(baseVectors, queryVectors, gtVectors);
    }

    public static void runCohere100k() throws IOException {
        System.out.println("Running Cohere-100k");

        var baseVectors = "/home/ted_willke/datasets-clean/cohere_english_v3_100k_base_99685.fvecs";
        var queryVectors = "/home/ted_willke/datasets-clean/cohere_english_v3_100k_query_10000.fvecs";
        var gtVectors = "/home/ted_willke/datasets-clean/cohere_english_v3_100k_gt_ip_100.ivecs";
        testASHEncodings(baseVectors, queryVectors, gtVectors);
    }

    public static void runADA() throws IOException {
        System.out.println("Running ada-002");

        var baseVectors = "./fvec/ada-002/ada_002_100000_base_vectors.fvec";
        var queryVectors = "./fvec/ada-002/ada_002_100000_query_vectors_10000.fvec";
        var gtVectors = "./fvec/ada-002/ada_002_100000_indices_query_10000.ivec";
        testASHEncodings(baseVectors, queryVectors, gtVectors);
    }

    public static void runADANoZeros() throws IOException {
        System.out.println("Running ada-002-no-zeros");

        var baseVectors = "./fvec/ada-002-no-zeros/ada_002_100000_base_vectors_no_zeros.fvec";
        var queryVectors = "./fvec/ada-002-no-zeros/ada_002_100000_query_vectors_10000_no_zeros.fvec";
        var gtVectors = "./fvec/ada-002-no-zeros/ada-002_gt_no_zeros.ivec";
        testASHEncodings(baseVectors, queryVectors, gtVectors);
    }

    public static void runColbert() throws IOException {
        System.out.println("Running colbertv2");

        var baseVectors = "./fvec/wikipedia_squad/1M/colbertv2.0_128_base_vectors_1000000.fvec";
        var queryVectors = "./fvec/wikipedia_squad/1M/colbertv2.0_128_query_vectors_100000.fvec";
        var gtVectors = "";
        testASHEncodings(baseVectors, queryVectors, gtVectors);
    }

    public static void runOpenai1536() throws IOException {
        System.out.println("Running text-embedding-3-large_1536");

        var baseVectors = "./fvec/openai-v3-large-1536-100k/openai_v3_large_1536_100k_base_98716.fvecs";
        var queryVectors = "./fvec/openai-v3-large-1536-100k/openai_v3_large_1536_100k_query_10000.fvecs";
        var gtVectors = "./fvec/openai-v3-large-1536-100k/openai_v3_large_1536_100k_gt_ip_100.ivecs";
        testASHEncodings(baseVectors, queryVectors, gtVectors);
    }

    public static void runOpenai3072() throws IOException {
        System.out.println("Running text-embedding-3-large_3072");

        var baseVectors = "./fvec/openai-v3-large-3072-100k/text-embedding-3-large_3072_100000_base_vectors.fvec";
        var queryVectors = "./fvec/openai-v3-large-3072-100k/text-embedding-3-large_3072_100000_query_vectors_10000.fvec";
        var gtVectors = "./fvec/openai-v3-large-3072-100k/text-embedding-3-large_3072_100000_indices_query_10000.ivec";
        testASHEncodings(baseVectors, queryVectors, gtVectors);
    }

    public static void runCap6m() throws IOException {
        System.out.println("Running cap-6m");

        var baseVectors = "./fvec/cap-6m/Caselaw_gte-Qwen2-1.5B_embeddings_base_6m_norm_shuffle.fvecs";
        var queryVectors = "./fvec/cap-6m/Caselaw_gte-Qwen2-1.5B_embeddings_query_10k_norm_shuffle.fvecs";
        var gtVectors = "./fvec/cap-6m/cap_6m_gt_norm_shuffle_ip_k100.ivecs";
        testASHEncodings(baseVectors, queryVectors, gtVectors);
    }

    public static void runCohere10m() throws IOException {
        System.out.println("Running cohere-10m");

        var baseVectors = "./fvec/cohere-10m/cohere_wiki_en_flat_base_10m_norm.fvecs";
        var queryVectors = "./fvec/cohere-10m/cohere_wiki_en_flat_query_10k_norm.fvecs";
        var gtVectors = "./fvec/cohere-10m/cohere_wiki_en_flat_gt_10m_ip_k100.ivecs";
        testASHEncodings(baseVectors, queryVectors, gtVectors);
    }

    public static void main(String[] args) throws IOException {
//        runSIFT();
//        runGIST();
//        runColbert();
//        runCohere100k();
//        runADA();
//        runADANoZeros();
        runOpenai1536();
//        runOpenai3072();
//        runCap6m();
//        runCohere10m();
    }
}
