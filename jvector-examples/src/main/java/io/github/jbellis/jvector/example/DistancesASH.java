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

    private static void validateRecallScorers(ASHVectors vectors, List<VectorFloat<?>> queries, int blockSize) {
        float maxBlockVsScalar = 0f;
        float maxBlockVsSingle = 0f;
        for (int qi : new int[]{0, queries.size() / 2}) {
            VectorFloat<?> query = queries.get(qi);
            ASHBlockScorer block = vectors.blockScorerFor(query, VectorSimilarityFunction.DOT_PRODUCT, blockSize);
            ASHBlockScorer scalar = vectors.blockScorerFor(query, VectorSimilarityFunction.DOT_PRODUCT);
            ScoreFunction.ApproximateScoreFunction single =
                    vectors.scoreFunctionFor(query, VectorSimilarityFunction.DOT_PRODUCT);
            printScorerInfo("recall-block", block);
            float[] blockScores = new float[blockSize];
            float[] scalarScores = new float[blockSize];
            for (int start = 0; start < vectors.count(); start += blockSize) {
                int count = Math.min(blockSize, vectors.count() - start);
                block.scoreRange(start, count, blockScores);
                scalar.scoreRange(start, count, scalarScores);
                for (int lane = 0; lane < count; lane++) {
                    float actual = blockScores[lane];
                    float reference = scalarScores[lane];
                    float singleScore = single.similarityTo(start + lane);
                    if (!Float.isFinite(actual) || !Float.isFinite(reference) || !Float.isFinite(singleScore)) {
                        throw new AssertionError("Non-finite ASH score at query=" + qi + " ordinal=" + (start + lane));
                    }
                    maxBlockVsScalar = Math.max(maxBlockVsScalar, Math.abs(actual - reference));
                    maxBlockVsSingle = Math.max(maxBlockVsSingle, Math.abs(actual - singleScore));
                }
            }
        }
        System.out.printf(java.util.Locale.ROOT,
                "\tRecall scorer validation: max block/scalar=%.8f, max block/single=%.8f%n",
                maxBlockVsScalar, maxBlockVsSingle);
        if (maxBlockVsScalar > 0.005f || maxBlockVsSingle > 0.005f) {
            throw new AssertionError("ASH block recall scorer disagrees with scalar or single scoring");
        }
    }

    private static void offerTop(java.util.PriorityQueue<long[]> candidates, int capacity,
                                 float score, int ordinal) {
        if (candidates.size() < capacity) {
            candidates.add(new long[]{Float.floatToRawIntBits(score), ordinal});
        } else if (score > Float.intBitsToFloat((int) candidates.peek()[0])) {
            candidates.poll();
            candidates.add(new long[]{Float.floatToRawIntBits(score), ordinal});
        }
    }

    private static int[] rankedOrdinals(java.util.PriorityQueue<long[]> candidates) {
        int[] ranked = new int[candidates.size()];
        for (int rank = ranked.length - 1; rank >= 0; rank--) ranked[rank] = (int) candidates.poll()[1];
        return ranked;
    }

    private static void addRecall(int[] ranked, int[] groundTruth, int[] newToOld,
                                  int recallK, int[] atValues, double[] totals, int offset) {
        for (int aIdx = 0; aIdx < atValues.length; aIdx++) {
            int at = atValues[aIdx];
            java.util.Set<Integer> topAtSet = new java.util.HashSet<>();
            for (int rank = 0; rank < Math.min(at, ranked.length); rank++) {
                topAtSet.add(newToOld == null ? ranked[rank] : newToOld[ranked[rank]]);
            }
            int matches = 0;
            java.util.HashSet<Integer> gtSeen = new java.util.HashSet<>(recallK * 2);
            for (int g = 0; g < recallK; g++) {
                int gtId = groundTruth[g];
                if (gtSeen.add(gtId) && topAtSet.contains(gtId)) matches++;
            }
            totals[offset + aIdx] += (double) matches / recallK;
        }
    }

    /** Independent component decoder for validation; never used in timed scoring. */
    private static double symmetricComponent(AsymmetricHashing.QuantizedVector v, int j, int bits) {
        if (bits == 1) return ((v.binaryVector[j / 64] >>> (j % 64)) & 1L) == 0 ? -1 : 1;
        if (bits != 2 && bits != 4) {
            int code = 0;
            for (int k = 0; k < bits - 1; k++) {
                int pos = j * (bits - 1) + k;
                code |= ((v.extraBits[pos / 8] >>> (pos % 8)) & 1) << k;
            }
            code |= ((v.binaryVector[j / 64] >>> (j % 64)) & 1L) << (bits - 1);
            return code - ((1 << (bits - 1)) - 0.5);
        }
        int field = (v.extraBits[j * bits / 8] >>> (j * bits % 8)) & ((1 << bits) - 1);
        double magnitude = (field & ((1 << (bits - 1)) - 1)) + 0.5;
        return (field & (1 << (bits - 1))) == 0 ? -magnitude : magnitude;
    }

    private static double decodedSymmetric(AsymmetricHashing ash,
                                           AsymmetricHashing.QuantizedVector a,
                                           AsymmetricHashing.QuantizedVector b) {
        int ca = a.landmark & 255, cb = b.landmark & 255;
        double dot = 0, ownA = 0, ownB = 0, crossA = 0, crossB = 0;
        for (int j = 0; j < ash.quantizedDim; j++) {
            double x = symmetricComponent(a, j, ash.bitsPerDimension);
            double y = symmetricComponent(b, j, ash.bitsPerDimension);
            dot += x * y;
            ownA += x * ash.landmarkProj[ca][j];
            ownB += y * ash.landmarkProj[cb][j];
            crossA += x * ash.landmarkProj[cb][j];
            crossB += y * ash.landmarkProj[ca][j];
        }
        double correctionA = a.offset, correctionB = b.offset;
        if (ash.bitsPerDimension != 2 && ash.bitsPerDimension != 4) {
            correctionA -= a.scale * ownA; correctionB -= b.scale * ownB;
        }
        double centroidDot = 0;
        for (int j = 0; j < ash.originalDimension; j++) {
            centroidDot += (double) ash.landmarks[ca].get(j) * ash.landmarks[cb].get(j);
        }
        return Math.max(0, (1 + (double) a.scale * b.scale * dot + a.scale * crossA
                + b.scale * crossB + correctionA + correctionB + centroidDot) / 2);
    }

    private static final class ScanScorer {
        final ScoreFunction single;
        final ASHBlockScorer block;
        ScanScorer(ScoreFunction single) { this.single = single; this.block = null; }
        ScanScorer(ASHBlockScorer block) { this.single = null; this.block = block; }
        void fill(int start, int count, float[] out) {
            if (block != null) block.scoreRange(start,count,out);
            else for (int i = 0; i < count; i++) out[i] = single.similarityTo(start+i);
        }
        double scan(int count, float[] scratch) {
            double sum = 0;
            if (single != null) {
                for (int node = 0; node < count; node++) sum += single.similarityTo(node);
            } else {
                for (int start = 0; start < count; start += scratch.length) {
                    int n = Math.min(scratch.length, count-start);
                    block.scoreRange(start,n,scratch);
                    for (int i = 0; i < n; i++) sum += scratch[i];
                }
            }
            return sum;
        }
    }

    /** Only query encoding is excluded. Every timed query creates its actual scoring state. */
    private static void benchmarkSymmetric(ASHVectors vectors, List<VectorFloat<?>> queries,
                                            List<List<Integer>> groundTruth, int[] newToOld,
                                            int recallK, boolean recall, ForkJoinPool executor) {
        var ash = vectors.getCompressor();
        int passes = Integer.getInteger("jvector.bench.scoringPasses", 5);
        int timingQueries = Math.min(queries.size(), Integer.getInteger("jvector.bench.scoringQueries", 128));
        int warmupQueries = Math.min(timingQueries, Integer.getInteger("jvector.bench.scoringWarmupQueries", 32));
        int timingVectors = Math.min(vectors.count(), Integer.getInteger("jvector.bench.scoringVectors", vectors.count()));
        if (passes < 1 || timingQueries < 1 || warmupQueries < 0 || timingVectors < 1) {
            throw new IllegalArgumentException("Invalid scoring pass/query/vector counts");
        }
        long start = System.nanoTime();
        var encodedQueries = new AsymmetricHashing.QuantizedVector[queries.size()];
        for (int q = 0; q < queries.size(); q++) encodedQueries[q] = ash.encode(queries.get(q));
        System.out.printf(java.util.Locale.ROOT, "QUERY_ENCODING seconds=%.6f queries=%d (excluded)%n",
                (System.nanoTime()-start)/1e9, queries.size());
        var scalar = new io.github.jbellis.jvector.quantization.ASHSymmetricScorer(vectors,
                io.github.jbellis.jvector.quantization.ASHSymmetricScorer.Kernel.SCALAR);
        var backend = io.github.jbellis.jvector.vector.VectorizationProvider.getInstance().getVectorUtilSupport();
        boolean hasSimd = (ash.bitsPerDimension == 1 || ash.bitsPerDimension == 2 || ash.bitsPerDimension == 4)
                && backend.supportsAshSymmetricScoring()
                && (ash.landmarkCount == 1 || (ash.bitsPerDimension == 1
                    ? backend.supportsAshMaskedLoad() : backend.supportsAshProjectionScoring()));
        boolean hasBlockSimd = ash.bitsPerDimension == 1 ? backend.supportsAshMaskedLoad()
                : (ash.bitsPerDimension == 2 || ash.bitsPerDimension == 4) && backend.supportsAshLutScoring();
        List<String> names = new ArrayList<>();
        List<java.util.function.IntFunction<ScanScorer>> factories = new ArrayList<>();
        names.add("symmetric-single-scalar");
        factories.add(q -> new ScanScorer(scalar.scoreFunctionFor(encodedQueries[q])));
        if (hasSimd) {
            var simd = new io.github.jbellis.jvector.quantization.ASHSymmetricScorer(vectors,
                    io.github.jbellis.jvector.quantization.ASHSymmetricScorer.Kernel.SIMD);
            names.add("symmetric-single-simd");
            factories.add(q -> new ScanScorer(simd.scoreFunctionFor(encodedQueries[q])));
        }
        names.add("symmetric-block-scalar");
        factories.add(q -> new ScanScorer(scalar.blockScorerFor(encodedQueries[q],32,
                io.github.jbellis.jvector.quantization.ASHSymmetricScorer.Kernel.SCALAR)));
        if (hasBlockSimd) {
            names.add("symmetric-block-simd");
            factories.add(q -> new ScanScorer(scalar.blockScorerFor(encodedQueries[q],32,
                    io.github.jbellis.jvector.quantization.ASHSymmetricScorer.Kernel.SIMD)));
        }
        names.add("asymmetric-single-" + singleMode);
        factories.add(q -> new ScanScorer(vectors.scoreFunctionFor(queries.get(q),VectorSimilarityFunction.DOT_PRODUCT)));
        names.add("asymmetric-block-" + blockMode);
        factories.add(q -> new ScanScorer(vectors.blockScorerFor(queries.get(q),VectorSimilarityFunction.DOT_PRODUCT,32)));

        String requestedModes = System.getProperty("jvector.bench.scoringModes", "").trim();
        if (!requestedModes.isEmpty()) {
            var selectedModes = new java.util.HashSet<>(java.util.Arrays.asList(requestedModes.split(",")));
            for (String name : selectedModes) if (!names.contains(name)) throw new IllegalArgumentException("Unavailable scoring mode: " + name);
            for (int i = names.size()-1; i >= 0; i--) if (!selectedModes.contains(names.get(i))) {
                names.remove(i); factories.remove(i);
            }
        }
        // Validate selected symmetric modes against an independent decoded/header formula.
        for (int mode = 0; mode < names.size(); mode++) {
            if (!names.get(mode).startsWith("symmetric-")) continue;
            double maxError = 0;
            for (int q = 0; q < Math.min(4,queries.size()); q++) {
                var scorer = factories.get(mode).apply(q);
                float[] one = new float[1];
                int samples = Math.min(512,vectors.count());
                for (int sample = 0; sample < samples; sample++) {
                    int node = samples == 1 ? 0 : (int) ((long) sample*(vectors.count()-1)/(samples-1));
                    double expected = decodedSymmetric(ash,encodedQueries[q],vectors.get(node));
                    scorer.fill(node,1,one);
                    double error = Math.abs(one[0]-expected);
                    if (!Float.isFinite(one[0]) || !Double.isFinite(expected)
                            || error > 5e-5*Math.max(1,Math.abs(expected))) {
                        throw new AssertionError(names.get(mode)+" decoder mismatch q="+q+" node="+node
                                +" expected="+expected+" actual="+one[0]);
                    }
                    maxError = Math.max(maxError,error);
                }
            }
            System.out.printf(java.util.Locale.ROOT,"SYMMETRIC_VALIDATION mode=%s max_abs_error=%.9g%n",names.get(mode),maxError);
        }
        if (recall) {
            if (recallK < 1 || recallK > vectors.count() || groundTruth.size() < queries.size()) {
                throw new IllegalArgumentException("Invalid recall size or missing ground truth");
            }
            double[] totals = new double[names.size()];
            float[] scores = new float[32];
            for (int q = 0; q < queries.size(); q++) {
                int[] gt = groundTruth.get(q).stream().mapToInt(Integer::intValue).toArray();
                if (gt.length < recallK) throw new IllegalArgumentException("Ground truth row too short");
                for (int mode = 0; mode < names.size(); mode++) {
                    var scorer = factories.get(mode).apply(q);
                    var heap = new java.util.PriorityQueue<long[]>((a,b) -> Float.compare(
                            Float.intBitsToFloat((int)a[0]),Float.intBitsToFloat((int)b[0])));
                    for (int node = 0; node < vectors.count(); node += scores.length) {
                        int n = Math.min(scores.length,vectors.count()-node);
                        scorer.fill(node,n,scores);
                        for (int i = 0; i < n; i++) offerTop(heap,recallK,scores[i],node+i);
                    }
                    addRecall(rankedOrdinals(heap),gt,newToOld,recallK,new int[]{recallK},totals,mode);
                }
            }
            for (int mode = 0; mode < names.size(); mode++) {
                System.out.printf(java.util.Locale.ROOT,"SCORING_RECALL mode=%s queries=%d %d-recall@%d=%.6f%n",
                        names.get(mode),queries.size(),recallK,recallK,totals[mode]/queries.size());
            }
        }
        System.out.println("SCORING_TIMING query encoding excluded; asymmetric projection and all scorer setup included once per scan; symmetric input already encoded");
        long minNanos = (long) (Double.parseDouble(System.getProperty("jvector.bench.scoringMinSeconds", "0.25")) * 1e9);
        if (minNanos < 0) throw new IllegalArgumentException("scoringMinSeconds must be nonnegative");
        for (var factory : factories) {
            if (warmupQueries == 0) continue;
            long until = System.nanoTime() + Math.max(minNanos, 1_000_000_000L);
            do { scanWithSetup(factory,warmupQueries,timingVectors,executor); }
            while (System.nanoTime() < until);
        }
        double[][] times = new double[names.size()][passes];
        for (int pass = 0; pass < passes; pass++) {
            for (int order = 0; order < names.size(); order++) {
                int mode = pass % 2 == 0 ? order : names.size()-1-order;
                start = System.nanoTime();
                double checksum = 0;
                int repetitions = 0;
                long elapsed;
                do {
                    checksum += scanWithSetup(factories.get(mode),timingQueries,timingVectors,executor);
                    repetitions++;
                    elapsed = System.nanoTime()-start;
                } while (elapsed < minNanos);
                times[mode][pass] = elapsed / (double) repetitions;
                System.out.printf(java.util.Locale.ROOT,"SCORING_PASS mode=%s pass=%d seconds=%.6f repetitions=%d checksum=%.6f%n",
                        names.get(mode),pass+1,elapsed/1e9,repetitions,checksum);
            }
        }
        long pairs = (long) timingQueries*timingVectors;
        for (int mode = 0; mode < names.size(); mode++) {
            java.util.Arrays.sort(times[mode]);
            double median = times[mode][passes/2];
            System.out.printf(java.util.Locale.ROOT,
                    "SCORING_THROUGHPUT mode=%s threads=%d queries=%d vectors=%d passes=%d median_Mdot_s=%.6f median_ns_pair=%.3f min_scan_s=%.6f max_scan_s=%.6f%n",
                    names.get(mode),executor.getParallelism(),timingQueries,timingVectors,passes,
                    pairs*1000.0/median,median/pairs,times[mode][0]/1e9,times[mode][passes-1]/1e9);
        }
    }

    private static double scanWithSetup(java.util.function.IntFunction<ScanScorer> factory,
                                         int queries, int count, ForkJoinPool executor) {
        if (queries == 0) return 0;
        int workers = Math.min(executor.getParallelism(),queries);
        List<ForkJoinTask<Double>> tasks = new ArrayList<>();
        for (int worker = 0; worker < workers; worker++) {
            final int first = worker;
            tasks.add(executor.submit(() -> {
                double sum = 0;
                float[] scratch = new float[32];
                for (int q = first; q < queries; q += workers) sum += factory.apply(q).scan(count,scratch);
                return sum;
            }));
        }
        double sum = 0;
        for (var task : tasks) sum += task.join();
        return sum;
    }

    public static void testASHEncodings(String filenameBase, String filenameQueries, String filenameGT) throws IOException {
        // ------------------------------------------------------------
        // Benchmark configuration (runtime flags)
        // ------------------------------------------------------------
        final boolean RUN_SANITY_CHECK =
                Boolean.parseBoolean(System.getProperty("jvector.bench.sanity-check", "false"));

        final boolean RUN_RECALL_CHECK =
                Boolean.parseBoolean(System.getProperty("jvector.bench.recall", "false"));

        final boolean RUN_BLOCK_RECALL =
                Boolean.parseBoolean(System.getProperty("jvector.bench.recall.block-scoring", "false"));

        final boolean COMPARE_ALL_RECALL =
                Boolean.parseBoolean(System.getProperty("jvector.bench.recall.compare-all", "false"));

        final boolean RECALL_ONLY_TEN =
                Boolean.parseBoolean(System.getProperty("jvector.bench.recall.only10", "false"));

        if (COMPARE_ALL_RECALL && !RUN_BLOCK_RECALL) {
            throw new IllegalArgumentException("compare-all requires recall.block-scoring=true");
        }

        final boolean RUN_BLOCK_TIMING =
                Boolean.parseBoolean(System.getProperty("jvector.bench.block-scoring", "true"));

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

        // How many ASH landmarks to use, C = [1, 256]
        final int landmarkCount = Integer.getInteger("jvector.ash.landmarkCount", 1);

        List<VectorFloat<?>> vectors = SiftLoader.readFvecs(filenameBase);
        List<VectorFloat<?>> allQueries = SiftLoader.readFvecs(filenameQueries);
        int maxQueries = Integer.getInteger("jvector.bench.maxQueries", allQueries.size());
        if (maxQueries <= 0) throw new IllegalArgumentException("maxQueries must be positive");
        List<VectorFloat<?>> queries = allQueries.subList(0, Math.min(maxQueries, allQueries.size()));
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

        if (vectors.isEmpty() || queries.isEmpty()) {
            throw new IllegalArgumentException("Base and query vectors must be nonempty");
        }
        int dimension = vectors.get(0).length();

        final int bitsPerDimension =
                Integer.getInteger("jvector.ash.bitsPerDimension", 4);

        final int quantizedDimensions =
                Integer.getInteger("jvector.ash.quantizedDimensions", dimension / 2);

        if (bitsPerDimension < 1 || bitsPerDimension > 9
                || (!Boolean.getBoolean("jvector.bench.symmetric-scoring")
                    && bitsPerDimension != 1 && bitsPerDimension != 2 && bitsPerDimension != 4)) {
            throw new IllegalArgumentException(
                    "bitsPerDimension must be 1, 2, or 4 (symmetric comparison also supports widths 3 through 9): " + bitsPerDimension);
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

        if (Boolean.getBoolean("jvector.bench.symmetric-scoring")) {
            // A separate comparison mode uses identical scan loops and prepared scorers.
            // Encoding, validation, recall, and warmup are excluded. Projection/setup are timed.
            try {
                benchmarkSymmetric(ashVectorsFinal, finalQueries, groundTruth, newToOldFinal,
                        RECALL_K, RUN_RECALL_CHECK, simdExecutor);
            } finally {
                simdExecutor.shutdown();
            }
            return;
        }

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
            int[] atValues = RECALL_ONLY_TEN ? new int[]{10} : new int[]{10, 15, 20, 30, 40, 50};
            int maxAt = atValues[atValues.length - 1];
            List<ForkJoinTask<double[]>> recallTasks = new ArrayList<>();

            if (RUN_BLOCK_RECALL) {
                validateRecallScorers(ashVectorsFinal, finalQueries, BLOCK_SIZES[0]);
            }

            logProgress("\t[stage] Computing " + RECALL_K + "-Recall@K with "
                    + (COMPARE_ALL_RECALL ? "ASH LUT block, single-vector and scalar"
                    : RUN_BLOCK_RECALL ? "ASH LUT block" : "ASH single-vector") + " scoring...");
            for (int start = 0; start < queries.size(); start += chunkSize) {
                final int s = start;
                final int e = Math.min(start + chunkSize, queries.size());

                recallTasks.add(simdExecutor.submit(() -> {
                    double[] localTotalRecall = new double[atValues.length * (COMPARE_ALL_RECALL ? 3 : 1)];
                    for (int i = s; i < e; i++) {
                        VectorFloat<?> q = finalQueries.get(i);
                        ScoreFunction.ApproximateScoreFunction f = RUN_BLOCK_RECALL && !COMPARE_ALL_RECALL ? null
                                : ashVecsFinal.scoreFunctionFor(q, VectorSimilarityFunction.DOT_PRODUCT);
                        ASHBlockScorer block = RUN_BLOCK_RECALL
                                ? ashVectorsFinal.blockScorerFor(q, VectorSimilarityFunction.DOT_PRODUCT, BLOCK_SIZES[0])
                                : null;
                        ASHBlockScorer scalar = COMPARE_ALL_RECALL
                                ? ashVectorsFinal.blockScorerFor(q, VectorSimilarityFunction.DOT_PRODUCT)
                                : null;
                        float[] blockScores = RUN_BLOCK_RECALL ? new float[BLOCK_SIZES[0]] : null;
                        float[] scalarScores = COMPARE_ALL_RECALL ? new float[BLOCK_SIZES[0]] : null;

                        // Min-heaps retain the top candidates from each scoring path.
                        var topCandidates = new java.util.PriorityQueue<long[]>((a, b) -> Float.compare(Float.intBitsToFloat((int) a[0]), Float.intBitsToFloat((int) b[0])));
                        java.util.PriorityQueue<long[]> singleCandidates = COMPARE_ALL_RECALL
                                ? new java.util.PriorityQueue<>((a, b) -> Float.compare(Float.intBitsToFloat((int) a[0]), Float.intBitsToFloat((int) b[0]))) : null;
                        java.util.PriorityQueue<long[]> scalarCandidates = COMPARE_ALL_RECALL
                                ? new java.util.PriorityQueue<>((a, b) -> Float.compare(Float.intBitsToFloat((int) a[0]), Float.intBitsToFloat((int) b[0]))) : null;
                        for (int startOrdinal = 0; startOrdinal < vectors.size(); startOrdinal += BLOCK_SIZES[0]) {
                            int count = Math.min(BLOCK_SIZES[0], vectors.size() - startOrdinal);
                            if (RUN_BLOCK_RECALL) block.scoreRange(startOrdinal, count, blockScores);
                            if (COMPARE_ALL_RECALL) scalar.scoreRange(startOrdinal, count, scalarScores);
                            for (int lane = 0; lane < count; lane++) {
                                int ordinal = startOrdinal + lane;
                                float score = RUN_BLOCK_RECALL ? blockScores[lane] : f.similarityTo(ordinal);
                                offerTop(topCandidates, maxAt, score, ordinal);
                                if (COMPARE_ALL_RECALL) {
                                    offerTop(singleCandidates, maxAt, f.similarityTo(ordinal), ordinal);
                                    offerTop(scalarCandidates, maxAt, scalarScores[lane], ordinal);
                                }
                            }
                        }

                        int[] queryGT = groundTruth.get(i).stream()
                                .mapToInt(Integer::intValue)
                                .toArray();
                        addRecall(rankedOrdinals(topCandidates), queryGT, newToOldFinal,
                                RECALL_K, atValues, localTotalRecall, 0);
                        if (COMPARE_ALL_RECALL) {
                            addRecall(rankedOrdinals(singleCandidates), queryGT, newToOldFinal,
                                    RECALL_K, atValues, localTotalRecall, atValues.length);
                            addRecall(rankedOrdinals(scalarCandidates), queryGT, newToOldFinal,
                                    RECALL_K, atValues, localTotalRecall, 2 * atValues.length);
                        }
                    }
                    return localTotalRecall;
                }));
            }

            double[] totalRecall = new double[atValues.length * (COMPARE_ALL_RECALL ? 3 : 1)];
            for (ForkJoinTask<double[]> t : recallTasks) {
                double[] local = t.join();
                for (int aIdx = 0; aIdx < totalRecall.length; aIdx++) totalRecall[aIdx] += local[aIdx];
            }

            String[] scorerNames = COMPARE_ALL_RECALL ? new String[]{"LUT-block", "single", "scalar"}
                    : new String[]{RUN_BLOCK_RECALL ? "LUT-block" : "single"};
            for (int scorer = 0; scorer < scorerNames.length; scorer++) {
                for (int aIdx = 0; aIdx < atValues.length; aIdx++) {
                    System.out.format(java.util.Locale.ROOT, "\tASH %s %d-recall@%d = %.6f%n",
                            scorerNames[scorer], RECALL_K, atValues[aIdx],
                            totalRecall[scorer * atValues.length + aIdx] / queries.size());
                }
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

        if (RUN_BLOCK_TIMING) for (int blockSize : BLOCK_SIZES) {

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
        if (args.length == 3) {
            testASHEncodings(args[0], args[1], args[2]);
            return;
        }
        if (args.length != 0) {
            throw new IllegalArgumentException("Expected base.fvecs query.fvecs ground-truth.ivecs");
        }
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
