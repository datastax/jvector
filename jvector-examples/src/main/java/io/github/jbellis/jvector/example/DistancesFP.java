/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.example;

import com.sun.management.ThreadMXBean;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSets;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.SplittableRandom;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.stream.IntStream;

/**
 * FP32 distance benchmark. Pass a dataset name from the public dataset catalog (default
 * {@value #DEFAULT_DATASET}; it is downloaded and cached on first use) or {@code --synthetic dimensions}
 * for random unit vectors. Use separate JVMs with {@code jvector.vectorization_provider} set to
 * {@code native}, {@code panama}, or {@code default} to compare backends.
 *
 * <p>Timing mode (the default) runs on one thread. Properties (prefix {@code jvector.bench.}):
 * {@code maxBaseVectors} (65536), {@code maxQueries} (16), {@code candidatesPerQuery} (entire base),
 * {@code scoringWarmupPasses} (3), and {@code scoringSamples} (3). A smaller candidate count gives
 * each query a different window of the selected traversal order. The direct mode calls the
 * production similarity. The construction mode creates the same exact scorer used for graph
 * insertion once per query and includes that setup in the measurement. Input loading and shuffling
 * happen outside the measurements. The first pass is reported separately because it can initialize
 * native vector views and compile code. Allocation measurements count Java heap bytes on the
 * scoring thread.</p>
 *
 * <p>Recall mode ({@code -Djvector.bench.recall=true}), like DistancesASH and DistancesPQ, scores
 * every query against the whole base with the dataset's similarity and reports exact FP32
 * {@code recall.k}-recall@N against the dataset's ground truth for N = 10, 15, 20, 30, 40 and 50.
 * It uses all base vectors and all queries (or {@code maxQueries}, if set), {@code recall.k}
 * (default 10), and {@code scoringThreads} threads (default 1), and skips the timing measurements.
 * A {@code recall.k}-recall@{@code recall.k} below 1.0 means the ground truth disagrees with an exact scan.</p>
 */
public final class DistancesFP {
    private DistancesFP() {}

    static final String DEFAULT_DATASET = "e5-small-v2-100k";

    public static void main(String[] args) throws IOException {
        if (args.length == 2 && args[0].equals("--synthetic")) {
            int dimension = Integer.parseInt(args[1]);
            if (dimension <= 0) throw new IllegalArgumentException("Dimensions must be positive");
            int maxBase = positiveProperty("maxBaseVectors", 65_536);
            int maxQueries = positiveProperty("maxQueries", 16);
            var random = new SplittableRandom(42);
            var provider = VectorizationProvider.getInstance();
            timing(syntheticVectors(provider, random, maxBase, dimension),
                    syntheticVectors(provider, random, maxQueries, dimension), VectorSimilarityFunction.DOT_PRODUCT);
            return;
        }
        if (args.length > 1) {
            throw new IllegalArgumentException("Usage: DistancesFP [dataset-name] | --synthetic <dimensions>");
        }
        String name = args.length == 0 ? DEFAULT_DATASET : args[0];
        try (DataSet dataset = DataSets.loadDataSet(name)
                .orElseThrow(() -> new IllegalArgumentException("Unknown dataset: " + name)).getDataSet()) {
            System.out.printf(Locale.ROOT, "Dataset %s, similarity %s%n", dataset.getName(), dataset.getSimilarityFunction());
            if (Boolean.getBoolean("jvector.bench.recall")) {
                recall(dataset);
            } else {
                var base = dataset.getBaseVectors();
                var queries = dataset.getQueryVectors();
                timing(base.subList(0, Math.min(base.size(), positiveProperty("maxBaseVectors", 65_536))),
                        queries.subList(0, Math.min(queries.size(), positiveProperty("maxQueries", 16))),
                        dataset.getSimilarityFunction());
            }
        }
    }

    private static void timing(List<VectorFloat<?>> base, List<VectorFloat<?>> queries, VectorSimilarityFunction similarity) {
        int warmups = positiveProperty("scoringWarmupPasses", 3);
        int samples = positiveProperty("scoringSamples", 3);
        var provider = VectorizationProvider.getInstance();
        if (base.isEmpty() || queries.isEmpty()) {
            throw new IllegalArgumentException("Base and query vectors must be nonempty");
        }
        int dimensions = base.get(0).length();
        if (base.stream().anyMatch(v -> v.length() != dimensions)
                || queries.stream().anyMatch(v -> v.length() != dimensions)) {
            throw new IllegalArgumentException("All base and query vectors must have the same dimension");
        }
        int candidates = positiveProperty("candidatesPerQuery", base.size());
        if (candidates > base.size()) throw new IllegalArgumentException("candidatesPerQuery exceeds the base size");

        var alloc = ManagementFactory.getPlatformMXBean(ThreadMXBean.class);
        if (alloc == null || !alloc.isThreadAllocatedMemorySupported()) {
            throw new IllegalStateException("This JVM does not support per-thread heap allocation measurements");
        }
        if (!alloc.isThreadAllocatedMemoryEnabled()) alloc.setThreadAllocatedMemoryEnabled(true);
        var scoreProvider = BuildScoreProvider.randomAccessScoreProvider(
                new ListRandomAccessVectorValues(base, dimensions), similarity);
        var sequential = IntStream.range(0, base.size()).toArray();
        var shuffled = sequential.clone();
        var random = new SplittableRandom(43);
        for (int i = shuffled.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int saved = shuffled[i];
            shuffled[i] = shuffled[j];
            shuffled[j] = saved;
        }

        System.out.printf(Locale.ROOT, "DistancesFP provider=%s dimensions=%d base=%d queries=%d candidates/query=%d%n",
                provider.getClass().getSimpleName(), dimensions, base.size(), queries.size(), candidates);
        System.out.println("phase,access,scorer,sample,scores,seconds,ns_per_score,million_scores_per_s,"
                + "allocated_bytes,bytes_per_score,bytes_per_query,checksum");
        run("sequential", sequential, base, queries, candidates, similarity, scoreProvider, alloc, warmups, samples);
        run("shuffled", shuffled, base, queries, candidates, similarity, scoreProvider, alloc, warmups, samples);
    }

    private static void run(String access, int[] order, List<VectorFloat<?>> base, List<VectorFloat<?>> queries,
                            int candidates, VectorSimilarityFunction similarity, BuildScoreProvider provider,
                            ThreadMXBean alloc, int warmups, int samples) {
        long thread = Thread.currentThread().getId();
        long comparisons = (long) candidates * queries.size();
        double expected = Double.NaN;
        for (String scorer : List.of("direct", "construction")) {
            boolean construction = scorer.equals("construction");
            for (int sample = -warmups; sample < samples; sample++) {
                long beforeBytes = alloc.getThreadAllocatedBytes(thread);
                long start = System.nanoTime();
                double checksum = score(order, base, queries, candidates, similarity, provider, construction);
                long elapsed = System.nanoTime() - start;
                long bytes = alloc.getThreadAllocatedBytes(thread) - beforeBytes;
                if (!Double.isFinite(checksum)) throw new IllegalStateException("Non-finite score checksum");
                if (sample >= 0) {
                    if (Double.isNaN(expected)) expected = checksum;
                    if (checksum != expected) throw new IllegalStateException("Direct and construction scores differ");
                }
                if (sample == -warmups || sample >= 0) {
                    System.out.printf(Locale.ROOT, "%s,%s,%s,%d,%d,%.6f,%.2f,%.3f,%d,%.6f,%.2f,%.9f%n",
                            sample < 0 ? "first-pass" : "steady", access, scorer, sample < 0 ? 1 : sample + 1,
                            comparisons, elapsed / 1e9, (double) elapsed / comparisons,
                            comparisons * 1e3 / elapsed, bytes, (double) bytes / comparisons,
                            (double) bytes / queries.size(), checksum);
                }
            }
        }
    }

    private static double score(int[] order, List<VectorFloat<?>> base, List<VectorFloat<?>> queries,
                                int candidates, VectorSimilarityFunction similarity, BuildScoreProvider provider,
                                boolean construction) {
        double checksum = 0;
        for (int q = 0; q < queries.size(); q++) {
            var query = queries.get(q);
            int start = (int) ((long) q * candidates % (order.length - candidates + 1));
            int end = start + candidates;
            if (construction) {
                var scorer = provider.searchProviderFor(query).scoreFunction();
                for (int i = start; i < end; i++) checksum += scorer.similarityTo(order[i]);
            } else {
                for (int i = start; i < end; i++) checksum += similarity.compare(query, base.get(order[i]));
            }
        }
        return checksum;
    }

    /** Exact FP32 recall against the dataset's ground truth, scoring every query against the whole base. */
    private static void recall(DataSet dataset) {
        int recallK = positiveProperty("recall.k", 10);
        int threads = positiveProperty("scoringThreads", 1);
        int[] atValues = {10, 15, 20, 30, 40, 50};
        int maxAt = atValues[atValues.length - 1];
        var similarity = dataset.getSimilarityFunction();
        List<VectorFloat<?>> base = dataset.getBaseVectors();
        List<VectorFloat<?>> allQueries = dataset.getQueryVectors();
        List<VectorFloat<?>> queries = allQueries.subList(0,
                Math.min(allQueries.size(), positiveProperty("maxQueries", Integer.MAX_VALUE)));
        List<? extends List<Integer>> groundTruth = dataset.getGroundTruth();
        if (base.isEmpty() || queries.isEmpty()) throw new IllegalArgumentException("Base and query vectors must be nonempty");
        if (groundTruth.size() < queries.size())
            throw new IllegalArgumentException("Ground truth has " + groundTruth.size() + " rows for " + queries.size() + " queries");
        if (maxAt > base.size()) throw new IllegalArgumentException("Recall needs at least " + maxAt + " base vectors");
        for (int q = 0; q < queries.size(); q++) {
            if (groundTruth.get(q).size() < recallK)
                throw new IllegalArgumentException("Ground truth row " + q + " has fewer than recall.k=" + recallK + " entries");
        }
        int dimensions = base.get(0).length();
        System.out.printf(Locale.ROOT, "DistancesFP recall provider=%s dimensions=%d base=%d queries=%d recall.k=%d threads=%d%n",
                VectorizationProvider.getInstance().getClass().getSimpleName(), dimensions, base.size(),
                queries.size(), recallK, threads);
        long start = System.nanoTime();
        double[] totals = new double[atValues.length];
        var pool = new ForkJoinPool(threads);
        try {
            int chunk = (queries.size() + threads - 1) / threads;
            var tasks = new ArrayList<ForkJoinTask<double[]>>();
            for (int first = 0; first < queries.size(); first += chunk) {
                int from = first, to = Math.min(first + chunk, queries.size());
                tasks.add(pool.submit(() -> {
                    double[] local = new double[atValues.length];
                    int[][] ids = new int[QUERY_BLOCK][maxAt];
                    float[][] scores = new float[QUERY_BLOCK][maxAt];
                    int[] sizes = new int[QUERY_BLOCK];
                    for (int block = from; block < to; block += QUERY_BLOCK) {
                        int count = Math.min(QUERY_BLOCK, to - block);
                        topScores(queries.subList(block, block + count), base, similarity, maxAt, ids, scores, sizes);
                        for (int b = 0; b < count; b++) {
                            var expected = new HashSet<Integer>(groundTruth.get(block + b).subList(0, recallK));
                            for (int a = 0; a < atValues.length; a++) {
                                int matches = 0;
                                for (int r = 0; r < Math.min(atValues[a], sizes[b]); r++) {
                                    if (expected.contains(ids[b][r])) matches++;
                                }
                                local[a] += (double) matches / expected.size();
                            }
                        }
                    }
                    return local;
                }));
            }
            for (var task : tasks) {
                double[] local = task.join();
                for (int a = 0; a < totals.length; a++) totals[a] += local[a];
            }
        } finally {
            pool.shutdown();
        }
        for (int a = 0; a < atValues.length; a++) {
            System.out.printf(Locale.ROOT, "\tFP32 exact %d-recall@%d = %.6f%n", recallK, atValues[a], totals[a] / queries.size());
        }
        System.out.printf(Locale.ROOT, "\tRecall scan time: %.1f s%n", (System.nanoTime() - start) / 1e9);
    }

    /** Queries scored per pass over the base, so each base vector is read from memory once per block. */
    private static final int QUERY_BLOCK = 32;

    /**
     * For each query, fills ids[q]/scores[q] with its n highest similarity scores, best first, and sizes[q]
     * with how many were filled. All queries share one pass over the base.
     */
    private static void topScores(List<VectorFloat<?>> queries, List<VectorFloat<?>> base,
                                  VectorSimilarityFunction similarity, int n, int[][] ids, float[][] scores, int[] sizes) {
        Arrays.fill(sizes, 0);
        for (int i = 0; i < base.size(); i++) {
            var vector = base.get(i);
            for (int q = 0; q < queries.size(); q++) {
                float score = similarity.compare(queries.get(q), vector);
                float[] top = scores[q];
                int size = sizes[q];
                if (size == n && score <= top[n - 1]) continue;
                int[] topIds = ids[q];
                int pos = size < n ? size++ : n - 1;
                while (pos > 0 && top[pos - 1] < score) {
                    top[pos] = top[pos - 1];
                    topIds[pos] = topIds[pos - 1];
                    pos--;
                }
                top[pos] = score;
                topIds[pos] = i;
                sizes[q] = size;
            }
        }
    }

    private static List<VectorFloat<?>> syntheticVectors(VectorizationProvider provider, SplittableRandom random,
                                                         int count, int dimensions) {
        var support = provider.getVectorTypeSupport();
        var vectors = new ArrayList<VectorFloat<?>>(count);
        for (int i = 0; i < count; i++) {
            var values = new float[dimensions];
            double squaredNorm = 0;
            for (int d = 0; d < dimensions; d++) {
                values[d] = (float) random.nextDouble(-1, 1);
                squaredNorm += (double) values[d] * values[d];
            }
            double scale = 1 / Math.sqrt(squaredNorm);
            for (int d = 0; d < dimensions; d++) values[d] = (float) (values[d] * scale);
            vectors.add(support.createFloatVector(values));
        }
        return vectors;
    }

    private static int positiveProperty(String name, int defaultValue) {
        int value = Integer.parseInt(System.getProperty("jvector.bench." + name, Integer.toString(defaultValue)));
        if (value <= 0) throw new IllegalArgumentException(name + " must be positive");
        return value;
    }
}
