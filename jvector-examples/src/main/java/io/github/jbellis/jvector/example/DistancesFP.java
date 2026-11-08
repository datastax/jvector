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
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.SplittableRandom;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.vector.VectorSimilarityFunction.DOT_PRODUCT;

/**
 * Single-thread FP32 distance benchmark. Pass {@code base.fvecs query.fvecs} or
 * {@code --synthetic dimensions}. Use separate JVMs with {@code jvector.vectorization_provider}
 * set to {@code native}, {@code panama}, or {@code default} to compare backends.
 *
 * <p>Properties (prefix {@code jvector.bench.}): {@code maxBaseVectors} (65536),
 * {@code maxQueries} (16), {@code candidatesPerQuery} (entire base),
 * {@code scoringWarmupPasses} (3), and {@code scoringSamples} (3). A smaller candidate
 * count gives each query a different window of the selected traversal order.</p>
 *
 * <p>The direct mode calls the production dot-product similarity. The construction mode
 * creates the same exact scorer used for graph insertion once per query and includes that
 * setup in the measurement. Input loading and shuffling happen outside the measurements.
 * The first pass is reported separately because it can initialize native vector views and
 * compile code. Allocation measurements count Java heap bytes on the scoring thread.</p>
 */
public final class DistancesFP {
    private DistancesFP() {}

    public static void main(String[] args) {
        if (args.length != 2) {
            throw new IllegalArgumentException("Usage: DistancesFP <base.fvecs> <query.fvecs> | --synthetic <dimensions>");
        }
        int maxBase = positiveProperty("maxBaseVectors", 65_536);
        int maxQueries = positiveProperty("maxQueries", 16);
        int warmups = positiveProperty("scoringWarmupPasses", 3);
        int samples = positiveProperty("scoringSamples", 3);
        var provider = VectorizationProvider.getInstance();
        List<VectorFloat<?>> base;
        List<VectorFloat<?>> queries;
        if (args[0].equals("--synthetic")) {
            int dimension = Integer.parseInt(args[1]);
            if (dimension <= 0) throw new IllegalArgumentException("Dimensions must be positive");
            var random = new SplittableRandom(42);
            base = syntheticVectors(provider, random, maxBase, dimension);
            queries = syntheticVectors(provider, random, maxQueries, dimension);
        } else {
            base = SiftLoader.readFvecs(args[0], maxBase);
            queries = SiftLoader.readFvecs(args[1], maxQueries);
        }
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
                new ListRandomAccessVectorValues(base, dimensions), DOT_PRODUCT);
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
        run("sequential", sequential, base, queries, candidates, scoreProvider, alloc, warmups, samples);
        run("shuffled", shuffled, base, queries, candidates, scoreProvider, alloc, warmups, samples);
    }

    private static void run(String access, int[] order, List<VectorFloat<?>> base, List<VectorFloat<?>> queries,
                            int candidates, BuildScoreProvider provider, ThreadMXBean alloc, int warmups, int samples) {
        long thread = Thread.currentThread().getId();
        long comparisons = (long) candidates * queries.size();
        double expected = Double.NaN;
        for (String scorer : List.of("direct", "construction")) {
            boolean construction = scorer.equals("construction");
            for (int sample = -warmups; sample < samples; sample++) {
                long beforeBytes = alloc.getThreadAllocatedBytes(thread);
                long start = System.nanoTime();
                double checksum = score(order, base, queries, candidates, provider, construction);
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
                                int candidates, BuildScoreProvider provider, boolean construction) {
        double checksum = 0;
        for (int q = 0; q < queries.size(); q++) {
            var query = queries.get(q);
            int start = (int) ((long) q * candidates % (order.length - candidates + 1));
            int end = start + candidates;
            if (construction) {
                var scorer = provider.searchProviderFor(query).scoreFunction();
                for (int i = start; i < end; i++) checksum += scorer.similarityTo(order[i]);
            } else {
                for (int i = start; i < end; i++) checksum += DOT_PRODUCT.compare(query, base.get(order[i]));
            }
        }
        return checksum;
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
