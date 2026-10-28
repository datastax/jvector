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

import io.github.jbellis.jvector.disk.SimpleWriter;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.Locale;

/**
 * DistancesPQ accuracy-only evaluation using the production precomputed ADC scorer.
 * Each query prepares one lookup table and scans the complete encoded base set.
 * Exact ties prefer smaller original ordinals, matching neighborhood ASH recall.
 * Optional outputs preserve per-query rankings and the trained/encoded PQ artifact.
 */
final class PQRecallEvaluator {
    private PQRecallEvaluator() {}

    static void run(PQVectors vectors, List<VectorFloat<?>> queries, List<List<Integer>> truth, int k,
            ForkJoinPool executor) throws IOException {
        if (queries.isEmpty() || k < 1 || k > vectors.count() || truth.size() < queries.size()) {
            throw new IllegalArgumentException("Invalid recall size or missing ground truth");
        }
        for (int q = 0; q < queries.size(); q++) {
            if (truth.get(q).size() < k || truth.get(q).subList(0, k).stream().distinct().count() != k) {
                throw new IllegalArgumentException("Ground truth must contain k distinct neighbors");
            }
            for (int id : truth.get(q).subList(0, k)) {
                if (id < 0 || id >= vectors.count()) {
                    throw new IllegalArgumentException("Ground-truth ordinal outside base set");
                }
            }
        }
        var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        try {
            var method = backend.getClass().getMethod(
                    "assembleAndSum", VectorFloat.class, int.class, ByteSequence.class, int.class, int.class);
            System.out.println("PQ ADC backend: " + backend.getClass().getName()
                    + "; assembly=" + method.getDeclaringClass().getName());
        } catch (NoSuchMethodException error) {
            throw new AssertionError("Missing PQ assembly contract", error);
        }
        var sample = vectors.precomputedScoreFunctionFor(queries.get(0), VectorSimilarityFunction.DOT_PRODUCT);
        System.out.println("PQ ADC scorer: " + sample.getClass().getName());
        System.out.println("Normalized query SHA-256: " + queryHash(queries));
        // Compare the lookup scorer with full reconstruction outside the accuracy scan.
        validate(vectors, queries);
        String artifact = System.getProperty("jvector.bench.pq.artifact");
        if (artifact != null) {
            Path target = Path.of(artifact);
            if (Files.exists(target)) {
                throw new IOException("PQ artifact already exists: " + target);
            }
            Path temporary = Files.createTempFile(target.toAbsolutePath().getParent(), ".pq-recall-", ".tmp");
            try {
                try (var writer = new SimpleWriter(temporary)) {
                    vectors.write(writer, OnDiskGraphIndex.CURRENT_VERSION);
                }
                Files.move(temporary, target);
            } finally {
                Files.deleteIfExists(temporary);
            }
        }
        String outputName = System.getProperty("jvector.bench.recall.output");
        if (outputName != null && Files.exists(Path.of(outputName))) {
            throw new IOException("Recall output already exists: " + outputName);
        }
        Result[] results = new Result[queries.size()];
        var finished = new AtomicInteger();
        List<ForkJoinTask<?>> tasks = new ArrayList<>();
        int workers = Math.min(executor.getParallelism(), queries.size());
        for (int worker = 0; worker < workers; worker++) {
            final int first = worker;
            tasks.add(executor.submit(() -> {
                for (int q = first; q < queries.size(); q += workers) {
                    var scorer =
                            vectors.precomputedScoreFunctionFor(queries.get(q), VectorSimilarityFunction.DOT_PRODUCT);
                    var result = rank(scorer, vectors.count(), truth.get(q).subList(0, k), k);
                    results[q] = result;
                    int count = finished.incrementAndGet();
                    if (count % 100 == 0 || count == queries.size()) {
                        System.out.printf(
                                Locale.ROOT, "PQ recall progress: %d/%d queries complete%n", count, queries.size());
                    }
                }
            }));
        }
        for (var task : tasks) {
            task.join();
        }
        long hits = Arrays.stream(results).mapToLong(r -> r.hits).sum();
        if (outputName != null) {
            try (var out = Files.newBufferedWriter(
                         Path.of(outputName), StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
                out.write("query\thits_at_" + k + "\tids_best_first\tscores_best_first\n");
                for (int q = 0; q < results.length; q++) {
                    var r = results[q];
                    out.write(q + "\t" + r.hits + "\t" + Arrays.toString(r.ids) + "\t" + Arrays.toString(r.scores)
                            + "\n");
                }
            }
        }
        System.out.printf(Locale.ROOT,
                "PQ_RECALL queries=%d baseVectors=%d subspaces=%d codeBytes=%d hits=%d denominator=%d recall_at_%d=%.9f%n",
                queries.size(), vectors.count(), vectors.getCompressor().getSubspaceCount(),
                vectors.getCompressor().compressedVectorSize(), hits, (long) k * queries.size(), k,
                hits / ((double) k * queries.size()));
    }

    /** Independent decoded-vector check; never used for ranking or reranking. */
    static void validate(PQVectors vectors, List<VectorFloat<?>> queries) {
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        var decoded = types.createFloatVector(queries.get(0).length());
        double maximum = 0;
        for (int q = 0; q < Math.min(4, queries.size()); q++) {
            var scorer = vectors.precomputedScoreFunctionFor(queries.get(q), VectorSimilarityFunction.DOT_PRODUCT);
            for (int i = 0; i < Math.min(256, vectors.count()); i++) {
                int id = (int) ((long) i * vectors.count() / Math.min(256, vectors.count()));
                vectors.getCompressor().decode(vectors.get(id), decoded);
                float expected = (1f + VectorUtil.dotProduct(queries.get(q), decoded)) / 2f;
                float actual = scorer.similarityTo(id);
                maximum = Math.max(maximum, Math.abs(expected - actual));
                if (!Float.isFinite(expected) || !Float.isFinite(actual)
                        || Math.abs(expected - actual) > 3e-5 + 3e-5 * Math.abs(expected)) {
                    throw new AssertionError("PQ lookup/reconstruction mismatch at query=" + q + " ordinal=" + id);
                }
            }
        }
        System.out.printf(Locale.ROOT, "PQ decoded-vector validation maxAbsError=%.9g%n", maximum);
    }

    static final class Result {
        final int[] ids;
        final float[] scores;
        final int hits;
        Result(int[] ids, float[] scores, int hits) {
            this.ids = ids;
            this.scores = scores;
            this.hits = hits;
        }
    }

    /** Keep precisely k results; input ordinals remain in their original base order. */
    static Result rank(ScoreFunction scorer, int count, List<Integer> truth, int k) {
        if (k < 1 || k > count) {
            throw new IllegalArgumentException("Recall k must be in [1, base count]");
        }
        Comparator<long[]> worst = (a, b) -> {
            int compare = Float.compare(Float.intBitsToFloat((int) a[0]), Float.intBitsToFloat((int) b[0]));
            return compare != 0 ? compare : Long.compare(b[1], a[1]);
        };
        var heap = new PriorityQueue<long[]>(k, worst);
        for (int id = 0; id < count; id++) {
            float score = scorer.similarityTo(id);
            if (!Float.isFinite(score)) {
                throw new IllegalStateException("Non-finite PQ score at ordinal " + id);
            }
            if (heap.size() < k) {
                heap.add(new long[] {Float.floatToRawIntBits(score), id});
            } else {
                var bottom = heap.peek();
                float threshold = Float.intBitsToFloat((int) bottom[0]);
                if (score > threshold || (score == threshold && id < bottom[1])) {
                    heap.poll();
                    heap.add(new long[] {Float.floatToRawIntBits(score), id});
                }
            }
        }
        int[] ids = new int[k];
        float[] scores = new float[k];
        int hits = 0;
        for (int rank = k - 1; rank >= 0; rank--) {
            var candidate = heap.remove();
            ids[rank] = (int) candidate[1];
            scores[rank] = Float.intBitsToFloat((int) candidate[0]);
            if (truth.contains(ids[rank])) {
                hits++;
            }
        }
        return new Result(ids, scores, hits);
    }

    private static String queryHash(List<VectorFloat<?>> queries) {
        try {
            var digest = MessageDigest.getInstance("SHA-256");
            var buffer = ByteBuffer.allocate(8192);
            for (var query : queries) {
                for (int d = 0; d < query.length(); d++) {
                    if (buffer.remaining() < Float.BYTES) {
                        digest.update(buffer.array(), 0, buffer.position());
                        buffer.clear();
                    }
                    buffer.putFloat(query.get(d));
                }
            }
            digest.update(buffer.array(), 0, buffer.position());
            var text = new StringBuilder();
            for (byte b : digest.digest()) {
                text.append(Character.forDigit((b >>> 4) & 15, 16)).append(Character.forDigit(b & 15, 16));
            }
            return text.toString();
        } catch (java.security.NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }
}
