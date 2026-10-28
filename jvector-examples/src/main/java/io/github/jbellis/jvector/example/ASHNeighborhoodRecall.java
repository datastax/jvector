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

import io.github.jbellis.jvector.quantization.ASHNeighborhoodScorer;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.PriorityQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Exact full-scan recall@10 evaluation through the production neighborhood scorer.
 *
 * <p>Each query is prepared once and scores every serialized neighborhood. Only real base
 * ordinals enter top-k; padded lanes are scored but excluded from ranking. Ties prefer the
 * smaller original ordinal in every backend. Query parallelism changes neither the scan work
 * nor its reduction order. This is an accuracy check, not a kernel throughput benchmark.</p>
 */
final class ASHNeighborhoodRecall {
    static final int K = 10;

    private ASHNeighborhoodRecall() {}

    static void run(ASHNeighborhoodFixture fixture, int threads) throws IOException {
        String groundTruthName = System.getProperty("jvector.bench.recall.groundTruth");
        if (groundTruthName == null) {
            throw new IllegalArgumentException("Set jvector.bench.recall.groundTruth to the matching .ivecs file");
        }
        int[][] truth = readGroundTruth(Path.of(groundTruthName), fixture.queries.size(), fixture.baseCount);
        String outputName = System.getProperty("jvector.bench.recall.output");
        if (outputName != null && Files.exists(Path.of(outputName))) {
            throw new IOException("Recall output already exists: " + outputName);
        }
        // Keep independent component validation bounded; recall itself uses every saved query.
        var validation =
                new ASHNeighborhoodFixture(fixture.ash, fixture.queries.subList(0, Math.min(4, fixture.queries.size())),
                        fixture.neighborhoods, fixture.capacity, fixture.blockSize, fixture.baseCount);
        ASHNeighborhoodValidator.validate(validation);
        System.out.printf(Locale.ROOT,
                "Full-scan recall@10: queries=%d baseVectors=%d neighborhoods=%d queryWorkers=%d; prepare once per query; no reranking; padding excluded from top-k%n",
                fixture.queries.size(), fixture.baseCount, fixture.neighborhoods.length, threads);
        var pool = Executors.newFixedThreadPool(threads);
        List<Result> results = new ArrayList<>(fixture.queries.size());
        try {
            var finished = new AtomicInteger();
            List<Callable<Result>> tasks = new ArrayList<>(fixture.queries.size());
            for (int q = 0; q < fixture.queries.size(); q++) {
                final int query = q;
                tasks.add(() -> {
                    Result result = scan(fixture, query, truth[query]);
                    int count = finished.incrementAndGet();
                    if (count % 100 == 0 || count == fixture.queries.size()) {
                        System.out.printf(Locale.ROOT, "Recall progress: %d/%d queries complete%n", count,
                                fixture.queries.size());
                    }
                    return result;
                });
            }
            int completed = 0;
            for (var future : pool.invokeAll(tasks)) {
                results.add(future.get());
                completed++;
            }
            if (completed != fixture.queries.size()) {
                throw new AssertionError("Incomplete full-scan recall");
            }
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted during full-scan recall", error);
        } catch (ExecutionException error) {
            throw new IllegalStateException("Full-scan recall worker failed", error.getCause());
        } finally {
            pool.shutdownNow();
            boolean interrupted = false;
            try {
                while (!pool.isTerminated()) {
                    try {
                        pool.awaitTermination(1, TimeUnit.SECONDS);
                    } catch (InterruptedException error) {
                        interrupted = true;
                    }
                }
            } finally {
                if (interrupted) {
                    Thread.currentThread().interrupt();
                }
            }
        }
        long hits = 0;
        for (var result : results) {
            hits += result.hits;
        }
        if (outputName != null) {
            try (var output = Files.newBufferedWriter(
                         Path.of(outputName), StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
                output.write("query\thits_at_10\tids_best_first\tscores_best_first\n");
                for (var result : results) {
                    output.write(result.query + "\t" + result.hits + "\t" + Arrays.toString(result.ids) + "\t"
                            + Arrays.toString(result.scores) + "\n");
                }
            }
        }
        System.out.printf(Locale.ROOT,
                "NEIGHBORHOOD_RECALL queries=%d baseVectors=%d hits=%d denominator=%d recall_at_10=%.9f%n",
                results.size(), fixture.baseCount, hits, (long) K * results.size(),
                hits / ((double) K * results.size()));
    }

    /** Scan exactly once; a worker owns the scorer, scratch, and top-k state for its query. */
    static Result scan(ASHNeighborhoodFixture fixture, int query, int[] truth) {
        var scorer =
                new ASHNeighborhoodScorer(fixture.ash, fixture.queries.get(query), fixture.capacity, fixture.blockSize);
        float[] scores = new float[fixture.capacity];
        var top = new TopK();
        for (int n = 0; n < fixture.neighborhoods.length; n++) {
            if (Thread.currentThread().isInterrupted()) {
                throw new IllegalStateException("Recall scan cancelled");
            }
            scorer.score(fixture.neighborhoods[n], scores);
            int base = n * fixture.capacity;
            int count = Math.min(fixture.capacity, fixture.baseCount - base);
            for (int lane = 0; lane < count; lane++) {
                float score = scores[lane];
                if (!Float.isFinite(score)) {
                    throw new IllegalStateException(
                            "Non-finite recall score at query=" + query + " ordinal=" + (base + lane));
                }
                top.offer(score, base + lane);
            }
        }
        Candidate[] ranked = top.ranked();
        int[] ids = new int[K];
        float[] bestScores = new float[K];
        int hits = 0;
        for (int rank = 0; rank < K; rank++) {
            ids[rank] = ranked[rank].ordinal;
            bestScores[rank] = ranked[rank].score;
            for (int expected : truth) {
                if (ids[rank] == expected) {
                    hits++;
                    break;
                }
            }
        }
        return new Result(query, ids, bestScores, hits);
    }

    /** Read little-endian ivecs; top ten must be unique, in-range original base ordinals. */
    static int[][] readGroundTruth(Path path, int queries, int baseCount) throws IOException {
        if (queries < 1 || baseCount < K) {
            throw new IllegalArgumentException("Recall@10 needs queries and at least ten real base vectors");
        }
        int[][] result = new int[queries][K];
        try (var input = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            for (int q = 0; q < queries; q++) {
                int width = Integer.reverseBytes(input.readInt());
                if (width < K || width > 100_000) {
                    throw new IOException("Invalid ground-truth width: " + width);
                }
                for (int rank = 0; rank < width; rank++) {
                    int id = Integer.reverseBytes(input.readInt());
                    if (id < 0 || id >= baseCount) {
                        throw new IOException("Ground-truth ordinal outside base set: " + id);
                    }
                    if (rank < K) {
                        for (int prior = 0; prior < rank; prior++) {
                            if (result[q][prior] == id) {
                                throw new IOException("Duplicate top-ten ground-truth ordinal at query " + q);
                            }
                        }
                        result[q][rank] = id;
                    }
                }
            }
        }
        return result;
    }

    static final class Result {
        final int query;
        final int[] ids;
        final float[] scores;
        final int hits;

        Result(int query, int[] ids, float[] scores, int hits) {
            this.query = query;
            this.ids = ids;
            this.scores = scores;
            this.hits = hits;
        }
    }

    static final class Candidate {
        final float score;
        final int ordinal;

        Candidate(float score, int ordinal) {
            this.score = score;
            this.ordinal = ordinal;
        }
    }

    /** Worst retained candidate first; equal scores evict the largest ordinal first. */
    static final class TopK {
        private static final Comparator<Candidate> WORST_FIRST = (a, b) -> {
            int comparison = Float.compare(a.score, b.score);
            return comparison != 0 ? comparison : Integer.compare(b.ordinal, a.ordinal);
        };
        private final PriorityQueue<Candidate> heap = new PriorityQueue<>(K, WORST_FIRST);

        void offer(float score, int ordinal) {
            Candidate worst = heap.peek();
            if (heap.size() < K) {
                heap.add(new Candidate(score, ordinal));
            } else if (score > worst.score || (score == worst.score && ordinal < worst.ordinal)) {
                heap.poll();
                heap.add(new Candidate(score, ordinal));
            }
        }

        Candidate[] ranked() {
            Candidate[] ranked = heap.toArray(new Candidate[0]);
            Arrays.sort(ranked, WORST_FIRST.reversed());
            return ranked;
        }
    }
}
