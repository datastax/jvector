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
package io.github.jbellis.jvector.gpu.bench;

import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.gpu.MappedVectors;
import io.github.jbellis.jvector.gpu.TornadoGraphBuildAccelerator;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Graph build on Cohere's English Wikipedia embeddings (embed-multilingual-v3, 1024 dimensions), as prepared by
 * {@code jvector-gpu/datasets/wikipedia.sh}: {@code base.f32} and {@code query.f32} (raw float32 rows) and
 * {@code gt_ip_100.ivecs} (exact neighbours). Same settings as the other benchmarks: M=32, efConstruction=100,
 * overflow 1.2, alpha 1.2, hierarchy.
 * <ul>
 * <li>{@code gpu}: jvector-gpu's accelerator over the memory-mapped base file (no heap copy);</li>
 * <li>{@code cpu}: JVector's own build over heap-resident vectors, its fastest setting (10M x 1024 needs a heap of
 * about 64 GB).</li>
 * </ul>
 * Both then search the same way: 1,000 queries, single-threaded, exact scores over the mapped file, recall@10 and
 * mean latency at beams 20 to 160.
 * <p>
 * Usage: WikipediaBench &lt;dataset dir&gt; gpu|cpu
 */
public final class WikipediaBench {
    private static final int DIM = 1024;
    private static final int QUERIES = 1000;

    private WikipediaBench() {}

    public static void main(String[] args) throws Exception {
        if (args.length < 2 || !Set.of("gpu", "cpu").contains(args[1])) {
            System.err.println("usage: WikipediaBench <dataset dir from datasets/wikipedia.sh> gpu|cpu");
            System.exit(1);
        }
        Path dir = Path.of(args[0]);
        String mode = args[1];
        var sim = VectorSimilarityFunction.COSINE;
        List<int[]> gt = ivecs(dir.resolve("gt_ip_100.ivecs"));
        try (var base = new MappedVectors(dir.resolve("base.f32"), DIM, -1);
             var queryFile = new MappedVectors(dir.resolve("query.f32"), DIM, QUERIES)) {
            List<VectorFloat<?>> queries = copy(queryFile);
            System.out.printf("Wikipedia (Cohere embed-multilingual-v3, English): %,d x %d, COSINE, %s build%n",
                    base.size(), DIM, mode);
            long t0 = System.nanoTime();
            ImmutableGraphIndex graph;
            if ("gpu".equals(mode)) {
                var builder = new GraphIndexBuilder(BuildScoreProvider.randomAccessScoreProvider(base, sim), DIM, 32, 100, 1.2f, 1.2f, true, true);
                var accelerator = new TornadoGraphBuildAccelerator();
                if (!accelerator.supports(builder, base, sim)) {
                    throw new IllegalStateException("the GPU build does not support this dataset here (see the log above)");
                }
                accelerator.build(builder, base, sim);
                builder.cleanup();
                graph = builder.getGraph();
            } else {
                var heap = new ListRandomAccessVectorValues(copy(base), DIM);
                System.out.printf("loaded the vectors onto the heap in %.1f s%n", (System.nanoTime() - t0) / 1e9);
                t0 = System.nanoTime();
                var builder = new GraphIndexBuilder(BuildScoreProvider.randomAccessScoreProvider(heap, sim), DIM, 32, 100, 1.2f, 1.2f, true, true);
                graph = builder.build(heap);
            }
            System.out.printf("%s graph build: %.1f s (a cold build in a fresh JVM)%n%n", mode.toUpperCase(), (System.nanoTime() - t0) / 1e9);

            System.out.printf("%6s %10s %10s%n", "beam", "recall@10", "mean us");
            var searcher = new GraphSearcher(graph);
            for (int beam : new int[] {20, 40, 80, 160}) {
                search(searcher, base, queries, gt, sim, beam); // warm-up pass
                double[] r = search(searcher, base, queries, gt, sim, beam);
                System.out.printf("%6d %10.4f %10.1f%n", beam, r[0], r[1]);
            }
        }
    }

    /** {recall@10, mean microseconds a query} of a pass over the queries. */
    private static double[] search(GraphSearcher searcher, RandomAccessVectorValues base, List<VectorFloat<?>> queries,
                                   List<int[]> gt, VectorSimilarityFunction sim, int beam) {
        long hits = 0;
        long nanos = 0;
        for (int q = 0; q < queries.size(); q++) {
            long t = System.nanoTime();
            var result = searcher.search(DefaultSearchScoreProvider.exact(queries.get(q), sim, base), 10, beam, 0.0f, 0.0f, Bits.ALL);
            nanos += System.nanoTime() - t;
            Set<Integer> truth = new HashSet<>();
            for (int i = 0; i < 10; i++) {
                truth.add(gt.get(q)[i]);
            }
            for (var ns : result.getNodes()) {
                if (truth.contains(ns.node)) {
                    hits++;
                }
            }
        }
        return new double[] {hits / (10.0 * queries.size()), nanos / 1e3 / queries.size()};
    }

    /** Heap copies of every row. */
    private static List<VectorFloat<?>> copy(MappedVectors vectors) {
        VectorFloat<?>[] rows = new VectorFloat<?>[vectors.size()];
        IntStream.range(0, rows.length).parallel().forEach(i -> rows[i] = vectors.getVector(i));
        return new ArrayList<>(List.of(rows));
    }

    /** An .ivecs file: each row a little-endian int count, then that many ints. */
    private static List<int[]> ivecs(Path path) throws IOException {
        List<int[]> rows = new ArrayList<>();
        try (var in = new DataInputStream(new java.io.BufferedInputStream(Files.newInputStream(path), 1 << 20))) {
            while (true) {
                int k;
                try {
                    k = Integer.reverseBytes(in.readInt());
                } catch (EOFException end) {
                    return rows;
                }
                int[] row = new int[k];
                for (int i = 0; i < k; i++) {
                    row[i] = Integer.reverseBytes(in.readInt());
                }
                rows.add(row);
            }
        }
    }
}
