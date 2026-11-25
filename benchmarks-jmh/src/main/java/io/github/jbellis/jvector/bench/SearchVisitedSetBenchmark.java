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
package io.github.jbellis.jvector.bench;

import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.diversity.VamanaDiversityProvider;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Generate each graph fixture once with this class's main method (directory, then graph size).
 * Pass the same directory to both JMH runs with -p fixtureDir=/absolute/path.
 * Use -prof gc to report allocation alongside query latency.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Fork(2)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Threads(1)
public class SearchVisitedSetBenchmark {
    private static final int DIMENSION = 128;
    private static final int QUERY_COUNT = 64;
    private static final float OVERFLOW = 1.2f;
    private static final float ALPHA = 1.2f;
    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.EUCLIDEAN;

    @Param({"."})
    String fixtureDir;

    @Param({"10000", "100000", "1000000"})
    int graphSize;

    @Param({"1", "2", "10", "100"})
    int topK;

    @Param({"true", "false"})
    boolean fresh;

    private RandomAccessVectorValues vectors;
    private OnHeapGraphIndex graph;
    private GraphSearcher reusedSearcher;
    private VectorFloat<?>[] queries;

    @Setup(Level.Trial)
    public void setup() throws IOException {
        vectors = vectors(graphSize);
        var buildScores = BuildScoreProvider.randomAccessScoreProvider(vectors, SIMILARITY);
        try (var supplier = ReaderSupplierFactory.open(fixturePath(Path.of(fixtureDir), graphSize));
             var reader = supplier.get()) {
            graph = OnHeapGraphIndex.load(reader, DIMENSION, OVERFLOW, new VamanaDiversityProvider(buildScores, ALPHA));
        }
        if (!fresh) {
            reusedSearcher = new GraphSearcher(graph);
        }
        var queryRandom = new Random(43);
        queries = new VectorFloat<?>[QUERY_COUNT];
        for (int i = 0; i < QUERY_COUNT; i++) {
            queries[i] = randomVector(queryRandom);
        }
        reportSearchWork();
    }

    @TearDown(Level.Trial)
    public void tearDown() throws IOException {
        if (reusedSearcher != null) {
            reusedSearcher.close();
        }
        graph.close();
    }

    @Benchmark
    @OperationsPerInvocation(QUERY_COUNT)
    public void search(Blackhole blackhole) throws IOException {
        for (var query : queries) {
            if (fresh) {
                try (var searcher = new GraphSearcher(graph)) {
                    blackhole.consume(searcher.search(scores(query), topK, Bits.ALL));
                }
            } else {
                blackhole.consume(reusedSearcher.search(scores(query), topK, Bits.ALL));
            }
        }
    }

    private SearchScoreProvider scores(VectorFloat<?> query) {
        return DefaultSearchScoreProvider.exact(query, SIMILARITY, vectors);
    }

    private void reportSearchWork() throws IOException {
        long visited = 0;
        long expanded = 0;
        try (var searcher = new GraphSearcher(graph)) {
            for (var query : queries) {
                var result = searcher.search(scores(query), topK, Bits.ALL);
                visited += result.getVisitedCount();
                expanded += result.getExpandedCount();
            }
        }
        System.out.printf("Search work per query: visited=%.2f, expanded=%.2f%n",
                          visited / (double) QUERY_COUNT, expanded / (double) QUERY_COUNT);
    }

    /** Measures construction separately from searches over the fixed fixture. */
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @State(Scope.Thread)
    public static class Build {
        private RandomAccessVectorValues vectors;

        @Setup(Level.Trial)
        public void setup() {
            vectors = vectors(10_000);
        }

        @Benchmark
        public int build() throws IOException {
            try (var builder = newBuilder(vectors); var graph = builder.build(vectors)) {
                return graph.size();
            }
        }
    }

    public static void main(String[] args) throws IOException {
        var directory = Files.createDirectories(Path.of(args[0]));
        int size = Integer.parseInt(args[1]);
        var vectors = vectors(size);
        try (var builder = newBuilder(vectors);
             var graph = (OnHeapGraphIndex) builder.build(vectors);
             var output = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(
                     fixturePath(directory, size), StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)))) {
            graph.save(output);
        }
    }

    private static GraphIndexBuilder newBuilder(RandomAccessVectorValues vectors) {
        var scores = BuildScoreProvider.randomAccessScoreProvider(vectors, SIMILARITY);
        return new GraphIndexBuilder(scores, DIMENSION, 16, 100, OVERFLOW, ALPHA, true);
    }

    private static Path fixturePath(Path directory, int size) {
        return directory.resolve("visited-set-" + size + ".graph");
    }

    private static RandomAccessVectorValues vectors(int size) {
        var random = new Random(42);
        var vectors = new ArrayList<VectorFloat<?>>(size);
        for (int i = 0; i < size; i++) {
            vectors.add(randomVector(random));
        }
        return new ListRandomAccessVectorValues(vectors, DIMENSION);
    }

    private static VectorFloat<?> randomVector(Random random) {
        var vector = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(DIMENSION);
        for (int i = 0; i < DIMENSION; i++) {
            vector.set(i, random.nextFloat() * 2.0f - 1.0f);
        }
        return vector;
    }
}
