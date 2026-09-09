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

package io.github.jbellis.jvector.example.tutorial;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.MultiGraphSearcher;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.ShardedSearchResult;
import io.github.jbellis.jvector.graph.disk.GraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.GraphIndexWriterTypes;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

// Demonstrates MultiGraphSearcher, described in docs/multi-index-search.md: fanning one query out
// across several independent on-disk graphs and merging the results, the way an embedding system
// like Cassandra or OpenSearch would when a logical dataset is sharded into per-segment indexes.
//
// Two things are demonstrated in turn: first the full single-query API (proportional initial sizing,
// the internal adaptive resume loop, and the caller-facing resume(int) call) against one illustrative
// query; then how a real embedding system actually gets parallelism out of MultiGraphSearcher once it's
// serving many concurrent client queries -- not by fanning one query's shards out across an executor
// (MultiGraphSearcher's per-call state and each shard's GraphSearcher/View aren't safe to share across
// concurrently-running searches), but by running queries concurrently with each worker thread owning
// its own MultiGraphSearcher, exactly like MultiShardBench/BenchYAML's own throughput measurement does.
public class MultiSearchExample {
    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.COSINE;
    private static final int DIMENSION = 8;

    public static void main(String[] args) throws Exception {
        // Simulate an embedding system whose data has been flushed into four independent on-disk
        // segments of very different sizes -- exactly the scenario docs/multi-index-search.md was
        // written for. The sizes are deliberately skewed, and (see buildAndWriteShard below) the
        // single best match to our query is planted in the smallest one, so that MultiGraphSearcher's
        // proportional initial sizing shortchanges it and its internal adaptive resume has real work
        // to do recovering it before we even get to the caller-facing resume() call below.
        int[] shardSizes = {800, 400, 150, 30};
        int needleShard = shardSizes.length - 1;

        Random random = new Random(42);
        VectorFloat<?> query = randomUnitVector(random, DIMENSION);

        System.out.println("Building " + shardSizes.length + " on-disk shards: " + Arrays.toString(shardSizes)
                + " vectors each (shard " + needleShard + " holds the true best match)");
        List<ShardHandle> shardHandles = new ArrayList<>();
        try {
            for (int s = 0; s < shardSizes.length; s++) {
                shardHandles.add(buildAndWriteShard(s, shardSizes[s], s == needleShard, query, random));
            }

            List<ImmutableGraphIndex> shards = shardHandles.stream()
                    .map(h -> (ImmutableGraphIndex) h.graph)
                    .collect(Collectors.toList());
            List<SearchScoreProvider> providers = shardHandles.stream()
                    .map(h -> DefaultSearchScoreProvider.exact(query, SIMILARITY, h.ravv))
                    .collect(Collectors.toList());

            int topK = 5;
            try (MultiGraphSearcher searcher = MultiGraphSearcher.builder(shards).build()) {
                // --- Search 1: a single call to search(), no caller-driven resume yet ---
                System.out.println("\n=== search(): merged top-" + topK + " across all " + shards.size() + " shards ===");
                ShardedSearchResult initial = searcher.search(providers, topK, topK * 2);
                printResult(initial, shardSizes);
                System.out.println("(rounds > 1 above means the internal adaptive resume already kicked in --");
                System.out.println(" some shard's proportional initial ask undershot and was topped up automatically)");

                // --- Search 2: the caller-facing resume(), for a pull-driven consumer ---
                // A real caller (e.g. Cassandra reconciling PrimaryKeyWithSortKey candidates against
                // live rows) often finds that some of these results don't pan out once it goes to
                // fetch the actual row -- overwritten, tombstoned, or otherwise stale. Simulate that
                // here by pretending two specific ranks turned out to be stale, and use resume() to
                // backfill without restarting the whole multi-shard search from scratch.
                var staleRanks = Set.of(1, 3);
                var staleIdentities = staleRanks.stream()
                        .map(rank -> shardNodeKey(initial.getNodes()[rank]))
                        .collect(Collectors.toSet());
                System.out.printf("%nDownstream filtering finds ranks %s stale (e.g. overwritten rows); "
                        + "%d of %d results are still valid%n", staleRanks, topK - staleRanks.size(), topK);

                System.out.println("\n=== resume(" + staleRanks.size() + "): grow to top-" + (topK + staleRanks.size())
                        + " without restarting the search ===");
                ShardedSearchResult grown = searcher.resume(staleRanks.size());
                printResult(grown, shardSizes);

                var finalValid = Arrays.stream(grown.getNodes())
                        .filter(n -> !staleIdentities.contains(shardNodeKey(n)))
                        .limit(topK)
                        .collect(Collectors.toList());
                System.out.println("\nFinal " + topK + " valid results after re-filtering the grown batch:");
                for (var n : finalValid) {
                    System.out.printf("  shard=%d node=%-5d score=%.4f%n", n.shardIndex, n.node, n.score);
                }
            }

            demonstrateConcurrentQueries(shardHandles, shards, topK, random);
        } finally {
            for (ShardHandle handle : shardHandles) {
                handle.close();
            }
        }
    }

    /**
     * Shows how a real embedding system actually gets parallelism out of {@link MultiGraphSearcher}
     * once it's serving many concurrent client queries: not by fanning one query's shards out across an
     * executor, but by running queries concurrently -- {@code IntStream.range(0, n).parallel()} here,
     * a thread pool in a real server -- with each worker thread owning its own {@link MultiGraphSearcher}
     * via {@link ExplicitThreadLocal}, doing plain sequential per-shard search within that thread.
     * <p>
     * This mirrors exactly how {@code MultiShardBench}/{@code BenchYAML} measure multi-shard throughput.
     * An earlier version of both this example and that benchmark instead tried to parallelize a single
     * query's shard fan-out via a shared executor -- {@link MultiGraphSearcher} holds mutable per-call
     * search state and each shard's {@code GraphSearcher}/{@code View} isn't safe to share across
     * concurrently-running searches, so that only pays off when there's truly one query in flight at a
     * time; real query concurrency (many independent callers, each searching all shards) is what
     * actually keeps every core busy, and per-thread searcher instances are what make that safe.
     */
    private static void demonstrateConcurrentQueries(List<ShardHandle> shardHandles, List<ImmutableGraphIndex> shards,
                                                       int topK, Random random) throws Exception
    {
        int numQueries = 2000;
        List<VectorFloat<?>> queries = new ArrayList<>(numQueries);
        for (int i = 0; i < numQueries; i++) {
            queries.add(randomUnitVector(random, DIMENSION));
        }

        System.out.printf("%n=== Concurrent queries: %d independent client searches across all %d shards ===%n",
                numQueries, shards.size());
        System.out.println("(each worker thread owns its own MultiGraphSearcher via ExplicitThreadLocal; "
                + "common pool parallelism=" + java.util.concurrent.ForkJoinPool.getCommonPoolParallelism() + ")");

        Set<Long> threadsUsed = ConcurrentHashMap.newKeySet();
        try (ExplicitThreadLocal<MultiGraphSearcher> searcherPool =
                     ExplicitThreadLocal.withInitial(() -> MultiGraphSearcher.builder(shards).build())) {
            long start = System.nanoTime();
            IntStream.range(0, numQueries).parallel().forEach(i -> {
                threadsUsed.add(Thread.currentThread().getId());
                MultiGraphSearcher searcher = searcherPool.get();
                VectorFloat<?> q = queries.get(i);
                List<SearchScoreProvider> providers = shardHandles.stream()
                        .map(h -> DefaultSearchScoreProvider.exact(q, SIMILARITY, h.ravv))
                        .collect(Collectors.toList());
                searcher.search(providers, topK, topK * 2);
            });
            long elapsedNanos = System.nanoTime() - start;

            double qps = numQueries / (elapsedNanos / 1_000_000_000.0);
            System.out.printf("%,d queries in %.3f ms across %d worker threads: %,.1f qps%n",
                    numQueries, elapsedNanos / 1_000_000.0, threadsUsed.size(), qps);
        }
    }

    private static void printResult(ShardedSearchResult result, int[] shardSizes) {
        System.out.printf("rounds=%d visited=%d expanded=%d reranked=%d%n",
                result.getRoundsUsed(), result.getVisitedCount(), result.getExpandedCount(), result.getRerankedCount());
        for (var n : result.getNodes()) {
            System.out.printf("  shard=%d (size=%-4d) node=%-5d score=%.4f%n", n.shardIndex, shardSizes[n.shardIndex], n.node, n.score);
        }
    }

    /**
     * Packs (shardIndex, node) into one key, used to recognize the same result across the pre- and
     * post-resume batches -- ordinals are only unique within a shard, so shardIndex must be part of
     * the identity.
     */
    private static long shardNodeKey(ShardedSearchResult.NodeScore n) {
        return ((long) n.shardIndex << 32) | (n.node & 0xffffffffL);
    }

    /**
     * Builds one shard's graph in memory, writes it to a temporary on-disk file exactly as a real
     * embedding system would persist a segment, then re-opens it as a read-only {@link OnDiskGraphIndex}
     * -- so the search half of this example runs against genuinely independent on-disk indexes, not
     * shared in-memory state.
     */
    private static ShardHandle buildAndWriteShard(int shardIndex, int size, boolean plantNeedle,
                                                    VectorFloat<?> query, Random random) throws IOException
    {
        List<VectorFloat<?>> vectors = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            vectors.add(randomUnitVector(random, DIMENSION));
        }
        if (plantNeedle) {
            // ordinal 0 becomes an exact match for the query -- the true global best result.
            vectors.set(0, query);
        }

        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vectors, DIMENSION);
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, SIMILARITY);

        ImmutableGraphIndex heapGraph;
        try (GraphIndexBuilder graphBuilder = new GraphIndexBuilder(bsp, DIMENSION, 16, 100, 1.2f, 1.2f, true, true)) {
            heapGraph = graphBuilder.build(ravv);
        }

        Path path = Files.createTempFile("multisearch-example-shard" + shardIndex, null);
        try (GraphIndexWriter writer = GraphIndexWriter.getBuilderFor(GraphIndexWriterTypes.RANDOM_ACCESS_PARALLEL, heapGraph, path)
                .with(new InlineVectors(DIMENSION))
                .build())
        {
            writer.write(Map.of(FeatureId.INLINE_VECTORS, nodeId -> new InlineVectors.State(ravv.getVector(nodeId))));
        }

        ReaderSupplier readerSupplier = ReaderSupplierFactory.open(path);
        OnDiskGraphIndex onDiskGraph = OnDiskGraphIndex.load(readerSupplier);
        var view = onDiskGraph.getView();
        var onDiskRavv = (RandomAccessVectorValues) view;
        return new ShardHandle(onDiskGraph, onDiskRavv, view, readerSupplier, path);
    }

    private static VectorFloat<?> randomUnitVector(Random random, int dimension) {
        float[] v = new float[dimension];
        float normSquared = 0;
        for (int i = 0; i < dimension; i++) {
            v[i] = random.nextFloat() * 2 - 1;
            normSquared += v[i] * v[i];
        }
        float norm = (float) Math.sqrt(normSquared);
        for (int i = 0; i < dimension; i++) {
            v[i] /= norm;
        }
        return VTS.createFloatVector(v);
    }

    /** One shard's on-disk graph plus everything needed to close it cleanly. */
    private static final class ShardHandle implements Closeable {
        final OnDiskGraphIndex graph;
        final RandomAccessVectorValues ravv;
        private final ImmutableGraphIndex.View view;
        private final ReaderSupplier readerSupplier;
        private final Path path;

        ShardHandle(OnDiskGraphIndex graph, RandomAccessVectorValues ravv, ImmutableGraphIndex.View view,
                    ReaderSupplier readerSupplier, Path path)
        {
            this.graph = graph;
            this.ravv = ravv;
            this.view = view;
            this.readerSupplier = readerSupplier;
            this.path = path;
        }

        @Override
        public void close() throws IOException {
            view.close();
            readerSupplier.close();
            Files.deleteIfExists(path);
        }
    }
}
