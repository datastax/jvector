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

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.disk.SimpleWriter;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.HnswIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.PersistableGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.GraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.diversity.VamanaDiversityProvider;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.index.HnswRecipe;
import io.github.jbellis.jvector.index.Index;
import io.github.jbellis.jvector.index.IndexSearcher;
import io.github.jbellis.jvector.index.Indexes;
import io.github.jbellis.jvector.ivf.IvfIndex;
import io.github.jbellis.jvector.management.CompressionType;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.IntFunction;
import java.util.function.IntPredicate;
import java.util.function.IntUnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * A walkthrough of building, persisting and searching a graph index with the {@link Indexes} API.
 * Every section builds, writes and/or searches a real index and prints its recall@{@value #TOP_K}
 * against brute force, so the example doubles as a check that each path works end to end.
 * <p>
 * The API is designed so the common case is one line, and everything else is an optional
 * {@code withXxx} call on the same builder:
 * <pre>{@code
 * PersistableGraphIndex graph = Indexes.hnswBuilder(vectors, VectorSimilarityFunction.COSINE)
 *         .withCompressionType(CompressionType.PQ)   // optional: the builder trains and encodes PQ itself
 *         .buildAndPopulate();
 * }</pre>
 * Sections, in the order {@link #main} runs them:
 * <ol>
 *     <li><b>Quickstart</b> &mdash; build a complete index in one call with every default, search it
 *     in memory, and use it through the backing-agnostic {@link Index} handle.</li>
 *     <li><b>Tuning and compression</b> &mdash; the optional settings, starting from a recipe, and
 *     building with PQ- or BQ-compressed scores by naming a {@link CompressionType}.</li>
 *     <li><b>Writing to disk</b> &mdash; persisting a populated graph with the sequential writer
 *     ({@code OnDiskSequentialGraphIndexWriter}) and the parallel writer
 *     ({@code OnDiskParallelGraphIndexWriter}), including NVQ-compressed vectors, then loading and
 *     searching the result.</li>
 *     <li><b>Incremental construction</b> &mdash; inserting from several threads while searching,
 *     deleting, and writing a graph with deleted nodes.</li>
 *     <li><b>Building from a score provider</b> &mdash; for callers that score with their own
 *     {@link BuildScoreProvider}, and rescoring a graph mid-build.</li>
 *     <li><b>Continuing a saved graph</b> &mdash; reloading a mutable graph from disk and appending
 *     new vectors to it.</li>
 *     <li><b>Search options</b> &mdash; reranking depth, filtered search, and resuming a search.</li>
 *     <li><b>Validation, recipes and IVF</b> &mdash; what the builders reject, and the parts of the
 *     API that are scaffolding for now.</li>
 * </ol>
 */
public class IndexApiExample {
    private static final int VECTOR_COUNT = 20_000;
    private static final int DIMENSION = 64;
    private static final int QUERY_COUNT = 50;
    private static final int TOP_K = 10;
    /** Vectors are drawn around this many random centroids; uniformly random vectors have no structure to index. */
    private static final int CLUSTER_COUNT = 100;
    /**
     * How many candidates a search collects before returning the best {@link #TOP_K}. With an
     * approximate score function these are reranked with higher-precision scores; with exact scores
     * alone it simply widens the search.
     */
    private static final int RERANK_K = 3 * TOP_K;
    private static final VectorSimilarityFunction SIMILARITY_FUNCTION = VectorSimilarityFunction.EUCLIDEAN;

    public static void main(String[] args) throws IOException {
        VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        Random random = new Random(42);
        System.out.printf("Generating %,d %d-dimensional vectors and %d queries...%n",
                VECTOR_COUNT, DIMENSION, QUERY_COUNT);
        List<VectorFloat<?>> centroids = randomVectors(vts, CLUSTER_COUNT, DIMENSION, random);
        Dataset ds = new Dataset(clusteredVectors(vts, centroids, VECTOR_COUNT, random),
                                 clusteredVectors(vts, centroids, QUERY_COUNT, random));

        Path workDir = Files.createTempDirectory("jvector-index-api-example");
        try {
            PersistableGraphIndex graph = section1Quickstart(ds);
            section2TuningAndCompression(ds);
            Path onDiskPath = section3WritingToDisk(ds, graph, workDir);
            section4IncrementalBuild(ds, workDir);
            section5ScoreProviderBuild(ds);
            section6ContinuingASavedGraph(ds, workDir);
            section7SearchOptions(ds, onDiskPath);
            section8ValidationRecipesAndIvf(ds);
        } finally {
            try (Stream<Path> files = Files.list(workDir)) {
                for (Path p : (Iterable<Path>) files::iterator) {
                    Files.deleteIfExists(p);
                }
            }
            Files.deleteIfExists(workDir);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // 1. Quickstart
    // ---------------------------------------------------------------------------------------------

    /**
     * The whole build in one call. {@code Indexes.hnswBuilder} picks the backing type first, so only
     * graph/HNSW settings exist on the builder, and every setting has a sensible default.
     */
    private static PersistableGraphIndex section1Quickstart(Dataset ds) throws IOException {
        header("1. Quickstart: build and populate in one call");

        long start = System.nanoTime();
        PersistableGraphIndex graph;
        // The builder is Closeable: closing it releases per-thread scratch space it used while
        // inserting. The graph it built is unaffected and stays usable.
        try (HnswIndexBuilder builder = Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION)) {
            graph = builder.buildAndPopulate();
        }
        System.out.printf("Built %,d nodes in %.1fs%n", graph.size(0), seconds(start));

        // graph.searcher() returns a GraphSearcher, which must be closed. Here every candidate is
        // scored exactly against the in-memory vectors.
        searchInMemory(ds, graph, "in memory, exact scores");

        // A PersistableGraphIndex is a GraphIndex, which is an Index: code that doesn't care which
        // backing it holds can work with Index alone.
        describe(graph);
        return graph;
    }

    /** What shared code that only holds an {@link Index} looks like. */
    private static void describe(Index index) throws IOException {
        System.out.printf("As a generic Index: %s, %,d bytes on heap%n",
                index.getClass().getSimpleName(), index.ramBytesUsed());
        // IndexSearcher is Closeable, so code holding only an Index can still release its searcher.
        try (IndexSearcher searcher = index.searcher()) {
            if (index instanceof GraphIndex) {
                GraphIndex graph = (GraphIndex) index;
                System.out.printf("  narrowed to GraphIndex: dimension=%d, maxDegree=%d, hierarchical=%s; searcher is a %s%n",
                        graph.getDimension(), graph.maxDegree(), graph.isHierarchical(),
                        searcher.getClass().getSimpleName());
            } else if (index instanceof IvfIndex) {
                System.out.println("  narrowed to IvfIndex (unreachable until IVF is implemented)");
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // 2. Tuning and compression
    // ---------------------------------------------------------------------------------------------

    /**
     * Every setting is optional. Compression is one setting too: name a {@link CompressionType} and
     * the builder trains the quantizer and encodes the vectors itself, then builds the graph with the
     * cheaper compressed scores. There is no codebook or encoding step for the caller to run.
     */
    private static void section2TuningAndCompression(Dataset ds) throws IOException {
        header("2. Tuning and compression");

        // The settings, shown with their default values. applyRecipe(DEFAULT) restates the same
        // defaults; it is the starting point that future recipes (HIGH_RECALL, ...) will tune.
        long start = System.nanoTime();
        PersistableGraphIndex tuned;
        try (HnswIndexBuilder builder = Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION)
                .applyRecipe(HnswRecipe.DEFAULT)
                .withMaxDegree(32)               // edges per node; withMaxDegrees(List) sets one per layer
                .withBeamWidth(100)              // search width while inserting
                .withNeighborOverflow(1.2f)      // temporary extra edges allowed while inserting
                .withAlpha(1.2f)                 // > 1 keeps some longer edges for better connectivity
                .withAddHierarchy(true)          // HNSW-style upper layers
                .withRefineFinalGraph(true)) {   // a second pass over every node during cleanup
            tuned = builder.buildAndPopulate();
        }
        System.out.printf("Tuned build: %,d nodes in %.1fs%n", tuned.size(0), seconds(start));

        // PQ: product quantization, sized from GraphIndexBuilderConfig (by default one subspace per
        // 8 dimensions, 256 centroids each). Training and encoding happen inside buildAndPopulate().
        start = System.nanoTime();
        PersistableGraphIndex pqGraph;
        try (HnswIndexBuilder builder = Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION)
                .withCompressionType(CompressionType.PQ)) {
            pqGraph = builder.buildAndPopulate();
        }
        System.out.printf("PQ-scored build: %,d nodes in %.1fs%n", pqGraph.size(0), seconds(start));

        // BQ: binary quantization, one bit per dimension. Much cheaper still, and coarser.
        start = System.nanoTime();
        PersistableGraphIndex bqGraph;
        try (HnswIndexBuilder builder = Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION)
                .withCompressionType(CompressionType.BQ)) {
            bqGraph = builder.buildAndPopulate();
        }
        System.out.printf("BQ-scored build: %,d nodes in %.1fs%n", bqGraph.size(0), seconds(start));

        // Compression only affects how the graph was built: the result is an ordinary graph, searched
        // here with exact scores. The executors used for building and for PQ/BQ encoding can be set
        // with withSimdExecutor and withParallelExecutor.
        searchInMemory(ds, tuned, "tuned build, exact search");
        searchInMemory(ds, pqGraph, "PQ-scored build, exact search");
        searchInMemory(ds, bqGraph, "BQ-scored build, exact search");
    }

    /** Searches an in-memory graph, scoring every candidate exactly against the dataset's vectors. */
    private static void searchInMemory(Dataset ds, GraphIndex graph, String label) throws IOException {
        try (GraphSearcher searcher = graph.searcher()) {
            report(label, recall(ds, searcher, RERANK_K,
                    q -> DefaultSearchScoreProvider.exact(q, SIMILARITY_FUNCTION, ds.ravv)));
        }
    }

    // ---------------------------------------------------------------------------------------------
    // 3. Writing to disk
    // ---------------------------------------------------------------------------------------------

    /**
     * A populated graph is a {@link PersistableGraphIndex}, which hands out a writer builder for each
     * on-disk writer. All of them produce the same format, loaded with {@link OnDiskGraphIndex#load}:
     * <ul>
     *     <li>{@code getWriterBuilder(IndexWriter)}: the sequential writer. One forward pass that never
     *     seeks, so it works on append-only outputs (object storage, a Lucene {@code IndexOutput}).
     *     The caller owns the output.</li>
     *     <li>{@code getParallelWriterBuilder(Path)}: the parallel writer. Encodes node records on worker
     *     threads and writes them asynchronously; worthwhile when encoding features (e.g. NVQ)
     *     dominates.</li>
     *     <li>{@code getWriterBuilder(Path)}: the single-threaded random-access writer, which writes the
     *     records in order and then seeks back to fill in the header.</li>
     * </ul>
     * Each node record carries the {@link Feature}s added with {@code with(...)}; {@code write} takes,
     * for each feature, a function from node to that node's state.
     *
     * @return the path of the sequentially written graph, searched again in section 7
     */
    private static Path section3WritingToDisk(Dataset ds, PersistableGraphIndex graph, Path workDir) throws IOException {
        header("3. Writing to disk");

        // Full-precision vectors stored inline with each node, used to rerank results exactly.
        Map<FeatureId, IntFunction<Feature.State>> inlineVectors = Feature.singleStateFactory(
                FeatureId.INLINE_VECTORS, node -> new InlineVectors.State(ds.ravv.getVector(node)));

        // --- Sequential writer. SimpleWriter is a plain file; any IndexWriter works. Closing the graph
        // writer does not close the output, so the output gets its own try-with-resources.
        Path sequentialPath = workDir.resolve("sequential.graph");
        try (SimpleWriter out = new SimpleWriter(sequentialPath);
             GraphIndexWriter writer = graph.getWriterBuilder(out)
                     .with(new InlineVectors(DIMENSION))
                     .build()) {
            writer.write(inlineVectors);
        }
        searchOnDisk(ds, sequentialPath, "sequential writer, inline vectors");

        // --- Parallel writer. Worker threads default to the number of available processors.
        Path parallelPath = workDir.resolve("parallel.graph");
        try (GraphIndexWriter writer = graph.getParallelWriterBuilder(parallelPath)
                .with(new InlineVectors(DIMENSION))
                .withParallelWorkerThreads(0)        // optional: 0 = all available processors
                .build()) {
            writer.write(inlineVectors);
        }
        searchOnDisk(ds, parallelPath, "parallel writer, inline vectors");

        // --- Parallel writer with NVQ: vectors stored roughly 4x smaller, so reranking reads less from
        // disk at the cost of approximate final scores. NVQ is computed from the vectors at write
        // time; each node is encoded on the writer's worker threads.
        NVQuantization nvq = NVQuantization.compute(ds.ravv, 2);
        Path nvqPath = workDir.resolve("nvq.graph");
        try (GraphIndexWriter writer = graph.getParallelWriterBuilder(nvqPath)
                .with(new NVQ(nvq))
                .build()) {
            writer.write(Feature.singleStateFactory(FeatureId.NVQ_VECTORS,
                    node -> new NVQ.State(nvq.encode(ds.ravv.getVector(node)))));
        }
        searchOnDisk(ds, nvqPath, "parallel writer, NVQ vectors");

        return sequentialPath;
    }

    /**
     * Loads a graph written in this section and searches it. The searcher's view reads the graph from
     * disk, and its reranker scores with whichever vectors the graph stores (inline or NVQ).
     */
    private static void searchOnDisk(Dataset ds, Path path, String label) throws IOException {
        try (ReaderSupplier rs = ReaderSupplierFactory.open(path);
             OnDiskGraphIndex onDisk = OnDiskGraphIndex.load(rs);
             GraphSearcher searcher = onDisk.searcher()) {
            report(String.format("%s (%,d bytes)", label, Files.size(path)), recall(ds, searcher, RERANK_K,
                    q -> new DefaultSearchScoreProvider(scoringView(searcher).rerankerFor(q, SIMILARITY_FUNCTION))));
        }
    }

    // ---------------------------------------------------------------------------------------------
    // 4. Incremental construction
    // ---------------------------------------------------------------------------------------------

    /**
     * For vectors that arrive over time, {@code build()} returns the empty graph and
     * {@code addGraphNode} inserts one node at a time. Inserts are thread-safe and the graph is
     * searchable while they run. Deleted nodes are hidden from searches immediately and removed by
     * {@code cleanup()}, which must run once the inserts and deletes are done, and before writing.
     */
    private static void section4IncrementalBuild(Dataset ds, Path workDir) throws IOException {
        header("4. Incremental construction");
        int n = ds.ravv.size();

        // Delete every 100th node, skipping the true nearest neighbors of the queries so that recall
        // stays comparable with the other sections.
        Set<Integer> trueNeighbors = new HashSet<>();
        ds.groundTruth.forEach(trueNeighbors::addAll);
        Set<Integer> deleted = IntStream.range(0, n).filter(i -> i % 100 == 0 && !trueNeighbors.contains(i))
                .boxed().collect(Collectors.toSet());

        try (HnswIndexBuilder builder = Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION)) {
            PersistableGraphIndex graph = builder.build();
            ExecutorService pool = Executors.newFixedThreadPool(5);
            try {
                AtomicInteger next = new AtomicInteger();
                AtomicLong bytesAdded = new AtomicLong();
                AtomicBoolean inserting = new AtomicBoolean(true);
                AtomicInteger searches = new AtomicInteger();

                // Four inserters. addGraphNode returns an estimate of the heap the insert used.
                List<Future<?>> inserters = new ArrayList<>();
                for (int t = 0; t < 4; t++) {
                    inserters.add(pool.submit(() -> {
                        int node;
                        while ((node = next.getAndIncrement()) < n) {
                            bytesAdded.addAndGet(builder.addGraphNode(node, ds.ravv.getVector(node)));
                            if (deleted.contains(node)) {
                                builder.markNodeDeleted(node);
                            }
                        }
                    }));
                }
                // One thread searching the graph while it grows. Take a new searcher per search, so
                // each sees the nodes added since the last one.
                Future<?> searcher = pool.submit(() -> {
                    Random random = new Random(0);
                    while (inserting.get()) {
                        try (GraphSearcher s = graph.searcher()) {
                            VectorFloat<?> q = ds.queries.get(random.nextInt(ds.queries.size()));
                            s.search(DefaultSearchScoreProvider.exact(q, SIMILARITY_FUNCTION, ds.ravv), TOP_K, Bits.ALL);
                            searches.incrementAndGet();
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    }
                });
                awaitAll(inserters);
                inserting.set(false);
                awaitAll(List.of(searcher));
                System.out.printf("Inserted %,d nodes from 4 threads (%,d bytes reported) while running %,d searches%n",
                        n, bytesAdded.get(), searches.get());
            } finally {
                pool.shutdownNow();
            }

            // Remove the deleted nodes and finish the graph.
            builder.cleanup();
            System.out.printf("After cleanup: %,d nodes (%,d deleted)%n", graph.size(0), deleted.size());
            searchInMemory(ds, graph, "incremental build, in memory");

            // Deletes leave holes in the graph's ordinals. By default the writers renumber the
            // remaining nodes 0..size-1 on disk (sequentialRenumbering below is that default, computed
            // here only so the example can map search results back to the original ordinals). Pass
            // withMap or withMapper to choose a different numbering, e.g. one matching your row ids.
            Map<Integer, Integer> graphToDisk = OnDiskGraphIndexWriter.sequentialRenumbering(graph);
            Path path = workDir.resolve("incremental.graph");
            try (SimpleWriter out = new SimpleWriter(path);
                 GraphIndexWriter writer = graph.getWriterBuilder(out)
                         .with(new InlineVectors(DIMENSION))
                         .build()) {
                // The feature functions receive graph ordinals, not on-disk ones.
                writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                        node -> new InlineVectors.State(ds.ravv.getVector(node))));
            }
            IntUnaryOperator diskToGraph = invert(graphToDisk);
            try (ReaderSupplier rs = ReaderSupplierFactory.open(path);
                 OnDiskGraphIndex onDisk = OnDiskGraphIndex.load(rs);
                 GraphSearcher searcher = onDisk.searcher()) {
                report(String.format("incremental build on disk, %,d nodes", onDisk.size(0)),
                        recall(ds, ds.groundTruth, searcher, RERANK_K, Bits.ALL, diskToGraph,
                                q -> new DefaultSearchScoreProvider(scoringView(searcher).rerankerFor(q, SIMILARITY_FUNCTION))));
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // 5. Building from a score provider
    // ---------------------------------------------------------------------------------------------

    /**
     * {@code Indexes.hnswBuilder(BuildScoreProvider, dimension)} is for callers that already score
     * vectors their own way, or that stream vectors in and never hold them all in one
     * {@link RandomAccessVectorValues}. The builder then has no vectors of its own: populate it with
     * {@code populateGraph(ravv)} or {@code addGraphNode}. For everything else, prefer the vector
     * builder from the earlier sections and {@code withCompressionType}.
     */
    private static void section5ScoreProviderBuild(Dataset ds) throws IOException {
        header("5. Building from a score provider, and rescoring");
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ds.ravv, SIMILARITY_FUNCTION);

        try (HnswIndexBuilder builder = Indexes.hnswBuilder(bsp, DIMENSION)) {
            // Insert the first half, then rescore and insert the rest.
            int half = ds.ravv.size() / 2;
            IntStream.range(0, half).parallel().forEach(i -> builder.addGraphNode(i, ds.ravv.getVector(i)));

            // rescore() copies the graph with every edge re-scored by a new provider, e.g. after
            // refining a PQ codebook partway through a build, and returns a builder that continues
            // with it. The copy keeps the original's settings; the original builder is unchanged.
            BuildScoreProvider refined = BuildScoreProvider.randomAccessScoreProvider(ds.ravv, SIMILARITY_FUNCTION);
            try (HnswIndexBuilder rescored = HnswIndexBuilder.rescore(builder, refined)) {
                IntStream.range(half, ds.ravv.size()).parallel()
                        .forEach(i -> rescored.addGraphNode(i, ds.ravv.getVector(i)));
                rescored.cleanup();
                PersistableGraphIndex graph = rescored.getGraph();
                System.out.printf("Built %,d nodes: %,d before the rescore, %,d after%n",
                        graph.size(0), half, graph.size(0) - half);
                searchInMemory(ds, graph, "score-provider build with a mid-build rescore");
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // 6. Continuing a saved graph
    // ---------------------------------------------------------------------------------------------

    /**
     * Adding vectors to an existing graph instead of rebuilding it, e.g. merging a smaller segment
     * into a larger one. The on-disk search format drops the neighbor scores a builder needs, so the
     * graph to continue is saved with {@code OnHeapGraphIndex.save} and reloaded with
     * {@code OnHeapGraphIndex.load}, then handed to {@code withExistingGraph}.
     */
    @SuppressWarnings("deprecation") // OnHeapGraphIndex.save/load are deprecated and experimental
    private static void section6ContinuingASavedGraph(Dataset ds, Path workDir) throws IOException {
        header("6. Continuing a saved graph");

        // The base graph: the first three quarters of the vectors.
        int baseCount = ds.ravv.size() * 3 / 4;
        var baseRavv = new ListRandomAccessVectorValues(ds.vectors.subList(0, baseCount), DIMENSION);
        Path savedPath = workDir.resolve("base.onheap");
        try (HnswIndexBuilder builder = Indexes.hnswBuilder(baseRavv, SIMILARITY_FUNCTION);
             SimpleWriter out = new SimpleWriter(savedPath)) {
            // save() is specific to the in-memory graph, so this is the one place a cast is needed.
            ((OnHeapGraphIndex) builder.buildAndPopulate()).save(out);
        }
        System.out.printf("Saved a %,d-node graph (%,d bytes)%n", baseCount, Files.size(savedPath));

        // load() needs the diversity provider the graph will be extended with. It must score the same
        // way as the builder below, and use the builder's alpha (1.2 by default).
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ds.ravv, SIMILARITY_FUNCTION);
        OnHeapGraphIndex loaded;
        try (ReaderSupplier rs = ReaderSupplierFactory.open(savedPath);
             var reader = rs.get()) {
            loaded = OnHeapGraphIndex.load(reader, DIMENSION, 1.2, new VamanaDiversityProvider(bsp, 1.2f));
        }

        // The builder's vectors are a superset: the existing nodes at their original ordinals, then
        // the new ones. buildAndPopulate() adds only the vectors past the end of the existing graph.
        long start = System.nanoTime();
        PersistableGraphIndex extended;
        try (HnswIndexBuilder builder = Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION)
                .withExistingGraph(loaded)) {
            extended = builder.buildAndPopulate();
        }
        System.out.printf("Appended %,d nodes in %.1fs, for %,d in total%n",
                extended.size(0) - baseCount, seconds(start), extended.size(0));
        searchInMemory(ds, extended, "reloaded and extended, in memory");

        // The existing graph fixes the max degrees and hierarchy, so setting them as well is an error.
        System.out.println("Setting the graph shape along with an existing graph is rejected:");
        expectFailure(() -> Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION)
                .withExistingGraph(loaded)
                .withMaxDegree(16)
                .build());
    }

    // ---------------------------------------------------------------------------------------------
    // 7. Search options
    // ---------------------------------------------------------------------------------------------

    /** Searching the on-disk graph from section 3 with the options a {@link GraphSearcher} offers. */
    private static void section7SearchOptions(Dataset ds, Path path) throws IOException {
        header("7. Search options (on the sequentially written graph from section 3)");
        try (ReaderSupplier rs = ReaderSupplierFactory.open(path);
             OnDiskGraphIndex onDisk = OnDiskGraphIndex.load(rs);
             GraphSearcher searcher = onDisk.searcher()) {
            ScoreProviderFactory exact = q -> new DefaultSearchScoreProvider(
                    scoringView(searcher).rerankerFor(q, SIMILARITY_FUNCTION));

            // rerankK: how many candidates to collect before keeping the best TOP_K. Wider is more
            // accurate and slower.
            for (int rerankK : new int[] {TOP_K, RERANK_K, 10 * TOP_K}) {
                report("rerankK = " + rerankK, recall(ds, searcher, rerankK, exact));
            }

            // Filtered search: only nodes for which the Bits return true are returned. Here, even
            // ordinals; ground truth is computed over the same subset.
            Bits evenOnly = node -> node % 2 == 0;
            List<Set<Integer>> evenTruth = ds.queries.stream()
                    .map(q -> bruteForce(ds, q, TOP_K, node -> node % 2 == 0))
                    .collect(Collectors.toList());
            report("filtered to even ordinals", recall(ds, evenTruth, searcher, RERANK_K, evenOnly, node -> node, exact));

            // Resuming: fetch the next results of the previous search without starting over.
            VectorFloat<?> query = ds.queries.get(0);
            SearchResult first = searcher.search(exact.forQuery(query), TOP_K, RERANK_K, 0.0f, 0.0f, Bits.ALL);
            SearchResult more = searcher.resume(TOP_K, RERANK_K);
            System.out.printf("  first search: %d results, best score %.4f; resumed: %d more, best score %.4f%n",
                    first.getNodes().length, first.getNodes()[0].score,
                    more.getNodes().length, more.getNodes()[0].score);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // 8. Validation, recipes and IVF
    // ---------------------------------------------------------------------------------------------

    private static void section8ValidationRecipesAndIvf(Dataset ds) {
        header("8. Validation, recipes and IVF");

        System.out.println("Out-of-range settings are rejected when the graph is built:");
        expectFailure(() -> Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION).withBeamWidth(0).build());
        expectFailure(() -> Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION).withNeighborOverflow(0.5f).build());

        // Several max degrees means one per layer, which needs the hierarchy.
        System.out.println("Per-layer max degrees need the hierarchy:");
        expectFailure(() -> Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION)
                .withMaxDegrees(List.of(32, 16))
                .withAddHierarchy(false)
                .build());

        // A score-provider builder has no vectors to populate from.
        System.out.println("buildAndPopulate() needs a builder created from vectors:");
        expectFailure(() -> Indexes.hnswBuilder(BuildScoreProvider.randomAccessScoreProvider(ds.ravv, SIMILARITY_FUNCTION), DIMENSION)
                .buildAndPopulate());

        // Recipes are experimental: DEFAULT is defined (section 2 uses it), the rest are named but
        // have no values yet.
        System.out.println("Recipes other than DEFAULT are not defined yet:");
        expectFailure(() -> Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION).applyRecipe(HnswRecipe.HIGH_RECALL));

        // IVF: the builder validates the inputs every backing shares, then refuses because IVF itself
        // isn't implemented yet. Code written against Index/IvfIndex compiles today.
        System.out.println("IVF validates its inputs, then refuses to build:");
        expectFailure(() -> Indexes.ivfBuilder().build());
        expectFailure(() -> Indexes.ivfBuilder()
                .withVectorValues(ds.ravv)
                .withSimilarityFunction(SIMILARITY_FUNCTION)
                .build());
    }

    // ---------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------

    /** The vectors, queries, and each query's true {@link #TOP_K} nearest neighbors. */
    private static final class Dataset {
        final List<VectorFloat<?>> vectors;
        final List<VectorFloat<?>> queries;
        final RandomAccessVectorValues ravv;
        final List<Set<Integer>> groundTruth;

        Dataset(List<VectorFloat<?>> vectors, List<VectorFloat<?>> queries) {
            this.vectors = vectors;
            this.queries = queries;
            this.ravv = new ListRandomAccessVectorValues(vectors, DIMENSION);
            this.groundTruth = queries.stream()
                    .map(q -> bruteForce(this, q, TOP_K, node -> true))
                    .collect(Collectors.toList());
        }
    }

    /** Creates the score provider for one query. */
    @FunctionalInterface
    private interface ScoreProviderFactory {
        SearchScoreProvider forQuery(VectorFloat<?> query);
    }

    /** Mean recall@{@value #TOP_K} over the dataset's queries. */
    private static double recall(Dataset ds, GraphSearcher searcher, int rerankK, ScoreProviderFactory ssp) {
        return recall(ds, ds.groundTruth, searcher, rerankK, Bits.ALL, node -> node, ssp);
    }

    /**
     * Mean recall@{@value #TOP_K} against {@code truth}, searching only {@code acceptOrds} and mapping
     * each result through {@code toGraphOrdinal} first (for graphs written with renumbered ordinals).
     */
    private static double recall(Dataset ds, List<Set<Integer>> truth, GraphSearcher searcher, int rerankK,
                                 Bits acceptOrds, IntUnaryOperator toGraphOrdinal, ScoreProviderFactory ssp) {
        int hits = 0;
        for (int i = 0; i < ds.queries.size(); i++) {
            SearchResult result = searcher.search(ssp.forQuery(ds.queries.get(i)), TOP_K, rerankK, 0.0f, 0.0f, acceptOrds);
            for (SearchResult.NodeScore ns : result.getNodes()) {
                if (truth.get(i).contains(toGraphOrdinal.applyAsInt(ns.node))) {
                    hits++;
                }
            }
        }
        return hits / (double) (ds.queries.size() * TOP_K);
    }

    private static Set<Integer> bruteForce(Dataset ds, VectorFloat<?> q, int k, IntPredicate accept) {
        return IntStream.range(0, ds.vectors.size()).filter(accept).boxed()
                .sorted(Comparator.comparingDouble(node -> -SIMILARITY_FUNCTION.compare(q, ds.vectors.get(node))))
                .limit(k)
                .collect(Collectors.toSet());
    }

    /**
     * The view of an on-disk graph is a {@link GraphIndex.ScoringView}, which builds score functions
     * from the vectors stored in the graph.
     */
    private static GraphIndex.ScoringView scoringView(GraphSearcher searcher) {
        return (GraphIndex.ScoringView) searcher.getView();
    }

    private static IntUnaryOperator invert(Map<Integer, Integer> oldToNew) {
        Map<Integer, Integer> newToOld = new HashMap<>();
        oldToNew.forEach((oldOrd, newOrd) -> newToOld.put(newOrd, oldOrd));
        return newOrd -> newToOld.get(newOrd);
    }

    private static void awaitAll(List<Future<?>> futures) {
        for (Future<?> f : futures) {
            try {
                f.get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            } catch (ExecutionException e) {
                throw new RuntimeException(e.getCause());
            }
        }
    }

    private static double seconds(long startNanos) {
        return (System.nanoTime() - startNanos) / 1e9;
    }

    private static void report(String label, double recall) {
        System.out.printf("  %-60s recall@%d = %.3f%n", label, TOP_K, recall);
    }

    private static void header(String title) {
        System.out.println();
        System.out.println("== " + title);
    }

    private static void expectFailure(Runnable action) {
        try {
            action.run();
            throw new AssertionError("expected an exception");
        } catch (IllegalArgumentException | IllegalStateException | UnsupportedOperationException e) {
            System.out.println("  -> " + e.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }

    /** Unit vectors scattered around the given centroids. */
    private static List<VectorFloat<?>> clusteredVectors(VectorTypeSupport vts, List<VectorFloat<?>> centroids,
                                                         int count, Random random) {
        List<VectorFloat<?>> vectors = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            VectorFloat<?> centroid = centroids.get(random.nextInt(centroids.size()));
            VectorFloat<?> v = vts.createFloatVector(centroid.length());
            for (int d = 0; d < v.length(); d++) {
                v.set(d, centroid.get(d) + (float) random.nextGaussian() * 0.1f);
            }
            VectorUtil.l2normalize(v);
            vectors.add(v);
        }
        return vectors;
    }

    private static List<VectorFloat<?>> randomVectors(VectorTypeSupport vts, int count, int dimension, Random random) {
        List<VectorFloat<?>> vectors = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            VectorFloat<?> v = vts.createFloatVector(dimension);
            for (int d = 0; d < dimension; d++) {
                v.set(d, random.nextFloat() * 2 - 1);
            }
            VectorUtil.l2normalize(v);
            vectors.add(v);
        }
        return vectors;
    }
}
