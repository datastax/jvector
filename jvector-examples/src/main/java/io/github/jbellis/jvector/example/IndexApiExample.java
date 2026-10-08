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

import io.github.jbellis.jvector.disk.RandomAccessWriter;
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
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.diversity.VamanaDiversityProvider;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.index.HnswRecipe;
import io.github.jbellis.jvector.api.Index;
import io.github.jbellis.jvector.api.IndexSearcher;
import io.github.jbellis.jvector.index.Indexes;
import io.github.jbellis.jvector.ivf.IvfIndex;
import io.github.jbellis.jvector.management.CompressionType;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.MutablePQVectors;
import io.github.jbellis.jvector.quantization.NVQVectors;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
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
 * A walkthrough of building, persisting and searching graph indexes with the {@link Indexes} API,
 * organized around how JVector is used in production by Cassandra (SAI vector indexes) and by the
 * OpenSearch jVector plugin. Every section builds, writes and/or searches a real index and prints its
 * recall@{@value #TOP_K} against brute force, so the example doubles as a check that each path works end
 * to end.
 * <p>
 * The common case is one call, and everything else is an optional {@code withXxx} setting on the same
 * builder. Compression is one of those settings: name a {@link CompressionType} and the builder trains the
 * quantizer and encodes the vectors itself, and {@link HnswIndexBuilder#getCompressedVectors()} hands back
 * what it trained, ready to search with or write to disk:
 * <pre>{@code
 * var builder = Indexes.hnswBuilder(vectors, VectorSimilarityFunction.COSINE)
 *         .withCompressionType(CompressionType.PQ);
 * PersistableGraphIndex graph = builder.buildAndPopulate();
 * PQVectors pq = (PQVectors) builder.getCompressedVectors();
 * }</pre>
 * Persisting and searching have shortcuts too: {@code graph.writeTo(path, vectors)} writes the graph with
 * its vectors stored inline, and {@code searcher.search(query, topK, rerankK, similarityFunction, filter)} searches an
 * on-disk graph with the vectors it stores, using fused PQ codes when it has them.
 * <p>
 * Sections, in the order {@link #main} runs them:
 * <ol>
 *     <li><b>Quickstart</b> &mdash; a complete index in one call, searched in memory, and used through
 *     the backing-agnostic {@link Index} handle.</li>
 *     <li><b>Tuning</b> &mdash; the optional settings, starting from a recipe.</li>
 *     <li><b>Building with compressed scores</b> &mdash; {@code withCompressionType(PQ)} and {@code BQ},
 *     and searching in memory with the compressed vectors the builder trained (Cassandra compaction,
 *     OpenSearch flush and merge with quantization enabled).</li>
 *     <li><b>Writer types</b> &mdash; the {@code writeTo} shortcut, then the sequential, parallel and
 *     random-access writers, embedding a
 *     graph at an offset in a larger file (Cassandra), and rewriting a graph loaded from disk.</li>
 *     <li><b>Fused PQ on disk</b> &mdash; each node's record carries its neighbors' PQ codes
 *     (Cassandra's current format), written from the builder's own PQ.</li>
 *     <li><b>NVQ and fused PQ, parallel writer</b> &mdash; compressed vectors for reranking, encoded on
 *     the parallel writer's threads (Cassandra compaction with parallel writing).</li>
 *     <li><b>Separately stored PQ codes</b> &mdash; inline or NVQ vectors in the graph, PQ codes beside
 *     it, written sequentially (OpenSearch).</li>
 *     <li><b>Incremental construction</b> &mdash; inserting from several threads while searching,
 *     deleting, and writing a graph with deleted nodes (Cassandra memtables).</li>
 *     <li><b>Your own score provider, and a PQ rescore</b> &mdash; streaming vectors in with PQ codes
 *     you maintain, refining the codebook partway through (Cassandra compaction).</li>
 *     <li><b>Continuing a saved graph</b> &mdash; reloading a mutable graph and adding vectors to it
 *     (OpenSearch leading-segment merge).</li>
 *     <li><b>Search options</b> &mdash; reranking depth, filtered search, and resuming a search.</li>
 *     <li><b>Validation, recipes and IVF</b> &mdash; what the builders reject, and the parts of the API
 *     that are scaffolding for now.</li>
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
     * How many candidates a search collects before returning the best {@link #TOP_K}. With approximate
     * (compressed) scores these are reranked with higher-precision ones; with exact scores alone it
     * simply widens the search. Cassandra's "rerankless" mode uses {@link #TOP_K}.
     */
    private static final int RERANK_K = 3 * TOP_K;
    private static final VectorSimilarityFunction SIMILARITY_FUNCTION = VectorSimilarityFunction.EUCLIDEAN;
    /** NVQ sub-vectors: Cassandra's default ({@code JVectorVersionUtil.NUM_SUB_VECTORS}). */
    private static final int NVQ_SUB_VECTORS = 2;

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
            PersistableGraphIndex fullPrecision = section1Quickstart(ds);
            section2Tuning(ds);
            PqBuild pqBuild = section3CompressedBuild(ds);
            section4WriterTypes(ds, fullPrecision, workDir);
            Path fusedPath = section5FusedPq(ds, pqBuild, workDir);
            section6NvqAndFusedPq(ds, pqBuild, workDir);
            section7SeparatePqCodes(ds, pqBuild, workDir);
            section8IncrementalBuild(ds, workDir);
            section9ScoreProviderAndRescore(ds, workDir);
            section10ContinuingASavedGraph(ds, workDir);
            section11SearchOptions(ds, fusedPath);
            section12ValidationRecipesAndIvf(ds);
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
     * graph settings exist on the builder, and every setting has a default.
     */
    private static PersistableGraphIndex section1Quickstart(Dataset ds) throws IOException {
        header("1. Quickstart: build and populate in one call");

        long start = System.nanoTime();
        PersistableGraphIndex graph;
        // The builder is Closeable: closing it releases per-thread scratch space used while inserting.
        // The graph it built is unaffected.
        try (HnswIndexBuilder builder = Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION)) {
            graph = builder.buildAndPopulate();
        }
        System.out.printf("Built %,d nodes in %.1fs%n", graph.size(0), seconds(start));

        // Searching in memory, scoring every candidate exactly against the vectors.
        searchInMemoryExact(ds, graph, "in memory, exact scores");

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
    // 2. Tuning
    // ---------------------------------------------------------------------------------------------

    /** Every setting is optional; these are the defaults, set explicitly. */
    private static void section2Tuning(Dataset ds) throws IOException {
        header("2. Tuning");

        // applyRecipe(DEFAULT) restates the defaults; it is the starting point that future recipes
        // (HIGH_RECALL, ...) will tune. Settings after it override the recipe's values.
        long start = System.nanoTime();
        PersistableGraphIndex graph;
        try (HnswIndexBuilder builder = Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION)
                .applyRecipe(HnswRecipe.DEFAULT)
                .withMaxDegree(32)               // edges per node; withMaxDegrees(List) sets one per layer
                .withBeamWidth(100)              // search width while inserting
                .withNeighborOverflow(1.2f)      // temporary extra edges allowed while inserting
                .withAlpha(1.2f)                 // > 1 keeps some longer edges for better connectivity
                .withAddHierarchy(true)          // HNSW-style upper layers
                .withRefineFinalGraph(true)) {   // a second pass over every node during cleanup
            // withBuildExecutor and withMaintenanceExecutor choose the thread pools for building and for
            // compressing; they default to the physical-core pool and the common pool.
            graph = builder.buildAndPopulate();
        }
        System.out.printf("Built %,d nodes in %.1fs%n", graph.size(0), seconds(start));
        searchInMemoryExact(ds, graph, "tuned build, exact scores");
    }

    // ---------------------------------------------------------------------------------------------
    // 3. Building with compressed scores
    // ---------------------------------------------------------------------------------------------

    /** A graph built with PQ scores, and the PQ codes the builder trained for it. */
    private static final class PqBuild {
        final PersistableGraphIndex graph;
        final PQVectors pqVectors;

        PqBuild(PersistableGraphIndex graph, PQVectors pqVectors) {
            this.graph = graph;
            this.pqVectors = pqVectors;
        }

        ProductQuantization pq() {
            return pqVectors.getCompressor();
        }
    }

    /**
     * Construction scored with compressed vectors instead of the raw ones is much cheaper per
     * comparison. Name the compression with {@code withCompressionType} and the builder does the rest:
     * it trains the quantizer on the vectors, encodes them, and builds with the compressed scores. This
     * is what Cassandra does in compaction and OpenSearch on flush and merge once a segment is large
     * enough to quantize. Both then search with compressed scores and rerank the best candidates with
     * higher-precision ones, as shown here in memory and in sections 5-7 on disk.
     */
    private static PqBuild section3CompressedBuild(Dataset ds) throws IOException {
        header("3. Building with compressed scores");

        // PQ: product quantization. Each vector is split into subspaces, each encoded in one byte. The
        // training settings default to Cassandra's: a code size that depends on the dimension (32 bytes for
        // these 64-dimension vectors), no centering, and unweighted training. withPqSubspaces,
        // withPqGlobalCentering and withPqAnisotropicThreshold override them.
        long start = System.nanoTime();
        PersistableGraphIndex pqGraph;
        PQVectors pqVectors;
        try (HnswIndexBuilder builder = Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION)
                .withCompressionType(CompressionType.PQ)) {
            pqGraph = builder.buildAndPopulate();
            // The PQ codes the graph was built with, and through getCompressor() the codebook. Reuse
            // them to search and to write to disk; there is no need to train a second quantizer.
            pqVectors = (PQVectors) builder.getCompressedVectors();
        }
        ProductQuantization pq = pqVectors.getCompressor();
        System.out.printf("PQ-scored build: %,d nodes in %.1fs; %d subspaces x %d clusters, %d bytes/vector (from %d)%n",
                pqGraph.size(0), seconds(start), pq.getSubspaceCount(), pq.getClusterCount(),
                pq.compressedVectorSize(), DIMENSION * Float.BYTES);

        // Search with PQ scores and rerank the best RERANK_K candidates exactly. The precomputed score
        // function computes the query's distance to every centroid once, then scores a node by table
        // lookups on its code.
        try (GraphSearcher searcher = pqGraph.searcher()) {
            report("PQ build: PQ scores + exact rerank", recall(ds, searcher, RERANK_K,
                    q -> new DefaultSearchScoreProvider(pqVectors.precomputedScoreFunctionFor(q, SIMILARITY_FUNCTION),
                                                        ds.ravv.rerankerFor(q, SIMILARITY_FUNCTION))));
            // Without a reranker the approximate scores are final: cheaper, and recall drops.
            report("PQ build: PQ scores only (rerankless)", recall(ds, searcher, TOP_K,
                    q -> new DefaultSearchScoreProvider(pqVectors.precomputedScoreFunctionFor(q, SIMILARITY_FUNCTION))));
        }

        // Fewer subspaces give smaller codes and coarser scores: here a quarter of the default.
        try (HnswIndexBuilder builder = Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION)
                .withCompressionType(CompressionType.PQ)
                .withPqSubspaces(DIMENSION / 8)) {
            PersistableGraphIndex smallCodesGraph = builder.buildAndPopulate();
            CompressedVectors smallCodes = builder.getCompressedVectors();
            try (GraphSearcher searcher = smallCodesGraph.searcher()) {
                report("PQ build, " + (DIMENSION / 8) + "-byte codes: PQ scores + exact rerank", recall(ds, searcher, RERANK_K,
                        q -> new DefaultSearchScoreProvider(smallCodes.precomputedScoreFunctionFor(q, SIMILARITY_FUNCTION),
                                                            ds.ravv.rerankerFor(q, SIMILARITY_FUNCTION))));
            }
        }

        // BQ: binary quantization, one bit per dimension. Much cheaper still, and coarser.
        start = System.nanoTime();
        PersistableGraphIndex bqGraph;
        CompressedVectors bqVectors;
        try (HnswIndexBuilder builder = Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION)
                .withCompressionType(CompressionType.BQ)) {
            bqGraph = builder.buildAndPopulate();
            bqVectors = builder.getCompressedVectors();
        }
        System.out.printf("BQ-scored build: %,d nodes in %.1fs%n", bqGraph.size(0), seconds(start));
        try (GraphSearcher searcher = bqGraph.searcher()) {
            report("BQ build: BQ scores + exact rerank", recall(ds, searcher, RERANK_K,
                    q -> new DefaultSearchScoreProvider(bqVectors.precomputedScoreFunctionFor(q, SIMILARITY_FUNCTION),
                                                        ds.ravv.rerankerFor(q, SIMILARITY_FUNCTION))));
        }

        return new PqBuild(pqGraph, pqVectors);
    }

    // ---------------------------------------------------------------------------------------------
    // 4. Writer types
    // ---------------------------------------------------------------------------------------------

    /**
     * A populated graph is a {@link PersistableGraphIndex}, which hands out each writer's builder. All
     * the writers produce the same format, loaded with {@link OnDiskGraphIndex#load}:
     * <ul>
     *     <li>{@code getWriterBuilder(IndexWriter)}: the sequential writer. One forward pass that never
     *     seeks, for append-only outputs (object storage, a Lucene {@code IndexOutput}); the caller owns
     *     the output. OpenSearch writes this way.</li>
     *     <li>{@code getParallelWriterBuilder(Path)}: the parallel writer. Encodes node records on worker
     *     threads and writes them asynchronously; worthwhile when encoding features (e.g. NVQ)
     *     dominates.</li>
     *     <li>{@code getWriterBuilder(Path)}: the single-threaded random-access writer. Writes the
     *     records in order, then seeks back to fill in the header; can start at an offset in a larger
     *     file. Cassandra's default.</li>
     * </ul>
     * Each accessor returns that writer's own builder, so its specific options are available. Each node
     * record carries the {@link Feature}s added with {@code with(...)}, and {@code write} takes, for each
     * feature, a function from a node to that node's state. Here the feature is the full-precision
     * vectors, stored inline for exact reranking.
     */
    private static void section4WriterTypes(Dataset ds, PersistableGraphIndex graph, Path workDir) throws IOException {
        header("4. Writer types (full-precision vectors inline)");

        // --- The shortcut: writeTo stores the vectors inline with the random-access writer. The result
        // is searched with searcher.search(query, topK, rerankK, similarityFunction, filter), which scores
        // with the vectors the graph stores, so no score provider is needed (see searchOnDisk).
        Path simplePath = workDir.resolve("simple.graph");
        graph.writeTo(simplePath, ds.ravv);
        searchOnDisk(ds, simplePath, "writeTo(path, vectors)", RERANK_K);

        // The writers themselves, for other outputs, features and options.
        Map<FeatureId, IntFunction<Feature.State>> inlineVectors = inlineVectorStates(ds);

        // --- Sequential writer. SimpleWriter is a plain file; any IndexWriter works. Closing the graph
        // writer does not close the output, so the output gets its own try-with-resources.
        Path sequentialPath = workDir.resolve("sequential.graph");
        try (SimpleWriter out = new SimpleWriter(sequentialPath);
             GraphIndexWriter writer = graph.getWriterBuilder(out)
                     .with(new InlineVectors(DIMENSION))
                     .build()) {
            writer.write(inlineVectors);
        }
        searchOnDisk(ds, sequentialPath, "sequential writer", RERANK_K);

        // --- Parallel writer.
        Path parallelPath = workDir.resolve("parallel.graph");
        try (GraphIndexWriter writer = graph.getParallelWriterBuilder(parallelPath)
                .with(new InlineVectors(DIMENSION))
                .withParallelWorkerThreads(0)        // optional: 0 = all available processors
                .build()) {
            writer.write(inlineVectors);
        }
        searchOnDisk(ds, parallelPath, "parallel writer", RERANK_K);

        // --- Random-access writer, embedded at an offset, as Cassandra does: its graph shares a file
        // with a header in front of it. The built writer is an OnDiskGraphIndexWriter, so getOutput()
        // and checksum() are available without a cast.
        Path embeddedPath = workDir.resolve("embedded.graph");
        byte[] prefix = "SAI-HEADER-STANDIN".getBytes(StandardCharsets.US_ASCII);
        try (OnDiskGraphIndexWriter writer = graph.getWriterBuilder(embeddedPath)
                .with(new InlineVectors(DIMENSION))
                .withStartOffset(prefix.length)
                .build()) {
            RandomAccessWriter out = writer.getOutput();
            out.seek(0);
            out.write(prefix);
            writer.write(inlineVectors);
            System.out.printf("Embedded graph written after a %d-byte prefix, checksum=%x%n", prefix.length, writer.checksum());
        }
        // Loaded by offset, without the footer, because the caller recorded where the graph starts.
        try (ReaderSupplier rs = ReaderSupplierFactory.open(embeddedPath);
             OnDiskGraphIndex onDisk = OnDiskGraphIndex.load(rs, prefix.length, false);
             GraphSearcher searcher = onDisk.searcher()) {
            report("random-access writer, at offset " + prefix.length, recallStored(ds, searcher, RERANK_K));
        }

        // --- An OnDiskGraphIndex is persistable too, so a loaded graph can be rewritten, here into the
        // sequential format, with the inline vectors read from the source graph itself.
        Path rewrittenPath = workDir.resolve("rewritten.graph");
        try (ReaderSupplier rs = ReaderSupplierFactory.open(sequentialPath);
             OnDiskGraphIndex source = OnDiskGraphIndex.load(rs);
             OnDiskGraphIndex.View sourceView = source.getView();
             SimpleWriter out = new SimpleWriter(rewrittenPath);
             GraphIndexWriter writer = source.getWriterBuilder(out)
                     .with(new InlineVectors(DIMENSION))
                     .build()) {
            writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                    node -> new InlineVectors.State(sourceView.getVector(node))));
        }
        searchOnDisk(ds, rewrittenPath, "on-disk graph rewritten sequentially", RERANK_K);
    }

    // ---------------------------------------------------------------------------------------------
    // 5. Fused PQ
    // ---------------------------------------------------------------------------------------------

    /**
     * {@link FusedPQ} stores, in each node's record, the PQ codes of that node's neighbors, so a search
     * reads the codes it needs from the same page as the adjacency list and nothing has to be loaded
     * into memory up front. This is Cassandra's current on-disk format. The feature needs the graph's
     * max degree and the codebook, and each node's state is built from a view of the graph and the PQ
     * codes: both straight from the builder in section 3.
     *
     * @return the path of the fused graph, searched again in section 11
     */
    private static Path section5FusedPq(Dataset ds, PqBuild pqBuild, Path workDir) throws IOException {
        header("5. Fused PQ: neighbors' PQ codes in each node record");

        Path graphPath = workDir.resolve("fused.graph");
        try (GraphIndex.View view = pqBuild.graph.getView();
             GraphIndexWriter writer = pqBuild.graph.getWriterBuilder(graphPath)
                     .with(new InlineVectors(DIMENSION))
                     .with(new FusedPQ(pqBuild.graph.maxDegree(), pqBuild.pq()))
                     .build()) {
            var states = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
            states.put(FeatureId.INLINE_VECTORS, node -> new InlineVectors.State(ds.ravv.getVector(node)));
            states.put(FeatureId.FUSED_PQ, node -> new FusedPQ.State(view, pqBuild.pqVectors, node));
            writer.write(states);
        }
        // Cassandra also writes the codebook on its own, so a later compaction can refine it rather than
        // retrain from scratch (section 9). Searching doesn't need it.
        Path codebookPath = workDir.resolve("fused.codebook");
        try (SimpleWriter out = new SimpleWriter(codebookPath)) {
            pqBuild.pq().write(out, OnDiskGraphIndex.CURRENT_VERSION);
        }
        System.out.printf("graph %,d bytes (PQ codes inside), codebook %,d bytes%n",
                Files.size(graphPath), Files.size(codebookPath));

        // The stored-vector search sees the fused codes and traverses with them, reranking the best RERANK_K
        // candidates with the inline vectors: the same scoring Cassandra builds by hand.
        searchOnDisk(ds, graphPath, "fused PQ + inline rerank", RERANK_K);
        // To choose the scoring yourself, build the score provider from the view. Here, Cassandra's
        // "rerankless" mode: no reranker, so the approximate scores are final.
        searchOnDisk(ds, graphPath, "fused PQ only (rerankless)", TOP_K, IndexApiExample::fusedOnly);
        return graphPath;
    }

    // ---------------------------------------------------------------------------------------------
    // 6. NVQ and fused PQ, parallel writer
    // ---------------------------------------------------------------------------------------------

    /**
     * {@link NVQ} replaces the full-precision inline vectors with NVQ-compressed ones, roughly 4x
     * smaller, so reranking reads less from disk at the cost of approximate final scores. Traversal still
     * uses fused PQ. Encoding every node's NVQ vector is the expensive part of the write, which is what
     * the parallel writer is for: the state functions run on its worker threads. This is Cassandra's
     * compaction with parallel writing enabled.
     */
    private static void section6NvqAndFusedPq(Dataset ds, PqBuild pqBuild, Path workDir) throws IOException {
        header("6. NVQ + fused PQ, parallel writer");

        // NVQ is trained at write time. Cassandra's memtable flush uses compute(ravv, n); its compaction
        // accumulates a global mean as rows stream in and uses NVQuantization.create(mean, n) instead.
        NVQuantization nvq = NVQuantization.compute(ds.ravv, NVQ_SUB_VECTORS);

        Path graphPath = workDir.resolve("nvq-fused.graph");
        try (GraphIndex.View view = pqBuild.graph.getView();
             GraphIndexWriter writer = pqBuild.graph.getParallelWriterBuilder(graphPath)
                     .with(new NVQ(nvq))
                     .with(new FusedPQ(pqBuild.graph.maxDegree(), pqBuild.pq()))
                     .build()) {
            var states = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
            states.put(FeatureId.NVQ_VECTORS, node -> new NVQ.State(nvq.encode(ds.ravv.getVector(node))));
            states.put(FeatureId.FUSED_PQ, node -> new FusedPQ.State(view, pqBuild.pqVectors, node));
            writer.write(states);
        }
        // The view's reranker scores with whichever vectors the graph stores: NVQ here.
        searchOnDisk(ds, graphPath, String.format("NVQ + fused PQ (%,d bytes)", Files.size(graphPath)), RERANK_K);
    }

    // ---------------------------------------------------------------------------------------------
    // 7. Separately stored PQ codes
    // ---------------------------------------------------------------------------------------------

    /**
     * The graph stores vectors for reranking, and the PQ codes are stored beside it and loaded into
     * memory for traversal. This is OpenSearch's layout, written sequentially because OpenSearch writes
     * through a Lucene {@code IndexOutput}: it appends the {@link PQVectors} after the graph in the same
     * file and reads each from a slice. Separate files here, for clarity. The codes are the builder's
     * own, from section 3.
     */
    private static void section7SeparatePqCodes(Dataset ds, PqBuild pqBuild, Path workDir) throws IOException {
        header("7. Separately stored PQ codes, sequential writer");

        Path pqPath = workDir.resolve("graph.pq");
        try (SimpleWriter out = new SimpleWriter(pqPath)) {
            pqBuild.pqVectors.write(out, OnDiskGraphIndex.CURRENT_VERSION);
        }
        PQVectors loadedPq;
        try (ReaderSupplier rs = ReaderSupplierFactory.open(pqPath);
             var reader = rs.get()) {
            loadedPq = PQVectors.load(reader);
        }

        // --- Full-precision vectors inline, PQ codes beside the graph.
        Path inlinePath = workDir.resolve("inline-with-pq.graph");
        try (SimpleWriter out = new SimpleWriter(inlinePath);
             GraphIndexWriter writer = pqBuild.graph.getWriterBuilder(out)
                     .with(new InlineVectors(DIMENSION))
                     .build()) {
            writer.write(inlineVectorStates(ds));
        }
        System.out.printf("graph %,d bytes, PQ codes %,d bytes%n", Files.size(inlinePath), Files.size(pqPath));
        searchOnDisk(ds, inlinePath, "in-memory PQ + inline rerank", RERANK_K,
                (view, q) -> new DefaultSearchScoreProvider(loadedPq.precomputedScoreFunctionFor(q, SIMILARITY_FUNCTION),
                                                            view.rerankerFor(q, SIMILARITY_FUNCTION)));

        // --- NVQ vectors inline, PQ codes beside the graph. OpenSearch encodes every vector with NVQ up
        // front, so the states just look the codes up.
        NVQVectors nvqVectors = NVQuantization.compute(ds.ravv, NVQ_SUB_VECTORS).encodeAll(ds.ravv, ForkJoinPool.commonPool());
        Path nvqPath = workDir.resolve("nvq-with-pq.graph");
        try (SimpleWriter out = new SimpleWriter(nvqPath);
             GraphIndexWriter writer = pqBuild.graph.getWriterBuilder(out)
                     .with(new NVQ(nvqVectors.getNVQuantization()))
                     .build()) {
            writer.write(Feature.singleStateFactory(FeatureId.NVQ_VECTORS, node -> new NVQ.State(nvqVectors.get(node))));
        }
        searchOnDisk(ds, nvqPath, String.format("in-memory PQ + NVQ rerank (%,d bytes)", Files.size(nvqPath)), RERANK_K,
                (view, q) -> new DefaultSearchScoreProvider(loadedPq.precomputedScoreFunctionFor(q, SIMILARITY_FUNCTION),
                                                            view.rerankerFor(q, SIMILARITY_FUNCTION)));
    }

    // ---------------------------------------------------------------------------------------------
    // 8. Incremental construction
    // ---------------------------------------------------------------------------------------------

    /**
     * For vectors that arrive over time, {@code build()} returns the empty graph and
     * {@code addGraphNode} inserts one node at a time. Inserts are thread-safe and the graph is
     * searchable while they run. Deleted nodes are hidden from searches immediately and removed by
     * {@code cleanup()}, which runs once the inserts and deletes are done, and before writing. This is
     * Cassandra's memtable index.
     */
    private static void section8IncrementalBuild(Dataset ds, Path workDir) throws IOException {
        header("8. Incremental construction, with deletes");
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

                // Four inserters. addGraphNode returns an estimate of the heap the insert used, which is
                // what Cassandra's memtable accounting adds up.
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
                // One thread searching the graph while it grows. It reuses one searcher, since creating one
                // allocates scratch space, and gives it a fresh view before each search so the search sees
                // the nodes added since the last one.
                Future<?> searcher = pool.submit(() -> {
                    Random random = new Random(0);
                    try (GraphSearcher s = graph.searcher()) {
                        while (inserting.get()) {
                            s.setView(graph.getView());
                            VectorFloat<?> q = ds.queries.get(random.nextInt(ds.queries.size()));
                            s.search(DefaultSearchScoreProvider.exact(q, SIMILARITY_FUNCTION, ds.ravv), TOP_K, Bits.ALL);
                            searches.incrementAndGet();
                        }
                    } catch (IOException e) {
                        throw new RuntimeException(e);
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

            // Remove the deleted nodes and finish the graph. Must not run during inserts.
            builder.cleanup();
            System.out.printf("After cleanup: %,d nodes (%,d deleted)%n", graph.size(0), deleted.size());
            searchInMemoryExact(ds, graph, "incremental build, in memory");

            // Deletes leave holes in the graph's ordinals. By default the writers renumber the remaining
            // nodes 0..size-1 on disk; sequentialRenumbering is that default, passed explicitly here so
            // the example can map search results back. Cassandra passes its own mapping from graph
            // ordinals to row ids.
            Map<Integer, Integer> graphToDisk = OnDiskGraphIndexWriter.sequentialRenumbering(graph);
            Path path = workDir.resolve("incremental.graph");
            try (GraphIndexWriter writer = graph.getWriterBuilder(path)
                    .with(new InlineVectors(DIMENSION))
                    .withMap(graphToDisk)
                    .build()) {
                // The state functions receive graph ordinals, not on-disk ones.
                writer.write(inlineVectorStates(ds));
            }
            IntUnaryOperator diskToGraph = invert(graphToDisk);
            try (ReaderSupplier rs = ReaderSupplierFactory.open(path);
                 OnDiskGraphIndex onDisk = OnDiskGraphIndex.load(rs);
                 GraphSearcher searcher = onDisk.searcher()) {
                report(String.format("incremental build on disk, %,d nodes", onDisk.size(0)),
                        recall(ds, ds.groundTruth, Bits.ALL, diskToGraph, storedVectorSearch(searcher, RERANK_K)));
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // 9. Your own score provider, and a PQ rescore
    // ---------------------------------------------------------------------------------------------

    /**
     * Cassandra's compaction streams rows in and never holds them all in one
     * {@link RandomAccessVectorValues}, so it maintains PQ codes itself and builds with
     * {@code Indexes.hnswBuilder(BuildScoreProvider, dimension)}: it trains a codebook on the first rows,
     * encodes each row as it arrives, and partway through refines the codebook on everything seen so far,
     * re-encodes, and rescores the graph with the refined codes. For everything else, prefer letting the
     * builder compress ({@code withCompressionType}).
     * <p>
     * The builder doesn't coordinate a rescore with inserts in flight. Here the inserts run in two
     * phases with the rescore between them; a caller inserting concurrently, as Cassandra does, holds its
     * own lock around the rescore.
     */
    private static void section9ScoreProviderAndRescore(Dataset ds, Path workDir) throws IOException {
        header("9. Your own score provider, and a PQ rescore");
        int n = ds.ravv.size();
        int half = n / 2;

        // An initial codebook from the first rows, trained the way Cassandra trains it: 32 subspaces (its
        // default for 64 dimensions), 256 clusters, as fused PQ requires, and no centering.
        var sample = new ListRandomAccessVectorValues(ds.vectors.subList(0, n / 10), DIMENSION);
        ProductQuantization initialPq = ProductQuantization.compute(sample, 32, 256, false);
        MutablePQVectors codes = new MutablePQVectors(initialPq);

        try (HnswIndexBuilder builder = Indexes.hnswBuilder(
                BuildScoreProvider.pqBuildScoreProvider(SIMILARITY_FUNCTION, codes), DIMENSION)) {
            // Phase 1. Each row's code must exist before the row is inserted, since inserts score the
            // nodes they visit by ordinal.
            for (int i = 0; i < half; i++) {
                codes.encodeAndSet(i, ds.ravv.getVector(i));
            }
            IntStream.range(0, half).parallel().forEach(i -> builder.addGraphNode(i, ds.ravv.getVector(i)));

            // Refine the codebook on everything seen so far, re-encode, and rescore. rescore() returns a
            // builder holding a copy of the graph with every edge re-scored, which carries on the build.
            ProductQuantization refinedPq = initialPq.refine(
                    new ListRandomAccessVectorValues(ds.vectors.subList(0, half), DIMENSION));
            MutablePQVectors refined = new MutablePQVectors(refinedPq);
            for (int i = 0; i < half; i++) {
                refined.encodeAndSet(i, ds.ravv.getVector(i));
            }
            try (HnswIndexBuilder rescored = HnswIndexBuilder.rescore(builder,
                    BuildScoreProvider.pqBuildScoreProvider(SIMILARITY_FUNCTION, refined))) {
                // Phase 2, with the refined codes.
                for (int i = half; i < n; i++) {
                    refined.encodeAndSet(i, ds.ravv.getVector(i));
                }
                IntStream.range(half, n).parallel().forEach(i -> rescored.addGraphNode(i, ds.ravv.getVector(i)));
                rescored.cleanup();
                PersistableGraphIndex graph = rescored.getGraph();
                System.out.printf("Built %,d nodes: %,d before the rescore, %,d after%n", graph.size(0), half, n - half);

                // Written in Cassandra's format, with the refined codebook and codes.
                Path path = workDir.resolve("compaction.graph");
                try (GraphIndex.View view = graph.getView();
                     GraphIndexWriter writer = graph.getWriterBuilder(path)
                             .with(new InlineVectors(DIMENSION))
                             .with(new FusedPQ(graph.maxDegree(), refinedPq))
                             .build()) {
                    var states = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
                    states.put(FeatureId.INLINE_VECTORS, node -> new InlineVectors.State(ds.ravv.getVector(node)));
                    states.put(FeatureId.FUSED_PQ, node -> new FusedPQ.State(view, refined, node));
                    writer.write(states);
                }
                searchOnDisk(ds, path, "streamed, rescored, fused PQ + inline rerank", RERANK_K);
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // 10. Continuing a saved graph
    // ---------------------------------------------------------------------------------------------

    /**
     * Adding vectors to an existing graph instead of rebuilding it. OpenSearch's leading-segment merge
     * does this: it saves each flushed graph, and when merging reloads the largest segment's graph and
     * inserts only the other segments' vectors. The search format drops the neighbor scores a builder
     * needs, so the graph to continue is saved with {@code OnHeapGraphIndex.save} and reloaded with
     * {@code OnHeapGraphIndex.load}, then handed to {@code withExistingGraph}.
     */
    @SuppressWarnings("deprecation") // OnHeapGraphIndex.save/load are deprecated and experimental
    private static void section10ContinuingASavedGraph(Dataset ds, Path workDir) throws IOException {
        header("10. Continuing a saved graph");

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

        // The builder scores with all the vectors: the existing nodes at their original ordinals, then
        // the new ones. Add the new nodes from the existing graph's id upper bound up, then clean up.
        // (populateGraph and buildAndPopulate only populate an empty graph.) OpenSearch also marks the
        // leading segment's deleted documents with markNodeDeleted before cleanup().
        long start = System.nanoTime();
        PersistableGraphIndex extended;
        try (HnswIndexBuilder builder = Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION)
                .withExistingGraph(loaded)) {
            extended = builder.build();
            IntStream.range(loaded.getIdUpperBound(), ds.ravv.size()).parallel()
                    .forEach(node -> builder.addGraphNode(node, ds.ravv.getVector(node)));
            builder.cleanup();
        }
        System.out.printf("Appended %,d nodes in %.1fs, for %,d in total%n",
                extended.size(0) - baseCount, seconds(start), extended.size(0));
        searchInMemoryExact(ds, extended, "reloaded and extended, in memory");

        // The existing graph fixes the max degrees and hierarchy, so setting them as well is an error.
        System.out.println("Setting the graph shape along with an existing graph is rejected:");
        expectFailure(() -> Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION)
                .withExistingGraph(loaded)
                .withMaxDegree(16)
                .build());
    }

    // ---------------------------------------------------------------------------------------------
    // 11. Search options
    // ---------------------------------------------------------------------------------------------

    /** Searching the fused-PQ graph from section 5 with the options a {@link GraphSearcher} offers. */
    private static void section11SearchOptions(Dataset ds, Path fusedPath) throws IOException {
        header("11. Search options (on the fused-PQ graph from section 5)");
        try (ReaderSupplier rs = ReaderSupplierFactory.open(fusedPath);
             OnDiskGraphIndex onDisk = OnDiskGraphIndex.load(rs);
             GraphSearcher searcher = onDisk.searcher()) {
            // rerankK: how many candidates to collect by approximate score before reranking and keeping
            // the best TOP_K. Wider is more accurate and slower.
            for (int rerankK : new int[] {TOP_K, RERANK_K, 10 * TOP_K}) {
                report("rerankK = " + rerankK, recallStored(ds, searcher, rerankK));
            }

            // Filtered search: only nodes for which the Bits return true are returned (Cassandra passes the
            // rows that match the query's other predicates). Ground truth is over the same subset.
            Bits evenOnly = node -> node % 2 == 0;
            List<Set<Integer>> evenTruth = ds.queries.stream()
                    .map(q -> bruteForce(ds, q, TOP_K, node -> node % 2 == 0))
                    .collect(Collectors.toList());
            report("filtered to even ordinals", recall(ds, evenTruth, evenOnly, node -> node, storedVectorSearch(searcher, RERANK_K)));

            // Resuming: fetch the next results of the previous search without starting over (Cassandra
            // does this when filtering leaves a page short of its limit).
            VectorFloat<?> query = ds.queries.get(0);
            SearchResult first = searcher.search(query, TOP_K, RERANK_K, SIMILARITY_FUNCTION, Bits.ALL);
            SearchResult more = searcher.resume(TOP_K, RERANK_K);
            System.out.printf("  first search: %d results, best score %.4f; resumed: %d more, best score %.4f%n",
                    first.getNodes().length, first.getNodes()[0].score,
                    more.getNodes().length, more.getNodes()[0].score);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // 12. Validation, recipes and IVF
    // ---------------------------------------------------------------------------------------------

    private static void section12ValidationRecipesAndIvf(Dataset ds) {
        header("12. Validation, recipes and IVF");

        System.out.println("Out-of-range settings are rejected when the graph is built:");
        expectFailure(() -> Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION).withBeamWidth(0).build());
        expectFailure(() -> Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION).withNeighborOverflow(0.5f).build());

        // Several max degrees means one per layer, which needs the hierarchy.
        System.out.println("Per-layer max degrees need the hierarchy:");
        expectFailure(() -> Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION)
                .withMaxDegrees(List.of(32, 16))
                .withAddHierarchy(false)
                .build());

        System.out.println("Compressed vectors exist only once the graph is built:");
        expectFailure(() -> Indexes.hnswBuilder(ds.ravv, SIMILARITY_FUNCTION)
                .withCompressionType(CompressionType.PQ)
                .getCompressedVectors());

        // A score-provider builder has no vectors to populate from.
        System.out.println("buildAndPopulate() needs a builder created from vectors:");
        expectFailure(() -> Indexes.hnswBuilder(BuildScoreProvider.randomAccessScoreProvider(ds.ravv, SIMILARITY_FUNCTION), DIMENSION)
                .buildAndPopulate());

        // Recipes are experimental: DEFAULT is defined (section 2 uses it), the rest are named but have
        // no values yet.
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
    // Scoring on disk
    // ---------------------------------------------------------------------------------------------

    /**
     * Builds a query's score provider from the view of an on-disk graph, for scoring other than what the
     * stored-vector search chooses (e.g. rerankless, or with separately stored PQ codes). The view of an
     * {@link OnDiskGraphIndex} is a {@link GraphIndex.ScoringView}: its reranker scores with whichever
     * vectors the graph stores (inline or NVQ), and for a fused graph its approximate score function
     * uses the fused PQ codes.
     */
    @FunctionalInterface
    private interface ViewScoring {
        SearchScoreProvider forQuery(GraphIndex.ScoringView view, VectorFloat<?> query);
    }

    /** Traverses with the fused PQ codes and returns their approximate scores as final. */
    private static SearchScoreProvider fusedOnly(GraphIndex.ScoringView view, VectorFloat<?> q) {
        return new DefaultSearchScoreProvider(view.approximateScoreFunctionFor(q, SIMILARITY_FUNCTION));
    }

    /**
     * Loads a graph from {@code path} and reports its recall, searching with the vectors the graph stores:
     * {@code searcher.search(query, topK, rerankK, similarityFunction, filter)}. With fused PQ it traverses
     * with the PQ codes and reranks with the stored vectors; otherwise it scores with the stored vectors.
     */
    private static void searchOnDisk(Dataset ds, Path path, String label, int rerankK) throws IOException {
        try (ReaderSupplier rs = ReaderSupplierFactory.open(path);
             OnDiskGraphIndex onDisk = OnDiskGraphIndex.load(rs);
             GraphSearcher searcher = onDisk.searcher()) {
            report(label, recallStored(ds, searcher, rerankK));
        }
    }

    /** Loads a graph from {@code path} and reports its recall, scoring as {@code scoring} says. */
    private static void searchOnDisk(Dataset ds, Path path, String label, int rerankK, ViewScoring scoring) throws IOException {
        try (ReaderSupplier rs = ReaderSupplierFactory.open(path);
             OnDiskGraphIndex onDisk = OnDiskGraphIndex.load(rs);
             GraphSearcher searcher = onDisk.searcher()) {
            report(label, recall(ds, searcher, rerankK, q -> scoring.forQuery(scoringView(searcher), q)));
        }
    }

    private static GraphIndex.ScoringView scoringView(GraphSearcher searcher) {
        return (GraphIndex.ScoringView) searcher.getView();
    }

    /** The full-precision vectors, as the state of an {@link InlineVectors} feature. */
    private static Map<FeatureId, IntFunction<Feature.State>> inlineVectorStates(Dataset ds) {
        return Feature.singleStateFactory(FeatureId.INLINE_VECTORS, node -> new InlineVectors.State(ds.ravv.getVector(node)));
    }

    /** Searches an in-memory graph, scoring every candidate exactly against the dataset's vectors. */
    private static void searchInMemoryExact(Dataset ds, GraphIndex graph, String label) throws IOException {
        try (GraphSearcher searcher = graph.searcher()) {
            report(label, recall(ds, searcher, RERANK_K,
                    q -> DefaultSearchScoreProvider.exact(q, SIMILARITY_FUNCTION, ds.ravv)));
        }
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

    /** Runs one query, returning only nodes in {@code acceptOrds}. */
    @FunctionalInterface
    private interface QuerySearch {
        SearchResult search(VectorFloat<?> query, Bits acceptOrds);
    }

    /** Mean recall@{@value #TOP_K} over the dataset's queries, scoring with {@code ssp}. */
    private static double recall(Dataset ds, GraphSearcher searcher, int rerankK, ScoreProviderFactory ssp) {
        return recall(ds, ds.groundTruth, Bits.ALL, node -> node,
                (q, acceptOrds) -> searcher.search(ssp.forQuery(q), TOP_K, rerankK, 0.0f, 0.0f, acceptOrds));
    }

    /** Mean recall@{@value #TOP_K} over the dataset's queries, scoring with the vectors the graph stores. */
    private static double recallStored(Dataset ds, GraphSearcher searcher, int rerankK) {
        return recall(ds, ds.groundTruth, Bits.ALL, node -> node, storedVectorSearch(searcher, rerankK));
    }

    private static QuerySearch storedVectorSearch(GraphSearcher searcher, int rerankK) {
        return (q, acceptOrds) -> searcher.search(q, TOP_K, rerankK, SIMILARITY_FUNCTION, acceptOrds);
    }

    /**
     * Mean recall@{@value #TOP_K} against {@code truth}, searching only {@code acceptOrds} and mapping
     * each result through {@code toGraphOrdinal} first (for graphs written with renumbered ordinals).
     */
    private static double recall(Dataset ds, List<Set<Integer>> truth, Bits acceptOrds,
                                 IntUnaryOperator toGraphOrdinal, QuerySearch search) {
        int hits = 0;
        for (int i = 0; i < ds.queries.size(); i++) {
            SearchResult result = search.search(ds.queries.get(i), acceptOrds);
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
