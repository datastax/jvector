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
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.MutableHnswIndex;
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
import io.github.jbellis.jvector.index.Index;
import io.github.jbellis.jvector.index.IndexSearcher;
import io.github.jbellis.jvector.index.Indexes;
import io.github.jbellis.jvector.ivf.IvfIndex;
import io.github.jbellis.jvector.quantization.MutablePQVectors;
import io.github.jbellis.jvector.quantization.NVQVectors;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
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
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.IntFunction;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * A walkthrough of the generic {@link Index} hierarchy, organized around how JVector is actually
 * used in production today by Cassandra (SAI vector indexes) and by the OpenSearch jVector plugin.
 * Every section builds, persists and/or searches a real index and prints its recall@{@value #TOP_K}
 * against brute force, so the example doubles as a check that each path still works end to end.
 * <p>
 * Sections, in the order {@link #main} runs them:
 * <ol>
 *     <li><b>API tour</b> &mdash; {@link Indexes} as the type-first entry point, aggregate
 *     validation, recipe scaffolding, and the IVF seam.</li>
 *     <li><b>Full-precision build</b> &mdash; exact scoring during construction (Cassandra
 *     memtables, OpenSearch segments too small to quantize), searched in memory, and the generic
 *     {@link Index} handle narrowed back to {@link GraphIndex}.</li>
 *     <li><b>PQ-scored build</b> &mdash; construction scored by product-quantized vectors
 *     (Cassandra compaction, OpenSearch flush/merge with quantization enabled).</li>
 *     <li><b>Writer types</b> &mdash; the three on-disk writer strategies, how to obtain each one
 *     (new {@link PersistableGraphIndex} accessors vs. the writer classes directly), and embedding a
 *     graph at an offset inside a larger file the way Cassandra does.</li>
 *     <li><b>Non-fused PQ on disk</b> &mdash; full-precision vectors inline, PQ codes stored beside
 *     the graph (OpenSearch's PQ layout, Cassandra's pre-fused formats).</li>
 *     <li><b>Fused PQ on disk</b> &mdash; PQ codes of each node's neighbors packed into the node
 *     record (Cassandra's current format).</li>
 *     <li><b>NVQ on disk</b> &mdash; NVQ-compressed vectors inline, both the Cassandra layout (NVQ +
 *     fused PQ) and the OpenSearch layout (NVQ + an auxiliary PQ blob, written sequentially).</li>
 *     <li><b>Disk to memory, and memory to disk</b> &mdash; loading a graph for search
 *     ({@link OnDiskGraphIndex}), and reloading a <em>mutable</em> in-memory graph from disk and
 *     appending to it through {@link io.github.jbellis.jvector.graph.HnswIndexBuilder#withExistingGraph}
 *     (OpenSearch's leading-segment merge).</li>
 *     <li><b>Legacy construction reference</b> &mdash; every way a graph was built before this
 *     branch, run side by side with its new-API counterpart, including incremental construction
 *     with deletes (Cassandra memtables) and a mid-build PQ rescore (Cassandra compaction).</li>
 * </ol>
 *
 * <h2>Old construction path &rarr; new API, at a glance</h2>
 * Section 9 runs each of these rows; the "who uses it" column names the call site in each consumer.
 * <table class="striped">
 *   <caption>Graph construction</caption>
 *   <tr><th>Before</th><th>Who uses it</th><th>Now</th></tr>
 *   <tr><td>{@code new GraphIndexBuilder(ravv, vsf, M, beam, overflow, alpha, addHierarchy)}
 *           then {@code build(ravv)}</td>
 *       <td>Cassandra tests ({@code VectorTester}), most JVector examples</td>
 *       <td>{@code Indexes.hnswBuilder().withVectorValues(ravv).withSimilarityFunction(vsf)...build()}</td></tr>
 *   <tr><td>{@code new GraphIndexBuilder(bsp, dim, M, beam, overflow, alpha, addHierarchy)} then
 *           parallel {@code addGraphNode} + {@code cleanup()}</td>
 *       <td>OpenSearch {@code JVectorWriter.getGraph}</td>
 *       <td>{@code ...withVectorValues(ravv).withScoreProvider(bsp)...build()}</td></tr>
 *   <tr><td>10-arg constructor with {@code refineFinalGraph}, SIMD pool and parallel pool</td>
 *       <td>Cassandra {@code CompactionGraph}</td>
 *       <td>{@code withRefineFinalGraph}, {@code withSimdExecutor}, {@code withParallelExecutor}</td></tr>
 *   <tr><td>{@code List<Integer> maxDegrees} constructors</td><td>&mdash;</td>
 *       <td>{@code withMaxDegrees(List)}</td></tr>
 *   <tr><td>{@code GraphIndexBuilder.builder(...)} fluent builder (addHierarchy/refine from JMX)</td>
 *       <td>new in 4.0.x</td>
 *       <td>{@code Indexes.hnswBuilder()} (addHierarchy/refine explicit)</td></tr>
 *   <tr><td>{@code new GraphIndexBuilder(bsp, dim, existingGraph, ...)} + {@code addGraphNode}</td>
 *       <td>OpenSearch leading-segment merge</td>
 *       <td>{@code withExistingGraph(graph)} with a superset {@code withVectorValues}</td></tr>
 *   <tr><td>{@code GraphIndexBuilder.buildAndMergeNewNodes(reader, ...)}</td><td>&mdash;</td>
 *       <td>{@code OnHeapGraphIndex.load(...)} + {@code withExistingGraph}</td></tr>
 *   <tr><td>Incremental {@code addGraphNode} interleaved with search, {@code markNodeDeleted},
 *           {@code removeDeletedNodes}, {@code GraphIndexBuilder.rescore}</td>
 *       <td>Cassandra {@code CassandraOnHeapGraph} (memtable) and {@code CompactionGraph} (PQ
 *           fine-tuning); OpenSearch merge deletes</td>
 *       <td>{@code Indexes.hnswBuilder()...buildMutable()}, returning a {@link MutableHnswIndex}:
 *           {@code addNode}, {@code markDeleted}, {@code removeDeletedNodes}, {@code cleanup},
 *           {@code rescore(Supplier)} (which also replaces the caller's own insert/rescore lock),
 *           searchable throughout</td></tr>
 * </table>
 * <table class="striped">
 *   <caption>Search and persistence</caption>
 *   <tr><th>Before</th><th>Now</th></tr>
 *   <tr><td>{@code ImmutableGraphIndex}</td><td>{@link GraphIndex} (extends {@link Index})</td></tr>
 *   <tr><td>{@code new GraphSearcher(graph)}</td><td>{@code graph.searcher()} (still a {@link GraphSearcher})</td></tr>
 *   <tr><td>{@code new OnDiskGraphIndexWriter.Builder(graph, path)} /
 *           {@code GraphIndexWriter.getBuilderFor(RANDOM_ACCESS, graph, path)}</td>
 *       <td>{@code persistable.getWriterBuilder(path)}</td></tr>
 *   <tr><td>{@code new OnDiskParallelGraphIndexWriter.Builder(graph, path)} /
 *           {@code getBuilderFor(RANDOM_ACCESS_PARALLEL, ...)}</td>
 *       <td>{@code persistable.getParallelWriterBuilder(path)}</td></tr>
 *   <tr><td>{@code new OnDiskSequentialGraphIndexWriter.Builder(graph, indexWriter)} /
 *           {@code getBuilderFor(ON_DISK_SEQUENTIAL, graph, indexWriter)}</td>
 *       <td>{@code persistable.getWriterBuilder(indexWriter)}</td></tr>
 * </table>
 * The old spellings all still compile; the new ones are thin accessors over the same writer classes.
 */
public class IndexApiExample {
    private static final int VECTOR_COUNT = 20_000;
    private static final int DIMENSION = 64;
    private static final int QUERY_COUNT = 50;
    private static final int TOP_K = 10;
    /** Vectors are drawn around this many random centroids; uniformly random vectors have no structure to index. */
    private static final int CLUSTER_COUNT = 100;
    /**
     * How many candidates the search collects before returning the best {@link #TOP_K}: with an
     * approximate score function these are reranked with higher-precision scores, and with exact
     * scores alone it simply widens the search. Cassandra's "rerankless" mode uses {@link #TOP_K}.
     */
    private static final int RERANK_K = 3 * TOP_K;
    private static final VectorSimilarityFunction SIMILARITY_FUNCTION = VectorSimilarityFunction.EUCLIDEAN;

    // Construction parameters shared by every section. These are the defaults Cassandra falls back to
    // for vectors of more than 3 dimensions (IndexWriterConfig / CompactionGraph).
    private static final int MAX_DEGREE = 32;
    private static final int BEAM_WIDTH = 100;
    private static final float NEIGHBOR_OVERFLOW = 1.2f;
    private static final float ALPHA = 1.2f;
    private static final boolean ADD_HIERARCHY = true;

    // PQ: one subspace per 4 dimensions and 256 centroids per subspace, as both consumers default to.
    private static final int PQ_SUBSPACES = DIMENSION / 4;
    private static final int PQ_CLUSTERS = 256;
    // NVQ: Cassandra's JVectorVersionUtil.NUM_SUB_VECTORS default.
    private static final int NVQ_SUB_VECTORS = 2;

    public static void main(String[] args) throws IOException {
        VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        Random random = new Random(42);
        System.out.printf("Generating %,d random %d-dimensional vectors and %d queries...%n",
                VECTOR_COUNT, DIMENSION, QUERY_COUNT);
        List<VectorFloat<?>> centroids = randomVectors(vts, CLUSTER_COUNT, DIMENSION, random);
        Dataset ds = new Dataset(clusteredVectors(vts, centroids, VECTOR_COUNT, random),
                                 clusteredVectors(vts, centroids, QUERY_COUNT, random));

        Path workDir = Files.createTempDirectory("jvector-index-api-example");
        try {
            section1ApiTour(ds);
            PersistableGraphIndex fullPrecisionGraph = section2FullPrecisionBuild(ds);
            PqBuild pqBuild = section3PqScoredBuild(ds);
            section4WriterTypes(ds, fullPrecisionGraph, workDir);
            section5NonFusedPq(ds, pqBuild, workDir);
            section6FusedPq(ds, pqBuild, workDir);
            section7Nvq(ds, pqBuild, workDir);
            section8DiskToMemory(ds, fullPrecisionGraph, workDir);
            section9LegacyReference(ds.subset(5_000));
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
    // 1. API tour
    // ---------------------------------------------------------------------------------------------

    /**
     * The parts of the new API that don't need a built index: {@link Indexes} picks the backing type
     * first so only that backing's parameters exist on the builder, {@code build()} reports every
     * missing value at once, and recipes and IVF are wired end to end but refuse at runtime until
     * their values/algorithm exist.
     */
    private static void section1ApiTour(Dataset ds) {
        header("1. API tour: Indexes, validation, recipes, IVF");

        // Indexes.hnswBuilder() replaces the old Index.hnswBuilder(): Index lives in jvector-api and
        // can no longer reference a concrete builder, so the factory lives in jvector-base with the
        // builders it constructs. There is no .nlist()/.nprobe() on this builder -- choosing "hnsw"
        // first turns an IVF parameter into a compile error rather than a runtime one.
        System.out.println("build() with nothing set reports every missing value in one exception:");
        expectFailure(() -> Indexes.hnswBuilder().build());

        // The two ways of supplying scoring are mutually exclusive, checked before the missing-value
        // scan.
        System.out.println("Setting both a similarity function and a score provider is rejected:");
        expectFailure(() -> Indexes.hnswBuilder()
                .withVectorValues(ds.ravv)
                .withSimilarityFunction(SIMILARITY_FUNCTION)
                .withScoreProvider(BuildScoreProvider.randomAccessScoreProvider(ds.ravv, SIMILARITY_FUNCTION))
                .build());

        // withDimension() is optional and only a cross-check against withVectorValues().dimension().
        System.out.println("A withDimension() that disagrees with the vectors is rejected:");
        expectFailure(() -> Indexes.hnswBuilder()
                .withVectorValues(ds.ravv)
                .withSimilarityFunction(SIMILARITY_FUNCTION)
                .withDimension(DIMENSION + 1)
                .withMaxDegree(MAX_DEGREE)
                .withBeamWidth(BEAM_WIDTH)
                .withNeighborOverflow(NEIGHBOR_OVERFLOW)
                .withAlpha(ALPHA)
                .withAddHierarchy(ADD_HIERARCHY)
                .build());

        // Recipes (HnswRecipe, IvfRecipe) are named but have no fixed values defined yet.
        System.out.println("Recipes are scaffolding only:");
        expectFailure(() -> Indexes.hnswBuilder().applyRecipe(HnswRecipe.HIGH_RECALL));

        // IVF: the builder validates the inputs every backing shares, then refuses because IVF's own
        // parameters and algorithm don't exist yet. Code written against Index/IvfIndex/IvfSearcher
        // compiles today, which is the point of the seam.
        System.out.println("IVF validates the shared inputs, then refuses to build:");
        expectFailure(() -> Indexes.ivfBuilder().build());
        expectFailure(() -> Indexes.ivfBuilder()
                .withVectorValues(ds.ravv)
                .withSimilarityFunction(SIMILARITY_FUNCTION)
                .build());
    }

    // ---------------------------------------------------------------------------------------------
    // 2. Full-precision build
    // ---------------------------------------------------------------------------------------------

    /**
     * Construction scored by exact comparisons against the raw vectors. This is what Cassandra's
     * memtable index ({@code CassandraOnHeapGraph}) does, and what OpenSearch does for segments below
     * its quantization threshold ({@code JVectorWriter.quantizeForFlush}).
     */
    private static PersistableGraphIndex section2FullPrecisionBuild(Dataset ds) throws IOException {
        header("2. Full-precision build (exact scoring during construction)");

        long start = System.nanoTime();
        // build() returns PersistableGraphIndex: a GraphIndex that can also be written to disk (section 4).
        PersistableGraphIndex graph = Indexes.hnswBuilder()
                .withVectorValues(ds.ravv)                  // always required: drives insertion
                .withSimilarityFunction(SIMILARITY_FUNCTION) // derives an exact BuildScoreProvider
                .withMaxDegree(MAX_DEGREE)
                .withBeamWidth(BEAM_WIDTH)
                .withNeighborOverflow(NEIGHBOR_OVERFLOW)
                .withAlpha(ALPHA)
                .withAddHierarchy(ADD_HIERARCHY)
                // optional, shown with their defaults:
                .withRefineFinalGraph(true)
                .withSimdExecutor(PhysicalCoreExecutor.pool())
                .withParallelExecutor(ForkJoinPool.commonPool())
                .build();
        System.out.printf("Built %,d nodes in %.1fs (%s)%n",
                graph.size(0), (System.nanoTime() - start) / 1e9, graph.getClass().getSimpleName());

        // graph.searcher() is declared to return GraphSearcher (a covariant override of
        // Index.searcher()), so no cast is needed when you hold the concrete GraphIndex.
        try (GraphSearcher searcher = graph.searcher()) {
            report("in-memory, exact", recall(ds, searcher, RERANK_K,
                    q -> DefaultSearchScoreProvider.exact(q, SIMILARITY_FUNCTION, ds.ravv)));
        }

        // Code that is backing-agnostic holds an Index and narrows with instanceof when it needs the
        // concrete type (jvector-api targets Java 11: no sealed interfaces or pattern switches).
        describe(graph);
        return graph;
    }

    /** What shared infrastructure that only holds an {@link Index} looks like. */
    private static void describe(Index index) throws IOException {
        System.out.printf("Generic handle: %s, %,d bytes on heap%n",
                index.getClass().getSimpleName(), index.ramBytesUsed());
        // IndexSearcher is Closeable, so code that only holds an Index can still release the searcher
        // (a GraphSearcher over an on-disk graph holds a file reader) without knowing the backing.
        try (IndexSearcher searcher = index.searcher()) {
            if (index instanceof GraphIndex) {
                GraphIndex graph = (GraphIndex) index;
                System.out.printf("  narrowed to GraphIndex: dimension=%d, maxDegree=%d, hierarchical=%s; searcher is a %s%n",
                        graph.getDimension(), graph.maxDegree(), graph.isHierarchical(),
                        searcher.getClass().getSimpleName());
            } else if (index instanceof IvfIndex) {
                System.out.println("  narrowed to IvfIndex (unreachable until IVF exists)");
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // 3. PQ-scored build
    // ---------------------------------------------------------------------------------------------

    /** A graph built with PQ scoring, plus the codebook and codes used to build it. */
    private static final class PqBuild {
        final PersistableGraphIndex graph;
        final ProductQuantization pq;
        final PQVectors pqVectors;

        PqBuild(PersistableGraphIndex graph, ProductQuantization pq, PQVectors pqVectors) {
            this.graph = graph;
            this.pq = pq;
            this.pqVectors = pqVectors;
        }
    }

    /**
     * Construction scored against PQ codes instead of raw vectors, which is much cheaper per
     * comparison. Cassandra does this during compaction ({@code CompactionGraph}, with a
     * {@code MutablePQVectors} filled in as rows arrive) and OpenSearch does it on flush and merge
     * once a segment passes its quantization threshold. Both then write the graph with a
     * full-precision or NVQ reranking feature (sections 5-7).
     */
    private static PqBuild section3PqScoredBuild(Dataset ds) throws IOException {
        header("3. PQ-scored build (compressed scoring during construction)");

        // OpenSearch centers the data for EUCLIDEAN only; Cassandra passes globallyCenter=false.
        ProductQuantization pq = ProductQuantization.compute(ds.ravv, PQ_SUBSPACES, PQ_CLUSTERS,
                SIMILARITY_FUNCTION == VectorSimilarityFunction.EUCLIDEAN);
        PQVectors pqVectors = pq.encodeAll(ds.ravv, PhysicalCoreExecutor.pool());
        BuildScoreProvider bsp = BuildScoreProvider.pqBuildScoreProvider(SIMILARITY_FUNCTION, pqVectors);
        System.out.printf("PQ: %d subspaces x %d clusters, %d bytes/vector (from %d)%n",
                PQ_SUBSPACES, PQ_CLUSTERS, pq.compressedVectorSize(), DIMENSION * Float.BYTES);

        long start = System.nanoTime();
        PersistableGraphIndex graph = Indexes.hnswBuilder()
                // Required by build() even with a score provider: it drives insertion (node count and
                // the vector for each node). Callers that stream vectors in and never hold them all in
                // one RandomAccessVectorValues -- Cassandra's CompactionGraph -- use buildMutable() with
                // withDimension() instead; see section 9(h).
                .withVectorValues(ds.ravv)
                .withScoreProvider(bsp)
                .withMaxDegree(MAX_DEGREE)
                .withBeamWidth(BEAM_WIDTH)
                .withNeighborOverflow(NEIGHBOR_OVERFLOW)
                .withAlpha(ALPHA)
                .withAddHierarchy(ADD_HIERARCHY)
                .build();
        System.out.printf("Built %,d nodes in %.1fs%n", graph.size(0), (System.nanoTime() - start) / 1e9);

        // In memory: traverse with PQ scores, rerank the top RERANK_K candidates exactly.
        try (GraphSearcher searcher = graph.searcher()) {
            report("in-memory, PQ + exact rerank", recall(ds, searcher, RERANK_K,
                    q -> new DefaultSearchScoreProvider(pqVectors.precomputedScoreFunctionFor(q, SIMILARITY_FUNCTION),
                                                        ds.ravv.rerankerFor(q, SIMILARITY_FUNCTION))));
        }
        return new PqBuild(graph, pq, pqVectors);
    }

    // ---------------------------------------------------------------------------------------------
    // 4. Writer types
    // ---------------------------------------------------------------------------------------------

    /**
     * The three on-disk writer strategies. All three produce the same on-disk graph format, readable
     * with {@link OnDiskGraphIndex#load}; they differ in what they need from the output and how they
     * produce it:
     * <table class="striped">
     *   <caption>Graph writers</caption>
     *   <tr><th>{@code GraphIndexWriterTypes}</th><th>Class</th><th>New accessor</th><th>Output</th><th>Used by</th></tr>
     *   <tr><td>{@code RANDOM_ACCESS}</td><td>{@code OnDiskGraphIndexWriter}</td>
     *       <td>{@code getWriterBuilder(Path)}</td>
     *       <td>{@code RandomAccessWriter}: writes a placeholder header, the node records in order,
     *           then seeks back to fill in the header. Supports {@code withStartOffset} and
     *           per-node {@code writeFeaturesInline}.</td>
     *       <td>Cassandra memtable flush; Cassandra compaction by default</td></tr>
     *   <tr><td>{@code RANDOM_ACCESS_PARALLEL}</td><td>{@code OnDiskParallelGraphIndexWriter}</td>
     *       <td>{@code getParallelWriterBuilder(Path)}</td>
     *       <td>A {@code Path}: node records are encoded on worker threads and written with an
     *           {@code AsynchronousFileChannel}. Worthwhile when feature encoding (e.g. NVQ) dominates.</td>
     *       <td>Cassandra compaction with parallel encoding enabled</td></tr>
     *   <tr><td>{@code ON_DISK_SEQUENTIAL}</td><td>{@code OnDiskSequentialGraphIndexWriter}</td>
     *       <td>{@code getWriterBuilder(IndexWriter)}</td>
     *       <td>Any {@code IndexWriter}: one forward pass, never seeks, header metadata also written
     *           as a footer. For append-only outputs (Lucene {@code IndexOutput}, object storage). Does
     *           not own or flush the output. No {@code withStartOffset}: it starts wherever the
     *           output is positioned. Cannot write ordinal "holes".</td>
     *       <td>OpenSearch (every write goes through a Lucene {@code IndexOutput})</td></tr>
     * </table>
     * Also available: {@code RandomAccessOnDiskGraphIndexWriter.Builder}, which picks between the
     * first two at build time from the JMX {@code GraphIndexBuilderConfig.isParallelBuild()} setting.
     * <p>
     * Options that apply to all: {@code with(Feature)} for what each node record carries,
     * {@code withMapper}/{@code withMap} to renumber ordinals on the way out (Cassandra maps graph
     * ordinals to row ids; the default compacts away deleted nodes), and {@code withVersion} to write
     * an older format for mixed-version clusters.
     */
    private static void section4WriterTypes(Dataset ds, PersistableGraphIndex persistable, Path workDir) throws IOException {
        header("4. Writer types (persisting an in-memory graph)");

        // The getXWriterBuilder accessors live on PersistableGraphIndex, implemented by OnHeapGraphIndex
        // and OnDiskGraphIndex. Indexes.hnswBuilder().build() and MutableHnswIndex.graph() both return
        // that type, so the result of a build can be written without a cast.
        EnumMap<FeatureId, IntFunction<Feature.State>> inlineVectors = Feature.singleStateFactory(
                FeatureId.INLINE_VECTORS, node -> new InlineVectors.State(ds.ravv.getVector(node)));

        // RANDOM_ACCESS. Previously: new OnDiskGraphIndexWriter.Builder(graph, path), or
        // GraphIndexWriter.getBuilderFor(GraphIndexWriterTypes.RANDOM_ACCESS, graph, path).
        Path randomAccessPath = workDir.resolve("random-access.graph");
        try (GraphIndexWriter writer = persistable.getWriterBuilder(randomAccessPath)
                .with(new InlineVectors(DIMENSION))
                .build()) {
            writer.write(inlineVectors);
        }
        searchInlineVectorsGraph(ds, randomAccessPath, "RANDOM_ACCESS");

        // RANDOM_ACCESS_PARALLEL. Previously: new OnDiskParallelGraphIndexWriter.Builder(graph, path),
        // or getBuilderFor(RANDOM_ACCESS_PARALLEL, ...). Worker threads: 0 = available processors.
        // withExecutor(ExecutorService) is only on the concrete builder, not the accessor's interface.
        Path parallelPath = workDir.resolve("parallel.graph");
        try (GraphIndexWriter writer = persistable.getParallelWriterBuilder(parallelPath)
                .with(new InlineVectors(DIMENSION))
                .withParallelWorkerThreads(0)
                .withParallelDirectBuffers(false)
                .build()) {
            writer.write(inlineVectors);
        }
        searchInlineVectorsGraph(ds, parallelPath, "RANDOM_ACCESS_PARALLEL");

        // ON_DISK_SEQUENTIAL. Previously: new OnDiskSequentialGraphIndexWriter.Builder(graph, out), or
        // getBuilderFor(ON_DISK_SEQUENTIAL, graph, out). The caller owns the IndexWriter; closing the
        // graph writer does not close or flush it. OpenSearch passes a JVectorIndexWriter wrapping a
        // Lucene IndexOutput here, after writing its own codec header.
        Path sequentialPath = workDir.resolve("sequential.graph");
        try (SimpleWriter out = new SimpleWriter(sequentialPath)) {
            try (GraphIndexWriter writer = persistable.getWriterBuilder(out)
                    .with(new InlineVectors(DIMENSION))
                    .build()) {
                writer.write(inlineVectors);
            }
        }
        searchInlineVectorsGraph(ds, sequentialPath, "ON_DISK_SEQUENTIAL");

        // Embedding at an offset, as Cassandra does: the graph shares the SAI TERMS_DATA file with a
        // codec header in front of it (and other segments' graphs before that). Cassandra also needs
        // writer.getOutput() and writer.checksum(), which only exist on the concrete writer type, so
        // this is one place where the direct builder is still the natural spelling; the accessor's
        // GraphIndexWriter return type would need a cast.
        Path embeddedPath = workDir.resolve("embedded.graph");
        byte[] prefix = "SAI-HEADER-STANDIN".getBytes(StandardCharsets.US_ASCII);
        try (OnDiskGraphIndexWriter writer = new OnDiskGraphIndexWriter.Builder(persistable, embeddedPath)
                .withStartOffset(prefix.length)
                .withVersion(OnDiskGraphIndex.CURRENT_VERSION)
                .with(new InlineVectors(DIMENSION))
                .build()) {
            RandomAccessWriter out = writer.getOutput();
            out.seek(0);
            out.write(prefix);
            writer.write(inlineVectors);
            System.out.printf("Embedded graph written after a %d-byte prefix, checksum=%x%n",
                    prefix.length, writer.checksum());
        }
        // Cassandra loads with useFooter=false, locating the graph by the offset it stored itself.
        try (ReaderSupplier rs = ReaderSupplierFactory.open(embeddedPath);
             OnDiskGraphIndex onDisk = OnDiskGraphIndex.load(rs, prefix.length, false);
             GraphSearcher searcher = onDisk.searcher()) {
            report("embedded at offset " + prefix.length, recall(ds, searcher, RERANK_K,
                    q -> new DefaultSearchScoreProvider(scoringView(searcher).rerankerFor(q, SIMILARITY_FUNCTION))));
        }

        // An OnDiskGraphIndex is persistable too, so a loaded graph can be rewritten: here into the
        // sequential format, taking the inline vectors from the source graph's own view.
        Path rewrittenPath = workDir.resolve("rewritten.graph");
        try (ReaderSupplier rs = ReaderSupplierFactory.open(randomAccessPath);
             OnDiskGraphIndex source = OnDiskGraphIndex.load(rs);
             OnDiskGraphIndex.View sourceView = source.getView();
             SimpleWriter out = new SimpleWriter(rewrittenPath)) {
            try (GraphIndexWriter writer = source.getWriterBuilder(out)
                    .with(new InlineVectors(DIMENSION))
                    .build()) {
                writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                        node -> new InlineVectors.State(sourceView.getVector(node))));
            }
        }
        searchInlineVectorsGraph(ds, rewrittenPath, "on-disk graph rewritten sequentially");
    }

    /** Loads a graph with inline full-precision vectors and searches it with exact scores only. */
    private static void searchInlineVectorsGraph(Dataset ds, Path path, String label) throws IOException {
        try (ReaderSupplier rs = ReaderSupplierFactory.open(path);
             OnDiskGraphIndex onDisk = OnDiskGraphIndex.load(rs);
             GraphSearcher searcher = onDisk.searcher()) {
            // Cassandra's "no compression" path: the view's reranker reads the inline vectors.
            report(label + String.format(" (%,d bytes)", Files.size(path)), recall(ds, searcher, RERANK_K,
                    q -> new DefaultSearchScoreProvider(scoringView(searcher).rerankerFor(q, SIMILARITY_FUNCTION))));
        }
    }

    // ---------------------------------------------------------------------------------------------
    // 5. Non-fused PQ
    // ---------------------------------------------------------------------------------------------

    /**
     * Full-precision vectors inline in the graph for reranking, with the PQ codes stored separately
     * and loaded into memory for traversal. OpenSearch's PQ layout appends the {@link PQVectors} right
     * after the graph in the same Lucene file and loads them from a slice; Cassandra's pre-fused
     * formats keep them in a separate PQ component. A separate file here, for clarity.
     */
    private static void section5NonFusedPq(Dataset ds, PqBuild pqBuild, Path workDir) throws IOException {
        header("5. Non-fused PQ: inline vectors + separate PQ codes");

        Path graphPath = workDir.resolve("pq.graph");
        Path pqPath = workDir.resolve("pq.codes");
        try (GraphIndexWriter writer = pqBuild.graph.getWriterBuilder(graphPath)
                .with(new InlineVectors(DIMENSION))
                .build()) {
            writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                    node -> new InlineVectors.State(ds.ravv.getVector(node))));
        }
        try (SimpleWriter out = new SimpleWriter(pqPath)) {
            pqBuild.pqVectors.write(out, OnDiskGraphIndex.CURRENT_VERSION);
        }
        System.out.printf("graph %,d bytes, PQ codes %,d bytes%n", Files.size(graphPath), Files.size(pqPath));

        PQVectors loadedPq;
        try (ReaderSupplier rs = ReaderSupplierFactory.open(pqPath);
             var reader = rs.get()) {
            loadedPq = PQVectors.load(reader);
        }
        try (ReaderSupplier rs = ReaderSupplierFactory.open(graphPath);
             OnDiskGraphIndex onDisk = OnDiskGraphIndex.load(rs);
             GraphSearcher searcher = onDisk.searcher()) {
            // Traverse with in-memory PQ, rerank with the inline vectors (read from disk).
            // Cassandra switches DOT_PRODUCT to COSINE for the PQ scores of unit vectors, since PQ
            // does not preserve unit length; the reranker keeps the original function.
            report("PQ traversal + inline rerank", recall(ds, searcher, RERANK_K,
                    q -> new DefaultSearchScoreProvider(loadedPq.precomputedScoreFunctionFor(q, SIMILARITY_FUNCTION),
                                                        scoringView(searcher).rerankerFor(q, SIMILARITY_FUNCTION))));
            // Cassandra's "rerankless" mode (rerankK <= 0): no reranker, rerankK = limit, and the
            // approximate scores are returned as-is. Cheaper, and recall drops accordingly.
            report("PQ traversal, rerankless", recall(ds, searcher, TOP_K,
                    q -> new DefaultSearchScoreProvider(loadedPq.precomputedScoreFunctionFor(q, SIMILARITY_FUNCTION))));
        }
    }

    // ---------------------------------------------------------------------------------------------
    // 6. Fused PQ
    // ---------------------------------------------------------------------------------------------

    /**
     * {@link FusedPQ} stores, in each node's record, the PQ codes of that node's neighbors, so
     * traversal reads codes from the same page as the adjacency list and nothing needs to be loaded
     * into memory up front. Cassandra writes this for current on-disk versions
     * ({@code JVectorVersionUtil.shouldWriteFused}). The feature needs the graph's max degree and the
     * codebook; each node's state is built from a view of the graph being written.
     */
    private static void section6FusedPq(Dataset ds, PqBuild pqBuild, Path workDir) throws IOException {
        header("6. Fused PQ: neighbor codes packed into each node record");

        Path graphPath = workDir.resolve("fused.graph");
        Path codebookPath = workDir.resolve("fused.codebook");
        try (GraphIndex.View view = pqBuild.graph.getView();
             GraphIndexWriter writer = pqBuild.graph.getWriterBuilder(graphPath)
                     .with(new InlineVectors(DIMENSION))
                     .with(new FusedPQ(pqBuild.graph.maxDegree(), pqBuild.pq))
                     .build()) {
            var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
            suppliers.put(FeatureId.INLINE_VECTORS, node -> new InlineVectors.State(ds.ravv.getVector(node)));
            suppliers.put(FeatureId.FUSED_PQ, node -> new FusedPQ.State(view, pqBuild.pqVectors, node));
            writer.write(suppliers);
        }
        // Cassandra still writes the codebook (only) to its PQ component so a later compaction can
        // refine() it rather than retrain from scratch. Search does not need it.
        try (SimpleWriter out = new SimpleWriter(codebookPath)) {
            pqBuild.pq.write(out, OnDiskGraphIndex.CURRENT_VERSION);
        }
        System.out.printf("graph %,d bytes (codes are inside it), codebook %,d bytes%n",
                Files.size(graphPath), Files.size(codebookPath));

        try (ReaderSupplier rs = ReaderSupplierFactory.open(graphPath);
             OnDiskGraphIndex onDisk = OnDiskGraphIndex.load(rs);
             GraphSearcher searcher = onDisk.searcher()) {
            System.out.println("Loaded features: " + onDisk.getFeatureSet()
                    + "; the codebook is also recoverable via ((FusedPQ) getFeatures().get(FUSED_PQ)).getPQ()");
            // The same call sites as Cassandra's CassandraDiskAnn.search: approximate scores come from
            // the view (fused codes), the reranker from the inline vectors.
            report("fused PQ traversal + inline rerank", recall(ds, searcher, RERANK_K, q -> {
                GraphIndex.ScoringView view = scoringView(searcher);
                return new DefaultSearchScoreProvider(view.approximateScoreFunctionFor(q, SIMILARITY_FUNCTION),
                                                      view.rerankerFor(q, SIMILARITY_FUNCTION));
            }));
            report("fused PQ traversal, rerankless", recall(ds, searcher, TOP_K,
                    q -> new DefaultSearchScoreProvider(scoringView(searcher).approximateScoreFunctionFor(q, SIMILARITY_FUNCTION))));
        }
    }

    // ---------------------------------------------------------------------------------------------
    // 7. NVQ
    // ---------------------------------------------------------------------------------------------

    /**
     * {@link NVQ} replaces full-precision inline vectors with NVQ-compressed ones (roughly 4x smaller),
     * so the reranker reads less from disk at the cost of approximate final scores. NVQ is only a
     * reranking feature; traversal still needs PQ, which the two consumers supply differently.
     */
    private static void section7Nvq(Dataset ds, PqBuild pqBuild, Path workDir) throws IOException {
        header("7. NVQ: compressed inline vectors for reranking");

        // Cassandra computes NVQ at write time. On memtable flush it is NVQuantization.compute(ravv, n);
        // during compaction it accumulates a global mean as rows stream in and uses
        // NVQuantization.create(globalMean, n) instead, so no second pass over the vectors is needed.
        NVQuantization nvq = NVQuantization.compute(ds.ravv, NVQ_SUB_VECTORS);

        // --- Cassandra layout: NVQ + fused PQ, parallel writer ---
        Path cassandraPath = workDir.resolve("nvq-fused.graph");
        try (GraphIndex.View view = pqBuild.graph.getView();
             GraphIndexWriter writer = pqBuild.graph.getParallelWriterBuilder(cassandraPath)
                     .with(new NVQ(nvq))
                     .with(new FusedPQ(pqBuild.graph.maxDegree(), pqBuild.pq))
                     .build()) {
            var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
            suppliers.put(FeatureId.NVQ_VECTORS, node -> new NVQ.State(nvq.encode(ds.ravv.getVector(node))));
            suppliers.put(FeatureId.FUSED_PQ, node -> new FusedPQ.State(view, pqBuild.pqVectors, node));
            writer.write(suppliers);
        }
        try (ReaderSupplier rs = ReaderSupplierFactory.open(cassandraPath);
             OnDiskGraphIndex onDisk = OnDiskGraphIndex.load(rs);
             GraphSearcher searcher = onDisk.searcher()) {
            // view.rerankerFor dispatches on the features present: here it scores against NVQ.
            report(String.format("Cassandra: NVQ + fused PQ (%,d bytes)", Files.size(cassandraPath)),
                    recall(ds, searcher, RERANK_K, q -> {
                        GraphIndex.ScoringView view = scoringView(searcher);
                        return new DefaultSearchScoreProvider(view.approximateScoreFunctionFor(q, SIMILARITY_FUNCTION),
                                                              view.rerankerFor(q, SIMILARITY_FUNCTION));
                    }));
        }

        // --- OpenSearch layout: NVQ inline + auxiliary PQ blob, sequential writer ---
        // OpenSearch's NVQ quantization builds the graph with an auxiliary PQ (as in section 3),
        // encodes every vector with NVQ up front, and appends the aux PQVectors after the graph in the
        // same Lucene file. Separate files here.
        NVQVectors nvqVectors = nvq.encodeAll(ds.ravv, PhysicalCoreExecutor.pool());
        Path openSearchPath = workDir.resolve("nvq-seq.graph");
        Path auxPqPath = workDir.resolve("nvq-seq.auxpq");
        try (SimpleWriter out = new SimpleWriter(openSearchPath)) {
            try (GraphIndexWriter writer = pqBuild.graph.getWriterBuilder(out)
                    .with(new NVQ(nvqVectors.getNVQuantization()))
                    .build()) {
                writer.write(Feature.singleStateFactory(FeatureId.NVQ_VECTORS,
                        node -> new NVQ.State(nvqVectors.get(node))));
            }
        }
        try (SimpleWriter out = new SimpleWriter(auxPqPath)) {
            pqBuild.pqVectors.write(out, OnDiskGraphIndex.CURRENT_VERSION);
        }
        PQVectors auxPq;
        try (ReaderSupplier rs = ReaderSupplierFactory.open(auxPqPath);
             var reader = rs.get()) {
            auxPq = PQVectors.load(reader);
        }
        try (ReaderSupplier rs = ReaderSupplierFactory.open(openSearchPath);
             OnDiskGraphIndex onDisk = OnDiskGraphIndex.load(rs);
             GraphSearcher searcher = onDisk.searcher()) {
            // The same branches as JVectorReader.FieldEntry.buildScoreFunctionProvider.
            report(String.format("OpenSearch: NVQ + aux PQ (%,d bytes)", Files.size(openSearchPath)),
                    recall(ds, searcher, RERANK_K,
                            q -> new DefaultSearchScoreProvider(auxPq.precomputedScoreFunctionFor(q, SIMILARITY_FUNCTION),
                                                                scoringView(searcher).rerankerFor(q, SIMILARITY_FUNCTION))));
            report("OpenSearch: NVQ only, no aux PQ", recall(ds, searcher, RERANK_K,
                    q -> new DefaultSearchScoreProvider(scoringView(searcher).rerankerFor(q, SIMILARITY_FUNCTION))));
        }
    }

    // ---------------------------------------------------------------------------------------------
    // 8. Disk to memory, memory to disk
    // ---------------------------------------------------------------------------------------------

    /**
     * Two different things are meant by "loading a graph from disk":
     * <ol>
     *     <li><b>For searching.</b> {@link OnDiskGraphIndex#load} keeps only the header, the upper
     *     (hierarchy) layers, and for fused graphs their fused features in memory; base-layer records
     *     are read on demand. This is Cassandra's {@code CassandraDiskAnn} and OpenSearch's
     *     {@code JVectorReader}. The result is immutable.</li>
     *     <li><b>For further construction.</b> A mutable {@link OnHeapGraphIndex} is written with
     *     {@code save} and read back with {@code OnHeapGraphIndex.load}, then extended. That is
     *     OpenSearch's leading-segment merge: it saves every flushed graph as a "neighbors score cache"
     *     file, and when merging reloads the largest segment's graph and inserts only the other
     *     segments' vectors instead of rebuilding. There is no API that turns an
     *     {@link OnDiskGraphIndex} into an {@link OnHeapGraphIndex}: the search format drops the
     *     neighbor scores the builder needs, so the save format is the only route back. Cassandra
     *     does not do this today; its compaction rebuilds the graph from the merged vectors.</li>
     * </ol>
     */
    @SuppressWarnings("deprecation") // OnHeapGraphIndex.save/load are @Deprecated @Experimental
    private static void section8DiskToMemory(Dataset ds, PersistableGraphIndex fullPrecisionGraph, Path workDir) throws IOException {
        header("8. Disk to memory, and memory to disk");

        // --- (a) memory -> disk -> memory for search: compare what each representation keeps on heap
        Path searchPath = workDir.resolve("search.graph");
        try (GraphIndexWriter writer = fullPrecisionGraph.getWriterBuilder(searchPath)
                .with(new InlineVectors(DIMENSION))
                .build()) {
            writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                    node -> new InlineVectors.State(ds.ravv.getVector(node))));
        }
        try (ReaderSupplier rs = ReaderSupplierFactory.open(searchPath);
             OnDiskGraphIndex onDisk = OnDiskGraphIndex.load(rs)) {
            System.out.printf("On-heap graph: %,d bytes of heap. Same graph loaded for search: %,d bytes of heap, %,d on disk%n",
                    fullPrecisionGraph.ramBytesUsed(), onDisk.ramBytesUsed(), Files.size(searchPath));
        }

        // --- (b) memory -> disk -> mutable memory, then append (OpenSearch leading-segment merge)
        int baseCount = ds.ravv.size() * 3 / 4;
        var baseRavv = new ListRandomAccessVectorValues(ds.vectors.subList(0, baseCount), DIMENSION);
        PersistableGraphIndex baseGraph = Indexes.hnswBuilder()
                .withVectorValues(baseRavv)
                .withSimilarityFunction(SIMILARITY_FUNCTION)
                .withMaxDegree(MAX_DEGREE)
                .withBeamWidth(BEAM_WIDTH)
                .withNeighborOverflow(NEIGHBOR_OVERFLOW)
                .withAlpha(ALPHA)
                .withAddHierarchy(ADD_HIERARCHY)
                .build();

        // save() is on OnHeapGraphIndex only (not PersistableGraphIndex); OpenSearch writes it to a
        // Lucene IndexOutput after flushing each segment's search-format graph.
        Path savedPath = workDir.resolve("base.onheap");
        try (SimpleWriter out = new SimpleWriter(savedPath)) {
            ((OnHeapGraphIndex) baseGraph).save(out);
        }
        System.out.printf("Saved a %,d-node mutable graph (%,d bytes)%n", baseGraph.size(0), Files.size(savedPath));

        // The score provider and RAVV cover the base vectors at their original ordinals plus the new
        // vectors after them. OpenSearch has to remap here (its heap graph accumulates ordinal holes
        // from deletes); in this example the ordinals already line up.
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ds.ravv, SIMILARITY_FUNCTION);
        PersistableGraphIndex extended;
        try (ReaderSupplier rs = ReaderSupplierFactory.open(savedPath);
             var reader = rs.get()) {
            // load() requires the DiversityProvider used when mutating the graph. OpenSearch passes a
            // DelayedInitDiversityProvider because its remapping isn't known until after the load.
            OnHeapGraphIndex loaded = OnHeapGraphIndex.load(reader, DIMENSION, NEIGHBOR_OVERFLOW,
                                                            new VamanaDiversityProvider(bsp, ALPHA));
            System.out.printf("Reloaded %,d nodes into a mutable graph; appending %,d more%n",
                    loaded.size(0), ds.ravv.size() - loaded.getIdUpperBound());

            // Before: new GraphIndexBuilder(bsp, dim, loaded, beamWidth, overflow, alpha, true,
            // simdPool, parallelPool), then addGraphNode(ord, vector) for every ord from
            // loaded.getIdUpperBound() up, then cleanup(). HnswIndexBuilder does exactly that.
            // No maxDegree or addHierarchy: both come from the existing graph.
            long start = System.nanoTime();
            extended = Indexes.hnswBuilder()
                    .withExistingGraph(loaded)
                    .withVectorValues(ds.ravv)   // superset: [0, baseCount) existing, the rest appended
                    .withScoreProvider(bsp)
                    .withBeamWidth(BEAM_WIDTH)
                    .withNeighborOverflow(NEIGHBOR_OVERFLOW)
                    .withAlpha(ALPHA)
                    .build();
            System.out.printf("Extended to %,d nodes in %.1fs%n", extended.size(0), (System.nanoTime() - start) / 1e9);
        }
        // OpenSearch also marks the leading segment's deleted docs with markNodeDeleted before cleanup().
        // build() is the shortcut for "append everything, then clean up"; to delete as well, call
        // buildMutable() on the same configuration, addNode the new ordinals, markDeleted, then
        // cleanup() -- section 9(g) shows that pattern.
        try (GraphSearcher searcher = extended.searcher()) {
            report("reloaded + extended, in memory", recall(ds, searcher, RERANK_K,
                    q -> DefaultSearchScoreProvider.exact(q, SIMILARITY_FUNCTION, ds.ravv)));
        }

        // ...and the extended graph is persistable like any other in-memory graph.
        Path extendedPath = workDir.resolve("extended.graph");
        try (GraphIndexWriter writer = extended.getWriterBuilder(extendedPath)
                .with(new InlineVectors(DIMENSION))
                .build()) {
            writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                    node -> new InlineVectors.State(ds.ravv.getVector(node))));
        }
        searchInlineVectorsGraph(ds, extendedPath, "reloaded + extended, written back to disk");
    }

    // ---------------------------------------------------------------------------------------------
    // 9. Legacy construction reference
    // ---------------------------------------------------------------------------------------------

    /**
     * Every construction path in use before this branch, each run next to its new-API counterpart
     * with the same parameters. Recall won't match to the last digit (concurrent insertion order is
     * nondeterministic) but should be equivalent.
     */
    @SuppressWarnings("deprecation") // the legacy GraphIndexBuilder constructors are @Deprecated
    private static void section9LegacyReference(Dataset ds) throws IOException {
        header(String.format("9. Legacy construction reference (%,d vectors)", ds.ravv.size()));
        BuildScoreProvider exactBsp = BuildScoreProvider.randomAccessScoreProvider(ds.ravv, SIMILARITY_FUNCTION);

        // (a) Raw vectors + similarity function, batch build(ravv).
        //     Cassandra's VectorTester and most JVector examples/benchmarks.
        GraphIndex legacy;
        try (GraphIndexBuilder builder = new GraphIndexBuilder(ds.ravv, SIMILARITY_FUNCTION,
                MAX_DEGREE, BEAM_WIDTH, NEIGHBOR_OVERFLOW, ALPHA, ADD_HIERARCHY)) {
            legacy = builder.build(ds.ravv);
        }
        compare(ds, "(a) new GraphIndexBuilder(ravv, vsf, M, beam, overflow, alpha, hierarchy).build(ravv)", legacy,
                "Indexes.hnswBuilder().withVectorValues(ravv).withSimilarityFunction(vsf)...",
                baseBuilder(ds).withSimilarityFunction(SIMILARITY_FUNCTION).build());

        // (b) Score provider + explicit parallel addGraphNode loop + cleanup().
        //     OpenSearch JVectorWriter.getGraph (with a PQ or exact BuildScoreProvider).
        try (GraphIndexBuilder builder = new GraphIndexBuilder(exactBsp, DIMENSION,
                MAX_DEGREE, BEAM_WIDTH, NEIGHBOR_OVERFLOW, ALPHA, ADD_HIERARCHY)) {
            var vv = ds.ravv.threadLocalSupplier();
            PhysicalCoreExecutor.pool().submit(() -> IntStream.range(0, ds.ravv.size()).parallel()
                    .forEach(ord -> builder.addGraphNode(ord, vv.get().getVector(ord)))).join();
            builder.cleanup();
            legacy = builder.getGraph();
        }
        compare(ds, "(b) new GraphIndexBuilder(bsp, dim, ...) + addGraphNode loop + cleanup()", legacy,
                "...withVectorValues(ravv).withScoreProvider(bsp)...",
                baseBuilder(ds).withScoreProvider(exactBsp).build());

        // (c) Full control: refineFinalGraph and both executors. Cassandra CompactionGraph, which
        //     builds on dedicated compaction pools.
        ForkJoinPool simdPool = new ForkJoinPool(2);
        ForkJoinPool parallelPool = new ForkJoinPool(2);
        try {
            try (GraphIndexBuilder builder = new GraphIndexBuilder(exactBsp, DIMENSION, MAX_DEGREE, BEAM_WIDTH,
                    NEIGHBOR_OVERFLOW, ALPHA, ADD_HIERARCHY, true, simdPool, parallelPool)) {
                legacy = builder.build(ds.ravv);
            }
            compare(ds, "(c) 10-arg constructor (refineFinalGraph, simd + parallel pools)", legacy,
                    "...withRefineFinalGraph(true).withSimdExecutor(p).withParallelExecutor(p)",
                    baseBuilder(ds).withScoreProvider(exactBsp)
                            .withRefineFinalGraph(true)
                            .withSimdExecutor(simdPool)
                            .withParallelExecutor(parallelPool)
                            .build());
        } finally {
            simdPool.shutdown();
            parallelPool.shutdown();
        }

        // (d) Per-layer max degrees.
        List<Integer> maxDegrees = List.of(MAX_DEGREE, MAX_DEGREE / 2);
        try (GraphIndexBuilder builder = new GraphIndexBuilder(exactBsp, DIMENSION, maxDegrees, BEAM_WIDTH,
                NEIGHBOR_OVERFLOW, ALPHA, ADD_HIERARCHY, true)) {
            legacy = builder.build(ds.ravv);
        }
        compare(ds, "(d) new GraphIndexBuilder(bsp, dim, List<Integer> maxDegrees, ...)", legacy,
                "...withMaxDegrees(List.of(32, 16))",
                baseBuilder(ds).withScoreProvider(exactBsp).withMaxDegrees(maxDegrees).build());

        // (e) GraphIndexBuilder.builder(): the non-deprecated fluent builder from 4.0.x. It reads
        //     addHierarchy, refineFinalGraph and build-time compression from the JMX-managed
        //     GraphIndexBuilderConfig; HnswIndexBuilder takes them explicitly instead.
        try (GraphIndexBuilder builder = GraphIndexBuilder.builder(ds.ravv, SIMILARITY_FUNCTION, MAX_DEGREE)
                .withBeamWidth(BEAM_WIDTH)
                .withNeighborOverflow(NEIGHBOR_OVERFLOW)
                .withAlpha(ALPHA)
                .build()) {
            legacy = builder.build(ds.ravv);
        }
        compare(ds, "(e) GraphIndexBuilder.builder(ravv, vsf, M)...build().build(ravv)", legacy,
                "Indexes.hnswBuilder()... (addHierarchy/refine explicit, not JMX)",
                baseBuilder(ds).withSimilarityFunction(SIMILARITY_FUNCTION).build());

        // (f) Existing graph + appended nodes, and GraphIndexBuilder.buildAndMergeNewNodes: see
        //     section 8. Both become OnHeapGraphIndex.load(...) + withExistingGraph(...).
        System.out.println("(f) new GraphIndexBuilder(bsp, dim, existingGraph, ...) / buildAndMergeNewNodes");
        System.out.println("      -> OnHeapGraphIndex.load(...) + Indexes.hnswBuilder().withExistingGraph(...)  [section 8]");

        // (g) Incremental construction with deletes. Cassandra's memtable index (CassandraOnHeapGraph)
        //     adds each row as it is written, searches concurrently, deletes with markNodeDeleted, and
        //     calls cleanup() before flush -- with a lower neighborOverflow for faster flushes.
        try (GraphIndexBuilder builder = new GraphIndexBuilder(ds.ravv, SIMILARITY_FUNCTION,
                MAX_DEGREE, BEAM_WIDTH, 1.0f, ALPHA, ADD_HIERARCHY)) {
            IntStream.range(0, ds.ravv.size()).parallel().forEach(ord -> builder.addGraphNode(ord, ds.ravv.getVector(ord)));
            IntStream.range(0, ds.ravv.size()).filter(ord -> ord % 100 == 0).forEach(builder::markNodeDeleted);
            builder.cleanup(); // removes the deleted nodes and enforces max degree before writing
            legacy = builder.getGraph();
        }
        GraphIndex modern;
        try (MutableHnswIndex index = baseBuilder(ds)
                .withSimilarityFunction(SIMILARITY_FUNCTION)
                .withNeighborOverflow(1.0f)
                .buildMutable()) {                                       // nothing inserted yet
            IntStream.range(0, ds.ravv.size()).parallel().forEach(ord -> index.addNode(ord, ds.ravv.getVector(ord)));
            IntStream.range(0, ds.ravv.size()).filter(ord -> ord % 100 == 0).forEach(index::markDeleted);
            index.cleanup();
            modern = index.graph();                                      // still usable after close()
        }
        compare(ds, "(g) new GraphIndexBuilder(ravv, vsf, ...) + addGraphNode + markNodeDeleted + cleanup()", legacy,
                "...buildMutable() + addNode + markDeleted + cleanup()", modern);

        // (h) Streaming compaction with a mid-build PQ codebook refinement, as Cassandra's
        //     CompactionGraph does: vectors arrive one at a time, are PQ-encoded into MutablePQVectors,
        //     and inserted on a pool; once enough have arrived the codebook is refined, the codes so far
        //     re-encoded, and the graph rescored against them. Previously the caller guarded
        //     addGraphNode and the codebook swap with its own ReadWriteLock; MutableHnswIndex.rescore
        //     runs the swap with inserts locked out itself.
        compare(ds, "(h) GraphIndexBuilder + caller's ReadWriteLock + GraphIndexBuilder.rescore(builder, bsp)",
                legacyStreamingCompaction(ds),
                "...withScoreProvider(bsp).withDimension(d).buildMutable() + rescore(() -> refine...)",
                streamingCompaction(ds));
        System.out.println("      Once a MutableHnswIndex is built, nothing is left without a new-API counterpart:");
        System.out.println("      its graph() is a PersistableGraphIndex, so no cast is needed to write it.");

        // Searching: the static convenience and new GraphSearcher(graph) both still work;
        // graph.searcher() is the Index-hierarchy spelling of the latter.
        GraphIndex graph = baseBuilder(ds).withSimilarityFunction(SIMILARITY_FUNCTION).build();
        VectorFloat<?> q = ds.queries.get(0);
        SearchResult legacyResult = GraphSearcher.search(q, TOP_K, ds.ravv, SIMILARITY_FUNCTION, graph, Bits.ALL);
        SearchResult modernResult;
        try (GraphSearcher searcher = graph.searcher()) {
            modernResult = searcher.search(DefaultSearchScoreProvider.exact(q, SIMILARITY_FUNCTION, ds.ravv), TOP_K, Bits.ALL);
        }
        System.out.printf("Search: GraphSearcher.search(q, k, ravv, vsf, graph, bits) and graph.searcher().search(ssp, k, bits) "
                + "agree on the top result: %s%n", legacyResult.getNodes()[0].node == modernResult.getNodes()[0].node);
    }

    /**
     * Section 9(h), new API: Cassandra's compaction pattern on {@link MutableHnswIndex}. One thread
     * reads vectors, encodes them, and hands each to a pool for insertion; halfway through it refines
     * the PQ codebook. The supplier passed to {@code rescore} runs with every insert locked out, so it
     * can replace the codes that the old score provider reads.
     */
    private static GraphIndex streamingCompaction(Dataset ds) throws IOException {
        int n = ds.ravv.size();
        ProductQuantization initialPq = ProductQuantization.compute(
                new ListRandomAccessVectorValues(ds.vectors.subList(0, n / 10), DIMENSION), PQ_SUBSPACES, PQ_CLUSTERS, false);
        var codes = new AtomicReference<>(new MutablePQVectors(initialPq));

        try (MutableHnswIndex index = Indexes.hnswBuilder()
                .withScoreProvider(BuildScoreProvider.pqBuildScoreProvider(SIMILARITY_FUNCTION, codes.get()))
                .withDimension(DIMENSION)   // no RandomAccessVectorValues: vectors arrive one at a time
                .withMaxDegree(MAX_DEGREE)
                .withBeamWidth(BEAM_WIDTH)
                .withNeighborOverflow(NEIGHBOR_OVERFLOW)
                .withAlpha(ALPHA)
                .withAddHierarchy(ADD_HIERARCHY)
                .buildMutable()) {
            // Cassandra inserts from its compaction threads, not from the pools the builder uses.
            ExecutorService insertThreads = Executors.newFixedThreadPool(4);
            List<Future<?>> inserts = new ArrayList<>(n);
            for (int ord = 0; ord < n; ord++) {
                VectorFloat<?> v = ds.ravv.getVector(ord);
                codes.get().encodeAndSet(ord, v);
                int o = ord;
                inserts.add(insertThreads.submit(() -> index.addNode(o, v)));

                if (ord == n / 2) {
                    int encodedSoFar = ord + 1;
                    index.rescore(() -> {
                        var trainingVectors = new ListRandomAccessVectorValues(ds.vectors.subList(0, encodedSoFar), DIMENSION);
                        MutablePQVectors refined = new MutablePQVectors(initialPq.refine(trainingVectors));
                        for (int i = 0; i < encodedSoFar; i++) {
                            refined.encodeAndSet(i, ds.ravv.getVector(i));
                        }
                        codes.set(refined);
                        return BuildScoreProvider.pqBuildScoreProvider(SIMILARITY_FUNCTION, refined);
                    });
                }
            }
            awaitAll(inserts);
            insertThreads.shutdown();
            index.cleanup();
            return index.graph();
        }
    }

    /** Section 9(h), before: the same pattern on {@link GraphIndexBuilder}, with the caller's own lock. */
    @SuppressWarnings("deprecation")
    private static GraphIndex legacyStreamingCompaction(Dataset ds) throws IOException {
        int n = ds.ravv.size();
        ProductQuantization initialPq = ProductQuantization.compute(
                new ListRandomAccessVectorValues(ds.vectors.subList(0, n / 10), DIMENSION), PQ_SUBSPACES, PQ_CLUSTERS, false);
        var codes = new AtomicReference<>(new MutablePQVectors(initialPq));
        var trainingLock = new ReentrantReadWriteLock();
        var builder = new AtomicReference<>(new GraphIndexBuilder(
                BuildScoreProvider.pqBuildScoreProvider(SIMILARITY_FUNCTION, codes.get()), DIMENSION,
                MAX_DEGREE, BEAM_WIDTH, NEIGHBOR_OVERFLOW, ALPHA, ADD_HIERARCHY, true,
                PhysicalCoreExecutor.pool(), ForkJoinPool.commonPool()));

        // Inserts must not run on a pool that the locked section also uses (refine() and rescore()
        // both run parallel work on the common pool): workers parked on this plain lock would leave
        // nothing to run it, and the build deadlocks. MutableHnswIndex waits cooperatively instead.
        ExecutorService insertThreads = Executors.newFixedThreadPool(4);
        List<Future<?>> inserts = new ArrayList<>(n);
        for (int ord = 0; ord < n; ord++) {
            VectorFloat<?> v = ds.ravv.getVector(ord);
            codes.get().encodeAndSet(ord, v);
            int o = ord;
            inserts.add(insertThreads.submit(() -> {
                trainingLock.readLock().lock();
                try {
                    builder.get().addGraphNode(o, v);
                } finally {
                    trainingLock.readLock().unlock();
                }
            }));

            if (ord == n / 2) {
                int encodedSoFar = ord + 1;
                trainingLock.writeLock().lock();
                try {
                    var trainingVectors = new ListRandomAccessVectorValues(ds.vectors.subList(0, encodedSoFar), DIMENSION);
                    MutablePQVectors refined = new MutablePQVectors(initialPq.refine(trainingVectors));
                    for (int i = 0; i < encodedSoFar; i++) {
                        refined.encodeAndSet(i, ds.ravv.getVector(i));
                    }
                    codes.set(refined);
                    builder.set(GraphIndexBuilder.rescore(builder.get(),
                            BuildScoreProvider.pqBuildScoreProvider(SIMILARITY_FUNCTION, refined)));
                } finally {
                    trainingLock.writeLock().unlock();
                }
            }
        }
        awaitAll(inserts);
        insertThreads.shutdown();
        try (GraphIndexBuilder b = builder.get()) {
            b.cleanup();
            return b.getGraph();
        }
    }

    private static void awaitAll(List<Future<?>> futures) {
        try {
            for (Future<?> f : futures) {
                f.get();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        } catch (ExecutionException e) {
            throw new RuntimeException(e.getCause());
        }
    }

    /** An {@link io.github.jbellis.jvector.graph.HnswIndexBuilder} with the shared shape parameters set. */
    private static io.github.jbellis.jvector.graph.HnswIndexBuilder baseBuilder(Dataset ds) {
        return Indexes.hnswBuilder()
                .withVectorValues(ds.ravv)
                .withMaxDegree(MAX_DEGREE)
                .withBeamWidth(BEAM_WIDTH)
                .withNeighborOverflow(NEIGHBOR_OVERFLOW)
                .withAlpha(ALPHA)
                .withAddHierarchy(ADD_HIERARCHY);
    }

    private static void compare(Dataset ds, String legacyLabel, GraphIndex legacy,
                                String modernLabel, GraphIndex modern) throws IOException {
        double legacyRecall;
        double modernRecall;
        try (GraphSearcher searcher = legacy.searcher()) {
            legacyRecall = recall(ds, searcher, RERANK_K, q -> DefaultSearchScoreProvider.exact(q, SIMILARITY_FUNCTION, ds.ravv));
        }
        try (GraphSearcher searcher = modern.searcher()) {
            modernRecall = recall(ds, searcher, RERANK_K, q -> DefaultSearchScoreProvider.exact(q, SIMILARITY_FUNCTION, ds.ravv));
        }
        System.out.printf("%s%n      recall %.3f  ->  %s%n      recall %.3f%n",
                legacyLabel, legacyRecall, modernLabel, modernRecall);
    }

    // ---------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------

    /** Base vectors, queries, and brute-force top-K ground truth for each query. */
    private static final class Dataset {
        final List<VectorFloat<?>> vectors;
        final RandomAccessVectorValues ravv;
        final List<VectorFloat<?>> queries;
        final List<Set<Integer>> groundTruth;

        Dataset(List<VectorFloat<?>> vectors, List<VectorFloat<?>> queries) {
            this.vectors = vectors;
            this.ravv = new ListRandomAccessVectorValues(vectors, DIMENSION);
            this.queries = queries;
            this.groundTruth = new ArrayList<>(queries.size());
            for (VectorFloat<?> q : queries) {
                Set<Integer> top = new HashSet<>();
                IntStream.range(0, vectors.size()).boxed()
                        .sorted(Comparator.comparingDouble(i -> -SIMILARITY_FUNCTION.compare(q, vectors.get(i))))
                        .limit(TOP_K)
                        .forEach(top::add);
                groundTruth.add(top);
            }
        }

        Dataset subset(int count) {
            return new Dataset(vectors.subList(0, count), queries);
        }
    }

    /** Builds the per-query {@link SearchScoreProvider}. */
    private interface ScoreProviderFactory {
        SearchScoreProvider forQuery(VectorFloat<?> query);
    }

    /** Mean recall@{@value #TOP_K} over the dataset's queries. */
    private static double recall(Dataset ds, GraphSearcher searcher, int rerankK, ScoreProviderFactory ssp) {
        int hits = 0;
        for (int i = 0; i < ds.queries.size(); i++) {
            SearchResult result = searcher.search(ssp.forQuery(ds.queries.get(i)), TOP_K, rerankK, 0.0f, 0.0f, Bits.ALL);
            for (SearchResult.NodeScore ns : result.getNodes()) {
                if (ds.groundTruth.get(i).contains(ns.node)) {
                    hits++;
                }
            }
        }
        return hits / (double) (ds.queries.size() * TOP_K);
    }

    /**
     * Views of an {@link OnDiskGraphIndex} are {@link GraphIndex.ScoringView}s: they build the reranker
     * (from inline or NVQ vectors) and, for fused graphs, the approximate score function.
     */
    private static GraphIndex.ScoringView scoringView(GraphSearcher searcher) {
        return (GraphIndex.ScoringView) searcher.getView();
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
        } catch (IllegalStateException | UnsupportedOperationException e) {
            System.out.println("  -> " + e.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }

    /** Unit vectors scattered around randomly chosen centroids. */
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
