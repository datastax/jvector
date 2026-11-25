# Upgrading from 4.0.x to 4.1.x

## Critical API changes

If you only read one thing, read this!

- **`ImmutableGraphIndex` has been renamed to `GraphIndex`**, which now extends the new generic `Index` interface.
  There is no deprecated alias: code that names `ImmutableGraphIndex` no longer compiles. Replace the name:

  | Before | Now |
  |---|---|
  | `ImmutableGraphIndex` | `GraphIndex` |
  | `ImmutableGraphIndex.View`, `.ScoringView`, `.NodeAtLevel`, `.IntMarker`, `.NeighborProcessor` | `GraphIndex.View`, `.ScoringView`, `.NodeAtLevel`, `.IntMarker`, `.NeighborProcessor` |
  | `ImmutableGraphIndex.ENTRY_NODE_ABSENT` | `GraphIndex.ENTRY_NODE_ABSENT` |
  | `ImmutableGraphIndex.prettyPrint(graph)` | `GraphIndex.prettyPrint(graph)` |

  The rename is mechanical: the methods and nested types are unchanged.
- **`GraphIndexBuilder.build(RandomAccessVectorValues)` and `getGraph()` now return `PersistableGraphIndex`**, a
  `GraphIndex` that can also be written to disk (see below), and `buildAndMergeNewNodes(...)` returns `GraphIndex`.
  Code that assigns these results to `GraphIndex` compiles unchanged.
- **Recompile against 4.1.** Besides the rename, the nested types are now members of `GraphIndex`, and many
  signatures that took or returned `ImmutableGraphIndex` (for example `new GraphSearcher(graph)`,
  `GraphSearcher.getView()`, `OnDiskGraphIndex.write`, and the graph writer builders) now use `GraphIndex`.
  Classes compiled against 4.0.x will fail to link.
- **`StreamingASHIndex.queryPreparer()` returns `Optional<QueryPreparer>`** instead of a nullable `QueryPreparer`
  (experimental). `StreamingASHIndex` is now a graph index (see below), and `GraphIndex.queryPreparer()` returns
  `Optional`. Replace a null check with `queryPreparer().orElse(null)` or `ifPresent`.

## On-disk format version 8
- Version 8 records the index's similarity function in the header: the float function (`VectorSimilarityFunction`)
  for float graphs, or the byte function (`ByteVectorSimilarityFunction`) for int8 graphs, or none when the graph
  was built with a custom score provider that names none. A fused ASH graph records its model's function.
- Version 8 also records the compression of the codes the graph is searched with (`CompressionType`): PQ or ASH
  for fused codes, automatically; for codes stored beside the graph, the type declared with the new
  `withCompressionType(type)` on the graph writer builders (`NONE` for a graph searched with its stored vectors
  alone). Undeclared, it is unknown. `OnDiskGraphIndex.compressionType()` reports it, also inferring fused codes in
  older files. A declared type needs version 8 and must agree with any fused codes, or `build()` throws
  `IllegalArgumentException`.
- The header grows by 12 bytes; otherwise a version 8 file has the version 7 layout.
- `OnDiskGraphIndex.CURRENT_VERSION`, the default for the graph writer builders, `OnDiskGraphIndexCompactor` and
  `CompressedVectors.write(out)`, is now 8. A version passed with `withVersion` is honored.
- JVector 4.0.x reads formats up to 6 and 4.1.0-rc.1 up to 7; both reject version 8 files. To keep a rollback
  possible, pass `withVersion(7)` (or 6 for 4.0.x) until every reader runs this release.
- Files written before version 8 load and search exactly as before. They record no function (except fused ASH
  files, through their model), so `similarityFunction()` is empty for them and they are searched with an explicit
  function, as before.
- `OnDiskGraphIndexCompactor` reads sources of versions 2 to 8 and records the function it was constructed with.
  A source that recorded a different function is compacted anyway, with a warning. New constructors that take no
  function (`(sources, liveNodes, executor)` and `(sources, sourceCompressed, liveNodes, executor)`) compact by the
  function the sources record, and throw `IllegalArgumentException` if one records none or they disagree. Compressed
  vectors given for a source that records another compression type are compacted with a warning by the existing
  constructors, and rejected by the new ones. The output records the type of its fused codes, or of the codes
  `compact(graphPath, compressedPath)` writes beside it.

## On-disk format version 7
- Version 7 of the on-disk graph format adds three features: fused ASH (`FeatureId.FUSED_ASH`), ASH codes
  stored in the graph alongside each node's neighbors; page alignment (`FeatureId.PAGE_ALIGNMENT`),
  written only when a graph writer builder is given `withPageAlignment(pageBytes)`; and inline int8
  vectors (`FeatureId.INLINE_BYTE_VECTORS`), each node's byte vector stored inline, one byte per
  dimension. It also stores the largest vector length that adaptive dot-product stopping needs: one float
  at the end of each serialized PQ model, and of the inline-vector and separated-vector feature headers.
  The inline int8 header stores it too, as the largest length of the stored byte vectors; nothing reads it
  yet, and int8 adaptive stopping, when added, must map it through int8 scores rather than the float rule.
  Otherwise a version 7 file has the version 6 layout. ASH models (`AsymmetricHashing`, `ASHVectors`) also
  require version 7 or later when they are written or loaded. A model stores its similarity and, for
  `EUCLIDEAN`, a scale unit (8 bytes in all); ASH models written before these fields existed cannot be
  loaded.
- Dot-product searches over indexes and PQ models written before version 7 stop by the standard rule,
  since their largest vector length is unknown; rewriting them at version 7 does not change that for codes
  encoded before, so a PQ model loaded from an older format keeps it unknown.
- JVector 4.0.x reads formats up to 6 and rejects version 7 files. (`OnDiskGraphIndex.CURRENT_VERSION` is now 8;
  see above.)
- To keep a rollback to 4.0.x possible, pass the version explicitly until every reader runs 4.1:
  `withVersion(6)` on graph writer builders and `write(out, 6)` for compressors and compressed vectors.
  Versions 6 and below still write inline vectors, NVQ, separated vectors and fused PQ exactly as before,
  and 4.0.x reads them. A version 6 writer rejects fused ASH, page alignment and inline int8 vectors, and
  ASH models reject version 6.
- `OnDiskGraphIndexCompactor` writes `CURRENT_VERSION` by default; pass `withVersion(6)` to keep a rollback
  to 4.0.x possible (its compressed sidecar is written at the same version). It does not compact page-aligned
  indexes or int8 indexes.

## New features
- Asymmetric Hashing (ASH) quantization, stored separately or fused into the graph (format version 7), with
  native FP16, FP32 and INT8 query projection and bounded queued query execution. See
  [ASH configuration and defaults](docs/ash-configuration.md) and
  [Integrating ASH queries into a database](docs/ash-query-integration.md).
- `AsymmetricHashing.setQueryOptions(ASHQueryProjection, maxBatchSize)` sets a model's query projection
  precision and batch size in code; the `jvector.ash.queryProjection` and `jvector.ash.queryBatchSize`
  properties remain the defaults.
- Page-aligned base-layer records for on-disk indexes: `withPageAlignment(pageBytes)` on graph writer
  builders places records so none crosses a page boundary, so serving a larger-than-memory index reads
  fewer pages per node. See
  [Building and serving larger-than-memory indexes](docs/larger-than-memory-indexes.md).
- ASH supports `EUCLIDEAN` and `COSINE` as well as `DOT_PRODUCT`. A model's similarity is chosen when it is trained,
  with `AsymmetricHashing.initialize(vectors, similarity, ...)`, and saved with it; its scorers reject any
  other similarity with `IllegalArgumentException`. The similarity is a required argument; there is no
  default. ASH does not require normalized vectors; `COSINE` normalizes them itself and throws
  `IllegalArgumentException` for zero vectors and queries, so callers must reject them first. `CompressedVectors.similarityFunction()` and
  `ScoreFunction.similarityFunction()` report a fixed similarity (null when the caller chooses it, as for PQ).
- JVector sets the native BLAS thread count for ASH training: one thread by default, or the size of the
  executor passed to `AsymmetricHashing.initialize(..., ForkJoinPool executor)`.
- **A generic index API.** `io.github.jbellis.jvector.api.Index` is a backing-agnostic handle (`searcher()`, `ramBytesUsed()`, `close()`), and
  `IndexSearcher` is the matching searcher type. `GraphIndex` implements `Index`, and `GraphIndex.searcher()`
  returns a `GraphSearcher` with no cast. Code that holds only an `Index` recovers the concrete type with
  `instanceof GraphIndex`. `IndexSearcher` is `Closeable` and `Index.close()` throws only `IOException`, so both
  work with try-with-resources:
  ```java
  try (IndexSearcher searcher = index.searcher()) { ... }
  ```
- **`Indexes.hnswBuilder(...)`**, a fluent builder for graph indexes (`HnswIndexBuilder`). You choose how vectors
  are scored when you create it, and every other setting has a default (the same defaults as `GraphIndexBuilder`:
  max degree 32, beam width 100, neighbor overflow 1.2, alpha 1.2, hierarchy on, refinement on). The common case is
  one call:
  ```java
  PersistableGraphIndex graph = Indexes.hnswBuilder(ravv, VectorSimilarityFunction.COSINE).buildAndPopulate();
  ```
  - `Indexes.hnswBuilder(RandomAccessVectorValues, VectorSimilarityFunction)` scores with the vectors themselves.
    `withCompressionType(CompressionType.PQ)` or `BQ` makes the builder train the quantizer, encode the vectors
    and build with the compressed scores, with no codebook or encoding step for you to run. Once the graph is
    built, `getCompressedVectors()` returns what it trained (`PQVectors`, whose `getCompressor()` is the
    `ProductQuantization`, or `BQVectors`), to search with or to write to disk, e.g. as a `FusedPQ` feature.
    The PQ training defaults match Cassandra's, and each can be overridden: `withPqSubspaces(n)` sets the code size
    in bytes (one subspace per byte; by default a dimension-dependent rule, e.g. 64 for 128 dimensions and 192 for
    1536), `withPqGlobalCentering(boolean)` whether to center the vectors first (default `false`), and
    `withPqAnisotropicThreshold(float)` the anisotropic weighting (default `-1.0`, unweighted). The cluster count
    (256) still comes from `GraphIndexBuilderConfig`.
    `withCompressionType(CompressionType.ASH)` builds with ASH scores instead. The builder trains ASH for its
    similarity function, with ITQ around a single landmark. `withAshProjectedDimensions(n)` sets the
    projected dimension (default: the vector dimension), and `withAshBitsPerDimension(b)` the bits per projected
    dimension (default `AsymmetricHashing.DEFAULT_BITS_PER_DIMENSION`, 2). `getCompressedVectors()` then returns
    `ASHVectors`.
  - `Indexes.hnswBuilder(BuildScoreProvider, dimension)` scores with your own provider, for callers that stream
    vectors in or already have a provider. It has no vectors of its own, so populate it with
    `populateGraph(ravv)` or `addGraphNode`.
  - `buildAndPopulate()` builds and inserts the builder's own vectors in one call. `build()` returns the graph
    empty, for incremental construction with `addGraphNode` (thread-safe, and the graph can be searched while it
    grows), `markNodeDeleted`, `removeDeletedNodes` and `cleanup()`. `populateGraph(ravv)` populates an empty
    graph from a whole `RandomAccessVectorValues` and calls `cleanup()`; it throws if the graph already has nodes.
    `build()` is idempotent: every call returns the same graph. Settings changed after the graph is built are
    ignored, with a logged warning.
  - `withExistingGraph(OnHeapGraphIndex)` continues building on a graph reloaded with `OnHeapGraphIndex.load`.
    Add the new nodes with `addGraphNode`, from the graph's `getIdUpperBound()` up, then call `cleanup()`.
    The graph's dimension must match the builder's.
  - `HnswIndexBuilder.rescore(builder, newProvider)` copies the graph with every edge re-scored by a new
    provider, for example after refining a PQ codebook, and returns a builder that continues with the copy.
  - `HnswIndexBuilder` is `Closeable`: closing it releases the per-thread scratch space used while inserting. The
    graph it built stays usable.
- **`PersistableGraphIndex`**, implemented by `OnHeapGraphIndex` and `OnDiskGraphIndex`, adds accessors for the
  three graph writers, so a built graph can be written without naming the writer classes. Each accessor returns
  that writer's own builder type, so its specific options (such as `withStartOffset`, `withParallelWorkerThreads`
  or `withExecutor`) are available, and building gives the concrete writer (with `getOutput()` and `checksum()`
  on the random-access writers):

  | Accessor | Writer | `GraphIndexWriterTypes` |
  |---|---|---|
  | `getWriterBuilder(Path)` | `OnDiskGraphIndexWriter` (random access, single-threaded) | `RANDOM_ACCESS` |
  | `getParallelWriterBuilder(Path)` | `OnDiskParallelGraphIndexWriter` | `RANDOM_ACCESS_PARALLEL` |
  | `getWriterBuilder(IndexWriter)` | `OnDiskSequentialGraphIndexWriter` | `ON_DISK_SEQUENTIAL` |

  ```java
  try (var writer = graph.getParallelWriterBuilder(path).with(new InlineVectors(dim)).build()) {
      writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
              node -> new InlineVectors.State(ravv.getVector(node))));
  }
  ```
  Constructing the writer builders directly and `GraphIndexWriter.getBuilderFor(...)` still work.
- **Shortcuts for the simplest on-disk index.** `PersistableGraphIndex.writeTo(path, vectors)` writes a graph with
  its vectors stored inline. `IndexSearcher.search(query, topK, rerankK, acceptOrds)` searches a graph that
  stores its vectors (an `OnDiskGraphIndex` with inline or NVQ vectors) by the similarity function it records,
  without building a score provider: with fused PQ or fused ASH it traverses with those codes and reranks the best
  `rerankK` with the stored vectors, otherwise it scores with the stored vectors. It throws `IllegalStateException`
  on an in-memory graph, on a file that records no function (written before version 8), and on a searcher created
  from a view. `GraphSearcher.search(query, topK, rerankK, similarityFunction, acceptOrds)` does the same with a
  function you pass, and is the way to search older files; for the same function and `rerankK` both return the same
  results as building that score provider by hand. `GraphIndex.ScoringView.hasApproximateScores()` (default
  `false`) reports whether a view supports `approximateScoreFunctionFor`.
  ```java
  graph.writeTo(path, ravv);
  try (var rs = ReaderSupplierFactory.open(path);
       var onDisk = OnDiskGraphIndex.load(rs);
       var searcher = onDisk.searcher()) {
      SearchResult result = searcher.search(query, 10, 30, Bits.ALL);
  }
  ```
- **Indexes know their similarity function.** `Index.similarityFunction()` (and `byteSimilarityFunction()` for
  int8) report the function an index was built with: for an in-memory graph, its build score provider's; for an
  on-disk graph, the one its file records (version 8). `IndexSearcher.search` scores by it, so a search can't use a
  function the index wasn't built for. The default implementations return empty.
- **Query preparation for every on-disk layout.** `GraphIndex.queryPreparer()` returns a `QueryPreparer` for
  version 8 graphs with fused PQ, NVQ or inline vectors, as it already did for fused ASH: preparing a query once
  computes fused PQ's lookup table and NVQ's per-query state, and `ScoringView.scoreProviderFor(prepared, rerank)`
  binds it to a view. Prepared and unprepared searches return the same results. It is empty for files that record
  no function and for int8 graphs.
- **`StreamingASHIndex` is a graph index** (experimental). It implements the new `ASHGraphIndex`, a
  `PersistableGraphIndex`: its views score with its ASH codes and rerank from the caller's vectors, its searcher
  supports `search(query, topK, rerankK, acceptOrds)`, and it is written as its graph is (call `prepareForFlush()`
  and `builder().cleanup()` first).
- **Experimental:** recipes and IVF are marked `@Experimental`. `HnswRecipe.DEFAULT` is defined (it restates the
  builder's defaults) and can be applied with `HnswIndexBuilder.applyRecipe`; the other recipes, and
  `IvfIndexBuilder.build()`, throw `UnsupportedOperationException` until their values and implementation exist.
  The IVF types are `Indexes.ivfBuilder()`, `IvfIndexBuilder`, `IvfIndex` and `IvfSearcher`.
- `jvector-examples/.../IndexApiExample.java` walks through the new API end to end, organized around Cassandra's and
  OpenSearch's usage: the one-call build and tuning; building with PQ or BQ via `withCompressionType` and searching
  with the trained codes; every writer type, including a graph embedded at an offset; fused PQ, NVQ with fused PQ,
  and separately stored PQ codes on disk, all from the builder's own PQ; incremental construction with deletes;
  streaming with caller-maintained PQ codes and a mid-build rescore; continuing a saved graph; and search options.
  Run it from the project root with `mvn compile exec:exec@index-api-example`.

## Moving from GraphIndexBuilder to HnswIndexBuilder (optional)

`GraphIndexBuilder` remains supported; moving is optional in this release.

| `GraphIndexBuilder` | `HnswIndexBuilder` |
|---|---|
| `new GraphIndexBuilder(ravv, vsf, M, beamWidth, overflow, alpha, addHierarchy).build(ravv)` | `Indexes.hnswBuilder(ravv, vsf).withMaxDegree(M).withBeamWidth(beamWidth).withNeighborOverflow(overflow).withAlpha(alpha).withAddHierarchy(addHierarchy).buildAndPopulate()`, omitting any setting that matches the default |
| constructors taking a `BuildScoreProvider` and dimension | `Indexes.hnswBuilder(bsp, dimension)` |
| computing PQ, BQ or ASH vectors yourself to build with compressed scores | `Indexes.hnswBuilder(ravv, vsf).withCompressionType(CompressionType.PQ)` (or `BQ`, or `ASH`) |
| `List<Integer>` max degrees | `withMaxDegrees(list)` |
| `refineFinalGraph`, SIMD and parallel executor arguments | `withRefineFinalGraph`, `withBuildExecutor` (the SIMD executor), `withMaintenanceExecutor` (the parallel executor) |
| `addGraphNode`, `markNodeDeleted`, `removeDeletedNodes`, `cleanup`, `insertsInProgress` | the same methods on `HnswIndexBuilder`, after `build()` |
| `GraphIndexBuilder.rescore(builder, newBsp)` | `HnswIndexBuilder.rescore(builder, newBsp)` |
| existing-graph constructor, or `buildAndMergeNewNodes` | `OnHeapGraphIndex.load(...)`, then `withExistingGraph(graph)` |
| `builder.getGraph()` | `builder.getGraph()` or `builder.build()` (a `PersistableGraphIndex`) |

Differences to be aware of:
- `addHierarchy`, `refineFinalGraph` and the build compression are always explicit (defaulting to `true`, `true`
  and `CompressionType.NONE`). The `GraphIndexBuilder.builder(...)` fluent builder from 4.0.x reads them from the
  JMX `GraphIndexBuilderConfig` instead. So are the PQ subspace count, centering and anisotropic threshold
  (`withPqSubspaces`, `withPqGlobalCentering`, `withPqAnisotropicThreshold`); only the PQ cluster count still comes
  from `GraphIndexBuilderConfig`.
- Settings are validated when the graph is built, by `GraphIndexBuilder`, which throws an
  `IllegalArgumentException` for the first invalid value. Null inputs and a non-positive dimension are rejected
  immediately. Settings changed after the graph is built are ignored, with a logged warning.
- With `withExistingGraph`, calling `withMaxDegree(s)` or `withAddHierarchy` as well is rejected with an
  `IllegalStateException`, because the existing graph fixes both. So is scoring of a different kind than the graph
  was built with: a graph built with exact scores can't be continued with compressed ones, or the reverse, and
  `withCompressionType` can't be combined with an existing graph, since it would train a new quantizer. To continue
  a graph built with compressed vectors, use `Indexes.hnswBuilder(scoreProvider, dimension)` with the score provider
  the graph was built with. The existing graph keeps the diversity provider
  it was created with, which must be able to score the ordinals you append.
- Concurrency is unchanged from `GraphIndexBuilder`: `addGraphNode` and `markNodeDeleted` are thread-safe, but
  `cleanup()`, `removeDeletedNodes()`, `rescore` and `close()` must not run while inserts are in progress.
- As with `GraphIndexBuilder.addGraphNode`, every node must be scoreable by ordinal through the build score
  provider before (or when) it is added: add the vector to your vector values, or encode its PQ code, first.
  Otherwise a later insert or `cleanup()` fails with an `IndexOutOfBoundsException`.

## Behavior changes

- Searching an index by a similarity function other than the one it records logs a warning, and the search
  proceeds with the function given, as before: once per loaded graph for `GraphSearcher.search(query, topK,
  rerankK, similarityFunction, acceptOrds)` and the on-disk views' `rerankerFor` and `approximateScoreFunctionFor`,
  and once per process for the static `GraphSearcher.search(query, topK, vectors, similarityFunction, graph,
  acceptOrds)`. Some mismatches are deliberate, such as `COSINE` over a `DOT_PRODUCT` graph of unit vectors.
  Fused ASH indexes still reject a mismatch with `IllegalArgumentException`.

- Adding nodes to an `OnHeapGraphIndex` after `cleanup()` now works correctly. Previously the graph stayed
  "frozen" after cleanup, so a later insert could link a node to itself (tripping an assertion) and concurrent
  searches could see half-inserted nodes. Adding a node now unfreezes the graph; call `cleanup()` again before
  writing it, and don't reuse views or searchers obtained while it was frozen. This also affects continuing to
  build on a graph that has already been cleaned up, including one returned by `GraphIndexBuilder.build`.
- `GraphIndexBuilder.rescore` now keeps nodes that were marked deleted (with `markNodeDeleted`) but not yet
  removed. Previously the rescored graph dropped those marks, so the nodes became visible to searches again and
  survived `cleanup()`. Callers that don't delete before rescoring, such as a compaction refining its PQ
  codebook, see no change.
- The writer builders that take a `Path` (`OnDiskGraphIndexWriter.Builder`, `OnDiskParallelGraphIndexWriter.Builder`,
  `RandomAccessOnDiskGraphIndexWriter.Builder`, and the `PersistableGraphIndex` accessors) now open the file in
  `build()` rather than in the constructor. A builder that is never built, or whose `build()` fails, no longer leaks
  an open file. An error opening the file (for example, a missing directory) is now thrown by `build()`.
- When `GraphIndexBuilderConfig` (JMX) selects PQ or BQ build compression for `GraphIndexBuilder`, the vectors are
  now encoded on `PhysicalCoreExecutor.pool()` rather than `ForkJoinPool.commonPool()`. Results are unchanged.
  The JMX build compression type also accepts `ASH`, with the same training as `HnswIndexBuilder`'s defaults
  (projected to the vector dimension, 2 bits per dimension), trained for the graph's similarity function.

## Deprecations

These still work; each `@deprecated` note names the replacement.

- `GraphSearcher.search(query, topK, rerankK, similarityFunction, acceptOrds)`: use `search(query, topK, rerankK,
  acceptOrds)`, which scores by the function the graph records. Files written before version 8 record none and keep
  using the deprecated method.
- `FusedPQ.approximateScoreFunctionFor(query, similarityFunction, view, reranker)`,
  `FusedASH.approximateScoreFunctionFor(query, similarityFunction, view, reranker)` and
  `NVQ.rerankerFor(query, similarityFunction, source)`: search with `search(query, topK, rerankK, acceptOrds)`, or
  prepare the query with `GraphIndex.queryPreparer()` and bind it with `ScoringView.scoreProviderFor`.

## Other changes to public classes

- `GraphSearcher` implements `IndexSearcher`.
- `AbstractGraphIndexWriter.Builder` implements `PersistableGraphIndex.GraphIndexWriterBuilder`, the options
  every writer supports (`with`, `withMapper`, `withMap`, `withVersion`, `build`). `OnDiskGraphIndexWriter.Builder`,
  `OnDiskParallelGraphIndexWriter.Builder` and `OnDiskSequentialGraphIndexWriter.Builder` override those to return
  their own type, so writer-specific options can follow them in a chain.
- `@Experimental` is now `@Documented`, so it appears in the generated Javadoc.
- `GraphIndex` extends `Accountable`, as `ImmutableGraphIndex` did.
- `CompressionType` has a new constant, `ASH`. A `switch` over `CompressionType` without a default branch needs a
  case for it.
- `GraphIndexFormat` has a new method, `supportsFusedFeatures()` (true from version 6). Implementations outside
  JVector must add it.
- `AsymmetricHashing.DEFAULT_BITS_PER_DIMENSION` is now public.
- Writer documentation corrected, with no behavior change: `withParallelWorkerThreads` and
  `withParallelDirectBuffers` apply only to the parallel writer, and `RandomAccessOnDiskGraphIndexWriter.Builder`
  (which picks a writer from the JMX configuration) ignores them when it picks the single-threaded one. A worker
  count of 0 or less means "use all available processors".

# Upgrading from 3.0.x to 4.0.x

## New features
- Support for Non-uniform Vector Quantization (NVQ, pronounced as "new vec"). This new technique quantizes the values
  in each vector with high accuracy by first applying a nonlinear transformation that is individually fit to each
  vector. These nonlinearities are designed to be lightweight and have a negligible impact on distance computation
  performance.
- Support for hierarchical graph indices. This new type of index blends HNSW and DiskANN in a novel way. An
  HNSW-like hierarchy resides in memory for quickly seeding the search. This also reduces the need for caching the
  DiskANN graph near the entrypoint. The base layer of the hierarchy is a DiskANN-like index and inherits its
  properties. This hierarchical structure can be disabled, ending up with just the base DiskANN layer.
- The feature previously known as Fused ADC has been renamed to Fused PQ. This feature allows to offload the PQ
  codebooks from memory during search, storing them within the graph in a way that does not slow down the search.
  Implementation notes: The implementation of this feature has been overhauled to not require native code acceleration.
  This explores a design space allowing for packed representations of vectors fused into the graph in shapes optimal 
  for approximate score calculation. This new feature of graph indexes is opt-in but fully functional now. Any graph
  degree limitations have been lifted. At this time, only 256-cluster ProductQuantization can use fused PQ.
  Version 6 or greater of the file disk format is required to use this feature.


## API changes
- MemorySegmentReader.Supplier and SimpleMappedReader.Supplier must now be explicitly closed, instead of being
  closed by the first Reader created from them.
- OnDiskGraphIndex no longer closes its ReaderSupplier
- The constructor of GraphIndexBuilder takes an additional parameter which allows to enable or disable the use of the
  hierarchy.
- GraphSearcher can be configured to run pruned searches using GraphSearcher.usePruning. When this is set to true,
  we do early termination of the search. In certain cases, this can accelerate the search at the potential cost of some
  accuracy. It is set to false by default.
- The constructors of GraphIndexBuilder allow to specify different maximum out-degrees for the graphs in each layer.

### API changes in 3.0.6

These were released in 3.0.6 but are spiritually part of 4.0.

- `VectorCompressor.encodeAll()` now returns a `CompressedVectors` object instead of a `ByteSequence<?>[]`.
  This provides better encapsulation of the compression functionality while also allowing for more efficient
  creation of the `CompressedVectors` object.
- The `ByteSequence` interface now includes an `offset()` method to provide offset information for the sequence.
  any time the method `ByteSequence::get` is called, the full backing data is returned, and as such, the `offset()`
  method is necessary to determine the offset of the data in the backing array.
- `PQVectors` has been split into `MutablePQVectors` and `ImmutablePQVectors`.  Generally you will use `new MutablePQVectors()`
  directly, while `ImmutablePQVectors` will usually be accessed via `PQVectors.load` (whose signature has not changed).
  These changes allow PQVectors to represent compressed vectors more efficiently under the hood.
- `BQVectors` has similarly been split into mutable and immutable implementations.
- The `VectorCompressor.createCompressedVectors(Object[])` method is now deprecated in favor of the new API that returns
  `CompressedVectors` directly from `encodeAll()`.
- `PQVectors::getProductQuantization` is removed; it duplicated `CompressedVectors::getCompressor` unnecessarily

# Upgrading from 2.0.x to 3.0.x

## Critical API changes

If you only read one thing, read this!

- `GraphIndexBuilder` `M` parameter now represents the maximum degree of the graph,
  instead of half the maximum degree.  (The former behavior was motivated by making
  it easy to make apples-to-apples comparisons with Lucene HNSW graphs.)  So,
  if you were building a graph of M=16 with JVector2, you should build it with M=32
  with JVector3.
- Support for indexes over byte vectors has been removed. This is because the implementation
  was becoming increasingly specialized for float vectors, leaving byte vector support as a 
  secondary concern. This specializes many structures that were previously generic over vector type.
- JVector 3 adds several optional features to the on-disk storage format, but remains
  compatible with indexes written by JVector 1 and 2.


## New features
- Experimental support for native code acceleration has been added. This currently only supports Linux x86-64 
  with certain AVX-512 extensions. This is opt-in and requires the use of MemorySegment `VectorFloat`/`ByteSequence`
  representations.
- Experimental support for fused ADC graph indexes has been added. These work best in concert with native code acceleration.
  Without the NativeVectorizationProvider, results using fused ADC will be valid but performance will degrade.
  This explores a design space allowing for packed representations of vectors fused into the graph in shapes optimal
  for approximate score calculation. This is a new feature of graph indexes and is opt-in. At this time, only graphs with
  a maximum degree of 32 and 256-cluster ProductQuantization can use fused ADC.
- Support for larger-than-memory graph construction by using quantized vectors + rerank for the searches
  performed during construction.
- Support for Anisotropic Product Quantization as described in "Accelerating Large-Scale Inference with Anisotropic Vector Quantization"
  (https://arxiv.org/abs/1908.10396)
- `GraphIndexBuilder.markNodeDeleted` is now threadsafe
- `GraphIndexBuilder::removeDeletedNodes` is parallelized and significantly faster.

## API changes supporting new features
- `GraphIndexBuilder` and `GraphSearcher` scoring are encapsulated by `BuildScoreProvider` and `SearchScoreProvider`,
  respectively.  `BuildScoreProvider.randomAccessScoreProvider()` and `BuildScoreProvider.pqBuildScoreProvider()`
  offer convenient ways to construct a BSP from full-resolution vectors in memory or with PQ-compressed vectors
  with reranking, respectively.
- `addGraphNode(int node, VectorFloat<?> vector)` is now the preferred way to construct a graph incrementally.
- `GraphIndexSearcher::resume` is added to allow resuming a previous search from where it left off.
- `ProductQuantization::refine` allows fine-tuning a new PQ object with additional vectors, starting with an existing PQ
- Changes to KMeansPlusPlusClusterer to support Anisotropic PQ

## Refactored APIs
- `VectorFloat` and `ByteSequence` are introduced as abstractions over float vectors and byte sequences.
  These are used in place of `float[]` and `byte[]` in many places in the API. This is to permit the
  possibility of alternative implementations of these types. This requires changes to many internal/external API
  surfaces.
- `NodeSimilarity` has been removed.  `ScoreFunction` is now a top-level interface; grouping of functions
  for build and for search are now done by `BuildScoreProvider` and `SearchScoreProvider`.
  - BuildScoreProvider allows the creation of larger-than-memory indexes by using compressed vectors
    during graph construction.
  - Reranking is done using `ExactScoreFunction::similarityTo(int[])` rather than with a Map parameter.
    The map change is because we discovered that (in contrast with the original DiskANN design) it is more
    performant to read vectors lazily from disk at reranking time, since this will only have to fetch vectors for the topK 
    nodes instead of all nodes visited.  Additionally, the extra method taking `int[]` allows native implementations 
    to perform more work per FFM call.
  - `example/Grid.java` shows how to use these.
- `OnDiskGraphIndex`, `CachingGraphIndex`, and `GraphCache` have moved to the package `jvector.graph.disk`
- Writing graphs using the new feature (FusedADC) is performed with `OnDiskGraphIndexWriter`; see `OnDiskGraphIndex.write` for an example of how to use it
- `RandomAccessVectorValues::vectorValue` is deprecated, replaced by `getVector` (which has the same semantics
  as `vectorValue`) and `getVectorInto`.  The latter allows JVector to avoid an unnecessary copy when there
  is a specific destination already created that needs the data.
- `CompressedVectors::approximateScoreFunctionFor` is deprecated, replaced by `precomputedScoreFunctionFor`
  (which has the same semantics as `approximateScoreFunctionFor`) and `scoreFunctionFor`, which does not
  precompute partial similarities across the codebooks and is more suitable for cases when only a few
  similarities will be calculated.
- `VectorUtil.divInPlace` is replaced by its inverse, `VectorUtil.scale`
- `PoolingSupport` is removed in favor of direct usage of `ExplicitThreadLocal`
- `ExplicitThreadLocal` and `GraphIndexBuilder` implement AutoCloseable to make it easier to clean up pooled Views

## Other changes to public classes
- `FixedBitSet.nextSetBit` behaves as expected
- Removed vestigal references to node level in several places that were left over from old HNSW code
- Centering of binary quantization makes things worse, not better, and has been removed.  Saved BQ and BQVectors
  that have centering data will ignore it on load.

# Upgrading from 1.0.x to 2.0.x

## New features

- In-graph deletes are supported through `GraphIndexBuilder.markNodeDeleted`.  Deleted nodes
  are removed when `GraphIndexBuilder.cleanup` is called (which is not threadsafe wrt other concurrent changes).
  To write a graph with deleted nodes to disk, a `Map` must be supplied indicating what ordinals
  to change the remaining node ids to -- on-disk graphs may not contain "holes" in the ordinal sequence.
- `GraphSearcher.search` now has an experimental overload that takes a
  `float threshold` parameter that may be used instead of topK; (approximately) all the nodes with simlarities greater than the given threshold will be returned.
- Binary Quantization is available as an alternative to Product Quantization. Our tests show that it's primarily suitable for ada002 embedding vectors and loses too much accuracy with smaller embeddings.

## Primary API changes

- `GraphIndexBuilder.complete` is now `cleanup`.
- The `Bits` parameter to `GraphSearcher.search` is no longer nullable;
  pass `Bits.ALL` instead of `null` to indicate that all ordinals are acceptable.

## Other changes to public classes

- `NeighborQueue`, `NeighborArray`, and `NeighborSimilarity` have been renamed to
  `NodeQueue`, `NodeArray`, and `NodeSimilarity`, respectively.
