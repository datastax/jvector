# Upgrading from 4.0.1 to 4.0.2

## Critical API changes

If you only read one thing, read this!

- `ImmutableGraphIndex` has been renamed to `GraphIndex`, which now extends the new generic `Index` interface.
  `ImmutableGraphIndex` remains for this release only, as a deprecated interface extending `GraphIndex`
  (`@Deprecated(forRemoval = true)`), so existing source code compiles with removal warnings. **It will be
  removed in the release after 4.0.2**; replace the name now:

  | Before | Now |
  |---|---|
  | `ImmutableGraphIndex` | `GraphIndex` |
  | `ImmutableGraphIndex.View`, `.ScoringView`, `.NodeAtLevel`, `.IntMarker`, `.NeighborProcessor` | `GraphIndex.View`, `.ScoringView`, `.NodeAtLevel`, `.IntMarker`, `.NeighborProcessor` |
  | `ImmutableGraphIndex.ENTRY_NODE_ABSENT` | `GraphIndex.ENTRY_NODE_ABSENT` |
  | `ImmutableGraphIndex.prettyPrint(graph)` | `GraphIndex.prettyPrint(graph)` |

  The old nested names are the same types as the new ones, so the two spellings can be mixed while you
  migrate. `OnHeapGraphIndex` and `OnDiskGraphIndex` still implement `ImmutableGraphIndex`, and
  `GraphIndexBuilder.build(RandomAccessVectorValues)`, `getGraph()` and `buildAndMergeNewNodes(...)` still return
  it, so assignments such as `ImmutableGraphIndex graph = builder.build(ravv)` keep compiling. When
  `ImmutableGraphIndex` is removed, those four methods will return `GraphIndex`; code that already assigns their
  results to `GraphIndex` is unaffected.
- **Recompile against 4.0.2.** This is a source-compatible change, not a binary-compatible one: the nested types
  are now members of `GraphIndex`, and many signatures that took or returned `ImmutableGraphIndex` (for example
  `new GraphSearcher(graph)`, `GraphSearcher.getView()`, `OnDiskGraphIndex.write`, and the graph writer builders)
  now use `GraphIndex`. Classes compiled against 4.0.1 may fail to link.
- A new module, `jvector-api`, holds the public contract, and `jvector-base` depends on it. It contains the new
  `Index`, `IndexSearcher`, `HnswRecipe` and `IvfRecipe`, plus `Accountable` and the `@Experimental` annotation,
  which moved there from `jvector-base` with their package and class names unchanged. If you depend on the
  published `io.github.jbellis:jvector` artifact (as Cassandra and OpenSearch do), nothing changes: it bundles
  `jvector-api`. If you build against the individual modules, add `jvector-api`.

## New features

- **A generic index API.** `Index` is a backing-agnostic handle (`searcher()`, `ramBytesUsed()`, `close()`), and
  `IndexSearcher` is the matching searcher type. `GraphIndex` implements `Index`, and `GraphIndex.searcher()`
  returns a `GraphSearcher` with no cast. Code that holds only an `Index` recovers the concrete type with
  `instanceof GraphIndex`. `IndexSearcher` is `Closeable` and `Index.close()` throws only `IOException`, so both
  work with try-with-resources:
  ```java
  try (IndexSearcher searcher = index.searcher()) { ... }
  ```
- **`Indexes.hnswBuilder()`**, a fluent builder for graph indexes (`HnswIndexBuilder`). It validates the whole
  configuration at once and reports every missing value, out-of-range value and conflicting setting in a single
  `IllegalStateException`. It finishes in one of two ways:
  - `build()` inserts every vector from `withVectorValues(...)` in parallel, calls `cleanup()`, and returns a
    `PersistableGraphIndex`.
  - `buildMutable()` returns a `MutableHnswIndex` for incremental construction: `addNode` (safe to call from
    many threads while the graph is searched), `markDeleted`, `removeDeletedNodes`, `cleanup`, and
    `rescore(Supplier<BuildScoreProvider>)`. The handle serializes `cleanup`, `removeDeletedNodes` and `rescore`
    against inserts itself, and waits cooperatively when inserts run on a `ForkJoinPool`, so callers no longer
    need their own lock around these calls. `rescore` runs the supplier with inserts locked out, so it can safely
    refine and re-encode PQ codes before returning the new score provider.
  - `withExistingGraph(OnHeapGraphIndex)` continues building on a graph reloaded with `OnHeapGraphIndex.load`.
- **`PersistableGraphIndex`**, implemented by `OnHeapGraphIndex` and `OnDiskGraphIndex`, adds accessors for the
  three graph writers:

  | Accessor | Writer | `GraphIndexWriterTypes` |
  |---|---|---|
  | `getWriterBuilder(Path)` | `OnDiskGraphIndexWriter` (random access, single-threaded) | `RANDOM_ACCESS` |
  | `getParallelWriterBuilder(Path)` | `OnDiskParallelGraphIndexWriter` | `RANDOM_ACCESS_PARALLEL` |
  | `getWriterBuilder(IndexWriter)` | `OnDiskSequentialGraphIndexWriter` | `ON_DISK_SEQUENTIAL` |

  Constructing the writer builders directly and `GraphIndexWriter.getBuilderFor(...)` still work.
- **Experimental:** `HnswRecipe` and `IvfRecipe` (with `HnswIndexBuilder.applyRecipe`), and the IVF types
  (`Indexes.ivfBuilder()`, `IvfIndexBuilder`, `IvfIndex`, `IvfSearcher`) are marked `@Experimental`. They are
  placeholders: every recipe, and `IvfIndexBuilder.build()`, currently throws `UnsupportedOperationException`.
- `jvector-examples/.../IndexApiExample.java` shows every way to build, write, load and search an index with the
  new API, including each compression type (PQ, fused PQ, NVQ) and each writer, and runs each older
  `GraphIndexBuilder` pattern next to its equivalent.

## Moving from GraphIndexBuilder to HnswIndexBuilder (optional)

`GraphIndexBuilder` remains supported; moving is optional in this release.

| `GraphIndexBuilder` | `HnswIndexBuilder` |
|---|---|
| `new GraphIndexBuilder(ravv, vsf, M, beamWidth, overflow, alpha, addHierarchy).build(ravv)` | `Indexes.hnswBuilder().withVectorValues(ravv).withSimilarityFunction(vsf).withMaxDegree(M).withBeamWidth(beamWidth).withNeighborOverflow(overflow).withAlpha(alpha).withAddHierarchy(addHierarchy).build()` |
| constructors taking a `BuildScoreProvider` and dimension | `withScoreProvider(bsp)`; with `buildMutable()`, `withDimension(d)` instead of vector values |
| `List<Integer>` max degrees | `withMaxDegrees(list)` |
| `refineFinalGraph`, SIMD and parallel executor arguments | `withRefineFinalGraph`, `withSimdExecutor`, `withParallelExecutor` |
| `addGraphNode`, `markNodeDeleted`, `removeDeletedNodes`, `cleanup` | `MutableHnswIndex.addNode`, `markDeleted`, `removeDeletedNodes`, `cleanup` |
| `builder = GraphIndexBuilder.rescore(builder, newBsp)` under your own lock | `mutableIndex.rescore(() -> ...)` |
| existing-graph constructor, or `buildAndMergeNewNodes` | `OnHeapGraphIndex.load(...)`, then `withExistingGraph(graph)` |
| `builder.getGraph()` | `mutableIndex.graph()` (a `PersistableGraphIndex`) |

Differences to be aware of:
- `addHierarchy` and `refineFinalGraph` are always explicit (`refineFinalGraph` defaults to `true`). The
  `GraphIndexBuilder.builder(...)` fluent builder from 4.0.x reads them from the JMX `GraphIndexBuilderConfig`
  instead.
- Invalid configuration fails with one `IllegalStateException` listing every problem, rather than
  `GraphIndexBuilder`'s `IllegalArgumentException` for the first one.
- With `withExistingGraph`, setting `withMaxDegree(s)` or `withAddHierarchy` is rejected as a conflict (the
  existing graph fixes both). The existing graph keeps the diversity provider it was created with, which must be
  able to score the ordinals you append.
- As with `GraphIndexBuilder.addGraphNode`, every node must be scoreable by ordinal through the build score
  provider before (or when) it is added: add the vector to your vector values, or encode its PQ code, first.
  Otherwise a later insert or `cleanup()` fails with an `IndexOutOfBoundsException`.

## Behavior changes

- Adding nodes to an `OnHeapGraphIndex` after `cleanup()` now works correctly. Previously the graph stayed
  "frozen" after cleanup, so a later insert could link a node to itself (tripping an assertion) and concurrent
  searches could see half-inserted nodes. Adding a node now unfreezes the graph; call `cleanup()` again before
  writing it, and don't reuse views or searchers obtained while it was frozen. This also affects continuing to
  build on a graph that has already been cleaned up, including one returned by `GraphIndexBuilder.build`.
- `GraphIndexBuilder.rescore` now keeps nodes that were marked deleted (with `markNodeDeleted`) but not yet
  removed. Previously the rescored graph dropped those marks, so the nodes became visible to searches again and
  survived `cleanup()`. Callers that don't delete before rescoring, such as a compaction refining its PQ
  codebook, see no change.

## Other changes to public classes

- `GraphSearcher` implements `IndexSearcher`.
- `AbstractGraphIndexWriter.Builder` implements `PersistableGraphIndex.GraphIndexWriterBuilder`.
- `@Experimental` has moved from `jvector-base` to `jvector-api` (same package and name) and is now `@Documented`,
  so it appears in the generated Javadoc.
- Writer documentation corrected, with no behavior change: `withParallelWorkerThreads` and
  `withParallelDirectBuffers` apply only to the parallel writer. The single-threaded random-access and sequential
  writer builders throw `UnsupportedOperationException` for them, and `RandomAccessOnDiskGraphIndexWriter.Builder`
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
