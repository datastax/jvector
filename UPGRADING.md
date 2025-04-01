# Upgrading from 3.0.x to 4.0.x

## New features
- Support for Non-uniform Vector Quantization (NVQ, pronounced as "new vec"). This new technique quantizes the values
  in each vector with high accuracy by first applying a nonlinear transformation that is individually fit to each
  vector. These nonlinearities are designed to be lightweight and have a negligible impact on distance computation
  performance.
- Support for hierarchical graph indices. These new type of indices blends HNSW and DiskANN in a novel way. An
  HNSW-like hierarchy resides in memory for quickly seeding the search. This also reduces the need for caching the
  DiskANN graph near the entrypoint. The base layer of the hierarchy is a DiskANN-like index and inherits its
  properties. This hierarchical structure can be disabled, ending up with just the base DiskANN layer.  

## API changes
- MemorySegmentReader.Supplier and SimpleMappedReader.Supplier must now be explicitly closed, instead of being
  closed by the first Reader created from them.
- OnDiskGraphIndex no longer closes its ReaderSupplier
- The constructor of GraphIndexBuilder takes an additional parameter which allows to enable or disable the use of the
  hierarchy.
- GraphSearcher can be configured to run pruned searches using GraphSearcher.usePruning. When this is set to true,
  we do early termination of the search. In certain cases, this can accelerate the search at the potential cost of some
  accuracy. It is set to false by default.

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
