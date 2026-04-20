# Graph Construction Guide

This guide explains how to build, configure, and persist graph indexes using JVector.
It covers both `OnHeapGraphIndex` (built and searched entirely in memory) and
`OnDiskGraphIndex` (persisted to disk and searched with memory-mapped I/O), as well as
the relationship between the two.

---

## Interface hierarchy

```
GraphIndex
├── MutableGraphIndex     — a live graph that accepts concurrent insertions and deletions
└── PersistableGraphIndex — a fully-built graph that can be written to disk
```

`OnHeapGraphIndex` implements **both** interfaces: it starts as a `MutableGraphIndex`
during construction and becomes a `PersistableGraphIndex` when building is finished.

`OnDiskGraphIndex` implements only `PersistableGraphIndex` — it is always read-only.

In practice you interact with `GraphIndexBuilder` rather than these interfaces
directly; the builder owns the mutable graph and hands you a `PersistableGraphIndex`
when construction is complete.

---

## Building an OnHeapGraphIndex

### 1. Provide your vectors

Implement or instantiate a `RandomAccessVectorValues` (RAVV) over your dataset.
The library ships `ListRandomAccessVectorValues` for convenience:

```java
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

int dimension = 1536; // must match the dimensionality of your model
List<VectorFloat<?>> vectors = ...; // your dataset

RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vectors, dimension);
```

A RAVV maps integer **node ordinals** (0, 1, 2, …) to vectors. The ordinal you use
when searching is the same ordinal you pass to the RAVV to retrieve the original vector.

### 2. Choose a similarity function

```java
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

// Choose one:
VectorSimilarityFunction vsf = VectorSimilarityFunction.COSINE;      // normalised dot product
VectorSimilarityFunction vsf = VectorSimilarityFunction.DOT_PRODUCT;  // raw dot product
VectorSimilarityFunction vsf = VectorSimilarityFunction.EUCLIDEAN;    // L2 distance
```

Use `COSINE` or `DOT_PRODUCT` for embedding models that produce unit-normalised vectors
(OpenAI, Cohere, etc.). Use `EUCLIDEAN` when absolute Euclidean distance is meaningful.

### 3. Create a BuildScoreProvider

The `BuildScoreProvider` tells the builder how to compute similarity between vectors
during graph construction. The simplest form wraps your RAVV directly:

```java
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;

BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, vsf);
```

### 4. Understand the construction parameters

| Parameter | Type | Description |
|---|---|---|
| `M` | `int` | Maximum number of edges per node (maximum out-degree). Higher values improve recall at the cost of memory and build time. **Typical range: 16–64.** |
| `beamWidth` (efConstruction) | `int` | Number of candidate nodes examined per insertion during construction. Higher values improve graph quality at the cost of build time. **Typical range: 64–200.** |
| `neighborOverflow` | `float` | Temporary per-node edge budget during construction, expressed as a multiple of `M`. A value of 1.2 allows 20% extra edges while building, which are trimmed during cleanup. Must be ≥ 1.0. **Typical value: 1.2–1.5.** |
| `alpha` | `float` | Diversity pruning aggressiveness. Values > 1.0 allow longer edges, improving recall in high-dimensional spaces. Setting alpha = 1.0 produces a flat HNSW-style graph. **Typical value: 1.2.** |
| `addHierarchy` | `boolean` | When `true`, adds HNSW-style coarser layers above the base layer (layer 0). Improves search performance on large graphs. Disable only for very small datasets or special use cases. |
| `refineFinalGraph` | `boolean` | When `true`, performs a second pass over each node to improve its connections. Improves recall at the cost of build time. Defaults to `true`. |

### 5. Construct the builder and build

```java
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.PersistableGraphIndex;

int M             = 32;
int beamWidth     = 100;
float overflow    = 1.2f;
float alpha       = 1.2f;
boolean hierarchy = true;

PersistableGraphIndex graph;
try (GraphIndexBuilder builder = new GraphIndexBuilder(
        bsp, dimension, M, beamWidth, overflow, alpha, hierarchy)) {
    graph = builder.build(ravv);
}
```

`build()` inserts all vectors from the RAVV in parallel, then calls `cleanup()` which
finalises edge lists, removes any deleted nodes, and marks the graph as immutable.
The returned `PersistableGraphIndex` is an `OnHeapGraphIndex` under the hood.

### 6. Incremental insertions

For streaming workloads you can add nodes one at a time instead of calling `build()`:

```java
try (GraphIndexBuilder builder = new GraphIndexBuilder(
        bsp, dimension, M, beamWidth, overflow, alpha, hierarchy)) {

    for (int i = 0; i < vectors.size(); i++) {
        builder.addGraphNode(i, vectors.get(i));
    }

    // Must call cleanup() before writing to disk or treating the graph as finished.
    builder.cleanup();
    PersistableGraphIndex graph = (PersistableGraphIndex) builder.getGraph();
}
```

`addGraphNode` is thread-safe and can be called from multiple threads concurrently.
`cleanup()` is not thread-safe — ensure all insertions are complete before calling it.

### 7. Deleting nodes

Mark a node for deletion before calling `cleanup()`:

```java
builder.markNodeDeleted(nodeOrdinal);
// ...
builder.cleanup(); // removal and reconnection happen here
```

Deleted nodes are reconnected in the graph (their in-edges are bridged to their
neighbours) to preserve connectivity before the nodes are physically removed.

---

## Searching an OnHeapGraphIndex

Use a `GraphSearcher` backed by the built index. `GraphSearcher` is not thread-safe —
create one per thread.

```java
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;

VectorFloat<?> queryVector = ...; // the vector you want to search for

// The SearchScoreProvider knows how to score candidates against the query.
// "exact" means it computes exact similarity from the RAVV at search time.
SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(queryVector, vsf, ravv);

int topK = 10;

try (GraphSearcher searcher = new GraphSearcher(graph)) {
    SearchResult result = searcher.search(ssp, topK, Bits.ALL);
    for (var ns : result.getNodes()) {
        System.out.printf("node %d  score %.4f%n", ns.node, ns.score);
    }
}
```

`Bits.ALL` means no filter — all nodes are considered candidates. Pass a custom `Bits`
implementation to restrict results to a subset of nodes (e.g. for metadata filtering).

---

## Persisting to disk

After building, write the graph to a file using a `WriteBuilder`. You must declare
which **features** to embed alongside the graph structure. The most common feature is
`InlineVectors`, which stores the raw vectors inside the graph file so that an
`OnDiskGraphIndex` can score candidates without a separate vector file.

### Writing with inline vectors

```java
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;

import java.nio.file.Path;
import java.util.Map;

Path outputPath = Path.of("/tmp/my-graph.jvector");

try (GraphIndex.WriteBuilder writer = graph.writer(outputPath)
        .with(new InlineVectors(dimension))) {
    writer.write(Map.of(
        FeatureId.INLINE_VECTORS,
        nodeId -> new InlineVectors.State(ravv.getVector(nodeId))
    ));
}
```

The `write()` method takes a map from `FeatureId` to a per-node supplier function
(`IntFunction<Feature.State>`). The supplier is called once per node ordinal and must
return the feature data for that node.

### Writing without vectors (graph structure only)

If you store vectors separately and do not need them inline:

```java
try (GraphIndex.WriteBuilder writer = graph.writer(outputPath)) {
    writer.write(Map.of()); // no features
}
```

### Ordinal remapping

When nodes have been deleted, gaps remain in the ordinal space. Use sequential
remapping to compact the on-disk representation:

```java
import io.github.jbellis.jvector.graph.disk.AbstractGraphIndexWriter;

Map<Integer, Integer> oldToNew = AbstractGraphIndexWriter.sequentialRenumbering(graph);

try (GraphIndex.WriteBuilder writer = graph.writer(outputPath)
        .with(new InlineVectors(dimension))
        .withMap(oldToNew)) {
    writer.write(Map.of(
        FeatureId.INLINE_VECTORS,
        nodeId -> new InlineVectors.State(ravv.getVector(nodeId))
    ));
}
```

The remapped ordinals in the on-disk file will differ from the original in-memory
ordinals, so keep track of this mapping if you need to correlate results back to
your original IDs.

### Parallel writing (large graphs)

For large graphs, parallel writing can reduce write time significantly:

```java
try (GraphIndex.WriteBuilder writer = graph.writer(outputPath)
        .with(new InlineVectors(dimension))
        .withParallelWorkerThreads(-1)) { // -1 = use all available processors
    writer.write(Map.of(
        FeatureId.INLINE_VECTORS,
        nodeId -> new InlineVectors.State(ravv.getVector(nodeId))
    ));
}
```

### Convenience write method

For simple use cases without custom features or ordinal mappings, there is a one-liner:

```java
OnDiskGraphIndex.write(graph, ravv, outputPath);
```

This writes inline vectors using sequential ordinal renumbering.

---

## Loading and searching an OnDiskGraphIndex

### Loading

```java
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;

ReaderSupplier readerSupplier = ReaderSupplierFactory.open(outputPath);
OnDiskGraphIndex diskGraph = OnDiskGraphIndex.load(readerSupplier);
```

`ReaderSupplierFactory` automatically selects the most efficient reader available on
the current platform (memory-mapped files, native I/O, etc.). Keep `readerSupplier`
open for the lifetime of the `OnDiskGraphIndex` and close it when done.

### Searching

An `OnDiskGraphIndex.View` implements `RandomAccessVectorValues`, so when inline
vectors are present you can use the view as your RAVV at search time — no separate
vector store is needed:

```java
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;

int topK = 10;

try (GraphSearcher searcher = new GraphSearcher(diskGraph)) {
    // Extract the view from the searcher to use as a RAVV.
    // Reuse the same searcher (and its view) across queries on the same thread.
    var graphRavv = (RandomAccessVectorValues) searcher.getView();

    SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(queryVector, vsf, graphRavv);
    SearchResult result = searcher.search(ssp, topK, Bits.ALL);
}
```

### Over-querying for higher recall

Fetch more candidates than `topK` from the graph and re-rank them, then return only
the best `topK`. This trades latency for recall:

```java
int topK    = 10;
int refineK = 50; // fetch 5× more candidates than needed

SearchResult result = searcher.search(ssp, topK, refineK, 0.0f, 0.0f, Bits.ALL);
```

`refineK` must be ≥ `topK`. As `refineK` increases toward the dataset size, recall
approaches 1.0 at the cost of more I/O and computation.

### Multi-threaded search

Create one `GraphSearcher` per thread. `OnDiskGraphIndex` itself is thread-safe for
concurrent reads; only the searcher (and its underlying `View`) is per-thread:

```java
// Example using a thread pool
ExecutorService pool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

// Each task creates its own searcher
Callable<SearchResult> task = () -> {
    try (GraphSearcher searcher = new GraphSearcher(diskGraph)) {
        var graphRavv = (RandomAccessVectorValues) searcher.getView();
        SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(queryVector, vsf, graphRavv);
        return searcher.search(ssp, topK, Bits.ALL);
    }
};
```

### Closing

```java
diskGraph.close();
readerSupplier.close();
```

---

## Typical end-to-end workflow

```java
// 1. Build in memory
BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, vsf);
PersistableGraphIndex heapGraph;
try (GraphIndexBuilder builder = new GraphIndexBuilder(bsp, dim, 32, 100, 1.2f, 1.2f, true)) {
    heapGraph = builder.build(ravv);
}

// 2. Write to disk with inline vectors
Path graphPath = Path.of("/data/my-index.jvector");
try (GraphIndex.WriteBuilder writer = heapGraph.writer(graphPath)
        .with(new InlineVectors(dim))) {
    writer.write(Map.of(
        FeatureId.INLINE_VECTORS,
        nodeId -> new InlineVectors.State(ravv.getVector(nodeId))
    ));
}

// 3. Load from disk
ReaderSupplier rs = ReaderSupplierFactory.open(graphPath);
OnDiskGraphIndex diskGraph = OnDiskGraphIndex.load(rs);

// 4. Search
try (GraphSearcher searcher = new GraphSearcher(diskGraph)) {
    var graphRavv = (RandomAccessVectorValues) searcher.getView();
    SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(queryVector, vsf, graphRavv);
    SearchResult result = searcher.search(ssp, topK, Bits.ALL);
}

// 5. Clean up
diskGraph.close();
rs.close();
```

---

## Parameter tuning reference

| Goal | Adjust |
|---|---|
| Higher recall | Increase `M`, `beamWidth`, or `refineK` at search time |
| Faster build | Decrease `beamWidth`; set `refineFinalGraph = false` |
| Less memory during build | Decrease `neighborOverflow` (minimum 1.0) |
| Faster search | Decrease `refineK`; decrease `M` |
| Better graph quality for high-dimensional data | Increase `alpha` (e.g. 1.4–1.5) |
| Flat (single-layer) graph | Set `addHierarchy = false` |
