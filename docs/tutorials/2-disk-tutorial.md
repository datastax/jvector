# JVector Tutorial Part 2: The OnDiskGraphIndex

In-memory indexes are relatively simple to create, but they are not persistent and are fundamentally limited by the amount of memory available on the machine. To solve this, JVector provides an `OnDiskGraphIndex` backed by a file.

Unlike the previous example, this time we'll use a proper ANN dataset based on OpenAI's text-embedding-ada-002 embedding model.

```java
// This is a preconfigured dataset that will be downloaded automatically.
DataSet dataset = DataSets.loadDataSet("ada002-100k").orElseThrow(() ->
    new RuntimeException("Dataset doesn't exist or wasn't configured correctly")
);
```

> [!TIP]
> A `DataSet` provides the base vectors used to be indexed, the query vectors, and the expected or "ground truth" results used for computing accuracy metrics.

We'll create the graph in-memory in exactly the same way as we did in the introductory tutorial. You can't create larger-than-memory indexes this way, but we'll cover how to do that in a later tutorial.

```java
// The loaded DataSet provides a RAVV over the base vectors
RandomAccessVectorValues ravv = dataset.getBaseRavv();
VectorSimilarityFunction vsf = dataset.getSimilarityFunction();
int dim = dataset.getDimension();

// reasonable defaults
int M = 32;
int ef = 100;
float overflow = 1.2f;
float alpha = 1.2f;
boolean addHierarchy = true;

BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, vsf);

// nothing new here
ImmutableGraphIndex heapGraph;
try (GraphIndexBuilder builder = new GraphIndexBuilder(bsp, dim, M, ef, overflow, alpha, addHierarchy)) {
    heapGraph = builder.build(ravv);
}
```

We can write this graph to the disk using a `GraphIndexWriter`.

```java
Path graphPath = Files.createTempFile("jvector-example-graph", null);  // or wherever you want to save the graph

// Create a writer for the on-heap graph we just built.
// Remember to close when done.
GraphIndexWriter writer = GraphIndexWriter.getBuilderFor(GraphIndexWriterTypes.RANDOM_ACCESS_PARALLEL, heapGraph, graphPath)
    // Let the writer know that we'll also be passing in the actual vector data
    // to be saved "inline" with the data for each corresponding graph node.
    .with(new InlineVectors(dim))
    .build();
```

This is a good time to discuss "Features" associated with on-disk JVector indexes. A Feature is any kind of additional information about each node in the graph that gets saved to the same index file. Here, we'll be writing "Inline Vectors" as a feature, which means that the vector value will be stored alongside the neighbor lists for each node in the graph. This will be useful later.

Since we want to include a feature, we need to tell the writer how to obtain the feature's "state" (or value) for each node. For inline vectors, this just means we need to specify the vector associated with each node.

```java
// Supply one map entry for each feature.
// The key is a FeatureId enum corresponding to the feature
// and the value is a function which generates the feature state for each graph node.
writer.write(Map.of(
    FeatureId.INLINE_VECTORS,
    // we already have a RAVV, so we'll just use that to supply the writer.
    nodeId -> new InlineVectors.State(ravv.getVector(nodeId))));
// writer.close() if not using try-with-resources
```

At this point the graph index has been written to disk and can be used by creating an instance of `OnDiskGraphIndex`.

To do so, we'll need to specify a `RandomAccessReader` implementation which JVector will use to read parts of the file as needed. This interface isn't thread safe, so when creating an `OnDiskGraphIndex` we pass in a `ReaderSupplier` object which can be used by the `OnDiskGraphIndex` to create `RandomAccessReader`s as needed.

```java
// ReaderSupplierFactory automatically picks an available RandomAccessReader implementation
ReaderSupplier readerSupplier = ReaderSupplierFactory.open(graphPath);
OnDiskGraphIndex graph = OnDiskGraphIndex.load(readerSupplier);
```

Now we can perform searches exactly as with the in-memory index, with one minor difference: for the in-memory index, we had to keep track of the RAVV we used to build the graph in order to create an exact `SearchScoreProvider` for each query. In this case, since Inline vectors are available as a feature, we can acquire a RAVV from the index itself.

```java
GraphSearcher searcher = new GraphSearcher(graph);
// Views of an OnDiskGraphIndex with inline or separated vectors can be used as RAVVs!
// In multi-threaded scenarios you should have one searcher per thread
// and extract a view for each thread from the associated searcher.
RandomAccessVectorValues graphRavv = (RandomAccessVectorValues) searcher.getView();

// number of search results we want
int topK = 10;
// `rerankK` controls the number of nodes to fetch from the initial graph search.
// which are then re-ranked to return the actual topK results.
// Increasing rerankK improves accuracy at the cost of latency and throughput.
int rerankK = 20;
VectorFloat<?> query = dataset.getQueryVectors().get(0);
// use the RAVV from the graph instead of the one from the original dataSet
SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(query, vsf, graphRavv);
// A slightly more complex overload of `search` which adds three extra parameters.
// Right now we only care about `rerankK`.
SearchResult sr = searcher.search(ssp, topK, rerankK, 0.0f, 0.0f, Bits.ALL);
```

The code from this tutorial is available in `DiskIntro.java`. Run it from the root of this repo using
```sh
mvn compile exec:exec@tutorial -Dtutorial=disk -pl jvector-examples -am
```
The full example also illustrates the impact that adjusting `rerankK` has on recall.

Next tutorial:
- Larger than memory indexes with Product Quantization
