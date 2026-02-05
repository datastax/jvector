# JVector Tutorial

JVector provides a graph index for ANN search which is a hybrid of DiskANN and HNSW. You can think of it as a Vamana index with an HNSW-style hierarchy. The rest of this tutorial assumes you have a basic understanding of Vector search, but no prior understanding of HNSW or DiskANN is assumed.

JVector provides a `VectorFloat` datatype for representing vectors, as an abstraction over the physical vector type. Therefore, the first step to using JVector is to understand how to create a `VectorFloat`:

```java
// `VectorizationProvider` is automatically picked based on the system, language version and runtime flags
// and determines the actual type of the vector data, and provides implementations for common operations
// like the inner product.
VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

int dimension = 3;

// Create a `VectorFloat` from a `float[]`.
// The types that can be converted to a VectorFloat are technically dependent on which VectorizationProvider is picked,
// but `float[]` is generally a safe bet.
float[] vector0Array = new float[]{0.1f, 0.2f, 0.3f};
VectorFloat<?> vector0 = vts.createFloatVector(vector0Array);
```

> [!TIP]
> For other ways to create vectors, refer to the javadoc for `VectorTypeSupport`.

Before creating the vector index, we will group all of our base vectors into a container which implements the `RandomAccessVectorValues` interface. Many APIs in JVector accept an instance of `RandomAccessVectorValues` as input. In this case, we'll use it to specify the vectors to be used to build the index.

```java
// This toy example uses only three vectors, in practical cases you might have millions or more.
List<VectorFloat<?>> baseVectors = List.of(
    vector0,
    vts.createFloatVector(new float[]{0.01f, 0.15f, -0.3f}),
    vts.createFloatVector(new float[]{-0.2f, 0.1f, 0.35f})
);

// RAVV or `ravv` is convenient shorthand for a RandomAccessVectorValues instance
RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, dimension /* 3 */);
```

> [!TIP]
> In this example, all vectors are loaded in-memory, but RAVVs are quite versatile. For example, you might have a RAVV backed by disk (check out `MMapRandomAccessVectorValues.java`) or write your own custom RAVV that transfers data over a network interface.

> [!NOTE]
> A note on terminology:
> - "Base" vectors are the vectors used to build the index. Each vector becomes a node in the graph. May also be referred to as the "train" set.
> - "Query" vectors are vectors used as queries for ANN search after the index has been built. In some cases you may want to use some base vectors as queries. Also referred to as the "test" set.

We're now ready to create a Graph-based vector index. We'll do this using a `GraphIndexBuilder` as an intermediate. Let's take a look at the signature of one of it's constructors:

```java
public GraphIndexBuilder(BuildScoreProvider scoreProvider,
                         int dimension,
                         int M,
                         int beamWidth,
                         float neighborOverflow,
                         float alpha,
                         boolean addHierarchy,
                         boolean refineFinalGraph);
```

This constructor asks for something called a `BuildScoreProvider`, the vector dimension, and a set of graph parameters.

The `BuildScoreProvider` is used by the graph builder to compute the similarity scores between any two vectors at build time. We'll use the RAVV we created earlier to generate a BuildScoreProvider:

```java
// The type of similarity score to use. JVector supports EUCLIDEAN (L2 distance), DOT_PRODUCT and COSINE.
VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.EUCLIDEAN;

// A simple score provider which can compute exact similarity scores by holding a reference to all the base vectors.
BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, similarityFunction);
```

Let's also initialize the graph parameters. For now we will not worry about the exact function of the parameters, except to note that these are reasonable defaults. Refer the DiskANN and HNSW papers for more details.

<!-- TODO describe graph parameters in a separate doc -->

```java
// Graph construction parameters
int M = 32;  // maximum degree of each node
int efConstruction = 100;  // search depth during construction
float neighborOverflow = 1.2f;
float alpha = 1.2f;  // note: not the best setting for 3D vectors, but good in the general case
boolean addHierarchy = true;  // use an HNSW-style hierarchy
boolean refineFinalGraph = true;
```

and now we can create the graph index

```java
// Build the graph index using a Builder
// Remember to close the builder using builder.close() or a try-with-resources block
GraphIndexBuilder builder = new GraphIndexBuilder(bsp,
                                                  dimension,
                                                  M,
                                                  efConstruction,
                                                  neighborOverflow,
                                                  alpha,
                                                  addHierarchy,
                                                  refineFinalGraph);
ImmutableGraphIndex graph = builder.build(ravv);
```

> [!NOTE]
> You may notice that we supplied the same `ravv` to `builder.build`, even though we'd already passed in the RAVV while creating the `BuildScoreProvider`. This is necessary since generally speaking, the `BuildScoreProvider` won't keep a reference to the actual base vectors, it just so happens that we're using an "exact" score provider that does so.

At this point, you have a completed Graph Index that resides in-memory.

To perform a search operation, you need to first create a `GraphSearcher`.

> [!IMPORTANT]
> The graph index itself can be shared between threads, but `GraphSearcher`s maintain internal state and are therefore NOT thread-safe. To run concurrent searches across multiple threads, each thread should have it's own `GraphSearcher`. The same searcher can be re-used across different queries in the same thread.

```java
// Remember to close the searcher using searcher.close() or a try-with-resources block
var searcher = new GraphSearcher(graph);
```

Generally speaking, you can't pass in a `VectorFloat<?>` directly to the `GraphSearcher`. You need to wrap the query vector with a `SearchScoreProvider`, similar in spirit to the `BuildScoreProvider` we created earlier.

```java
VectorFloat<?> queryVector = vts.createFloatVector(new float[]{0.2f, 0.3f, 0.4f});  // for example
// The in-memory graph index doesn't own the actual vectors used to construct it.
// To compute exact scores at search time, you need to pass in the base RAVV again,
// in addition to the actual query vector
SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(queryVector, similarityFunction, ravv);
```

Now we can run a search

```java
int topK = 10;  // number of approximate nearest neighbors to fetch
// You can provide a filter to the query as a bit mask.
// In this case we want the actual topK neighbors without filtering,
// so we pass in a virtual bit mask representing all ones.
SearchResult result = searcher.search(ssp, topK, Bits.ALL);

for (NodeScore ns : result.getNodes()) {
    int id = ns.node;  // you can look up this ID in the RAVV
    float score = ns.score;  // the similarity score between this vector and the query vector (higher -> more similar)
    System.out.println("ID: " + id + ", Score: " + score + ", Vector: " + ravv.getVector(id));
}
```

For the full example, refer `jvector-examples/../VectorIntro.java`. Run it using
```sh
mvn compile exec:exec@example -Dexample=intro -pl jvector-examples -am
```

Next steps:
- Understand index construction parameters
- Overquerying to improve search accuracy
- Quantization for space efficiency
- Building indexes for larger-than-memory datasets on disk
- VectorizationProviders
