# JVector Tutorial Part 3: Larger-Than-Memory Indexes with Product Quantization

In the previous tutorials, we built indexes in-memory then wrote them to disk. However, this requires enough memory to hold all the vectors during construction. In this tutorial we'll build larger-than-memory indexes using two key techniques:
- Write vectors to disk as they are streamed in
- Keep compressed versions of the vectors in the memory to allow JVector to compute similarities during construction. We'll use Product Quantization (PQ).

> ![TIP]
> Product Quantization is a lossy compression technique that works by:
> 1. Dividing each vector into subspaces (e.g., a 128-dimensional vector into 16 subspaces of 8 dimensions each)
> 2. Learning a codebook of centroids for each subspace
> 3. Representing each subspace by the index of its nearest centroid
> 
> For example, consider 128 dimension vectors. With 16 subspaces of 256 centroids per subspace, each vector is represented by 16 bytes. instead of 512 bytes, achieving a 32x compression ratio.
>
> Read the [PQ paper](https://ieeexplore.ieee.org/document/5432202) for more details.

## Loading the Dataset

We'll change things up with a different dataset:

```java
// The DataSet provided by loadDataSet is in-memory,
// but you can apply the same technique even when you don't have
// the base vectors in-memory.
DataSet dataset = DataSetLoader.loadDataSet("e5-small-v2-100k");

// Remember that RAVVs need not be in-memory in the general case.
// We will sample from this RAVV to compute the PQ codebooks.
// In general you don't need to have a RAVV over all the vectors to
// build PQ codebooks, but you do need a "representative set".
RandomAccessVectorValues ravv = dataset.getBaseRavv();
VectorSimilarityFunction vsf = dataset.getSimilarityFunction();
int dim = dataset.getDimension();
```

## Computing the Product Quantization Codebook

Before we can compress vectors, we need to compute a PQ codebook using a representative set of vectors. This is just a set of vectors whose distribution matches or approximates the distribution of the entire set of base vectors. In this case we'll select the representative set by randomly sampling from the entire set of vectors.

```java
// PQ parameters
int subspaces = 64;  // number of subspaces to divide each vector into
int centroidsPerSubspace = 256;  // number of centroids per subspace (256 => 1 byte)
boolean centerDataset = false;  // we won't ask to center the dataset before quantization

// This method randomly samples at most MAX_PQ_TRAINING_SET_SIZE vectors
// from the RAVV and considers that a "representative set" used to build the codebooks.
ProductQuantization pq = ProductQuantization.compute(ravv, subspaces, centroidsPerSubspace, centerDataset);
```

## Setting Up for Incremental Construction

Now we'll set up the structures needed for incremental index construction:

```java
// MutablePQVectors is a thread-safe, dynamically growing container for compressed vectors.
// As we add vectors to the index, we'll compress them and store them here.
// These compressed vectors are used during graph construction for approximate distance calculations.
var pqVectors = new MutablePQVectors(pq);

// Provides approximate scores during graph construction using the compressed vectors
BuildScoreProvider bsp = BuildScoreProvider.pqBuildScoreProvider(vsf, pqVectors);
```

Since we have a BuildScoreProvider, we're now ready to create and use a `GraphIndexBuilder`. We will also create the corresponding `GraphIndexWriter` at the same time. Unlike the last tutorial we created the writer only after building the complete graph, we'll write full-resolution vectors to disk in tandem with adding them to the graph.

```java
// Initialize the graph parameters M, ef etc
// ...

Path graphPath = Files.createTempFile("jvector-ltm-graph", null);

// remember to close these manually or use try-with-resources
GraphIndexBuilder builder = new GraphIndexBuilder(bsp, dim, M, ef, overflow, alpha, addHierarchy);
OnDiskGraphIndexWriter writer = new OnDiskGraphIndexWriter.Builder(builder.getGraph(), graphPath)
        .with(new InlineVectors(dim))
        // Since we start with an empty graph, the writer will, by default,
        // assume an ordinal mapping of size 0 (which is obviously incorrect).
        // This is easy to rectify if you know the number of vectors beforehand,
        // if not you may need to implement OrdinalMapper yourself.
        .withMapper(new OrdinalMapper.IdentityMapper(ravv.size() - 1))
        .build();
```

## Incremental Index Construction

We'll proceed to add vectors to the index one at a time. The general procedure is:
- Encode the vector using the PQ codebooks created earlier, and add it to the collection of PQ vectors
- Write the full-resolution vector to disk
- Add the new vector to the graph

```java
for (int ordinal = 0; ordinal < ravv.size(); ordinal++) {
    VectorFloat<?> v = ravv.getVector(ordinal);

    // Encode and add the vector to the working set of PQ vectors,
    // which allows the graph builder to access it through the BuildScoreProvder.
    pqVectors.encodeAndSet(ordinal, v);

    // Write the feature (full-resolution vector) for a single vector instead of all at once.
    writer.writeInline(ordinal, Feature.singleState(FeatureId.INLINE_VECTORS, new InlineVectors.State(v)));

    builder.addGraphNode(ordinal, v);
}
```

> ![TIP]
> When you call `builder.build(ravv)`, vectors from the RAVV are added to the graph in parallel. There's no parallelization if you add vectors one by one in the same thread. But you can parallelize manually by inserting from multiple threads at once, see the full example in LargerThanMemory.java.

Once we've added all the vectors we can write the graph structure to disk:

```java
// Must be done manually for incrementally built graphs.
// Enforces maximum degree constraint among other things.
builder.cleanup();

// Write the graph structure (neighbor lists) to disk.
// No need to pass-in a feature supplier since we wrote the features incrementally
// using writer.writeInLine
writer.write(Map.of());
```

We should also save the PQ vectors we created since we'll need them later on when it's time to run searches:

```java
// PQ codebooks and vectors also need to be saved somewhere! (we'll not concern ourselves with Fused PQ)
Path pqPath = Files.createTempFile("jvector-ltm-pq", null);
try (DataOutputStream pqOut = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(pqPath)))) {
    pqVectors.write(pqOut);
}
```

> ![NOTE]
> JVector supports a Graph feature called "Fused PQ" using which you can embed the PQ vectors and codebooks in the on-disk graph, similar to how full-resolution vectors are embedded. This saves you from the need to store them separately, but we won't cover that in this tutorial.

## Searching with Two-Pass Approach

Now that we've built and saved the index and the PQ vectors, we're ready to start searching. The only things that need to be held in memory are the PQ Vectors and the upper layers of the graph (if the graph is hierarchial). The full-resolution vectors and the lower layer of the graph remains on-disk and are read on demand.

Searching is done in two phases:
- The first phase searches the graph to identify a set of candidates (as many as `rerankK`). In-memory PQ vectors are used to compute scores.
- In the second phase reranks the candidates and returns the top `topK` results. Exact scores are computed using full-resolution vectors from disk.

```java
// nothing new here
ReaderSupplier graphSupplier = ReaderSupplierFactory.open(graphPath);
OnDiskGraphIndex graph = OnDiskGraphIndex.load(graphSupplier);
// except that we also need a reader for the PQ vectors
ReaderSupplier pqSupplier = ReaderSupplierFactory.open(pqPath);
RandomAccessReader pqReader = pqSupplier.get()
// don't forget to close all of the above! (or just use try-with-resources)

// we need to have the PQ vectors in memory
PQVectors pqVectorsSearch = PQVectors.load(pqReader);

int topK = 10;
float overqueryFactor = 32.0f;
int rerankK = (int) (topK * overqueryFactor);

// a friendly reminder that searchers need closing too
GraphSearcher searcher = new GraphSearcher(graph);
var graphRavv = (RandomAccessVectorValues) searcher.getView();
VectorFloat<?> query : dataset.getQueryVectors().get(0);

// Two-phase search:
// 1. ApproximateScoreFunction (ASF) uses compressed vectors for fast initial search
// 2. Reranker uses full-resolution vectors from disk for accurate final ranking
var asf = pqVectorsSearch.precomputedScoreFunctionFor(query, vsf);
var reranker = graphRavv.rerankerFor(query, vsf);
SearchScoreProvider ssp = new DefaultSearchScoreProvider(asf, reranker);

SearchResult sr = searcher.search(ssp, topK, rerankK, 0.0f, 0.0f, Bits.ALL);
```

## Full Example

The complete code from this tutorial is available in `LargerThanMemory.java`. Run it from the root of this repo using:

```sh
mvn compile exec:exec@example -Dexample=ltm -pl jvector-examples -am
```

## Next Steps

- Fused PQ
- NVQ
