# NVQ

[NVQ](https://arxiv.org/abs/2509.18471) (non-uniform vector quantization) stores a compact,
high-accuracy copy of each vector for **reranking**: the second search pass that rescores the best
candidates found with the traversal codes. It splits each vector into subvectors, centers them on
the dataset mean, and quantizes each subvector to 8 bits per dimension with a non-uniform
quantizer learned for that subvector.

## When to use it

- Reranking on-disk indexes, in place of full FP32 vectors: about a quarter of the size, so each
  candidate reads far fewer bytes, with accuracy close to FP32.
- Together with any traversal quantizer (PQ, ASH, BQ) or fused codes. NVQ is not used for
  traversal.

## Create the model

NVQ needs only the dataset mean, so it can be created without another pass over the vectors:

```java
// from a mean collected during an earlier scan (for example VectorMean while encoding ASH)
NVQuantization nvq = NVQuantization.create(mean, 2);       // 2 subvectors
// or let it compute the mean
NVQuantization nvq = NVQuantization.compute(vectors, 2);
```

The second argument is the number of subvectors. Each subvector adds 28 bytes of parameters, so
fewer subvectors are smaller; the examples in this repository use 1 or 2.

## Write and search

Write NVQ vectors into the graph with the `NVQ` feature (inline in each record) or `SeparatedNVQ`
(in their own block; see [Fused and separated layouts](fused-and-separated.md)):

```java
writer = new OnDiskParallelGraphIndexWriter.Builder(graph, path)
        .with(new FusedASH(32, ash))                    // or FusedPQ, or no fused codes
        .with(new NVQ(nvq))
        .build();
writer.write(Map.of(
        FeatureId.FUSED_ASH,   node -> new FusedASH.State(view, codes, node),
        FeatureId.NVQ_VECTORS, node -> new NVQ.State(nvq.encode(vectors.getVector(node)))));
```

At search time the index's reranker uses the stored NVQ vectors automatically.
[Larger-than-memory indexes](larger-than-memory-indexes.md#construction) has a complete example,
and [Deferred batch serialization](deferred-batch-writer.md) shows how to encode NVQ vectors while
the graph is written instead of holding them all in memory.

## Size

`4 + Σ over subvectors of (subvector dimensions + 28)` bytes per vector: dimension + 60 with two
subvectors, for example 1,084 bytes for a 1,024-dimension vector (FP32: 4,096). See
[Memory and disk sizing](sizing.md).
