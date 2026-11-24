# Binary quantization (BQ)

Binary quantization keeps one bit per dimension: the sign of each component. Scores are Hamming
distances between bit strings, which approximate cosine similarity. It needs no training and is
the cheapest code to compute and compare, but also the coarsest.

## When to use it

- Very high-dimensional embeddings, where even one bit per dimension leaves enough information to
  steer the graph search, and cheap construction and traversal matter most.
- Cosine similarity, or vectors normalized to unit length. BQ ignores the similarity function and
  always scores by Hamming distance, so it is a poor fit for unnormalized dot products and
  Euclidean distance.
- Always with reranking (FP32 or NVQ), and usually with a larger rerank depth than PQ or ASH need.

For most data, [ASH](ash-configuration.md) at 1 bit per projected dimension, or
[PQ](pq.md), is more accurate at the same or smaller size; measure on your data.

## Use it

```java
var bq = new BinaryQuantization(vectors.dimension());
var codes = (BQVectors) bq.encodeAll(vectors);

// construction scored with BQ codes
var buildScores = BuildScoreProvider.bqBuildScoreProvider(codes);

// search: BQ traversal, reranked with full vectors
var asf = codes.scoreFunctionFor(query, similarity);
var reranker = vectors.rerankerFor(query, similarity);
var result = searcher.search(new DefaultSearchScoreProvider(asf, reranker), topK, rerankK,
                             0.0f, 0.0f, Bits.ALL);
```

BQ codes are kept in memory beside the graph; there is no fused BQ layout. The
[in-memory BQ example configuration](../jvector-examples/yaml-configs/index-parameters/in-memory-bq/README.md)
runs this pipeline in BenchYAML.

## Size

`8 × ⌈dimension / 64⌉` bytes per vector: 128 bytes for 1,024 dimensions (32× smaller than FP32).
