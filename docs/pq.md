# Product quantization (PQ)

[Product quantization](https://doi.org/10.1109/TPAMI.2010.57) splits each vector into `M`
subvectors ("subspaces") and replaces each with the nearest of a small set of centroids learned for
that subspace. With 256 centroids per subspace, each subvector becomes one byte, so a PQ code is
`M` bytes. PQ is JVector's stable, widely deployed quantizer for graph traversal.

## When to use it

- Traversal codes for indexes in memory or on disk, fused or separate.
- Any similarity function.
- When API stability matters more than the extra accuracy or build speed of
  [ASH](ash-configuration.md); see [Choosing an index](choosing-an-index.md).

## Train and encode

```java
int M = 64;                       // subspaces = bytes per code
ProductQuantization pq = ProductQuantization.compute(vectors, M, 256, false);
PQVectors codes = pq.encodeAll(vectors, PhysicalCoreExecutor.pool());
```

| Parameter | Meaning |
| --- | --- |
| `M` | Number of subspaces, and bytes per code. Larger is more accurate and larger. A common start is dimension / 8 (BenchYAML `mFactor: 8`, 16× compression) or the code size of the ASH model you are comparing with. |
| `clusterCount` | Centroids per subspace. 256 gives one byte per subspace and is required for fused PQ. |
| `globallyCenter` | Subtract the dataset mean before training. Not recommended for dot-product similarity. |
| `anisotropicThreshold` | Optional. Weights quantization error parallel to each vector more heavily ([anisotropic PQ](https://arxiv.org/abs/1908.10396)); requires unit-length vectors. `UNWEIGHTED` (the default) turns it off. |

Training samples a bounded subset of the vectors, so it does not need the whole dataset in memory;
see [Batched input vector access](batched-vector-access.md) for training straight from a file.

## Build and search

- **Construction:** `BuildScoreProvider.pqBuildScoreProvider(similarity, codes)` scores codes
  instead of full vectors; see [Graph construction](graph-construction.md). Diversity scoring
  compares code pairs straight from the codebooks, in one pass per comparison.
- **Search with codes in memory:** `codes.precomputedScoreFunctionFor(query, similarity)` gives
  the traversal scores; rerank with full or NVQ vectors from the index.
  [Tutorial 3](tutorials/3-larger-than-memory-tutorial.md) walks through the whole flow.
- **Fused PQ:** write the graph with `new FusedPQ(maxDegree, pq)` so each record carries its
  neighbors' codes; see [Fused and separated layouts](fused-and-separated.md).
- **Saving codes:** `codes.write(out)` and `PQVectors.load(reader)` store and reload separate
  codes.

## Sizes

| Item | Bytes |
| --- | --- |
| Code | `M` per vector |
| Codebooks | about 4 × 256 × dimension, once per model |
| Diversity codebook copy | about 4 × 256 × dimension, once diversity is first scored |

See [Memory and disk sizing](sizing.md) for whole-index estimates.
