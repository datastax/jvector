# Choosing an index

A JVector index is a graph plus up to two kinds of vector data:

- **Traversal codes** score candidates while the search walks the graph. They are compact and
  approximate: full vectors, or codes from PQ, ASH or BQ.
- **Reranking vectors** rescore the best candidates at the end: full-precision (FP32) vectors or
  NVQ codes.

Choosing an index means choosing where the graph lives, which traversal codes to use, where those
codes are stored, and what reranks.

## Step 1: memory or disk

| Your situation | Start with |
| --- | --- |
| A small collection whose full vectors fit comfortably in memory, or low-dimensional data where quantization saves little | In-memory graph, full vectors, no quantization |
| Vectors fit in memory but you want faster traversal | In-memory graph, ASH or PQ traversal, FP32 rerank |
| The index must persist, or vectors do not fit in memory | On-disk index, fused ASH or PQ traversal, NVQ rerank |
| The index is many times larger than memory | The same, with page-aligned records; see [Larger-than-memory indexes](larger-than-memory-indexes.md) |

The on-disk layout with fused codes and NVQ reranking is the default in the benchmark tools and
the usual starting point for a database. The
[example configurations](../jvector-examples/yaml-configs/index-parameters/README.md) give one
ready-to-run file per layout.

## Step 2: the traversal quantizer

| Quantizer | Choose it when | Notes |
| --- | --- | --- |
| **[PQ](pq.md)** | You want the stable, widely deployed default | Any similarity; optional anisotropic weighting; fused PQ requires 256 clusters per subspace |
| **[ASH](ash-configuration.md)** (experimental) | You want the best accuracy for a given code size, or faster builds | Any similarity (chosen when the model is trained); fused or in memory; API may change between releases |
| **[BQ](bq.md)** | Dimension is very high, vectors are normalized, and a coarse first pass is enough | One bit per dimension, scored by Hamming distance as an approximation of cosine similarity; not fused |
| **None** | The collection is small or low-dimensional | Scores full vectors; no training |

On large datasets at the same code size, fused ASH builds faster than fused PQ and reaches higher
recall at a given throughput; see [ASH performance](ash-performance.md). PQ remains the choice
when API stability matters most.

The code size sets the accuracy and memory trade-off. A 32× compression (for example, 128 bytes
for a 1,024-dimension FP32 vector) is a common starting point; see
[Memory and disk sizing](sizing.md).

## Step 3: fused or separate codes

| Layout | How it works | Best for |
| --- | --- | --- |
| **Fused** | Each node's on-disk record carries its neighbors' codes, so a search reads the codes with the graph | Indexes that should not hold codes in memory; fewest reads per query |
| **Separate** | Codes are kept in memory (or in a file beside the index) and the graph holds only neighbor ids | Smaller files; streaming indexes that keep their codes beside the flushed index |

Fused codes repeat each node's code in every record that links to it, so a fused index is
larger on disk (degree × code size per node). Reranking vectors have a similar choice: inline in
each record, or separated into their own block. [Fused and separated layouts](fused-and-separated.md)
compares all of these in detail. Fused ASH, stored ASH codes and page-aligned records need on-disk
format version 7, which JVector 4.0.x cannot read; see [UPGRADING.md](../UPGRADING.md).

## Step 4: reranking

| Reranker | Size per vector | Use it when |
| --- | --- | --- |
| **[NVQ](nvq.md)** | About a quarter of FP32 (dimension + 60 bytes with two subvectors) | On-disk indexes: high accuracy at a fraction of the read size |
| **FP32** | 4 × dimension bytes | Exact final scores, or in-memory indexes that already hold the vectors |
| **None** | 0 | Full-vector traversal, or when approximate scores are good enough |

Rerank more candidates to raise recall; see [Tuning](tuning.md).

## Similarity

JVector supports `DOT_PRODUCT`, `COSINE` and `EUCLIDEAN`. An ASH model is trained for one
similarity and saved with it; ASH does not require normalized vectors. See
[ASH similarity](ash-configuration.md#similarity). Int8 indexes use `ByteVectorSimilarityFunction`,
with the same three functions. Under `COSINE`, ASH and int8 indexes both reject a zero vector with
`IllegalArgumentException` when it is added or used as a query.
