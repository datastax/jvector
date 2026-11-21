# Fused and separated layouts

An on-disk JVector index stores up to two kinds of vector data next to the graph, and each can be
placed in more than one way:

- **Traversal codes** (ASH or PQ) can be **fused** into the graph or kept **separate**.
- **Reranking vectors** (FP32 or NVQ) can be **inline** in each node's record or **separated**
  into their own block.

These choices decide what must be in memory, how large the index file is, and how much a query
reads from disk. They do not change the graph itself, so the same graph can be written either way.

## Traversal codes: fused or separate

| | Fused | Separate |
| --- | --- | --- |
| Where the codes live | In each node's record, next to its neighbor list: the codes of that node's neighbors | One code per vector, held in memory (`PQVectors`, `ASHVectors`), optionally saved to a file beside the index |
| Memory for codes | None beyond the page cache | Every code: vector count × code bytes |
| Index size on disk | Adds degree × code bytes per node | Adds nothing to the graph; a codes file adds code bytes per node |
| Reads per expanded node | One record gives the neighbors and their codes | One record for the neighbors; codes come from memory |
| Writer feature | `FusedASH`, `FusedPQ` (`FeatureId.FUSED_ASH`, `FUSED_PQ`) | None in the graph; write codes with `CompressedVectors.write` and load them with `ASHVectors.load` or `PQVectors.load` |
| Limits | Fused ASH: 1–9 bits per dimension, fastest at 1, 2 and 4; fused PQ: 256 clusters per subspace; needs an FP32 or NVQ reranker in the graph | Works with ASH, PQ and BQ |

**Choose fused** when memory is the constraint: the codes cost no memory, and each disk read during
traversal brings the codes it needs. This is the default in the benchmark tools.

**Choose separate** when the codes fit in memory and disk space or write volume matters more:
the index file is much smaller, because each code is stored once instead of once per incoming
edge, and traversal scores neighbors from memory without reading their codes from disk. Streaming
indexes (`StreamingASHIndex`) already hold one code per vector, so they are usually flushed this
way; BenchYAML's streaming cases write those codes beside the flushed index and load them when the
index is opened.

As a rough guide, fused codes add degree × code bytes to every record. At degree 32 with 128-byte
codes that is 4 KB per node, more than the rest of a typical record; see
[Memory and disk sizing](sizing.md).

## Reranking vectors: inline or separated

| | Inline | Separated |
| --- | --- | --- |
| Where the vectors live | In each node's record (`InlineVectors`, `NVQ`) | In one contiguous block of the index file, after the node records (`SeparatedVectors`, `SeparatedNVQ`) |
| Node record size | Graph + codes + reranking vector | Graph + codes only |
| Reading a reranking vector | Comes with the node's record | A separate read at a fixed offset in the vector block |

**Inline** is the common choice and what the examples use: a record read for any reason also
brings its reranking vector, and page-aligned records keep each read to one page.

**Separated** keeps node records small, so traversal touches less data per node and more of the
graph fits in the page cache; each reranked candidate then costs one read from the vector block.
Consider it when records are large (high dimension, FP32 reranking) and the rerank depth is small
compared with the nodes a search expands. Measure both on your data before committing.

Fused ASH, stored ASH codes and page-aligned records require on-disk format version 7, which
JVector 4.0.x cannot read; see [UPGRADING.md](../UPGRADING.md).

## Common layouts

| Layout | In memory | On disk per node | Notes |
| --- | --- | --- | --- |
| Fused ASH or PQ + inline NVQ | Graph's upper layers | Neighbors, neighbors' codes, NVQ vector | Default for indexes served from disk; see [Larger-than-memory indexes](larger-than-memory-indexes.md) |
| Separate ASH or PQ codes + inline NVQ | All codes | Neighbors, NVQ vector | Smallest file with accurate reranking; codes must fit in memory |
| Separate codes + inline FP32 | All codes | Neighbors, full vector | Exact final scores; largest records |
| Separate codes + separated NVQ or FP32 | All codes | Neighbors only, plus the vector block | Smallest traversal reads |

See [Choosing an index](choosing-an-index.md) for picking the quantizer and reranker, and the
[example configurations](../jvector-examples/yaml-configs/index-parameters/README.md) for
ready-to-run fused and separate layouts.
